import torch
import torch.nn.functional as F

# 简单的 Sobel 卷积核缓存，避免每步重复构造
_SOBEL_KERNEL_CACHE = {}

def _get_sobel_kernels(channels: int, device: torch.device, dtype: torch.dtype):
    key = (channels, device.type, getattr(device, "index", None), str(dtype))
    if key in _SOBEL_KERNEL_CACHE:
        return _SOBEL_KERNEL_CACHE[key]
    kx = torch.tensor([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ], device=device, dtype=dtype).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = torch.tensor([
        [-1.0, -2.0, -1.0],
        [0.0,  0.0,  0.0],
        [1.0,  2.0,  1.0],
    ], device=device, dtype=dtype).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    _SOBEL_KERNEL_CACHE[key] = (kx, ky)
    return kx, ky


def distillation_loss(
    eps_student_cfg: torch.Tensor,
    eps_teacher_cfg: torch.Tensor,
) -> torch.Tensor:
    """
    主蒸馏损失: 约束 Student 的噪声预测贴近 Teacher。
    同时在图像空间增加一个简单的梯度感知项, 减少高频噪点。
    """
    mse_loss = torch.mean((eps_student_cfg - eps_teacher_cfg) ** 2)

    # 可选: 对 4D 特征增加梯度感知正则
    if eps_student_cfg.dim() == 4:
        channels = eps_student_cfg.shape[1]
        device = eps_student_cfg.device
        dtype = eps_student_cfg.dtype

        # 使用缓存的 Sobel 3x3 卷积核，减少每步构建开销
        kx, ky = _get_sobel_kernels(channels, device, dtype)

        grad_student_x = torch.conv2d(eps_student_cfg, kx, padding=1, groups=channels)
        grad_student_y = torch.conv2d(eps_student_cfg, ky, padding=1, groups=channels)

        grad_teacher_x = torch.conv2d(eps_teacher_cfg, kx, padding=1, groups=channels)
        grad_teacher_y = torch.conv2d(eps_teacher_cfg, ky, padding=1, groups=channels)

        grad_loss = (
            torch.mean((grad_student_x - grad_teacher_x) ** 2)
            + torch.mean((grad_student_y - grad_teacher_y) ** 2)
        ) * 0.1  # 权重可调

        mse_loss = mse_loss + grad_loss

    return mse_loss


def consistency_loss(
    student_unet: torch.nn.Module,
    z0: torch.Tensor,
    tau,
    cond_emb: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    一致性 / flow-matching 风格的损失（修复版）：
      - 仅使用 GPU 端索引和张量操作，避免 CPU<->GPU 往返；
      - 对 alpha 进行数值下界裁剪，避免除零或梯度爆炸；
      - 当 tau 长度不足时返回 0 损失。
    """
    device = z0.device
    B = z0.shape[0]

    # tau -> Tensor[Len] on device
    if isinstance(tau, torch.Tensor):
        tau_tensor = tau.to(device=device, dtype=torch.long)
    else:
        tau_tensor = torch.tensor(list(tau), device=device, dtype=torch.long)

    if tau_tensor.numel() < 2:
        return torch.zeros((), device=device)

    # 假设 tau 已按从大到小排序 (t_T -> t_0)
    idx = torch.randint(0, tau_tensor.numel() - 1, (B,), device=device)
    t1_idx = tau_tensor.index_select(0, idx)
    t2_idx = tau_tensor.index_select(0, idx + 1)

    alpha_bar = alphas_cumprod.to(device=device, dtype=torch.float32)
    # gather alpha_bar[t]
    alpha1_bar = alpha_bar.gather(0, t1_idx)
    alpha2_bar = alpha_bar.gather(0, t2_idx)

    # 数值稳定性：对 sqrt 输入与分母 alpha 加下界
    eps_small = 1e-8
    alpha1 = torch.clamp(alpha1_bar, min=eps_small).sqrt().view(B, 1, 1, 1)
    sigma1 = torch.clamp(1.0 - alpha1_bar, min=0.0).sqrt().view(B, 1, 1, 1)
    alpha2 = torch.clamp(alpha2_bar, min=eps_small).sqrt().view(B, 1, 1, 1)
    sigma2 = torch.clamp(1.0 - alpha2_bar, min=0.0).sqrt().view(B, 1, 1, 1)

    # 同一噪声 eps, 构造两个时间步的带噪 latent
    eps = torch.randn_like(z0)
    z_t1 = alpha1 * z0 + sigma1 * eps
    z_t2 = alpha2 * z0 + sigma2 * eps

    # Student 在两个时间步的噪声预测 (有梯度)
    eps_hat1 = student_unet(z_t1, t1_idx, context=cond_emb)
    eps_hat2 = student_unet(z_t2, t2_idx, context=cond_emb)

    # 重建对应的 z0_hat（数值稳定）
    z0_hat1 = (z_t1 - sigma1 * eps_hat1) / torch.clamp(alpha1, min=eps_small)
    z0_hat2 = (z_t2 - sigma2 * eps_hat2) / torch.clamp(alpha2, min=eps_small)

    loss = F.mse_loss(z0_hat1, z0_hat2)
    return loss
