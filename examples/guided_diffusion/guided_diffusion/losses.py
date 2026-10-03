"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th


_SOBEL_KERNEL_CACHE = {}


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


# =============================================================================
# Distillation losses (added for NAS-guided dynamic policy distillation)
# =============================================================================

def split_eps_from_model_output(model_out: th.Tensor) -> th.Tensor:
    """
    guided-diffusion 的 learn_sigma=True 时，UNet out_channels=6：
      [eps(3ch), var(3ch)]
    这里返回 eps 部分。
    """
    if model_out.shape[1] == 6:
        eps, _ = th.split(model_out, 3, dim=1)
        return eps
    return model_out


def distill_eps_mse(
    student_out: th.Tensor,
    teacher_out: th.Tensor,
    reduction: str = "mean",
) -> th.Tensor:
    """
    蒸馏主损失：MSE(student_eps, teacher_eps).

    逻辑含义：
    - diffusion UNet 在 (x_t, t, y) 输入下预测噪声 eps（或 eps+var）
    - teacher 产生 target eps_teacher（detach）
    - student 学习逼近 eps_teacher

    reduction:
      - "mean": 返回标量
      - "none": 返回逐样本 loss (B,)
    """
    s = split_eps_from_model_output(student_out)
    t = split_eps_from_model_output(teacher_out).detach()

    # per-pixel MSE
    per = (s - t) ** 2  # (B, C, H, W)

    if reduction == "none":
        # (B,)
        return per.mean(dim=(1, 2, 3))
    if reduction == "sum":
        return per.sum()
    return per.mean()


def split_var_from_model_output(model_out: th.Tensor):
    """
    guided-diffusion 的 learn_sigma=True 时，UNet out_channels=6：
      [eps(3ch), var(3ch)]
    这里返回 var 部分；不存在时返回 None。
    """
    if model_out.shape[1] == 6:
        _, var = th.split(model_out, 3, dim=1)
        return var
    return None


def _extract_coefficients(arr, t: th.Tensor, shape) -> th.Tensor:
    if not isinstance(arr, th.Tensor):
        arr = th.tensor(arr, device=t.device, dtype=th.float32)
    else:
        arr = arr.to(device=t.device, dtype=th.float32)
    out = arr[t.long()]
    while len(out.shape) < len(shape):
        out = out.unsqueeze(-1)
    return out.expand(shape)


def predict_xstart_from_eps(
    diffusion,
    x_t: th.Tensor,
    t: th.Tensor,
    eps: th.Tensor,
    clip_denoised: bool = False,
) -> th.Tensor:
    pred_xstart = (
        _extract_coefficients(diffusion.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - _extract_coefficients(diffusion.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )
    if clip_denoised:
        pred_xstart = pred_xstart.clamp(-1.0, 1.0)
    return pred_xstart


def _get_sobel_kernels(channels: int, device: th.device, dtype: th.dtype):
    key = (channels, device.type, getattr(device, "index", None), str(dtype))
    if key in _SOBEL_KERNEL_CACHE:
        return _SOBEL_KERNEL_CACHE[key]
    kx = th.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = th.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    _SOBEL_KERNEL_CACHE[key] = (kx, ky)
    return kx, ky


def _gradient_alignment_loss(student_tensor: th.Tensor, teacher_tensor: th.Tensor) -> th.Tensor:
    channels = student_tensor.shape[1]
    kx, ky = _get_sobel_kernels(channels, student_tensor.device, student_tensor.dtype)
    grad_s_x = th.conv2d(student_tensor, kx, padding=1, groups=channels)
    grad_s_y = th.conv2d(student_tensor, ky, padding=1, groups=channels)
    grad_t_x = th.conv2d(teacher_tensor, kx, padding=1, groups=channels)
    grad_t_y = th.conv2d(teacher_tensor, ky, padding=1, groups=channels)
    return ((grad_s_x - grad_t_x) ** 2).mean() + ((grad_s_y - grad_t_y) ** 2).mean()


def distill_adm_hybrid_loss(
    *,
    student_out: th.Tensor,
    teacher_out: th.Tensor,
    x_t: th.Tensor,
    t: th.Tensor,
    diffusion,
    eps_weight: float = 1.0,
    xstart_weight: float = 0.25,
    var_weight: float = 0.0,
    grad_weight: float = 0.05,
    clip_xstart: bool = True,
):
    """
    比纯 eps-MSE 更稳的蒸馏目标：
    - eps 对齐：保持原始 teacher score target
    - x0 对齐：约束最终去噪方向，和采样质量更相关
    - var 对齐：仅在 learn_sigma=True 时启用
    """
    eps_s = split_eps_from_model_output(student_out)
    eps_t = split_eps_from_model_output(teacher_out).detach()

    loss_eps = ((eps_s - eps_t) ** 2).mean()
    total = float(eps_weight) * loss_eps

    loss_grad = th.zeros((), device=student_out.device, dtype=student_out.dtype)
    if grad_weight > 0.0 and eps_s.dim() == 4:
        loss_grad = _gradient_alignment_loss(eps_s, eps_t)
        total = total + float(grad_weight) * loss_grad

    loss_xstart = th.zeros((), device=student_out.device, dtype=student_out.dtype)
    if xstart_weight > 0.0:
        x0_s = predict_xstart_from_eps(diffusion, x_t, t, eps_s, clip_denoised=clip_xstart)
        x0_t = predict_xstart_from_eps(diffusion, x_t, t, eps_t, clip_denoised=clip_xstart).detach()
        loss_xstart = ((x0_s - x0_t) ** 2).mean()
        total = total + float(xstart_weight) * loss_xstart

    loss_var = th.zeros((), device=student_out.device, dtype=student_out.dtype)
    if var_weight > 0.0:
        var_s = split_var_from_model_output(student_out)
        var_t = split_var_from_model_output(teacher_out)
        if var_s is not None and var_t is not None:
            loss_var = ((var_s - var_t.detach()) ** 2).mean()
            total = total + float(var_weight) * loss_var

    metrics = {
        "loss_eps": float(loss_eps.detach().item()),
        "loss_grad": float(loss_grad.detach().item()),
        "loss_xstart": float(loss_xstart.detach().item()),
        "loss_var": float(loss_var.detach().item()),
    }
    return total, metrics


def distill_adm_consistency_loss(
    *,
    student_model_fn,
    x_start: th.Tensor,
    y: th.Tensor,
    tau_list,
    diffusion,
    clip_xstart: bool = True,
):
    """
    ADM 版 consistency loss:
    - 从 tau_list 中随机选相邻两个时间步 t_i, t_{i+1}
    - 用同一份噪声构造 x_ti / x_t{i+1}
    - 让 student 在两个时间步重建的 x0 保持一致

    student_model_fn 需要接受:
      student_model_fn(x_t, t, y) -> model_out
    """
    device = x_start.device
    batch_size = x_start.shape[0]
    tau_tensor = th.as_tensor(list(tau_list), device=device, dtype=th.long)
    if tau_tensor.numel() < 2:
        return th.zeros((), device=device, dtype=x_start.dtype)

    pair_idx = th.randint(0, tau_tensor.numel() - 1, (batch_size,), device=device)
    t_hi = tau_tensor.index_select(0, pair_idx)
    t_lo = tau_tensor.index_select(0, pair_idx + 1)

    noise = th.randn_like(x_start)
    x_t_hi = diffusion.q_sample(x_start, t_hi, noise=noise)
    x_t_lo = diffusion.q_sample(x_start, t_lo, noise=noise)

    student_out_hi = student_model_fn(x_t_hi, t_hi, y)
    student_out_lo = student_model_fn(x_t_lo, t_lo, y)

    eps_hi = split_eps_from_model_output(student_out_hi)
    eps_lo = split_eps_from_model_output(student_out_lo)

    x0_hi = predict_xstart_from_eps(diffusion, x_t_hi, t_hi, eps_hi, clip_denoised=clip_xstart)
    x0_lo = predict_xstart_from_eps(diffusion, x_t_lo, t_lo, eps_lo, clip_denoised=clip_xstart)
    return ((x0_hi - x0_lo) ** 2).mean()


def distill_adm_sampler_rollout_loss(
    *,
    student_sample: th.Tensor,
    teacher_sample: th.Tensor,
    grad_weight: float = 0.0,
):
    """
    对齐同一初始噪声、同一离散 tau、同一 sampler rollout 下的最终输出。
    用于减少 student 只适配 DDIM 局部监督、切到 ACDMS 后轨迹失配的问题。
    """
    teacher_sample = teacher_sample.detach()
    loss_img = ((student_sample - teacher_sample) ** 2).mean()
    total = loss_img

    loss_grad = th.zeros((), device=student_sample.device, dtype=student_sample.dtype)
    if grad_weight > 0.0 and student_sample.dim() == 4:
        loss_grad = _gradient_alignment_loss(student_sample, teacher_sample)
        total = total + float(grad_weight) * loss_grad

    metrics = {
        "loss_sampler_img": float(loss_img.detach().item()),
        "loss_sampler_grad": float(loss_grad.detach().item()),
    }
    return total, metrics
