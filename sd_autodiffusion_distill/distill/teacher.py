import os
import sys
import warnings

# -------------------- 抑制警告信息 --------------------
# 抑制 TensorFlow AVX2 FMA 和 TF-TRT 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=全部显示, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 OneDNN 优化警告

# 抑制 PyTorch weights_only 警告
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

# 抑制 transformers 库的 FutureWarning（关于 weights_only）
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message='.*You are using `torch.load` with `weights_only=False`.*')
warnings.filterwarnings('ignore', category=FutureWarning,
                       message='.*the default value for `weights_only` will be flipped to `True`.*')

import numpy as np
import torch
import torch.cuda.amp as amp
from typing import List
from omegaconf import OmegaConf
import random

# --- 1. 设置 Stable Diffusion 路径 ---
current_file = os.path.abspath(__file__)
distill_dir = os.path.dirname(os.path.dirname(current_file))           # sd_autodiffusion_distill
project_root = os.path.dirname(distill_dir)                            # AutoDiffusion-main
stable_diffusion_path = os.path.join(project_root, "examples", "Stable Diffusion")

if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)

ROOT = stable_diffusion_path

# UniPC 基础组件：噪声调度器 + model_wrapper
from ldm.models.diffusion.unipc import NoiseScheduleVP, model_wrapper as unipc_model_wrapper, UniPC

# 我们自己的 Flow Matching 采样器
from ldm.models.diffusion.acdms import ACDMSFM

# DPM-Solver++ 2M + Flow Matching Corrector 采样器
from .dpm_solver_wrapper import DPMSolverV2FMSampler, teacher_dpm_solver_sample

from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

# PyTorch Lightning 用于安全加载 checkpoint
try:
    import pytorch_lightning.callbacks.model_checkpoint
    _HAS_PYTORCH_LIGHTNING = True
except ImportError:
    _HAS_PYTORCH_LIGHTNING = False


# -------------------- 随机种子 --------------------
def set_seed(seed: int = 42, deterministic: bool = True):
    import os as _os
    try:
        import numpy as _np
    except Exception:
        _np = None

    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed)
    _os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# -------------------- 从 ckpt 实例化 LatentDiffusion --------------------
def _load_model_from_ckpt(config, ckpt, verbose=True):
    print(f"[Teacher] Loading model from {ckpt}")
    
    # 添加 PyTorch Lightning 安全全局对象（如果可用）
    if _HAS_PYTORCH_LIGHTNING:
        try:
            torch.serialization.add_safe_globals([
                pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
            ])
        except Exception:
            pass  # 如果添加失败，继续使用回退机制
    
    # 添加 numpy scalar 到安全全局对象列表
    try:
        import numpy.core.multiarray as multiarray
        torch.serialization.add_safe_globals([multiarray.scalar])
    except Exception:
        pass
    
    try:
        # 首先尝试使用 weights_only=True（更安全）
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as e:
        # 如果失败，说明 checkpoint 包含非张量对象，回退到 weights_only=False
        # 注意：这要求 checkpoint 来源可信
        # 使用更简洁的错误信息，避免显示冗长的警告
        if verbose:
            print(f"[Teacher] Using legacy checkpoint format (weights_only=False)")
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if verbose:
        if m:
            print("[Teacher] missing keys:", m)
        if u:
            print("[Teacher] unexpected keys:", u)

    return model


# -------------------- TeacherPipeline 封装 --------------------
class TeacherPipeline:
    def __init__(
        self,
        model,
        noise_schedule,
        unet_config=None,
        device="cuda",
        dtype=torch.float16,
    ):
        self.model = model
        self.unet = model.model.diffusion_model
        self.vae = model.first_stage_model
        self.text_encoder = model.cond_stage_model  # FrozenCLIPEmbedder
        self.noise_schedule = noise_schedule
        self.alphas_cumprod = model.alphas_cumprod
        self.unet_config = unet_config
        self.device = device
        self.dtype = dtype

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
            self.model.to(device)
        if dtype is not None:
            self.dtype = dtype
            self.unet.to(dtype=dtype)
            self.vae.to(dtype=dtype)
        return self

    # ------------------ 新增 encode_text 方法 ------------------
    def encode_text(self, prompts):
        """
        返回 Tensor-like 条件向量，用于 NAS / evaluator。
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        with torch.no_grad():
            cond = self.text_encoder(prompts)
        return cond.to(device=self.device, dtype=torch.float32)


# -------------------- load_teacher_pipeline --------------------
def load_teacher_pipeline(
    model_id: str,
    device: str = "cuda",
    use_fp16: bool = True,
) -> TeacherPipeline:
    default_cfg = os.path.join(ROOT, "configs/stable-diffusion/v1-inference_coco.yaml")
    config_path = os.getenv("SD_CONFIG_PATH", default_cfg)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. 可通过环境变量 SD_CONFIG_PATH 指定。"
        )
    config = OmegaConf.load(config_path)

    # 从 ckpt 实例化 LatentDiffusion
    try:
        model = _load_model_from_ckpt(config, model_id, verbose=True)
    except Exception as e:
        raise RuntimeError(f"加载模型权重失败: {model_id}. 错误: {e}")

    dtype = torch.float16 if use_fp16 else torch.float32

    # 构造 NoiseScheduleVP（离散）
    alphas_cumprod = model.alphas_cumprod.float()
    noise_schedule = NoiseScheduleVP(
        schedule="discrete",
        alphas_cumprod=alphas_cumprod,
        dtype=torch.float32,
    )

    # 封装成 TeacherPipeline
    pipe = TeacherPipeline(
        model=model,
        noise_schedule=noise_schedule,
        unet_config=config.model.params.unet_config,
        device=device,
        dtype=dtype,
    )

    # 冻结 Teacher/UNet/VAE/TextEncoder，并切换到 eval 模式
    try:
        pipe.model.eval()
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        for p in pipe.model.parameters():
            p.requires_grad_(False)
    except Exception:
        pass

    print("[Teacher] TeacherPipeline loaded (LatentDiffusion + NoiseScheduleVP) [frozen eval]")
    return pipe


# -------------------- VAE & 文本编码辅助函数 --------------------
def _get_vae_and_scale(pipe: TeacherPipeline):
    vae = pipe.vae
    scale = getattr(vae, "scale_factor", 0.18215)
    return vae, scale


def encode_text(pipe: TeacherPipeline, prompts, device="cuda"):
    text_encoder: FrozenCLIPEmbedder = pipe.text_encoder
    if isinstance(prompts, str):
        prompts = [prompts]
    with torch.no_grad():
        cond = text_encoder(prompts)
    return cond.to(device=device, dtype=torch.float32)


def get_uncond_embeddings(pipe: TeacherPipeline, batch_size: int, device="cuda"):
    text_encoder: FrozenCLIPEmbedder = pipe.text_encoder
    with torch.no_grad():
        uncond = text_encoder([""] * batch_size)
    return uncond.to(device=device, dtype=torch.float32)


def encode_image_to_latent(pipe: TeacherPipeline, images: torch.Tensor) -> torch.Tensor:
    vae, scale = _get_vae_and_scale(pipe)
    device = pipe.device
    dtype = pipe.dtype

    images = images.to(device=device, dtype=dtype)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(dtype == torch.float16), dtype=torch.float16):
        z = vae.encode(images)
        if hasattr(z, "latent_dist"):
            z = z.latent_dist.sample()
        elif hasattr(z, "sample"):
            z = z.sample()
        z = z * scale
    return z


def decode_latents(pipe: TeacherPipeline, latents: torch.Tensor) -> torch.Tensor:
    vae, scale = _get_vae_and_scale(pipe)
    device = pipe.device
    dtype = pipe.dtype

    latents = latents.to(device=device, dtype=dtype)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dtype == torch.float16), dtype=torch.float16):
        x = vae.decode(latents / scale)
        if hasattr(x, "sample"):
            x = x.sample
        images = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return images


# -------------------- 通用采样函数（支持 ACDMS 和 DPM-Solver++） --------------------
def teacher_sample(
    pipe: TeacherPipeline,
    latents: torch.Tensor,
    tau: list,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    cfg_scale: float = 4.5,
    num_inference_steps: int = 6,
    fm_strength: float = 0.04,
    fm_t_start: float = 0.80,
    sampler_type: str = "acdms",
    return_latents: bool = False,  # ✅ 新增: 是否只返回 latents 而不进行 VAE decode
) -> torch.Tensor:
    if sampler_type == "acdms":
        return teacher_acdms_sample(
            pipe=pipe,
            latents=latents,
            tau=tau,
            cond_emb=cond_emb,
            uncond_emb=uncond_emb,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            fm_strength=fm_strength,
            fm_t_start=fm_t_start,
            return_latents=return_latents,  # ✅ 传递参数
        )
    elif sampler_type == "dpm_solver":
        return teacher_dpm_solver_sample(
            pipe=pipe,
            latents=latents,
            tau=tau,
            cond_emb=cond_emb,
            uncond_emb=uncond_emb,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            fm_strength=fm_strength,
            fm_t_start=fm_t_start,
            return_latents=return_latents,  # ✅ 传递参数
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Must be 'acdms' or 'dpm_solver'.")


# -------------------- Teacher 噪声预测 & tau --------------------
def teacher_eps(
    pipe: TeacherPipeline,
    z_t: torch.Tensor,
    t: torch.Tensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    cfg_scale: float = 4.5,
) -> torch.Tensor:
    device = pipe.device
    dtype = pipe.dtype
    unet = pipe.unet

    z_t = z_t.to(device=device, dtype=dtype)
    t = t.to(device=device)
    cond_emb = cond_emb.to(device=device, dtype=torch.float32)
    uncond_emb = uncond_emb.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=(dtype == torch.float16), dtype=torch.float16):
            z_cat = torch.cat([z_t, z_t], dim=0)
            t_cat = torch.cat([t, t], dim=0)
            cond_cat = torch.cat([uncond_emb, cond_emb], dim=0)

            eps_cat = unet(z_cat, t_cat, context=cond_cat)
            eps_uncond, eps_cond = eps_cat.chunk(2, dim=0)
            eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    return eps_cfg


# -------------------- Tau 加载与采样函数 --------------------
def load_tau(tau_path: str, default_tau: list = None) -> list:
    """
    从 .npy 文件加载 tau（离散时间步列表）。
    如果文件不存在，返回 default_tau。
    """
    if tau_path is None or not os.path.exists(tau_path):
        if default_tau is not None:
            print(f"[load_tau] tau_path not found, using default_tau: {default_tau[:10]}...")
            return default_tau
        else:
            raise FileNotFoundError(f"tau_path not found: {tau_path}, and no default_tau provided.")
    
    tau = np.load(tau_path).tolist()
    print(f"[load_tau] Loaded tau from {tau_path}: {tau[:10]}... (len={len(tau)})")
    return tau


def sample_t_from_tau(pipe: TeacherPipeline, tau: list, batch_size: int = 1) -> torch.Tensor:
    """
    从 tau 列表中随机采样时间步 t。
    返回 LongTensor [batch_size]，值在 [0, 999] 范围内。
    """
    device = pipe.device
    t_idx = np.random.choice(len(tau), size=batch_size, replace=True)
    t_values = [tau[idx] for idx in t_idx]
    t_tensor = torch.tensor(t_values, dtype=torch.long, device=device)
    return t_tensor


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    设置随机种子以确保可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# -------------------- teacher_acdms_sample 函数 --------------------
def teacher_acdms_sample(
    pipe: TeacherPipeline,
    latents: torch.Tensor,
    tau: list,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    cfg_scale: float = 4.5,
    num_inference_steps: int = 6,
    fm_strength: float = 0.04,
    fm_t_start: float = 0.85,
    return_latents: bool = False,  # ✅ 新增: 是否只返回 latents 而不进行 VAE decode
) -> torch.Tensor:
    from ldm.models.diffusion.acdms import ACDMSSampler

    device = pipe.device
    dtype = pipe.dtype

    # 创建 ACDMS 采样器
    sampler = ACDMSSampler(
        model=pipe.model,
        noise_schedule=pipe.noise_schedule,
        fm_strength=fm_strength,
        fm_t_start=fm_t_start,
        dynamic_thresholding_ratio=0.995,
        thresholding_max_val=1.0,
        fm_lambda_clip=0.9,
        low_noise_threshold_index=150,
        cfg_high=cfg_scale,
        cfg_low=cfg_scale,
        cfg_schedule="two_stage",
    )

    # 获取 latent shape
    B, C, H, W = latents.shape

    # 运行采样
    with torch.inference_mode():
        latents_out, _ = sampler.sample(
            S=num_inference_steps,
            batch_size=B,
            shape=(C, H, W),
            conditioning=cond_emb,
            unconditional_conditioning=uncond_emb,
            unconditional_guidance_scale=cfg_scale,
            sampled_timestep=tau,
            verbose=False,
            x_T=latents,
        )

    # ✅ 如果 return_latents=True,直接返回 latents 而不进行 VAE decode
    if return_latents:
        return latents_out

    # 使用 VAE 解码 latents
    vae = pipe.vae
    scale = getattr(vae, "scale_factor", 0.18215)

    latents_out = latents_out.to(device=device, dtype=dtype)
    with torch.no_grad():
        if dtype == torch.float16:
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                x = vae.decode(latents_out / scale)
        else:
            x = vae.decode(latents_out / scale)

        if hasattr(x, "sample"):
            x = x.sample

        # 转换到 [0, 1] 范围
        images = (x.clamp(-1.0, 1.0) + 1.0) / 2.0

    return images
