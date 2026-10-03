"""
DPM-Solver++ 2M + Flow Matching Corrector 包装类
只包含项目特定的配置/默认参数，主实现在 ldm.models.diffusion.dpmsolver_two
"""

from typing import Optional, Any, Tuple, Sequence, Dict

import torch

from ldm.models.diffusion.dpmsolver_two import (
    NoiseScheduleVP,
    DPMSolverV2FM,
    prepare_tau_for_dpmsolver,
)


# ============================================================
# Wrapper Sampler - 项目特定的采样器配置
# ============================================================

class DPMSolverV2FMSampler:
    """
    DPM-Solver++ 2M + FM Corrector 的包装类
    提供项目特定默认参数，并把关键控制参数透传到 DPMSolverV2FM：
    - fm_step_clip：每步 FM 校正最大步长（压“过艳”的最强旋钮）
    - step_ramp_start：FM 渐进启用的开始位置（0~1，越大越晚启用，越稳越不艳）
    - inner_steps_low_noise：低噪声阶段的迭代次数（建议先 1）
    """

    def __init__(
        self,
        model,
        betas: torch.Tensor,  # 直接传 model.betas
        fm_strength: float = 0.04,
        fm_t_start: float = 0.85,
        fm_lambda_clip: float = 1.0,
        fm_step_clip: float = 0.02,        # ✅ 新增：建议 0.02
        step_ramp_start: float = 0.60,     # ✅ 新增：建议 0.55~0.70
        low_noise_threshold_index: int = 150,
        inner_steps_low_noise: int = 1,    # ✅ 新增：建议先 1
        cfg_schedule: str = "two_stage",
    ):
        self.model = model
        self.betas = betas
        self.noise_schedule = NoiseScheduleVP("discrete", betas=self.betas)

        self.fm_strength = float(fm_strength)
        self.fm_t_start = float(fm_t_start)
        self.fm_lambda_clip = float(fm_lambda_clip)
        self.fm_step_clip = float(fm_step_clip)
        self.step_ramp_start = float(step_ramp_start)

        self.low_noise_threshold_index = int(low_noise_threshold_index)
        self.inner_steps_low_noise = int(inner_steps_low_noise)

        self.cfg_schedule = str(cfg_schedule)

    @torch.no_grad()
    def sample(
        self,
        S: int,
        batch_size: int,
        shape: Tuple[int, int, int],
        conditioning: Any,
        unconditional_conditioning: Any,
        unconditional_guidance_scale: float = 4.5,
        sampled_timestep: Optional[Sequence[int]] = None,
        verbose: bool = False,
        x_T: Optional[torch.Tensor] = None,
        use_fp16: bool = True,  # ✅ 默认使用FP16加速
        **kwargs
    ):
        device = x_T.device if x_T is not None else self.betas.device

        # ✅ 根据use_fp16选择数据类型
        dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32

        if x_T is None:
            C, H, W = shape
            x_T = torch.randn(batch_size, C, H, W, device=device, dtype=dtype)
        else:
            x_T = x_T.to(device=device, dtype=dtype)

        N = int(getattr(self.noise_schedule, "total_N", 1000))

        # ✅ 使用主实现：强制包含 999 和 0，并对齐到 S+1
        tau_desc = prepare_tau_for_dpmsolver(sampled_timestep, S=int(S), N=N)

        if verbose:
            print(f"[DPMSolverV2FMSampler] S={S}, tau_desc(len={len(tau_desc)}): {tau_desc.tolist()}")
            print(f"[DPMSolverV2FMSampler] cfg_scale={float(unconditional_guidance_scale)} "
                  f"fm_strength={self.fm_strength} fm_t_start={self.fm_t_start} "
                  f"fm_step_clip={self.fm_step_clip} step_ramp_start={self.step_ramp_start}")

        cfg_high = float(unconditional_guidance_scale)
        cfg_low = float(unconditional_guidance_scale)

        solver = DPMSolverV2FM(
            model=self.model,
            noise_schedule=self.noise_schedule,
            conditioning=conditioning,
            unconditional_conditioning=unconditional_conditioning,
            timesteps_idx_desc=tau_desc,
            cfg_high=cfg_high,
            cfg_low=cfg_low,
            cfg_schedule=self.cfg_schedule,
            fm_strength_base=self.fm_strength,
            fm_t_start=self.fm_t_start,
            fm_lambda_clip=self.fm_lambda_clip,
            fm_step_clip=self.fm_step_clip,             # ✅ 透传
            step_ramp_start=self.step_ramp_start,       # ✅ 透传
            low_noise_threshold_index=self.low_noise_threshold_index,
            inner_steps_low_noise=self.inner_steps_low_noise,
        )

        latents = solver.sample(x_T, apply_fm_corrector=True)

        info_dict: Dict[str, Any] = {
            "sampler": "dpm_solver_v2_fm",
            "steps": int(S),
            "tau_desc": tau_desc.tolist(),
            "cfg_scale": float(unconditional_guidance_scale),
            "fm_strength": self.fm_strength,
            "fm_t_start": self.fm_t_start,
            "fm_step_clip": self.fm_step_clip,
            "step_ramp_start": self.step_ramp_start,
            "low_noise_threshold_index": self.low_noise_threshold_index,
            "inner_steps_low_noise": self.inner_steps_low_noise,
        }
        return latents, info_dict


# ============================================================
# Teacher helper - 项目特定的教师采样辅助函数
# ============================================================

@torch.no_grad()
def teacher_dpm_solver_sample(
    pipe,
    latents: torch.Tensor,
    tau: Sequence[int],
    cond_emb: Any,
    uncond_emb: Any,
    cfg_scale: float = 4.5,
    num_inference_steps: int = 6,
    fm_strength: float = 0.04,
    fm_t_start: float = 0.90,
    fm_lambda_clip: float = 1.0,
    fm_step_clip: float = 0.02,        # ✅ 默认更不艳
    step_ramp_start: float = 0.60,     # ✅ 默认平滑启用
    verbose: bool = False,
    return_latents: bool = False,  # ✅ 新增: 是否只返回 latents 而不进行 VAE decode
    use_fp16: bool = True,  # ✅ 默认使用FP16加速
    **kwargs
) -> torch.Tensor:
    betas = pipe.model.betas

    sampler = DPMSolverV2FMSampler(
        model=pipe.model,
        betas=betas,
        fm_strength=fm_strength,
        fm_t_start=fm_t_start,
        fm_lambda_clip=fm_lambda_clip,
        fm_step_clip=fm_step_clip,
        step_ramp_start=step_ramp_start,
        low_noise_threshold_index=150,
        inner_steps_low_noise=1,
        cfg_schedule="two_stage",
    )

    B, C, H, W = latents.shape

    latents_out, info = sampler.sample(
        S=int(num_inference_steps),
        batch_size=B,
        shape=(C, H, W),
        conditioning=cond_emb,
        unconditional_conditioning=uncond_emb,
        unconditional_guidance_scale=float(cfg_scale),
        sampled_timestep=tau,
        verbose=verbose,
        x_T=latents,
        use_fp16=use_fp16,  # ✅ 传递use_fp16参数
    )

    if verbose:
        print("[teacher_dpm_solver_sample] info:", info)

    # ✅ 如果 return_latents=True,直接返回 latents 而不进行 VAE decode
    if return_latents:
        return latents_out

    # VAE decode
    vae = pipe.vae
    scale = getattr(vae, "scale_factor", 0.18215)
    dtype = pipe.dtype
    device = pipe.device

    latents_out = latents_out.to(device=device, dtype=dtype)
    with torch.no_grad():
        if dtype == torch.float16:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                x = vae.decode(latents_out / scale)
        else:
            x = vae.decode(latents_out / scale)

        if hasattr(x, "sample"):
            x = x.sample

        images = (x.clamp(-1.0, 1.0) + 1.0) / 2.0

    return images
