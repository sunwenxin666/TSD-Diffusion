import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th


stable_diffusion_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../Stable Diffusion")
)
if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)

from ldm.models.diffusion.acdms import ACDMSFM
from ldm.models.diffusion.unipc import NoiseScheduleVP

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.losses import split_eps_from_model_output


def _extract_into_tensor(arr: th.Tensor, timesteps: th.Tensor, broadcast_shape) -> th.Tensor:
    res = arr.to(device=timesteps.device)[timesteps].float()
    while res.dim() < len(broadcast_shape):
        res = res[..., None]
    return res


def _arch_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tau_to_step_index(t_scalar: int, tau_list: List[int]) -> int:
    tau_int = [int(x) for x in tau_list]
    t_scalar = int(t_scalar)
    if t_scalar in tau_int:
        return tau_int.index(t_scalar)
    diffs = [abs(x - t_scalar) for x in tau_int]
    return int(np.argmin(diffs))


def gstar_to_skip_layers_for_scalar_t(
    g_star_obj,
    tau_list: List[int],
    t_scalar: int,
    allow_hard_skip: bool = True,
) -> List[int]:
    controls = gstar_to_policy_controls_for_scalar_t(
        g_star_obj,
        tau_list,
        t_scalar,
        allow_hard_skip=allow_hard_skip,
    )
    return list(controls["skip_layers"])


def gstar_to_policy_controls_for_scalar_t(
    g_star_obj,
    tau_list: List[int],
    t_scalar: int,
    allow_hard_skip: bool = True,
) -> Dict[str, Any]:
    if g_star_obj is None:
        return {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0}
    step_policies = _arch_get(g_star_obj, "step_policies", None)
    if not step_policies:
        return {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0}

    si = _tau_to_step_index(int(t_scalar), tau_list)
    si = max(0, min(si, len(step_policies) - 1))
    pol = step_policies[si]

    layer_mask = _arch_get(pol, "layer_mask", None)
    if layer_mask is not None:
        mask = [bool(x) for x in layer_mask]
        skip_layers = [i for i, keep in enumerate(mask) if not keep]
    else:
        skip_blocks = _arch_get(pol, "skip_blocks", None)
        if skip_blocks is None:
            skip_layers = []
        else:
            skip_layers = [int(x) for x in skip_blocks]

    layer_gates = _arch_get(pol, "layer_gates", None)
    layer_gate_map: Dict[int, float] = {}
    if isinstance(layer_gates, dict):
        for layer_id, gate in layer_gates.items():
            layer_gate_map[int(layer_id)] = float(gate)
    elif layer_gates is not None:
        for item in layer_gates:
            if isinstance(item, dict):
                layer_id = item.get("layer_id", None)
                gate = item.get("gate", None)
            else:
                try:
                    layer_id, gate = item
                except Exception:
                    continue
            if layer_id is None or gate is None:
                continue
            layer_gate_map[int(layer_id)] = float(gate)

    class_gain = _arch_get(pol, "class_gain", 1.0)
    if not bool(allow_hard_skip):
        skip_layers = []

    return {
        "skip_layers": [int(x) for x in skip_layers],
        "layer_gate_map": {int(k): float(v) for k, v in layer_gate_map.items()},
        "class_gain": float(class_gain if class_gain is not None else 1.0),
    }


def build_adm_noise_schedule(teacher_defaults: Dict[str, Any], device: str) -> NoiseScheduleVP:
    betas = gd.get_named_beta_schedule(
        str(teacher_defaults["noise_schedule"]),
        int(teacher_defaults["diffusion_steps"]),
    )
    betas = th.tensor(betas, device=device, dtype=th.float32)
    return NoiseScheduleVP("discrete", betas=betas, dtype=th.float32)


def resolve_adm_acdms_params(
    *,
    tau_list: List[int],
    total_n: int,
    fm_strength: float,
    low_noise_threshold_index: int,
    fm_apply_last_k_steps: int,
    adaptive_low_noise: bool = True,
    adaptive_fm_strength: bool = True,
    adaptive_last_k: bool = True,
    low_noise_step_ratio: float = 0.34,
    low_noise_margin: int = 16,
):
    tau_desc = sorted((int(x) for x in tau_list), reverse=True)
    steps_total = max(1, len(tau_desc))

    resolved_threshold = int(low_noise_threshold_index)
    resolved_last_k = int(fm_apply_last_k_steps)
    resolved_strength = float(fm_strength)

    low_noise_steps = max(1, int(round(steps_total * float(low_noise_step_ratio))))
    low_noise_steps = min(steps_total, low_noise_steps)

    if adaptive_low_noise:
        pivot = tau_desc[-low_noise_steps]
        resolved_threshold = min(int(total_n - 1), int(pivot + max(1, int(low_noise_margin))))

    if adaptive_last_k:
        resolved_last_k = max(1, low_noise_steps)

    if adaptive_fm_strength:
        gaps = np.array([tau_desc[i] - tau_desc[i + 1] for i in range(len(tau_desc) - 1)], dtype=np.float32)
        if gaps.size > 0:
            low_gap = float(np.mean(gaps[-min(low_noise_steps, gaps.size):]))
            mid_lo = max(0, gaps.size // 3)
            mid_hi = max(mid_lo + 1, gaps.size - low_noise_steps)
            mid_slice = gaps[mid_lo:mid_hi]
            if mid_slice.size == 0:
                mid_slice = gaps
            mid_gap = float(np.mean(mid_slice))
            gap_ratio = low_gap / max(1e-6, mid_gap)
            gap_scale = float(np.clip(gap_ratio, 0.8, 1.8))
        else:
            gap_scale = 1.0

        step_scale = float(np.sqrt(6.0 / float(max(1, steps_total))))
        resolved_strength = float(fm_strength) * step_scale * gap_scale

    meta = {
        "tau_desc": tau_desc,
        "steps_total": int(steps_total),
        "low_noise_steps": int(low_noise_steps),
        "resolved_low_noise_threshold_index": int(resolved_threshold),
        "resolved_fm_apply_last_k_steps": int(resolved_last_k),
        "resolved_fm_strength": float(resolved_strength),
    }
    return resolved_strength, resolved_threshold, resolved_last_k, meta


def _sample_acdms_with_explicit_timesteps(
    sampler: ACDMSFM,
    x: th.Tensor,
    timesteps: th.Tensor,
    order: int = 2,
):
    """
    Run UniPC/ACDMS on an explicit descending continuous-time grid instead of a
    uniform grid. This keeps the original discrete tau schedule.
    """
    device = x.device
    timesteps = timesteps.to(device=device, dtype=th.float32)
    assert timesteps.dim() == 1 and timesteps.numel() >= 2
    assert bool(th.all(timesteps[:-1] >= timesteps[1:])), "timesteps must be descending"

    unipc = sampler.unipc
    steps = int(timesteps.numel() - 1)
    sampler._reset_fm_state()
    sampler._steps_total = steps

    t_prev_list = [timesteps[0]]
    # 注意：这里不使用 th.no_grad()，因为 model_fn 内部可能需要计算梯度（通过 cond_fn）
    model_prev_list = [unipc.model_fn(x, timesteps[0]).detach()]
    if unipc.correcting_xt_fn is not None:
        x = unipc.correcting_xt_fn(x, timesteps[0], 0).detach()

    for step in range(1, min(order, steps + 1)):
        t_cur = timesteps[step]
        x, model_x = unipc.multistep_uni_pc_update(
            x,
            model_prev_list,
            t_prev_list,
            t_cur,
            step,
            use_corrector=True,
        )
        if model_x is None:
            model_x = unipc.model_fn(x, t_cur)
        model_x = model_x.detach()
        if unipc.correcting_xt_fn is not None:
            x = unipc.correcting_xt_fn(x, t_cur, step).detach()
        t_prev_list.append(t_cur)
        model_prev_list.append(model_x)

    for step in range(order, steps + 1):
        t_cur = timesteps[step]
        step_order = min(order, steps + 1 - step)
        use_corrector = step < steps
        x, model_x = unipc.multistep_uni_pc_update(
            x,
            model_prev_list,
            t_prev_list,
            t_cur,
            step_order,
            use_corrector=use_corrector,
        )
        x = x.detach()
        if unipc.correcting_xt_fn is not None:
            x = unipc.correcting_xt_fn(x, t_cur, step).detach()
        for i in range(order - 1):
            t_prev_list[i] = t_prev_list[i + 1]
            model_prev_list[i] = model_prev_list[i + 1]
        t_prev_list[-1] = t_cur
        if step < steps:
            if model_x is None:
                model_x = unipc.model_fn(x, t_cur)
            model_prev_list[-1] = model_x.detach()
    return x


def sample_with_adm_acdms(
    *,
    student,
    teacher_defaults: Dict[str, Any],
    tau_list: List[int],
    g_star_obj,
    shape,
    device: str,
    class_labels: Optional[th.Tensor] = None,
    clip_denoised: bool = True,
    fm_strength: float = 0.04,
    fm_t_start: float = 0.8,
    fm_lambda_clip: float = 0.9,
    dynamic_thresholding_ratio: float = 0.995,
    thresholding_max_val: float = 1.0,
    low_noise_threshold_index: int = 150,
    fm_apply_last_k_steps: int = 1,
    adaptive_low_noise: bool = True,
    adaptive_fm_strength: bool = True,
    adaptive_last_k: bool = True,
    low_noise_step_ratio: float = 0.34,
    low_noise_margin: int = 16,
    x_init: Optional[th.Tensor] = None,
    return_meta: bool = False,
    allow_hard_skip: bool = True,
    cond_fn=None,
):
    """
    ADM student wrapper for ACDMSFM using the exact discrete tau schedule.
    """
    noise_schedule = build_adm_noise_schedule(teacher_defaults, device=device)
    total_n = int(noise_schedule.total_N)
    betas = gd.get_named_beta_schedule(
        str(teacher_defaults["noise_schedule"]),
        int(teacher_defaults["diffusion_steps"]),
    )
    alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
    alphas_cumprod = th.tensor(alphas_cumprod, device=device, dtype=th.float32)
    tau_desc = sorted((int(x) for x in tau_list), reverse=True)
    steps_total = max(1, len(tau_desc))
    resolved_fm_strength, resolved_low_noise_threshold_index, resolved_last_k, meta = resolve_adm_acdms_params(
        tau_list=tau_desc,
        total_n=total_n,
        fm_strength=float(fm_strength),
        low_noise_threshold_index=int(low_noise_threshold_index),
        fm_apply_last_k_steps=int(fm_apply_last_k_steps),
        adaptive_low_noise=bool(adaptive_low_noise),
        adaptive_fm_strength=bool(adaptive_fm_strength),
        adaptive_last_k=bool(adaptive_last_k),
        low_noise_step_ratio=float(low_noise_step_ratio),
        low_noise_margin=int(low_noise_margin),
    )
    meta["adaptive_low_noise"] = bool(adaptive_low_noise)
    meta["adaptive_fm_strength"] = bool(adaptive_fm_strength)
    meta["adaptive_last_k"] = bool(adaptive_last_k)
    meta["input_fm_strength"] = float(fm_strength)
    meta["input_low_noise_threshold_index"] = int(low_noise_threshold_index)
    meta["input_fm_apply_last_k_steps"] = int(fm_apply_last_k_steps)

    # 预计算所有时间步的控制参数
    precomputed_controls = {}
    if g_star_obj is not None:
        for t_scalar in tau_list:
            precomputed_controls[t_scalar] = gstar_to_policy_controls_for_scalar_t(
                g_star_obj,
                tau_list,
                t_scalar,
                allow_hard_skip=allow_hard_skip,
            )

    def model_fn_noise(x_in: th.Tensor, t_in: th.Tensor):
        if t_in.dim() == 0:
            t_in = t_in.expand(x_in.shape[0])
        t_discrete = (t_in.float() * float(total_n) - 1.0).clamp(0, total_n - 1).round().long()
        t_scalar = int(t_discrete[0].item())
        controls = precomputed_controls.get(
            t_scalar,
            {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0},
        )

        with th.no_grad():
            if class_labels is None:
                out = student(
                    x_in,
                    t_discrete,
                    skip_layer=controls["skip_layers"],
                    layer_gate_map=controls["layer_gate_map"],
                    class_gain=controls["class_gain"],
                )
            else:
                out = student(
                    x_in,
                    t_discrete,
                    y=class_labels,
                    skip_layer=controls["skip_layers"],
                    layer_gate_map=controls["layer_gate_map"],
                    class_gain=controls["class_gain"],
                )
            eps = split_eps_from_model_output(out).detach()

        if cond_fn is not None and class_labels is not None:
            # 确保 x_in 具有 requires_grad=True，这样 cond_fn 才能正常计算梯度
            x_in_for_grad = x_in.detach().requires_grad_(True)
            grad = cond_fn(x_in_for_grad, t_discrete, y=class_labels).detach()
            sigma_t = th.sqrt(1.0 - _extract_into_tensor(alphas_cumprod, t_discrete, x_in.shape))
            eps = eps - sigma_t.to(dtype=eps.dtype) * grad.to(dtype=eps.dtype)

        return eps


    sampler = ACDMSFM(
        model_fn_noise=model_fn_noise,
        noise_schedule=noise_schedule,
        fm_strength=float(resolved_fm_strength),
        fm_t_start=float(fm_t_start),
        fm_lambda_clip=float(fm_lambda_clip),
        dynamic_thresholding_ratio=float(dynamic_thresholding_ratio),
        thresholding_max_val=float(thresholding_max_val),
        low_noise_threshold_index=int(resolved_low_noise_threshold_index),
        fm_apply_last_k_steps=int(resolved_last_k),
    )

    if x_init is None:
        x_t = th.randn(shape, device=device)
    else:
        x_t = x_init.to(device=device).clone()
    tau_cont = th.tensor(
        [(int(ti) + 1.0) / float(total_n) for ti in tau_desc],
        device=device,
        dtype=th.float32,
    )
    sample = _sample_acdms_with_explicit_timesteps(
        sampler=sampler,
        x=x_t,
        timesteps=tau_cont,
        order=2,
    )
    if clip_denoised:
        sample = sample.clamp(-1.0, 1.0)
    if return_meta:
        return sample, meta
    return sample
