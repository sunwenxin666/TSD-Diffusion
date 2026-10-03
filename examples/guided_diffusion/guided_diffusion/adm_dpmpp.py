import os
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import importlib.util
import numpy as np
import torch as th

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.losses import split_eps_from_model_output


def _load_dpmsolver_symbols():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(this_dir, "dpmsolver_two.py"),
        os.path.join(this_dir, "nas", "dpmsolver_two.py"),
        os.path.normpath(
            os.path.join(
                this_dir,
                "..",
                "..",
                "Stable Diffusion",
                "ldm",
                "models",
                "diffusion",
                "dpmsolver_two.py",
            )
        ),
    ]

    # normal imports first
    try:
        from .dpmsolver_two import NoiseScheduleVP as _NoiseScheduleVP
        from .dpmsolver_two import DPMSolverV2FM as _DPMSolverV2FM
        return _NoiseScheduleVP, _DPMSolverV2FM
    except Exception:
        pass
    try:
        from dpmsolver_two import NoiseScheduleVP as _NoiseScheduleVP
        from dpmsolver_two import DPMSolverV2FM as _DPMSolverV2FM
        return _NoiseScheduleVP, _DPMSolverV2FM
    except Exception:
        pass

    last_err = None
    for mod_path in candidate_paths:
        if not os.path.isfile(mod_path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("external_dpmsolver_two", mod_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module.NoiseScheduleVP, module.DPMSolverV2FM
        except Exception as e:
            last_err = e

    raise ImportError(
        "Cannot import NoiseScheduleVP / DPMSolverV2FM. Tried normal imports and file paths:\n"
        + "\n".join(candidate_paths)
        + (f"\nLast error: {last_err}" if last_err is not None else "")
    )


NoiseScheduleVP, DPMSolverV2FM = _load_dpmsolver_symbols()


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


class _CallableModelAdapter:
    """Adapter for LDM-style DPMSolver code which expects model.apply_model(...)."""
    def __init__(self, fn):
        self.fn = fn

    def apply_model(self, x, t, cond=None):
        return self.fn(x, t)


@th.no_grad()
def sample_with_adm_dpmpp(
    *,
    student,
    teacher_defaults: Dict[str, Any],
    tau_list: List[int],
    g_star_obj,
    shape,
    device: str,
    class_labels: Optional[th.Tensor] = None,
    clip_denoised: bool = True,
    fm_strength: float = 0.0,
    fm_t_start: float = 0.85,
    fm_lambda_clip: float = 1.0,
    fm_step_clip: float = 0.02,
    step_ramp_start: float = 0.60,
    low_noise_threshold_index: int = 150,
    inner_steps_low_noise: int = 0,
    fm_apply_last_k_steps: int = 1,
    continuous_grid: bool = True,
    dynamic_thresholding: bool = True,
    threshold_ratio: float = 0.998,
    threshold_max_val: float = 1.0,
    x_init: Optional[th.Tensor] = None,
    return_meta: bool = False,
    allow_hard_skip: bool = True,
    cond_fn=None,
):
    """
    ADM wrapper aligned to the modified dpmsolver_two.py.

    Key semantics:
    - continuous_grid=True: time_step=K -> exactly K solver updates.
    - tau_list is used as stage anchors for NAS policy / classifier-guidance control.
    - FM corrector can be restricted to the last K steps.
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
    if len(tau_desc) < 2:
        raise ValueError("tau_list 至少需要两个离散时间点（例如 [999, ..., 0]）。")

    precomputed_controls = {}
    if g_star_obj is not None:
        for t_scalar in tau_desc:
            precomputed_controls[t_scalar] = gstar_to_policy_controls_for_scalar_t(
                g_star_obj,
                tau_desc,
                t_scalar,
                allow_hard_skip=allow_hard_skip,
            )

    if bool(continuous_grid):
        solver_t_nodes = np.linspace(
            (float(tau_desc[0]) + 1.0) / float(total_n),
            (float(tau_desc[-1]) + 1.0) / float(total_n),
            num=len(tau_desc) + 1,
            dtype=np.float32,
        )

        def _continuous_to_stage_idx(t_in: th.Tensor) -> int:
            t_val = float(t_in.reshape(-1)[0].item())
            idx = int(np.argmin(np.abs(solver_t_nodes[:-1] - t_val)))
            return int(max(0, min(idx, len(tau_desc) - 1)))

        def model_fn_noise(x_in: th.Tensor, t_in: th.Tensor):
            if t_in.dim() == 0:
                t_in = t_in.expand(x_in.shape[0])
            stage_idx = _continuous_to_stage_idx(t_in)
            t_scalar = int(tau_desc[stage_idx])
            t_discrete = th.full((x_in.shape[0],), t_scalar, device=x_in.device, dtype=th.long)
            controls = precomputed_controls.get(
                t_scalar,
                {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0},
            )

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
                x_in_for_grad = x_in.detach().requires_grad_(True)
                grad = cond_fn(x_in_for_grad, t_discrete, y=class_labels).detach()
                sigma_t = th.sqrt(1.0 - _extract_into_tensor(alphas_cumprod, t_discrete, x_in.shape))
                eps = eps - sigma_t.to(dtype=eps.dtype) * grad.to(dtype=eps.dtype)
            return eps
    else:
        def model_fn_noise(x_in: th.Tensor, t_in: th.Tensor):
            if t_in.dim() == 0:
                t_in = t_in.expand(x_in.shape[0])
            t_discrete = (t_in.float() * float(total_n) - 1.0).clamp(0, total_n - 1).round().long()
            t_scalar = int(t_discrete[0].item())
            controls = precomputed_controls.get(
                t_scalar,
                {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0},
            )

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
                x_in_for_grad = x_in.detach().requires_grad_(True)
                grad = cond_fn(x_in_for_grad, t_discrete, y=class_labels).detach()
                sigma_t = th.sqrt(1.0 - _extract_into_tensor(alphas_cumprod, t_discrete, x_in.shape))
                eps = eps - sigma_t.to(dtype=eps.dtype) * grad.to(dtype=eps.dtype)
            return eps

    solver = DPMSolverV2FM(
        model=_CallableModelAdapter(model_fn_noise),
        noise_schedule=noise_schedule,
        conditioning=None,
        unconditional_conditioning=None,
        timesteps_idx_desc=tau_desc,
        cfg_high=1.0,
        cfg_low=1.0,
        cfg_schedule="two_stage",
        fm_strength_base=float(fm_strength),
        fm_t_start=float(fm_t_start),
        fm_lambda_clip=float(fm_lambda_clip),
        fm_step_clip=float(fm_step_clip),
        step_ramp_start=float(step_ramp_start),
        low_noise_threshold_index=int(low_noise_threshold_index),
        inner_steps_low_noise=int(inner_steps_low_noise),
        fm_apply_last_k_steps=int(fm_apply_last_k_steps),
        continuous_grid=bool(continuous_grid),
        dynamic_thresholding=bool(dynamic_thresholding),
        threshold_ratio=float(threshold_ratio),
        threshold_max_val=float(threshold_max_val),
    )

    if x_init is None:
        x_t = th.randn(shape, device=device)
    else:
        x_t = x_init.to(device=device).clone()

    sample = solver.sample(x_t, apply_fm_corrector=bool(float(fm_strength) > 0.0))
    if clip_denoised:
        sample = sample.clamp(-1.0, 1.0)

    if return_meta:
        meta = {
            "sampler": "adm_dpm_solver_pp_2m",
            "tau_desc": [int(x) for x in tau_desc],
            "steps_total": int(solver.steps_total),
            "continuous_grid": bool(continuous_grid),
            "fm_enabled": bool(float(fm_strength) > 0.0),
            "fm_strength_input": float(fm_strength),
            "fm_strength_effective": float(solver.fm_strength),
            "fm_t_start": float(fm_t_start),
            "fm_lambda_clip": float(fm_lambda_clip),
            "fm_step_clip": float(fm_step_clip),
            "step_ramp_start": float(step_ramp_start),
            "low_noise_threshold_index": int(low_noise_threshold_index),
            "inner_steps_low_noise": int(inner_steps_low_noise),
            "fm_apply_last_k_steps": int(fm_apply_last_k_steps),
            "dynamic_thresholding": bool(dynamic_thresholding),
            "threshold_ratio": float(threshold_ratio),
            "threshold_max_val": float(threshold_max_val),
        }
        return sample, meta
    return sample
