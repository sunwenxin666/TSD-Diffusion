import math
from typing import Optional, Any, Tuple, Sequence, Dict

import numpy as np
import torch

# ============================================================
# Utils
# ============================================================

def expand_dims(v: torch.Tensor, dims: int) -> torch.Tensor:
    return v[(...,) + (None,) * (dims - 1)]


def interpolate_fn(x: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)

    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)

    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)

    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)

    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x + 1e-12)
    return cand


def _expand_cond_to_batch(cond: Any, B: int) -> Any:
    if isinstance(cond, torch.Tensor):
        if cond.shape[0] == B:
            return cond
        if cond.shape[0] == 1:
            return cond.expand(B, *cond.shape[1:])
        raise RuntimeError(f"conditioning batch={cond.shape[0]} != x batch={B}")

    if isinstance(cond, dict):
        out: Dict[str, Any] = {}
        for k, v in cond.items():
            if isinstance(v, list):
                out_list = []
                for vv in v:
                    if vv.shape[0] == B:
                        out_list.append(vv)
                    elif vv.shape[0] == 1:
                        out_list.append(vv.expand(B, *vv.shape[1:]))
                    else:
                        raise RuntimeError(f"cond[{k}] batch={vv.shape[0]} != x batch={B}")
                out[k] = out_list
            else:
                if v.shape[0] == B:
                    out[k] = v
                elif v.shape[0] == 1:
                    out[k] = v.expand(B, *v.shape[1:])
                else:
                    raise RuntimeError(f"cond[{k}] batch={v.shape[0]} != x batch={B}")
        return out

    raise TypeError(f"Unsupported cond type: {type(cond)}")


def _cat_cond(uc: Any, cond: Any) -> Any:
    if isinstance(cond, torch.Tensor):
        return torch.cat([uc, cond], dim=0)

    if isinstance(cond, dict):
        out: Dict[str, Any] = {}
        for k in cond.keys():
            v_uc = uc[k]
            v_c = cond[k]
            if isinstance(v_c, list):
                out[k] = [torch.cat([a, b], dim=0) for a, b in zip(v_uc, v_c)]
            else:
                out[k] = torch.cat([v_uc, v_c], dim=0)
        return out

    raise TypeError(f"Unsupported cond type: {type(cond)}")


def prepare_tau_for_dpmsolver(tau: Optional[Sequence[int]], S: int, N: int = 1000) -> np.ndarray:
    target = int(S) + 1
    if tau is None:
        t = np.linspace(N - 1, 0, target, dtype=np.int64)
        return np.sort(t)[::-1]

    t = np.array(tau, dtype=np.int64).reshape(-1)
    # ✅ 强制加两端点：999 和 0
    t = np.concatenate([t, np.array([N - 1, 0], dtype=np.int64)])
    t = np.unique(t).astype(np.int64)
    t = np.sort(t)[::-1]

    if len(t) > target:
        t = t[:target]
        t[0] = N - 1
        t[-1] = 0
        t = np.sort(t)[::-1]
    elif len(t) < target:
        pad = np.linspace(N - 1, 0, target, dtype=np.int64)
        t = np.unique(np.concatenate([t, pad])).astype(np.int64)
        t = np.sort(t)[::-1]
        if len(t) > target:
            t = t[:target]
            t[0] = N - 1
            t[-1] = 0
            t = np.sort(t)[::-1]

    return t


# ============================================================
# NoiseScheduleVP
# ============================================================

class NoiseScheduleVP:
    def __init__(
        self,
        schedule: str = "discrete",
        betas: Optional[torch.Tensor] = None,
        alphas_cumprod: Optional[torch.Tensor] = None,
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
        dtype: torch.dtype = torch.float32,
    ):
        if schedule not in ["discrete", "linear"]:
            raise ValueError(f"Unsupported schedule {schedule}. Need 'discrete' or 'linear'.")

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)

            self.T = 1.0
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.0
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas: torch.Tensor, clipped_lambda: float = -5.1) -> torch.Tensor:
        log_sigmas = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)),
                self.t_array.to(t.device),
                self.log_alpha_array.to(t.device),
            ).reshape((-1,))
        else:
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


# ============================================================
# DPMSolver++ 2M + FM corrector
# (平滑渐进式：step_ramp + trust=u^2 + bell + local lambda + fm_step_clip)
# ============================================================

class DPMSolverV2FM:
    def __init__(
        self,
        model,
        noise_schedule: NoiseScheduleVP,
        conditioning: Any,
        unconditional_conditioning: Optional[Any],
        timesteps_idx_desc: Sequence[int],
        cfg_high: float = 5.0,
        cfg_low: float = 3.0,
        cfg_schedule: str = "two_stage",
        fm_strength_base: float = 0.04,
        fm_t_start: float = 0.85,
        fm_lambda_clip: float = 1.0,
        fm_step_clip: float = 0.02,      # ✅ 建议先用 0.02（更不艳）
        step_ramp_start: float = 0.60,   # ✅ 新增：FM从采样进度 p>=start 开始渐进启用（0..1）
        low_noise_threshold_index: int = 150,
        inner_steps_low_noise: int = 1,
    ):
        self.model = model
        self.ns = noise_schedule
        self.cond = conditioning
        self.uc = unconditional_conditioning

        t_idx = np.array(timesteps_idx_desc, dtype=np.int64).reshape(-1)
        t_idx = np.sort(t_idx)[::-1]
        if len(t_idx) < 2:
            raise ValueError("timesteps_idx_desc 至少需要 2 个点。")
        self.t_idx_desc = t_idx

        self.N = int(getattr(self.ns, "total_N", 1000))
        self.steps_total = int(len(self.t_idx_desc) - 1)

        self.cfg_high = float(cfg_high)
        self.cfg_low = float(cfg_low)
        self.cfg_schedule = str(cfg_schedule)

        self.fm_t_start = float(fm_t_start)
        self.fm_lambda_clip = float(fm_lambda_clip)
        self.fm_step_clip = float(fm_step_clip)
        self.step_ramp_start = float(step_ramp_start)
        self.low_noise_threshold_index = int(low_noise_threshold_index)
        self.inner_steps_low_noise = int(inner_steps_low_noise)

        # few-step scaling
        self.fm_strength = float(fm_strength_base * math.sqrt(6.0 / max(1, self.steps_total)))

        self._fm_prev_x = None
        self._fm_prev_v = None
        self._fm_prev_t_fm = None

        self.idx_to_pos = {int(v): int(i) for i, v in enumerate(self.t_idx_desc.tolist())}

    def _cfg_scale_at_step(self, step_idx: int) -> float:
        if self.steps_total <= 1:
            return self.cfg_high
        if self.cfg_schedule == "linear":
            a = float(step_idx) / float(self.steps_total - 1)
            return (1.0 - a) * self.cfg_high + a * self.cfg_low
        half = self.steps_total // 2
        return self.cfg_high if step_idx < half else self.cfg_low

    def _t_in_to_step_idx(self, t_cont_scalar: torch.Tensor) -> int:
        td = int(torch.round(t_cont_scalar.float() * float(self.N) - 1.0).clamp(0, self.N - 1).item())
        pos = self.idx_to_pos.get(td, None)
        if pos is None:
            diffs = np.abs(self.t_idx_desc.astype(np.int64) - np.int64(td))
            pos = int(np.argmin(diffs))
        return int(max(0, min(self.steps_total - 1, pos)))

    @torch.no_grad()
    def model_fn_noise(self, x_in: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        B = x_in.shape[0]
        t0 = t_cont[0] if t_cont.dim() != 0 else t_cont
        t_discrete = (t0 * float(self.N) - 1.0).clamp(0, self.N - 1).round().long().expand(B)

        if self.uc is None:
            return self.model.apply_model(x_in, t_discrete, self.cond)

        step_idx = self._t_in_to_step_idx(t0)
        cfg_s = self._cfg_scale_at_step(step_idx)

        cond = _expand_cond_to_batch(self.cond, B)
        uc = _expand_cond_to_batch(self.uc, B)

        x_cat = torch.cat([x_in, x_in], dim=0)
        t_cat = torch.cat([t_discrete, t_discrete], dim=0)
        c_cat = _cat_cond(uc, cond)

        e_uc, e_c = self.model.apply_model(x_cat, t_cat, c_cat).chunk(2)
        return e_uc + float(cfg_s) * (e_c - e_uc)

    def _reset_fm_state(self):
        self._fm_prev_x = None
        self._fm_prev_v = None
        self._fm_prev_t_fm = None

    @torch.no_grad()
    def _predict_x0(self, x: torch.Tensor, t_cont: torch.Tensor):
        t_b = t_cont.expand(x.shape[0]) if t_cont.dim() == 0 else t_cont
        eps = self.model_fn_noise(x, t_b)

        alpha_t = self.ns.marginal_alpha(t_b)
        sigma_t = self.ns.marginal_std(t_b)

        a = alpha_t
        s = sigma_t
        while a.dim() < x.dim():
            a = a.view(-1, *([1] * (x.dim() - 1)))
            s = s.view(-1, *([1] * (x.dim() - 1)))

        x0 = (x - s * eps) / (a + 1e-8)
        return x0, alpha_t, sigma_t

    @torch.no_grad()
    def _t_to_discrete_index(self, t_cont_scalar: torch.Tensor) -> int:
        td = torch.round(t_cont_scalar.float() * float(self.N) - 1.0)
        td = torch.clamp(td, 0.0, float(self.N - 1))
        return int(td.item())

    @torch.no_grad()
    def flow_matching_correct_step(self, x: torch.Tensor, t_cont: torch.Tensor, step: int) -> torch.Tensor:
        if self.fm_strength <= 0.0:
            return x

        x0_pred, alpha_t, sigma_t = self._predict_x0(x, t_cont)

        eps_num = 1e-8
        t_fm = alpha_t / (alpha_t + sigma_t + eps_num)
        t_fm_val = float(t_fm[0].item())

        v_fm = x0_pred - x

        # fm_t_start gate：过早不校正，但会更新状态（平滑）
        if t_fm_val < self.fm_t_start:
            self._fm_prev_x = x.detach()
            self._fm_prev_v = v_fm.detach()
            self._fm_prev_t_fm = t_fm.detach()
            return x

        # bell envelope
        denom_u = max(1.0 - self.fm_t_start, eps_num)
        u = (t_fm - self.fm_t_start) / denom_u
        u = torch.clamp(u, 0.0, 1.0)
        lambda_env = self.fm_strength * (4.0 * u * (1.0 - u))  # [B]

        # ✅ step_ramp：按采样进度平滑启用（解决“突跃”）
        if self.steps_total <= 1:
            ramp = torch.ones_like(lambda_env)
        else:
            p = float(step) / float(max(1, self.steps_total - 1))  # 0..1
            ramp_val = (p - self.step_ramp_start) / max(1e-8, (1.0 - self.step_ramp_start))
            ramp_val = float(max(0.0, min(1.0, ramp_val)))
            ramp_val = ramp_val * ramp_val  # 平滑
            ramp = torch.ones_like(lambda_env) * ramp_val
        lambda_env = lambda_env * ramp

        # conservative early direction: trust=u^2
        trust = u * u
        v_fm = v_fm * trust.view(-1, *([1] * (x.dim() - 1)))

        # local lambda (A-FloPS)
        if self._fm_prev_x is not None and self._fm_prev_v is not None:
            delta_x = (x - self._fm_prev_x).reshape(x.shape[0], -1)
            delta_v = (v_fm - self._fm_prev_v).reshape(x.shape[0], -1)
            num = (delta_v * delta_x).sum(dim=1)
            den = delta_x.pow(2).sum(dim=1) + 1e-12
            lambda_local = torch.clamp(num / den, -self.fm_lambda_clip, self.fm_lambda_clip)
        else:
            lambda_local = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

        lambda_env = lambda_env.to(device=x.device, dtype=x.dtype)
        lambda_local = lambda_local.to(device=x.device, dtype=x.dtype)

        while lambda_env.dim() < x.dim():
            lambda_env = lambda_env.view(-1, *([1] * (x.dim() - 1)))
        while lambda_local.dim() < x.dim():
            lambda_local = lambda_local.view(-1, *([1] * (x.dim() - 1)))

        lambda_eff = lambda_env * lambda_local

        # ✅ 饱和度刹车
        lambda_eff = torch.clamp(lambda_eff, -self.fm_step_clip, self.fm_step_clip)

        # low-noise inner correction (可保持1，先别加太大)
        t0 = t_cont[0] if t_cont.dim() != 0 else t_cont
        t_discrete = self._t_to_discrete_index(t0)
        is_low_noise = (t_discrete < self.low_noise_threshold_index)

        if is_low_noise and self.inner_steps_low_noise > 1:
            x_new = x
            for _ in range(self.inner_steps_low_noise):
                x_new = x_new + lambda_eff * (x0_pred - x_new)
        else:
            x_new = x + lambda_eff * v_fm

        # update state
        self._fm_prev_x = x.detach()
        self._fm_prev_v = v_fm.detach()
        self._fm_prev_t_fm = t_fm.detach()

        return x_new

    @torch.no_grad()
    def _x0_model(self, x: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        x0, _, _ = self._predict_x0(x, t_cont)
        return x0

    @torch.no_grad()
    def first_order_update(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, model_s: Optional[torch.Tensor]) -> torch.Tensor:
        ns = self.ns
        lambda_s = ns.marginal_lambda(s)
        lambda_t = ns.marginal_lambda(t)
        h = lambda_t - lambda_s

        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s = ns.marginal_std(s)
        sigma_t = ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self._x0_model(x, s)

        return (sigma_t / (sigma_s + 1e-12)) * x - expand_dims(alpha_t, x.dim()) * expand_dims(phi_1, x.dim()) * model_s

    @torch.no_grad()
    def second_order_multistep_update(self, x: torch.Tensor,
                                     model_prev_1: torch.Tensor, model_prev_0: torch.Tensor,
                                     t_prev_1: torch.Tensor, t_prev_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ns = self.ns
        lambda_prev_1 = ns.marginal_lambda(t_prev_1)
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / (h + 1e-12)

        D1_0 = (1.0 / (r0 + 1e-12)) * (model_prev_0 - model_prev_1)

        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0 = ns.marginal_std(t_prev_0)
        sigma_t = ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(-h)

        return (
            (sigma_t / (sigma_prev_0 + 1e-12)) * x
            - expand_dims(alpha_t, x.dim()) * expand_dims(phi_1, x.dim()) * model_prev_0
            - 0.5 * expand_dims(alpha_t, x.dim()) * expand_dims(phi_1, x.dim()) * D1_0
        )

    @torch.no_grad()
    def sample(self, x_T: torch.Tensor, apply_fm_corrector: bool = True) -> torch.Tensor:
        self._reset_fm_state()
        t_list = (self.t_idx_desc.astype(np.float32) + 1.0) / float(self.N)
        t_list = torch.from_numpy(t_list).to(device=x_T.device, dtype=torch.float32)

        K = int(t_list.shape[0])
        x = x_T

        t_prev_0 = t_list[0:1]
        model_prev_0 = self._x0_model(x, t_prev_0)

        if apply_fm_corrector:
            x = self.flow_matching_correct_step(x, t_prev_0, step=0)

        t1 = t_list[1:2]
        x = self.first_order_update(x, t_prev_0, t1, model_s=model_prev_0)
        if apply_fm_corrector:
            x = self.flow_matching_correct_step(x, t1, step=0)

        t_prev_1 = t_prev_0
        model_prev_1 = model_prev_0
        t_prev_0 = t1
        model_prev_0 = self._x0_model(x, t_prev_0)

        for step in range(1, K - 1):
            t_next = t_list[step + 1 : step + 2]
            x = self.second_order_multistep_update(x, model_prev_1, model_prev_0, t_prev_1, t_prev_0, t_next)
            if apply_fm_corrector:
                x = self.flow_matching_correct_step(x, t_next, step=step)
            t_prev_1, t_prev_0 = t_prev_0, t_next
            model_prev_1, model_prev_0 = model_prev_0, self._x0_model(x, t_prev_0)

        return x


# ============================================================
# Wrapper sampler + teacher helper
# ============================================================

class DPMSolverV2FMSampler:
    def __init__(
        self,
        model,
        betas: torch.Tensor,
        fm_strength: float = 0.08,
        fm_t_start: float = 0.85,
        fm_lambda_clip: float = 1.0,
        fm_step_clip: float = 0.02,
        step_ramp_start: float = 0.60,
        low_noise_threshold_index: int = 150,
        cfg_schedule: str = "two_stage",
    ):
        self.model = model
        self.betas = betas
        self.noise_schedule = NoiseScheduleVP("discrete", betas=betas)

        self.fm_strength = float(fm_strength)
        self.fm_t_start = float(fm_t_start)
        self.fm_lambda_clip = float(fm_lambda_clip)
        self.fm_step_clip = float(fm_step_clip)
        self.step_ramp_start = float(step_ramp_start)
        self.low_noise_threshold_index = int(low_noise_threshold_index)
        self.cfg_schedule = str(cfg_schedule)

    @torch.no_grad()
    def sample(
        self,
        S: int,
        batch_size: int,
        shape: Tuple[int, int, int],
        conditioning: Any,
        unconditional_conditioning: Any,
        unconditional_guidance_scale: float,
        sampled_timestep: Optional[Sequence[int]],
        x_T: torch.Tensor,
        verbose: bool = False,
    ):
        N = int(getattr(self.noise_schedule, "total_N", 1000))
        tau_desc = prepare_tau_for_dpmsolver(sampled_timestep, S=int(S), N=N)

        if verbose:
            print(f"[DPMSolverV2FMSampler] tau_desc(len={len(tau_desc)}): {tau_desc.tolist()}")

        solver = DPMSolverV2FM(
            model=self.model,
            noise_schedule=self.noise_schedule,
            conditioning=conditioning,
            unconditional_conditioning=unconditional_conditioning,
            timesteps_idx_desc=tau_desc,
            cfg_high=float(unconditional_guidance_scale),
            cfg_low=float(unconditional_guidance_scale),
            cfg_schedule=self.cfg_schedule,
            fm_strength_base=self.fm_strength,
            fm_t_start=self.fm_t_start,
            fm_lambda_clip=self.fm_lambda_clip,
            fm_step_clip=self.fm_step_clip,
            step_ramp_start=self.step_ramp_start,
            low_noise_threshold_index=self.low_noise_threshold_index,
            inner_steps_low_noise=1,
        )
        latents = solver.sample(x_T, apply_fm_corrector=True)
        return latents, {"tau_desc": tau_desc.tolist()}


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
    fm_t_start: float = 0.85,
    fm_lambda_clip: float = 1.0,
    fm_step_clip: float = 0.02,
    step_ramp_start: float = 0.60,
    verbose: bool = False,
):
    betas = pipe.model.betas
    sampler = DPMSolverV2FMSampler(
        model=pipe.model,
        betas=betas,
        fm_strength=fm_strength,
        fm_t_start=fm_t_start,
        fm_lambda_clip=fm_lambda_clip,
        fm_step_clip=fm_step_clip,
        step_ramp_start=step_ramp_start,
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
        x_T=latents,
        verbose=verbose,
    )

    if verbose:
        print("[teacher_dpm_solver_sample] info:", info)

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


"""
调参建议（解决“过艳”的最有效旋钮）：
- fm_step_clip：0.015 ~ 0.03（越小越不艳）
- step_ramp_start：0.55 ~ 0.70（越大越晚启用FM，越不艳/越稳）
- fm_t_start：0.80 ~ 0.90（越大越晚启用FM）
"""
