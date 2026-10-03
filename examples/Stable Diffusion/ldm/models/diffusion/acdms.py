import torch
import numpy as np
import math

# ============================================================
# ACDMS-FM (UniPC + Flow Matching Corrector) for VERY FEW steps
# timesteps.npy is DISCRETE IDX (0..N-1), <= 8 steps typical
#
# MODIFIED: reduce extra UNet evals while preserving quality
#   - Do NOT run _predict_x0 (UNet forward) unless correction is actually applied
#   - Apply FM correction only on last K steps (fm_apply_last_k_steps)
#   - Skip early steps + below fm_t_start without paying UNet cost
# ============================================================

class ACDMSFM(torch.nn.Module):

    def __init__(
        self,
        model_fn_noise,
        noise_schedule,
        # Flow matching parameters
        fm_strength: float = 0.04,           # base strength (will be scaled by steps outside)
        fm_t_start: float = 0.8,            # flow-time threshold to start FM
        fm_lambda_clip: float = 0.9,
        # Dynamic thresholding parameters
        dynamic_thresholding_ratio: float = 0.995,
        thresholding_max_val: float = 1.0,
        # Low noise parameters
        inner_steps_low_noise: int = 4,      # will be overwritten by sampler rule
        low_noise_threshold_index: int = 150, # IMPORTANT: discrete index threshold
        # NEW: only apply FM in the last K solver steps (biggest speed win)
        fm_apply_last_k_steps: int = 1,
    ):
        super().__init__()

        # ==================== Core model setup ====================
        self.model_fn_noise = model_fn_noise
        self.noise_schedule = noise_schedule

        # ==================== UniPC initialization ====================
        from ldm.models.diffusion.unipc import UniPC

        self.unipc = UniPC(
            model_fn=model_fn_noise,
            noise_schedule=noise_schedule,
            algorithm_type="noise_prediction",
            correcting_x0_fn=None,
            correcting_xt_fn=self.flow_matching_correct_step,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            thresholding_max_val=thresholding_max_val,
            variant="bh1",
        )

        # ==================== Flow matching configuration ====================
        self.fm_strength = float(fm_strength)
        self.fm_t_start = float(fm_t_start)
        self.fm_lambda_clip = float(fm_lambda_clip)

        # ==================== Low noise configuration ====================
        self.inner_steps_low_noise = int(inner_steps_low_noise)
        self.low_noise_threshold_index = int(low_noise_threshold_index)

        # ==================== NEW: last-k-step gating ====================
        self.fm_apply_last_k_steps = int(fm_apply_last_k_steps)
        self._steps_total = None

        # ==================== FM state initialization ====================
        self._fm_prev_x = None
        self._fm_prev_v = None
        self._fm_prev_t_fm = None

        # ==================== Cache total timesteps ====================
        try:
            self._N = int(self.noise_schedule.total_N)  # usually 1000
        except Exception:
            self._N = 1000

    def _reset_fm_state(self):
        self._fm_prev_x = None
        self._fm_prev_v = None
        self._fm_prev_t_fm = None
        self._steps_total = None

    @torch.no_grad()
    def _predict_x0(self, x: torch.Tensor, t: torch.Tensor):
        # t -> [B]
        if t.dim() == 0:
            t_batched = t.expand(x.shape[0])
        else:
            t_batched = t

        eps = self.unipc.noise_prediction_fn(x, t_batched)

        alpha_t = self.noise_schedule.marginal_alpha(t_batched)  # [B]
        sigma_t = self.noise_schedule.marginal_std(t_batched)    # [B]

        # broadcast to x dims
        alpha_view = alpha_t
        sigma_view = sigma_t
        while alpha_view.dim() < x.dim():
            alpha_view = alpha_view.view(-1, *([1] * (x.dim() - 1)))
            sigma_view = sigma_view.view(-1, *([1] * (x.dim() - 1)))

        x0_pred = (x - sigma_view * eps) / (alpha_view + 1e-8)
        x0_pred = self.unipc.dynamic_thresholding_fn(x0_pred)

        return x0_pred, alpha_t, sigma_t

    @torch.no_grad()
    def _t_to_discrete_index(self, t: torch.Tensor) -> int:
        """
        Your sampler uses: t_cont = (idx+1)/N  in (0,1]
        so inverse: idx = round(t*N - 1)
        """
        N = self._N
        td = torch.round(t.float() * float(N) - 1.0)
        td = torch.clamp(td, 0.0, float(N - 1))
        return int(td.item())

    @torch.no_grad()
    def flow_matching_correct_step(self, x: torch.Tensor, t: torch.Tensor, step: int):
        # ==================== Early exit if FM disabled ====================
        if self.fm_strength <= 0.0:
            return x

        # ==================== Gate 0: only apply on last K steps ====================
        # This avoids extra UNet eval on early steps.
        if self.fm_apply_last_k_steps is not None and self.fm_apply_last_k_steps > 0:
            if self._steps_total is not None:
                if step < (self._steps_total - self.fm_apply_last_k_steps):
                    # keep minimal state
                    self._fm_prev_x = x.detach()
                    self._fm_prev_v = None
                    self._fm_prev_t_fm = None
                    return x

        # ==================== Prepare scalar t0 ====================
        t0 = t if t.dim() == 0 else t[0]

        # ==================== Compute discrete idx & low-noise (NO UNet) ====================
        t_discrete = self._t_to_discrete_index(t0)
        is_low_noise = (t_discrete < self.low_noise_threshold_index)

        # ==================== Compute flow-time t_fm (NO UNet) ====================
        t_batched = t0.expand(x.shape[0])
        alpha_t = self.noise_schedule.marginal_alpha(t_batched)  # [B]
        sigma_t = self.noise_schedule.marginal_std(t_batched)    # [B]
        eps_num = 1e-8
        t_fm = alpha_t / (alpha_t + sigma_t + eps_num)  # [B]
        t_fm_val = float(t_fm[0].item())

        # ==================== Gate 1: below fm_t_start and not low-noise => skip WITHOUT UNet ====================
        if (t_fm_val < self.fm_t_start) and (not is_low_noise):
            self._fm_prev_x = x.detach()
            self._fm_prev_v = None
            self._fm_prev_t_fm = t_fm.detach()
            return x

        # ==================== Now we REALLY need x0_pred => pay 1 UNet eval ====================
        x0_pred, _, _ = self._predict_x0(x, t0)

        # ==================== Compute FM direction ====================
        v_fm = x0_pred - x

        # ==================== If below start threshold but low-noise: allow gentle correction ====================
        if t_fm_val < self.fm_t_start:
            # very conservative: tiny lambda to avoid artifacts
            base = float(self.fm_strength * 0.25)
            lambda_env = torch.full((x.shape[0],), base, device=x.device, dtype=x.dtype)
            while lambda_env.dim() < x.dim():
                lambda_env = lambda_env.view(-1, *([1] * (x.dim() - 1)))
            lambda_eff = lambda_env

            if is_low_noise and self.inner_steps_low_noise > 1:
                x_new = x
                for _ in range(self.inner_steps_low_noise):
                    x_new = x_new + lambda_eff * (x0_pred - x_new)
            else:
                x_new = x + lambda_eff * v_fm

            self._fm_prev_x = x.detach()
            self._fm_prev_v = v_fm.detach()
            self._fm_prev_t_fm = t_fm.detach()
            return x_new

        # ==================== 1) Envelope on flow-time (bell curve) ====================
        denom_u = max(1.0 - self.fm_t_start, eps_num)
        u = (t_fm - self.fm_t_start) / denom_u
        u = torch.clamp(u, 0.0, 1.0)

        # bell envelope: 4u(1-u)
        lambda_env = self.fm_strength * (4.0 * u * (1.0 - u))  # [B]

        # ==================== 2) Conservative FM direction early ====================
        trust = u * u
        v_fm = v_fm * trust.view(-1, *([1] * (x.dim() - 1)))

        # ==================== 3) A-FloPS local lambda ====================
        if self._fm_prev_x is not None and self._fm_prev_v is not None:
            delta_x = (x - self._fm_prev_x).reshape(x.shape[0], -1)
            delta_v = (v_fm - self._fm_prev_v).reshape(x.shape[0], -1)
            num = (delta_v * delta_x).sum(dim=1)
            den = delta_x.pow(2).sum(dim=1) + 1e-12
            lambda_local = num / den
            lambda_local = torch.clamp(lambda_local, -self.fm_lambda_clip, self.fm_lambda_clip)
        else:
            lambda_local = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

        # ==================== Broadcast lambdas ====================
        lambda_env = lambda_env.to(device=x.device, dtype=x.dtype)
        lambda_local = lambda_local.to(device=x.device, dtype=x.dtype)

        while lambda_env.dim() < x.dim():
            lambda_env = lambda_env.view(-1, *([1] * (x.dim() - 1)))
        while lambda_local.dim() < x.dim():
            lambda_local = lambda_local.view(-1, *([1] * (x.dim() - 1)))

        lambda_eff = lambda_env * lambda_local

        # ==================== 4) Low-noise extra inner corrections ====================
        if is_low_noise and self.inner_steps_low_noise > 1:
            x_new = x
            for _ in range(self.inner_steps_low_noise):
                x_new = x_new + lambda_eff * (x0_pred - x_new)
        else:
            x_new = x + lambda_eff * v_fm

        # ==================== Update FM state ====================
        self._fm_prev_x = x.detach()
        self._fm_prev_v = v_fm.detach()
        self._fm_prev_t_fm = t_fm.detach()

        return x_new

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        steps: int,
        t_start: float,
        t_end: float,
        order: int = 2,
        skip_type: str = "time_uniform",
        return_intermediate: bool = False,
    ):
        self._reset_fm_state()
        self._steps_total = int(steps)  # NEW: save total steps for last-k gating
        return self.unipc.sample(
            x=x,
            steps=int(steps),
            t_start=float(t_start),
            t_end=float(t_end),
            order=int(order),
            skip_type=str(skip_type),
            method="multistep",
            lower_order_final=True,
            denoise_to_zero=False,
            return_intermediate=return_intermediate,
        )


# ============================================================
# ACDMS Sampler wrapper (supports discrete idx timesteps + CFG schedule)
# ============================================================

from ldm.models.diffusion.unipc import NoiseScheduleVP

class ACDMSSampler(object):
    """
    - timesteps.npy should be a 1D list/np.ndarray of DISCRETE IDX (0..N-1),
      recommended to be DESCENDING and include endpoints (e.g., 999 ... 0).
    - Few steps support:
      * fm_strength scaling by steps
      * inner correction count rule for low-noise
      * CFG schedule: first half high, second half low
    """

    def __init__(
        self,
        model,
        noise_schedule: NoiseScheduleVP,
        # Flow matching parameters
        fm_strength: float = 0.04,   # base (will be scaled by sqrt(6/steps))
        fm_t_start: float = 0.8,
        fm_lambda_clip: float = 0.9,
        # Dynamic thresholding parameters
        dynamic_thresholding_ratio: float = 0.995,
        thresholding_max_val: float = 1.0,
        # Few-step parameters
        low_noise_threshold_index: int = 150,
        # CFG parameters
        cfg_high: float = 5.0,
        cfg_low: float = 3.0,
        cfg_schedule: str = "two_stage",  # "two_stage" or "linear"
        # NEW: last-k-step FM gating
        fm_apply_last_k_steps: int = 1,
    ):
        super().__init__()

        # ==================== Core model setup ====================
        self.model = model
        self.noise_schedule = noise_schedule
        self.ddpm_num_timesteps = model.num_timesteps

        # ==================== Flow matching configuration ====================
        self.fm_strength_base = float(fm_strength)
        self.fm_t_start = float(fm_t_start)
        self.fm_lambda_clip = float(fm_lambda_clip)
        self.fm_apply_last_k_steps = int(fm_apply_last_k_steps)

        # ==================== Dynamic thresholding configuration ====================
        self.dynamic_thresholding_ratio = float(dynamic_thresholding_ratio)
        self.thresholding_max_val = float(thresholding_max_val)

        # ==================== Low noise configuration ====================
        self.low_noise_threshold_index = int(low_noise_threshold_index)

        # ==================== CFG configuration ====================
        self.cfg_high = float(cfg_high)
        self.cfg_low = float(cfg_low)
        self.cfg_schedule = str(cfg_schedule)

        # ==================== Device setup ====================
        self.device = model.betas.device if hasattr(model, "betas") else torch.device("cuda")

    def _inner_steps_rule(self, steps_total: int) -> int:
        """
        Standard for very few steps:
          2 steps -> 8
          4 steps -> 6
          6 steps -> 4
          8 steps -> 3
        """
        table = {2: 8, 4: 4, 6: 4, 8: 2}
        if steps_total in table:
            return table[steps_total]
        return max(2, int(round(24.0 / max(1, steps_total))))

    def _fm_strength_scaled(self, steps_total: int) -> float:
        # fm_strength_eff = fm_strength * sqrt(6/S)
        S = max(1, int(steps_total))
        return float(self.fm_strength_base * math.sqrt(6.0 / float(S)))

    def _cfg_scale_at_step(self, step_idx: int, steps_total: int) -> float:
        if steps_total <= 1:
            return self.cfg_high
        if self.cfg_schedule == "linear":
            a = float(step_idx) / float(steps_total - 1)
            return (1.0 - a) * self.cfg_high + a * self.cfg_low
        half = steps_total // 2
        return self.cfg_high if step_idx < half else self.cfg_low

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        sampled_timestep=None,  # <-- discrete idx list from timesteps.npy
        **kwargs,
    ):
        device = self.device
        C, H, W = shape

        # 1) init x_T
        if x_T is None:
            x = torch.randn((batch_size, C, H, W), device=device)
        else:
            x = x_T.to(device)

        # 2) Build discrete time points (IDX) and continuous time grid
        N = int(getattr(self.noise_schedule, "total_N", 1000))

        if sampled_timestep is not None:
            t_idx = np.array(sampled_timestep, dtype=np.int64).reshape(-1)
        else:
            t_idx = np.linspace(N - 1, 0, int(S) + 1, dtype=np.int64)

        # sort descending (high noise -> low noise)
        t_idx = np.sort(t_idx)[::-1]

        # Continuous time points (0,1]: t = (idx+1)/N
        t_cont = (t_idx.astype(np.float32) + 1.0) / float(N)
        t_cont = torch.from_numpy(t_cont).to(device=device, dtype=torch.float32)

        if sampled_timestep is not None:
            steps_total = max(1, int(len(t_cont)))
        else:
            steps_total = max(1, int(len(t_cont) - 1))

        idx_to_pos = {int(v): int(i) for i, v in enumerate(t_idx.tolist())}

        def t_in_to_step_idx(t_in: torch.Tensor) -> int:
            if t_in.dim() != 0:
                t0 = t_in[0]
            else:
                t0 = t_in
            td = int(torch.round(t0.float() * float(N) - 1.0).clamp(0, N - 1).item())
            pos = idx_to_pos.get(td, None)
            if pos is None:
                diffs = np.abs(t_idx.astype(np.int64) - np.int64(td))
                pos = int(np.argmin(diffs))
            return int(max(0, min(steps_total - 1, pos)))

        # 3) step-aware model_fn_noise with CFG schedule
        def model_fn_noise(x_in: torch.Tensor, t_in: torch.Tensor):
            B = x_in.shape[0]

            if self.noise_schedule.schedule == "discrete":
                t_discrete = t_in * float(N) - 1.0
                t_discrete = t_discrete.clamp(0, N - 1)
                t_discrete = t_discrete.round().long()
            else:
                t_discrete = t_in

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                return self.model.apply_model(x_in, t_discrete, conditioning)

            assert not isinstance(conditioning, dict), \
                "ACDMSSampler: current implementation only supports tensor conditioning."

            cond = conditioning
            uc = unconditional_conditioning

            if cond.shape[0] != B:
                if cond.shape[0] == 1:
                    cond = cond.expand(B, *cond.shape[1:])
                else:
                    raise RuntimeError(f"conditioning batch={cond.shape[0]} != x batch={B}")
            if uc.shape[0] != B:
                if uc.shape[0] == 1:
                    uc = uc.expand(B, *uc.shape[1:])
                else:
                    raise RuntimeError(f"unconditional_conditioning batch={uc.shape[0]} != x batch={B}")

            step_idx = t_in_to_step_idx(t_in if t_in.dim() == 0 else t_in[:1])
            cfg_s = self._cfg_scale_at_step(step_idx, steps_total)

            x_cat = torch.cat([x_in, x_in], dim=0)
            t_cat = torch.cat([t_discrete, t_discrete], dim=0)
            c_in = torch.cat([uc, cond], dim=0)

            e_uncond, e_cond = self.model.apply_model(x_cat, t_cat, c_in).chunk(2)
            e_t = e_uncond + float(cfg_s) * (e_cond - e_uncond)
            return e_t

        # 4) Construct solver with few-step tuned params
        fm_strength_eff = self._fm_strength_scaled(steps_total)
        inner_steps_low_noise = self._inner_steps_rule(steps_total)

        acdms_fm = ACDMSFM(
            model_fn_noise=model_fn_noise,
            noise_schedule=self.noise_schedule,
            fm_strength=fm_strength_eff,
            fm_t_start=self.fm_t_start,
            dynamic_thresholding_ratio=self.dynamic_thresholding_ratio,
            thresholding_max_val=self.thresholding_max_val,
            fm_lambda_clip=self.fm_lambda_clip,
            inner_steps_low_noise=inner_steps_low_noise,
            low_noise_threshold_index=self.low_noise_threshold_index,
            fm_apply_last_k_steps=self.fm_apply_last_k_steps,  # NEW
        )

        # 5) Run sampling on this time grid
        t_start = float(t_cont[0].item())
        t_end = float(t_cont[-1].item())

        x_out = acdms_fm.sample(
            x=x,
            steps=steps_total,
            t_start=t_start,
            t_end=t_end,
            order=2,
            skip_type="time_uniform",
            return_intermediate=False,
        )

        return x_out, None
