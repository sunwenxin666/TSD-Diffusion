from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import logging
import copy
import random

logger = logging.getLogger(__name__)


# ============================================================
# 1) Core Policy Definitions
# ============================================================

@dataclass(frozen=True)
class SegmentPolicy:
    layer_mask: Tuple[bool, ...]
    stage_cond_gain: Tuple[float, ...]
    stage_gate_scale: Tuple[float, ...]


# ============================================================
# 2) Sampler Policy Definition
# ============================================================

@dataclass(frozen=True)
class SamplerPolicy:
    """Sampler hyperparameters to be searched.

    支持两种模式：
    1. 统一模式：所有时间步使用相同参数
    2. 分段模式：在中间阶段（0.7-0.25）使用更强的矫正策略
    """
    # 基础参数（所有阶段）
    fm_strength: float = 0.04
    fm_t_start: float = 0.85
    order: int = 2


# ============================================================
# 3) Genotype
# ============================================================

@dataclass(frozen=True)
class Genotype:
    """Individual in EA."""
    # ------------------------------------------------------------------
    # Two compatible modes
    # ------------------------------------------------------------------
    # (A) Legacy mode (schedule + segment policies):
    #     - u_schedule participates in search
    #     - policy is selected by u via boundaries
    #
    # (B) NEW mode (fixed timesteps + per-step policies):
    #     - timesteps are given externally (config .npy)
    #     - timesteps DO NOT participate in search
    #     - we search a potentially different policy for EACH step
    #
    # This file keeps both to avoid breaking your stage0 / older scripts.
    # Your current需求 uses (B).

    # --- Legacy: searched schedule (keep for backward compat) ---
    u_schedule: Tuple[float, ...] = field(default_factory=tuple)
    segment_policies: Tuple[SegmentPolicy, ...] = field(default_factory=tuple)
    segment_boundaries: Tuple[float, ...] = field(default_factory=tuple)

    # --- New: per-step policies (length == num_steps == len(timesteps)) ---
    step_policies: Tuple[SegmentPolicy, ...] = field(default_factory=tuple)
    sampler_policy: SamplerPolicy = field(default_factory=SamplerPolicy)

    fitness: float = -1.0
    metrics: Dict[str, float] = field(default_factory=dict)

    # -------------------------
    # Deterministic policy repair
    # -------------------------
    def _repair_policy(
        self,
        policy: SegmentPolicy,
        mask_len: int,
        total_stages: int,
        deterministic: bool = True,
        rng: Optional[np.random.RandomState] = None,
        # 兼容：若你允许 full-mask（覆盖全部 wrapped blocks），把 full_mask_len 也传进来
        full_mask_len: Optional[int] = None,
    ) -> SegmentPolicy:
        """
        Ensure policy fields have correct shapes and are non-empty.

        mask_len:
          - 推荐传 “skippable mask 长度”
          - 如果你仍用旧逻辑，也可以传 “total_layers”（full mask 长度）

        full_mask_len:
          - 如果提供，则允许 layer_mask 长度为 mask_len 或 full_mask_len（二选一合法）
        """
        if rng is None:
            # Only used when deterministic=False (or if caller passes rng explicitly).
            rng = np.random.RandomState(0)

        mask_len = int(mask_len)
        total_stages = int(total_stages)
        if full_mask_len is not None:
            full_mask_len = int(full_mask_len)

        # --------------------
        # layer_mask
        # --------------------
        lm = policy.layer_mask

        def _make_all_true(n: int) -> Tuple[bool, ...]:
            return tuple([True] * int(n)) if n > 0 else tuple()

        # Determine which length we want to repair to
        # Priority:
        #   1) if full_mask_len provided and current mask matches it -> keep it
        #   2) else repair to mask_len
        target_len = mask_len
        if full_mask_len is not None and lm is not None and len(lm) == full_mask_len:
            target_len = full_mask_len

        if lm is None or len(lm) != target_len:
            lm = _make_all_true(target_len)

        if len(lm) > 0 and (not any(lm)):
            lm_list = list(lm)
            if deterministic:
                lm_list[0] = True
            else:
                lm_list[int(rng.randint(0, target_len))] = True
            lm = tuple(lm_list)

        # --------------------
        # stage_cond_gain
        # --------------------
        cg = policy.stage_cond_gain
        if cg is None or len(cg) != total_stages:
            cg = tuple([1.0] * total_stages)

        # --------------------
        # stage_gate_scale
        # --------------------
        gs = policy.stage_gate_scale
        if gs is None or len(gs) != total_stages:
            gs = tuple([1.0] * total_stages)

        return SegmentPolicy(layer_mask=lm, stage_cond_gain=cg, stage_gate_scale=gs)

    def _normalized_boundaries(self) -> Tuple[float, ...]:
        """Return boundaries sorted descending and clipped to [0,1]."""
        if not self.segment_boundaries:
            return tuple()
        b = [float(np.clip(x, 0.0, 1.0)) for x in self.segment_boundaries]
        b_sorted = tuple(sorted(b, reverse=True))
        return b_sorted

    def get_policy_for_u(
        self,
        u: float,
        mask_len: Optional[int] = None,
        total_stages: Optional[int] = None,
        full_mask_len: Optional[int] = None,
    ) -> SegmentPolicy:
        """
        Get segment policy for time u.

        - mask_len：推荐传 skippable mask 长度（len(adapter._skippable_indices)）
        - full_mask_len：可选，传 adapter.total_layers 以兼容 full-mask

        Boundary semantics (descending boundaries):
          boundaries = [b0, b1, ..., b_{k-1}] with b0 > b1 > ...
          segments = k+1
          seg0: u >= b0
          seg1: b0 > u >= b1
          ...
          segk: u < b_{k-1}
        """
        if not self.segment_policies:
            raise ValueError("Genotype has no segment policies defined.")

        u = float(np.clip(u, 0.0, 1.0))
        boundaries = self._normalized_boundaries()

        if boundaries:
            expected = len(boundaries) + 1
            if len(self.segment_policies) != expected:
                logger.warning(
                    "segment_policies count (%d) != boundaries+1 (%d). Will clamp segment index.",
                    len(self.segment_policies),
                    expected,
                )
            idx = 0
            for boundary in boundaries:
                if u >= boundary:
                    break
                idx += 1
            idx = min(idx, len(self.segment_policies) - 1)
        else:
            idx = 0

        policy = self.segment_policies[idx]

        if mask_len is not None and total_stages is not None:
            policy = self._repair_policy(
                policy,
                mask_len=int(mask_len),
                total_stages=int(total_stages),
                deterministic=True,
                rng=None,
                full_mask_len=full_mask_len,
            )

        return policy

    # -------------------------
    # NEW: per-step policy access (for fixed discrete timesteps)
    # -------------------------
    def get_policy_for_step(
        self,
        step_idx: int,
        mask_len: Optional[int] = None,
        total_stages: Optional[int] = None,
        full_mask_len: Optional[int] = None,
    ) -> SegmentPolicy:
        """Get policy for a discrete step index.

        This is the entry used by the *fixed timesteps* pipeline:
        - timesteps are loaded from npy and sorted externally
        - model_fn will map current t to a step index
        - step_policies[step_idx] is applied
        """
        if not self.step_policies:
            # If caller mistakenly uses step mode but genotype has no step_policies,
            # fallback to legacy single segment policy if possible.
            if self.segment_policies:
                pol = self.segment_policies[0]
            else:
                raise ValueError("Genotype has no step_policies nor segment_policies defined.")
        else:
            idx = int(np.clip(int(step_idx), 0, len(self.step_policies) - 1))
            pol = self.step_policies[idx]

        if mask_len is not None and total_stages is not None:
            pol = self._repair_policy(
                pol,
                mask_len=int(mask_len),
                total_stages=int(total_stages),
                deterministic=True,
                rng=None,
                full_mask_len=full_mask_len,
            )
        return pol


# ============================================================
# Adapter-aligned helper
# ============================================================

def infer_mask_len_from_adapter(adapter: Any) -> int:
    """
    Recommended mask length:
      - if adapter has _skippable_indices (improved UNetAdapter), use that
      - else fallback to adapter.total_layers
    """
    if hasattr(adapter, "_skippable_indices"):
        try:
            n = len(getattr(adapter, "_skippable_indices"))
            if n > 0:
                return int(n)
        except Exception:
            pass
    if hasattr(adapter, "total_layers"):
        return int(adapter.total_layers)
    raise AttributeError("Cannot infer mask length: adapter has no _skippable_indices nor total_layers")


# ============================================================
# Convenience helpers (baseline policies)
# ============================================================

def make_all_true_segment_policy(mask_len: int, total_stages: int) -> SegmentPolicy:
    lm = tuple([True] * int(mask_len)) if mask_len and mask_len > 0 else tuple()
    cg = tuple([1.0] * int(total_stages)) if total_stages and total_stages > 0 else tuple()
    gs = tuple([1.0] * int(total_stages)) if total_stages and total_stages > 0 else tuple()
    return SegmentPolicy(layer_mask=lm, stage_cond_gain=cg, stage_gate_scale=gs)


# ============================================================
# Mask / vector operators
# ============================================================

def _ensure_at_least_one_true(mask: np.ndarray, rng: np.random.RandomState, deterministic: bool = False) -> np.ndarray:
    if mask.size == 0:
        return mask
    if not mask.any():
        if deterministic:
            mask[0] = True
        else:
            mask[rng.randint(0, mask.size)] = True
    return mask


def random_layer_mask(
    mask_len: int,
    rng: np.random.RandomState,
    p_keep_range: Tuple[float, float] = (0.5, 1.0),
) -> Tuple[bool, ...]:
    if mask_len <= 0:
        return tuple()
    p_keep = float(rng.uniform(p_keep_range[0], p_keep_range[1]))
    mask = rng.rand(mask_len) < p_keep
    mask = _ensure_at_least_one_true(mask, rng, deterministic=False)
    return tuple(bool(x) for x in mask)


def random_stage_vector(
    total_stages: int,
    rng: np.random.RandomState,
    low: float = 0.8,
    high: float = 1.2,
) -> Tuple[float, ...]:
    if total_stages <= 0:
        return tuple()
    v = rng.uniform(low, high, size=total_stages).astype(np.float32)
    return tuple(float(x) for x in v)


def make_random_segment_policy(
    mask_len: int,
    total_stages: int,
    rng: np.random.RandomState,
    p_keep_range: Tuple[float, float] = (0.5, 1.0),
    gain_range: Tuple[float, float] = (0.8, 1.2),
    gate_range: Tuple[float, float] = (0.8, 1.2),
) -> SegmentPolicy:
    return SegmentPolicy(
        layer_mask=random_layer_mask(mask_len, rng, p_keep_range=p_keep_range),
        stage_cond_gain=random_stage_vector(total_stages, rng, low=gain_range[0], high=gain_range[1]),
        stage_gate_scale=random_stage_vector(total_stages, rng, low=gate_range[0], high=gate_range[1]),
    )


def mutate_layer_mask(
    layer_mask: Tuple[bool, ...],
    rng: np.random.RandomState,
    flip_prob: float = 0.1,
) -> Tuple[bool, ...]:
    if layer_mask is None or len(layer_mask) == 0:
        return layer_mask
    mask = np.array(layer_mask, dtype=np.bool_)
    flips = rng.rand(mask.size) < float(flip_prob)
    mask = np.logical_xor(mask, flips)
    mask = _ensure_at_least_one_true(mask, rng, deterministic=False)
    return tuple(bool(x) for x in mask)


def mutate_stage_vector(
    vec: Tuple[float, ...],
    rng: np.random.RandomState,
    sigma: float = 0.05,
    clamp: Tuple[float, float] = (0.5, 1.5),
    mutate_prob: float = 0.7,
) -> Tuple[float, ...]:
    if vec is None or len(vec) == 0:
        return vec
    v = np.array(vec, dtype=np.float32)
    if rng.rand() < float(mutate_prob):
        v = v + rng.normal(0.0, float(sigma), size=v.shape).astype(np.float32)
    v = np.clip(v, float(clamp[0]), float(clamp[1]))
    return tuple(float(x) for x in v)


def mutate_segment_policy(
    policy: SegmentPolicy,
    mask_len: int,
    total_stages: int,
    rng: np.random.RandomState,
    mask_flip_prob: float = 0.08,
    gain_sigma: float = 0.05,
    gate_sigma: float = 0.05,
) -> SegmentPolicy:
    if policy is None:
        return make_random_segment_policy(mask_len, total_stages, rng)

    lm = policy.layer_mask
    if lm is None or len(lm) != mask_len:
        lm = random_layer_mask(mask_len, rng, p_keep_range=(0.5, 1.0))
    lm = mutate_layer_mask(lm, rng, flip_prob=mask_flip_prob)

    cg = policy.stage_cond_gain
    if cg is None or len(cg) != total_stages:
        cg = random_stage_vector(total_stages, rng, low=0.8, high=1.2)
    cg = mutate_stage_vector(cg, rng, sigma=gain_sigma)

    gs = policy.stage_gate_scale
    if gs is None or len(gs) != total_stages:
        gs = random_stage_vector(total_stages, rng, low=0.8, high=1.2)
    gs = mutate_stage_vector(gs, rng, sigma=gate_sigma)

    return SegmentPolicy(layer_mask=lm, stage_cond_gain=cg, stage_gate_scale=gs)


def crossover_layer_mask(a: Tuple[bool, ...], b: Tuple[bool, ...], rng: np.random.RandomState) -> Tuple[bool, ...]:
    if a is None or b is None or len(a) == 0 or len(a) != len(b):
        return a
    ma = np.array(a, dtype=np.bool_)
    mb = np.array(b, dtype=np.bool_)
    choose_a = rng.rand(ma.size) < 0.5
    child = np.where(choose_a, ma, mb)
    child = _ensure_at_least_one_true(child, rng, deterministic=False)
    return tuple(bool(x) for x in child)


def crossover_stage_vector(a: Tuple[float, ...], b: Tuple[float, ...], rng: np.random.RandomState) -> Tuple[float, ...]:
    if a is None or b is None or len(a) == 0 or len(a) != len(b):
        return a
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    choose_a = rng.rand(va.size) < 0.5
    child = np.where(choose_a, va, vb)
    return tuple(float(x) for x in child)


def crossover_segment_policy(
    pa: SegmentPolicy,
    pb: SegmentPolicy,
    mask_len: int,
    total_stages: int,
    rng: np.random.RandomState,
) -> SegmentPolicy:
    if pa is None:
        pa = make_random_segment_policy(mask_len, total_stages, rng)
    if pb is None:
        pb = make_random_segment_policy(mask_len, total_stages, rng)

    # Repair lengths if needed by re-sampling
    if pa.layer_mask is None or len(pa.layer_mask) != mask_len:
        pa = make_random_segment_policy(mask_len, total_stages, rng)
    if pb.layer_mask is None or len(pb.layer_mask) != mask_len:
        pb = make_random_segment_policy(mask_len, total_stages, rng)
    if pa.stage_cond_gain is None or len(pa.stage_cond_gain) != total_stages:
        pa = make_random_segment_policy(mask_len, total_stages, rng)
    if pb.stage_cond_gain is None or len(pb.stage_cond_gain) != total_stages:
        pb = make_random_segment_policy(mask_len, total_stages, rng)

    lm = crossover_layer_mask(pa.layer_mask, pb.layer_mask, rng)
    cg = crossover_stage_vector(pa.stage_cond_gain, pb.stage_cond_gain, rng)
    gs = crossover_stage_vector(pa.stage_gate_scale, pb.stage_gate_scale, rng)
    return SegmentPolicy(layer_mask=lm, stage_cond_gain=cg, stage_gate_scale=gs)


# ============================================================
# u_schedule utilities (stable, innovation-preserving)
# ============================================================

def make_default_schedule(num_steps: int, N: int = 1000) -> Tuple[float, ...]:
    if num_steps <= 1:
        return (1.0,)
    eps = 1.0 / float(N)
    vals = np.linspace(1.0, eps, num_steps, dtype=np.float32)
    return tuple(float(x) for x in vals)


def sanitize_u_schedule(
    u_schedule: Tuple[float, ...],
    N: int = 1000,
    target_length: Optional[int] = None,
    low_noise_min_count: int = 2,
    low_noise_t_threshold: int = 100,
) -> Tuple[float, ...]:
    if not u_schedule:
        return u_schedule

    eps = 1.0 / float(N)
    u = np.array(u_schedule, dtype=np.float32)

    if target_length is not None and len(u) != int(target_length):
        x_old = np.linspace(0.0, 1.0, len(u), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, int(target_length), dtype=np.float32)
        u = np.interp(x_new, x_old, u).astype(np.float32)

    u = np.clip(u, eps, 1.0)
    u[0] = 1.0
    u[-1] = max(u[-1], eps)

    delta = eps / (len(u) * 2.0)
    for i in range(len(u) - 1):
        if u[i] <= u[i + 1]:
            u[i + 1] = max(eps, u[i] - delta)

    low_thr = float(low_noise_t_threshold) / float(N)
    if low_noise_min_count > 0 and len(u) >= max(2, low_noise_min_count):
        cnt = int(np.sum(u < low_thr))
        if cnt < low_noise_min_count:
            for k in range(low_noise_min_count):
                idx = len(u) - 1 - k
                if idx <= 0:
                    break
                upper = min(low_thr - eps, u[idx - 1] - delta)
                lower = eps
                u[idx] = float(np.clip(u[idx], lower, upper))

            for i in range(len(u) - 1):
                if u[i] <= u[i + 1]:
                    u[i + 1] = max(eps, u[i] - delta)

            cnt2 = int(np.sum(u < low_thr))
            if cnt2 < low_noise_min_count:
                for k in range(low_noise_min_count):
                    idx = len(u) - 1 - k
                    if idx <= 0:
                        break
                    target = max(eps, min(low_thr - eps, (k + 1) * (low_thr / (low_noise_min_count + 2.0))))
                    u[idx] = min(u[idx], target)
                for i in range(len(u) - 1):
                    if u[i] <= u[i + 1]:
                        u[i + 1] = max(eps, u[i] - delta)

    return tuple(float(x) for x in u)


def mutate_schedule(
    u_schedule: Tuple[float, ...],
    rng: np.random.RandomState,
    N: int = 1000,
    target_length: Optional[int] = None,
) -> Tuple[float, ...]:
    if u_schedule is None or len(u_schedule) <= 1:
        return u_schedule

    eps = 1.0 / float(N)
    u = np.array(u_schedule, dtype=np.float32)

    if target_length is None:
        target_length = len(u)

    for i in range(1, len(u) - 1):
        if rng.rand() < 0.5:
            u[i] = u[i] + rng.normal(0.0, 0.02)

    u = np.clip(u, eps, 1.0)
    u[0] = 1.0
    u[-1] = max(u[-1], eps)

    delta = eps / (len(u) * 2.0)
    for i in range(len(u) - 1):
        if u[i] <= u[i + 1]:
            u[i + 1] = max(eps, u[i] - delta)

    return sanitize_u_schedule(tuple(float(x) for x in u), N=N, target_length=int(target_length))


# ============================================================
# Boundaries utilities
# ============================================================

def make_random_boundaries(num_segments: int, rng: np.random.RandomState) -> Tuple[float, ...]:
    if num_segments <= 1:
        return tuple()
    b = rng.uniform(0.1, 0.9, size=num_segments - 1).astype(np.float32)
    b.sort()
    b = b[::-1]
    return tuple(float(x) for x in b)


def mutate_boundaries(boundaries: Tuple[float, ...], rng: np.random.RandomState) -> Tuple[float, ...]:
    if boundaries is None or len(boundaries) == 0:
        return boundaries
    b = np.array(boundaries, dtype=np.float32)
    for i in range(len(b)):
        if rng.rand() < 0.5:
            b[i] = b[i] + rng.normal(0.0, 0.05)
    b = np.clip(b, 0.05, 0.95)
    b.sort()
    b = b[::-1]
    return tuple(float(x) for x in b)


# ============================================================
# Sampler policy operators
# ============================================================

def mutate_sampler_policy(sp: SamplerPolicy, rng: np.random.RandomState) -> SamplerPolicy:
    fm_strength = float(np.clip(sp.fm_strength + rng.normal(0.0, 0.03), 0.0, 0.3))
    fm_t_start = float(np.clip(sp.fm_t_start + rng.normal(0.0, 0.05), 0.5, 0.95))
    order = int(sp.order)
    if rng.rand() < 0.2:
        order = int(rng.choice([1, 2, 3]))
    return SamplerPolicy(fm_strength=fm_strength, fm_t_start=fm_t_start, order=order)


def crossover_sampler_policy(a: SamplerPolicy, b: SamplerPolicy, rng: np.random.RandomState) -> SamplerPolicy:
    fm_strength = float(a.fm_strength if rng.rand() < 0.5 else b.fm_strength)
    fm_t_start = float(a.fm_t_start if rng.rand() < 0.5 else b.fm_t_start)
    order = int(a.order if rng.rand() < 0.5 else b.order)
    return SamplerPolicy(fm_strength=fm_strength, fm_t_start=fm_t_start, order=order)


# ============================================================
# Improved schedule init/mutation with low-noise bias (keeps diversity)
# ============================================================

def make_random_schedule_low_noise_bias(
    num_steps: int,
    N: int = 1000,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, ...]:
    if rng is None:
        rng = np.random.RandomState(0)
    if num_steps <= 1:
        return (1.0,)

    eps = 1.0 / float(N)

    if num_steps >= 6:
        # 优化后的时间步长分配策略
        # 粗阶段 (1.0-0.7): 最少步数，仅用于初步去噪
        high_count = max(1, num_steps // 10)  # 10%的步数，至少1步
        high_noise = np.linspace(1.0, 0.7, high_count, dtype=np.float32)
        
        # 中间阶段 (0.7-0.25): 最多步数，这是图像质量决定性阶段
        mid_count = max(2, int(num_steps * 0.6))  # 60%的步数，至少2步
        mid_noise = np.linspace(0.69, 0.25, mid_count, dtype=np.float32)
        
        # 精细阶段 (0.25-0.02): 中等步数，用于细节优化
        low_count = num_steps - high_count - mid_count
        low_noise = np.linspace(0.24, max(eps * 2.0, 0.02), low_count, dtype=np.float32)
        
        schedule = np.concatenate([high_noise, mid_noise, low_noise])
    else:
        schedule = np.linspace(1.0, eps, num_steps, dtype=np.float32)

    schedule = schedule + rng.normal(0.0, 0.01, size=schedule.shape).astype(np.float32)
    schedule = np.clip(schedule, eps, 1.0)

    return sanitize_u_schedule(
        tuple(float(x) for x in schedule),
        N=N,
        target_length=num_steps,
        low_noise_min_count=2,
        low_noise_t_threshold=100,
    )


def mutate_schedule_improved(
    u_schedule: Tuple[float, ...],
    rng: np.random.RandomState,
    N: int = 1000,
    target_length: Optional[int] = None,
    mutation_prob: float = 0.5,
) -> Tuple[float, ...]:
    if u_schedule is None or len(u_schedule) <= 1:
        return u_schedule

    if target_length is None:
        target_length = len(u_schedule)

    eps = 1.0 / float(N)
    low_thr = 100.0 / float(N)

    u = np.array(u_schedule, dtype=np.float32)

    for i in range(1, len(u) - 1):
        base_p = float(mutation_prob)
        p = base_p * (0.3 if u[i] < low_thr else 1.0)
        if rng.rand() < p:
            u[i] = u[i] + rng.normal(0.0, 0.02)

    u = np.clip(u, eps, 1.0)
    u[0] = 1.0
    u[-1] = max(u[-1], eps)

    delta = eps / (len(u) * 2.0)
    for i in range(len(u) - 1):
        if u[i] <= u[i + 1]:
            u[i + 1] = max(eps, u[i] - delta)

    return sanitize_u_schedule(
        tuple(float(x) for x in u),
        N=N,
        target_length=int(target_length),
        low_noise_min_count=2,
        low_noise_t_threshold=100,
    )


# ============================================================
# Validation (aligns with sanitize) + mask_len/full_mask_len support
# ============================================================

def validate_policy(
    genotype: Genotype,
    mask_len: Optional[int] = None,
    total_stages: Optional[int] = None,
    expected_num_steps: Optional[int] = None,
    N: int = 1000,
    low_noise_t_threshold: int = 100,
    low_noise_min_count: int = 2,
    full_mask_len: Optional[int] = None,
) -> bool:
    """
    Validate genotype legality.

    mask_len:
      - 推荐传 skippable mask 长度
    full_mask_len:
      - 可选：传 total_layers，允许 full-mask 合法
    """
    try:
        # ------------------------------------------------------------
        # Mode B (preferred): fixed timesteps + per-step policies
        # ------------------------------------------------------------
        if genotype.step_policies and len(genotype.step_policies) > 0:
            if expected_num_steps is not None and len(genotype.step_policies) != int(expected_num_steps):
                logging.warning(
                    "validate_policy: step_policies length=%d != expected_num_steps=%d",
                    len(genotype.step_policies), int(expected_num_steps)
                )
                return False

            tol = 1e-8

            # step policies shapes
            for i, policy in enumerate(genotype.step_policies):
                if mask_len is not None:
                    ok = (policy.layer_mask is not None and len(policy.layer_mask) == int(mask_len))
                    if (not ok) and (full_mask_len is not None):
                        ok = (policy.layer_mask is not None and len(policy.layer_mask) == int(full_mask_len))
                    if not ok:
                        logging.warning("validate_policy: step %d layer_mask wrong length", i)
                        return False
                    if len(policy.layer_mask) > 0 and (not any(policy.layer_mask)):
                        logging.warning("validate_policy: step %d layer_mask has no True", i)
                        return False

                if total_stages is not None:
                    if policy.stage_cond_gain is None or len(policy.stage_cond_gain) != int(total_stages):
                        logging.warning("validate_policy: step %d stage_cond_gain wrong shape", i)
                        return False
                    if policy.stage_gate_scale is None or len(policy.stage_gate_scale) != int(total_stages):
                        logging.warning("validate_policy: step %d stage_gate_scale wrong shape", i)
                        return False

            # sampler policy ranges (common)
            sp = genotype.sampler_policy
            if sp.fm_strength < 0.0 or sp.fm_strength > 0.3:
                logging.warning("validate_policy: fm_strength out of range")
                return False
            if sp.fm_t_start < 0.5 or sp.fm_t_start > 0.95:
                logging.warning("validate_policy: fm_t_start out of range")
                return False
            if sp.order not in [1, 2, 3]:
                logging.warning("validate_policy: order invalid")
                return False

            return True

        # ------------------------------------------------------------
        # Mode A (legacy): searched u_schedule + segment policies
        # ------------------------------------------------------------
        if genotype.u_schedule is None or len(genotype.u_schedule) == 0:
            logging.warning("validate_policy: u_schedule is empty")
            return False

        eps = 1.0 / float(N)
        tol = 1e-8

        for u in genotype.u_schedule:
            if (u < eps - tol) or (u > 1.0 + tol):
                logging.warning("validate_policy: u=%f out of range [%f, 1.0]", u, eps)
                return False

        for i in range(len(genotype.u_schedule) - 1):
            if not (genotype.u_schedule[i] > genotype.u_schedule[i + 1] + tol):
                logging.warning("validate_policy: u_schedule not strictly decreasing at index %d", i)
                return False

        low_thr = float(low_noise_t_threshold) / float(N)
        low_cnt = sum(1 for u in genotype.u_schedule if u < low_thr)
        if low_cnt < int(low_noise_min_count):
            logging.warning("validate_policy: only %d timesteps in low-noise region (need >= %d)", low_cnt, low_noise_min_count)
            return False

        # boundaries
        boundaries = (
            tuple(sorted([float(np.clip(b, 0.0, 1.0)) for b in genotype.segment_boundaries], reverse=True))
            if genotype.segment_boundaries else tuple()
        )
        if boundaries:
            for i in range(len(boundaries) - 1):
                if not (boundaries[i] > boundaries[i + 1] + tol):
                    logging.warning("validate_policy: boundaries not strictly descending")
                    return False
            if genotype.segment_policies and (len(genotype.segment_policies) != len(boundaries) + 1):
                logging.warning("validate_policy: segment_policies count != boundaries+1")
                return False

        # segment policies shapes
        if genotype.segment_policies:
            for i, policy in enumerate(genotype.segment_policies):
                if mask_len is not None:
                    ok = (policy.layer_mask is not None and len(policy.layer_mask) == int(mask_len))
                    if (not ok) and (full_mask_len is not None):
                        ok = (policy.layer_mask is not None and len(policy.layer_mask) == int(full_mask_len))
                    if not ok:
                        logging.warning("validate_policy: segment %d layer_mask wrong length", i)
                        return False
                    if len(policy.layer_mask) > 0 and (not any(policy.layer_mask)):
                        logging.warning("validate_policy: segment %d layer_mask has no True", i)
                        return False

                if total_stages is not None:
                    if policy.stage_cond_gain is None or len(policy.stage_cond_gain) != int(total_stages):
                        logging.warning("validate_policy: segment %d stage_cond_gain wrong shape", i)
                        return False
                    if policy.stage_gate_scale is None or len(policy.stage_gate_scale) != int(total_stages):
                        logging.warning("validate_policy: segment %d stage_gate_scale wrong shape", i)
                        return False

        # sampler policy ranges
        sp = genotype.sampler_policy
        if sp.fm_strength < 0.0 or sp.fm_strength > 0.3:
            logging.warning("validate_policy: fm_strength out of range")
            return False
        if sp.fm_t_start < 0.5 or sp.fm_t_start > 0.95:
            logging.warning("validate_policy: fm_t_start out of range")
            return False
        if sp.order not in [1, 2, 3]:
            logging.warning("validate_policy: order invalid")
            return False

        return True

    except Exception as e:
        logging.error("validate_policy: error validating genotype: %s", e)
        return False


# ============================================================
# DPM timestep sampling/mutation (unchanged)
# ============================================================

def sample_dpm_timesteps(
    num_steps: int,
    full_timesteps: List[int],
    rng: Optional[random.Random] = None
) -> List[int]:
    if rng is None:
        rng = random.Random()
    use_timesteps = copy.deepcopy(full_timesteps)
    rng.shuffle(use_timesteps)
    selected = use_timesteps[:num_steps]
    selected.sort(reverse=True)
    return selected


def mutate_dpm_timesteps(
    current_timesteps: List[int],
    full_timesteps: List[int],
    mutation_prob: float,
    rng: Optional[random.Random] = None
) -> List[int]:
    if rng is None:
        rng = random.Random()
    candidates = [t for t in full_timesteps if t not in current_timesteps]
    for i in range(len(current_timesteps)):
        if rng.random() < mutation_prob and len(candidates) > 0:
            new_t = rng.choice(candidates)
            current_timesteps[i] = new_t
            candidates.remove(new_t)
    current_timesteps.sort(reverse=True)
    return current_timesteps
