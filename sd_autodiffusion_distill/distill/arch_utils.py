"""
架构编码/解码与操作工具
- 联合候选：tau(时间步序列) + channel_scales(通道缩放)
- 支持：固定 tau 长度（用于更稳定的联合搜索）
- Scheme A：通道量化/整除约束的 build_compressed_unet
"""
import random
import copy
import numpy as np
import torch
from typing import List, Optional
from ldm.util import instantiate_from_config
from torch import nn


# =========================
# Candidate
# =========================
class ArchCandidate:
    """联合搜索的候选个体：tau + 通道缩放"""

    def __init__(self, tau: List[int], channel_scales: List[float]):
        """
        Args:
            tau: 时间步序列（会强制降序），如 [999, 800, 600, ...]
            channel_scales: 通道缩放比例，长度为 num_layers
        """
        self.tau = sorted(list(map(int, tau)), reverse=True)
        self.channel_scales = list(map(float, channel_scales))
        self.fid = None
        self.params = None
        self.fitness = None

    def __repr__(self):
        return f"ArchCandidate(tau={self.tau}, scales={self.channel_scales[:3]}..., FID={self.fid}, params={self.params}M)"

    def to_dict(self):
        return {
            "tau": self.tau,
            "channel_scales": self.channel_scales,
            "fid": float(self.fid) if self.fid is not None else None,
            "params": float(self.params) if self.params is not None else None,
            "fitness": float(self.fitness) if self.fitness is not None else None,
        }

    @classmethod
    def from_dict(cls, d):
        cand = cls(d["tau"], d["channel_scales"])
        cand.fid = d.get("fid")
        cand.params = d.get("params")
        cand.fitness = d.get("fitness")
        return cand


# =========================
# tau helpers (seed / sanitize / fixed-length mutation)
# =========================
def load_tau_seed(npy_path: str, mode: str = "index", min_idx: int = 1, max_idx: int = 999) -> List[int]:
    """
    mode:
      - "index": npy 是 [0..999] 或 [1..999] 的整数 timestep
      - "float01": npy 是 [0..1] 浮点，会映射到 [0..max_idx]
    """
    arr = np.load(npy_path)
    arr = np.array(arr).reshape(-1)

    if mode == "float01":
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.rint(arr * max_idx).astype(np.int64)
    else:
        arr = np.rint(arr).astype(np.int64)

    tau = arr.tolist()
    tau = sanitize_tau(tau, min_len=3, fixed_len=None, min_idx=min_idx, max_idx=max_idx)
    return tau


def sanitize_tau(
    tau: List[int],
    min_len: int,
    fixed_len: Optional[int],
    min_idx: int,
    max_idx: int,
) -> List[int]:
    """裁剪、去重、排序(降序)、补齐/截断到固定长度。"""
    tau = [int(max(min_idx, min(max_idx, int(t)))) for t in tau]
    tau = sorted(list(set(tau)), reverse=True)

    while len(tau) < min_len:
        tau.append(random.randint(min_idx, max_idx))
        tau = sorted(list(set(tau)), reverse=True)

    if fixed_len is not None:
        if len(tau) > fixed_len:
            tau = tau[:fixed_len]
        while len(tau) < fixed_len:
            tau.append(random.randint(min_idx, max_idx))
            tau = sorted(list(set(tau)), reverse=True)
            if len(tau) > fixed_len:
                tau = tau[:fixed_len]
        # 极端：去重后仍短，继续补
        while len(tau) < fixed_len:
            tau.append(random.randint(min_idx, max_idx))
            tau = sorted(list(set(tau)), reverse=True)
            if len(tau) > fixed_len:
                tau = tau[:fixed_len]

    return tau


def random_tau(min_steps: int, max_steps: int, min_idx: int, max_idx: int, fixed_len: Optional[int] = None) -> List[int]:
    """生成随机 tau 序列（降序）。"""
    min_steps = max(3, int(min_steps))  # len(tau) >= 3 => steps=len-1 >= 2
    max_steps = max(min_steps, int(max_steps))

    if fixed_len is not None:
        num_steps = int(fixed_len)
    else:
        num_steps = random.randint(min_steps, max_steps)

    num_steps = max(3, num_steps)
    # sample without replacement
    if (max_idx - min_idx + 1) < num_steps:
        # 不够就允许重复后再去重补齐
        tau = [random.randint(min_idx, max_idx) for _ in range(num_steps)]
    else:
        tau = random.sample(range(min_idx, max_idx + 1), num_steps)

    tau = sanitize_tau(tau, min_len=3, fixed_len=fixed_len, min_idx=min_idx, max_idx=max_idx)
    return tau


def mutate_tau_fixedlen(
    tau: List[int],
    prob: float,
    mutation_range: int,
    min_idx: int,
    max_idx: int,
    fixed_len: int,
    min_len: int = 3,
) -> List[int]:
    """固定长度 tau 的“位置微调”变异。"""
    new_tau = list(map(int, tau))

    # 逐位置扰动
    for i in range(len(new_tau)):
        if random.random() < prob:
            delta = random.randint(-int(mutation_range), int(mutation_range))
            new_tau[i] = max(min_idx, min(max_idx, new_tau[i] + delta))

    # 少量随机重采样（跳出局部）
    if random.random() < 0.2:
        j = random.randint(0, len(new_tau) - 1)
        new_tau[j] = random.randint(min_idx, max_idx)

    new_tau = sanitize_tau(new_tau, min_len=min_len, fixed_len=fixed_len, min_idx=min_idx, max_idx=max_idx)
    return new_tau


def crossover_tau_fixedlen(
    tau1: List[int],
    tau2: List[int],
    min_idx: int,
    max_idx: int,
    fixed_len: int,
    min_len: int = 3,
) -> List[int]:
    """固定长度 tau 的交叉：集合混合 + 截断/补齐。"""
    pool = list(set(list(map(int, tau1)) + list(map(int, tau2))))
    random.shuffle(pool)
    child = pool[:fixed_len]
    child = sanitize_tau(child, min_len=min_len, fixed_len=fixed_len, min_idx=min_idx, max_idx=max_idx)
    return child


# =========================
# channel scales helpers
# =========================
def random_channel_scales(num_layers: int, choices: List[float]) -> List[float]:
    return [float(random.choice(choices)) for _ in range(int(num_layers))]


def mutate_channel_scales(scales: List[float], prob: float, choices: List[float]) -> List[float]:
    new_scales = list(map(float, scales))
    for i in range(len(new_scales)):
        if random.random() < prob:
            new_scales[i] = float(random.choice(choices))
    return new_scales


def crossover_channel_scales(scales1: List[float], scales2: List[float]) -> List[float]:
    point = random.randint(1, len(scales1) - 1)
    return list(map(float, scales1[:point] + scales2[point:]))


# =========================
# candidate mutate/crossover
# =========================
def mutate_candidate(
    cand: ArchCandidate,
    tau_prob: float,
    channel_prob: float,
    tau_range: int,
    tau_min_idx: int,
    tau_max_idx: int,
    channel_choices: List[float],
    fixed_tau_len: Optional[int] = None,
) -> ArchCandidate:
    if fixed_tau_len is not None:
        new_tau = mutate_tau_fixedlen(
            cand.tau,
            prob=tau_prob,
            mutation_range=tau_range,
            min_idx=tau_min_idx,
            max_idx=tau_max_idx,
            fixed_len=int(fixed_tau_len),
            min_len=3,
        )
    else:
        # 兼容旧逻辑：允许增删步数（不推荐用于你的当前目标）
        new_tau = cand.tau.copy()
        for i in range(len(new_tau)):
            if random.random() < tau_prob:
                delta = random.randint(-tau_range, tau_range)
                new_tau[i] = max(tau_min_idx, min(tau_max_idx, new_tau[i] + delta))
        new_tau = sanitize_tau(new_tau, min_len=3, fixed_len=None, min_idx=tau_min_idx, max_idx=tau_max_idx)

    new_scales = mutate_channel_scales(cand.channel_scales, channel_prob, channel_choices)
    return ArchCandidate(new_tau, new_scales)


def crossover_candidates(
    cand1: ArchCandidate,
    cand2: ArchCandidate,
    tau_min_idx: int = 1,
    tau_max_idx: int = 999,
    fixed_tau_len: Optional[int] = None,
) -> ArchCandidate:
    if fixed_tau_len is not None:
        new_tau = crossover_tau_fixedlen(
            cand1.tau, cand2.tau,
            min_idx=tau_min_idx, max_idx=tau_max_idx,
            fixed_len=int(fixed_tau_len),
            min_len=3,
        )
    else:
        # 旧逻辑：随机混合 tau（长度可能变化）
        combined = cand1.tau + cand2.tau
        k = max(3, min(len(combined) // 2, len(combined)))
        new_tau = sorted(random.sample(combined, k), reverse=True)
        new_tau = sanitize_tau(new_tau, min_len=3, fixed_len=None, min_idx=tau_min_idx, max_idx=tau_max_idx)

    new_scales = crossover_channel_scales(cand1.channel_scales, cand2.channel_scales)
    return ArchCandidate(new_tau, new_scales)


# =========================
# Scheme A: quantized build_compressed_unet
# =========================
def _safe_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        try:
            return getattr(cfg, key)
        except Exception:
            pass
    return default


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(x)
    except Exception:
        return [x]


def _to_int_list(x):
    xs = _to_list(x)
    out = []
    for v in xs:
        try:
            out.append(int(v))
        except Exception:
            out.append(int(float(v)))
    return out


def _to_float_list(x):
    xs = _to_list(x)
    out = []
    for v in xs:
        try:
            out.append(float(v))
        except Exception:
            out.append(float(str(v)))
    return out


def make_divisible(v: int, divisor: int, min_value: int) -> int:
    v = int(v)
    if v < min_value:
        v = min_value
    new_v = (v + divisor // 2) // divisor * divisor
    if new_v < min_value:
        new_v = min_value
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def _compute_stage_scales(channel_scales: List[float], num_stages: int) -> List[float]:
    scales = _to_float_list(channel_scales)
    if len(scales) == 0:
        return [1.0] * num_stages
    chunk = int(np.ceil(len(scales) / num_stages))
    stage_scales = []
    for i in range(num_stages):
        start = i * chunk
        end = min((i + 1) * chunk, len(scales))
        seg = scales[start:end]
        if len(seg) == 0:
            seg = [scales[-1]]
        stage_scales.append(float(sum(seg) / len(seg)))
    return stage_scales[:num_stages]


def build_compressed_unet(
    teacher_unet,
    unet_config,
    channel_scales: List[float],
    min_channels: int = 16,
    strategy: str = "stage_wise",
):
    """
    Scheme A (Quantization):
      - enforce channels divisible by norm_num_groups / attention constraints
      - enforce integer channel_mult (LD M UNetModel expects ints)
    """
    config_copy = copy.deepcopy(unet_config)

    # -------- infer constraints --------
    norm_groups = _safe_get(config_copy.params, "norm_num_groups", 32)
    norm_groups = int(norm_groups) if norm_groups is not None else 32

    num_head_channels = _safe_get(config_copy.params, "num_head_channels", None)
    num_heads = _safe_get(config_copy.params, "num_heads", None)

    min_channels = int(max(int(min_channels), norm_groups))

    divisor = 8
    divisor = int(np.lcm(divisor, norm_groups))
    if num_head_channels is not None:
        divisor = int(np.lcm(divisor, int(num_head_channels)))
    if num_heads is not None:
        divisor = int(np.lcm(divisor, int(num_heads)))
    if divisor <= 0:
        divisor = 32

    if strategy == "global":
        scales = _to_float_list(channel_scales)
        avg_scale = float(sum(scales) / len(scales)) if len(scales) > 0 else 1.0

        base_channels = int(config_copy.params.model_channels)
        raw_new_base = int(round(base_channels * avg_scale))
        new_base = make_divisible(raw_new_base, divisor=divisor, min_value=min_channels)

        config_copy.params.model_channels = int(new_base)

        channel_mult = _to_int_list(_safe_get(config_copy.params, "channel_mult", [1, 2, 4, 4]))
        config_copy.params.channel_mult = channel_mult

        print(f"[Arch] Global (quantize x{divisor}): scale={avg_scale:.2f}, base {base_channels}->{new_base}")

    elif strategy == "stage_wise":
        base_channels = int(config_copy.params.model_channels)
        channel_mult = _to_int_list(_safe_get(config_copy.params, "channel_mult", [1, 2, 4, 4]))
        num_stages = len(channel_mult)

        stage_scales = _compute_stage_scales(channel_scales, num_stages)
        teacher_stage_channels = [base_channels * int(m) for m in channel_mult]

        target_stage_channels = []
        for ch_t, s in zip(teacher_stage_channels, stage_scales):
            raw = int(round(float(ch_t) * float(s)))
            raw = max(raw, min_channels)
            q = make_divisible(raw, divisor=divisor, min_value=min_channels)
            target_stage_channels.append(int(q))

        new_base = make_divisible(target_stage_channels[0], divisor=divisor, min_value=min_channels)

        new_mult = []
        final_stage_channels = []
        for i in range(num_stages):
            m = max(1, int(round(target_stage_channels[i] / float(new_base))))
            ch = new_base * m
            ch = make_divisible(ch, divisor=divisor, min_value=min_channels)
            m = max(1, int(ch // new_base))
            ch = new_base * m
            new_mult.append(int(m))
            final_stage_channels.append(int(ch))

        config_copy.params.model_channels = int(new_base)
        config_copy.params.channel_mult = new_mult

        print(f"[Arch] Stage-wise (quantize x{divisor}):")
        print(f"  base_channels: {base_channels}->{new_base}")
        print(f"  channel_mult: {channel_mult}->{new_mult}")
        print(f"  stage_scales: {[f'{s:.2f}' for s in stage_scales]}")
        print(f"  stage_channels teacher: {teacher_stage_channels}")
        print(f"  stage_channels targetQ: {target_stage_channels}")
        print(f"  stage_channels final  : {final_stage_channels}")

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    student_unet = instantiate_from_config(config_copy)
    if isinstance(student_unet, nn.DataParallel):
        student_unet = student_unet.module

    try:
        missing, unexpected = student_unet.load_state_dict(teacher_unet.state_dict(), strict=False)
        if len(missing) > 0:
            print(f"[Arch] Partial init: {len(missing)} missing keys (expected for changed arch)")
    except Exception as e:
        print(f"[Arch] Teacher init failed: {repr(e)}")

    return student_unet


# =========================
# metrics / selection
# =========================
def count_parameters(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def calculate_fitness(
    fid: float,
    params: float,
    weight_fid: float = 1.0,
    weight_params: float = 0.01,
) -> float:
    return weight_fid * fid + weight_params * params


def pareto_sort(candidates: List[ArchCandidate]) -> List[ArchCandidate]:
    """
    简化版 Pareto：按非支配层依次输出（先前层更优）
    目标：最小化 FID 和 params
    """
    if not candidates:
        return []

    objectives = [(c.fid, c.params) for c in candidates]
    pareto_front = []
    remaining = list(range(len(candidates)))

    while remaining:
        non_dominated = []
        for i in remaining:
            is_dominated = False
            for j in remaining:
                if i == j:
                    continue
                if (objectives[j][0] <= objectives[i][0] and objectives[j][1] <= objectives[i][1] and
                    (objectives[j][0] < objectives[i][0] or objectives[j][1] < objectives[i][1])):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(i)

        pareto_front.extend([candidates[i] for i in non_dominated])
        remaining = [i for i in remaining if i not in non_dominated]

    return pareto_front


def candidate_to_hash(cand: ArchCandidate) -> str:
    tau_str = "_".join(map(str, cand.tau))
    scales_str = "_".join(f"{s:.2f}" for s in cand.channel_scales)
    return f"{tau_str}|{scales_str}"
