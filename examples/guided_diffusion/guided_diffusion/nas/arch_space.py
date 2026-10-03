#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage NAS search space for guided-diffusion ADM.

Stage0:
- Search a static ADM UNet skeleton via attention placement + global resblock depth.

Stage1:
- Timesteps are fixed externally.
- Search a per-step skip policy on top of the selected Stage0 architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class UNetArchConfig:
    attention_resolutions: Tuple[int, ...]
    num_res_blocks: int

    def validate(self) -> None:
        if int(self.num_res_blocks) <= 0:
            raise ValueError(f"num_res_blocks must be positive, got {self.num_res_blocks}")
        if any(int(x) <= 0 for x in self.attention_resolutions):
            raise ValueError(
                f"attention_resolutions must be positive, got {self.attention_resolutions}"
            )

    def key(self) -> str:
        return str(
            (
                tuple(int(x) for x in self.attention_resolutions),
                int(self.num_res_blocks),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attention_resolutions"] = [int(x) for x in self.attention_resolutions]
        d["num_res_blocks"] = int(self.num_res_blocks)
        return d


@dataclass(frozen=True)
class StepPolicy:
    skip_blocks: Tuple[int, ...] = field(default_factory=tuple)
    layer_gates: Tuple[Tuple[int, float], ...] = field(default_factory=tuple)
    class_gain: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skip_blocks": [int(x) for x in self.skip_blocks],
            "layer_gates": [
                {"layer_id": int(layer_id), "gate": float(gate)}
                for layer_id, gate in self.layer_gates
            ],
            "class_gain": float(self.class_gain),
        }


@dataclass(frozen=True)
class SamplerPolicy:
    # DDIM in this pipeline is deterministic. Keep a placeholder field so the
    # Stage1 data model matches the SD pipeline structurally.
    ddim_eta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"ddim_eta": float(self.ddim_eta)}


@dataclass(frozen=True)
class Genotype:
    step_policies: Tuple[StepPolicy, ...] = field(default_factory=tuple)
    sampler_policy: SamplerPolicy = field(default_factory=SamplerPolicy)
    fitness: float = -1.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def get_policy_for_step(
        self,
        step_idx: int,
        total_blocks: Optional[int] = None,
        searchable_block_ids: Optional[Sequence[int]] = None,
    ) -> StepPolicy:
        if not self.step_policies:
            return StepPolicy(skip_blocks=tuple())
        idx = int(np.clip(int(step_idx), 0, len(self.step_policies) - 1))
        pol = self.step_policies[idx]
        return repair_step_policy(
            pol,
            total_blocks=total_blocks,
            searchable_block_ids=searchable_block_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_policies": [p.to_dict() for p in self.step_policies],
            "sampler_policy": self.sampler_policy.to_dict(),
            "fitness": float(self.fitness),
            "metrics": dict(self.metrics or {}),
        }


def _normalize_searchable_block_ids(
    total_blocks: Optional[int] = None,
    searchable_block_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    if searchable_block_ids is not None:
        return tuple(sorted({int(x) for x in searchable_block_ids if int(x) >= 0}))
    total_blocks = max(0, int(total_blocks or 0))
    return tuple(range(total_blocks))


def _normalize_layer_gates(
    gate_values: Optional[Any],
    valid_ids: Sequence[int],
    *,
    default_gate: float = 1.0,
    gate_min: float = 0.0,
    gate_max: float = 1.5,
) -> Tuple[Tuple[int, float], ...]:
    valid_id_set = {int(x) for x in valid_ids}
    items: List[Tuple[int, float]] = []
    if gate_values is None:
        return tuple()

    if isinstance(gate_values, dict):
        iterator = gate_values.items()
    else:
        iterator = gate_values

    for item in iterator:
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
        layer_id = int(layer_id)
        if layer_id not in valid_id_set:
            continue
        gate = float(np.clip(float(gate), gate_min, gate_max))
        if abs(gate - float(default_gate)) <= 1e-6:
            continue
        items.append((layer_id, gate))

    items.sort(key=lambda x: x[0])
    return tuple(items)


def repair_step_policy(
    policy: Optional[StepPolicy],
    total_blocks: Optional[int] = None,
    searchable_block_ids: Optional[Sequence[int]] = None,
) -> StepPolicy:
    if policy is None:
        return StepPolicy(skip_blocks=tuple())

    valid_ids = set(
        _normalize_searchable_block_ids(
            total_blocks=total_blocks, searchable_block_ids=searchable_block_ids
        )
    )
    skips = sorted(
        {
            int(x)
            for x in getattr(policy, "skip_blocks", tuple()) or tuple()
            if int(x) in valid_ids
        }
    )
    gate_values = getattr(policy, "layer_gates", None)
    if gate_values is None and isinstance(policy, dict):
        gate_values = policy.get("layer_gates", None)
    layer_gates = _normalize_layer_gates(gate_values, sorted(valid_ids))

    class_gain = getattr(policy, "class_gain", None)
    if class_gain is None and isinstance(policy, dict):
        class_gain = policy.get("class_gain", None)
    if class_gain is None:
        class_gain = 1.0
    class_gain = float(np.clip(float(class_gain), 0.5, 1.5))

    return StepPolicy(
        skip_blocks=tuple(skips),
        layer_gates=layer_gates,
        class_gain=class_gain,
    )


def make_dense_policy() -> StepPolicy:
    return StepPolicy(skip_blocks=tuple(), layer_gates=tuple(), class_gain=1.0)


def make_random_step_policy(
    total_blocks: Optional[int],
    rng: np.random.RandomState,
    p_skip_range: Tuple[float, float],
    max_skip_ratio: float,
    searchable_block_ids: Optional[Sequence[int]] = None,
    gate_scale_range: Tuple[float, float] = (1.0, 1.0),
    class_gain_range: Tuple[float, float] = (1.0, 1.0),
) -> StepPolicy:
    candidates = list(
        _normalize_searchable_block_ids(
            total_blocks=total_blocks, searchable_block_ids=searchable_block_ids
        )
    )
    if not candidates:
        return StepPolicy(skip_blocks=tuple())

    lo, hi = float(p_skip_range[0]), float(p_skip_range[1])
    p_skip = float(np.clip(rng.uniform(lo, hi), 0.0, 1.0))
    max_skip = int(round(len(candidates) * float(max_skip_ratio)))
    skips: List[int] = []
    for idx in candidates:
        if rng.rand() < p_skip:
            skips.append(int(idx))
    if len(skips) > max_skip:
        keep = rng.choice(np.array(skips), size=max_skip, replace=False)
        skips = sorted(int(x) for x in keep.tolist())

    gate_lo, gate_hi = float(gate_scale_range[0]), float(gate_scale_range[1])
    raw_gates = []
    for idx in candidates:
        gate = float(np.clip(rng.uniform(gate_lo, gate_hi), 0.0, 1.5))
        raw_gates.append((int(idx), gate))
    layer_gates = _normalize_layer_gates(raw_gates, candidates)

    class_lo, class_hi = float(class_gain_range[0]), float(class_gain_range[1])
    class_gain = float(np.clip(rng.uniform(class_lo, class_hi), 0.5, 1.5))
    if abs(class_gain - 1.0) <= 1e-6:
        class_gain = 1.0
    return StepPolicy(skip_blocks=tuple(skips), layer_gates=layer_gates, class_gain=class_gain)


def mutate_step_policy(
    policy: StepPolicy,
    total_blocks: Optional[int],
    rng: np.random.RandomState,
    flip_prob: float,
    max_skip_ratio: float,
    searchable_block_ids: Optional[Sequence[int]] = None,
    gate_mutate_prob: Optional[float] = None,
    gate_sigma: float = 0.03,
    class_gain_sigma: float = 0.02,
) -> StepPolicy:
    candidates = list(
        _normalize_searchable_block_ids(
            total_blocks=total_blocks, searchable_block_ids=searchable_block_ids
        )
    )
    if not candidates:
        return StepPolicy(skip_blocks=tuple())

    skips = set(
        repair_step_policy(
            policy,
            total_blocks=total_blocks,
            searchable_block_ids=searchable_block_ids,
        ).skip_blocks
    )
    for idx in candidates:
        if rng.rand() < float(flip_prob):
            if idx in skips:
                skips.remove(idx)
            else:
                skips.add(idx)

    max_skip = int(round(len(candidates) * float(max_skip_ratio)))
    if len(skips) > max_skip:
        keep = rng.choice(np.array(sorted(skips)), size=max_skip, replace=False)
        skips = set(int(x) for x in keep.tolist())

    if gate_mutate_prob is None:
        gate_mutate_prob = float(flip_prob)
    base_gate_map = {int(idx): 1.0 for idx in candidates}
    for layer_id, gate in repair_step_policy(
        policy,
        total_blocks=total_blocks,
        searchable_block_ids=searchable_block_ids,
    ).layer_gates:
        base_gate_map[int(layer_id)] = float(gate)
    for idx in candidates:
        if rng.rand() < float(gate_mutate_prob):
            base_gate_map[int(idx)] = float(
                np.clip(base_gate_map[int(idx)] + rng.normal(0.0, float(gate_sigma)), 0.0, 1.5)
            )
    layer_gates = _normalize_layer_gates(base_gate_map, candidates)

    class_gain = float(getattr(policy, "class_gain", 1.0))
    class_gain = float(np.clip(class_gain + rng.normal(0.0, float(class_gain_sigma)), 0.5, 1.5))
    if abs(class_gain - 1.0) <= 1e-6:
        class_gain = 1.0
    return StepPolicy(
        skip_blocks=tuple(sorted(skips)),
        layer_gates=layer_gates,
        class_gain=class_gain,
    )


def crossover_step_policy(
    a: StepPolicy,
    b: StepPolicy,
    total_blocks: Optional[int],
    rng: np.random.RandomState,
    max_skip_ratio: float,
    searchable_block_ids: Optional[Sequence[int]] = None,
) -> StepPolicy:
    candidates = list(
        _normalize_searchable_block_ids(
            total_blocks=total_blocks, searchable_block_ids=searchable_block_ids
        )
    )
    if not candidates:
        return StepPolicy(skip_blocks=tuple())

    a_skips = set(
        repair_step_policy(
            a,
            total_blocks=total_blocks,
            searchable_block_ids=searchable_block_ids,
        ).skip_blocks
    )
    b_skips = set(
        repair_step_policy(
            b,
            total_blocks=total_blocks,
            searchable_block_ids=searchable_block_ids,
        ).skip_blocks
    )
    out = set()
    for idx in candidates:
        if (idx in a_skips) and (idx in b_skips):
            out.add(idx)
        elif (idx in a_skips) or (idx in b_skips):
            if rng.rand() < 0.5:
                out.add(idx)

    max_skip = int(round(len(candidates) * float(max_skip_ratio)))
    if len(out) > max_skip:
        keep = rng.choice(np.array(sorted(out)), size=max_skip, replace=False)
        out = set(int(x) for x in keep.tolist())

    a_rep = repair_step_policy(a, total_blocks=total_blocks, searchable_block_ids=searchable_block_ids)
    b_rep = repair_step_policy(b, total_blocks=total_blocks, searchable_block_ids=searchable_block_ids)
    a_gate_map = {int(idx): 1.0 for idx in candidates}
    b_gate_map = {int(idx): 1.0 for idx in candidates}
    for layer_id, gate in a_rep.layer_gates:
        a_gate_map[int(layer_id)] = float(gate)
    for layer_id, gate in b_rep.layer_gates:
        b_gate_map[int(layer_id)] = float(gate)
    out_gate_map = {}
    for idx in candidates:
        out_gate_map[int(idx)] = float(a_gate_map[int(idx)] if rng.rand() < 0.5 else b_gate_map[int(idx)])
    layer_gates = _normalize_layer_gates(out_gate_map, candidates)
    class_gain = float(a_rep.class_gain if rng.rand() < 0.5 else b_rep.class_gain)
    if abs(class_gain - 1.0) <= 1e-6:
        class_gain = 1.0
    return StepPolicy(
        skip_blocks=tuple(sorted(out)),
        layer_gates=layer_gates,
        class_gain=class_gain,
    )


def enumerate_candidate_archs(
    teacher_num_res_blocks: int,
    rng: Optional[np.random.RandomState] = None,
    max_candidates: int = 12,
    *,
    allow_depth_search: bool = False,
    depth_candidates: Optional[Sequence[int]] = None,
    attention_patterns: Optional[Sequence[Sequence[int]]] = None,
) -> List[UNetArchConfig]:
    """Enumerate Stage0 ADM skeleton candidates.

    LSUN-Bedroom 256 ADM is very sensitive to changing the global UNet depth.
    The safe default is therefore teacher-isomorphic: keep num_res_blocks equal to
    teacher_num_res_blocks and keep attention_resolutions=(32,16,8).

    Set allow_depth_search=true explicitly if you really want to ablate shallower
    or deeper students. Do not enable it for the main LSUN-Bedroom result.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    if attention_patterns is None:
        attn_patterns = [(32, 16, 8)]
    else:
        attn_patterns = [tuple(int(x) for x in attn) for attn in attention_patterns]
        if not attn_patterns:
            attn_patterns = [(32, 16, 8)]

    teacher_depth = int(teacher_num_res_blocks)
    if depth_candidates is not None:
        depths = sorted({int(x) for x in depth_candidates})
    elif allow_depth_search:
        depths = sorted({1, 2, 3, teacher_depth})
    else:
        depths = [teacher_depth]

    # Always put the exact teacher architecture first.
    baseline = UNetArchConfig(attention_resolutions=(32, 16, 8), num_res_blocks=teacher_depth)
    cands = [
        UNetArchConfig(attention_resolutions=tuple(int(x) for x in attn), num_res_blocks=int(depth))
        for attn in attn_patterns
        for depth in depths
    ]
    if baseline in cands:
        cands.remove(baseline)
    cands.insert(0, baseline)

    if len(cands) > 1:
        tail = cands[1:]
        rng.shuffle(tail)
        cands = [cands[0]] + tail

    cands = cands[: int(max_candidates)]
    for cand in cands:
        cand.validate()
    return cands
