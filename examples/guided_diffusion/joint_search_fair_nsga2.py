#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot joint NAS baseline for fair comparison with the user's two-stage NAS.

What is jointly searched here:
  - timestep schedule
  - static UNet architecture (attention placement + global resblock depth)
  - step-wise dynamic policy (per-step skip / gate policy)

This is the correct one-shot baseline when the two-stage method is:
  Stage0: search timestep schedule + static architecture
  Stage1: search step-wise dynamic policy on top of Stage0 result

Important reporting semantics:
  - Final FID / latency come from the FINAL selected candidate only.
  - Search-cost metrics (time / evaluated candidates) are tracked for the whole run.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from guided_diffusion.nas.arch_space import (
    Genotype,
    SamplerPolicy,
    StepPolicy,
    UNetArchConfig,
    crossover_step_policy,
    enumerate_candidate_archs,
    make_dense_policy,
    make_random_step_policy,
    mutate_step_policy,
)
from guided_diffusion.nas.evaluator_with_timesteps import Evaluator

print = functools.partial(print, flush=True)
choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))


# -------------------------
#  NSGA-II helpers
# -------------------------
def dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
    not_worse = True
    strictly_better = False
    for k in keys:
        av = float(a.get(k, float("inf")))
        bv = float(b.get(k, float("inf")))
        if av > bv:
            not_worse = False
            break
        if av < bv:
            strictly_better = True
    return not_worse and strictly_better


def fast_non_dominated_sort(pop_infos: List[Dict[str, float]], obj_keys: List[str]):
    dominates_list = [[] for _ in range(len(pop_infos))]
    dominated_count = [0 for _ in range(len(pop_infos))]
    fronts = [[]]

    for p in range(len(pop_infos)):
        for q in range(len(pop_infos)):
            if p == q:
                continue
            if dominates(pop_infos[p], pop_infos[q], obj_keys):
                dominates_list[p].append(q)
            elif dominates(pop_infos[q], pop_infos[p], obj_keys):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def crowding_distance(front_indices: List[int], pop_infos: List[Dict[str, float]], obj_keys: List[str]):
    dist = {idx: 0.0 for idx in front_indices}
    if len(front_indices) <= 2:
        for idx in front_indices:
            dist[idx] = float("inf")
        return dist

    for key in obj_keys:
        ordered = sorted(front_indices, key=lambda idx: float(pop_infos[idx].get(key, float("inf"))))
        dist[ordered[0]] = float("inf")
        dist[ordered[-1]] = float("inf")
        vmin = float(pop_infos[ordered[0]].get(key, float("inf")))
        vmax = float(pop_infos[ordered[-1]].get(key, float("inf")))
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        for pos in range(1, len(ordered) - 1):
            prev_v = float(pop_infos[ordered[pos - 1]].get(key, float("inf")))
            next_v = float(pop_infos[ordered[pos + 1]].get(key, float("inf")))
            dist[ordered[pos]] += (next_v - prev_v) / denom
    return dist


def nsga2_select(candidates: List[str], vis_dict: Dict[str, Dict[str, Any]], k: int, obj_keys: List[str]) -> List[str]:
    pop_infos = []
    valid_candidates = []
    for c in candidates:
        if c not in vis_dict or not vis_dict[c].get("visited", False):
            continue
        valid_candidates.append(c)
        pop_infos.append({key: float(vis_dict[c].get(key, float("inf"))) for key in obj_keys})

    if not valid_candidates:
        return []

    fronts = fast_non_dominated_sort(pop_infos, obj_keys)
    selected: List[str] = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend([valid_candidates[idx] for idx in front])
            continue
        cd = crowding_distance(front, pop_infos, obj_keys)
        ordered = sorted(front, key=lambda idx: cd[idx], reverse=True)
        remain = k - len(selected)
        selected.extend([valid_candidates[idx] for idx in ordered[:remain]])
        break
    return selected


# -------------------------
#  schedule helpers
# -------------------------
def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def _resolve_diffusion_num_steps(config_num_steps: int, timesteps: Sequence[int]) -> int:
    resolved = int(config_num_steps)
    if timesteps:
        min_required = max(int(t) for t in timesteps) + 1
        if resolved < min_required:
            print(
                "[Config][Warn] num_steps is smaller than the max searched timestep. "
                f"Auto-correcting from {resolved} to {min_required}."
            )
            resolved = min_required
    return resolved


def _normalize_timesteps(cand: Sequence[int], time_step: int) -> List[int]:
    uniq = sorted({int(x) for x in cand}, reverse=True)
    return uniq[: int(time_step)]


# -------------------------
#  genotype (de)serialization
# -------------------------
def _serialize_step_policy(pol: StepPolicy) -> Dict[str, Any]:
    return {
        "skip_blocks": [int(x) for x in getattr(pol, "skip_blocks", tuple())],
        "layer_gates": [
            {"layer_id": int(layer_id), "gate": float(gate)}
            for layer_id, gate in (getattr(pol, "layer_gates", tuple()) or tuple())
        ],
        "class_gain": float(getattr(pol, "class_gain", 1.0)),
    }


def _deserialize_step_policy(payload: Dict[str, Any]) -> StepPolicy:
    return StepPolicy(
        skip_blocks=tuple(sorted({int(x) for x in payload.get("skip_blocks", [])})),
        layer_gates=tuple(
            sorted(
                (
                    (int(g["layer_id"]), float(g["gate"]))
                    for g in payload.get("layer_gates", [])
                ),
                key=lambda x: x[0],
            )
        ),
        class_gain=float(payload.get("class_gain", 1.0)),
    )


def _serialize_genotype(g: Genotype) -> Dict[str, Any]:
    return {
        "step_policies": [_serialize_step_policy(p) for p in g.step_policies],
        "sampler_policy": {"ddim_eta": float(getattr(g.sampler_policy, "ddim_eta", 0.0))},
    }


def _deserialize_genotype(payload: Dict[str, Any]) -> Genotype:
    return Genotype(
        step_policies=tuple(_deserialize_step_policy(p) for p in payload.get("step_policies", [])),
        sampler_policy=SamplerPolicy(ddim_eta=float(payload.get("sampler_policy", {}).get("ddim_eta", 0.0))),
        fitness=-1.0,
        metrics={},
    )


def _json_key(timesteps: Sequence[int], arch_idx: int, genotype: Genotype) -> str:
    payload = {
        "timesteps": [int(x) for x in timesteps],
        "arch_idx": int(arch_idx),
        "genotype": _serialize_genotype(genotype),
    }
    return json.dumps(payload, sort_keys=True)


def _parse_json_key(cand: str) -> Tuple[List[int], int, Genotype]:
    payload = json.loads(cand)
    return [int(x) for x in payload["timesteps"]], int(payload["arch_idx"]), _deserialize_genotype(payload["genotype"])


def _arch_to_str(arch: UNetArchConfig) -> str:
    return json.dumps(arch.to_dict(), sort_keys=True)


class JointFullEvolutionSearcher(object):
    def __init__(self, config: Dict[str, Any], schedule_search_space: Optional[List[int]] = None):
        self.cfg = dict(config)
        self.search_space = schedule_search_space

        self.max_epochs = int(config["generations"])
        self.select_num = int(config["select_num"])
        self.population_num = int(config["population"])
        self.m_prob = float(config.get("schedule_mutation_prob", config.get("mutation_prob", 0.1)))
        self.crossover_num = int(config["crossover_num"])
        self.mutation_num = int(config["mutation_num"])
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates: List[str] = []
        self.vis_dict: Dict[str, Dict[str, Any]] = {}

        # Fairness: keep joint objective definition aligned with Stage1.
        requested_obj_keys = self.cfg.get("obj_keys", None)
        self.obj_keys = ["fid", "latency_ms_per_img"]
        if requested_obj_keys is not None and list(requested_obj_keys) != self.obj_keys:
            print(
                f"[Config][Warn] Overriding obj_keys from {list(requested_obj_keys)} "
                f"to {self.obj_keys} for fair comparison with Stage1."
            )

        self.mo_eval_repeats = int(config.get("mo_eval_repeats", 2))
        self.use_median_fid = bool(config.get("use_median_fid", True))
        self.use_median_latency = bool(config.get("use_median_latency", True))
        self.outdir = Path(config["outdir"])
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.time_step = int(config.get("time_step", 4))
        self.seed = int(config.get("seed", 0))
        self.rng = np.random.RandomState(self.seed)

        # Schedule mutation bias
        self.high_noise_threshold = float(config.get("high_noise_threshold", 0.7))
        self.mid_noise_threshold = float(config.get("mid_noise_threshold", 0.25))
        self.high_stage_ratio = float(config.get("high_stage_ratio", 0.15))
        self.mid_stage_ratio = float(config.get("mid_stage_ratio", 0.7))
        self.low_stage_ratio = float(config.get("low_stage_ratio", 0.15))
        self.stage_refill_order = tuple(
            s.strip()
            for s in str(config.get("stage_refill_order", "mid,low,high")).split(",")
            if s.strip() in ("high", "mid", "low")
        ) or ("mid", "low", "high")
        self.arch_mutate_prob = float(config.get("arch_mutate_prob", 0.35))

        # Quality constraints / final selection
        self.quality_fid_baseline = self.cfg.get("quality_fid_baseline", None)
        if self.quality_fid_baseline is not None:
            self.quality_fid_baseline = float(self.quality_fid_baseline)
        self.quality_fid_delta = float(self.cfg.get("quality_fid_delta", float("inf")))
        self.quality_constraint_strict = bool(self.cfg.get("quality_constraint_strict", False))
        self.std_max = float(self.cfg.get("std_max", float("inf")))
        self.protected_last_n_steps = max(0, int(self.cfg.get("protected_last_n_steps", 0)))
        self.policy_search_mode = str(self.cfg.get("policy_search_mode", "hybrid")).lower()
        if self.policy_search_mode not in {"hybrid", "soft_only"}:
            raise ValueError(
                f"Unsupported policy_search_mode={self.policy_search_mode!r}. Expected 'hybrid' or 'soft_only'."
            )

        # Search-cost / convergence trackers
        self.search_start_time = time.perf_counter()
        self.total_eval_wall_time_s = 0.0
        self.eval_counter = 0
        self.search_peak_gpu_allocated_bytes = 0
        self.search_peak_gpu_reserved_bytes = 0
        self.convergence_history: List[Dict[str, Any]] = []
        self.convergence_jsonl_path = self.outdir / "convergence_history.jsonl"
        self.convergence_csv_path = self.outdir / "convergence_history.csv"

        # arch pool
        self.teacher_num_res_blocks = int(config.get("teacher_num_res_blocks", 3))
        arch_pool = enumerate_candidate_archs(
            teacher_num_res_blocks=self.teacher_num_res_blocks,
            rng=np.random.RandomState(self.seed),
            max_candidates=int(config.get("arch_max_candidates", 8)),
        )
        teacher_attn = tuple(int(x) for x in config.get("teacher_attention_resolutions", [32, 16, 8]))
        base_arch = UNetArchConfig(attention_resolutions=teacher_attn, num_res_blocks=self.teacher_num_res_blocks)
        if base_arch.key() not in {a.key() for a in arch_pool}:
            arch_pool = [base_arch] + arch_pool
        self.arch_pool = arch_pool
        self.base_arch_idx = next(i for i, a in enumerate(self.arch_pool) if a.key() == base_arch.key())
        (self.outdir / "joint_arch_pool.json").write_text(
            json.dumps([a.to_dict() for a in self.arch_pool], indent=2), encoding="utf-8"
        )

        # evaluator cache by arch key
        self.evaluator_cache: Dict[str, Evaluator] = {}

        # evaluator base kwargs
        self.base_eval_kwargs = dict(
            ckpt_path=str(config["ckpt_path"]),
            image_size=int(config.get("image_size", 64)),
            num_steps=int(config["num_steps"]),
            device=str(config.get("device", "cuda")),
            fp16=bool(config.get("fp16", False)),
            ref_mu=config.get("ref_mu"),
            ref_sigma=config.get("ref_sigma"),
            batch_size=int(config.get("batch_size", 16)),
            num_samples=int(config.get("num_samples", 2000)),
            seed=self.seed,
            max_resblocks_hint=int(config.get("max_resblocks", 256)),
            outdir=str(self.outdir),
            latents_shape=config.get("latents_shape"),
            calculate_is=bool(config.get("calculate_is", False)),
            calculate_precision_recall=bool(config.get("calculate_precision_recall", False)),
            calculate_kid=bool(config.get("calculate_kid", False)),
            is_splits=int(config.get("is_splits", 10)),
            pr_k=int(config.get("pr_k", 3)),
            kid_num_subsets=int(config.get("kid_num_subsets", 100)),
            kid_max_subset_size=int(config.get("kid_max_subset_size", 1000)),
            real_images_path=config.get("real_images_path"),
            mo_eval_repeats=self.mo_eval_repeats,
            use_median_fid=self.use_median_fid,
            use_median_latency=self.use_median_latency,
            class_sampling_mode=str(config.get("class_sampling_mode", "uniform")),
            searchable_block_mode=str(config.get("searchable_block_mode", "all")),
            protected_attention_resolutions=config.get("protected_attention_resolutions"),
            protect_middle_attention=bool(config.get("protect_middle_attention", True)),
            protected_block_types=config.get("protected_block_types"),
        )

        init_t = self._default_ddim_schedule()
        num_steps = _resolve_diffusion_num_steps(int(config["num_steps"]), init_t)
        self.base_eval_kwargs["num_steps"] = int(num_steps)

        resolved_cfg_path = self.outdir / "resolved_joint_full_search_config.json"
        with resolved_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, ensure_ascii=False)
        print(f"[Config] saved resolved config to {resolved_cfg_path}")

    # -------------------------
    # bookkeeping helpers
    # -------------------------
    def _update_search_peak_gpu(self, alloc_bytes: int, reserved_bytes: int) -> None:
        self.search_peak_gpu_allocated_bytes = max(self.search_peak_gpu_allocated_bytes, int(alloc_bytes))
        self.search_peak_gpu_reserved_bytes = max(self.search_peak_gpu_reserved_bytes, int(reserved_bytes))

    def _current_search_wall_time_s(self) -> float:
        return float(time.perf_counter() - self.search_start_time)

    def quality_fid_limit(self) -> float:
        limit = float("inf")
        if self.quality_fid_baseline is not None and np.isfinite(self.quality_fid_delta):
            limit = min(limit, float(self.quality_fid_baseline) + float(self.quality_fid_delta))
        return limit

    def _is_quality_feasible_metrics(self, metrics: Dict[str, Any]) -> bool:
        fid_v = float(metrics.get("fid_raw", metrics.get("fid", float("inf"))))
        std_v = float(metrics.get("std", metrics.get("fid_std", float("inf"))))
        if fid_v > self.quality_fid_limit():
            return False
        if std_v > self.std_max:
            return False
        return True

    def _global_pareto_keys(self) -> List[str]:
        evaluated = [k for k, info in self.vis_dict.items() if info.get("visited", False)]
        if not evaluated:
            return []
        pop_infos = [self.vis_dict[k] for k in evaluated]
        fronts, _ = fast_non_dominated_sort(pop_infos, self.obj_keys), None
        if not fronts:
            return []
        return [evaluated[i] for i in fronts[0]]

    def _best_key_by(self, keys: List[str], metric: str) -> Optional[str]:
        if not keys:
            return None
        return min(keys, key=lambda k: float(self.vis_dict[k].get(metric, float("inf"))))

    def _append_convergence_row(self, event: str) -> None:
        evaluated = [k for k, info in self.vis_dict.items() if info.get("visited", False)]
        feasible = [k for k in evaluated if self.vis_dict[k].get("quality_feasible", False)]
        pareto = self._global_pareto_keys()

        best_fid_key = self._best_key_by(evaluated, "fid")
        best_lat_key = self._best_key_by(evaluated, "latency_ms_per_img")
        best_feasible_fid_key = self._best_key_by(feasible, "fid")
        best_feasible_lat_key = self._best_key_by(feasible, "latency_ms_per_img")

        row = {
            "event": str(event),
            "epoch": int(self.epoch),
            "evaluated_candidates": int(len(evaluated)),
            "quality_feasible_candidates": int(len(feasible)),
            "pareto_size": int(len(pareto)),
            "best_fid_so_far": float(self.vis_dict[best_fid_key]["fid"]) if best_fid_key else float("inf"),
            "best_latency_so_far": float(self.vis_dict[best_lat_key]["latency_ms_per_img"]) if best_lat_key else float("inf"),
            "best_feasible_fid_so_far": float(self.vis_dict[best_feasible_fid_key]["fid"]) if best_feasible_fid_key else float("inf"),
            "best_feasible_latency_so_far": float(self.vis_dict[best_feasible_lat_key]["latency_ms_per_img"]) if best_feasible_lat_key else float("inf"),
            "total_eval_wall_time_s": float(self.total_eval_wall_time_s),
            "total_search_wall_time_s": float(self._current_search_wall_time_s()),
            "search_peak_gpu_allocated_gb": float(self.search_peak_gpu_allocated_bytes) / (1024 ** 3),
            "search_peak_gpu_reserved_gb": float(self.search_peak_gpu_reserved_bytes) / (1024 ** 3),
        }
        self.convergence_history.append(row)
        with self.convergence_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        header = list(row.keys())
        write_header = not self.convergence_csv_path.exists()
        with self.convergence_csv_path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(header) + "\n")
            f.write(",".join(str(row[k]) for k in header) + "\n")

    # -------------------------
    # schedule helpers
    # -------------------------
    def _split_timesteps_by_stage(self, timesteps):
        total_steps = max(1, int(self.base_eval_kwargs["num_steps"]) - 1)
        high, mid, low = [], [], []
        for t in sorted({int(x) for x in timesteps}, reverse=True):
            ratio = float(t) / float(total_steps)
            if ratio >= self.high_noise_threshold:
                high.append(int(t))
            elif ratio >= self.mid_noise_threshold:
                mid.append(int(t))
            else:
                low.append(int(t))
        return {"high": high, "mid": mid, "low": low}

    def _sample_biased_timesteps(self, all_timesteps):
        pools = self._split_timesteps_by_stage(all_timesteps)
        total_k = int(self.time_step)
        targets = {
            "high": int(round(total_k * self.high_stage_ratio)),
            "mid": int(round(total_k * self.mid_stage_ratio)),
            "low": int(round(total_k * self.low_stage_ratio)),
        }
        total_assigned = sum(targets.values())
        while total_assigned < total_k:
            targets["mid"] += 1
            total_assigned += 1
        while total_assigned > total_k:
            if targets["mid"] > 0:
                targets["mid"] -= 1
            elif targets["low"] > 0:
                targets["low"] -= 1
            elif targets["high"] > 0:
                targets["high"] -= 1
            total_assigned -= 1

        selected = []
        for stage in ("high", "mid", "low"):
            pool = pools[stage]
            if not pool:
                continue
            k = min(len(pool), max(0, targets[stage]))
            if k > 0:
                selected.extend(self.rng.choice(pool, size=k, replace=False).tolist())

        if len(selected) < total_k:
            remaining = [t for t in sorted({int(x) for x in all_timesteps}, reverse=True) if t not in selected]
            for stage in self.stage_refill_order:
                pool = [t for t in pools[stage] if t in remaining]
                if not pool:
                    continue
                need = total_k - len(selected)
                take = min(len(pool), need)
                selected.extend(self.rng.choice(pool, size=take, replace=False).tolist())
                remaining = [t for t in remaining if t not in selected]
                if len(selected) >= total_k:
                    break
            if len(selected) < total_k:
                need = total_k - len(selected)
                selected.extend(remaining[:need])

        selected = sorted({int(x) for x in selected}, reverse=True)
        if len(selected) < total_k:
            global_sorted = sorted({int(x) for x in all_timesteps}, reverse=True)
            for t in global_sorted:
                if t not in selected:
                    selected.append(int(t))
                if len(selected) >= total_k:
                    break
        return sorted(selected[:total_k], reverse=True)

    def _mutate_timesteps_stage_aware(self, cand, m_prob):
        all_index = self.search_space if self.search_space is not None else list(range(int(self.base_eval_kwargs["num_steps"])))
        pools = self._split_timesteps_by_stage(all_index)
        out = list(int(x) for x in cand)
        all_index_sorted = sorted({int(x) for x in all_index}, reverse=True)
        for i, t in enumerate(list(out)):
            if np.random.random_sample() >= m_prob:
                continue
            ratio = float(t) / float(max(1, int(self.base_eval_kwargs["num_steps"]) - 1))
            if ratio >= self.high_noise_threshold:
                stage = "high"
            elif ratio >= self.mid_noise_threshold:
                stage = "mid"
            else:
                stage = "low"
            cand_pool = [x for x in pools.get(stage, []) if x not in out or x == t]
            if not cand_pool:
                cand_pool = [x for x in all_index_sorted if x not in out or x == t]
            out[i] = int(choice(cand_pool))

        out = _normalize_timesteps(out, self.time_step)
        if len(out) < self.time_step:
            for t in all_index_sorted:
                if t not in out:
                    out.append(int(t))
                if len(out) >= self.time_step:
                    break
        return sorted(out[: self.time_step], reverse=True)

    def _default_ddim_schedule(self) -> List[int]:
        steps = int(self.base_eval_kwargs["num_steps"])
        timestep_respacing = f"ddim{self.time_step}"
        init_x = list(space_timesteps(steps, timestep_respacing))
        return sorted([int(x) for x in init_x], reverse=True)

    def sample_active_subnet(self) -> Tuple[List[int], int]:
        use_timestep = self.search_space if self.search_space is not None else list(range(int(self.base_eval_kwargs["num_steps"])))
        timesteps = self._sample_biased_timesteps(use_timestep)
        arch_idx = int(np.random.randint(len(self.arch_pool)))
        return timesteps, arch_idx

    # -------------------------
    # evaluator / policy helpers
    # -------------------------
    def _get_evaluator(self, arch_cfg: UNetArchConfig, timesteps: Sequence[int]) -> Evaluator:
        arch_key = arch_cfg.key()
        if arch_key not in self.evaluator_cache:
            sub_outdir = self.outdir / f"arch_{arch_key.replace(' ', '').replace('/', '_')}"
            sub_outdir.mkdir(parents=True, exist_ok=True)
            kwargs = dict(self.base_eval_kwargs)
            kwargs["outdir"] = str(sub_outdir)
            kwargs["arch_cfg"] = arch_cfg
            kwargs["fixed_timesteps"] = [int(x) for x in timesteps]
            self.evaluator_cache[arch_key] = Evaluator(**kwargs)
        evaluator = self.evaluator_cache[arch_key]
        evaluator.fixed_timesteps = tuple(int(x) for x in timesteps)
        evaluator.sampling_diffusion = evaluator._build_sampling_diffusion()
        return evaluator

    def _policy_space(self, arch_cfg: UNetArchConfig, timesteps: Sequence[int]) -> Tuple[int, Tuple[int, ...]]:
        evaluator = self._get_evaluator(arch_cfg, timesteps)
        total_blocks = int(getattr(evaluator, "total_blocks", 0))
        searchable_block_ids = tuple(int(x) for x in getattr(evaluator, "searchable_block_ids", tuple()))
        return total_blocks, searchable_block_ids

    def _apply_step_guards(self, step_policies: Sequence[StepPolicy]) -> Tuple[StepPolicy, ...]:
        policies = list(step_policies)
        if self.policy_search_mode == "soft_only":
            policies = [StepPolicy(skip_blocks=tuple(), layer_gates=tuple(getattr(pol, "layer_gates", tuple())), class_gain=float(getattr(pol, "class_gain", 1.0))) for pol in policies]
        if self.protected_last_n_steps > 0 and policies:
            keep_dense_from = max(0, len(policies) - self.protected_last_n_steps)
            for idx in range(keep_dense_from, len(policies)):
                policies[idx] = StepPolicy(skip_blocks=tuple(), layer_gates=tuple(), class_gain=1.0)
        return tuple(policies)

    def _dense_genotype(self, num_steps: int) -> Genotype:
        return Genotype(
            step_policies=tuple(make_dense_policy() for _ in range(int(num_steps))),
            sampler_policy=SamplerPolicy(ddim_eta=0.0),
            fitness=-1.0,
            metrics={},
        )

    def _random_genotype(self, num_steps: int, arch_cfg: UNetArchConfig, timesteps: Sequence[int]) -> Genotype:
        total_blocks, searchable_block_ids = self._policy_space(arch_cfg, timesteps)
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))
        init_cfg = dict(self.cfg.get("step_policy_init", {}))
        p_skip_high = tuple(init_cfg.get("p_skip_high", (0.0, 0.1)))
        p_skip_mid = tuple(init_cfg.get("p_skip_mid", (0.05, 0.2)))
        p_skip_low = tuple(init_cfg.get("p_skip_low", (0.1, 0.35)))
        gate_scale_range = tuple(init_cfg.get("gate_scale_range", (1.0, 1.0)))
        class_gain_range = tuple(init_cfg.get("class_gain_range", (1.0, 1.0)))
        max_skip_ratio = float(self.cfg.get("skip_max_ratio", 0.35))

        step_policies: List[StepPolicy] = []
        for i in range(int(num_steps)):
            pos = i / max(1.0, float(num_steps - 1))
            if pos < 0.33:
                p_skip = p_skip_high
            elif pos < 0.66:
                p_skip = p_skip_mid
            else:
                p_skip = p_skip_low
            step_policies.append(
                make_random_step_policy(
                    total_blocks=total_blocks,
                    searchable_block_ids=searchable_block_ids,
                    rng=rng,
                    p_skip_range=(float(p_skip[0]), float(p_skip[1])),
                    max_skip_ratio=max_skip_ratio,
                    gate_scale_range=(float(gate_scale_range[0]), float(gate_scale_range[1])),
                    class_gain_range=(float(class_gain_range[0]), float(class_gain_range[1])),
                )
            )

        return Genotype(
            step_policies=self._apply_step_guards(step_policies),
            sampler_policy=SamplerPolicy(ddim_eta=0.0),
            fitness=-1.0,
            metrics={},
        )

    def _mutate_genotype(self, g: Genotype, arch_cfg: UNetArchConfig, timesteps: Sequence[int]) -> Genotype:
        total_blocks, searchable_block_ids = self._policy_space(arch_cfg, timesteps)
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))
        mut_cfg = dict(self.cfg.get("step_policy_mutation", {}))
        per_step_p = float(mut_cfg.get("per_step_mutate_prob", 0.35))
        flip_prob = float(mut_cfg.get("mask_flip_prob", 0.08))
        gate_mutate_prob = float(mut_cfg.get("gate_mutate_prob", flip_prob))
        gate_sigma = float(mut_cfg.get("gate_sigma", 0.03))
        class_gain_sigma = float(mut_cfg.get("class_gain_sigma", 0.02))
        max_skip_ratio = float(self.cfg.get("skip_max_ratio", 0.35))

        step_policies: List[StepPolicy] = []
        for pol in g.step_policies:
            if rng.rand() < per_step_p:
                step_policies.append(
                    mutate_step_policy(
                        pol,
                        total_blocks=total_blocks,
                        searchable_block_ids=searchable_block_ids,
                        rng=rng,
                        flip_prob=flip_prob,
                        max_skip_ratio=max_skip_ratio,
                        gate_mutate_prob=gate_mutate_prob,
                        gate_sigma=gate_sigma,
                        class_gain_sigma=class_gain_sigma,
                    )
                )
            else:
                step_policies.append(pol)

        return Genotype(
            step_policies=self._apply_step_guards(step_policies),
            sampler_policy=g.sampler_policy,
            fitness=-1.0,
            metrics={},
        )

    def _crossover_genotype(self, a: Genotype, b: Genotype, arch_cfg: UNetArchConfig, timesteps: Sequence[int]) -> Genotype:
        total_blocks, searchable_block_ids = self._policy_space(arch_cfg, timesteps)
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))
        max_skip_ratio = float(self.cfg.get("skip_max_ratio", 0.35))
        n = min(len(a.step_policies), len(b.step_policies))
        step_policies: List[StepPolicy] = []
        for idx in range(n):
            step_policies.append(
                crossover_step_policy(
                    a.step_policies[idx],
                    b.step_policies[idx],
                    total_blocks=total_blocks,
                    searchable_block_ids=searchable_block_ids,
                    rng=rng,
                    max_skip_ratio=max_skip_ratio,
                )
            )
        if len(a.step_policies) > n:
            step_policies.extend(a.step_policies[n:])
        elif len(b.step_policies) > n:
            step_policies.extend(b.step_policies[n:])
        return Genotype(
            step_policies=self._apply_step_guards(step_policies[: len(timesteps)]),
            sampler_policy=a.sampler_policy,
            fitness=-1.0,
            metrics={},
        )

    # -------------------------
    # evaluation
    # -------------------------
    def evaluate_candidate(self, cand: str) -> bool:
        if cand in self.vis_dict and self.vis_dict[cand].get("visited", False):
            return True

        timesteps, arch_idx, genotype = _parse_json_key(cand)
        timesteps = sorted([int(x) for x in timesteps], reverse=True)
        arch_cfg = self.arch_pool[int(arch_idx)]
        evaluator = self._get_evaluator(arch_cfg, timesteps)

        cuda_enabled = torch.cuda.is_available() and str(self.cfg.get("device", "cuda")).lower() == "cuda"
        if cuda_enabled:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        eval_t0 = time.perf_counter()

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_cpu_state = torch.get_rng_state()
        torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            evaluated = evaluator.evaluate_genotype(genotype)
        except Exception as exc:
            eval_wall = float(time.perf_counter() - eval_t0)
            self.total_eval_wall_time_s += eval_wall
            peak_alloc = int(torch.cuda.max_memory_allocated()) if cuda_enabled else 0
            peak_reserved = int(torch.cuda.max_memory_reserved()) if cuda_enabled else 0
            self._update_search_peak_gpu(peak_alloc, peak_reserved)
            self.eval_counter += 1
            self.vis_dict[cand] = {
                "visited": True,
                "fid": float("inf"),
                "latency_ms_per_img": float("inf"),
                "std": float("inf"),
                "error": repr(exc),
                "arch_idx": int(arch_idx),
                "arch": arch_cfg.to_dict(),
                "timesteps": [int(x) for x in timesteps],
                "genotype": _serialize_genotype(genotype),
                "eval_wall_time_s": float(eval_wall),
                "peak_gpu_allocated_gb": float(peak_alloc) / (1024 ** 3),
                "peak_gpu_reserved_gb": float(peak_reserved) / (1024 ** 3),
            }
            self._append_convergence_row(event="eval_fail")
            print(f"[Eval][Fail] arch_idx={arch_idx} error={exc!r}")
            return False
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_state)
            if cuda_enabled:
                torch.cuda.synchronize()

        eval_wall = float(time.perf_counter() - eval_t0)
        self.total_eval_wall_time_s += eval_wall
        peak_alloc = int(torch.cuda.max_memory_allocated()) if cuda_enabled else 0
        peak_reserved = int(torch.cuda.max_memory_reserved()) if cuda_enabled else 0
        self._update_search_peak_gpu(peak_alloc, peak_reserved)
        self.eval_counter += 1

        metrics = dict(evaluated.metrics or {})
        info = {
            "visited": True,
            "fid": float(metrics.get("fid", float("inf"))),
            "latency_ms_per_img": float(metrics.get("latency_ms_per_img", float("inf"))),
            "std": float(metrics.get("std", metrics.get("fid_std", float("inf")))),
            "fid_std": float(metrics.get("fid_std", float("inf"))),
            "latency_std": float(metrics.get("latency_std", float("inf"))),
            "arch_idx": int(arch_idx),
            "arch": arch_cfg.to_dict(),
            "timesteps": [int(x) for x in timesteps],
            "genotype": _serialize_genotype(evaluated),
            "quality_feasible": bool(self._is_quality_feasible_metrics(metrics)),
            "eval_wall_time_s": float(eval_wall),
            "peak_gpu_allocated_gb": float(peak_alloc) / (1024 ** 3),
            "peak_gpu_reserved_gb": float(peak_reserved) / (1024 ** 3),
        }
        for k, v in metrics.items():
            if k not in info:
                info[k] = v
        self.vis_dict[cand] = info
        self._append_convergence_row(event="eval")
        print(
            f"[Eval] fid={info['fid']:.6f} lat={info['latency_ms_per_img']:.6f} std={info['std']:.6f} "
            f"peakGPU={info['peak_gpu_allocated_gb']:.3f}GB evalT={info['eval_wall_time_s']:.3f}s "
            f"timesteps={info['timesteps']} arch={_arch_to_str(arch_cfg)}"
        )
        return True

    def evaluate_population(self, population: Sequence[str]) -> None:
        for cand in population:
            self.evaluate_candidate(cand)

    def is_legal_before_search(self, cand: str) -> bool:
        if cand in self.vis_dict and self.vis_dict[cand].get("visited", False):
            print(f"cand already visited, skip: {cand[:120]}...")
            return False
        return self.evaluate_candidate(cand)

    def is_legal(self, cand: str) -> bool:
        return self.is_legal_before_search(cand)

    # -------------------------
    # evolution operators
    # -------------------------
    def update_top_k(self, candidates, *, k):
        assert k in self.keep_top_k
        t = self.keep_top_k[k]
        t += list(candidates)
        t = list(dict.fromkeys(t))
        t = [c for c in t if (c in self.vis_dict and self.vis_dict[c].get("visited", False))]
        self.keep_top_k[k] = nsga2_select(t, self.vis_dict, k, self.obj_keys)

    def dump_pareto_front(self) -> None:
        evaluated = [c for c, info in self.vis_dict.items() if info.get("visited", False)]
        if not evaluated:
            return
        pop_infos = [self.vis_dict[c] for c in evaluated]
        fronts = fast_non_dominated_sort(pop_infos, self.obj_keys)
        if not fronts:
            return
        pareto = [evaluated[i] for i in fronts[0]]
        pareto_sorted = sorted(pareto, key=lambda c: self.vis_dict[c].get("fid", float("inf")))
        latest_path = self.outdir / "pareto_front_latest.jsonl"
        all_path = self.outdir / "pareto_front_all.jsonl"
        with latest_path.open("w", encoding="utf-8") as f:
            for c in pareto_sorted:
                payload = dict(self.vis_dict[c])
                payload["candidate_key"] = c
                payload["epoch"] = int(self.epoch)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        with all_path.open("a", encoding="utf-8") as f:
            for c in pareto_sorted:
                payload = dict(self.vis_dict[c])
                payload["candidate_key"] = c
                payload["epoch"] = int(self.epoch)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[MO] Pareto latest -> {latest_path}; all -> {all_path} (pareto N={len(pareto_sorted)})")

    def save_population_snapshot(self, filename: str, population: Sequence[str]) -> None:
        rows = []
        for cand in population:
            info = dict(self.vis_dict.get(cand, {}))
            info["candidate_key"] = cand
            rows.append(info)
        (self.outdir / filename).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def _make_random_candidate(self, *, use_dense_policy: bool = False) -> str:
        timesteps, arch_idx = self.sample_active_subnet()
        arch_cfg = self.arch_pool[int(arch_idx)]
        genotype = self._dense_genotype(len(timesteps)) if use_dense_policy else self._random_genotype(len(timesteps), arch_cfg, timesteps)
        return _json_key(timesteps, arch_idx, genotype)

    def get_random_before_search(self, num):
        print("random select ........")
        while len(self.candidates) < num:
            cand = self._make_random_candidate(use_dense_policy=False)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            print(f"random {len(self.candidates)}/{num}")
        print(f"random_num = {len(self.candidates)}")

    def get_random(self, num):
        print("random select ........")
        while len(self.candidates) < num:
            cand = self._make_random_candidate(use_dense_policy=False)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print(f"random {len(self.candidates)}/{num}")
        print(f"random_num = {len(self.candidates)}")

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        print("cross ......")
        res = []
        max_iters = cross_num * 12

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])
            ts1, arch1, g1 = _parse_json_key(cand1)
            ts2, arch2, g2 = _parse_json_key(cand2)
            new_ts = []
            for i in range(len(ts1)):
                new_ts.append(ts1[i] if np.random.random_sample() < 0.5 else ts2[i])
            new_ts = _normalize_timesteps(new_ts, self.time_step)
            if len(new_ts) < self.time_step:
                all_index = self.search_space if self.search_space is not None else list(range(int(self.base_eval_kwargs["num_steps"])))
                for t in sorted(all_index, reverse=True):
                    if t not in new_ts:
                        new_ts.append(int(t))
                    if len(new_ts) >= self.time_step:
                        break
            new_ts = sorted(new_ts[: self.time_step], reverse=True)
            new_arch = arch1 if np.random.random_sample() < 0.5 else arch2
            new_arch_cfg = self.arch_pool[int(new_arch)]
            new_g = self._crossover_genotype(g1, g2, new_arch_cfg, new_ts)
            return _json_key(new_ts, new_arch, new_g)

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print(f"cross {len(res)}/{cross_num}")

        print(f"cross_num = {len(res)}")
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print("mutation ......")
        res = []
        max_iters = mutation_num * 12

        def random_func():
            cand = choice(self.keep_top_k[k])
            timesteps, arch_idx, genotype = _parse_json_key(cand)
            timesteps = self._mutate_timesteps_stage_aware(timesteps, m_prob)
            if np.random.random_sample() < self.arch_mutate_prob:
                arch_idx = int(np.random.randint(len(self.arch_pool)))
            arch_cfg = self.arch_pool[int(arch_idx)]
            genotype = self._mutate_genotype(genotype, arch_cfg, timesteps)
            return _json_key(timesteps, arch_idx, genotype)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print(f"mutation {len(res)}/{mutation_num}")

        print(f"mutation_num = {len(res)}")
        return res

    def mutate_init_x(self, x0, mutation_num, m_prob):
        print("mutation x0 ......")
        res = []
        max_iters = mutation_num * 12

        def random_func():
            timesteps, arch_idx, genotype = _parse_json_key(x0)
            timesteps = self._mutate_timesteps_stage_aware(timesteps, m_prob)
            arch_cfg = self.arch_pool[int(arch_idx)]
            genotype = self._mutate_genotype(genotype, arch_cfg, timesteps)
            return _json_key(timesteps, arch_idx, genotype)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print(f"mutation x0 {len(res)}/{mutation_num}")

        print(f"mutation_num = {len(res)}")
        return res

    # -------------------------
    # final selection
    # -------------------------
    def select_final_candidate(self) -> Dict[str, Any]:
        evaluated = [k for k, info in self.vis_dict.items() if info.get("visited", False)]
        feasible = [k for k in evaluated if self.vis_dict[k].get("quality_feasible", False)]
        report: Dict[str, Any] = {
            "evaluated_population_size": int(len(evaluated)),
            "quality_feasible_size": int(len(feasible)),
            "eval_counter": int(self.eval_counter),
            "quality_fid_baseline": self.quality_fid_baseline,
            "quality_fid_delta": float(self.quality_fid_delta),
            "quality_fid_limit": self.quality_fid_limit(),
            "std_max": float(self.std_max),
            "quality_constraint_strict": bool(self.quality_constraint_strict),
            "obj_keys": list(self.obj_keys),
            "protected_last_n_steps": int(self.protected_last_n_steps),
            "total_eval_wall_time_s": float(self.total_eval_wall_time_s),
            "total_search_wall_time_s": float(self._current_search_wall_time_s()),
            "search_peak_gpu_allocated_gb": float(self.search_peak_gpu_allocated_bytes) / (1024 ** 3),
            "search_peak_gpu_reserved_gb": float(self.search_peak_gpu_reserved_bytes) / (1024 ** 3),
            "convergence_csv": str(self.convergence_csv_path),
            "convergence_jsonl": str(self.convergence_jsonl_path),
        }

        if feasible:
            best_key = min(
                feasible,
                key=lambda k: (
                    float(self.vis_dict[k].get("latency_ms_per_img", float("inf"))),
                    float(self.vis_dict[k].get("fid", float("inf"))),
                    float(self.vis_dict[k].get("std", float("inf"))),
                ),
            )
            report["selection_mode"] = "quality_feasible_min_latency"
        else:
            best_key = min(
                evaluated,
                key=lambda k: (
                    float(self.vis_dict[k].get("fid", float("inf"))),
                    float(self.vis_dict[k].get("latency_ms_per_img", float("inf"))),
                ),
            )
            report["selection_mode"] = (
                "best_proxy_no_quality_fallback" if not self.quality_constraint_strict else "strict_no_feasible_found"
            )

        best_info = dict(self.vis_dict[best_key])
        report["best_candidate_key"] = best_key
        report["best_candidate"] = best_info
        return report

    # -------------------------
    # main search loop
    # -------------------------
    def search(self) -> Dict[str, Any]:
        print(
            "population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}".format(
                self.population_num,
                self.select_num,
                self.mutation_num,
                self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num,
                self.max_epochs,
            )
        )

        if not bool(self.cfg.get("use_ddim_init_x", False)):
            self.get_random_before_search(self.population_num)
        else:
            init_x = self._default_ddim_schedule()
            init_g = self._dense_genotype(num_steps=len(init_x))
            init_cand = _json_key(init_x, self.base_arch_idx, init_g)
            self.evaluate_candidate(init_cand)
            self.candidates.append(init_cand)
            self.get_random_before_search(self.population_num // 2 + 1)
            res = self.mutate_init_x(
                x0=init_cand,
                mutation_num=self.population_num - self.population_num // 2 - 1,
                m_prob=float(self.cfg.get("init_schedule_mutation_prob", 0.1)),
            )
            self.candidates += res

        self.save_population_snapshot("population_epoch0.json", self.candidates)
        self.dump_pareto_front()
        self._append_convergence_row(event="epoch_end")

        while self.epoch < self.max_epochs:
            print(f"epoch = {self.epoch}")
            self.update_top_k(self.candidates, k=self.select_num)
            self.update_top_k(self.candidates, k=50)

            print(f"epoch = {self.epoch} : top {len(self.keep_top_k[50])} (multi-objective) result")
            for i, cand in enumerate(self.keep_top_k[50]):
                info = self.vis_dict[cand]
                print(
                    "No.{} fid={:.6f} lat={:.4f} std={:.6f} ts={} arch={}".format(
                        i + 1,
                        info.get("fid", float("inf")),
                        info.get("latency_ms_per_img", float("inf")),
                        info.get("std", float("inf")),
                        info.get("timesteps", []),
                        json.dumps(info.get("arch", {}), sort_keys=True),
                    )
                )

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            self.candidates = mutation
            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand
            self.get_random(self.population_num)
            self.epoch += 1
            self.save_population_snapshot(f"population_epoch{self.epoch}.json", self.candidates)
            self.dump_pareto_front()
            self._append_convergence_row(event="epoch_end")

        report = self.select_final_candidate()
        final_path = self.outdir / "final_selection_report.json"
        final_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Done] final report -> {final_path}")
        print(f"[Done] convergence csv -> {self.convergence_csv_path}")
        print(f"[Done] convergence jsonl -> {self.convergence_jsonl_path}")
        return report


def build_schedule_search_space(cfg: Dict[str, Any]) -> Optional[List[int]]:
    search_space = cfg.get("schedule_search_space", None)
    search_space_path = cfg.get("schedule_search_space_path", None)
    ddim_seed_path = cfg.get("timesteps_npy_path", None)

    core = None
    if search_space is not None:
        core = sorted({int(x) for x in search_space})
    elif search_space_path:
        p = Path(str(search_space_path))
        if not p.exists():
            raise FileNotFoundError(f"schedule_search_space_path not found: {p}")
        arr = np.load(str(p))
        core = sorted({int(x) for x in arr.reshape(-1).tolist()})
    elif ddim_seed_path and bool(cfg.get("reuse_stage0_core_as_joint_space", False)):
        p = Path(str(ddim_seed_path))
        if p.exists():
            arr = np.load(str(p))
            core = sorted({int(x) for x in arr.reshape(-1).tolist()})

    if core is None:
        return None

    num_steps = int(cfg["num_steps"])
    radius = int(cfg.get("search_space_expand_radius", max(1, num_steps // 100)))
    expanded = []
    for s in core:
        left = max(int(s) - radius, 0)
        right = min(int(s) + radius, num_steps)
        expanded += list(range(left, right))
    expanded = sorted(set(expanded))
    return expanded


def main():
    parser = argparse.ArgumentParser(description="One-shot joint NSGA-II search over timestep + arch + step-wise policy")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must parse to a dict.")

    schedule_search_space = build_schedule_search_space(cfg)
    if schedule_search_space is not None:
        outdir = Path(cfg["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "joint_schedule_search_space.json").write_text(
            json.dumps([int(x) for x in schedule_search_space], indent=2), encoding="utf-8"
        )

    searcher = JointFullEvolutionSearcher(cfg, schedule_search_space=schedule_search_space)
    searcher.search()


if __name__ == "__main__":
    main()
