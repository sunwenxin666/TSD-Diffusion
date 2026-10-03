#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 evolutionary search for guided-diffusion NAS.

This mirrors the SD distill pipeline structure:
- timesteps are fixed externally
- each step has its own policy
- objectives are minimized jointly: FID + latency
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import json
import time
import random

import numpy as np
import sys
import os

# 添加上级目录到 sys.path，使同一包内的模块可以被导入
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from arch_space import (
    Genotype,
    SamplerPolicy,
    StepPolicy,
    crossover_step_policy,
    make_random_step_policy,
    mutate_step_policy,
)

logger = logging.getLogger(__name__)


def dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
    not_worse = True
    strictly_better = False
    for key in keys:
        av = float(a.get(key, float("inf")))
        bv = float(b.get(key, float("inf")))
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


def nsga2_select(candidates: List[Genotype], k: int, obj_keys: List[str]) -> List[Genotype]:
    pop_infos = []
    for g in candidates:
        m = g.metrics or {}
        pop_infos.append({key: float(m.get(key, float("inf"))) for key in obj_keys})

    fronts = fast_non_dominated_sort(pop_infos, obj_keys)
    selected: List[Genotype] = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend([candidates[idx] for idx in front])
            continue
        cd = crowding_distance(front, pop_infos, obj_keys)
        ordered = sorted(front, key=lambda idx: cd[idx], reverse=True)
        remain = k - len(selected)
        selected.extend([candidates[idx] for idx in ordered[:remain]])
        break
    return selected


class EvolutionarySearcher:
    def __init__(
        self,
        *,
        search_config: Dict[str, Any],
        total_blocks: int,
        searchable_block_ids: Optional[Sequence[int]] = None,
    ):
        self.cfg = dict(search_config)
        self.total_blocks = int(total_blocks)
        if searchable_block_ids is None:
            self.searchable_block_ids = tuple(range(max(0, self.total_blocks)))
        else:
            self.searchable_block_ids = tuple(sorted({int(x) for x in searchable_block_ids}))

        self.population_num = int(self.cfg.get("population_num", 12))
        self.select_num = int(self.cfg.get("select_num", 4))
        self.mutation_num = int(self.cfg.get("mutation_num", 4))
        self.crossover_num = int(self.cfg.get("crossover_num", 4))
        self.m_prob = float(self.cfg.get("m_prob", 0.35))
        self.max_skip_ratio = float(self.cfg.get("skip_max_ratio", 0.35))
        self.obj_keys = ["fid", "latency_ms_per_img"]
        self.fid_max = float(self.cfg.get("fid_max", float("inf")))
        self.std_max = float(self.cfg.get("std_max", float("inf")))
        self.std_penalty_weight = float(self.cfg.get("std_penalty_weight", 0.0))
        self.is_min = self.cfg.get("is_min", None)
        self.is_penalty_weight = float(self.cfg.get("is_penalty_weight", 0.0))
        self.precision_min = self.cfg.get("precision_min", None)
        self.recall_min = self.cfg.get("recall_min", None)
        self.pr_penalty_weight = float(self.cfg.get("pr_penalty_weight", 0.0))
        self.kid_max = self.cfg.get("kid_max", None)
        self.kid_penalty_weight = float(self.cfg.get("kid_penalty_weight", 0.0))
        self.quality_fid_baseline = self.cfg.get("quality_fid_baseline", None)
        if self.quality_fid_baseline is not None:
            self.quality_fid_baseline = float(self.quality_fid_baseline)
        self.quality_fid_delta = float(self.cfg.get("quality_fid_delta", float("inf")))
        self.quality_constraint_strict = bool(self.cfg.get("quality_constraint_strict", False))
        self.protected_last_n_steps = max(0, int(self.cfg.get("protected_last_n_steps", 0)))
        self.policy_search_mode = str(self.cfg.get("policy_search_mode", "hybrid")).lower()
        if self.policy_search_mode not in {"hybrid", "soft_only"}:
            raise ValueError(
                f"Unsupported policy_search_mode={self.policy_search_mode!r}. "
                "Expected 'hybrid' or 'soft_only'."
            )
        self.final_selection_priority = str(self.cfg.get("final_selection_priority", "fid_first")).lower()
        if self.final_selection_priority not in {"fid_first", "latency_first"}:
            raise ValueError(
                f"Unsupported final_selection_priority={self.final_selection_priority!r}. "
                "Expected 'fid_first' or 'latency_first'."
            )

        self.population: List[Genotype] = []
        self.top_k: List[Genotype] = []

        self.epoch = 0
        self.search_start_wall_time = time.time()
        self.search_peak_gpu_allocated_gb = 0.0
        self.search_peak_gpu_reserved_gb = 0.0
        self.evaluated_archive: Dict[str, Dict[str, Any]] = {}
        self.total_evaluated_candidates = 0
        self.convergence_history: List[Dict[str, Any]] = []
        self.search_artifact_dir = str(self.cfg.get("search_artifact_dir", self.cfg.get("save_dir", "")) or "")

    def genotype_key(self, g: Genotype) -> str:
        step_key = tuple(
            (
                tuple(int(x) for x in p.skip_blocks),
                tuple((int(layer_id), round(float(gate), 4)) for layer_id, gate in (p.layer_gates or tuple())),
                round(float(getattr(p, "class_gain", 1.0)), 4),
            )
            for p in g.step_policies
        )
        return str((step_key, round(float(g.sampler_policy.ddim_eta), 6)))

    def _snapshot_metrics(self, metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not metrics:
            return out
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                out[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
                out[k] = list(v)
        return out

    def _update_search_bookkeeping(self, evaluated_population: Sequence[Genotype]) -> None:
        self.total_evaluated_candidates += int(len(evaluated_population))
        for g in evaluated_population:
            metrics = dict(g.metrics or {})
            if len(metrics) == 0:
                continue
            key = self.genotype_key(g)
            self.evaluated_archive[key] = self._snapshot_metrics(metrics)
            self.search_peak_gpu_allocated_gb = max(
                self.search_peak_gpu_allocated_gb,
                float(metrics.get("peak_gpu_allocated_gb", 0.0) or 0.0),
            )
            self.search_peak_gpu_reserved_gb = max(
                self.search_peak_gpu_reserved_gb,
                float(metrics.get("peak_gpu_reserved_gb", 0.0) or 0.0),
            )

    def _best_archive_key(self, metric_name: str) -> Optional[str]:
        valid = [k for k, m in self.evaluated_archive.items() if np.isfinite(float(m.get(metric_name, float("inf"))))]
        if not valid:
            return None
        return min(valid, key=lambda k: float(self.evaluated_archive[k].get(metric_name, float("inf"))))

    def _archive_fronts(self):
        keys = list(self.evaluated_archive.keys())
        if not keys:
            return keys, []
        pop_infos = [
            {obj_key: float(self.evaluated_archive[k].get(obj_key, float("inf"))) for obj_key in self.obj_keys}
            for k in keys
        ]
        fronts = fast_non_dominated_sort(pop_infos, self.obj_keys)
        return keys, fronts

    def _append_convergence_row(self) -> None:
        if not self.evaluated_archive:
            return
        keys, fronts = self._archive_fronts()
        best_fid_key = self._best_archive_key("fid")
        best_lat_key = self._best_archive_key("latency_ms_per_img")
        feasible = [k for k, m in self.evaluated_archive.items() if np.isfinite(float(m.get("fid", float("inf"))))]
        row = {
            "epoch": int(self.epoch),
            "evaluated_candidates": int(self.total_evaluated_candidates),
            "unique_evaluated_candidates": int(len(self.evaluated_archive)),
            "quality_feasible_candidates": int(len(feasible)),
            "pareto_size": int(len(fronts[0])) if fronts else 0,
            "best_fid_so_far": float(self.evaluated_archive[best_fid_key].get("fid", float("inf"))) if best_fid_key else float("inf"),
            "best_latency_so_far": float(self.evaluated_archive[best_lat_key].get("latency_ms_per_img", float("inf"))) if best_lat_key else float("inf"),
            "search_wall_time_s": float(time.time() - self.search_start_wall_time),
            "search_peak_gpu_allocated_gb": float(self.search_peak_gpu_allocated_gb),
            "search_peak_gpu_reserved_gb": float(self.search_peak_gpu_reserved_gb),
        }
        self.convergence_history.append(row)

    def dump_search_artifacts(
        self,
        outdir: Optional[str] = None,
        *,
        final_population: Optional[Sequence[Genotype]] = None,
        selected: Optional[Genotype] = None,
        selection_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        outdir = str(outdir or self.search_artifact_dir or "")
        if not outdir:
            return
        os.makedirs(outdir, exist_ok=True)

        if self.convergence_history:
            jsonl_path = os.path.join(outdir, "convergence_history.jsonl")
            csv_path = os.path.join(outdir, "convergence_history.csv")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for row in self.convergence_history:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            keys = list(self.convergence_history[0].keys())
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for row in self.convergence_history:
                    f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

        archive_keys, fronts = self._archive_fronts()
        if archive_keys and fronts:
            pareto_keys = [archive_keys[idx] for idx in fronts[0]]
            pareto_path = os.path.join(outdir, "pareto_front_latest.jsonl")
            with open(pareto_path, "w", encoding="utf-8") as f:
                for key in pareto_keys:
                    row = {"genotype_key": key}
                    row.update(self.evaluated_archive[key])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        report = dict(selection_report or {})
        report.update({
            "obj_keys": list(self.obj_keys),
            "epochs_completed": int(self.epoch),
            "evaluated_candidates": int(self.total_evaluated_candidates),
            "unique_evaluated_candidates": int(len(self.evaluated_archive)),
            "search_wall_time_s": float(time.time() - self.search_start_wall_time),
            "search_peak_gpu_allocated_gb": float(self.search_peak_gpu_allocated_gb),
            "search_peak_gpu_reserved_gb": float(self.search_peak_gpu_reserved_gb),
        })
        if final_population is not None:
            report["final_population_size"] = int(len(list(final_population)))
        if selected is not None:
            report["selected_genotype_key"] = self.genotype_key(selected)
            report["selected_metrics"] = self._snapshot_metrics(selected.metrics or {})
        with open(os.path.join(outdir, "final_selection_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def quality_fid_limit(self) -> float:
        limit = float(self.fid_max)
        if self.quality_fid_baseline is not None and np.isfinite(self.quality_fid_delta):
            limit = min(limit, float(self.quality_fid_baseline) + float(self.quality_fid_delta))
        return limit

    def _apply_step_guards(self, step_policies: Sequence[StepPolicy]) -> Tuple[StepPolicy, ...]:
        policies = list(step_policies)
        if self.policy_search_mode == "soft_only":
            policies = [replace(pol, skip_blocks=tuple()) for pol in policies]
        if self.protected_last_n_steps > 0 and policies:
            keep_dense_from = max(0, len(policies) - self.protected_last_n_steps)
            for idx in range(keep_dense_from, len(policies)):
                policies[idx] = StepPolicy(skip_blocks=tuple(), layer_gates=tuple(), class_gain=1.0)
        return tuple(policies)

    def is_quality_feasible(self, g: Genotype) -> bool:
        metrics = dict(g.metrics or {})
        fid_v = float(metrics.get("fid_raw", metrics.get("fid", float("inf"))))
        std_v = float(metrics.get("std", float("inf")))
        if fid_v > self.quality_fid_limit():
            return False
        if std_v > self.std_max:
            return False
        return True

    def quality_feasible_population(self, population: Sequence[Genotype]) -> List[Genotype]:
        return [g for g in population if self.is_quality_feasible(g)]

    def select_final_candidate(
        self,
        population: Sequence[Genotype],
        *,
        fallback: Optional[Genotype] = None,
    ) -> Tuple[Genotype, Dict[str, Any]]:
        feasible = self.quality_feasible_population(population)
        report = {
            "quality_fid_baseline": self.quality_fid_baseline,
            "quality_fid_delta": float(self.quality_fid_delta),
            "quality_fid_limit": float(self.quality_fid_limit()),
            "quality_constraint_strict": bool(self.quality_constraint_strict),
            "final_selection_priority": str(self.final_selection_priority),
            "population_size": int(len(population)),
            "quality_feasible_size": int(len(feasible)),
            "protected_last_n_steps": int(self.protected_last_n_steps),
            "evaluated_candidates": int(self.total_evaluated_candidates),
            "unique_evaluated_candidates": int(len(self.evaluated_archive)),
            "search_wall_time_s": float(time.time() - self.search_start_wall_time),
            "search_peak_gpu_allocated_gb": float(self.search_peak_gpu_allocated_gb),
            "search_peak_gpu_reserved_gb": float(self.search_peak_gpu_reserved_gb),
        }
        if feasible:
            if self.final_selection_priority == "latency_first":
                key_fn = lambda g: (
                    float((g.metrics or {}).get("latency_ms_per_img", float("inf"))),
                    float((g.metrics or {}).get("fid", float("inf"))),
                    float((g.metrics or {}).get("std", float("inf"))),
                )
                report["selection_mode"] = "quality_feasible_min_latency"
            else:
                # Safer for LSUN-Bedroom 256: do not sacrifice FID for tiny latency gains.
                key_fn = lambda g: (
                    float((g.metrics or {}).get("fid", float("inf"))),
                    float((g.metrics or {}).get("std", float("inf"))),
                    float((g.metrics or {}).get("latency_ms_per_img", float("inf"))),
                )
                report["selection_mode"] = "quality_feasible_min_fid"
            chosen = sorted(feasible, key=key_fn)[0]
            return chosen, report
        if fallback is not None and self.quality_constraint_strict:
            report["selection_mode"] = "fallback_dense_policy"
            return fallback, report
        chosen = sorted(
            list(population),
            key=lambda g: (
                float((g.metrics or {}).get("fid", float("inf"))),
                float((g.metrics or {}).get("latency_ms_per_img", float("inf"))),
            ),
        )[0]
        report["selection_mode"] = "best_proxy_no_quality_fallback"
        return chosen, report

    def init_population(self, num_steps: int) -> List[Genotype]:
        self.population = [self._random_genotype(num_steps=num_steps) for _ in range(self.population_num)]
        return self.population

    def _random_genotype(self, num_steps: int) -> Genotype:
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))
        init_cfg = dict(self.cfg.get("step_policy_init", {}))
        p_skip_high = tuple(init_cfg.get("p_skip_high", (0.0, 0.1)))
        p_skip_mid = tuple(init_cfg.get("p_skip_mid", (0.05, 0.2)))
        p_skip_low = tuple(init_cfg.get("p_skip_low", (0.1, 0.35)))
        gate_scale_range = tuple(init_cfg.get("gate_scale_range", (1.0, 1.0)))
        class_gain_range = tuple(init_cfg.get("class_gain_range", (1.0, 1.0)))

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
                    total_blocks=self.total_blocks,
                    searchable_block_ids=self.searchable_block_ids,
                    rng=rng,
                    p_skip_range=(float(p_skip[0]), float(p_skip[1])),
                    max_skip_ratio=self.max_skip_ratio,
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

    def _mutate(self, g: Genotype) -> Genotype:
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))
        mut_cfg = dict(self.cfg.get("step_policy_mutation", {}))
        per_step_p = float(mut_cfg.get("per_step_mutate_prob", 0.35))
        flip_prob = float(mut_cfg.get("mask_flip_prob", 0.08))
        gate_mutate_prob = float(mut_cfg.get("gate_mutate_prob", flip_prob))
        gate_sigma = float(mut_cfg.get("gate_sigma", 0.03))
        class_gain_sigma = float(mut_cfg.get("class_gain_sigma", 0.02))

        step_policies: List[StepPolicy] = []
        for pol in g.step_policies:
            if rng.rand() < per_step_p:
                step_policies.append(
                    mutate_step_policy(
                        pol,
                        total_blocks=self.total_blocks,
                        searchable_block_ids=self.searchable_block_ids,
                        rng=rng,
                        flip_prob=flip_prob,
                        max_skip_ratio=self.max_skip_ratio,
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

    def _crossover(self, a: Genotype, b: Genotype) -> Genotype:
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))
        n = min(len(a.step_policies), len(b.step_policies))
        step_policies: List[StepPolicy] = []
        for idx in range(n):
            step_policies.append(
                crossover_step_policy(
                    a.step_policies[idx],
                    b.step_policies[idx],
                    total_blocks=self.total_blocks,
                    searchable_block_ids=self.searchable_block_ids,
                    rng=rng,
                    max_skip_ratio=self.max_skip_ratio,
                )
            )
        if len(a.step_policies) > n:
            step_policies.extend(a.step_policies[n:])
        elif len(b.step_policies) > n:
            step_policies.extend(b.step_policies[n:])

        return Genotype(
            step_policies=self._apply_step_guards(step_policies),
            sampler_policy=a.sampler_policy,
            fitness=-1.0,
            metrics={},
        )

    def evolve(self, evaluated_population: List[Genotype]) -> None:
        self._update_search_bookkeeping(evaluated_population)
        if not evaluated_population:
            self.population = []
            self.top_k = []
            self._append_convergence_row()
            if self.search_artifact_dir:
                self.dump_search_artifacts(self.search_artifact_dir)
            self.epoch += 1
            return

        feasible_pool = []
        quality_limit = self.quality_fid_limit()
        for g in evaluated_population:
            metrics = g.metrics or {}
            fid_v = float(metrics.get("fid_raw", metrics.get("fid", float("inf"))))
            std_v = float(metrics.get("std", float("inf")))
            if fid_v > quality_limit:
                continue
            if std_v > self.std_max:
                continue
            feasible_pool.append(g)

        pool = feasible_pool if len(feasible_pool) >= max(2, self.select_num) else list(evaluated_population)
        pool = self._apply_soft_penalties(pool)

        elites = nsga2_select(pool, self.select_num, self.obj_keys)
        if not elites:
            elites = pool[: min(len(pool), self.select_num)]
        self.top_k = elites

        next_pop: List[Genotype] = list(elites)
        seen = {self.genotype_key(g) for g in next_pop}

        mutation_target = min(self.population_num, len(next_pop) + self.mutation_num)
        while len(next_pop) < mutation_target:
            child = self._mutate(random.choice(elites))
            key = self.genotype_key(child)
            if key in seen:
                continue
            seen.add(key)
            next_pop.append(child)

        crossover_target = min(self.population_num, len(next_pop) + self.crossover_num)
        while len(next_pop) < crossover_target:
            if len(elites) >= 2:
                pa, pb = random.sample(elites, k=2)
            else:
                pa = pb = elites[0]
            child = self._crossover(pa, pb)
            key = self.genotype_key(child)
            if key in seen:
                continue
            seen.add(key)
            next_pop.append(child)

        while len(next_pop) < self.population_num:
            child = self._random_genotype(num_steps=len(elites[0].step_policies))
            key = self.genotype_key(child)
            if key in seen:
                continue
            seen.add(key)
            next_pop.append(child)

        self.population = next_pop[: self.population_num]
        self._append_convergence_row()
        if self.search_artifact_dir:
            self.dump_search_artifacts(self.search_artifact_dir)
        self.epoch += 1

    def _apply_soft_penalties(self, population: List[Genotype]) -> List[Genotype]:
        adjusted: List[Genotype] = []
        for g in population:
            metrics = dict(g.metrics or {})
            if "fid" in metrics:
                fid = float(metrics["fid"])
                if self.std_penalty_weight > 0.0:
                    fid += self.std_penalty_weight * float(metrics.get("std", 0.0))
                if self.is_penalty_weight > 0.0 and self.is_min is not None and "is_mean" in metrics:
                    fid += self.is_penalty_weight * max(0.0, float(self.is_min) - float(metrics.get("is_mean", 0.0)))
                if self.pr_penalty_weight > 0.0:
                    deficit = 0.0
                    if self.precision_min is not None and "precision" in metrics:
                        deficit += max(0.0, float(self.precision_min) - float(metrics.get("precision", 0.0)))
                    if self.recall_min is not None and "recall" in metrics:
                        deficit += max(0.0, float(self.recall_min) - float(metrics.get("recall", 0.0)))
                    fid += self.pr_penalty_weight * deficit
                if self.kid_penalty_weight > 0.0 and self.kid_max is not None and "kid_mean" in metrics:
                    fid += self.kid_penalty_weight * max(0.0, float(metrics.get("kid_mean", 0.0)) - float(self.kid_max))
                metrics["fid_raw"] = float(g.metrics.get("fid", float("inf")))
                metrics["fid"] = float(fid)
            lat_v = float(metrics.get("latency_ms_per_img", float("inf")))
            fid_v = float(metrics.get("fid", float("inf")))
            fitness = float("inf") if (not np.isfinite(fid_v) or not np.isfinite(lat_v)) else (fid_v + 0.001 * lat_v)
            adjusted.append(replace(g, metrics=metrics, fitness=fitness))
        return adjusted
