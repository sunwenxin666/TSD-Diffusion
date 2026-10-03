import random
import time
import copy
import logging
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

from .search_space import Genotype, SegmentPolicy, SamplerPolicy
from .search_space import (
    make_random_schedule_low_noise_bias,
    sanitize_u_schedule,
    mutate_schedule,
    make_random_boundaries,
    make_random_segment_policy,
    mutate_segment_policy,
    crossover_segment_policy,
    mutate_sampler_policy,        
    crossover_sampler_policy,
)

logger = logging.getLogger(__name__)


# -------------------------
#  NSGA-II utilities
# -------------------------

def dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
    """All objectives are MINIMIZE."""
    not_worse_all = True
    strictly_better = False
    for k in keys:
        av = float(a.get(k, float("inf")))
        bv = float(b.get(k, float("inf")))
        if av > bv:
            not_worse_all = False
            break
        if av < bv:
            strictly_better = True
    return not_worse_all and strictly_better


def fast_non_dominated_sort(pop_infos: List[Dict[str, float]], obj_keys: List[str]):
    """Return fronts: list of list of indices."""
    S = [[] for _ in range(len(pop_infos))]
    n = [0 for _ in range(len(pop_infos))]
    rank = [0 for _ in range(len(pop_infos))]
    fronts = [[]]

    for p in range(len(pop_infos)):
        for q in range(len(pop_infos)):
            if p == q:
                continue
            if dominates(pop_infos[p], pop_infos[q], obj_keys):
                S[p].append(q)
            elif dominates(pop_infos[q], pop_infos[p], obj_keys):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    fronts.pop()
    return fronts, rank


def crowding_distance(front_indices: List[int], pop_infos: List[Dict[str, float]], obj_keys: List[str]):
    dist = {idx: 0.0 for idx in front_indices}
    if len(front_indices) <= 2:
        for idx in front_indices:
            dist[idx] = float("inf")
        return dist

    for k in obj_keys:
        front_sorted = sorted(front_indices, key=lambda i: float(pop_infos[i].get(k, float("inf"))))
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")

        vmin = float(pop_infos[front_sorted[0]].get(k, float("inf")))
        vmax = float(pop_infos[front_sorted[-1]].get(k, float("inf")))
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0

        for j in range(1, len(front_sorted) - 1):
            prev_v = float(pop_infos[front_sorted[j - 1]].get(k, float("inf")))
            next_v = float(pop_infos[front_sorted[j + 1]].get(k, float("inf")))
            dist[front_sorted[j]] += (next_v - prev_v) / denom

    return dist


def nsga2_select(candidates: List[Genotype], k: int, obj_keys: List[str]) -> List[Genotype]:
    """
    candidates must already have metrics for keys.
    Missing metrics are treated as +inf.
    """
    pop_infos = []
    for g in candidates:
        m = g.metrics or {}
        pop_infos.append({kk: float(m.get(kk, float("inf"))) for kk in obj_keys})

    fronts, _ = fast_non_dominated_sort(pop_infos, obj_keys)
    selected: List[Genotype] = []

    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend([candidates[i] for i in front])
        else:
            cd = crowding_distance(front, pop_infos, obj_keys)
            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
            remain = k - len(selected)
            selected.extend([candidates[i] for i in front_sorted[:remain]])
            break

    return selected


# -------------------------
#  Evolution Searcher (MO)
# -------------------------

class EvolutionSearcher:
    """
    Multi-objective EA (NSGA-II style) over Genotype.
    Objectives (minimize): fid, latency

    Key changes vs your original:
    1) mask_len / full_mask_len support:
       - mask_len: recommended, skippable blocks count (len(adapter._skippable_indices))
       - full_mask_len: legacy, total wrapped blocks count (adapter.total_layers)
    2) genotype_key includes stage_cond_gain and stage_gate_scale (critical once adapter uses them)
    3) random init/mutation/crossover now operate in the SAME search_space definitions:
       - schedule: make_random_schedule_low_noise_bias + sanitize_u_schedule + mutate_schedule
       - segment policy: make_random_segment_policy + crossover_segment_policy
    4) mutation projects any legacy full-mask down to skippable mask_len (simple deterministic projection)
    5) optional de-dup during offspring generation to reduce redundant evals
    """

    def __init__(self, search_config: Dict[str, Any], adapter_info: Dict[str, int]):
        self.cfg = search_config

        # "step" (recommended): fixed timesteps + per-step policies
        # "legacy": searched u_schedule + segment boundaries
        self.policy_mode = str(self.cfg.get("policy_mode", "step")).lower()

        # stages (must match adapter total_stages)
        self.total_stages = int(adapter_info["total_stages"])

        # legacy full mask length (all wrapped blocks)
        self.full_mask_len = int(adapter_info["total_layers"])

        # recommended mask length (skippable blocks only)
        self.mask_len = int(adapter_info.get("skippable_layers", self.full_mask_len))

        # EA hyperparameters
        self.population_num = int(self.cfg.get("population_num", 50))
        self.select_num = int(self.cfg.get("select_num", 10))
        self.mutation_num = int(self.cfg.get("mutation_num", 20))
        self.crossover_num = int(self.cfg.get("crossover_num", 20))
        self.m_prob = float(self.cfg.get("m_prob", 0.1))

        # dual objectives (all minimize)
        self.obj_keys = ["fid", "latency"]

        # optional stability constraint / penalty
        self.std_max = float(self.cfg.get("std_max", float("inf")))  # hard constraint
        self.std_penalty_weight = float(self.cfg.get("std_penalty_weight", 0.0))  # soft penalty

        # Multi-objective evaluation parameters (kept for config compatibility)
        self.mo_eval_repeats = int(self.cfg.get("mo_eval_repeats", 3))
        self.mo_fid_warmup_repeats = int(self.cfg.get("mo_fid_warmup_repeats", 0))
        self.mo_seed_base = int(self.cfg.get("mo_seed_base", 12345))

        # Early stopping parameters
        self.early_stop_patience = int(self.cfg.get("early_stop_patience", 10))
        self.early_stop_delta = float(self.cfg.get("early_stop_delta", 0.001))
        self.early_stop_enabled = bool(self.cfg.get("early_stop_enabled", True))

        # Prompts configuration (max_prompts for truncation)
        self.max_prompts = self.cfg.get("max_prompts", None)
        if self.max_prompts is not None:
            logger.info("EvolutionSearcher: max_prompts=%d (prompts will be truncated)", self.max_prompts)
        else:
            logger.info("EvolutionSearcher: max_prompts=None (using all prompts)")

        # Tracking for early stopping
        self.best_fid_history: List[float] = []
        self.no_improve_epochs = 0
        self.epoch = 0

        self.population: List[Genotype] = []
        self.top_k: List[Genotype] = []
        self.vis_dict: Dict[str, Dict[str, Any]] = {}  # Cache for evaluated genotypes

        logger.info(
            "EvolutionSearcher init: population=%d select=%d mutation=%d crossover=%d m_prob=%.3f | "
            "mask_len(skippable)=%d full_mask_len=%d stages=%d",
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.m_prob,
            self.mask_len, self.full_mask_len, self.total_stages
        )

    # ---- genotype key for de-dup / caching (stable string) ----
    def genotype_key(self, g: Genotype) -> str:
        # NOTE: timesteps are fixed externally, so in "step" mode we do NOT
        # include timesteps in the key; only include the per-step policies.
        us = tuple(round(float(x), 6) for x in g.u_schedule)
        sb = tuple(round(float(x), 6) for x in g.segment_boundaries)

        sp = g.sampler_policy
        sp_key = (round(float(sp.fm_strength), 6), round(float(sp.fm_t_start), 6), int(sp.order))

        def _pol_key(pol: SegmentPolicy):
            mask_s = "".join("1" if b else "0" for b in pol.layer_mask)
            cg = tuple(round(float(x), 6) for x in pol.stage_cond_gain)
            gs = tuple(round(float(x), 6) for x in pol.stage_gate_scale)
            return (mask_s, cg, gs)

        if getattr(g, "step_policies", None):
            step_keys = tuple(_pol_key(pol) for pol in g.step_policies)
            return str(("step", sp_key, step_keys))

        seg_keys = tuple(_pol_key(pol) for pol in g.segment_policies)
        return str(("legacy", us, sb, sp_key, seg_keys))

    # ---- RNG state management (kept) ----
    def _capture_rng_state(self) -> Dict[str, Any]:
        import torch
        state = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": None}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        import torch
        random.setstate(state["py"])
        np.random.set_state(state["np"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    # ---- early stopping check ----
    def _check_early_stop_with_reason(self, current_best_fid: float) -> Tuple[bool, str]:
        if not self.early_stop_enabled:
            return False, ""

        self.best_fid_history.append(float(current_best_fid))
        self.epoch += 1

        if len(self.best_fid_history) < 2:
            return False, ""

        improvement = self.best_fid_history[-2] - float(current_best_fid)
        if improvement > self.early_stop_delta:
            self.no_improve_epochs = 0
            logger.info("Epoch %d: Improvement detected (ΔFID=%.6f > %.6f)", self.epoch, improvement, self.early_stop_delta)
            return False, ""

        self.no_improve_epochs += 1
        logger.info(
            "Epoch %d: No improvement for %d/%d epochs (best FID: %.6f)",
            self.epoch, self.no_improve_epochs, self.early_stop_patience, float(current_best_fid)
        )

        if self.no_improve_epochs >= self.early_stop_patience:
            reason = f"No improvement for {self.no_improve_epochs} epochs (patience={self.early_stop_patience}, delta={self.early_stop_delta})"
            logger.warning("Early stopping: %s", reason)
            return True, reason

        return False, ""

    # ---- random init ----
    def init_population(self, num_steps: int, num_segments: int) -> List[Genotype]:
        """Initialize population.

        In "step" mode, num_segments is ignored (kept only for API compatibility).
        """
        pop = [self._random_genotype(num_steps=num_steps, num_segments=num_segments) for _ in range(self.population_num)]
        self.population = pop
        return pop

    def _random_sampler_policy(self, rng: np.random.RandomState) -> SamplerPolicy:
        fm_strength = float(np.clip(rng.uniform(0.03, 0.12), 0.0, 0.3))
        fm_t_start = float(np.clip(rng.uniform(0.7, 0.9), 0.5, 0.95))
        order = int(rng.choice([1, 2]))
        return SamplerPolicy(fm_strength=fm_strength, fm_t_start=fm_t_start, order=order)

    def _random_genotype(self, num_steps: int, num_segments: int) -> Genotype:
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))

        # ------------------------------
        # NEW: fixed timesteps => per-step policies
        # ------------------------------
        if self.policy_mode == "step":
            init_cfg = dict(self.cfg.get("step_policy_init", {}))
            p_keep_high = tuple(init_cfg.get("p_keep_high", (0.2, 0.6)))
            p_keep_mid = tuple(init_cfg.get("p_keep_mid", (0.5, 1.0)))
            p_keep_low = tuple(init_cfg.get("p_keep_low", (0.8, 1.0)))
            gain_range = tuple(init_cfg.get("gain_range", (0.85, 1.15)))
            gate_range = tuple(init_cfg.get("gate_range", (0.85, 1.15)))

            # We don't know the exact meaning of each timestep here (could be sigma/t/index).
            # But we can still bias by step position: early steps ~ high noise, late ~ low noise.
            step_policies: List[SegmentPolicy] = []
            for i in range(int(num_steps)):
                pos = i / max(1.0, float(num_steps - 1))  # 0..1
                # early (pos small) => high noise
                if pos < 0.33:
                    pk = p_keep_high
                elif pos < 0.66:
                    pk = p_keep_mid
                else:
                    pk = p_keep_low
                step_policies.append(
                    make_random_segment_policy(
                        self.mask_len,
                        self.total_stages,
                        rng=rng,
                        p_keep_range=(float(pk[0]), float(pk[1])),
                        gain_range=(float(gain_range[0]), float(gain_range[1])),
                        gate_range=(float(gate_range[0]), float(gate_range[1])),
                    )
                )

            return Genotype(
                u_schedule=tuple(),
                segment_policies=tuple(),
                segment_boundaries=tuple(),
                step_policies=tuple(step_policies),
                sampler_policy=self._random_sampler_policy(rng),
                fitness=-1.0,
                metrics={},
            )

        # ------------------------------
        # LEGACY: searched schedule + segment policies
        # ------------------------------
        u_schedule = make_random_schedule_low_noise_bias(num_steps=num_steps, N=1000, rng=rng)
        if num_segments <= 1:
            boundaries = tuple()
            seg_policies = (make_random_segment_policy(self.mask_len, self.total_stages, rng),)
        else:
            boundaries = make_random_boundaries(num_segments=num_segments, rng=rng)
            seg_policies = tuple(make_random_segment_policy(self.mask_len, self.total_stages, rng) for _ in range(num_segments))

        return Genotype(
            u_schedule=u_schedule,
            segment_policies=seg_policies,
            segment_boundaries=boundaries,
            step_policies=tuple(),
            sampler_policy=self._random_sampler_policy(rng),
            fitness=-1.0,
            metrics={},
        )

    # ---- helpers: project any legacy full-mask to skippable mask_len ----
    def _project_mask(self, lm: Tuple[bool, ...]) -> Tuple[bool, ...]:
        """
        Ensure mask length equals self.mask_len.
        If a legacy full-mask appears, we project by taking the first mask_len bits.
        (Precise mapping would require adapter._skippable_indices; searcher typically doesn't have adapter.)
        """
        if lm is None:
            lm = tuple()
        if len(lm) == self.mask_len:
            out = lm
        elif len(lm) == self.full_mask_len and self.mask_len <= self.full_mask_len:
            out = tuple(bool(x) for x in lm[: self.mask_len])
        else:
            out = tuple([True] * self.mask_len)

        if len(out) > 0 and (not any(out)):
            out = (True,) + tuple(out[1:])  # deterministic fix
        return out

    # ---- genetic operators ----
    def _mutate(self, g: Genotype) -> Genotype:
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))

        # ------------------------------------------------------------
        # STEP MODE: mutate per-step policies (timesteps fixed)
        # ------------------------------------------------------------
        if self.policy_mode == "step":
            mut_cfg = dict(self.cfg.get("step_policy_mutation", {}))
            per_step_p = float(mut_cfg.get("per_step_mutate_prob", 0.35))
            mask_flip_prob = float(mut_cfg.get("mask_flip_prob", 0.08))
            gain_sigma = float(mut_cfg.get("gain_sigma", 0.05))
            gate_sigma = float(mut_cfg.get("gate_sigma", 0.05))

            step_policies: List[SegmentPolicy] = []
            for pol in (g.step_policies or tuple()):
                if rng.rand() < per_step_p:
                    step_policies.append(
                        mutate_segment_policy(
                            pol,
                            mask_len=self.mask_len,
                            total_stages=self.total_stages,
                            rng=rng,
                            mask_flip_prob=mask_flip_prob,
                            gain_sigma=gain_sigma,
                            gate_sigma=gate_sigma,
                        )
                    )
                else:
                    step_policies.append(pol)

            sp = g.sampler_policy
            if rng.rand() < self.m_prob:
                sp = mutate_sampler_policy(sp, rng)

            return Genotype(
                u_schedule=tuple(),
                segment_policies=tuple(),
                segment_boundaries=tuple(),
                step_policies=tuple(step_policies),
                sampler_policy=sp,
                fitness=-1.0,
                metrics={},
            )

        # schedule mutation (stable)
        u = mutate_schedule(
            g.u_schedule,
            rng=rng,
            N=1000,
            target_length=len(g.u_schedule),
        )

        # policies mutation
        seg_policies: List[SegmentPolicy] = []
        for pol in g.segment_policies:
            # mask: project to mask_len then flip
            mask = list(self._project_mask(pol.layer_mask))
            for i in range(len(mask)):
                if random.random() < self.m_prob:
                    mask[i] = not mask[i]
            if len(mask) > 0 and (not any(mask)):
                mask[random.randrange(len(mask))] = True

            # stage vectors: enforce length == total_stages then perturb
            if pol.stage_cond_gain is None or len(pol.stage_cond_gain) != self.total_stages:
                scg0 = [1.0] * self.total_stages
            else:
                scg0 = list(pol.stage_cond_gain)

            if pol.stage_gate_scale is None or len(pol.stage_gate_scale) != self.total_stages:
                sgs0 = [1.0] * self.total_stages
            else:
                sgs0 = list(pol.stage_gate_scale)

            scg = [float(np.clip(x + rng.normal(0.0, 0.05), 0.5, 1.5)) for x in scg0]
            sgs = [float(np.clip(x + rng.normal(0.0, 0.05), 0.5, 1.5)) for x in sgs0]

            seg_policies.append(SegmentPolicy(layer_mask=tuple(mask), stage_cond_gain=tuple(scg), stage_gate_scale=tuple(sgs)))

        # sampler policy mutation
        sp = g.sampler_policy
        if random.random() < self.m_prob:
            sp = SamplerPolicy(
                fm_strength=float(np.clip(sp.fm_strength + rng.normal(0.0, 0.02), 0.0, 0.3)),
                fm_t_start=float(np.clip(sp.fm_t_start + rng.normal(0.0, 0.05), 0.5, 0.95)),
                order=int(rng.choice([1, 2, 3])),
            )

        return Genotype(
            u_schedule=u,
            segment_policies=tuple(seg_policies),
            segment_boundaries=g.segment_boundaries,
            step_policies=tuple(),
            sampler_policy=sp,
            fitness=-1.0,
            metrics={}
        )

    def _crossover(self, a: Genotype, b: Genotype) -> Genotype:
        rng = np.random.RandomState(random.randint(0, 2**32 - 1))

        # ------------------------------------------------------------
        # STEP MODE: crossover per-step policies (timesteps fixed)
        # ------------------------------------------------------------
        if self.policy_mode == "step":
            n = min(len(a.step_policies), len(b.step_policies))
            step_policies: List[SegmentPolicy] = []
            for i in range(n):
                step_policies.append(
                    crossover_segment_policy(
                        a.step_policies[i],
                        b.step_policies[i],
                        mask_len=self.mask_len,
                        total_stages=self.total_stages,
                        rng=rng,
                    )
                )
            # append remainder from the longer parent
            if len(a.step_policies) > n:
                step_policies.extend([a.step_policies[i] for i in range(n, len(a.step_policies))])
            elif len(b.step_policies) > n:
                step_policies.extend([b.step_policies[i] for i in range(n, len(b.step_policies))])

            sp = crossover_sampler_policy(a.sampler_policy, b.sampler_policy, rng)

            return Genotype(
                u_schedule=tuple(),
                segment_policies=tuple(),
                segment_boundaries=tuple(),
                step_policies=tuple(step_policies),
                sampler_policy=sp,
                fitness=-1.0,
                metrics={},
            )

        # schedule: per-position pick then sanitize
        ua = list(a.u_schedule)
        ub = list(b.u_schedule)
        u = [1.0]
        for i in range(1, len(ua)):
            u.append(float(ua[i] if random.random() < 0.5 else ub[i]))
        u = sorted(u, reverse=True)
        u[0] = 1.0
        if u[-1] < 0.001:
            u[-1] = 0.001
        u = sanitize_u_schedule(tuple(u), N=1000, target_length=len(a.u_schedule))

        # segment policy: element-wise crossover (keeps innovation)
        seg_policies: List[SegmentPolicy] = []
        nseg = min(len(a.segment_policies), len(b.segment_policies))
        for i in range(nseg):
            seg_policies.append(
                crossover_segment_policy(
                    a.segment_policies[i],
                    b.segment_policies[i],
                    mask_len=self.mask_len,
                    total_stages=self.total_stages,
                    rng=rng,
                )
            )

        # if segment counts differ, append remaining from the longer one
        if len(a.segment_policies) > nseg:
            seg_policies.extend([a.segment_policies[i] for i in range(nseg, len(a.segment_policies))])
        elif len(b.segment_policies) > nseg:
            seg_policies.extend([b.segment_policies[i] for i in range(nseg, len(b.segment_policies))])

        sp = a.sampler_policy if random.random() < 0.5 else b.sampler_policy

        # boundaries: keep from a (consistent with your original)
        return Genotype(
            u_schedule=tuple(u),
            segment_policies=tuple(seg_policies),
            segment_boundaries=a.segment_boundaries,
            step_policies=tuple(),
            sampler_policy=sp,
            fitness=-1.0,
            metrics={}
        )

    # ---- evolve ----
    def evolve(self, evaluated_population: List[Genotype]) -> Tuple[bool, str]:
        logger.info("=== Evolving population (size=%d) ===", len(evaluated_population))

        if not evaluated_population:
            self.population = []
            self.top_k = []
            return False, ""

        # ---------- 1) Optional: apply std hard constraint ----------
        feasible_pool: List[Genotype] = []
        if self.std_max < float("inf"):
            for g in evaluated_population:
                m = g.metrics or {}
                s = float(m.get("std", 0.0))
                if s <= self.std_max:
                    feasible_pool.append(g)
            logger.info("Hard constraint (std_max=%s): %d/%d feasible", str(self.std_max), len(feasible_pool), len(evaluated_population))
        else:
            feasible_pool = evaluated_population

        pool = feasible_pool if len(feasible_pool) >= max(2, self.select_num) else evaluated_population
        if len(pool) < len(evaluated_population):
            logger.warning("Using fallback pool (%d candidates) to avoid search collapse", len(pool))

        # ---------- 2) Optional: apply std soft penalty to fid ----------
        if self.std_penalty_weight > 0.0:
            for g in pool:
                m = g.metrics or {}
                if "fid" in m:
                    s = float(m.get("std", 0.0))
                    m["fid"] = float(m["fid"]) + self.std_penalty_weight * s

        # ---------- 3) Select elites (NSGA-II on fid+latency) ----------
        elites = nsga2_select(pool, self.select_num, self.obj_keys)
        if not elites:
            logger.warning("All candidates have inf metrics, using fallback selection")
            elites = pool[: min(len(pool), self.select_num)]
        self.top_k = elites

        elite_fids = [float(g.metrics.get("fid", float("inf"))) for g in elites]
        elite_lats = [float(g.metrics.get("latency", float("inf"))) for g in elites]
        logger.info(
            "Elites selected (n=%d): FID [%.4f, %.4f], Latency [%.4fs, %.4fs]",
            len(elites),
            min(elite_fids) if elite_fids else float("inf"),
            max(elite_fids) if elite_fids else float("inf"),
            min(elite_lats) if elite_lats else float("inf"),
            max(elite_lats) if elite_lats else float("inf"),
        )

        # ---------- 4) Check early stopping ----------
        current_best_fid = min(elite_fids) if elite_fids else float("inf")
        should_stop, reason = self._check_early_stop_with_reason(current_best_fid)
        if should_stop:
            return True, reason

        # ---------- 5) Build next population with correct counts ----------
        next_pop: List[Genotype] = []
        next_pop.extend(elites)

        # Optional: local de-dup inside generation
        seen = set(self.genotype_key(g) for g in next_pop)

        # 5.1 Mutation
        mutation_target = min(self.population_num, len(next_pop) + self.mutation_num)
        tries = 0
        while len(next_pop) < mutation_target and tries < self.mutation_num * 50:
            tries += 1
            parent = random.choice(elites)
            child = self._mutate(parent)
            k = self.genotype_key(child)
            if k in seen:
                continue
            seen.add(k)
            next_pop.append(child)
        logger.info("Generated %d mutations", max(0, len(next_pop) - len(elites)))

        # 5.2 Crossover
        crossover_target = min(self.population_num, len(next_pop) + self.crossover_num)
        tries = 0
        before = len(next_pop)
        while len(next_pop) < crossover_target and tries < self.crossover_num * 50:
            tries += 1
            if len(elites) >= 2:
                pa, pb = random.sample(elites, k=2)
            else:
                pa = pb = elites[0]
            child = self._crossover(pa, pb)
            k = self.genotype_key(child)
            if k in seen:
                continue
            seen.add(k)
            next_pop.append(child)
        logger.info("Generated %d crossovers", len(next_pop) - before)

        # 5.3 Fill with random
        if self.policy_mode == "step":
            num_steps = len(elites[0].step_policies)
            num_segments = 0
        else:
            num_steps = len(elites[0].u_schedule)
            num_segments = len(elites[0].segment_policies)
        before = len(next_pop)
        tries = 0
        while len(next_pop) < self.population_num and tries < self.population_num * 200:
            tries += 1
            child = self._random_genotype(num_steps=num_steps, num_segments=num_segments)
            k = self.genotype_key(child)
            if k in seen:
                continue
            seen.add(k)
            next_pop.append(child)
        logger.info("Generated %d random individuals", len(next_pop) - before)

        self.population = next_pop[: self.population_num]
        logger.info("Next population size: %d", len(self.population))

        return False, ""

    # ---- utility methods ----
    def get_pareto_front(self, population: List[Genotype]) -> List[Genotype]:
        if not population:
            return []

        pop_infos = []
        for g in population:
            m = g.metrics or {}
            pop_infos.append({kk: float(m.get(kk, float("inf"))) for kk in self.obj_keys})

        fronts, _ = fast_non_dominated_sort(pop_infos, self.obj_keys)
        if not fronts:
            return []
        return [population[i] for i in fronts[0]]

    def log_search_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "epoch": self.epoch,
            "population_size": len(self.population),
            "top_k_size": len(self.top_k),
            "no_improve_epochs": self.no_improve_epochs,
            "best_fid_history": self.best_fid_history.copy(),
            "mask_len": self.mask_len,
            "full_mask_len": self.full_mask_len,
            "total_stages": self.total_stages,
        }

        if self.top_k:
            elite_fids = [float(g.metrics.get("fid", float("inf"))) for g in self.top_k]
            elite_lats = [float(g.metrics.get("latency", float("inf"))) for g in self.top_k]
            stats["best_fid"] = min(elite_fids) if elite_fids else float("inf")
            stats["best_latency"] = min(elite_lats) if elite_lats else float("inf")
            stats["elite_fid_range"] = [min(elite_fids), max(elite_fids)] if elite_fids else [float("inf"), float("inf")]
            stats["elite_latency_range"] = [min(elite_lats), max(elite_lats)] if elite_lats else [float("inf"), float("inf")]

        return stats

    def reset_search_state(self) -> None:
        self.best_fid_history = []
        self.no_improve_epochs = 0
        self.epoch = 0
        self.population = []
        self.top_k = []
        self.vis_dict = {}
        logger.info("Search state reset")
