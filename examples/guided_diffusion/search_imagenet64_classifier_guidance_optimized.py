# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time

import argparse
import os

from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_classifier,
    classifier_defaults,
)

print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


def resolve_sampler_name(args) -> str:
    sampler = str(getattr(args, "sampler", "") or "").strip().lower()
    aliases = {
        "ddim": "ddim",
        "acdms": "acdms",
        "dpm": "dpmpp",
        "dpmpp": "dpmpp",
        "dpm++": "dpmpp",
        "dpm_solver++": "dpmpp",
        "dpm-solver++": "dpmpp",
    }
    if sampler in aliases:
        return aliases[sampler]
    if bool(getattr(args, "use_dpmpp", False)):
        return "dpmpp"
    if bool(getattr(args, "use_acdms", False)):
        return "acdms"
    if bool(getattr(args, "use_ddim", False)):
        return "ddim"
    return "ddim"


# -------------------------
#  Multi-objective utilities: non-dominated sorting + crowding distance (NSGA-II)
#  All objectives are assumed to be minimized.
# -------------------------
def dominates(a, b, keys):
    not_worse_all = True
    strictly_better = False
    for k in keys:
        av = a[k]
        bv = b[k]
        if av > bv:
            not_worse_all = False
            break
        if av < bv:
            strictly_better = True
    return not_worse_all and strictly_better

def fast_non_dominated_sort(pop_infos, obj_keys):
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

def crowding_distance(front_indices, pop_infos, obj_keys):
    dist = {idx: 0.0 for idx in front_indices}
    if len(front_indices) <= 2:
        for idx in front_indices:
            dist[idx] = float('inf')
        return dist
    for k in obj_keys:
        front_sorted = sorted(front_indices, key=lambda i: pop_infos[i][k])
        dist[front_sorted[0]] = float('inf')
        dist[front_sorted[-1]] = float('inf')
        vmin = pop_infos[front_sorted[0]][k]
        vmax = pop_infos[front_sorted[-1]][k]
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        for j in range(1, len(front_sorted) - 1):
            prev_v = pop_infos[front_sorted[j - 1]][k]
            next_v = pop_infos[front_sorted[j + 1]][k]
            dist[front_sorted[j]] += (next_v - prev_v) / denom
    return dist

def nsga2_select(candidates, vis_dict, k, obj_keys):
    pop_infos = [vis_dict[c] for c in candidates]
    fronts, _ = fast_non_dominated_sort(pop_infos, obj_keys)
    selected = []
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


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]  # [10, 20]
    size_per = num_timesteps // len(section_counts) # 500  1000
    extra = num_timesteps % len(section_counts)  # 0
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)  # 500
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)  # (500 - 1) / (10 - 1) = 55.4
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2
        import warnings
        from scipy import linalg
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class EvolutionSearcher(object):

    def __init__(self, args, model, base_diffusion, time_step, classifier, search_space=None):
        self.args = args
        self.model = model
        self.base_diffusion = base_diffusion
        self.classifier = classifier
        import copy
        self.active_diffusion = copy.deepcopy(base_diffusion)
        self.time_step = time_step
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.max_fid = args.max_fid
        self.thres = args.thres
        
        # self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
        self.rf_features = []
        self.rf_lebal = []

        self.search_space = search_space

        from evaluations.evaluator_v1 import Evaluator_v1
        import tensorflow.compat.v1 as tf
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        self.evaluator = Evaluator_v1(tf.Session(config=config))

        import pickle
        f = open(args.ref_path,'rb')
        self.ref_stats = pickle.load(f)

        # Multi-objective keys (all minimize): fid, latency, stability (std over repeated evals)
        self.obj_keys = ["fid", "latency", "std"]

        # --- multi-objective evaluation knobs ---
        self.mo_eval_repeats = int(getattr(args, "mo_eval_repeats", 3))
        self.mo_fid_warmup_repeats = int(getattr(args, "mo_fid_warmup_repeats", 0))
        self.latency_include_fid_time = bool(getattr(args, "latency_include_fid_time", False))
        self.mo_seed_base = int(getattr(args, "mo_seed_base", 12345))

    def _split_timesteps_by_stage(self, timesteps):
        high, mid, low = [], [], []
        max_step = max(1, self.base_diffusion.original_num_steps - 1)
        high_thr = float(getattr(self.args, "high_noise_threshold", 0.7))
        mid_thr = float(getattr(self.args, "mid_noise_threshold", 0.25))
        if high_thr < mid_thr:
            high_thr, mid_thr = mid_thr, high_thr
        for t in sorted({int(x) for x in timesteps}, reverse=True):
            u = float(t) / float(max_step)
            if u >= high_thr:
                high.append(t)
            elif u >= mid_thr:
                mid.append(t)
            else:
                low.append(t)
        return {"high": high, "mid": mid, "low": low}

    def _target_stage_counts(self):
        n = int(self.time_step)
        if n <= 1:
            return {"high": 1, "mid": 0, "low": 0}
        if n == 2:
            return {"high": 1, "mid": 1, "low": 0}
        if n == 3:
            return {"high": 1, "mid": 2, "low": 0}

        high_ratio = float(getattr(self.args, "high_stage_ratio", 0.15))
        mid_ratio = float(getattr(self.args, "mid_stage_ratio", 0.7))
        low_ratio = float(getattr(self.args, "low_stage_ratio", 0.15))
        ratio_sum = high_ratio + mid_ratio + low_ratio
        if ratio_sum <= 0:
            high_ratio, mid_ratio, low_ratio = 0.15, 0.7, 0.15
            ratio_sum = 1.0
        high_ratio /= ratio_sum
        mid_ratio /= ratio_sum
        low_ratio /= ratio_sum

        high = max(1, int(round(n * high_ratio)))
        mid = max(2, int(round(n * mid_ratio)))
        low = max(1, int(round(n * low_ratio)))
        while high + mid + low > n:
            if low > 1:
                low -= 1
            elif mid > 2:
                mid -= 1
            else:
                high -= 1
        while high + mid + low < n:
            mid += 1
        return {"high": high, "mid": mid, "low": low}

    def _sample_stage_values(self, pool, count):
        if count <= 0 or len(pool) == 0:
            return []
        if count >= len(pool):
            return list(pool)

        chosen = []
        used = set()
        for i in range(count):
            pos = (i + 0.5) * len(pool) / float(count)
            center = int(round(pos - 0.5))
            lo = max(0, center - 1)
            hi = min(len(pool) - 1, center + 1)
            local = [idx for idx in range(lo, hi + 1) if idx not in used]
            if not local:
                local = [idx for idx in range(len(pool)) if idx not in used]
            pick_idx = random.choice(local)
            used.add(pick_idx)
            chosen.append(pool[pick_idx])
        return chosen

    def _sample_biased_timesteps(self, all_timesteps):
        pools = self._split_timesteps_by_stage(all_timesteps)
        target = self._target_stage_counts()
        counts = {k: min(target[k], len(pools[k])) for k in target}
        shortfall = self.time_step - sum(counts.values())
        refill_order = tuple(str(getattr(self.args, "stage_refill_order", "mid,low,high")).split(","))
        refill_order = tuple(s.strip() for s in refill_order if s.strip() in ("high", "mid", "low"))
        if len(refill_order) == 0:
            refill_order = ("mid", "low", "high")
        for stage in refill_order:
            if shortfall <= 0:
                break
            cap = len(pools[stage]) - counts[stage]
            if cap <= 0:
                continue
            take = min(shortfall, cap)
            counts[stage] += take
            shortfall -= take

        selected = []
        for stage in ("high", "mid", "low"):
            selected.extend(self._sample_stage_values(pools[stage], counts[stage]))

        if len(selected) < self.time_step:
            remaining = [t for t in sorted({int(x) for x in all_timesteps}, reverse=True) if t not in selected]
            selected.extend(self._sample_stage_values(remaining, self.time_step - len(selected)))

        return sorted(selected, reverse=True)[:self.time_step]

    def _stage_name(self, timestep):
        max_step = max(1, self.base_diffusion.original_num_steps - 1)
        high_thr = float(getattr(self.args, "high_noise_threshold", 0.7))
        mid_thr = float(getattr(self.args, "mid_noise_threshold", 0.25))
        if high_thr < mid_thr:
            high_thr, mid_thr = mid_thr, high_thr
        u = float(int(timestep)) / float(max_step)
        if u >= high_thr:
            return "high"
        if u >= mid_thr:
            return "mid"
        return "low"

    def _mutate_candidate_stage_aware(self, cand, m_prob):
        cand = list(cand)
        if self.search_space is not None:
            all_index = self.search_space
        else:
            all_index = [j for j in range(self.base_diffusion.original_num_steps)]

        pools = self._split_timesteps_by_stage(all_index)
        used = set(cand)
        for i in range(len(cand)):
            if np.random.random_sample() >= m_prob:
                continue
            stage = self._stage_name(cand[i])
            stage_candidates = [t for t in pools[stage] if t not in used]
            fallback_candidates = [t for t in all_index if t not in used]
            candidates = stage_candidates if len(stage_candidates) > 0 else fallback_candidates
            if len(candidates) == 0:
                continue
            old_t = cand[i]
            new_t = random.choice(candidates)
            used.remove(old_t)
            cand[i] = new_t
            used.add(new_t)

        return sorted(cand, reverse=True)

    # -------------------------
    # RNG save/restore: avoid evaluation polluting EA randomness
    # -------------------------
    def _capture_rng_state(self):
        state = {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        random.setstate(state["py"])
        np.random.set_state(state["np"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def reset_diffusion(self, use_timesteps):
        use_timesteps = set(use_timesteps)
        self.active_diffusion.timestep_map = []
        last_alpha_cumprod = 1.0
        new_betas = []

        self.active_diffusion.use_timesteps = set(use_timesteps)

        for i, alpha_cumprod in enumerate(self.base_diffusion.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod) # 通过长序列的 \overline{alpha} 解 短序列的 \beta
                last_alpha_cumprod = alpha_cumprod
                self.active_diffusion.timestep_map.append(i)

        import numpy as np
        new_betas = np.array(new_betas, dtype=np.float64)

        self.active_diffusion.betas = new_betas
        assert len(new_betas.shape) == 1, "betas must be 1-D"
        assert (new_betas > 0).all() and (new_betas <= 1).all()

        self.active_diffusion.num_timesteps = int(new_betas.shape[0])

        alphas = 1.0 - new_betas  # alpha 递减
        self.active_diffusion.alphas_cumprod = np.cumprod(alphas, axis=0) # overliane_{x}
        self.active_diffusion.alphas_cumprod_prev = np.append(1.0, self.active_diffusion.alphas_cumprod[:-1]) # alpha[0], alpha[0], alpha[1], ...., alpha[T-1]
        self.active_diffusion.alphas_cumprod_next = np.append(self.active_diffusion.alphas_cumprod[1:], 0.0)  # alpha[1], alpha[2], ..., alpha[T], alpha[T]
        assert self.active_diffusion.alphas_cumprod_prev.shape == (self.active_diffusion.num_timesteps,)  

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.active_diffusion.sqrt_alphas_cumprod = np.sqrt(self.active_diffusion.alphas_cumprod)  # \sqrt{\overline{\alpha}}
        self.active_diffusion.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.active_diffusion.alphas_cumprod) # \sqrt{1 - \overline{\alpha}}
        self.active_diffusion.log_one_minus_alphas_cumprod = np.log(1.0 - self.active_diffusion.alphas_cumprod)  # \log{1 - \overline{\alpha}}
        self.active_diffusion.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.active_diffusion.alphas_cumprod)      # \frac{1}{\sqrt{\overline{\alpha}}}
        self.active_diffusion.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.active_diffusion.alphas_cumprod - 1)   

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.active_diffusion.posterior_variance = (
            new_betas * (1.0 - self.active_diffusion.alphas_cumprod_prev) / (1.0 - self.active_diffusion.alphas_cumprod)  # DDPM 式7 的 \hat{\beta}
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        if len(self.active_diffusion.posterior_variance) > 1:
            self.active_diffusion.posterior_log_variance_clipped = np.log(
                np.append(self.active_diffusion.posterior_variance[1], self.active_diffusion.posterior_variance[1:])
            )
        else:
            self.active_diffusion.posterior_log_variance_clipped = self.active_diffusion.posterior_variance
        self.active_diffusion.posterior_mean_coef1 = (
            new_betas * np.sqrt(self.active_diffusion.alphas_cumprod_prev) / (1.0 - self.active_diffusion.alphas_cumprod)
        )
        self.active_diffusion.posterior_mean_coef2 = (
            (1.0 - self.active_diffusion.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.active_diffusion.alphas_cumprod)
        )

    def update_top_k(self, candidates, *, k):
        assert k in self.keep_top_k
        logger.log('select (multi-objective NSGA-II) ......')
        t = self.keep_top_k[k]
        t += candidates
        # de-duplicate while preserving order
        t = list(dict.fromkeys(t))
        # must be evaluated
        t = [c for c in t if (c in self.vis_dict and self.vis_dict[c].get("visited", False))]
        if len(t) == 0:
            self.keep_top_k[k] = []
            return
        selected = nsga2_select(t, self.vis_dict, k, self.obj_keys)
        self.keep_top_k[k] = selected

    def sample_active_subnet(self):
        if self.search_space is not None:
            use_timestep = self.search_space
        else:
            original_num_steps = self.base_diffusion.original_num_steps
            use_timestep = [i for i in range(original_num_steps)]

        return self._sample_biased_timesteps(use_timestep)
    
    def is_legal_before_search(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logger.log('cand: {} has visited!'.format(cand))
            return False

        # 确保时间步是单调递减的（从高到低）
        cand_timesteps = eval(cand)
        if not all(cand_timesteps[i] > cand_timesteps[i+1] for i in range(len(cand_timesteps)-1)):
            cand_timesteps = sorted(cand_timesteps, reverse=True)
            cand = str(cand_timesteps)
            logger.log(f'Resorted timesteps to ensure monotonic decreasing: {cand}')

        fid_mean, latency_mean, fid_std = self.get_cand_metrics(cand=cand_timesteps, args=self.args)

        info['fid'] = float(fid_mean)
        info['latency'] = float(latency_mean)
        info['std'] = float(fid_std)
        info['visited'] = True

        logger.log('cand: {} fid(mean): {:.6f} latency(mean): {:.4f}s fid_std: {:.6f}'.format(
            cand, info['fid'], info['latency'], info['std']))
        return True

    def is_legal(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logger.log('cand: {} has visited!'.format(cand))
            return False

        # 确保时间步是单调递减的（从高到低）
        cand_timesteps = eval(cand)
        if not all(cand_timesteps[i] > cand_timesteps[i+1] for i in range(len(cand_timesteps)-1)):
            cand_timesteps = sorted(cand_timesteps, reverse=True)
            cand = str(cand_timesteps)
            logger.log(f'Resorted timesteps to ensure monotonic decreasing: {cand}')

        fid_mean, latency_mean, fid_std = self.get_cand_metrics(cand=cand_timesteps, args=self.args)

        info['fid'] = float(fid_mean)
        info['latency'] = float(latency_mean)
        info['std'] = float(fid_std)
        info['visited'] = True

        logger.log('cand: {} fid(mean): {:.6f} latency(mean): {:.4f}s fid_std: {:.6f}'.format(
            cand, info['fid'], info['latency'], info['std']))
        return True

    def get_cand_metrics(self, cand=None, args=None, device='cuda'):
        """Return fid_mean, latency_mean, fid_std by repeating evaluation.
        latency is computed from sample_time (optionally + fid_time) and averaged across repeats.
        """
        saved_state = self._capture_rng_state()
        try:
            # optional warmup to reduce first-run latency bias
            for w in range(self.mo_fid_warmup_repeats):
                _ = self.get_cand_fid(
                    cand=cand, args=args, override_seed=self.mo_seed_base + 100000 + w,
                    return_timing=False
                )

            fids = []
            lats = []
            for r in range(self.mo_eval_repeats):
                seed = self.mo_seed_base + r
                fid, sample_time, fid_time = self.get_cand_fid(
                    cand=cand, args=args, override_seed=seed, return_timing=True
                )
                fids.append(float(fid))
                if self.latency_include_fid_time:
                    lats.append(float(sample_time + fid_time))
                else:
                    lats.append(float(sample_time))

            fid_mean = float(np.mean(fids))
            fid_std = float(np.std(fids))
            latency_mean = float(np.mean(lats))
            return fid_mean, latency_mean, fid_std
        finally:
            self._restore_rng_state(saved_state)

    def get_cand_fid(self, cand=None, args=None, override_seed=None, return_timing=False):
        """Evaluate a candidate timestep subset.
        If return_timing=True: returns (fid, sample_time, fid_time).
        """
        if override_seed is not None:
            override_seed = int(override_seed)
            random.seed(override_seed)
            np.random.seed(override_seed % (2**32 - 1))
            torch.manual_seed(override_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(override_seed)

        # 确保时间步是单调递减的（从高到低）
        if not all(cand[i] > cand[i+1] for i in range(len(cand)-1)):
            cand = sorted(cand, reverse=True)
            logger.log(f'Resorted timesteps in get_cand_fid to ensure monotonic decreasing: {cand}')

        # active model
        t1 = time.time()
        self.reset_diffusion(cand)
        reset_time = time.time() - t1

        self.model.eval()
        self.classifier.eval()

        # sample image (classifier guidance)
        def cond_fn(x, t, y=None):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

        def model_fn(x, t, y=None):
            assert y is not None
            return self.model(x, t, y if args.class_cond else None)

        logger.log("sampling...")
        all_images = []
        all_labels = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_sample_start = time.time()

        sampler_name = resolve_sampler_name(args)
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

            if sampler_name == "acdms":
                from guided_diffusion.adm_acdms import sample_with_adm_acdms

                teacher_defaults = {
                    'noise_schedule': args.noise_schedule,
                    'diffusion_steps': args.diffusion_steps,
                }
                class_labels = model_kwargs.get('y', None)
                sample = sample_with_adm_acdms(
                    student=self.model,
                    teacher_defaults=teacher_defaults,
                    tau_list=cand,
                    g_star_obj=None,
                    shape=(args.batch_size, 3, args.image_size, args.image_size),
                    device=dist_util.dev(),
                    class_labels=class_labels,
                    clip_denoised=args.clip_denoised,
                    fm_strength=args.acdms_fm_strength,
                    fm_t_start=args.acdms_fm_t_start,
                    cond_fn=cond_fn,
                )
            elif sampler_name == "dpmpp":
                from guided_diffusion.adm_dpmpp import sample_with_adm_dpmpp

                teacher_defaults = {
                    'noise_schedule': args.noise_schedule,
                    'diffusion_steps': args.diffusion_steps,
                }
                class_labels = model_kwargs.get('y', None)
                fm_strength = args.dpmpp_fm_strength if bool(args.dpmpp_use_corrector) else 0.0
                sample = sample_with_adm_dpmpp(
                    student=self.model,
                    teacher_defaults=teacher_defaults,
                    tau_list=cand,
                    g_star_obj=None,
                    shape=(args.batch_size, 3, args.image_size, args.image_size),
                    device=dist_util.dev(),
                    class_labels=class_labels,
                    clip_denoised=args.clip_denoised,
                    fm_strength=fm_strength,
                    fm_t_start=args.dpmpp_fm_t_start,
                    fm_lambda_clip=args.dpmpp_fm_lambda_clip,
                    fm_step_clip=args.dpmpp_fm_step_clip,
                    step_ramp_start=args.dpmpp_step_ramp_start,
                    low_noise_threshold_index=args.dpmpp_low_noise_threshold_index,
                    inner_steps_low_noise=args.dpmpp_inner_steps_low_noise,
                    cond_fn=cond_fn,
                )
            else:
                sample_fn = (
                    self.active_diffusion.ddim_sample_loop if sampler_name == "ddim" else self.active_diffusion.p_sample_loop
                )
                sample = sample_fn(
                    model_fn,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=dist_util.dev(),
                )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log('created ' + str(len(all_images) * args.batch_size) + ' samples')

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        dist.barrier()
        logger.log("sampling complete")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sample_time = time.time() - t_sample_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fid_start = time.time()

        from evaluations.evaluator_v1 import cal_fid
        fid = cal_fid(arr, args.image_size, self.evaluator, ref_stats=self.ref_stats)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fid_time = time.time() - t_fid_start

        logger.log('reset_time: ' + str(reset_time) + ', sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))

        if return_timing:
            return fid, sample_time, fid_time
        return fid



    def dump_pareto_front(self, outpath):
        """Dump current evaluated Pareto front (rank=0) to txt."""
        evaluated = [c for c, info in self.vis_dict.items() if info.get("visited", False)]
        if len(evaluated) == 0:
            return
        pop_infos = [self.vis_dict[c] for c in evaluated]
        fronts, _ = fast_non_dominated_sort(pop_infos, self.obj_keys)
        if len(fronts) == 0:
            return
        pareto = [evaluated[i] for i in fronts[0]]
        pareto_sorted = sorted(pareto, key=lambda c: self.vis_dict[c].get("fid", float("inf")))
        os.makedirs(outpath, exist_ok=True)
        latest_path = os.path.join(outpath, "pareto_front_latest.txt")
        all_path = os.path.join(outpath, "pareto_front_all.txt")
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(f"epoch={self.epoch}\n")
            for c in pareto_sorted:
                info = self.vis_dict[c]
                f.write(f"{c}\tfid={info.get('fid', float('inf')):.6f}\tlat={info.get('latency', float('inf')):.4f}\tstd={info.get('std', float('inf')):.6f}\n")
        with open(all_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== epoch={self.epoch} ===\n")
            for c in pareto_sorted:
                info = self.vis_dict[c]
                f.write(f"{c}\tfid={info.get('fid', float('inf')):.6f}\tlat={info.get('latency', float('inf')):.4f}\tstd={info.get('std', float('inf')):.6f}\n")
        logger.log(f"[MO] Pareto latest -> {latest_path}; all -> {all_path} (pareto N={len(pareto_sorted)})")

    def get_random_before_search(self, num):
        logger.log('random select ........')
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logger.log('random {}/{}'.format(len(self.candidates), num))
        logger.log('random_num = {}'.format(len(self.candidates)))
    
    def get_random(self, num):
        logger.log('random select ........')
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.log('random {}/{}'.format(len(self.candidates), num))
        logger.log('random_num = {}'.format(len(self.candidates)))

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logger.log('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.log('cross {}/{}'.format(len(res), cross_num))

        logger.log('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logger.log('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)
            return self._mutate_candidate_stage_aware(cand, m_prob)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.log('mutation {}/{}'.format(len(res), mutation_num))

        logger.log('mutation_num = {}'.format(len(res)))
        return res
    
    def mutate_init_x(self, x0, mutation_num, m_prob):
        logger.log('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)
            return self._mutate_candidate_stage_aware(cand, m_prob)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logger.log('mutation x0 {}/{}'.format(len(res), mutation_num))

        logger.log('mutation_num = {}'.format(len(res)))
        return res

    def search(self):
        logger.log('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        if args.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)

        else:
            steps = self.base_diffusion.original_num_steps
            if resolve_sampler_name(args) == 'ddim':
                timestep_respacing = 'ddim'
            else:
                timestep_respacing = ''
            timestep_respacing += str(args.time_step)
            init_x = space_timesteps(steps, timestep_respacing)
            init_x = list(init_x)
            # 确保时间步是单调递减的（从高到低）
            init_x = sorted(init_x, reverse=True)
            self.is_legal_before_search(str(init_x))
            self.candidates.append(str(init_x))
            self.get_random_before_search(self.population_num // 2 + 1)
            res = self.mutate_init_x(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            self.candidates += res
        
        # self.rf_features = [eval(self.candidates[j]) for j in range(len(self.candidates))]
        # rf_features = np.asarray(self.rf_features, dtype='float')

        def dec(fid):
            return fid < self.max_fid

        # try:    
        #     self.rf_lebal = [dec(self.vis_dict[self.candidates[j]]['fid']) for j in range(len(self.candidates))]
        #     self.RandomForestClassifier.fit(rf_features, self.rf_lebal)
        # except:
        #     import pdb
        #     pdb.set_trace()

        while self.epoch < self.max_epochs:
            logger.log('epoch = {}'.format(self.epoch))

            self.update_top_k(self.candidates, k=self.select_num)
            self.update_top_k(self.candidates, k=50)

            logger.log('epoch = {} : top {} (multi-objective) result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                info = self.vis_dict[cand]
                logger.log('No.{} {} fid={:.6f} lat={:.4f}s std={:.6f}'.format(
                    i + 1, cand, info.get('fid', float('inf')), info.get('latency', float('inf')), info.get('std', float('inf'))))
            
            # dump pareto each epoch
            outdir = getattr(self.args, "save_dir", "")
            if outdir:
                self.dump_pareto_front(outdir)

            # sys.exit()
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            # rf_features = np.asarray(self.rf_features, dtype='float')
            # self.RandomForestClassifier.fit(rf_features, self.rf_lebal) # refit

            self.epoch += 1

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5000,
        batch_size=2,
        sampler="ddim",
        use_ddim=False,
        use_acdms=False,
        use_dpmpp=False,
        model_path="",
        save_dir="",
        time_step=100,
        # search config
        seed = 0,
        deterministic = False,
        local_rank = 0,
        max_epochs = 20,
        select_num = 10,
        population_num = 50,
        m_prob = 0.1,
        crossover_num = 25,
        mutation_num = 35,
        classifier_path="",
        classifier_scale=1.0,
        max_fid=48.0,
        thres=0.2,
        use_ddim_init_x=False,
        search_space='',
        ref_path='',
        MASTER_PORT='12344',
        # multi-objective evaluation
        mo_eval_repeats=3,
        mo_fid_warmup_repeats=0,
        latency_include_fid_time=False,
        mo_seed_base=12345,
        # stage-aware timestep bias
        high_noise_threshold=0.7,
        mid_noise_threshold=0.25,
        high_stage_ratio=0.15,
        mid_stage_ratio=0.7,
        low_stage_ratio=0.15,
        stage_refill_order="mid,low,high",
        # ACDMS parameters
        acdms_fm_strength=0.04,
        acdms_fm_t_start=0.85,
        dpmpp_use_corrector=False,
        dpmpp_fm_strength=0.04,
        dpmpp_fm_t_start=0.85,
        dpmpp_fm_lambda_clip=1.0,
        dpmpp_fm_step_clip=0.02,
        dpmpp_step_ramp_start=0.60,
        dpmpp_low_noise_threshold_index=150,
        dpmpp_inner_steps_low_noise=1,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == '__main__':

    args = create_argparser().parse_args()

    sampler_name = resolve_sampler_name(args)
    args.use_ddim = sampler_name == "ddim"
    args.use_acdms = sampler_name == "acdms"
    args.use_dpmpp = sampler_name == "dpmpp"

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.MASTER_PORT

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist_util.setup_dist()
    logger.configure(args.save_dir)

    logger.log(str(args))
    logger.log(f"[Sampler] active sampler = {sampler_name}")
    logger.log(
        "[TimestepBias] high_thr={:.3f} mid_thr={:.3f} ratios(high/mid/low)=({:.3f}/{:.3f}/{:.3f}) refill_order={}".format(
            float(args.high_noise_threshold),
            float(args.mid_noise_threshold),
            float(args.high_stage_ratio),
            float(args.mid_stage_ratio),
            float(args.low_stage_ratio),
            str(args.stage_refill_order),
        )
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    classifier.eval()

    if args.search_space != '':
        search_space_core = sorted(eval(args.search_space))
        if args.use_ddim_init_x:
            steps = diffusion.original_num_steps
            if resolve_sampler_name(args) == 'ddim':
                timestep_respacing = 'ddim'
            else:
                timestep_respacing = ''
            timestep_respacing += str(args.time_step)
            init_x = space_timesteps(steps, timestep_respacing)
            init_x = list(init_x)
            # 确保时间步是单调递减的（从高到低）
            init_x = sorted(init_x, reverse=True)
            search_space_core += init_x

        R = int(diffusion.original_num_steps / 100)
        search_space = []

        for s in search_space_core:
            left = max(s - R, 0)
            right = min(s + R, diffusion.original_num_steps)
            search_space += [i for i in range(left, right)]
        
        search_space = sorted(list(set(search_space)))
        logger.log('search space: ' + str(search_space))
    else:
        search_space = None
    
    ## build EA 
    t = time.time()
    searcher = EvolutionSearcher(args, model=model, base_diffusion=diffusion, time_step=args.time_step, classifier=classifier, search_space=search_space)
    searcher.search()
    logger.log('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))
