import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time

from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

from torch import autocast
from contextlib import contextmanager, nullcontext
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from scipy import linalg
from sklearn.ensemble import RandomForestClassifier

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.acdms import ACDMSSampler
from ldm.data.build_dataloader import build_dataloader

from transformers import AutoFeatureExtractor
import logging
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

# 修改导入路径，使用项目中的pytorch_fid模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))
from pytorch_fid.inception import InceptionV3
import copy

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def get_activations(data, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    model.eval()

    if batch_size > data.shape[0]:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = data.shape[0]

    pred_arr = np.empty((data.shape[0], dims))
    start_idx = 0

    for i in range(0, data.shape[0], batch_size):
        if i + batch_size > data.shape[0]:
            batch = data[i:, :, :, :]
        else:
            batch = data[i:i + batch_size, :, :, :]
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(datas, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    act = get_activations(datas, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(data1, ref_mu, ref_sigma, batch_size, device, dims, num_workers=1, use_fp16=True):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    # ✅ FP16优化：在CUDA环境下使用FP16加速
    if use_fp16 and device == 'cuda':
        model = model.half()

    try:
        if isinstance(data1, torch.Tensor):
            if data1.numel() == 0 or (data1.dim() > 0 and data1.shape[0] < 2):
                logging.warning("calculate_fid: insufficient samples (N=%d), returning inf FID",
                                data1.shape[0] if data1.dim() > 0 else 0)
                return float('inf')
            # ✅ FP16优化：根据use_fp16参数选择精度
            dtype = torch.float16 if use_fp16 and device == 'cuda' else torch.float32
            data1 = torch.nan_to_num(
                data1.to(dtype=dtype),
                nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
        else:
            N = data1.shape[0]
            if N < 2:
                logging.warning("calculate_fid: insufficient samples (N=%d), returning inf FID", N)
                return float('inf')
    except Exception:
        logging.warning("calculate_fid: invalid input data, returning inf FID")
        return float('inf')

    m1, s1 = calculate_activation_statistics(data1, model, batch_size, dims, device, num_workers)

    if np.any(np.isnan(m1)) or np.any(np.isinf(m1)):
        logging.warning("calculate_fid: m1 contains NaN or Inf values! Returning inf FID.")
        return float('inf')
    if np.any(np.isnan(s1)) or np.any(np.isinf(s1)):
        logging.warning("calculate_fid: s1 contains NaN or Inf values! Returning inf FID.")
        return float('inf')

    fid_value = calculate_frechet_distance(m1, s1, ref_mu, ref_sigma)
    return fid_value


# -------------------------
#  多目标工具：非支配排序 + 拥挤距离
# -------------------------

def dominates(a, b, keys):
    """
    a dominates b if:
      for all k: a[k] <= b[k] and exists k: a[k] < b[k]
    keys: list of objective keys, all are "minimize"
    """
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
    """
    pop_infos: list of dict with objectives
    return: fronts as list[list[int]] indices
    """
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

    fronts.pop()  # remove last empty
    return fronts, rank


def crowding_distance(front_indices, pop_infos, obj_keys):
    """
    return: dict idx -> crowding distance
    """
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
    """
    candidates: list[str] candidate keys in vis_dict
    vis_dict[cand] contains objectives
    Return: selected list[str] length <= k
    """
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


# -------------------------
#  进化搜索器：多目标版本（最小侵入式改造）
# -------------------------

class EvolutionSearcher(object):

    def __init__(self, opt, model, time_step, ref_mu, ref_sigma, sampler, dataloader_info, batch_size, dpm_params=None):
        self.opt = opt
        self.model = model
        self.sampler = sampler
        self.time_step = time_step
        self.dataloader_info = dataloader_info
        self.batch_size = batch_size

        # EA hyperparameters
        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num
        self.population_num = opt.population_num
        self.m_prob = opt.m_prob
        self.crossover_num = opt.crossover_num
        self.mutation_num = opt.mutation_num
        self.num_samples = opt.num_sample
        self.ddim_discretize = "uniform"

        # multi-objective keys (all minimize)
        self.obj_keys = ["fid", "latency", "std"]

        # tracking
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.max_fid = opt.max_fid

        self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
        self.rf_features = []
        self.rf_lebal = []

        self.use_ddim_init_x = opt.use_ddim_init_x

        self.ref_mu = np.load(ref_mu)
        self.ref_sigma = np.load(ref_sigma)

        self.dpm_params = dpm_params

        # --- MO knobs ---
        self.mo_eval_repeats = int(getattr(opt, "mo_eval_repeats", 3))
        self.mo_fid_warmup_repeats = int(getattr(opt, "mo_fid_warmup_repeats", 0))
        self.latency_include_fid_time = bool(getattr(opt, "latency_include_fid_time", False))
        self.mo_seed_base = int(getattr(opt, "mo_seed_base", 12345))

    # -------------------------
    #  RNG state save/restore：避免评估污染EA随机轨迹
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

    # -------------------------
    #  多目标选择：替换原 update_top_k 的排序方式
    # -------------------------
    def update_top_k(self, candidates, *, k):
        assert k in self.keep_top_k
        logging.info('select (multi-objective NSGA-II) ......')
        t = self.keep_top_k[k]
        t += candidates

        # 去重
        t = list(dict.fromkeys(t))

        # 要求候选必须已经评估过
        t = [c for c in t if (c in self.vis_dict and self.vis_dict[c].get("visited", False))]

        if len(t) == 0:
            self.keep_top_k[k] = []
            return

        selected = nsga2_select(t, self.vis_dict, k, self.obj_keys)
        self.keep_top_k[k] = selected

    # -------------------------
    #  候选合法性：同原版，但评估输出增加 latency/std
    # -------------------------
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False

        fid_mean, latency_mean, fid_std = self.get_cand_metrics(opt=self.opt, cand=eval(cand))

        info['fid'] = float(fid_mean)
        info['latency'] = float(latency_mean)
        info['std'] = float(fid_std)
        info['visited'] = True

        logging.info('cand: {}, fid(mean): {:.6f}, latency(mean): {:.4f}s, fid_std: {:.6f}'.format(
            cand, info['fid'], info['latency'], info['std']))
        return True

    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False

        fid_mean, latency_mean, fid_std = self.get_cand_metrics(opt=self.opt, cand=eval(cand))

        info['fid'] = float(fid_mean)
        info['latency'] = float(latency_mean)
        info['std'] = float(fid_std)
        info['visited'] = True

        logging.info('cand: {}, fid(mean): {:.6f}, latency(mean): {:.4f}s, fid_std: {:.6f}'.format(
            cand, info['fid'], info['latency'], info['std']))
        return True

    # -------------------------
    #  生成候选：保留原逻辑
    # -------------------------
    def get_random_before_search(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))

    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1_l = eval(cand1)
            cand2_l = eval(cand2)

            for i in range(len(cand1_l)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1_l[i])
                else:
                    new_cand.append(cand2_l[i])
            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del candidates[new_index]
                    cand[i] = new_c
                    if len(candidates) == 0:
                        break
            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def get_mutation_dpm(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del candidates[new_index]
                    cand[i] = new_c
                    if len(candidates) == 0:
                        break
            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = eval(x0)

            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del candidates[new_index]
                    cand[i] = new_c
                    if len(candidates) == 0:
                        break
            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = eval(x0)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del candidates[new_index]
                    cand[i] = new_c
                    if len(candidates) == 0:
                        break
            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def sample_active_subnet(self):
        original_num_steps = self.sampler.ddpm_num_timesteps
        use_timestep = [i for i in range(original_num_steps)]
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step]
        return use_timestep

    def sample_active_subnet_dpm(self):
        use_timestep = copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        return use_timestep

    # -------------------------
    #  多目标评估 wrapper
    #  - 重复 R 次得到 fid_std
    #  - latency 复用 sample_time（可选加 fid_time）
    #  - 评估前后保存/恢复 RNG state，避免污染 EA
    # -------------------------
    def get_cand_metrics(self, cand=None, opt=None, device='cuda'):
        """
        return: fid_mean, latency_mean, fid_std
        """
        # 评估前捕获EA随机状态
        saved_state = self._capture_rng_state()

        try:
            # warmup (可选)：避免首轮 cuda kernel 编译导致 latency 偏大
            for w in range(self.mo_fid_warmup_repeats):
                _ = self.get_cand_fid(
                    cand=cand, opt=opt, device=device,
                    override_seed=self.mo_seed_base + 100000 + w,
                    return_timing=False
                )

            fids = []
            lats = []
            for r in range(self.mo_eval_repeats):
                seed = self.mo_seed_base + r
                fid, sample_time, fid_time = self.get_cand_fid(
                    cand=cand, opt=opt, device=device,
                    override_seed=seed, return_timing=True
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
            # 无论评估成功与否，都恢复EA随机状态
            self._restore_rng_state(saved_state)

    # -------------------------
    #  原版 get_cand_fid：最小修改
    #  - 允许 override_seed
    #  - 可选 return_timing
    #  - 添加 cuda synchronize 让计时可信
    # -------------------------
    def get_cand_fid(self, cand=None, opt=None, device='cuda', override_seed=None, return_timing=False):
        start_code = None

        if override_seed is not None:
            random.seed(int(override_seed))
            np.random.seed(int(override_seed) % (2**32 - 1))
            torch.manual_seed(int(override_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(override_seed))

        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_sample_start = time.time()

                    all_samples = list()

                    for itr, batch in enumerate(self.dataloader_info['validation_loader']):
                        prompts = batch['text']
                        uc = None
                        if opt.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        sampled_timestep = np.array(cand)

                        samples_ddim, _ = self.sampler.sample(
                            S=opt.time_step,
                            conditioning=c,
                            batch_size=opt.n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code,
                            sampled_timestep=sampled_timestep
                        )
                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        for x_sample in x_checked_image_torch:
                            all_samples.append(x_sample.cpu().numpy())

                        logging.info('samples: ' + str(len(all_samples)))

                        if len(all_samples) > self.num_samples:
                            logging.info('samples: ' + str(len(all_samples)))
                            break

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    sample_time = time.time() - t_sample_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fid_start = time.time()

        all_samples = np.array(all_samples)
        all_samples = torch.Tensor(all_samples)

        fid = calculate_fid(
            data1=all_samples,
            ref_mu=self.ref_mu,
            ref_sigma=self.ref_sigma,
            batch_size=320,
            dims=2048,
            device='cuda'
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fid_time = time.time() - t_fid_start

        logging.info('FID: ' + str(fid))
        logging.info('sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))

        if return_timing:
            return fid, sample_time, fid_time
        return fid

    def dump_pareto_front(self, outpath):
        """
        输出当前已评估候选的 Pareto front（rank=0）到文件
        """
        evaluated = [c for c, info in self.vis_dict.items() if info.get("visited", False)]
        if len(evaluated) == 0:
            return

        pop_infos = [self.vis_dict[c] for c in evaluated]
        fronts, _ = fast_non_dominated_sort(pop_infos, self.obj_keys)
        if len(fronts) == 0:
            return

        pareto = [evaluated[i] for i in fronts[0]]
        pareto_sorted = sorted(pareto, key=lambda c: self.vis_dict[c]["fid"])

        latest_path = os.path.join(outpath, "pareto_front_latest.txt")
        all_path = os.path.join(outpath, "pareto_front_all.txt")

        # 1) 覆盖写 latest
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(f"epoch={self.epoch}\n")
            for c in pareto_sorted:
                info = self.vis_dict[c]
                f.write(f"{c}\tfid={info['fid']:.6f}\tlat={info['latency']:.4f}\tstd={info['std']:.6f}\n")

        # 2) 追加写 all（方便 grep/搜索）
        with open(all_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== epoch={self.epoch} ===\n")
            for c in pareto_sorted:
                info = self.vis_dict[c]
                f.write(f"{c}\tfid={info['fid']:.6f}\tlat={info['latency']:.4f}\tstd={info['std']:.6f}\n")

        logging.info(f"[MO] Pareto latest -> {latest_path}; all -> {all_path} (pareto N={len(pareto_sorted)})")

    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num,
            self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        if self.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)
        else:
            if self.opt.dpm_solver:
                init_x = self.dpm_params['init_timesteps']
            else:
                init_x = make_ddim_timesteps(
                    ddim_discr_method=self.ddim_discretize,
                    num_ddim_timesteps=self.time_step,
                    num_ddpm_timesteps=self.sampler.ddpm_num_timesteps,
                    verbose=False
                )
            init_x = sorted(list(init_x))
            self.is_legal_before_search(str(init_x))
            self.candidates.append(str(init_x))
            self.get_random_before_search(self.population_num // 2)

            if self.opt.dpm_solver:
                res = self.mutate_init_x_dpm(
                    x0=str(init_x),
                    mutation_num=self.population_num - self.population_num // 2 - 1,
                    m_prob=0.1
                )
            else:
                res = self.mutate_init_x(
                    x0=str(init_x),
                    mutation_num=self.population_num - self.population_num // 2 - 1,
                    m_prob=0.1
                )
            self.candidates += res

        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))

            # 多目标 top-k 更新（NSGA-II）
            self.update_top_k(self.candidates, k=self.select_num)
            self.update_top_k(self.candidates, k=50)

            logging.info('epoch = {} : top {} (multi-objective) result'.format(self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                info = self.vis_dict[cand]
                logging.info('No.{} {} fid={:.6f} lat={:.4f}s std={:.6f}'.format(
                    i + 1, cand, info['fid'], info['latency'], info['std']))

            # 每代输出 Pareto front
            outdir = getattr(self.opt, "outdir", "outputs/txt2img-samples")
            self.dump_pareto_front(outdir)

            if self.epoch + 1 == self.max_epochs:
                break

            if self.opt.dpm_solver:
                mutation = self.get_mutation_dpm(self.select_num, self.mutation_num, self.m_prob)
            else:
                mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num)

            self.epoch += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar")
    parser.add_argument("--outdir", type=str, nargs="?", default="outputs/txt2img-samples")
    parser.add_argument("--skip_grid", action='store_true')
    parser.add_argument("--skip_save", action='store_true')
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--plms", action='store_true')
    parser.add_argument("--dpm_solver", action='store_true')
    parser.add_argument("--dpm_solver_steps", type=int, default=50, help="DPM-Solver++ number of steps")
    parser.add_argument("--dpm_solver_order", type=int, default=2, help="DPM-Solver++ order (1 or 2)")
    parser.add_argument("--laion400m", action='store_true')
    parser.add_argument("--fixed_code", action='store_true')
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--acdms", action='store_true')
    parser.add_argument("--n_iter", type=int, default=2)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--n_rows", type=int, default=0)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--from-file", type=str)
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--ckpt", type=str, default="examples/Stable Diffusion/models/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--num_sample", type=int, default=5000)
    parser.add_argument("--cal_fid", type=bool, default=False)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--select_num", type=int, default=4)
    parser.add_argument("--population_num", type=int, default=12)
    parser.add_argument("--m_prob", type=float, default=0.1)
    parser.add_argument("--crossover_num", type=int, default=4)
    parser.add_argument("--mutation_num", type=int, default=4)
    parser.add_argument("--max_fid", type=float, default=3.)
    parser.add_argument("--thres", type=float, default=0.2)
    parser.add_argument("--ref_mu", type=str, default='')
    parser.add_argument("--ref_sigma", type=str, default='')
    parser.add_argument("--time_step", type=int, default=50)
    parser.add_argument("--use_ddim_init_x", type=bool, default=False)

    # 多目标评估参数（按你之前思路保留）
    parser.add_argument("--mo_eval_repeats", type=int, default=3,
                        help="repeat evaluation per candidate to estimate fid_std; latency is averaged too.")
    parser.add_argument("--mo_fid_warmup_repeats", type=int, default=0,
                        help="warmup runs per candidate to reduce first-run latency bias (optional).")
    parser.add_argument("--latency_include_fid_time", action="store_true",
                        help="if set, latency = sample_time + fid_time; otherwise only sample_time.")
    parser.add_argument("--mo_seed_base", type=int, default=12345,
                        help="base seed for repeated evaluations to compute fid_std.")

    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm_solver:
        # DPM-Solver++ 采样器
        from diffusers import DPMSolverMultistepScheduler
        from diffusers import StableDiffusionPipeline
        
        # 创建DPM-Solver++调度器
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=model.num_timesteps,
            beta_start=model.betas[0].item(),
            beta_end=model.betas[-1].item(),
            beta_schedule="scaled_linear",
            algorithm_type="dpmsolver++",
            solver_order=opt.dpm_solver_order
        )
        # 使用DPM-Solver++包装类
        from sd_autodiffusion_distill.distill.dpm_solver_wrapper import DPMSolverV2FMSampler
        sampler = DPMSolverV2FMSampler(
            model=model,
            scheduler=scheduler,
            num_steps=opt.dpm_solver_steps,
            fm_strength=0.04,
            fm_t_start=0.85
        )
    elif opt.acdms:
        from ldm.models.diffusion.unipc import NoiseScheduleVP
        from ldm.models.diffusion.acdms import ACDMSSampler

        alphas_cumprod = model.alphas_cumprod
        noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=alphas_cumprod)
        sampler = ACDMSSampler(
            model=model,
            noise_schedule=noise_schedule,
            fm_strength=0.04,
            fm_t_start=0.85,
            dynamic_thresholding_ratio=0.995,
            thresholding_max_val=1.0,
            fm_lambda_clip=1.0,
        )
    else:
        sampler = DDIMSampler(model)

    dataloader_info = build_dataloader(config, opt)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    os.makedirs(outpath, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(outpath, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    batch_size = opt.n_samples

    # 你当前脚本没有真的走 dpm_solver 分支（上面也没实现 DPMSolverSampler）
    # 如果你后续补齐 dpm_solver，可以在这里恢复原先那段 full_timesteps/init_timesteps 构造逻辑
    dpm_params = None

    t = time.time()
    searcher = EvolutionSearcher(
        opt=opt, model=model, time_step=opt.time_step,
        ref_mu=opt.ref_mu, ref_sigma=opt.ref_sigma,
        sampler=sampler, dataloader_info=dataloader_info,
        batch_size=batch_size, dpm_params=dpm_params
    )
    searcher.search()
    logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))


if __name__ == '__main__':
    main()
