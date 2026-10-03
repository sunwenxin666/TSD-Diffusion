# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import logging
from dataclasses import replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

# -----------------------------------------------------------------------------
# Project path bootstrap (按你项目结构保留；如你已在外部设置 PYTHONPATH，可删掉这段)
# -----------------------------------------------------------------------------
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
stable_diffusion_path = os.path.join(project_root, "examples", "Stable Diffusion")
if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# -----------------------------------------------------------------------------
# Imports from your project
# -----------------------------------------------------------------------------
from evaluation_metrics import _calculate_activation_statistics, _calculate_frechet_distance
from ldm.models.diffusion.acdms import ACDMSFM

from ..adapters.base import BaseAdapter
from ..nas.search_space import Genotype, SegmentPolicy

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# Optional: transformers CLIP (for local HF-format CLIP directory)
# -----------------------------------------------------------------------------
try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Local HF CLIP loading will be disabled.")

# -----------------------------------------------------------------------------
# Utilities: RNG capture/restore (critical for EA stability)
# -----------------------------------------------------------------------------
def capture_rng_state() -> Dict[str, Any]:
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Utilities: prompts (deterministic)
# -----------------------------------------------------------------------------
def _clean_prompts(prompts: Any) -> List[str]:
    """Accept list[str] or list[list[str]] or str. Return flat list[str]."""
    if prompts is None:
        return []
    if isinstance(prompts, str):
        s = prompts.strip()
        return [s] if s else []
    if isinstance(prompts, (tuple, list)):
        flat: List[str] = []
        if len(prompts) > 0 and isinstance(prompts[0], (tuple, list)):
            for group in prompts:
                if not group:
                    continue
                for p in group:
                    if p is None:
                        continue
                    s = str(p).strip()
                    if s:
                        flat.append(s)
        else:
            for p in prompts:
                if p is None:
                    continue
                s = str(p).strip()
                if s:
                    flat.append(s)
        return flat
    s = str(prompts).strip()
    return [s] if s else []


def _cycle_prompts(prompt_list: List[str], n: int, offset: int = 0) -> List[str]:
    """Deterministic prompt selection (no RNG)."""
    if not prompt_list:
        return [""] * n
    L = len(prompt_list)
    return [prompt_list[(offset + i) % L] for i in range(n)]

# -----------------------------------------------------------------------------
# Utilities: policy repair
# -----------------------------------------------------------------------------
def _repair_segment_policy(policy: Optional[SegmentPolicy], total_layers: int) -> SegmentPolicy:
    """Ensure layer_mask exists, has correct length, and not all-false."""
    if policy is None:
        mask = tuple([True] * total_layers)
        return SegmentPolicy(layer_mask=mask, stage_cond_gain=tuple(), stage_gate_scale=tuple())

    lm = getattr(policy, "layer_mask", None)
    if lm is None or len(lm) == 0 or len(lm) != total_layers:
        lm = tuple([True] * total_layers)

    if not any(lm):
        lm = list(lm)
        lm[random.randrange(total_layers)] = True
        lm = tuple(lm)

    return replace(policy, layer_mask=lm)

# -----------------------------------------------------------------------------
# Utilities: text encoder discovery + extraction
# -----------------------------------------------------------------------------
def _extract_text_embedding(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for x in out:
            if isinstance(x, torch.Tensor):
                return x
        raise TypeError(f"encode_text returned list/tuple but no tensor found: {type(out)}")
    if isinstance(out, dict):
        for k in ["cond", "c", "embedding", "emb", "context", "z"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("encode_text returned dict but no tensor value found.")
    raise TypeError(f"Unsupported encode_text output type: {type(out)}")


def _get_text_encoder_fn(cfg: Dict[str, Any]) -> Callable[[List[str]], Any]:
    """Priority: cfg['encode_text'] > cfg['text_encoder'] > cfg['teacher_pipeline'].encode_text"""
    if callable(cfg.get("encode_text", None)):
        return cfg["encode_text"]
    if callable(cfg.get("text_encoder", None)):
        return cfg["text_encoder"]
    tp = cfg.get("teacher_pipeline", None)
    if tp is not None and callable(getattr(tp, "encode_text", None)):
        return tp.encode_text
    raise KeyError("No usable text encoder found in eval_config.")

# -----------------------------------------------------------------------------
# FID (ref stats cache): supports .npz or (mu.npy, sigma.npy)
# -----------------------------------------------------------------------------
_fid_ref_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _load_fid_ref_stats(ref_stats_path: Union[str, Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load reference (mu, sigma) from:
    - a single .npz (with keys mu/sigma), OR
    - a tuple (mu_path.npy, sigma_path.npy)
    """
    import pickle

    cache_key = "_".join(ref_stats_path) if isinstance(ref_stats_path, tuple) else ref_stats_path
    if cache_key in _fid_ref_cache:
        return _fid_ref_cache[cache_key]

    if isinstance(ref_stats_path, tuple):
        mu_path, sigma_path = ref_stats_path
        if not os.path.exists(mu_path):
            raise FileNotFoundError(f"FID reference mu file not found: {mu_path}")
        if not os.path.exists(sigma_path):
            raise FileNotFoundError(f"FID reference sigma file not found: {sigma_path}")
        mu_r = np.load(mu_path)
        sigma_r = np.load(sigma_path)
    else:
        if not os.path.exists(ref_stats_path):
            raise FileNotFoundError(f"FID reference stats file not found: {ref_stats_path}")
        try:
            with np.load(ref_stats_path) as data:
                mu_r, sigma_r = data["mu"], data["sigma"]
        except Exception:
            with open(ref_stats_path, "rb") as f:
                mu_r, sigma_r = pickle.load(f)

    _fid_ref_cache[cache_key] = (mu_r, sigma_r)
    return mu_r, sigma_r


def calculate_fid_from_images(
    images: torch.Tensor,
    ref_stats_path: Union[str, Tuple[str, str]],
    device: str,
    batch_size: int = 16
) -> float:
    """
    Compute FID between generated images and precomputed reference stats.
    images: (N, 3, H, W) in [0,1]
    """
    if not isinstance(images, torch.Tensor) or images.numel() == 0:
        return float("inf")
    if images.dim() != 4 or images.shape[0] < 2:
        return float("inf")

    images = torch.nan_to_num(images.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    mu_r, sigma_r = _load_fid_ref_stats(ref_stats_path)
    mu_g, sigma_g = _calculate_activation_statistics(images, device=device, batch_size=int(batch_size))
    fid = _calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    return float(fid) if np.isfinite(fid) else float("inf")

# -----------------------------------------------------------------------------
# CLIP similarity (LOCAL HF CLIP directory via transformers)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _get_hf_clip(local_dir: str, device: str):
    """
    Cache HF CLIP model/processor loaded from local directory.
    local_dir should contain config.json, pytorch_model.bin, tokenizer.json, merges.txt, vocab.json ...
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required for local HF CLIP loading.")
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"CLIP local directory not found: {local_dir}")

    processor = CLIPProcessor.from_pretrained(local_dir)
    model = CLIPModel.from_pretrained(local_dir)
    model.eval()
    model.to(device)
    return model, processor


def _tensor_images_to_pil(images: torch.Tensor) -> List[Image.Image]:
    images = images.detach().clamp(0, 1).cpu()
    pil_list: List[Image.Image] = []
    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 255.0).round().astype(np.uint8)
        pil_list.append(Image.fromarray(img))
    return pil_list


def calculate_clip_similarity_local(
    images: torch.Tensor,
    texts: List[str],
    local_dir: str,
    device: str = "cuda",
) -> float:
    """
    Calculate average cosine similarity between image/text embeddings using LOCAL HF CLIP.
    images: (N,3,H,W) in [0,1]
    texts: list[str] length N (auto pad/truncate)
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers unavailable, returning 0.0 clip similarity")
        return 0.0
    if not isinstance(images, torch.Tensor) or images.numel() == 0:
        return 0.0

    n = int(images.shape[0])
    if not texts:
        texts = [""] * n
    if len(texts) != n:
        texts = texts[:n] + [""] * max(0, n - len(texts))

    try:
        model, processor = _get_hf_clip(local_dir, device)
        pil_list = _tensor_images_to_pil(images)

        inputs = processor(text=texts, images=pil_list, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_feat = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )

        img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True)
        sim = (img_feat * txt_feat).sum(dim=1)
        return float(sim.mean().detach().cpu().item())
    except Exception as e:
        logger.warning(f"Local CLIP similarity failed: {e}")
        return 0.0

# -----------------------------------------------------------------------------
# Improved Evaluator
# -----------------------------------------------------------------------------
class ImprovedEvaluator:
    """
    Multi-objective evaluator:
      - FID (lower better)
      - Latency (lower better)
      - Optional CLIP similarity constraint/penalty (using local HF CLIP directory)
    """

    def __init__(self, adapter: BaseAdapter, eval_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = eval_config
        prompts_path = self.config.get("prompts_path", None)
        if prompts_path and isinstance(prompts_path, str) and os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                self.config["prompts"] = loaded
                self.config["fid_prompts"] = loaded
        # device
        try:
            self.device = next(adapter.model.parameters()).device
        except Exception:
            self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # layers/stages info (for policy repair)
        self.total_layers = int(self.config.get("total_layers", getattr(adapter, "total_layers", 0) or 0))
        self.total_stages = int(self.config.get("total_stages", getattr(adapter, "total_stages", 0) or 0))
        if self.total_stages <= 0:
            self.total_stages = 1

        # recommended mask length: only skippable blocks (Attention / SpatialTransformer)
        if hasattr(adapter, "_skippable_indices"):
            try:
                self.mask_len = int(len(getattr(adapter, "_skippable_indices")))
            except Exception:
                self.mask_len = int(self.total_layers)
        else:
            self.mask_len = int(self.total_layers)

        # MO repeats
        self.mo_eval_repeats = int(self.config.get("mo_eval_repeats", 1))
        self.mo_seed_base = int(self.config.get("mo_seed_base", 12345))

        # latency knobs
        self.fixed_batch_size = int(self.config.get("fixed_batch_size", 1))
        self.num_warmup_runs = int(self.config.get("num_warmup_runs", 3))
        self.num_latency_measurements = int(self.config.get("num_latency_measurements", 5))
        self.use_median_latency = bool(self.config.get("use_median_latency", True))

        # fid knobs
        self.use_median_fid = bool(self.config.get("use_median_fid", True))
        self.fid_batch_size = int(self.config.get("fid_batch_size", self.config.get("eval_batch_size", 16)))
        self.target_num_images = self.config.get("target_num_images", None)  # e.g. 1000-5000

        # IMPORTANT: allow tuple (mu.npy, sigma.npy) directly (your case)
        self.fid_ref_stats_path: Union[str, Tuple[str, str]] = self.config["fid_ref_stats_path"]

        # prompts (stable deterministic)
        prompts = _clean_prompts(self.config.get("prompts", [])) or [""]
        max_prompts = self.config.get("max_prompts", None)
        
        # 应用max_prompts截断（如果设置）
        if max_prompts is not None and isinstance(max_prompts, int) and max_prompts > 0:
            if isinstance(prompts, list):
                if len(prompts) > 0 and isinstance(prompts[0], list):
                    # prompts是List[List[str]]，截断外层列表
                    prompts = prompts[:max_prompts]
                    print(f"[ImprovedEvaluator] Prompts截断为前 {max_prompts} 组（共 {len(prompts)} 组）")
                else:
                    # prompts是List[str]，直接截断
                    prompts = prompts[:max_prompts]
                    print(f"[ImprovedEvaluator] Prompts截断为前 {max_prompts} 个（共 {len(prompts)} 个）")
            else:
                print(f"[ImprovedEvaluator] 警告：prompts类型为 {type(prompts)}，无法截断")
        
        self.prompt_pool = prompts
        
        # FID专用prompts（同样支持截断）
        fid_prompts = _clean_prompts(self.config.get("fid_prompts", self.prompt_pool)) or self.prompt_pool
        if max_prompts is not None and isinstance(max_prompts, int) and max_prompts > 0:
            if isinstance(fid_prompts, list):
                if len(fid_prompts) > 0 and isinstance(fid_prompts[0], list):
                    fid_prompts = fid_prompts[:max_prompts]
                else:
                    fid_prompts = fid_prompts[:max_prompts]
        self.fid_prompt_pool = fid_prompts

        # stable seed list for FID
        fid_seeds_cfg = self.config.get("fid_seeds", None)
        if isinstance(fid_seeds_cfg, (list, tuple)) and len(fid_seeds_cfg) > 0:
            self.fid_seeds: List[int] = [int(s) for s in fid_seeds_cfg]
        else:
            default_k = int(self.config.get("fid_num_seeds", 50))
            self.fid_seeds = [self.mo_seed_base + i for i in range(default_k)]

        # Optional CLIP alignment (local dir)
        self.enable_text_alignment = bool(self.config.get("enable_text_alignment", False))
        self.clip_local_dir = str(self.config.get("clip_local_dir", ""))  # you will set this to your local path
        self.clip_similarity_threshold = float(self.config.get("clip_similarity_threshold", 0.15))
        self.clip_similarity_penalty_weight = float(self.config.get("clip_similarity_penalty_weight", 0.0))

        if self.enable_text_alignment:
            if (not TRANSFORMERS_AVAILABLE) or (not self.clip_local_dir) or (not os.path.isdir(self.clip_local_dir)):
                logger.warning(
                    "Text alignment enabled but transformers/clip_local_dir invalid. Disabling text alignment."
                )
                self.enable_text_alignment = False

    def _get_vae(self):
        vae = self.config.get("vae", None)
        if vae is not None and hasattr(vae, "decode"):
            vae = vae.to(self.device)
            vae.eval()
            return vae
        tp = self.config.get("teacher_pipeline", None)
        if tp is not None and hasattr(tp, "vae") and hasattr(tp.vae, "decode"):
            vae = tp.vae.to(self.device)
            vae.eval()
            return vae
        raise AttributeError("VAE is invalid or missing from eval_config.")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def evaluate_population(self, population: List[Genotype]) -> List[Genotype]:
        out: List[Genotype] = []
        for g in tqdm(population, desc="Evaluating Population"):
            out.append(self.evaluate_genotype(g))
        return out

    def evaluate_genotype(self, genotype: Genotype) -> Genotype:
        saved_state = capture_rng_state()
        try:
            # global warmup
            warmup_seed = self.mo_seed_base + 100_000
            for w in range(max(0, self.num_warmup_runs)):
                _ = self._generate_images_for_latency(genotype, seed=warmup_seed + w, warmup=True)

            fids: List[float] = []
            latencies: List[float] = []
            clip_sims: List[float] = []

            for r in range(self.mo_eval_repeats):
                rep_seed_base = self.mo_seed_base + r * 10_000

                # latency
                lat_center, _lat_std = self._measure_latency(genotype, seed=rep_seed_base + 777)
                latencies.append(lat_center)

                # fid (+ clip)
                fid_val, clip_val = self._evaluate_fid_and_clip(genotype, seed_offset=rep_seed_base)
                fids.append(fid_val)
                clip_sims.append(clip_val)

            fid_value = float(np.median(fids)) if self.use_median_fid else float(np.mean(fids))
            fid_std = float(np.std(fids))

            latency_value = float(np.median(latencies)) if self.use_median_latency else float(np.mean(latencies))
            latency_std = float(np.std(latencies))

            clip_similarity = float(np.mean(clip_sims)) if clip_sims else 0.0

            text_alignment_penalty = 0.0
            if self.enable_text_alignment:
                if clip_similarity < self.clip_similarity_threshold:
                    fid_value = float("inf")
                elif self.clip_similarity_penalty_weight > 0:
                    text_alignment_penalty = self.clip_similarity_penalty_weight * (1.0 - clip_similarity)
                    fid_value = fid_value + text_alignment_penalty

            genotype = replace(genotype, metrics={
                "fid": fid_value,
                "fid_std": fid_std,
                "latency": latency_value,
                "latency_std": latency_std,
                "clip_similarity": clip_similarity,
                "text_alignment_penalty": float(text_alignment_penalty),
            })

            if not np.isfinite(fid_value) or not np.isfinite(latency_value):
                genotype = replace(genotype, fitness=float("inf"))
            else:
                fid_weight = float(self.config.get("fitness_fid_weight", 1.0))
                lat_weight = float(self.config.get("fitness_latency_weight", 0.1))
                genotype = replace(genotype, fitness=fid_weight * fid_value + lat_weight * latency_value)

            return genotype

        except Exception as e:
            logger.exception(f"Error evaluating genotype: {e}")
            genotype = replace(genotype, metrics={
                "fid": float("inf"),
                "fid_std": 0.0,
                "latency": float("inf"),
                "latency_std": 0.0,
                "clip_similarity": 0.0,
                "text_alignment_penalty": 0.0,
            }, fitness=float("inf"))
            return genotype
        finally:
            restore_rng_state(saved_state)

    # -------------------------------------------------------------------------
    # Latency measurement
    # -------------------------------------------------------------------------
    def _measure_latency(self, genotype: Genotype, seed: int) -> Tuple[float, float]:
        timings: List[float] = []

        # warmup
        for w in range(max(0, self.num_warmup_runs)):
            _ = self._generate_images_for_latency(genotype, seed=seed + 10_000 + w, warmup=True)

        # measure
        for m in range(max(1, self.num_latency_measurements)):
            _, t = self._generate_images_for_latency(genotype, seed=seed + m, warmup=False)
            timings.append(float(t))

        center = float(np.median(timings)) if self.use_median_latency else float(np.mean(timings))
        spread = float(np.std(timings))
        return center, spread

    @torch.no_grad()
    def _generate_images_for_latency(self, genotype: Genotype, seed: int, warmup: bool) -> Tuple[torch.Tensor, float]:
        _seed_everything(seed)

        latents_shape_base: Tuple[int, ...] = tuple(self.config["latents_shape"])
        b = max(1, int(self.fixed_batch_size))
        latents_shape = (b,) + latents_shape_base[1:]

        prompt_offset = seed % max(1, len(self.prompt_pool))
        prompts = _cycle_prompts(self.prompt_pool, b, offset=prompt_offset)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time() if not warmup else None

        images = self._generate_images_once(genotype=genotype, latents_shape=latents_shape, prompts=prompts)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = 0.0 if warmup else (time.time() - float(t0))
        return images, float(elapsed)

    # -------------------------------------------------------------------------
    # FID (+ optional CLIP)
    # -------------------------------------------------------------------------
    def _evaluate_fid_and_clip(self, genotype: Genotype, seed_offset: int) -> Tuple[float, float]:
        if self.target_num_images is not None:
            target_n = max(2, int(self.target_num_images))
        else:
            target_n = max(2, int(self.fid_batch_size) * int(len(self.fid_seeds)))

        all_imgs: List[torch.Tensor] = []
        all_prompts: List[str] = []

        produced = 0
        batch_idx = 0

        while produced < target_n:
            bs = min(int(self.fid_batch_size), target_n - produced)
            latents_shape_base: Tuple[int, ...] = tuple(self.config["latents_shape"])
            latents_shape = (bs,) + latents_shape_base[1:]

            seed = int(self.fid_seeds[batch_idx % len(self.fid_seeds)]) + int(seed_offset)
            _seed_everything(seed)

            prompt_offset = (batch_idx * bs) % max(1, len(self.fid_prompt_pool))
            prompts = _cycle_prompts(self.fid_prompt_pool, bs, offset=prompt_offset)

            images = self._generate_images_once(genotype=genotype, latents_shape=latents_shape, prompts=prompts)

            all_imgs.append(images.detach().cpu())
            all_prompts.extend(prompts)

            produced += bs
            batch_idx += 1

        images_all = torch.cat(all_imgs, dim=0).clamp(0, 1)

        fid_value = calculate_fid_from_images(
            images_all.to(self.device),
            self.fid_ref_stats_path,
            device=str(self.device),
            batch_size=int(self.config.get("fid_inception_batch_size", 16)),
        )

        clip_val = 0.0
        if self.enable_text_alignment:
            clip_val = calculate_clip_similarity_local(
                images_all,
                all_prompts,
                local_dir=self.clip_local_dir,
                device=str(self.device),
            )

        return float(fid_value), float(clip_val)

    # -------------------------------------------------------------------------
    # Core generation (one batch)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _generate_images_once(self, genotype: Genotype, latents_shape: Tuple[int, ...], prompts: List[str]) -> torch.Tensor:
        noise_schedule = self.config["noise_schedule"]
        vae = self._get_vae()
        cfg_scale = float(self.config.get("cfg_scale", 4.5))
        sampler_type = str(self.config.get("sampler_type", "acdms"))

        batch_size = int(latents_shape[0])
        if len(prompts) != batch_size:
            prompts = prompts[:batch_size] + [""] * max(0, batch_size - len(prompts))

        # text embeddings
        text_encoder_fn = _get_text_encoder_fn(self.config)
        cond_out = text_encoder_fn(prompts)
        uncond_out = text_encoder_fn([""] * batch_size)
        cond_emb = _extract_text_embedding(cond_out).to(self.device)
        uncond_emb = _extract_text_embedding(uncond_out).to(self.device)

        latents = torch.randn(latents_shape, device=self.device)

        # ------------------------------------------------------------------
        # Timesteps
        # ------------------------------------------------------------------
        # NEW (recommended): fixed timesteps passed by config (loaded from npy in run_nas_search)
        # Legacy fallback: genotype.u_schedule (searched)
        ns = noise_schedule
        t0 = 1.0 / float(ns.total_N)
        tT = float(ns.T)

        if self.config.get("timesteps", None) is not None:
            ts = self.config["timesteps"]
            if isinstance(ts, np.ndarray):
                ts = torch.from_numpy(ts.copy())
            timesteps = ts.to(self.device).float().view(-1)
            timesteps = torch.clamp(timesteps, min=t0 + 1e-6, max=tT)
            timesteps, _ = torch.sort(timesteps, descending=True)
        else:
            u = torch.tensor(genotype.u_schedule, device=self.device, dtype=torch.float32)
            timesteps = t0 + u * (tT - t0)
            timesteps = torch.clamp(timesteps, min=t0 + 1e-6, max=tT)
            timesteps, _ = torch.sort(timesteps, descending=True)

        # Build a robust mapping from discrete timestep index -> step_idx.
        # We do this because model_fn_noise only receives t_in, not the step index.
        disc_ids = (timesteps * 999).round().long().detach().cpu().numpy()
        disc_to_step = {int(d): int(i) for i, d in enumerate(disc_ids.tolist())}
        disc_array = disc_ids.astype(np.int64)

        def model_fn_noise(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            t_discrete = (t_in * 999).round().long()

            # Map current t -> step index (robust to small floating mismatch)
            d0 = int(t_discrete[0].item())
            step_idx = disc_to_step.get(d0, None)
            if step_idx is None:
                # nearest neighbor fallback
                step_idx = int(np.argmin(np.abs(disc_array - d0)))

            # Apply per-step policy (or fallback to legacy in Genotype)
            policy = genotype.get_policy_for_step(
                step_idx,
                mask_len=self.mask_len,
                total_stages=self.total_stages,
                full_mask_len=self.total_layers,
            )
            self.adapter.set_policy(policy)

            x_cat = torch.cat([x_in, x_in], dim=0)
            # Use t_in (float) instead of t_discrete (long) to match model dtype
            t_cat = torch.cat([t_in, t_in], dim=0)
            c_in = torch.cat([uncond_emb, cond_emb], dim=0).to(self.device)

            # Get the model's dtype from the adapter to determine if autocast is needed
            model_dtype = next(self.adapter.model.parameters()).dtype
            use_autocast = (model_dtype == torch.float16)

            # Use amp.autocast for half precision models (same as teacher.py)
            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16):
                e = self.adapter.forward(x_cat, t_cat, c_in)
            e_uncond, e_cond = e.chunk(2)
            return e_uncond + cfg_scale * (e_cond - e_uncond)

        sampler_policy = genotype.sampler_policy
        if sampler_type == "acdms":
            sampler = ACDMSFM(
                model_fn_noise=model_fn_noise,
                noise_schedule=noise_schedule,
                fm_strength=float(sampler_policy.fm_strength),
                fm_t_start=float(sampler_policy.fm_t_start),
            )
        elif sampler_type == "dpm_solver":
            from ldm.models.diffusion.dpm_solver import DPMSolverSampler
            sampler = DPMSolverSampler(
                model_fn=model_fn_noise,
                noise_schedule=noise_schedule,
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
            )
        elif sampler_type == "unipc":
            from ldm.models.diffusion.unipc import UniPC
            sampler = UniPC(
                model_fn=model_fn_noise,
                noise_schedule=noise_schedule,
                algorithm_type="noise_prediction",
            )
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")

        # ACDMSFM.sample() expects steps, t_start, t_end (not timesteps)
        # Convert timesteps tensor to these parameters
        steps = len(timesteps)
        t_start = float(timesteps[0].item())
        t_end = float(timesteps[-1].item())
        out = sampler.sample(x=latents, steps=steps, t_start=t_start, t_end=t_end, order=int(sampler_policy.order))
        sampled_latents = out[0] if isinstance(out, (tuple, list)) else out

        # decode (same as teacher.py decode_latents)
        scale_factor = None
        if hasattr(vae, "config") and hasattr(vae.config, "get"):
            scale_factor = vae.config.get("scale_factor", None)
        if scale_factor is None:
            scale_factor = getattr(vae, "scale_factor", 0.18215)
        scale_factor = float(scale_factor) if scale_factor is not None else 0.18215

        # Get VAE's dtype and use amp.autocast (same as teacher.py)
        vae_dtype = next(vae.parameters()).dtype
        use_autocast = (vae_dtype == torch.float16)

        final_latents = sampled_latents / scale_factor
        final_latents = final_latents.to(dtype=vae_dtype)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16):
            decoded = vae.decode(final_latents)
            images = decoded.sample if hasattr(decoded, "sample") else decoded
            images = (images / 2.0 + 0.5).clamp(0.0, 1.0)

        return images
