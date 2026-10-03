#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage NAS evaluator for guided-diffusion ADM/ImageNet64.

Stage0:
- instantiate a static ADM architecture candidate
- partially load checkpoint weights by shape match
- evaluate with a dense Stage1 policy

Stage1:
- timesteps are fixed externally
- each step applies its own skip-block policy
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 添加上级目录到 sys.path，使同一包内的模块可以被导入
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from arch_space import Genotype, UNetArchConfig

# guided_diffusion package lives under examples/guided_diffusion
guided_diffusion_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if guided_diffusion_pkg_path not in sys.path:
    sys.path.insert(0, guided_diffusion_pkg_path)

autodiffusion_main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if autodiffusion_main_path not in sys.path:
    sys.path.insert(0, autodiffusion_main_path)

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.dynamic_unet import AttentionBlock, ResBlock
from guided_diffusion.respace import SpacedDiffusion

try:
    from imagenet64_evaluation_metrics import (
        calculate_fid_for_imagenet64,
        evaluate_generated_images_for_imagenet64,
    )

    IMAGENET64_EVAL_METRICS_AVAILABLE = True
except ImportError:
    IMAGENET64_EVAL_METRICS_AVAILABLE = False


def _seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        return ckpt_obj
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt_obj)}")


def _infer_model_channels(state: Dict[str, torch.Tensor]) -> Tuple[int, int, str]:
    key = None
    for k in ("time_embed.0.weight", "model.time_embed.0.weight"):
        if k in state:
            key = k
            break
    if key is None:
        for k in state.keys():
            if k.endswith("time_embed.0.weight"):
                key = k
                break
    if key is None:
        raise KeyError("Cannot find time_embed.0.weight in checkpoint.")
    w = state[key]
    return int(w.shape[1]), int(w.shape[0]), key


def _infer_out_channels(state: Dict[str, torch.Tensor]) -> Tuple[int, str]:
    key = None
    for k in ("out.2.weight", "model.out.2.weight"):
        if k in state:
            key = k
            break
    if key is None:
        for k in state.keys():
            if k.endswith("out.2.weight"):
                key = k
                break
    if key is None:
        raise KeyError("Cannot find out.2.weight in checkpoint.")
    return int(state[key].shape[0]), key


def _infer_class_conditional(state: Dict[str, torch.Tensor]) -> bool:
    for key in state.keys():
        if "label_emb" in key or "label_embed" in key:
            return True
    return False


def _infer_num_classes(state: Dict[str, torch.Tensor]) -> Optional[int]:
    for key in (
        "label_emb.weight",
        "label_embed.weight",
        "model.label_emb.weight",
        "model.label_embed.weight",
    ):
        if key in state:
            return int(state[key].shape[0])
    for key, value in state.items():
        if key.endswith("label_emb.weight") or key.endswith("label_embed.weight"):
            return int(value.shape[0])
    return None


def _build_class_schedule(num_samples: int, num_classes: int, mode: str, seed: int) -> np.ndarray:
    mode = str(mode).lower()
    if num_samples <= 0:
        return np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    if mode == "uniform":
        repeats = (num_samples + num_classes - 1) // num_classes
        labels = np.tile(np.arange(num_classes, dtype=np.int64), repeats)[:num_samples]
        rng.shuffle(labels)
        return labels
    if mode == "random":
        return rng.integers(0, num_classes, size=num_samples, dtype=np.int64)
    raise ValueError(f"Unsupported class_sampling_mode={mode!r}. Expected 'random' or 'uniform'.")


def _as_sorted_int_tuple(values: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if values is None:
        return tuple()
    return tuple(sorted({int(x) for x in values}))


def _normalize_protected_block_types(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Expand human-friendly protection aliases to actual block kinds.

    The block catalog uses concrete kinds such as res_in/res_out/res_down/res_up.
    A config value like "output" alone would not match anything without this
    expansion, which made earlier LSUN searches more aggressive than intended.
    """
    if values is None:
        return tuple()
    alias = {
        "input": {"res_in", "attn_in"},
        "output": {"res_out", "attn_out"},
        "middle": {"res_mid", "attn_mid"},
        "downsampling": {"res_down"},
        "upsampling": {"res_up"},
        "down": {"res_down"},
        "up": {"res_up"},
        "attention": {"attn_in", "attn_mid", "attn_out"},
        "residual": {"res_in", "res_mid", "res_out"},
    }
    out = set()
    for item in values:
        key = str(item).strip().lower()
        if not key:
            continue
        out.update(alias.get(key, {key}))
    return tuple(sorted(out))


def _validate_fixed_timesteps(timesteps: Sequence[int], *, num_steps: int) -> Tuple[int, ...]:
    ts = tuple(int(x) for x in timesteps)
    if len(ts) == 0:
        raise ValueError("fixed_timesteps is empty.")
    if len(ts) != len(set(ts)):
        raise ValueError(
            f"fixed_timesteps contains duplicates: {list(ts)}. "
            "For DDIM/ADM few-step sampling, duplicated timesteps collapse an update and usually destroy quality."
        )
    bad = [t for t in ts if t < 0 or t >= int(num_steps)]
    if bad:
        raise ValueError(f"fixed_timesteps out of [0,{int(num_steps)-1}]: {bad}; all={list(ts)}")
    if any(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)):
        raise ValueError(
            f"fixed_timesteps must be strictly descending high->low, got {list(ts)}. "
            "Example 4-step LSUN-safe seed: [848, 588, 361, 126]."
        )
    return ts


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        try:
            weights = torchvision.models.Inception_V3_Weights.DEFAULT
            inception = torchvision.models.inception_v3(
                weights=weights, transform_input=False, aux_logits=False
            )
        except Exception:
            inception = torchvision.models.inception_v3(
                weights=None, transform_input=False, aux_logits=False
            )
        inception.fc = nn.Identity()
        inception.eval()
        for p in inception.parameters():
            p.requires_grad_(False)
        self.inception = inception.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.inception(x)


def compute_stats_from_features(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fid)


class Evaluator:
    def __init__(
        self,
        *,
        ckpt_path: str,
        image_size: int = 64,
        num_steps: int = 1000,
        device: str = "cuda",
        fp16: bool = False,
        ref_mu: Optional[str] = None,
        ref_sigma: Optional[str] = None,
        batch_size: int = 16,
        num_samples: int = 2000,
        seed: int = 0,
        max_resblocks_hint: int = 256,
        outdir: str = "./outputs",
        latents_shape: Optional[List[int]] = None,
        calculate_is: bool = False,
        calculate_precision_recall: bool = False,
        calculate_kid: bool = False,
        is_splits: int = 10,
        pr_k: int = 3,
        kid_num_subsets: int = 100,
        kid_max_subset_size: int = 1000,
        real_images_path: Optional[str] = None,
        timesteps_npy_path: Optional[str] = None,
        arch_cfg: Optional[UNetArchConfig] = None,
        fixed_timesteps: Optional[List[int]] = None,
        mo_eval_repeats: int = 1,
        use_median_fid: bool = True,
        use_median_latency: bool = True,
        class_sampling_mode: str = "uniform",
        searchable_block_mode: str = "attention_only",
        protected_attention_resolutions: Optional[List[int]] = None,
        protect_middle_attention: bool = True,
        protected_block_types: Optional[List[str]] = None,
        eval_dataset: str = "imagenet64",
        ref_stats_pkl: Optional[str] = None,
        noise_schedule: str = "linear",
        classifier_path: Optional[str] = None,
        classifier_scale: float = 0.0,
        classifier_depth: int = 4,
        classifier_use_fp16: bool = False,
        require_project_fid: bool = False,
    ):
        self.ckpt_path = str(ckpt_path)
        self.image_size = int(image_size)
        self.num_steps = int(num_steps)
        self.noise_schedule = str(noise_schedule)
        self.classifier_scale = float(classifier_scale)
        self.classifier = None
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.max_resblocks_hint = int(max_resblocks_hint)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.latents_shape = latents_shape or [3, self.image_size, self.image_size]
        self.in_channels = int(self.latents_shape[0])

        self.calculate_is = bool(calculate_is)
        self.calculate_precision_recall = bool(calculate_precision_recall)
        self.calculate_kid = bool(calculate_kid)
        self.is_splits = int(is_splits)
        self.pr_k = int(pr_k)
        self.kid_num_subsets = int(kid_num_subsets)
        self.kid_max_subset_size = int(kid_max_subset_size)
        self.real_images_path = real_images_path
        self.mo_eval_repeats = max(1, int(mo_eval_repeats))
        self.use_median_fid = bool(use_median_fid)
        self.use_median_latency = bool(use_median_latency)
        self.class_sampling_mode = str(class_sampling_mode).lower()
        if self.class_sampling_mode not in {"random", "uniform"}:
            raise ValueError(
                f"Unsupported class_sampling_mode={self.class_sampling_mode!r}. Expected 'random' or 'uniform'."
            )
        self.searchable_block_mode = str(searchable_block_mode).lower()
        if self.searchable_block_mode not in {"all", "attention_only"}:
            raise ValueError(
                f"Unsupported searchable_block_mode={self.searchable_block_mode!r}. "
                "Expected 'all' or 'attention_only'."
            )
        self.protected_attention_resolutions = _as_sorted_int_tuple(protected_attention_resolutions)
        self.protect_middle_attention = bool(protect_middle_attention)
        self.protected_block_types = set(_normalize_protected_block_types(protected_block_types))

        # Dataset-aware evaluation switch.
        # - imagenet64: use the existing ImageNet64 branch when available.
        # - lsun_bedroom: use evaluator_v1.cal_fid with ref_stats_obj.pkl.
        # - other/generic: use ref_mu/ref_sigma with InceptionFeatureExtractor.
        self.eval_dataset = str(eval_dataset or "imagenet64").lower()
        if require_project_fid and not IMAGENET64_EVAL_METRICS_AVAILABLE:
            raise ImportError("Project ImageNet64 FID backend is unavailable; refusing metric fallback.")
        self.ref_stats_pkl = ref_stats_pkl
        self.lsun_evaluator = None
        self.lsun_ref_stats = None

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type == "cuda"

        self.fixed_timesteps = None
        if fixed_timesteps is not None:
            self.fixed_timesteps = _validate_fixed_timesteps(fixed_timesteps, num_steps=self.num_steps)
        elif timesteps_npy_path and os.path.exists(timesteps_npy_path):
            self.fixed_timesteps = _validate_fixed_timesteps(np.load(timesteps_npy_path).tolist(), num_steps=self.num_steps)

        _seed_all(self.seed)

        ckpt_obj = torch.load(self.ckpt_path, map_location="cpu")
        self.ckpt_state = _strip_prefix(_extract_state_dict(ckpt_obj))

        self.model, self.diffusion, self.model_cfg = self._build_model_and_diffusion(arch_cfg=arch_cfg)
        self.sampling_diffusion = self._build_sampling_diffusion()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.classifier_scale > 0:
            if not self.model_cfg["class_conditional"]:
                raise ValueError("ADM-G requires a class-conditional diffusion checkpoint.")
            if not classifier_path or not Path(classifier_path).is_file():
                raise FileNotFoundError(f"ADM-G classifier checkpoint missing: {classifier_path}")
            from guided_diffusion.script_util import create_classifier
            self.classifier = create_classifier(
                image_size=self.image_size,
                classifier_use_fp16=bool(classifier_use_fp16),
                classifier_width=128,
                classifier_depth=int(classifier_depth),
                classifier_attention_resolutions="32,16,8",
                classifier_use_scale_shift_norm=True,
                classifier_resblock_updown=True,
                classifier_pool="attention",
            )
            self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
            self.classifier.to(self.device)
            if classifier_use_fp16:
                self.classifier.convert_to_fp16()
            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad_(False)
        self.total_blocks = int(getattr(self.model, "layer_num", 0))
        self.block_catalog = self._build_block_catalog()
        self.searchable_block_ids = self._select_searchable_block_ids()
        self._write_search_space_summary(arch_cfg=arch_cfg)

        acp = getattr(self.diffusion, "alphas_cumprod", None)
        if acp is None:
            raise AttributeError("diffusion has no alphas_cumprod; check guided_diffusion version.")
        if isinstance(acp, np.ndarray):
            self.alphas_cumprod = torch.from_numpy(acp).to(self.device)
        else:
            self.alphas_cumprod = torch.tensor(acp, device=self.device)

        self.extractor = None
        self.ref_mu_path = ref_mu
        self.ref_sigma_path = ref_sigma
        self.ref_mu = None
        self.ref_sigma = None

        if self.eval_dataset in {"lsun", "lsun_bedroom", "bedroom"}:
            if not self.ref_stats_pkl:
                raise ValueError(
                    "LSUN-Bedroom evaluation requires ref_stats_pkl, e.g. "
                    ".../lsun_bedroom_ref_stats_obj.pkl"
                )
            import pickle
            import tensorflow.compat.v1 as tf
            from evaluations.evaluator_v1 import Evaluator_v1

            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.allow_growth = True
            self.lsun_evaluator = Evaluator_v1(tf.Session(config=tf_config))
            with open(self.ref_stats_pkl, "rb") as f:
                self.lsun_ref_stats = pickle.load(f)
        elif self.eval_dataset in {"imagenet64", "imagenet"} and IMAGENET64_EVAL_METRICS_AVAILABLE:
            # Use the project-specific ImageNet64 metrics implementation.
            pass
        elif ref_mu and ref_sigma:
            # Generic fallback. Here ref_mu/ref_sigma must be Inception feature statistics,
            # not RGB channel mean/std.
            self.ref_mu = np.load(ref_mu)
            self.ref_sigma = np.load(ref_sigma)
            self.extractor = InceptionFeatureExtractor(self.device)

    def _build_model_and_diffusion(self, arch_cfg: Optional[UNetArchConfig]):
        from guided_diffusion import script_util

        state = self.ckpt_state
        mc, te_dim, te_key = _infer_model_channels(state)
        out_ch, out_key = _infer_out_channels(state)
        class_cond = _infer_class_conditional(state)
        num_classes = _infer_num_classes(state)

        defaults = script_util.model_and_diffusion_defaults()
        defaults["image_size"] = self.image_size
        defaults["num_channels"] = mc
        defaults["class_cond"] = class_cond
        defaults["learn_sigma"] = True
        defaults["attention_resolutions"] = "32,16,8"
        defaults["dropout"] = 0.0
        defaults["diffusion_steps"] = self.num_steps
        defaults["noise_schedule"] = self.noise_schedule
        defaults["rescale_learned_sigmas"] = False
        defaults["rescale_timesteps"] = False
        defaults["num_head_channels"] = 64
        defaults["use_new_attention_order"] = True
        defaults["resblock_updown"] = True
        defaults["use_scale_shift_norm"] = True
        defaults["use_dynamic_unet"] = True

        # For guided-diffusion, channel_mult="" lets script_util choose the correct
        # built-in channel multiplier for each image_size. For 256x256 that is
        # (1,1,2,2,4,4). The old ImageNet64 value "1,2,3,4" silently builds
        # the wrong LSUN-Bedroom network.
        defaults["channel_mult"] = ""

        if arch_cfg is None:
            defaults["num_res_blocks"] = 2 if self.image_size == 256 else 3
        else:
            defaults["num_res_blocks"] = int(arch_cfg.num_res_blocks)
        defaults["attention_resolutions"] = (
            "32,16,8" if arch_cfg is None else ",".join(str(int(x)) for x in arch_cfg.attention_resolutions)
        )

        model, diffusion = script_util.create_model_and_diffusion(**defaults)
        model.to(self.device)

        model_state = model.state_dict()
        matched = 0
        loaded_state = dict(model_state)
        for key, value in state.items():
            if key in model_state and model_state[key].shape == value.shape:
                loaded_state[key] = value
                matched += 1
        missing, unexpected = model.load_state_dict(loaded_state, strict=False)

        report = {
            "ckpt_path": self.ckpt_path,
            "time_embed_key": te_key,
            "inferred_model_channels": mc,
            "inferred_time_embed_dim": te_dim,
            "out_weight_key": out_key,
            "inferred_out_channels": out_ch,
            "inferred_class_conditional": class_cond,
            "inferred_num_classes": num_classes,
            "class_sampling_mode": self.class_sampling_mode,
            "arch_cfg": None if arch_cfg is None else arch_cfg.to_dict(),
            "matched_keys_count": matched,
            "unmatched_model_keys": [
                key for key, value in model_state.items()
                if key not in state or state[key].shape != value.shape
            ],
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
            "missing_keys_preview": list(missing[:80]),
            "unexpected_keys_preview": list(unexpected[:80]),
            "model_defaults": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in defaults.items()
                if k in {
                    "image_size", "num_channels", "num_res_blocks", "attention_resolutions",
                    "channel_mult", "class_cond", "learn_sigma", "noise_schedule",
                    "rescale_timesteps", "rescale_learned_sigmas", "num_head_channels",
                }
            },
        }
        suffix = "teacher" if arch_cfg is None else arch_cfg.key().replace(" ", "")
        report_path = self.outdir / f"ckpt_load_report_{suffix}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        model_cfg = {
            "model_channels": mc,
            "out_channels": out_ch,
            "class_conditional": class_cond,
            "num_classes": int(num_classes) if num_classes is not None else int(defaults.get("num_classes", 1000)),
            "learn_sigma": bool(defaults["learn_sigma"]),
            "diffusion_steps": int(defaults["diffusion_steps"]),
            "noise_schedule": str(defaults["noise_schedule"]),
            "rescale_timesteps": bool(defaults["rescale_timesteps"]),
            "rescale_learned_sigmas": bool(defaults["rescale_learned_sigmas"]),
        }
        return model, diffusion, model_cfg

    def _build_sampling_diffusion(self) -> SpacedDiffusion:
        use_timesteps = sorted(int(x) for x in self._effective_timesteps())
        return SpacedDiffusion(
            use_timesteps=use_timesteps,
            betas=gd.get_named_beta_schedule(
                self.model_cfg["noise_schedule"],
                self.model_cfg["diffusion_steps"],
            ),
            model_mean_type=gd.ModelMeanType.EPSILON,
            model_var_type=(
                gd.ModelVarType.LEARNED_RANGE
                if self.model_cfg["learn_sigma"]
                else gd.ModelVarType.FIXED_LARGE
            ),
            loss_type=gd.LossType.MSE,
            rescale_timesteps=self.model_cfg["rescale_timesteps"],
        )

    def _make_block_item(
        self,
        *,
        module,
        kind: str,
        stage: str,
        level: Optional[int],
        ds: int,
        block_index: Optional[int] = None,
    ) -> Dict[str, object]:
        return {
            "layer_id": int(module.layer_id),
            "kind": str(kind),
            "stage": str(stage),
            "level": None if level is None else int(level),
            "block_index": None if block_index is None else int(block_index),
            "downsample_rate": int(ds),
            "resolution": int(self.image_size // max(1, ds)),
            "channels": int(getattr(module, "channels", getattr(module, "out_channels", 0))),
            "updown": bool(getattr(module, "updown", False)),
        }

    def _build_block_catalog(self) -> List[Dict[str, object]]:
        model = self.model
        if not hasattr(model, "input_blocks") or not hasattr(model, "middle_block") or not hasattr(model, "output_blocks"):
            raise TypeError(
                "NAS evaluator now expects a Dynamic_UNetModel-style model with "
                "input_blocks / middle_block / output_blocks."
            )

        catalog: List[Dict[str, object]] = []
        ds = 1

        input_idx = 1
        fallback_channel_mult = (1, 1, 2, 2, 4, 4) if self.image_size == 256 else (1, 2, 3, 4)
        channel_mult = tuple(int(x) for x in getattr(model, "channel_mult", fallback_channel_mult))
        num_res_blocks = int(getattr(model, "num_res_blocks", 0))

        for level, _mult in enumerate(channel_mult):
            for block_index in range(num_res_blocks):
                block = model.input_blocks[input_idx]
                input_idx += 1
                for layer in block:
                    if isinstance(layer, ResBlock):
                        catalog.append(
                            self._make_block_item(
                                module=layer,
                                kind="res_in",
                                stage="input",
                                level=level,
                                ds=ds,
                                block_index=block_index,
                            )
                        )
                    elif isinstance(layer, AttentionBlock):
                        catalog.append(
                            self._make_block_item(
                                module=layer,
                                kind="attn_in",
                                stage="input",
                                level=level,
                                ds=ds,
                                block_index=block_index,
                            )
                        )
            if level != len(channel_mult) - 1:
                block = model.input_blocks[input_idx]
                input_idx += 1
                for layer in block:
                    if isinstance(layer, ResBlock):
                        catalog.append(
                            self._make_block_item(
                                module=layer,
                                kind="res_down",
                                stage="down",
                                level=level,
                                ds=ds,
                            )
                        )
                ds *= 2

        for layer in model.middle_block:
            if isinstance(layer, ResBlock):
                catalog.append(
                    self._make_block_item(
                        module=layer,
                        kind="res_mid",
                        stage="middle",
                        level=None,
                        ds=ds,
                    )
                )
            elif isinstance(layer, AttentionBlock):
                catalog.append(
                    self._make_block_item(
                        module=layer,
                        kind="attn_mid",
                        stage="middle",
                        level=None,
                        ds=ds,
                    )
                )

        output_levels = list(enumerate(channel_mult))[::-1]
        output_idx = 0
        for level, _mult in output_levels:
            for block_index in range(num_res_blocks + 1):
                block = model.output_blocks[output_idx]
                output_idx += 1
                for layer in block:
                    if isinstance(layer, ResBlock):
                        if getattr(layer, "updown", False):
                            kind = "res_up"
                            stage = "up"
                        else:
                            kind = "res_out"
                            stage = "output"
                        catalog.append(
                            self._make_block_item(
                                module=layer,
                                kind=kind,
                                stage=stage,
                                level=level,
                                ds=ds,
                                block_index=block_index,
                            )
                        )
                    elif isinstance(layer, AttentionBlock):
                        catalog.append(
                            self._make_block_item(
                                module=layer,
                                kind="attn_out",
                                stage="output",
                                level=level,
                                ds=ds,
                                block_index=block_index,
                            )
                        )
                if level and block_index == num_res_blocks:
                    ds //= 2

        catalog = sorted(catalog, key=lambda item: int(item["layer_id"]))
        return catalog

    def _select_searchable_block_ids(self) -> Tuple[int, ...]:
        catalog = list(self.block_catalog)
        
        # 排除注意力块（用户要求注意力块只使用基线不允许裁剪）
        non_attention_items = [item for item in catalog if not str(item["kind"]).startswith("attn")]
        
        # 排除关键块（输入层、输出层、下采样/上采样转换点）
        searchable_items = []
        for item in non_attention_items:
            kind = str(item["kind"])
            stage = str(item["stage"])
            
            # 使用配置的保护块类型
            if kind in self.protected_block_types:
                continue
            # 保护输入层的第一个块
            if stage in ["input"] and item["block_index"] == 0:
                continue
            # 保护中间层的块
            if stage in ["middle"]:
                continue
            
            searchable_items.append(item)

        return tuple(int(item["layer_id"]) for item in searchable_items)

    def _write_search_space_summary(self, arch_cfg: Optional[UNetArchConfig]) -> None:
        suffix = "teacher" if arch_cfg is None else arch_cfg.key().replace(" ", "")
        payload = {
            "searchable_block_mode": self.searchable_block_mode,
            "protected_attention_resolutions": [int(x) for x in self.protected_attention_resolutions],
            "protect_middle_attention": bool(self.protect_middle_attention),
            "total_blocks": int(self.total_blocks),
            "searchable_blocks": [int(x) for x in self.searchable_block_ids],
            "searchable_blocks_count": int(len(self.searchable_block_ids)),
            "block_catalog": self.block_catalog,
        }
        summary_path = self.outdir / f"search_space_summary_{suffix}.json"
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _effective_timesteps(self) -> List[int]:
        if self.fixed_timesteps is not None:
            return list(int(x) for x in self.fixed_timesteps)
        return [self.num_steps - 1, 0]

    def _model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        step_idx: int,
        genotype: Optional[Genotype],
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        layer_gate_map: Dict[int, float] = {}
        class_gain = 1.0
        if genotype is None:
            skip_layer: List[int] = []
        else:
            pol = genotype.get_policy_for_step(
                step_idx=step_idx,
                total_blocks=self.total_blocks,
                searchable_block_ids=self.searchable_block_ids,
            )
            skip_layer = [int(x) for x in pol.skip_blocks]
            layer_gate_map = {int(layer_id): float(gate) for layer_id, gate in getattr(pol, "layer_gates", tuple())}
            class_gain = float(getattr(pol, "class_gain", 1.0))

        use_amp = self.fp16 and self.device.type == "cuda"

        if self.model_cfg.get("class_conditional", False) and y is None:
            raise ValueError("Missing class labels for class-conditional NAS evaluation.")

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = (
                self.model(x, t, skip_layer=skip_layer, layer_gate_map=layer_gate_map, class_gain=class_gain)
                if y is None
                else self.model(
                    x,
                    t,
                    y=y,
                    skip_layer=skip_layer,
                    layer_gate_map=layer_gate_map,
                    class_gain=class_gain,
                )
            )
        return out

    def _classifier_cond_fn(self, x, t, y=None):
        # Sampling runs under no_grad; guidance still needs gradients w.r.t. x.
        if y is None or self.classifier is None:
            raise ValueError("Classifier guidance requires labels and a loaded classifier.")
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_probs = F.log_softmax(self.classifier(x_in, t), dim=-1)
            selected = log_probs[torch.arange(len(log_probs), device=y.device), y]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale

    @torch.no_grad()
    def _ddim_sample(
        self,
        timesteps: List[int],
        n: int,
        genotype: Optional[Genotype],
        return_cpu: bool = True,
        class_seed: Optional[int] = None,
    ) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        class_schedule = None
        step_index_by_t = {int(t): idx for idx, t in enumerate(int(x) for x in timesteps)}
        if self.model_cfg.get("class_conditional", False):
            class_schedule = _build_class_schedule(
                num_samples=n,
                num_classes=int(self.model_cfg.get("num_classes", 1000)),
                mode=self.class_sampling_mode,
                seed=int(self.seed if class_seed is None else class_seed),
            )
        ddim_eta = 0.0
        if genotype is not None and getattr(genotype, "sampler_policy", None) is not None:
            ddim_eta = float(getattr(genotype.sampler_policy, "ddim_eta", 0.0) or 0.0)
        for start in range(0, n, self.batch_size):
            b = min(self.batch_size, n - start)
            y = None
            if class_schedule is not None:
                y = torch.from_numpy(class_schedule[start : start + b]).to(device=self.device, dtype=torch.long)
            model_kwargs: Dict[str, torch.Tensor] = {}
            if y is not None:
                model_kwargs["y"] = y

            def model_fn(x_in: torch.Tensor, t_in: torch.Tensor, y: Optional[torch.Tensor] = None):
                t_scalar = int(t_in[0].item())
                if t_scalar not in step_index_by_t:
                    raise KeyError(
                        f"Current timestep {t_scalar} is not in NAS fixed timesteps {list(step_index_by_t.keys())}."
                    )
                step_idx = step_index_by_t[t_scalar]
                return self._model_output(
                    x=x_in,
                    t=t_in,
                    step_idx=step_idx,
                    genotype=genotype,
                    y=y,
                )

            sample = self.sampling_diffusion.ddim_sample_loop(
                model_fn,
                (b, self.in_channels, self.image_size, self.image_size),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                cond_fn=self._classifier_cond_fn if self.classifier is not None else None,
                device=self.device,
                progress=False,
                eta=ddim_eta,
            )
            img = (sample.clamp(-1, 1) + 1.0) / 2.0
            outs.append(img.float().cpu() if return_cpu else img.float())

        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def compute_latency_ms_per_img(
        self,
        genotype: Optional[Genotype],
        warmup: int = 2,
        repeats: int = 5,
        class_seed: Optional[int] = None,
    ) -> float:
        timesteps = self._effective_timesteps()
        b = min(self.batch_size, 8)

        for _ in range(warmup):
            _ = self._ddim_sample(
                timesteps,
                b,
                genotype=genotype,
                return_cpu=False,
                class_seed=class_seed,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeats):
            _ = self._ddim_sample(
                timesteps,
                b,
                genotype=genotype,
                return_cpu=False,
                class_seed=class_seed,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        return float(((t1 - t0) / max(1, repeats * b)) * 1000.0)

    @torch.no_grad()
    def compute_fid(self, images01: torch.Tensor) -> float:
        # LSUN-Bedroom branch: use evaluator_v1 + ref_stats_obj.pkl.
        # This avoids accidentally using the ImageNet64 FID branch when
        # imagenet64_evaluation_metrics is importable in the environment.
        if self.eval_dataset in {"lsun", "lsun_bedroom", "bedroom"}:
            from evaluations.evaluator_v1 import cal_fid

            if self.lsun_evaluator is None or self.lsun_ref_stats is None:
                raise RuntimeError(
                    "LSUN evaluator is not initialized. Check eval_dataset=lsun_bedroom "
                    "and ref_stats_pkl in config_formal.yaml."
                )

            imgs = images01.detach().clamp(0, 1)
            arr = (imgs * 255.0).round().to(torch.uint8)
            arr = arr.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            return float(
                cal_fid(
                    arr,
                    self.batch_size,
                    self.lsun_evaluator,
                    ref_stats=self.lsun_ref_stats,
                )
            )

        # ImageNet64 branch.
        if self.eval_dataset in {"imagenet64", "imagenet"} and IMAGENET64_EVAL_METRICS_AVAILABLE:
            return calculate_fid_for_imagenet64(
                generated_images=images01,
                device=str(self.device),
                batch_size=self.batch_size,
                use_fp16=self.fp16,
                ref_mu=self.ref_mu_path,
                ref_sigma=self.ref_sigma_path,
            )

        # Generic fallback: ref_mu/ref_sigma must be Inception feature stats.
        if self.ref_mu is None or self.ref_sigma is None or self.extractor is None:
            return float("inf")

        feats: List[np.ndarray] = []
        for i in range(0, images01.shape[0], self.batch_size):
            x = images01[i : i + self.batch_size].to(self.device)
            f = self.extractor(x).detach().cpu().numpy()
            feats.append(f)
        feats_np = np.concatenate(feats, axis=0)
        mu, sigma = compute_stats_from_features(feats_np)
        return frechet_distance(mu, sigma, self.ref_mu, self.ref_sigma)

    def flops_proxy_g(self, genotype: Optional[Genotype]) -> float:
        total = max(len(self.searchable_block_ids), 1)
        if genotype is None or not genotype.step_policies:
            avg_skip = 0.0
        else:
            avg_skip = float(np.mean([len(p.skip_blocks) for p in genotype.step_policies]))
        ratio = max(0.0, 1.0 - avg_skip / max(total, 1))
        return float(50.0 * ratio)

    @torch.no_grad()
    def compute_all_metrics(self, images01: torch.Tensor) -> Dict[str, float]:
        # LSUN-Bedroom currently reports FID through evaluator_v1.
        if self.eval_dataset in {"lsun", "lsun_bedroom", "bedroom"}:
            return {"fid": self.compute_fid(images01)}

        if self.eval_dataset in {"imagenet64", "imagenet"} and IMAGENET64_EVAL_METRICS_AVAILABLE:
            return evaluate_generated_images_for_imagenet64(
                generated_images=images01,
                real_images=self.real_images_path,
                device=str(self.device),
                batch_size=self.batch_size,
                calculate_fid=True,
                calculate_is=self.calculate_is,
                calculate_precision_recall=self.calculate_precision_recall,
                calculate_kid=self.calculate_kid,
                is_splits=self.is_splits,
                pr_k=self.pr_k,
                kid_num_subsets=self.kid_num_subsets,
                kid_max_subset_size=self.kid_max_subset_size,
                use_fp16=self.fp16,
                ref_mu=self.ref_mu_path,
                ref_sigma=self.ref_sigma_path,
            )

        return {"fid": self.compute_fid(images01)}

    def evaluate_genotype(self, genotype: Genotype) -> Genotype:
        timesteps = self._effective_timesteps()
        rep_metrics: List[Dict[str, float]] = []
        latencies: List[float] = []
        fids: List[float] = []
        eval_start_time = time.time()
        peak_alloc_bytes = 0
        peak_reserved_bytes = 0
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

        for rep in range(self.mo_eval_repeats):
            rep_seed = self.seed + rep * 1000
            _seed_all(rep_seed)
            lat = self.compute_latency_ms_per_img(genotype, class_seed=rep_seed)
            keep_on_device = self.device.type == "cuda"
            imgs = self._ddim_sample(
                timesteps,
                self.num_samples,
                genotype=genotype,
                return_cpu=not keep_on_device,
                class_seed=rep_seed,
            )

            if (self.calculate_is or self.calculate_precision_recall or self.calculate_kid):
                metrics = self.compute_all_metrics(imgs)
            else:
                fid = self.compute_fid(imgs)
                metrics = {"fid": float(fid) if math.isfinite(fid) else float("inf")}

            metrics["latency_ms_per_img"] = float(lat)
            rep_metrics.append(metrics)
            latencies.append(float(metrics.get("latency_ms_per_img", float("inf"))))
            fids.append(float(metrics.get("fid", float("inf"))))
            if torch.cuda.is_available():
                try:
                    peak_alloc_bytes = max(peak_alloc_bytes, int(torch.cuda.max_memory_allocated(self.device)))
                    peak_reserved_bytes = max(peak_reserved_bytes, int(torch.cuda.max_memory_reserved(self.device)))
                except Exception:
                    try:
                        peak_alloc_bytes = max(peak_alloc_bytes, int(torch.cuda.max_memory_allocated()))
                        peak_reserved_bytes = max(peak_reserved_bytes, int(torch.cuda.max_memory_reserved()))
                    except Exception:
                        pass

        def _reduce(vals: List[float], use_median: bool) -> float:
            finite = [float(v) for v in vals if math.isfinite(float(v))]
            if not finite:
                return float("inf")
            return float(np.median(finite)) if use_median else float(np.mean(finite))

        metrics: Dict[str, float] = {
            "fid": _reduce(fids, self.use_median_fid),
            "fid_std": float(np.std([v for v in fids if math.isfinite(v)])) if any(math.isfinite(v) for v in fids) else float("inf"),
            "latency_ms_per_img": _reduce(latencies, self.use_median_latency),
            "latency_std": float(np.std([v for v in latencies if math.isfinite(v)])) if any(math.isfinite(v) for v in latencies) else float("inf"),
        }

        optional_keys = ["is_mean", "is_std", "precision", "recall", "kid_mean", "kid_std"]
        for key in optional_keys:
            vals = [float(m.get(key, float("nan"))) for m in rep_metrics if key in m]
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                metrics[key] = float(np.mean(finite))

        metrics["std"] = metrics["fid_std"]
        metrics["eval_wall_time_s"] = float(time.time() - eval_start_time)
        metrics["peak_gpu_allocated_gb"] = float(peak_alloc_bytes) / (1024 ** 3)
        metrics["peak_gpu_reserved_gb"] = float(peak_reserved_bytes) / (1024 ** 3)
        metrics["flops_g"] = float(self.flops_proxy_g(genotype))
        metrics["searchable_blocks_count"] = float(len(self.searchable_block_ids))
        metrics["num_samples"] = float(self.num_samples)
        metrics["timesteps"] = [int(t) for t in timesteps]
        metrics["mo_eval_repeats"] = float(self.mo_eval_repeats)

        # 生成详细的块使用日志
        block_usage_log = self._generate_block_usage_log(genotype, timesteps)
        metrics["block_usage_log"] = block_usage_log

        fid_v = float(metrics.get("fid", float("inf")))
        lat_v = float(metrics.get("latency_ms_per_img", float("inf")))
        fitness = float("inf") if (not math.isfinite(fid_v) or not math.isfinite(lat_v)) else (fid_v + 0.001 * lat_v)
        
        # 保存详细日志到文件
        self._save_block_usage_log(genotype, block_usage_log)
        
        return replace(genotype, metrics=metrics, fitness=fitness)

    def _generate_block_usage_log(self, genotype: Genotype, timesteps: List[int]) -> Dict[str, Any]:
        """生成详细的块使用日志"""
        log = {}
        
        # 按stage统计保留的块
        stage_stats = {}
        for item in self.block_catalog:
            stage = str(item["stage"])
            if stage not in stage_stats:
                stage_stats[stage] = {"total": 0, "kept": 0, "blocks": []}
            stage_stats[stage]["total"] += 1
            stage_stats[stage]["blocks"].append(item)
        
        # 统计每个timestep激活的块
        timestep_blocks = {}
        for step_idx, t in enumerate(timesteps):
            pol = genotype.get_policy_for_step(
                step_idx=step_idx,
                total_blocks=self.total_blocks,
                searchable_block_ids=self.searchable_block_ids,
            )
            skip_blocks = set(pol.skip_blocks)
            
            # 计算每个stage保留的块
            for stage, stats in stage_stats.items():
                kept_in_stage = 0
                for item in stats["blocks"]:
                    layer_id = item["layer_id"]
                    if layer_id not in skip_blocks:
                        kept_in_stage += 1
                stats["kept"] = kept_in_stage
            
            # 记录当前timestep激活的块
            activated_blocks = []
            for item in self.block_catalog:
                layer_id = item["layer_id"]
                if layer_id not in skip_blocks:
                    activated_blocks.append(layer_id)
            timestep_blocks[t] = activated_blocks
        
        # 计算attention和resblock的保留比例
        total_attn = 0
        kept_attn = 0
        total_res = 0
        kept_res = 0
        
        for item in self.block_catalog:
            layer_id = item["layer_id"]
            kind = str(item["kind"])
            
            # 检查是否为注意力块
            if kind.startswith("attn"):
                total_attn += 1
                # 注意力块总是保留
                kept_attn += 1
            elif kind.startswith("res"):
                total_res += 1
                # 检查是否在任何timestep中被保留
                for step_idx, t in enumerate(timesteps):
                    pol = genotype.get_policy_for_step(
                        step_idx=step_idx,
                        total_blocks=self.total_blocks,
                        searchable_block_ids=self.searchable_block_ids,
                    )
                    if layer_id not in pol.skip_blocks:
                        kept_res += 1
                        break
        
        log["stage_stats"] = stage_stats
        log["timestep_blocks"] = timestep_blocks
        log["attention_ratio"] = float(kept_attn) / max(1, total_attn)
        log["resblock_ratio"] = float(kept_res) / max(1, total_res)
        
        return log

    def _save_block_usage_log(self, genotype: Genotype, block_usage_log: Dict[str, Any]) -> None:
        """保存块使用日志到文件"""
        log_path = self.outdir / f"block_usage_log_{genotype.fitness:.4f}.json"
        
        # 格式化日志输出，使其更易读
        formatted_log = {
            "stage_statistics": {},
            "timestep_activated_blocks": {},
            "retention_ratios": {
                "attention_blocks": f"{block_usage_log['attention_ratio']:.2f}",
                "residual_blocks": f"{block_usage_log['resblock_ratio']:.2f}"
            }
        }
        
        # 格式化stage统计
        for stage, stats in block_usage_log["stage_stats"].items():
            formatted_log["stage_statistics"][stage] = {
                "kept": stats["kept"],
                "total": stats["total"],
                "ratio": f"{stats['kept']}/{stats['total']}"
            }
        
        # 格式化timestep激活的块
        for t, blocks in block_usage_log["timestep_blocks"].items():
            formatted_log["timestep_activated_blocks"][f"timestep_{t}"] = blocks
        
        log_path.write_text(json.dumps(formatted_log, indent=2), encoding="utf-8")
