import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.utils import save_image


guided_diffusion_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if guided_diffusion_pkg_path not in sys.path:
    sys.path.insert(0, guided_diffusion_pkg_path)

autodiffusion_main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if autodiffusion_main_path not in sys.path:
    sys.path.insert(0, autodiffusion_main_path)

from guided_diffusion import logger
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import model_and_diffusion_defaults, create_classifier
from guided_diffusion.adm_acdms import sample_with_adm_acdms
from guided_diffusion.train_distill_imagenet64_adm import (
    _extract_state_dict,
    _arch_get,
    _str2bool,
    build_student_from_teacher,
    build_student_isomorphic_from_teacher,
    build_teacher_and_diffusion,
    gstar_to_policy_controls_for_scalar_t,
    load_config_file,
    load_npy,
    load_pickle,
    set_seed,
    _teacher_model_overrides_from_cfg,
)
from imagenet64_evaluation_metrics import calculate_fid_for_imagenet64


DEFAULT_EVAL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "eval_student_distill_imagenet64_adm.config.yaml",
)


def _resolve_path(path: Optional[str], config_path: str) -> Optional[str]:
    if path in (None, ""):
        return None
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(config_path))
    candidate = os.path.abspath(os.path.join(base_dir, path))
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(path)


def _normalize_cfg_paths(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    out = dict(cfg)
    for key in [
        "train_config",
        "student_ckpt",
        "save_dir",
        "ckpt",
        "best_timesteps_npy",
        "best_arch_pkl",
        "g_star_pkl",
        "outdir",
        "ref_mu",
        "ref_sigma",
        "ref_stats_pkl",
        "classifier_path",
    ]:
        if key in out:
            out[key] = _resolve_path(out.get(key), config_path)
    return out


def _build_sampling_diffusion(teacher_defaults: Dict[str, Any], use_timesteps: List[int]) -> SpacedDiffusion:
    base = model_and_diffusion_defaults()
    base.update(
        {
            "learn_sigma": bool(teacher_defaults["learn_sigma"]),
            "diffusion_steps": int(teacher_defaults["diffusion_steps"]),
            "noise_schedule": str(teacher_defaults["noise_schedule"]),
            "rescale_learned_sigmas": bool(teacher_defaults["rescale_learned_sigmas"]),
            "rescale_timesteps": bool(teacher_defaults["rescale_timesteps"]),
        }
    )
    return SpacedDiffusion(
        use_timesteps=sorted(int(x) for x in use_timesteps),
        betas=gd.get_named_beta_schedule(base["noise_schedule"], base["diffusion_steps"]),
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE if base["learn_sigma"] else gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=base["rescale_timesteps"],
    )


def _sampler_eta(g_star_obj) -> float:
    sampler_policy = _arch_get(g_star_obj, "sampler_policy", None)
    if sampler_policy is None:
        return 0.0
    return float(_arch_get(sampler_policy, "ddim_eta", 0.0) or 0.0)


def _load_student_state(path: str) -> Dict[str, th.Tensor]:
    obj = th.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return _extract_state_dict(obj)


def _default_student_ckpt(outdir: str) -> str:
    ema_ckpt = os.path.join(outdir, "student_ema_final.pt")
    if os.path.exists(ema_ckpt):
        return ema_ckpt
    return os.path.join(outdir, "student_final.pt")


def _save_preview_pngs(arr: np.ndarray, outdir: str, max_images: int) -> None:
    preview_dir = os.path.join(outdir, "preview_pngs")
    os.makedirs(preview_dir, exist_ok=True)
    total = min(len(arr), max_images)
    for i in range(total):
        img = th.from_numpy(arr[i]).permute(2, 0, 1).float() / 255.0
        save_image(img, os.path.join(preview_dir, f"{i:05d}.png"))


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


def _default_ref_stats() -> Dict[str, str]:
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    return {
        "ref_mu": os.path.join(models_dir, "imagenet64_mu.npy"),
        "ref_sigma": os.path.join(models_dir, "imagenet64_sigma.npy"),
    }


def _compute_fid(images01: th.Tensor, cfg: Dict[str, Any], batch_size: int, use_fp16: bool, device: str) -> float:
    eval_dataset = str(cfg.get("eval_dataset", "imagenet64")).lower()
    if eval_dataset in {"lsun", "lsun_bedroom", "bedroom"}:
        ref_stats_pkl = cfg.get("ref_stats_pkl")
        if not ref_stats_pkl:
            raise ValueError("LSUN-Bedroom evaluation requires ref_stats_pkl.")
        import tensorflow.compat.v1 as tf
        from evaluations.evaluator_v1 import Evaluator_v1, cal_fid

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        evaluator = Evaluator_v1(sess)
        evaluator.warmup()
        with open(ref_stats_pkl, "rb") as f:
            ref_stats = pickle.load(f)
        arr = (images01.detach().clamp(0, 1) * 255.0).round().to(th.uint8)
        arr = arr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        try:
            return float(cal_fid(arr, max(1, min(batch_size, arr.shape[0])), evaluator, ref_stats=ref_stats))
        finally:
            sess.close()

    return float(
        calculate_fid_for_imagenet64(
            generated_images=images01,
            device=str(device),
            batch_size=batch_size,
            use_fp16=use_fp16,
            ref_mu=cfg["ref_mu"],
            ref_sigma=cfg["ref_sigma"],
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a distillation-trained ADM student with FID and latency.")
    parser.add_argument("--config", default=DEFAULT_EVAL_CONFIG_PATH)
    parser.add_argument("--train_config", default=None)
    parser.add_argument("--use_teacher_model", type=_str2bool, default=None)
    parser.add_argument("--student_ckpt", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_fp16", type=_str2bool, default=None)
    parser.add_argument("--clip_denoised", type=_str2bool, default=None)
    parser.add_argument("--sampler", choices=["ddim", "ddpm", "acdms"], default=None)
    parser.add_argument("--ddim_eta", type=float, default=None)
    parser.add_argument("--ddim_late_correct", type=_str2bool, default=None, help="Enable DDIM late-step corrector. Default: False (no corrector).")
    parser.add_argument("--ddim_corrector_strength", type=float, default=None, help="DDIM corrector strength. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_apply_last_k_steps", type=int, default=None, help="Apply DDIM corrector only to last K steps. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_low_noise_threshold_index", type=int, default=None, help="DDIM corrector low noise threshold index. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_lambda_clip", type=float, default=None, help="DDIM corrector lambda clip. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_inner_steps_low_noise", type=int, default=None, help="DDIM corrector inner steps low noise. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_use_noise_scale", type=_str2bool, default=None, help="DDIM corrector use noise scale. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_fm_t_start", type=float, default=None, help="DDIM corrector flow matching t start. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_use_adaptive_lambda", type=_str2bool, default=None, help="DDIM corrector use adaptive lambda. Only used if ddim_late_correct is True.")
    parser.add_argument("--ddim_corrector_debug", type=_str2bool, default=None, help="DDIM corrector debug mode. Only used if ddim_late_correct is True.")
    parser.add_argument("--acdms_fm_strength", type=float, default=None)
    parser.add_argument("--acdms_fm_t_start", type=float, default=None)
    parser.add_argument("--acdms_fm_lambda_clip", type=float, default=None)
    parser.add_argument("--acdms_threshold_ratio", type=float, default=None)
    parser.add_argument("--acdms_threshold_max_val", type=float, default=None)
    parser.add_argument("--acdms_low_noise_threshold_index", type=int, default=None)
    parser.add_argument("--acdms_apply_last_k_steps", type=int, default=None)
    parser.add_argument("--acdms_adaptive_low_noise", type=_str2bool, default=None)
    parser.add_argument("--acdms_adaptive_fm_strength", type=_str2bool, default=None)
    parser.add_argument("--acdms_adaptive_last_k", type=_str2bool, default=None)
    parser.add_argument("--acdms_low_noise_step_ratio", type=float, default=None)
    parser.add_argument("--acdms_low_noise_margin", type=int, default=None)
    parser.add_argument("--save_npz", type=_str2bool, default=None)
    parser.add_argument("--save_pngs", type=_str2bool, default=None)
    parser.add_argument("--max_pngs", type=int, default=None)
    parser.add_argument("--class_cond", type=_str2bool, default=None)
    parser.add_argument("--class_sampling_mode", choices=["random", "uniform"], default=None)
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--ref_mu", default=None)
    parser.add_argument("--ref_sigma", default=None)
    parser.add_argument("--ref_stats_pkl", default=None)
    parser.add_argument("--allow_hard_skip", type=_str2bool, default=None)
    # ADM-G classifier guidance parameters
    parser.add_argument("--classifier_path", default=None)
    parser.add_argument("--classifier_scale", type=float, default=None)
    parser.add_argument("--classifier_depth", type=int, default=None)
    parser.add_argument("--classifier_use_fp16", type=_str2bool, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    eval_config_path = os.path.abspath(args.config)
    eval_cfg = _normalize_cfg_paths(load_config_file(eval_config_path), eval_config_path)
    train_config_path = args.train_config or eval_cfg.get("train_config")
    if not train_config_path:
        raise ValueError("Missing train_config. Set it in the eval config or pass --train_config.")
    train_config_path = _resolve_path(train_config_path, eval_config_path)
    train_cfg = _normalize_cfg_paths(load_config_file(train_config_path), train_config_path)

    cfg = dict(train_cfg)
    if str(train_cfg.get("eval_dataset", "imagenet64")).lower() not in {"lsun", "lsun_bedroom", "bedroom"}:
        cfg.update(_default_ref_stats())
    cfg.update(eval_cfg)
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg.update(cli_overrides)

    sampler = cfg.get("sampler", "ddim")
    use_teacher_model = bool(cfg.get("use_teacher_model", False))
    student_ckpt = None
    if not use_teacher_model:
        student_ckpt = cfg.get("student_ckpt") or _default_student_ckpt(cfg["outdir"])
        student_ckpt = _resolve_path(student_ckpt, eval_config_path)
    default_save_subdir = f"{'teacher_eval' if use_teacher_model else 'eval'}_{sampler}"
    save_dir = cfg.get("save_dir") or os.path.join(cfg["outdir"], default_save_subdir)
    save_dir = _resolve_path(save_dir, eval_config_path)
    os.makedirs(save_dir, exist_ok=True)
    logger.configure(dir=save_dir)

    set_seed(int(cfg.get("seed", 0)))
    device = cfg.get("device", "cuda")
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    use_fp16 = bool(cfg.get("use_fp16", False))

    tau_list = load_npy(cfg["best_timesteps_npy"]).astype(np.int64).tolist()
    use_best_arch = bool(cfg.get("use_best_arch", True))
    use_step_policies = bool(cfg.get("use_step_policies", True))
    best_arch_obj = load_pickle(cfg.get("best_arch_pkl")) if use_best_arch else None
    g_star_obj = load_pickle(cfg.get("g_star_pkl")) if use_step_policies else None
    if cfg.get("strict_artifacts", False):
        if use_best_arch and best_arch_obj is None:
            raise FileNotFoundError(f"Required static architecture missing: {cfg.get('best_arch_pkl')}")
        if use_step_policies and g_star_obj is None:
            raise FileNotFoundError(f"Required dynamic policy missing: {cfg.get('g_star_pkl')}")

    teacher_model_overrides = _teacher_model_overrides_from_cfg(train_cfg)
    teacher, _, teacher_defaults = build_teacher_and_diffusion(
        ckpt_path=cfg["ckpt"],
        device=device,
        use_fp16=use_fp16,
        model_overrides=teacher_model_overrides,
    )
    if use_teacher_model:
        teacher.eval()
        model = teacher
        model_source = "teacher"
        g_star_for_sampling = None
    else:
        if use_best_arch and best_arch_obj is not None:
            student = build_student_from_teacher(
                teacher=teacher,
                teacher_defaults=teacher_defaults,
                best_arch_obj=best_arch_obj,
                device=device,
                use_fp16=use_fp16,
            )
        else:
            student = build_student_isomorphic_from_teacher(
                teacher=teacher,
                teacher_defaults=teacher_defaults,
                device=device,
                use_fp16=use_fp16,
            )
        student.load_state_dict(_load_student_state(student_ckpt), strict=True)
        if use_fp16 and hasattr(student, "convert_to_fp16"):
            student.convert_to_fp16()
        student.eval()
        model = student
        model_source = "student"
        g_star_for_sampling = g_star_obj if use_step_policies else None
        del teacher
        if th.cuda.is_available():
            th.cuda.empty_cache()

    diffusion = _build_sampling_diffusion(teacher_defaults, tau_list)
    ddim_eta = float(cfg["ddim_eta"]) if cfg.get("ddim_eta") is not None else _sampler_eta(g_star_obj)
    image_size = int(teacher_defaults["image_size"])
    num_samples = int(cfg.get("num_samples", 50000))
    batch_size = int(cfg.get("batch_size", 8))
    clip_denoised = bool(cfg.get("clip_denoised", True))
    class_cond = bool(cfg.get("class_cond", teacher_defaults.get("class_cond", True)))
    num_classes = int(teacher_defaults.get("num_classes", 1000) or 1000) if class_cond else 0
    class_sampling_mode = str(cfg.get("class_sampling_mode", "random")).lower()
    class_schedule = None
    if class_cond:
        class_schedule = _build_class_schedule(
            num_samples=num_samples,
            num_classes=num_classes,
            mode=class_sampling_mode,
            seed=int(cfg.get("seed", 0)),
        )
        class_counts = np.bincount(class_schedule, minlength=num_classes)
        logger.log(
            "[Eval] class sampling "
            f"mode={class_sampling_mode}, "
            f"min_per_class={int(class_counts.min())}, "
            f"max_per_class={int(class_counts.max())}"
        )

    logger.log(f"[Eval] model_source={model_source}")
    logger.log(f"[Eval] use_best_arch={use_best_arch}, use_step_policies={use_step_policies}")
    logger.log(f"[Eval] allow_hard_skip={bool(cfg.get('allow_hard_skip', True))}")
    if student_ckpt is not None:
        logger.log(f"[Eval] ckpt={student_ckpt}")
    else:
        logger.log(f"[Eval] ckpt={cfg['ckpt']}")
    logger.log(f"[Eval] timesteps={tau_list}")
    # ADM-G classifier guidance setup
    classifier = None
    classifier_scale = float(cfg.get("classifier_scale", 1.0))  # Default to 1.0
    if classifier_scale > 0:
        classifier_path = cfg.get("classifier_path")
        if not classifier_path:
            raise ValueError("classifier_path is required when classifier_scale > 0")
        classifier_depth = int(cfg.get("classifier_depth", 4))
        classifier_use_fp16 = bool(cfg.get("classifier_use_fp16", cfg.get("use_fp16", False)))
        
        # Create and load classifier
        classifier_args = {
            "image_size": image_size,
            "classifier_use_fp16": classifier_use_fp16,
            "classifier_width": 128,
            "classifier_depth": classifier_depth,
            "classifier_attention_resolutions": "32,16,8",
            "classifier_use_scale_shift_norm": True,
            "classifier_resblock_updown": True,
            "classifier_pool": "attention",
        }
        classifier = create_classifier(**classifier_args)
        classifier.load_state_dict(th.load(classifier_path, map_location="cpu"))
        classifier.to(device)
        if classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
        logger.log(f"[Eval] loaded classifier from {classifier_path}")
        logger.log(f"[Eval] classifier_scale={classifier_scale}, classifier_depth={classifier_depth}")

    ddim_late_correct = bool(cfg.get("ddim_late_correct", False))
    logger.log(f"[Eval] sampler={sampler}, eta={ddim_eta}, late_correct={ddim_late_correct}")
    if sampler == "ddim" and ddim_late_correct:
        logger.log(
            f"[Eval] DDIM corrector: strength={cfg.get('ddim_corrector_strength', 0.5)}, "
            f"last_k_steps={cfg.get('ddim_corrector_apply_last_k_steps', 2)}, "
            f"low_noise_threshold={cfg.get('ddim_corrector_low_noise_threshold_index', 320)}, "
            f"lambda_clip={cfg.get('ddim_corrector_lambda_clip', 0.03)}, "
            f"inner_steps_low_noise={cfg.get('ddim_corrector_inner_steps_low_noise', 1)}, "
            f"use_noise_scale={cfg.get('ddim_corrector_use_noise_scale', True)}, "
            f"fm_t_start={cfg.get('ddim_corrector_fm_t_start', 0.7)}, "
            f"use_adaptive_lambda={cfg.get('ddim_corrector_use_adaptive_lambda', True)}, "
            f"debug={cfg.get('ddim_corrector_debug', False)}"
        )
    logger.log(f"[Eval] num_samples={num_samples}, batch_size={batch_size}, device={device}")
    if sampler == "acdms":
        logger.log(
            "[Eval][ACDMS] input "
            f"fm_strength={cfg.get('acdms_fm_strength', 0.04)}, "
            f"fm_t_start={cfg.get('acdms_fm_t_start', 0.8)}, "
            f"low_noise_thr={cfg.get('acdms_low_noise_threshold_index', 150)}, "
            f"last_k={cfg.get('acdms_apply_last_k_steps', 1)}, "
            f"adaptive_low_noise={cfg.get('acdms_adaptive_low_noise', True)}, "
            f"adaptive_fm_strength={cfg.get('acdms_adaptive_fm_strength', True)}, "
            f"adaptive_last_k={cfg.get('acdms_adaptive_last_k', True)}"
        )

    # 预计算所有时间步的控制参数
    precomputed_controls = {}
    if not use_teacher_model and use_step_policies and g_star_obj is not None:
        for t_scalar in tau_list:
            precomputed_controls[t_scalar] = gstar_to_policy_controls_for_scalar_t(
                g_star_obj,
                tau_list,
                t_scalar,
                allow_hard_skip=bool(cfg.get("allow_hard_skip", True)),
            )

    def model_fn(x, t, y=None):
        if use_teacher_model:
            if class_cond:
                assert y is not None
                return model(x, t, y=y)
            return model(x, t)
        # 从预计算的控制参数中获取
        t_scalar = int(t[0].item())
        controls = precomputed_controls.get(t_scalar, {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0})
        if class_cond:
            assert y is not None
            return model(
                x,
                t,
                y=y,
                skip_layer=controls["skip_layers"],
                layer_gate_map=controls["layer_gate_map"],
                class_gain=controls["class_gain"],
            )
        return model(
            x,
            t,
            skip_layer=controls["skip_layers"],
            layer_gate_map=controls["layer_gate_map"],
            class_gain=controls["class_gain"],
        )

    def cond_fn(x, t, y=None):
        """Classifier guidance function for ADM-G"""
        assert y is not None and classifier is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[range(len(logits)), y]
            grad = th.autograd.grad(selected_log_probs.sum(), x_in)[0]
            return grad * classifier_scale

    all_images: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total = 0

    # 初始化延迟累加器
    lat_text_sum = 0.0
    lat_sample_wall_sum = 0.0
    lat_sample_gpu_sum = 0.0
    lat_decode_sum = 0.0
    lat_eval_sum = 0.0
    lat_count = 0

    if th.cuda.is_available() and str(device).startswith("cuda"):
        th.cuda.synchronize()
    sample_start = time.time()

    while total < num_samples:
        cur_bs = min(batch_size, num_samples - total)
        model_kwargs: Dict[str, Any] = {}
        
        # ---- A) 文本编码（ADM中如果没有文本编码，可以跳过） ----
        text_start = time.time()
        if class_cond:
            classes = th.from_numpy(class_schedule[total : total + cur_bs]).to(device=device, dtype=th.long)
            model_kwargs["y"] = classes
        text_end = time.time()
        text_sec = text_end - text_start
        
        # ---- B) 采样 ----
        if th.cuda.is_available() and str(device).startswith("cuda"):
            th.cuda.synchronize()
        sample_wall_start = time.time()
        
        if sampler == "acdms":
            acdms_result = sample_with_adm_acdms(
                student=model,
                teacher_defaults=teacher_defaults,
                tau_list=tau_list,
                g_star_obj=g_star_for_sampling,
                shape=(cur_bs, 3, image_size, image_size),
                device=device,
                class_labels=classes if class_cond else None,
                clip_denoised=clip_denoised,
                fm_strength=float(cfg.get("acdms_fm_strength", 0.04)),
                fm_t_start=float(cfg.get("acdms_fm_t_start", 0.8)),
                fm_lambda_clip=float(cfg.get("acdms_fm_lambda_clip", 0.9)),
                dynamic_thresholding_ratio=float(cfg.get("acdms_threshold_ratio", 0.995)),
                thresholding_max_val=float(cfg.get("acdms_threshold_max_val", 1.0)),
                low_noise_threshold_index=int(cfg.get("acdms_low_noise_threshold_index", 150)),
                fm_apply_last_k_steps=int(cfg.get("acdms_apply_last_k_steps", 1)),
                adaptive_low_noise=bool(cfg.get("acdms_adaptive_low_noise", True)),
                adaptive_fm_strength=bool(cfg.get("acdms_adaptive_fm_strength", True)),
                adaptive_last_k=bool(cfg.get("acdms_adaptive_last_k", True)),
                low_noise_step_ratio=float(cfg.get("acdms_low_noise_step_ratio", 0.34)),
                low_noise_margin=int(cfg.get("acdms_low_noise_margin", 16)),
                return_meta=(total == 0),
                allow_hard_skip=bool(cfg.get("allow_hard_skip", True)),
                cond_fn=cond_fn if classifier is not None else None,
            )
            if isinstance(acdms_result, tuple):
                sample, acdms_meta = acdms_result
                logger.log(
                    "[Eval][ACDMS] resolved "
                    f"fm_strength={acdms_meta['resolved_fm_strength']:.6f}, "
                    f"low_noise_steps={acdms_meta['low_noise_steps']}, "
                    f"low_noise_thr={acdms_meta['resolved_low_noise_threshold_index']}, "
                    f"last_k={acdms_meta['resolved_fm_apply_last_k_steps']}"
                )
            else:
                sample = acdms_result
        else:
            sample_fn = diffusion.ddim_sample_loop if sampler == "ddim" else diffusion.p_sample_loop
            sample_kwargs = {
                "model": model_fn,
                "shape": (cur_bs, 3, image_size, image_size),
                "clip_denoised": clip_denoised,
                "model_kwargs": model_kwargs,
                "device": device,
                "progress": False,
                "cond_fn": cond_fn if classifier is not None else None,
            }
            if sampler == "ddim":
                sample_kwargs["eta"] = ddim_eta
                sample_kwargs["ddim_late_correct"] = bool(cfg.get("ddim_late_correct", False))
                if sample_kwargs["ddim_late_correct"]:
                    sample_kwargs["ddim_corrector_strength"] = float(cfg.get("ddim_corrector_strength", 0.5))
                    sample_kwargs["ddim_corrector_apply_last_k_steps"] = int(cfg.get("ddim_corrector_apply_last_k_steps", 2))
                    sample_kwargs["ddim_corrector_low_noise_threshold_index"] = int(cfg.get("ddim_corrector_low_noise_threshold_index", 320))
                    sample_kwargs["ddim_corrector_lambda_clip"] = float(cfg.get("ddim_corrector_lambda_clip", 0.03))
                    sample_kwargs["ddim_corrector_inner_steps_low_noise"] = int(cfg.get("ddim_corrector_inner_steps_low_noise", 1))
                    sample_kwargs["ddim_corrector_use_noise_scale"] = bool(cfg.get("ddim_corrector_use_noise_scale", True))
                    sample_kwargs["ddim_corrector_fm_t_start"] = float(cfg.get("ddim_corrector_fm_t_start", 0.7))
                    sample_kwargs["ddim_corrector_use_adaptive_lambda"] = bool(cfg.get("ddim_corrector_use_adaptive_lambda", True))
                    sample_kwargs["ddim_corrector_debug"] = bool(cfg.get("ddim_corrector_debug", False))
            else:
                sample_kwargs["eta"] = 0.0
            sample = sample_fn(**sample_kwargs)
        
        if th.cuda.is_available() and str(device).startswith("cuda"):
            th.cuda.synchronize()
        sample_wall_end = time.time()
        sample_wall_sec = sample_wall_end - sample_wall_start
        
        # 对于ADM，我们使用wall time作为GPU time的近似
        sample_gpu_sec = sample_wall_sec
        
        # ---- C) 后处理（对应SD的decode步骤） ----
        if th.cuda.is_available() and str(device).startswith("cuda"):
            th.cuda.synchronize()
        decode_start = time.time()
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        
        if th.cuda.is_available() and str(device).startswith("cuda"):
            th.cuda.synchronize()
        decode_end = time.time()
        decode_sec = decode_end - decode_start
        
        all_images.append(sample)
        if class_cond:
            all_labels.append(classes.cpu().numpy())
        total += cur_bs
        logger.log(f"[Eval] generated {total}/{num_samples}")
        
        # 累加延迟
        lat_text_sum += text_sec
        lat_sample_wall_sum += sample_wall_sec
        lat_sample_gpu_sum += sample_gpu_sec
        lat_decode_sum += decode_sec
        lat_count += cur_bs

    if th.cuda.is_available() and str(device).startswith("cuda"):
        th.cuda.synchronize()
    sample_time = time.time() - sample_start

    arr = np.concatenate(all_images, axis=0)[:num_samples]
    images01 = th.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    # ---- D) 评估 ----
    if th.cuda.is_available() and str(device).startswith("cuda"):
        th.cuda.synchronize()
    eval_start = time.time()
    
    fid = _compute_fid(images01, cfg, batch_size=batch_size, use_fp16=use_fp16, device=str(device))
    
    if th.cuda.is_available() and str(device).startswith("cuda"):
        th.cuda.synchronize()
    eval_end = time.time()
    eval_sec = eval_end - eval_start
    fid_time = eval_sec
    
    # 累加评估延迟
    lat_eval_sum += eval_sec
    
    # 计算每图片平均延迟
    lat_count = max(1, lat_count)
    lat_text = lat_text_sum / lat_count
    lat_sample_wall = lat_sample_wall_sum / lat_count
    lat_sample_gpu = lat_sample_gpu_sum / lat_count
    lat_decode = lat_decode_sum / lat_count
    lat_eval = lat_eval_sum / lat_count
    lat_total = lat_text + lat_sample_wall + lat_decode + lat_eval
    
    latency_ms_per_img = float(sample_time / max(1, num_samples) * 1000.0)
    metrics = {
        "fid": float(fid),
        "sample_time_sec": float(sample_time),
        "fid_time_sec": float(fid_time),
        "latency_ms_per_img": latency_ms_per_img,
        "latency_text_sec_per_img": float(lat_text),
        "latency_sample_wall_sec_per_img": float(lat_sample_wall),
        "latency_sample_gpu_sec_per_img": float(lat_sample_gpu),
        "latency_decode_sec_per_img": float(lat_decode),
        "latency_eval_sec_per_img": float(lat_eval),
        "latency_total_sec_per_img": float(lat_total),
        "throughput_img_per_sec_total": float(1.0 / max(1e-12, lat_total)),
        "throughput_img_per_sec_sample_only_wall": float(1.0 / max(1e-12, lat_sample_wall)),
        "throughput_img_per_sec_sample_only_gpu": float(1.0 / max(1e-12, lat_sample_gpu)),
        "latency_note": "text=class conditioning (wall), sample=sampling process (wall+gpu-event), decode=post-processing (wall), eval=FID calculation (wall)",
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
        "sampler": str(sampler),
        "ddim_eta": float(ddim_eta),
        "class_sampling_mode": str(class_sampling_mode),
        "model_source": str(model_source),
        "student_ckpt": None if student_ckpt is None else str(student_ckpt),
        "timesteps": [int(x) for x in tau_list],
        "use_best_arch": use_best_arch,
        "use_step_policies": use_step_policies,
        "allow_hard_skip": bool(cfg.get("allow_hard_skip", True)),
        "classifier_scale": classifier_scale,
        "classifier_path": cfg.get("classifier_path"),
        "noise_schedule": teacher_defaults["noise_schedule"],
        "ddim_late_correct": ddim_late_correct,
        "corrector_config": {k: v for k, v in cfg.items() if k.startswith("ddim_corrector_")},
    }

    if bool(cfg.get("save_npz", True)):
        out_path = os.path.join(save_dir, f"samples_{arr.shape[0]}x{image_size}x{image_size}x3.npz")
        if class_cond:
            label_arr = np.concatenate(all_labels, axis=0)[:num_samples]
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        metrics["samples_npz"] = out_path
        logger.log(f"[Eval] saved npz to {out_path}")

    if bool(cfg.get("save_pngs", False)):
        _save_preview_pngs(arr, save_dir, int(cfg.get("max_pngs", 16)))
        logger.log(f"[Eval] saved preview PNGs to {os.path.join(save_dir, 'preview_pngs')}")

    metrics_path = Path(save_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.log(f"[Eval] metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
