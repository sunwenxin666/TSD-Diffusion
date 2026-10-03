import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.utils import save_image


guided_diffusion_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if guided_diffusion_pkg_path not in sys.path:
    sys.path.insert(0, guided_diffusion_pkg_path)

from guided_diffusion import logger
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_classifier,
    create_model_and_diffusion,
)
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

DEFAULT_SAMPLE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "sample_student_distill_imagenet64_adm.config.yaml",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample from a distillation-trained ADM student with ADM-G classifier guidance.")
    parser.add_argument("--config", default=DEFAULT_SAMPLE_CONFIG_PATH)
    parser.add_argument("--train_config", default=None)
    parser.add_argument("--use_teacher_model", type=_str2bool, default=None)
    parser.add_argument("--teacher_backend", choices=["dynamic", "official"], default=None)
    parser.add_argument("--teacher_use_searched_timesteps", type=_str2bool, default=None)
    parser.add_argument("--timestep_respacing", default=None)
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
    parser.add_argument("--ddim_late_correct", type=_str2bool, default=None)
    parser.add_argument("--ddim_corrector_strength", type=float, default=None)
    parser.add_argument("--ddim_corrector_apply_last_k_steps", type=int, default=None)
    parser.add_argument("--ddim_corrector_low_noise_threshold_index", type=int, default=None)
    parser.add_argument("--ddim_corrector_lambda_clip", type=float, default=None)
    parser.add_argument("--ddim_corrector_inner_steps_low_noise", type=int, default=None)
    parser.add_argument("--ddim_corrector_use_noise_scale", type=_str2bool, default=None)
    parser.add_argument("--ddim_corrector_fm_t_start", type=float, default=None)
    parser.add_argument("--ddim_corrector_use_adaptive_lambda", type=_str2bool, default=None)
    parser.add_argument("--ddim_corrector_debug", type=_str2bool, default=None)
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
    parser.add_argument("--save_pngs", type=_str2bool, default=None)
    parser.add_argument("--max_pngs", type=int, default=None)
    parser.add_argument("--class_cond", type=_str2bool, default=None)
    parser.add_argument("--class_sampling_mode", choices=["random", "uniform"], default=None)
    parser.add_argument("--allow_hard_skip", type=_str2bool, default=None)
    parser.add_argument("--classifier_path", default=None)
    parser.add_argument("--classifier_scale", type=float, default=None)
    parser.add_argument("--classifier_depth", type=int, default=None)
    parser.add_argument("--classifier_use_fp16", type=_str2bool, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    sample_config_path = os.path.abspath(args.config)
    sample_cfg = _normalize_cfg_paths(load_config_file(sample_config_path), sample_config_path)
    train_config_path = args.train_config or sample_cfg.get("train_config")
    if not train_config_path:
        raise ValueError("Missing train_config. Set it in the sample config or pass --train_config.")
    train_config_path = _resolve_path(train_config_path, sample_config_path)
    train_cfg = _normalize_cfg_paths(load_config_file(train_config_path), train_config_path)

    cfg = dict(train_cfg)
    cfg.update(sample_cfg)
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg.update(cli_overrides)

    sampler = cfg.get("sampler", "ddim")
    use_teacher_model = bool(cfg.get("use_teacher_model", False))
    student_ckpt = None
    if not use_teacher_model:
        student_ckpt = cfg.get("student_ckpt") or _default_student_ckpt(cfg["outdir"])
        student_ckpt = _resolve_path(student_ckpt, sample_config_path)
    default_save_subdir = f"{'teacher' if use_teacher_model else 'samples'}_{sampler}"
    save_dir = cfg.get("save_dir") or os.path.join(cfg["outdir"], default_save_subdir)
    save_dir = _resolve_path(save_dir, sample_config_path)
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

    teacher_model_overrides = _teacher_model_overrides_from_cfg(cfg if use_teacher_model else train_cfg)
    teacher_backend = str(cfg.get("teacher_backend", "dynamic")).lower()
    teacher_use_searched_timesteps = bool(cfg.get("teacher_use_searched_timesteps", True))
    if use_teacher_model and teacher_backend == "official":
        teacher_defaults = model_and_diffusion_defaults()
        supported_teacher_keys = set(teacher_defaults.keys())
        teacher_defaults.update(
            {k: v for k, v in teacher_model_overrides.items() if k in supported_teacher_keys}
        )
        teacher_defaults["use_fp16"] = use_fp16
        teacher_defaults["timestep_respacing"] = str(cfg.get("timestep_respacing") or "ddim100")
        teacher_defaults["use_dynamic_unet"] = False
        logger.log(
            "[Sample][official teacher args] "
            f"image_size={teacher_defaults['image_size']}, "
            f"num_channels={teacher_defaults['num_channels']}, "
            f"num_res_blocks={teacher_defaults['num_res_blocks']}, "
            f"attention_resolutions={teacher_defaults['attention_resolutions']}, "
            f"channel_mult={teacher_defaults['channel_mult']!r}, "
            f"num_heads={teacher_defaults['num_heads']}, "
            f"num_head_channels={teacher_defaults['num_head_channels']}, "
            f"learn_sigma={teacher_defaults['learn_sigma']}, "
            f"class_cond={teacher_defaults['class_cond']}, "
            f"noise_schedule={teacher_defaults['noise_schedule']}, "
            f"resblock_updown={teacher_defaults['resblock_updown']}, "
            f"use_scale_shift_norm={teacher_defaults['use_scale_shift_norm']}, "
            f"use_new_attention_order={teacher_defaults['use_new_attention_order']}, "
            f"timestep_respacing={teacher_defaults['timestep_respacing']}"
        )
        teacher, diffusion = create_model_and_diffusion(**teacher_defaults)
        teacher.load_state_dict(_extract_state_dict(th.load(cfg["ckpt"], map_location="cpu")), strict=True)
        teacher.to(device)
        if use_fp16 and hasattr(teacher, "convert_to_fp16"):
            teacher.convert_to_fp16()
        teacher.eval()
        model = teacher
        model_source = "teacher_official"
        g_star_for_sampling = None
    else:
        teacher, _, teacher_defaults = build_teacher_and_diffusion(
            ckpt_path=cfg["ckpt"],
            device=device,
            use_fp16=use_fp16,
            model_overrides=teacher_model_overrides,
        )
        if use_teacher_model:
            teacher.eval()
            model = teacher
            model_source = "teacher_dynamic"
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

    if use_teacher_model and teacher_backend == "official" and teacher_use_searched_timesteps:
        diffusion = _build_sampling_diffusion(teacher_defaults, tau_list)
    elif not (use_teacher_model and teacher_backend == "official"):
        diffusion = _build_sampling_diffusion(teacher_defaults, tau_list)
    ddim_eta = float(cfg["ddim_eta"]) if cfg.get("ddim_eta") is not None else _sampler_eta(g_star_obj)
    image_size = int(teacher_defaults["image_size"])
    class_cond = bool(cfg.get("class_cond", teacher_defaults.get("class_cond", True)))
    num_classes = int(teacher_defaults.get("num_classes", 1000) or 1000) if class_cond else 0
    logger.log(f"[Sample] model_source={model_source}")
    if use_teacher_model:
        logger.log(f"[Sample] teacher_backend={teacher_backend}")
        if teacher_backend == "official":
            logger.log(f"[Sample] teacher_use_searched_timesteps={teacher_use_searched_timesteps}")
            if not teacher_use_searched_timesteps:
                logger.log(f"[Sample] timestep_respacing={teacher_defaults['timestep_respacing']}")
    logger.log(f"[Sample] use_best_arch={use_best_arch}, use_step_policies={use_step_policies}")
    logger.log(f"[Sample] allow_hard_skip={bool(cfg.get('allow_hard_skip', True))}")
    if student_ckpt is not None:
        logger.log(f"[Sample] ckpt={student_ckpt}")
    else:
        logger.log(f"[Sample] ckpt={cfg['ckpt']}")
    if use_teacher_model and teacher_backend == "official" and not teacher_use_searched_timesteps:
        logger.log("[Sample] timesteps=official timestep_respacing, best_timesteps_npy is not used")
    else:
        logger.log(f"[Sample] timesteps={tau_list}")

    classifier = None
    classifier_scale = float(cfg.get("classifier_scale", 1.0))
    if classifier_scale > 0:
        classifier_path = cfg.get("classifier_path")
        if not classifier_path:
            raise ValueError("classifier_path is required when classifier_scale > 0")
        classifier_depth = int(cfg.get("classifier_depth", 4))
        classifier_use_fp16 = bool(cfg.get("classifier_use_fp16", cfg.get("use_fp16", False)))

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
        logger.log(f"[Sample] loaded classifier from {classifier_path}")
        logger.log(f"[Sample] classifier_scale={classifier_scale}, classifier_depth={classifier_depth}")

    logger.log(f"[Sample] sampler={sampler}, eta={ddim_eta}")
    if sampler == "ddim":
        logger.log(
            "[Sample][DDIM-FM-CORR] "
            f"enabled={bool(cfg.get('ddim_late_correct', False))}, "
            f"strength={cfg.get('ddim_corrector_strength', 0.01)}, "
            f"last_k={cfg.get('ddim_corrector_apply_last_k_steps', 2)}, "
            f"low_noise_thr={cfg.get('ddim_corrector_low_noise_threshold_index', 320)}, "
            f"lambda_clip={cfg.get('ddim_corrector_lambda_clip', 0.03)}, "
            f"inner_steps={cfg.get('ddim_corrector_inner_steps_low_noise', 1)}, "
            f"use_noise_scale={cfg.get('ddim_corrector_use_noise_scale', True)}, "
            f"fm_t_start={cfg.get('ddim_corrector_fm_t_start', 0.7)}, "
            f"adaptive_lambda={cfg.get('ddim_corrector_use_adaptive_lambda', True)}, "
            f"debug={cfg.get('ddim_corrector_debug', False)}"
        )
    if sampler == "acdms":
        logger.log(
            "[Sample][ACDMS] input "
            f"fm_strength={cfg.get('acdms_fm_strength', 0.04)}, "
            f"fm_t_start={cfg.get('acdms_fm_t_start', 0.8)}, "
            f"low_noise_thr={cfg.get('acdms_low_noise_threshold_index', 150)}, "
            f"last_k={cfg.get('acdms_apply_last_k_steps', 1)}, "
            f"adaptive_low_noise={cfg.get('acdms_adaptive_low_noise', True)}, "
            f"adaptive_fm_strength={cfg.get('acdms_adaptive_fm_strength', True)}, "
            f"adaptive_last_k={cfg.get('acdms_adaptive_last_k', True)}"
        )

    def model_fn(x, t, y=None):
        if use_teacher_model:
            if class_cond:
                assert y is not None
                return model(x, t, y=y)
            return model(x, t)
        controls = {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0}
        if use_step_policies and g_star_obj is not None:
            t_scalar = int(t[0].item())
            controls = gstar_to_policy_controls_for_scalar_t(
                g_star_obj,
                tau_list,
                t_scalar,
                allow_hard_skip=bool(cfg.get("allow_hard_skip", True)),
            )
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
        assert y is not None and classifier is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    all_images: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total = 0
    num_samples = int(cfg.get("num_samples", 50000))
    batch_size = int(cfg.get("batch_size", 16))
    clip_denoised = bool(cfg.get("clip_denoised", True))
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
            "[Sample] class sampling "
            f"mode={class_sampling_mode}, "
            f"min_per_class={int(class_counts.min())}, "
            f"max_per_class={int(class_counts.max())}"
        )

    while total < num_samples:
        cur_bs = min(batch_size, num_samples - total)
        model_kwargs: Dict[str, Any] = {}
        if class_cond:
            classes = th.from_numpy(class_schedule[total : total + cur_bs]).to(device=device, dtype=th.long)
            model_kwargs["y"] = classes

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
            )
            if isinstance(acdms_result, tuple):
                sample, acdms_meta = acdms_result
                logger.log(
                    "[Sample][ACDMS] resolved "
                    f"fm_strength={acdms_meta['resolved_fm_strength']:.6f}, "
                    f"low_noise_steps={acdms_meta['low_noise_steps']}, "
                    f"low_noise_thr={acdms_meta['resolved_low_noise_threshold_index']}, "
                    f"last_k={acdms_meta['resolved_fm_apply_last_k_steps']}"
                )
            else:
                sample = acdms_result

        elif sampler == "ddim":
            sample = diffusion.ddim_sample_loop(
                model_fn,
                (cur_bs, 3, image_size, image_size),
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn if classifier is not None else None,
                device=device,
                progress=False,
                eta=ddim_eta,
                ddim_late_correct=bool(cfg.get("ddim_late_correct", False)),
                ddim_corrector_strength=float(cfg.get("ddim_corrector_strength", 0.01)),
                ddim_corrector_apply_last_k_steps=int(cfg.get("ddim_corrector_apply_last_k_steps", 2)),
                ddim_corrector_low_noise_threshold_index=int(cfg.get("ddim_corrector_low_noise_threshold_index", 320)),
                ddim_corrector_lambda_clip=float(cfg.get("ddim_corrector_lambda_clip", 0.03)),
                ddim_corrector_inner_steps_low_noise=int(cfg.get("ddim_corrector_inner_steps_low_noise", 1)),
                ddim_corrector_use_noise_scale=bool(cfg.get("ddim_corrector_use_noise_scale", True)),
                ddim_corrector_fm_t_start=float(cfg.get("ddim_corrector_fm_t_start", 0.7)),
                ddim_corrector_use_adaptive_lambda=bool(cfg.get("ddim_corrector_use_adaptive_lambda", True)),
                ddim_corrector_debug=bool(cfg.get("ddim_corrector_debug", False)),
            )

        else:
            sample = diffusion.p_sample_loop(
                model_fn,
                (cur_bs, 3, image_size, image_size),
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn if classifier is not None else None,
                device=device,
                progress=False,
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        all_images.append(sample)
        if class_cond:
            all_labels.append(classes.cpu().numpy())
        total += cur_bs
        logger.log(f"[Sample] generated {total}/{num_samples}")

    arr = np.concatenate(all_images, axis=0)[:num_samples]
    out_path = os.path.join(save_dir, f"samples_{arr.shape[0]}x{image_size}x{image_size}x3.npz")
    if class_cond:
        label_arr = np.concatenate(all_labels, axis=0)[:num_samples]
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)
    logger.log(f"[Sample] saved npz to {out_path}")

    if bool(cfg.get("save_pngs", False)):
        _save_preview_pngs(arr, save_dir, int(cfg.get("max_pngs", 64)))
        logger.log(f"[Sample] saved preview PNGs to {os.path.join(save_dir, 'preview_pngs')}")


if __name__ == "__main__":
    main()
