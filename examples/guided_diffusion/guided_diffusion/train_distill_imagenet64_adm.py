import copy
import os
import json
import math
import pickle
import random
import sys
import argparse
import importlib
import warnings
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Suppress PyTorch FutureWarning about torch.load
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import numpy as np
import torch as th
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Make the top-level `guided_diffusion` package importable when running this
# file directly via `python .../train_distill_imagenet64_adm.py`.
guided_diffusion_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if guided_diffusion_pkg_path not in sys.path:
    sys.path.insert(0, guided_diffusion_pkg_path)

nas_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "nas"))
if nas_pkg_path not in sys.path:
    sys.path.insert(0, nas_pkg_path)

# Older NAS pickles were serialized with top-level module names like `arch_space`.
# Register the current module under that legacy name so pickle can resolve it.
sys.modules.setdefault("arch_space", importlib.import_module("guided_diffusion.nas.arch_space"))

# === guided-diffusion core ===
from guided_diffusion import dist_util
from guided_diffusion import logger
from guided_diffusion.nn import update_ema
from guided_diffusion.script_util import model_and_diffusion_defaults, create_gaussian_diffusion
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.dynamic_unet import Dynamic_UNetModel as DynamicUNetModel  # 支持 skip_layer
from guided_diffusion.losses import (
    distill_adm_consistency_loss,
    distill_adm_hybrid_loss,
    distill_adm_sampler_rollout_loss,
)
from guided_diffusion.adm_acdms import sample_with_adm_acdms


# -----------------------------
# utils
# -----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def load_pickle(p: Optional[str]):
    if p is None:
        return None
    p = str(p)
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def load_npy(p: str) -> np.ndarray:
    return np.load(p)


def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config_file(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return load_yaml(path)
    if suffix == ".json":
        return load_json(path)
    raise ValueError(f"Unsupported config format: {path}")


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, th.Tensor]:
    """
    更稳健地从 ckpt 里拿 state_dict。
    guided-diffusion 常见 key：ema / model / state_dict
    优先级：ema > model > state_dict > 直接当作 state_dict
    """
    if isinstance(ckpt_obj, dict):
        for k in ["ema", "model", "state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format.")


def _infer_model_channels(state: Dict[str, th.Tensor]) -> Optional[int]:
    for key in ("time_embed.0.weight", "model.time_embed.0.weight"):
        if key in state:
            return int(state[key].shape[1])
    for key, value in state.items():
        if key.endswith("time_embed.0.weight"):
            return int(value.shape[1])
    return None


def _infer_learn_sigma(state: Dict[str, th.Tensor]) -> Optional[bool]:
    for key in ("out.2.weight", "model.out.2.weight"):
        if key in state:
            return bool(int(state[key].shape[0]) == 6)
    for key, value in state.items():
        if key.endswith("out.2.weight"):
            return bool(int(value.shape[0]) == 6)
    return None


def _infer_class_conditional(state: Dict[str, th.Tensor]) -> bool:
    for key in state.keys():
        if "label_emb" in key or "label_embed" in key:
            return True
    return False


def _infer_num_classes(state: Dict[str, th.Tensor]) -> Optional[int]:
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


def _get_state_tensor_by_suffix(state: Dict[str, th.Tensor], suffix: str) -> Optional[th.Tensor]:
    if suffix in state:
        return state[suffix]
    for key, value in state.items():
        if key.endswith(suffix):
            return value
    return None


def _infer_lsun_bedroom_teacher_overrides(state: Dict[str, th.Tensor], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Detect the released LSUN-Bedroom ADM layout from checkpoint tensor shapes."""
    image_size = int(defaults.get("image_size", 0) or 0)
    model_channels = int(defaults.get("num_channels", 0) or 0)
    class_cond = bool(defaults.get("class_cond", False))
    block5 = _get_state_tensor_by_suffix(state, "input_blocks.5.0.in_layers.2.weight")

    if (
        image_size == 256
        and model_channels > 0
        and not class_cond
        and block5 is not None
        and int(block5.shape[0]) == model_channels
    ):
        return {
            "num_res_blocks": 2,
            "channel_mult": (1, 1, 2, 2, 4, 4),
            "noise_schedule": "linear",
            "rescale_learned_sigmas": False,
            "rescale_timesteps": False,
        }
    return {}


# -----------------------------
# NAS objects compatibility
# -----------------------------
def _arch_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def best_arch_to_model_overrides(best_arch_obj) -> Dict[str, Any]:
    """
    best_arch 常见字段：
      - num_res_blocks
      - attn_ds / attention_resolutions
    转成 attention_resolutions="32,16,8" 的字符串形式（后续再 parse 成 tuple[int]）
    """
    num_res_blocks = _arch_get(best_arch_obj, "num_res_blocks", None)
    attn_ds = _arch_get(best_arch_obj, "attn_ds", None)
    if attn_ds is None:
        attn_ds = _arch_get(best_arch_obj, "attention_resolutions", None)

    overrides: Dict[str, Any] = {}
    if num_res_blocks is not None:
        overrides["num_res_blocks"] = int(num_res_blocks)

    if attn_ds is not None:
        attn_list = [int(x) for x in list(attn_ds)]
        overrides["attention_resolutions"] = ",".join(str(x) for x in attn_list)

    return overrides


def normalize_attention_resolutions(attn, image_size: int) -> Tuple[int, ...]:
    if isinstance(attn, str):
        values = [int(x) for x in attn.split(",") if x.strip() != ""]
    else:
        values = [int(x) for x in attn]
    if not values:
        return tuple()
    # The original guided-diffusion config uses spatial resolutions like
    # "32,16,8". Dynamic_UNetModel expects downsample factors like (2,4,8).
    if max(values) > 8:
        values = [image_size // x for x in values]
    return tuple(values)


def resolve_channel_mult(channel_mult, image_size: int) -> Tuple[Any, ...]:
    if channel_mult in (None, "", "auto"):
        if image_size == 512:
            return (0.5, 1, 1, 2, 2, 4, 4)
        if image_size == 256:
            return (1, 1, 2, 2, 4, 4)
        if image_size == 128:
            return (1, 1, 2, 3, 4)
        if image_size == 64:
            return (1, 2, 3, 4)
        if image_size == 32:
            return (1, 2, 2, 2)
        raise ValueError(f"unsupported image size: {image_size}")
    if isinstance(channel_mult, str):
        out = []
        for item in channel_mult.split(","):
            item = item.strip()
            if not item:
                continue
            value = float(item)
            out.append(int(value) if value.is_integer() else value)
        return tuple(out)
    return tuple(channel_mult)


def tau_to_step_index(t_scalar: int, tau_list: List[int]) -> int:
    tau_int = [int(x) for x in tau_list]
    t_scalar = int(t_scalar)
    if t_scalar in tau_int:
        return tau_int.index(t_scalar)
    diffs = [abs(x - t_scalar) for x in tau_int]
    return int(np.argmin(diffs))


def gstar_to_skip_layers_for_scalar_t(
    g_star_obj, tau_list: List[int], t_scalar: int, allow_hard_skip: bool = True
) -> List[int]:
    """
    输入单个 t_scalar，返回该 t 对应 policy 的 skip_layers（全局层索引列表）。
    """
    controls = gstar_to_policy_controls_for_scalar_t(
        g_star_obj,
        tau_list,
        t_scalar,
        allow_hard_skip=allow_hard_skip,
    )
    return list(controls["skip_layers"])


def gstar_to_policy_controls_for_scalar_t(
    g_star_obj,
    tau_list: List[int],
    t_scalar: int,
    allow_hard_skip: bool = True,
) -> Dict[str, Any]:
    """
    将某个离散 t 对应的 NAS policy 统一解析成：
    - skip_layers: List[int]
    - layer_gate_map: Dict[int, float]
    - class_gain: float
    """
    if g_star_obj is None:
        return {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0}
    step_policies = _arch_get(g_star_obj, "step_policies", None)
    if not step_policies:
        return {"skip_layers": [], "layer_gate_map": {}, "class_gain": 1.0}

    si = tau_to_step_index(int(t_scalar), tau_list)
    si = max(0, min(si, len(step_policies) - 1))
    pol = step_policies[si]

    layer_mask = _arch_get(pol, "layer_mask", None)
    if layer_mask is None and isinstance(pol, dict):
        layer_mask = pol.get("layer_mask", None)
    if layer_mask is not None:
        mask = [bool(x) for x in layer_mask]
        skip_layers = [i for i, keep in enumerate(mask) if not keep]
    else:
        skip_blocks = _arch_get(pol, "skip_blocks", None)
        if skip_blocks is None and isinstance(pol, dict):
            skip_blocks = pol.get("skip_blocks", None)
        if skip_blocks is None:
            skip_layers = []
        else:
            skip_layers = [int(x) for x in skip_blocks]

    layer_gates = _arch_get(pol, "layer_gates", None)
    if layer_gates is None and isinstance(pol, dict):
        layer_gates = pol.get("layer_gates", None)
    layer_gate_map: Dict[int, float] = {}
    if isinstance(layer_gates, dict):
        for key, value in layer_gates.items():
            layer_gate_map[int(key)] = float(value)
    elif layer_gates is not None:
        for item in layer_gates:
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
            layer_gate_map[int(layer_id)] = float(gate)

    class_gain = _arch_get(pol, "class_gain", None)
    if class_gain is None and isinstance(pol, dict):
        class_gain = pol.get("class_gain", None)
    if class_gain is None:
        class_gain = 1.0

    if not bool(allow_hard_skip):
        skip_layers = []

    return {
        "skip_layers": [int(x) for x in skip_layers],
        "layer_gate_map": {int(k): float(v) for k, v in layer_gate_map.items()},
        "class_gain": float(class_gain),
    }


def summarize_nas_artifacts(best_arch_obj, g_star_obj) -> Dict[str, Any]:
    metrics = dict(_arch_get(g_star_obj, "metrics", {}) or {})
    num_res_blocks = _arch_get(best_arch_obj, "num_res_blocks", None)
    attention_resolutions = _arch_get(best_arch_obj, "attention_resolutions", None)
    step_policies = list(_arch_get(g_star_obj, "step_policies", []) or [])
    non_empty_policies = 0
    for pol in step_policies:
        skip_blocks = _arch_get(pol, "skip_blocks", []) or []
        layer_gates = _arch_get(pol, "layer_gates", []) or []
        class_gain = _arch_get(pol, "class_gain", 1.0)
        if len(skip_blocks) > 0 or len(layer_gates) > 0 or abs(float(class_gain) - 1.0) > 1e-6:
            non_empty_policies += 1
    return {
        "num_res_blocks": None if num_res_blocks is None else int(num_res_blocks),
        "attention_resolutions": None if attention_resolutions is None else list(attention_resolutions),
        "fid": metrics.get("fid", None),
        "fid_raw": metrics.get("fid_raw", metrics.get("fid", None)),
        "latency_ms_per_img": metrics.get("latency_ms_per_img", None),
        "num_steps": len(step_policies),
        "non_empty_step_policies": int(non_empty_policies),
    }


def _teacher_model_overrides_from_cfg(cfg_like) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in [
        "image_size",
        "num_channels",
        "num_res_blocks",
        "learn_sigma",
        "class_cond",
        "num_classes",
        "attention_resolutions",
        "dropout",
        "diffusion_steps",
        "noise_schedule",
        "rescale_learned_sigmas",
        "rescale_timesteps",
        "use_scale_shift_norm",
        "use_checkpoint",
        "channel_mult",
        "num_head_channels",
        "num_heads",
        "num_heads_upsample",
        "resblock_updown",
        "use_new_attention_order",
    ]:
        has_value = key in cfg_like if isinstance(cfg_like, dict) else hasattr(cfg_like, key)
        if has_value:
            value = cfg_like[key] if isinstance(cfg_like, dict) else getattr(cfg_like, key)
            if value is not None:
                overrides[key] = value
    return overrides


# -----------------------------
# build model & diffusion
# -----------------------------
def build_teacher_and_diffusion(
    ckpt_path: str,
    device: str,
    use_fp16: bool,
    model_overrides: Dict[str, Any],
) -> Tuple[nn.Module, SpacedDiffusion, Dict[str, Any]]:
    """
    构造 teacher(UNet) + diffusion，并加载 ckpt。
    用 DynamicUNetModel 是为了让 student/teacher forward 签名一致（支持 skip_layer）。
    """
    defaults = model_and_diffusion_defaults()
    defaults.update({
        "image_size": 64,
        "num_channels": 192,
        "num_res_blocks": 3,
        "learn_sigma": True,
        "class_cond": True,
        "num_classes": 1000,
        "use_fp16": use_fp16,
        "attention_resolutions": "32,16,8",
        "dropout": 0.1,
        "diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "rescale_learned_sigmas": False,
        "rescale_timesteps": False,
        "use_scale_shift_norm": True,
        "use_checkpoint": False,
        "channel_mult": "",
        "num_head_channels": 64,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "resblock_updown": True,
        "use_new_attention_order": True,
    })
    defaults.update(model_overrides or {})

    ckpt_obj = th.load(ckpt_path, map_location="cpu", weights_only=True)
    state = _extract_state_dict(ckpt_obj)
    inferred_overrides: Dict[str, Any] = {}
    inferred_num_channels = _infer_model_channels(state)
    if inferred_num_channels is not None:
        inferred_overrides["num_channels"] = int(inferred_num_channels)
    inferred_learn_sigma = _infer_learn_sigma(state)
    if inferred_learn_sigma is not None:
        inferred_overrides["learn_sigma"] = bool(inferred_learn_sigma)
    inferred_overrides["class_cond"] = bool(_infer_class_conditional(state))
    inferred_num_classes = _infer_num_classes(state)
    if inferred_num_classes is not None:
        inferred_overrides["num_classes"] = int(inferred_num_classes)
    defaults.update(inferred_overrides)
    defaults.update(model_overrides or {})
    defaults.update(_infer_lsun_bedroom_teacher_overrides(state, defaults))

    defaults["channel_mult"] = resolve_channel_mult(
        defaults.get("channel_mult", ""),
        int(defaults["image_size"]),
    )

    # attention_resolutions "32,16,8" -> (32,16,8)
    defaults["attention_resolutions"] = normalize_attention_resolutions(
        defaults.get("attention_resolutions", "32,16,8"),
        int(defaults["image_size"]),
    )
    logger.log(
        "[Teacher] resolved arch "
        f"image_size={defaults['image_size']}, "
        f"num_channels={defaults['num_channels']}, "
        f"num_res_blocks={defaults['num_res_blocks']}, "
        f"channel_mult={tuple(defaults['channel_mult'])}, "
        f"class_cond={defaults['class_cond']}, "
        f"noise_schedule={defaults['noise_schedule']}"
    )

    model = DynamicUNetModel(
        image_size=int(defaults["image_size"]),
        in_channels=3,
        model_channels=int(defaults["num_channels"]),
        out_channels=6 if bool(defaults["learn_sigma"]) else 3,
        num_res_blocks=int(defaults["num_res_blocks"]),
        attention_resolutions=defaults["attention_resolutions"],
        dropout=float(defaults["dropout"]),
        channel_mult=tuple(defaults["channel_mult"]),
        num_classes=int(defaults["num_classes"]) if bool(defaults["class_cond"]) else None,
        use_checkpoint=bool(defaults["use_checkpoint"]),
        use_fp16=bool(defaults["use_fp16"]),
        num_heads=int(defaults["num_heads"]),
        num_head_channels=int(defaults["num_head_channels"]),
        num_heads_upsample=int(defaults["num_heads_upsample"]),
        use_scale_shift_norm=bool(defaults["use_scale_shift_norm"]),
        resblock_updown=bool(defaults["resblock_updown"]),
        use_new_attention_order=bool(defaults["use_new_attention_order"]),
    )

    diffusion = create_gaussian_diffusion(
        steps=int(defaults["diffusion_steps"]),
        learn_sigma=bool(defaults["learn_sigma"]),
        noise_schedule=str(defaults["noise_schedule"]),
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=bool(defaults["rescale_timesteps"]),
        rescale_learned_sigmas=bool(defaults["rescale_learned_sigmas"]),
        timestep_respacing="",
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.log(f"[Teacher] load ckpt done. missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0 or len(unexpected) > 0:
        save_json(
            {"missing_keys": missing, "unexpected_keys": unexpected, "defaults": _jsonify(defaults)},
            os.path.join("ckpt_load_report_teacher.json"),
        )

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, diffusion, defaults


def build_student_from_teacher(
    teacher: nn.Module,
    teacher_defaults: Dict[str, Any],
    best_arch_obj,
    device: str,
    use_fp16: bool,
) -> nn.Module:
    """
    student = teacher_defaults 基底 + best_arch 覆盖（结构剪枝）
    然后按 shape-match 从 teacher 拷贝权重，剩余随机初始化。
    """
    overrides = best_arch_to_model_overrides(best_arch_obj)
    defaults = dict(teacher_defaults)
    defaults.update(overrides)

    defaults["attention_resolutions"] = normalize_attention_resolutions(
        defaults.get("attention_resolutions", (32, 16, 8)),
        int(defaults["image_size"]),
    )

    model = DynamicUNetModel(
        image_size=int(defaults["image_size"]),
        in_channels=3,
        model_channels=int(defaults["num_channels"]),
        out_channels=6 if bool(defaults["learn_sigma"]) else 3,
        num_res_blocks=int(defaults["num_res_blocks"]),
        attention_resolutions=defaults["attention_resolutions"],
        dropout=float(defaults["dropout"]),
        channel_mult=tuple(defaults["channel_mult"]),
        num_classes=int(defaults["num_classes"]) if bool(defaults["class_cond"]) else None,
        use_checkpoint=bool(defaults["use_checkpoint"]),
        use_fp16=use_fp16,
        num_heads=int(defaults["num_heads"]),
        num_head_channels=int(defaults["num_head_channels"]),
        num_heads_upsample=int(defaults["num_heads_upsample"]),
        use_scale_shift_norm=bool(defaults["use_scale_shift_norm"]),
        resblock_updown=bool(defaults["resblock_updown"]),
        use_new_attention_order=bool(defaults["use_new_attention_order"]),
    )

    dst = model.state_dict()
    src = teacher.state_dict()
    matched = 0
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            matched += 1
    model.load_state_dict(dst, strict=False)
    logger.log(f"[Student] built. shape-matched keys={matched}/{len(dst)}")

    model.to(device)
    model.train()
    return model


def build_student_isomorphic_from_teacher(
    teacher: nn.Module,
    teacher_defaults: Dict[str, Any],
    device: str,
    use_fp16: bool,
) -> nn.Module:
    """
    不使用 NAS best_arch 时，构建与 teacher 同构的 student，
    并完整继承 teacher 权重。
    """
    defaults = dict(teacher_defaults)
    model = DynamicUNetModel(
        image_size=int(defaults["image_size"]),
        in_channels=3,
        model_channels=int(defaults["num_channels"]),
        out_channels=6 if bool(defaults["learn_sigma"]) else 3,
        num_res_blocks=int(defaults["num_res_blocks"]),
        attention_resolutions=tuple(defaults["attention_resolutions"]),
        dropout=float(defaults["dropout"]),
        channel_mult=tuple(defaults["channel_mult"]),
        num_classes=int(defaults["num_classes"]) if bool(defaults["class_cond"]) else None,
        use_checkpoint=bool(defaults["use_checkpoint"]),
        use_fp16=use_fp16,
        num_heads=int(defaults["num_heads"]),
        num_head_channels=int(defaults["num_head_channels"]),
        num_heads_upsample=int(defaults["num_heads_upsample"]),
        use_scale_shift_norm=bool(defaults["use_scale_shift_norm"]),
        resblock_updown=bool(defaults["resblock_updown"]),
        use_new_attention_order=bool(defaults["use_new_attention_order"]),
    )
    model.load_state_dict(teacher.state_dict(), strict=True)
    logger.log("[Student] built teacher-isomorphic student with strict teacher weight inheritance.")
    model.to(device)
    model.train()
    return model


def _jsonify(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (tuple, list)):
            out[k] = list(v)
        elif isinstance(v, (np.ndarray,)):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# -----------------------------
# training
# -----------------------------
@dataclass
class TrainCfg:
    ckpt: str
    data_dir: str
    outdir: str

    best_timesteps_npy: str
    best_arch_pkl: Optional[str] = None
    g_star_pkl: Optional[str] = None
    use_best_arch: bool = True
    use_step_policies: bool = True
    allow_hard_skip: bool = True
    strict_artifacts: bool = False
    eval_dataset: str = "imagenet64"
    ref_stats_pkl: Optional[str] = None

    image_size: int = 64
    num_channels: int = 192
    num_res_blocks: int = 3
    learn_sigma: bool = True
    class_cond: bool = True
    num_classes: int = 1000
    attention_resolutions: str = "32,16,8"
    dropout: float = 0.1
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"
    rescale_learned_sigmas: bool = False
    rescale_timesteps: bool = False
    use_scale_shift_norm: bool = True
    use_checkpoint: bool = False
    channel_mult: str = ""
    num_head_channels: int = 64
    num_heads: int = 4
    num_heads_upsample: int = -1
    resblock_updown: bool = True
    use_new_attention_order: bool = True

    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_schedule: str = "constant"  # constant / cosine
    lr_warmup_steps: int = 0
    lr_min_scale: float = 0.0
    total_steps: int = 20000
    log_every: int = 50
    save_every: int = 2000
    num_workers: int = 8
    seed: int = 0
    ema_decay: float = 0.0

    use_fp16: bool = True
    device: str = "cuda"

    sampler_type: str = "ddim"  # ddim / acdms
    acdms_fm_strength: float = 0.04
    acdms_fm_t_start: float = 0.8
    acdms_fm_lambda_clip: float = 0.9
    acdms_threshold_ratio: float = 0.995
    acdms_threshold_max_val: float = 1.0
    acdms_low_noise_threshold_index: int = 150
    acdms_apply_last_k_steps: int = 1
    sampler_distill_weight: float = 0.25
    sampler_distill_grad_weight: float = 0.0
    sampler_distill_batch_size: int = 2
    sampler_distill_every: int = 4

    lambda_consistency: float = 0.1
    tau_sample_mode: str = "per_sample"  # per_batch / per_sample
    eps_loss_weight: float = 1.0
    grad_loss_weight: float = 0.05
    xstart_loss_weight: float = 0.25
    var_loss_weight: float = 0.0
    clip_xstart_loss: bool = True
    warn_on_poor_nas: bool = True
    poor_fid_threshold: float = 80.0
    distributed: bool = False
    dist_backend: Optional[str] = None
    dist_init_method: str = "env://"


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "train_distill_imagenet64_adm.config.yaml",
)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--outdir", default=None)

    parser.add_argument("--best_timesteps_npy", default=None)
    parser.add_argument("--best_arch_pkl", default=None)
    parser.add_argument("--g_star_pkl", default=None)
    parser.add_argument("--use_best_arch", type=_str2bool, default=None)
    parser.add_argument("--use_step_policies", type=_str2bool, default=None)
    parser.add_argument("--allow_hard_skip", type=_str2bool, default=None)
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--ref_stats_pkl", default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--num_channels", type=int, default=None)
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--learn_sigma", type=_str2bool, default=None)
    parser.add_argument("--class_cond", type=_str2bool, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--attention_resolutions", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--noise_schedule", default=None)
    parser.add_argument("--rescale_learned_sigmas", type=_str2bool, default=None)
    parser.add_argument("--rescale_timesteps", type=_str2bool, default=None)
    parser.add_argument("--use_scale_shift_norm", type=_str2bool, default=None)
    parser.add_argument("--use_checkpoint", type=_str2bool, default=None)
    parser.add_argument("--channel_mult", default=None)
    parser.add_argument("--num_head_channels", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_heads_upsample", type=int, default=None)
    parser.add_argument("--resblock_updown", type=_str2bool, default=None)
    parser.add_argument("--use_new_attention_order", type=_str2bool, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr_schedule", default=None, choices=["constant", "cosine"])
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--lr_min_scale", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", type=_str2bool, default=None)
    parser.add_argument("--sampler_type", default=None, choices=["ddim", "acdms"])
    parser.add_argument("--acdms_fm_strength", type=float, default=None)
    parser.add_argument("--acdms_fm_t_start", type=float, default=None)
    parser.add_argument("--acdms_fm_lambda_clip", type=float, default=None)
    parser.add_argument("--acdms_threshold_ratio", type=float, default=None)
    parser.add_argument("--acdms_threshold_max_val", type=float, default=None)
    parser.add_argument("--acdms_low_noise_threshold_index", type=int, default=None)
    parser.add_argument("--acdms_apply_last_k_steps", type=int, default=None)
    parser.add_argument("--sampler_distill_weight", type=float, default=None)
    parser.add_argument("--sampler_distill_grad_weight", type=float, default=None)
    parser.add_argument("--sampler_distill_batch_size", type=int, default=None)
    parser.add_argument("--sampler_distill_every", type=int, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None)
    parser.add_argument("--tau_sample_mode", default=None, choices=["per_batch", "per_sample"])
    parser.add_argument("--eps_loss_weight", type=float, default=None)
    parser.add_argument("--grad_loss_weight", type=float, default=None)
    parser.add_argument("--xstart_loss_weight", type=float, default=None)
    parser.add_argument("--var_loss_weight", type=float, default=None)
    parser.add_argument("--clip_xstart_loss", type=_str2bool, default=None)
    parser.add_argument("--warn_on_poor_nas", type=_str2bool, default=None)
    parser.add_argument("--poor_fid_threshold", type=float, default=None)
    parser.add_argument("--distributed", type=_str2bool, default=None)
    parser.add_argument("--dist_backend", default=None)
    parser.add_argument("--dist_init_method", default=None)
    return parser


def load_train_cfg(args: argparse.Namespace) -> TrainCfg:
    cfg_dict = asdict(TrainCfg(
        ckpt="",
        data_dir="",
        outdir="",
        best_timesteps_npy="",
    ))
    if args.config:
        config_path = os.path.abspath(args.config)
        if os.path.isfile(config_path):
            cfg_dict.update(load_config_file(config_path))
        elif args.config != DEFAULT_CONFIG_PATH:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    if "fp16" in cli_overrides:
        cli_overrides["use_fp16"] = cli_overrides.pop("fp16")
    cfg_dict.update(cli_overrides)

    required_fields = ["ckpt", "data_dir", "outdir", "best_timesteps_npy"]
    missing = [name for name in required_fields if not cfg_dict.get(name)]
    if missing:
        raise ValueError(
            "Missing required config values: "
            + ", ".join(missing)
            + f". Edit {args.config} or pass them on the command line."
        )

    return TrainCfg(**cfg_dict)


def sample_t_from_tau(tau_list: List[int], batch_size: int, mode: str) -> th.Tensor:
    tau = [int(x) for x in tau_list]
    if mode == "per_sample":
        t = np.random.choice(tau, size=(batch_size,), replace=True)
        return th.from_numpy(t).long()
    t0 = int(np.random.choice(tau, size=(1,), replace=True)[0])
    return th.full((batch_size,), t0, dtype=th.long)


def _build_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def _set_optimizer_lr(opt: th.optim.Optimizer, lr: float) -> None:
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def _lr_scale_for_step(step: int, cfg: TrainCfg) -> float:
    schedule = str(cfg.lr_schedule).lower()
    warmup_steps = max(0, int(cfg.lr_warmup_steps))
    total_steps = max(1, int(cfg.total_steps))
    min_scale = min(max(float(cfg.lr_min_scale), 0.0), 1.0)

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))

    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule={cfg.lr_schedule!r}. Expected 'constant' or 'cosine'.")

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * cosine


def student_forward_with_policies(
    student: nn.Module,
    x_t: th.Tensor,
    t: th.Tensor,
    y: Optional[th.Tensor],
    g_star_obj,
    tau_list: List[int],
    allow_hard_skip: bool = True,
) -> th.Tensor:
    """
    针对 per-sample tau，把不同 t 的样本分组后分别注入对应 skip_layer。
    """
    unique_t = [int(v) for v in th.unique(t).tolist()]
    if len(unique_t) == 1:
        controls = gstar_to_policy_controls_for_scalar_t(
            g_star_obj,
            tau_list,
            unique_t[0],
            allow_hard_skip=allow_hard_skip,
        )
        kwargs = {
            "skip_layer": controls["skip_layers"],
            "layer_gate_map": controls["layer_gate_map"],
            "class_gain": controls["class_gain"],
        }
        if y is not None:
            kwargs["y"] = y
        return student(x_t, t, **kwargs)

    outputs = None
    for tu in unique_t:
        idx = (t == tu)
        controls = gstar_to_policy_controls_for_scalar_t(
            g_star_obj,
            tau_list,
            tu,
            allow_hard_skip=allow_hard_skip,
        )
        kwargs = {
            "skip_layer": controls["skip_layers"],
            "layer_gate_map": controls["layer_gate_map"],
            "class_gain": controls["class_gain"],
        }
        if y is not None:
            kwargs["y"] = y[idx]
        out_g = student(x_t[idx], t[idx], **kwargs)
        if outputs is None:
            outputs = th.empty(
                (x_t.shape[0],) + tuple(out_g.shape[1:]),
                device=out_g.device,
                dtype=out_g.dtype,
            )
        outputs[idx] = out_g
    return outputs


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_train_cfg(args)

    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    logger.configure(dir=cfg.outdir)
    save_json(asdict(cfg), os.path.join(cfg.outdir, "train_config.resolved.json"))

    set_seed(cfg.seed)
    dist_util.setup_dist(
        distributed=cfg.distributed,
        backend=cfg.dist_backend,
        init_method=cfg.dist_init_method,
    )

    device = cfg.device
    if device == "auto":
        device = str(dist_util.dev())
    use_fp16 = cfg.use_fp16

    tau_list = load_npy(cfg.best_timesteps_npy).astype(np.int64).tolist()
    logger.log(f"[Tau] loaded best timesteps: {tau_list}")

    best_arch_obj = load_pickle(cfg.best_arch_pkl) if cfg.use_best_arch else None
    g_star_obj = load_pickle(cfg.g_star_pkl) if cfg.use_step_policies else None
    if cfg.strict_artifacts:
        if cfg.use_best_arch and best_arch_obj is None:
            raise FileNotFoundError(f"Required static architecture missing: {cfg.best_arch_pkl}")
        if cfg.use_step_policies and g_star_obj is None:
            raise FileNotFoundError(f"Required dynamic policy missing: {cfg.g_star_pkl}")
    nas_summary = summarize_nas_artifacts(best_arch_obj, g_star_obj)
    logger.log(f"[NAS] summary: {json.dumps(nas_summary, ensure_ascii=True)}")
    if cfg.use_step_policies and not cfg.allow_hard_skip:
        logger.log("[NAS] hard skip disabled; only soft layer gates and class_gain will be applied to the student.")
    if cfg.use_best_arch and cfg.warn_on_poor_nas:
        fid_raw = nas_summary.get("fid_raw", None)
        if fid_raw is not None and float(fid_raw) > float(cfg.poor_fid_threshold):
            logger.log(
                "[NAS][Warn] upstream NAS result already has very poor raw FID="
                f"{float(fid_raw):.4f} (> {float(cfg.poor_fid_threshold):.4f}). "
                "Distillation is unlikely to recover a strong student from this seed."
            )
        if cfg.use_step_policies and nas_summary.get("non_empty_step_policies", 0) == 0:
            logger.log("[NAS][Warn] g_star step policies are all empty; current run is effectively static architecture distillation.")

    teacher, diffusion, teacher_defaults = build_teacher_and_diffusion(
        ckpt_path=cfg.ckpt,
        device=device,
        use_fp16=use_fp16,
        model_overrides=_teacher_model_overrides_from_cfg(cfg),
    )

    if cfg.use_best_arch and best_arch_obj is not None:
        student = build_student_from_teacher(
            teacher=teacher,
            teacher_defaults=teacher_defaults,
            best_arch_obj=best_arch_obj,
            device=device,
            use_fp16=use_fp16,
        )
    else:
        if cfg.use_best_arch and best_arch_obj is None:
            logger.log(
                f"[NAS][Fallback] best_arch not found at {cfg.best_arch_pkl}; "
                "fall back to teacher-isomorphic student."
            )
        student = build_student_isomorphic_from_teacher(
            teacher=teacher,
            teacher_defaults=teacher_defaults,
            device=device,
            use_fp16=use_fp16,
        )

    opt = th.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema_student = None
    if float(cfg.ema_decay) > 0.0:
        ema_student = _build_ema_model(student)
        logger.log(f"[EMA] enabled decay={float(cfg.ema_decay):.6f}")
    else:
        logger.log("[EMA] disabled")
    initial_lr = float(cfg.lr) * _lr_scale_for_step(0, cfg)
    _set_optimizer_lr(opt, initial_lr)
    logger.log(
        "[Optim] "
        f"base_lr={float(cfg.lr):.6g}, "
        f"schedule={cfg.lr_schedule}, "
        f"warmup_steps={int(cfg.lr_warmup_steps)}, "
        f"min_scale={float(cfg.lr_min_scale):.4f}, "
        f"initial_lr={initial_lr:.6g}"
    )
    amp_device_type = "cuda" if th.cuda.is_available() else str(device)
    amp_enabled = bool(use_fp16 and amp_device_type == "cuda")
    scaler = th.amp.GradScaler(amp_device_type, enabled=amp_enabled)

    image_size = int(teacher_defaults["image_size"])
    class_cond = bool(teacher_defaults.get("class_cond", True))
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
    ])
    ds = ImageFolder(cfg.data_dir, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    it = iter(dl)

    student.train()
    global_step = 0

    def save_ckpt(tag: str, *, save_ema: bool = False):
        path = os.path.join(cfg.outdir, f"{tag}.pt")
        th.save(
            {"model": student.state_dict(), "opt": opt.state_dict(), "step": global_step, "tau": tau_list},
            path,
        )
        logger.log(f"[Save] {path}")
        if save_ema and ema_student is not None:
            ema_tag = tag.replace("student_", "student_ema_", 1) if tag.startswith("student_") else f"{tag}_ema"
            ema_path = os.path.join(cfg.outdir, f"{ema_tag}.pt")
            th.save(
                {"model": ema_student.state_dict(), "step": global_step, "tau": tau_list, "ema_decay": cfg.ema_decay},
                ema_path,
            )
            logger.log(f"[Save][EMA] {ema_path}")

    pbar = tqdm(total=cfg.total_steps, dynamic_ncols=True)
    while global_step < cfg.total_steps:
        try:
            x0, y = next(it)
        except StopIteration:
            it = iter(dl)
            x0, y = next(it)

        x0 = x0.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long() if class_cond else None
        B = x0.shape[0]

        t = sample_t_from_tau(tau_list, B, cfg.tau_sample_mode).to(device)

        noise = th.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise=noise)

        with th.no_grad(), th.amp.autocast(amp_device_type, enabled=amp_enabled):
            teacher_kwargs = {"skip_layer": []}
            if class_cond:
                teacher_kwargs["y"] = y
            teacher_out = teacher(x_t, t, **teacher_kwargs)

        with th.amp.autocast(amp_device_type, enabled=amp_enabled):
            student_out = student_forward_with_policies(
                student=student,
                x_t=x_t,
                t=t,
                y=y,
                g_star_obj=g_star_obj,
                tau_list=tau_list,
                allow_hard_skip=cfg.allow_hard_skip,
            )
            loss_distill, loss_metrics = distill_adm_hybrid_loss(
                student_out=student_out,
                teacher_out=teacher_out,
                x_t=x_t,
                t=t,
                diffusion=diffusion,
                eps_weight=cfg.eps_loss_weight,
                grad_weight=cfg.grad_loss_weight,
                xstart_weight=cfg.xstart_loss_weight,
                var_weight=cfg.var_loss_weight,
                    clip_xstart=cfg.clip_xstart_loss,
                )
            loss_consistency = th.zeros((), device=x_t.device, dtype=loss_distill.dtype)
            if cfg.lambda_consistency > 0.0:
                loss_consistency = distill_adm_consistency_loss(
                    student_model_fn=lambda x_in, t_in, y_in: student_forward_with_policies(
                        student=student,
                        x_t=x_in,
                        t=t_in,
                        y=y_in,
                        g_star_obj=g_star_obj,
                        tau_list=tau_list,
                        allow_hard_skip=cfg.allow_hard_skip,
                    ),
                    x_start=x0,
                    y=y,
                    tau_list=tau_list,
                    diffusion=diffusion,
                    clip_xstart=cfg.clip_xstart_loss,
                )
            loss_sampler = th.zeros((), device=x_t.device, dtype=loss_distill.dtype)
            sampler_metrics = {
                "loss_sampler_img": 0.0,
                "loss_sampler_grad": 0.0,
            }
            do_sampler_distill = (
                cfg.sampler_type == "acdms"
                and float(cfg.sampler_distill_weight) > 0.0
                and int(cfg.sampler_distill_every) > 0
                and (global_step % int(cfg.sampler_distill_every) == 0)
            )
            if do_sampler_distill:
                sampler_b = max(1, min(int(cfg.sampler_distill_batch_size), B))
                sampler_idx = th.randperm(B, device=x_t.device)[:sampler_b]
                y_sampler = y.index_select(0, sampler_idx)
                x_init_sampler = th.randn(
                    (sampler_b,) + tuple(x0.shape[1:]),
                    device=x_t.device,
                    dtype=x0.dtype,
                )
                with th.no_grad(), th.amp.autocast(amp_device_type, enabled=amp_enabled):
                    teacher_sample = sample_with_adm_acdms(
                        student=teacher,
                        teacher_defaults=teacher_defaults,
                        tau_list=tau_list,
                        g_star_obj=None,
                        shape=tuple(x_init_sampler.shape),
                        device=device,
                        class_labels=y_sampler,
                        clip_denoised=False,
                        fm_strength=cfg.acdms_fm_strength,
                        fm_t_start=cfg.acdms_fm_t_start,
                        fm_lambda_clip=cfg.acdms_fm_lambda_clip,
                        dynamic_thresholding_ratio=cfg.acdms_threshold_ratio,
                        thresholding_max_val=cfg.acdms_threshold_max_val,
                        low_noise_threshold_index=cfg.acdms_low_noise_threshold_index,
                        fm_apply_last_k_steps=cfg.acdms_apply_last_k_steps,
                        x_init=x_init_sampler,
                    )
                student_sample = sample_with_adm_acdms(
                    student=student,
                    teacher_defaults=teacher_defaults,
                    tau_list=tau_list,
                    g_star_obj=g_star_obj,
                    shape=tuple(x_init_sampler.shape),
                    device=device,
                    class_labels=y_sampler,
                    clip_denoised=False,
                    fm_strength=cfg.acdms_fm_strength,
                    fm_t_start=cfg.acdms_fm_t_start,
                    fm_lambda_clip=cfg.acdms_fm_lambda_clip,
                    dynamic_thresholding_ratio=cfg.acdms_threshold_ratio,
                    thresholding_max_val=cfg.acdms_threshold_max_val,
                    low_noise_threshold_index=cfg.acdms_low_noise_threshold_index,
                    fm_apply_last_k_steps=cfg.acdms_apply_last_k_steps,
                    x_init=x_init_sampler,
                    allow_hard_skip=cfg.allow_hard_skip,
                )
                loss_sampler, sampler_metrics = distill_adm_sampler_rollout_loss(
                    student_sample=student_sample,
                    teacher_sample=teacher_sample,
                    grad_weight=float(cfg.sampler_distill_grad_weight),
                )

            loss_total = (
                loss_distill
                + float(cfg.lambda_consistency) * loss_consistency
                + float(cfg.sampler_distill_weight) * loss_sampler
            )

        batch_loss_metrics = dict(loss_metrics)
        batch_loss_metrics["loss_consistency"] = float(loss_consistency.detach().item())
        batch_loss_metrics["loss_sampler"] = float(loss_sampler.detach().item())
        batch_loss_metrics.update(sampler_metrics)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss_total).backward()
        scaler.step(opt)
        scaler.update()
        if ema_student is not None:
            update_ema(ema_student.parameters(), student.parameters(), rate=float(cfg.ema_decay))

        if global_step % cfg.log_every == 0:
            logger.logkv("step", global_step)
            logger.logkv("lr", float(opt.param_groups[0]["lr"]))
            logger.logkv("loss_total", float(loss_total.item()))
            logger.logkv("loss_eps", float(batch_loss_metrics["loss_eps"]))
            logger.logkv("loss_grad", float(batch_loss_metrics["loss_grad"]))
            logger.logkv("loss_xstart", float(batch_loss_metrics["loss_xstart"]))
            logger.logkv("loss_var", float(batch_loss_metrics["loss_var"]))
            logger.logkv("loss_consistency", float(batch_loss_metrics["loss_consistency"]))
            logger.logkv("loss_sampler", float(batch_loss_metrics["loss_sampler"]))
            logger.logkv("loss_sampler_img", float(batch_loss_metrics["loss_sampler_img"]))
            logger.logkv("loss_sampler_grad", float(batch_loss_metrics["loss_sampler_grad"]))
            logger.logkv("lambda_consistency", float(cfg.lambda_consistency))
            logger.logkv("sampler_type", cfg.sampler_type)
            logger.logkv("sampler_distill_weight", float(cfg.sampler_distill_weight))
            logger.logkv("tau_mode", cfg.tau_sample_mode)
            logger.dumpkvs()

        if global_step > 0 and global_step % cfg.save_every == 0:
            save_ckpt(f"student_step{global_step}")

        global_step += 1
        if global_step < cfg.total_steps:
            next_lr = float(cfg.lr) * _lr_scale_for_step(global_step, cfg)
            _set_optimizer_lr(opt, next_lr)
        pbar.update(1)

    save_ckpt("student_final", save_ema=True)
    logger.log("Done.")


if __name__ == "__main__":
    main()
