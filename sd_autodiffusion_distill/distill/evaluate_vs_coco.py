import os
import sys
import json
import gc
import random
import time
from pathlib import Path
from typing import Optional, Any, Tuple, Sequence, Dict

import torch
import numpy as np

current_file = os.path.abspath(__file__)
distill_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(distill_dir)
stable_diffusion_path = os.path.join(project_root, "examples", "Stable Diffusion")

if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from evaluation_metrics import (
    calculate_fid_score,
    calculate_inception_score,
    calculate_clip_score,   # ✅ 统一入口（支持本地 transformers CLIP）
)

# 用于 latency 端到端生成
from .teacher import (
    load_teacher_pipeline,
    encode_text,
    get_uncond_embeddings,
    load_tau,
    teacher_sample,
)
from .train_config import Config
from ldm.util import instantiate_from_config


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup(light: bool = True):
    """
    light=True: 只做 gc + synchronize（更快）
    light=False: 额外 empty_cache（更慢，必要时才用）
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if not light:
            torch.cuda.empty_cache()


def measure_latency_seconds(fn, warmup: int = 10, runs: int = 50) -> float:
    """
    标准 latency 测法：warmup + synchronize + 多次平均
    """
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / runs


def load_student_unet(teacher_pipe, ckpt_path: str, device, dtype):
    """
    和你生成脚本一致：从 teacher_pipe.unet_config instantiate student，然后 load_state_dict
    """
    student = instantiate_from_config(teacher_pipe.unet_config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    student.load_state_dict(state, strict=False)
    student.to(device=device, dtype=dtype)
    student.eval()
    return student


@torch.no_grad()
def measure_generation_latency_sampling_only(
    student_ckpt_path: str,
    device: str,
    sampler_type: str,
    steps: int,
    cfg_scale: float,
    fm_strength: float,
    fm_t_start: float,
    tau_path: str,
    default_tau: Sequence[int],
    image_size: int = 512,
    warmup: int = 10,
    runs: int = 50,
) -> Dict[str, float]:
    """
    ✅ 只测量采样/UNet部分（公平对比 step 数）
    - cond_emb/uncond_emb：计时外预先计算
    - VAE decode：计时外（teacher_sample return_latents=True 跳过）
    """
    dev = torch.device(device)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = load_teacher_pipeline(Config.model_id, device=dev, use_fp16=True)
    pipe.to(device=dev, dtype=dtype)

    # 替换成 student UNet（如果给了 ckpt）
    is_student = False
    if student_ckpt_path and Path(student_ckpt_path).is_file():
        student_unet = load_student_unet(pipe, student_ckpt_path, device=dev, dtype=dtype)
        pipe.unet = student_unet
        is_student = True

    # 固定一个 prompt（可改成你的 benchmark prompt）
    prompt = ["a photo of a dog in the park"]

    # ✅ cond_emb/uncond_emb 在计时外预先计算
    cond_emb = encode_text(pipe, prompt, device=dev)
    uncond_emb = get_uncond_embeddings(pipe, batch_size=1, device=dev)

    latent_h = image_size // 8
    latent_w = image_size // 8

    # tau
    tau_full = load_tau(tau_path, list(default_tau))
    tau = tau_full[: int(steps) + 1]  # time points

    def _one_run():
        latents = torch.randn((1, 4, latent_h, latent_w), device=dev, dtype=dtype)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            _ = teacher_sample(
                pipe=pipe,
                latents=latents,
                tau=tau,
                cond_emb=cond_emb,
                uncond_emb=uncond_emb,
                cfg_scale=float(cfg_scale),
                num_inference_steps=int(steps),
                fm_strength=float(fm_strength),
                fm_t_start=float(fm_t_start),
                sampler_type=str(sampler_type),
                return_latents=True,  # ✅ 关键：跳过 VAE decode
            )
        return

    sec_per_img = measure_latency_seconds(_one_run, warmup=warmup, runs=runs)
    ips = 1.0 / max(1e-12, sec_per_img)

    _cleanup(light=True)
    return {
        "latency_sec_per_img_sampling_only": float(sec_per_img),
        "throughput_img_per_sec_sampling_only": float(ips),
        "latency_warmup": int(warmup),
        "latency_runs": int(runs),
        "latency_batch": 1,
        "latency_is_student_unet": bool(is_student),
        "latency_note": "Only UNet sampling measured (text encoder and VAE decode excluded)",
    }


def evaluate_vs_coco_aligned_official(
    coco_val_image_dir: str,
    generated_paths_json: str,
    selected_file_names_json: str,
    selected_captions_json: str,
    student_ckpt_path: str = "",
    do_latency: bool = True,
    latency_warmup: int = 10,
    latency_runs: int = 50,
    clip_model_path: Optional[str] = None,
):
    cfg = Config
    _set_all_seeds(int(cfg.seed))

    device = cfg.device
    safe_batch = int(min(max(1, cfg.eval_batch_size), 1024))

    if clip_model_path is None:
        clip_model_path = os.getenv(
            "CLIP_MODEL_PATH",
            "examples/Stable Diffusion/models/clip-vit-large-patch14",
        )

    # 1) 生成图路径
    with open(generated_paths_json, "r", encoding="utf-8") as f:
        generated_paths = json.load(f)
    if not isinstance(generated_paths, list) or len(generated_paths) == 0:
        raise RuntimeError("[Error] generated_paths_json 为空或非 list")

    # 2) 真实图 file_names
    with open(selected_file_names_json, "r", encoding="utf-8") as f:
        file_names = json.load(f)
    if not isinstance(file_names, list) or len(file_names) != len(generated_paths):
        raise RuntimeError("[Error] selected_file_names_json 长度必须与 generated_paths 相同")

    # 3) captions（用于 CLIPScore）
    with open(selected_captions_json, "r", encoding="utf-8") as f:
        captions = json.load(f)
    if not isinstance(captions, list) or len(captions) != len(generated_paths):
        raise RuntimeError("[Error] selected_captions_json 长度必须与 generated_paths 相同")

    coco_val_image_dir = Path(coco_val_image_dir)
    real_paths = [str(coco_val_image_dir / fn) for fn in file_names]

    missing = [p for p in real_paths if not Path(p).is_file()]
    if len(missing) > 0:
        raise FileNotFoundError(f"[Error] 有 {len(missing)} 张真实图不存在，示例: {missing[0]}")

    _cleanup(light=True)
    print(f"[Aligned Eval] num_images={len(generated_paths)}, batch={safe_batch}, device={device}")
    print(f"[Aligned Eval] CLIP model path: {clip_model_path}")

    # ========== 质量指标 ==========
    fid = calculate_fid_score(real_paths, generated_paths, device=device, batch_size=safe_batch)
    _cleanup(light=True)

    is_mean, is_std = calculate_inception_score(generated_paths, device=device, batch_size=safe_batch)
    _cleanup(light=True)

    # ========== CLIPScore ==========
    # clip 一般更吃显存，单独限制 batch
    clip_score = None
    clip_batch = int(min(16, max(1, safe_batch)))
    try:
        clip_score = calculate_clip_score(
            generated_paths,
            captions,
            device=device,
            batch_size=clip_batch,
            clip_model_path=clip_model_path,
            fp16=False,  # ✅ 改为 False 以避免数据类型不匹配问题
        )
    except Exception as e:
        print(f"[WARN] CLIPScore 计算失败：{e}")
        clip_score = None
    _cleanup(light=True)

    # ========== Latency（sampling-only） ==========
    latency_info = {}
    if do_latency:
        if cfg.sampler_type == "acdms":
            steps = int(cfg.acdms_num_steps)
            fm_strength = float(cfg.acdms_fm_strength)
            fm_t_start = float(cfg.acdms_fm_t_start)
        elif cfg.sampler_type == "dpm_solver":
            steps = int(cfg.dpm_solver_num_steps)
            fm_strength = float(cfg.dpm_solver_fm_strength)
            fm_t_start = float(cfg.dpm_solver_fm_t_start)
        else:
            raise ValueError(f"Unknown sampler_type: {cfg.sampler_type}")

        latency_info = measure_generation_latency_sampling_only(
            student_ckpt_path=student_ckpt_path,
            device=device,
            sampler_type=str(cfg.sampler_type),
            steps=int(steps),
            cfg_scale=float(cfg.cfg_scale),
            fm_strength=float(fm_strength),
            fm_t_start=float(fm_t_start),
            tau_path=str(cfg.tau_path),
            default_tau=list(cfg.default_tau),
            image_size=int(cfg.image_size),
            warmup=int(latency_warmup),
            runs=int(latency_runs),
        )
        _cleanup(light=True)

    result = {
        "student_ckpt": student_ckpt_path,
        "num_images": int(len(generated_paths)),

        "FID": float(fid),
        "IS_mean": float(is_mean),
        "IS_std": float(is_std),
        "CLIPScore": None if clip_score is None else float(clip_score),

        "clip_model_path": clip_model_path,
        "generated_paths_json": generated_paths_json,
        "selected_file_names_json": selected_file_names_json,
        "selected_captions_json": selected_captions_json,

        "seed": int(cfg.seed),
        "eval_batch_size_used": int(safe_batch),
        **latency_info,
    }

    out_path = Path("student_metrics_vs_coco_aligned_official.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[OK] 保存到 {out_path.resolve()}")
    return result


if __name__ == "__main__":
    raise SystemExit("Use python scripts/sd_pipeline.py evaluate --help from the repository root")
