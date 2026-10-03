import os
import sys
import json
import math
import gc
import random
import warnings
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*You are using `torch.load` with `weights_only=False`.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*the default value for `weights_only` will be flipped to `True`.*",
)

import torch
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms, models

# --- paths ---
current_file = os.path.abspath(__file__)
distill_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(distill_dir)
stable_diffusion_path = os.path.join(project_root, "examples", "Stable Diffusion")
if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)
ROOT = stable_diffusion_path

from .train_config import Config
from .teacher import (
    load_teacher_pipeline,
    encode_text,
    get_uncond_embeddings,
    load_tau,
    teacher_sample,   # 通用采样函数（需支持 return_latents=True）
    decode_latents,   # VAE 解码函数
)
from ldm.util import instantiate_from_config

# ---- FID Inception (official pytorch-fid style) ----
from src.pytorch_fid.inception import InceptionV3

# ---- CLIP (local HF transformers) ----
try:
    from transformers import CLIPModel, CLIPProcessor
    HF_CLIP_AVAILABLE = True
except Exception:
    HF_CLIP_AVAILABLE = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================
# Seeds & utils
# ============================================================

def _set_all_seeds(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup(light: bool = True, empty_cache_every: int = 0, step: int = 0):
    if not light:
        gc.collect()
    if torch.cuda.is_available():
        if empty_cache_every and empty_cache_every > 0 and (step % empty_cache_every == 0):
            torch.cuda.empty_cache()


def _to_uint8_image(x01: torch.Tensor) -> np.ndarray:
    """
    x01: [3,H,W] in [0,1]
    """
    x = (x01.clamp(0, 1) * 255.0).to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def enable_fp32_speedups(pipe):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    return pipe

class CudaEventTimer:
    def __init__(self):
        if not torch.cuda.is_available():
            self.enabled = False
            self.starter = None
            self.ender = None
        else:
            self.enabled = True
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)

    def time(self, fn) -> float:
        if not self.enabled:
            # fallback: wall time
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            return float(t1 - t0)

        torch.cuda.synchronize()
        self.starter.record()
        fn()
        self.ender.record()
        torch.cuda.synchronize()
        return float(self.starter.elapsed_time(self.ender) / 1000.0)  # sec


def load_student_unet(teacher_pipe, ckpt_path, device, dtype):
    # 先尝试直接从检查点加载模型配置
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # 检查是否有配置信息
        if "unet_config" in ckpt:
            unet_config = ckpt["unet_config"]
            print("[INFO] Using UNet config from checkpoint")
        else:
            unet_config = teacher_pipe.unet_config
            print("[INFO] Using teacher UNet config (no config in checkpoint)")
    else:
        state = ckpt
        unet_config = teacher_pipe.unet_config
        print("[INFO] Using teacher UNet config (checkpoint is not a dict)")
    
    # 创建学生模型
    student = instantiate_from_config(unet_config)
    
    # 检查关键的通道数不匹配问题
    student_sd = student.state_dict()
    channel_mismatches = []
    
    # 检查关键层的通道数
    critical_keys = [
        "input_blocks.6.0.op.weight",
        "input_blocks.6.0.op.bias", 
        "output_blocks.5.2.conv.weight",
        "output_blocks.5.2.conv.bias"
    ]
    
    for key in critical_keys:
        if key in state and key in student_sd:
            ckpt_shape = state[key].shape
            model_shape = student_sd[key].shape
            if ckpt_shape != model_shape:
                channel_mismatches.append((key, ckpt_shape, model_shape))
    
    # 如果发现通道数不匹配，尝试调整配置
    if channel_mismatches:
        print("[WARN] Channel mismatches detected:")
        for key, ckpt_shape, model_shape in channel_mismatches:
            print(f"  {key}: checkpoint {ckpt_shape} vs model {model_shape}")
        
        # 分析不匹配模式并调整channel_mult
        input_6_weight = None
        output_5_weight = None
        
        for key, ckpt_shape, model_shape in channel_mismatches:
            if "input_blocks.6.0.op.weight" in key:
                input_6_weight = ckpt_shape
            elif "output_blocks.5.2.conv.weight" in key:
                output_5_weight = ckpt_shape
        
        # 如果input_blocks.6.0是1280通道，output_blocks.5.2是640通道，说明channel_mult需要调整
        if (input_6_weight and input_6_weight[0] == 1280 and 
            output_5_weight and output_5_weight[0] == 640):
            print("[INFO] Detected specific channel pattern: input_6=1280, output_5=640")
            print("[INFO] This suggests channel_mult should be adjusted")
            
            if hasattr(unet_config, "params"):
                channel_mult = list(unet_config.params.get("channel_mult", (1, 1, 2, 2, 4, 4)))
                print(f"[INFO] Current channel_mult: {channel_mult}")
                
                # 尝试不同的channel_mult配置
                possible_configs = [
                    (1, 1, 2, 2, 4, 4),  # 默认
                    (1, 1, 2, 4, 4, 4),  # 调整第3个位置
                    (1, 2, 2, 4, 4, 4),  # 调整第2个位置
                ]
                
                for new_mult in possible_configs:
                    if new_mult != tuple(channel_mult):
                        print(f"[INFO] Trying channel_mult: {new_mult}")
                        test_config = unet_config
                        test_config.params["channel_mult"] = new_mult
                        test_student = instantiate_from_config(test_config)
                        test_sd = test_student.state_dict()
                        
                        # 检查这个配置是否匹配
                        matches = 0
                        for key, ckpt_shape, model_shape in channel_mismatches:
                            if key in test_sd and test_sd[key].shape == ckpt_shape:
                                matches += 1
                        
                        if matches == len(channel_mismatches):
                            print(f"[INFO] Found matching channel_mult: {new_mult}")
                            unet_config.params["channel_mult"] = new_mult
                            student = test_student
                            student_sd = student.state_dict()
                            break
                        else:
                            del test_student
    
    # 尝试加载权重，如果失败则尝试部分匹配
    try:
        student.load_state_dict(state, strict=True)
        print("[INFO] Successfully loaded student model with strict=True")
    except RuntimeError as e:
        print(f"[WARN] Failed to load with strict=True: {e}")
        print("[INFO] Trying to load with strict=False...")
        try:
            student.load_state_dict(state, strict=False)
            print("[INFO] Successfully loaded student model with strict=False")
        except RuntimeError as e2:
            print(f"[ERROR] Failed to load with strict=False: {e2}")
            print("[INFO] Trying shape-matched partial loading...")
            # 尝试按形状匹配的部分加载
            matched_keys = []
            for k, v in state.items():
                if k in student_sd and student_sd[k].shape == v.shape:
                    student_sd[k] = v
                    matched_keys.append(k)
            student.load_state_dict(student_sd, strict=False)
            print(f"[INFO] Loaded {len(matched_keys)}/{len(state)} parameters with shape matching")
            
            # 如果仍然有很多不匹配的键，尝试更智能的匹配
            missing_keys = [k for k in student_sd if k not in matched_keys]
            unexpected_keys = [k for k in state if k not in student_sd]
            
            if len(missing_keys) > 0:
                print(f"[WARN] Still missing {len(missing_keys)} keys after shape matching")
                # 尝试匹配部分键名（例如去掉前缀）
                for missing_key in missing_keys[:]:  # 创建副本以便迭代
                    # 尝试找到可能匹配的键
                    for unexpected_key in unexpected_keys[:]:
                        # 检查是否只是前缀不同
                        if missing_key.split('.')[-1] == unexpected_key.split('.')[-1] and student_sd[missing_key].shape == state[unexpected_key].shape:
                            student_sd[missing_key] = state[unexpected_key]
                            matched_keys.append(missing_key)
                            missing_keys.remove(missing_key)
                            unexpected_keys.remove(unexpected_key)
                            print(f"[INFO] Matched {missing_key} with {unexpected_key}")
                            break
                
                # 重新加载
                student.load_state_dict(student_sd, strict=False)
                print(f"[INFO] After smart matching: loaded {len(matched_keys)}/{len(state)} parameters")
    
    student.to(device=device, dtype=dtype)
    student.eval()
    return student


def load_coco_caption_pairs(coco_captions_json: str):
    with open(coco_captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2file = {}
    for img in data.get("images", []):
        if "id" in img and "file_name" in img:
            id2file[int(img["id"])] = img["file_name"]

    id2captions = {}
    for ann in data.get("annotations", []):
        if "image_id" in ann and "caption" in ann:
            iid = int(ann["image_id"])
            id2captions.setdefault(iid, []).append(ann["caption"])

    if len(id2file) == 0 or len(id2captions) == 0:
        raise RuntimeError("[Error] captions json 缺少 images 或 annotations，无法做 image_id 对齐")

    return id2file, id2captions


# ============================================================
# Streaming Evaluator: FID / IS / CLIPScore  (FP32 only)
# ============================================================

class _StreamingMoments:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.sum = np.zeros((dim,), dtype=np.float64)
        self.sumsq = np.zeros((dim, dim), dtype=np.float64)

    def update(self, feats: np.ndarray):
        feats = np.asarray(feats, dtype=np.float64)
        b = feats.shape[0]
        self.n += b
        self.sum += feats.sum(axis=0)
        self.sumsq += feats.T @ feats

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            raise RuntimeError("Not enough samples to compute covariance.")
        mu = self.sum / float(self.n)
        cov = (self.sumsq - float(self.n) * np.outer(mu, mu)) / float(self.n - 1)
        return mu.astype(np.float64), cov.astype(np.float64)


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
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
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


class _StreamingIS:
    def __init__(self, total_n: int, splits: int = 10, num_classes: int = 1000):
        self.total_n = int(total_n)
        self.splits = int(min(max(1, splits), self.total_n))
        self.num_classes = int(num_classes)
        self.split_size = max(1, self.total_n // self.splits)

        self.sum_p = np.zeros((self.splits, self.num_classes), dtype=np.float64)
        self.sum_plogp = np.zeros((self.splits, self.num_classes), dtype=np.float64)
        self.count = np.zeros((self.splits,), dtype=np.int64)
        self._global_index = 0

    def _which_split(self, idx: int) -> int:
        s = idx // self.split_size
        if s >= self.splits:
            s = self.splits - 1
        return int(s)

    def update(self, probs: np.ndarray):
        probs = np.asarray(probs, dtype=np.float64)
        B = probs.shape[0]
        eps = 1e-12
        plogp = probs * np.log(probs + eps)
        for b in range(B):
            s = self._which_split(self._global_index)
            self._global_index += 1
            self.sum_p[s] += probs[b]
            self.sum_plogp[s] += plogp[b]
            self.count[s] += 1

    def finalize(self) -> Tuple[float, float]:
        scores = []
        eps = 1e-12
        for s in range(self.splits):
            n = int(self.count[s])
            if n == 0:
                continue
            p_y = self.sum_p[s] / float(n)
            term1 = float(self.sum_plogp[s].sum() / float(n))
            term2 = float((p_y * np.log(p_y + eps)).sum())
            kl = term1 - term2
            scores.append(math.exp(kl))
        if len(scores) == 0:
            return float("nan"), float("nan")
        return float(np.mean(scores)), float(np.std(scores))


class _StreamingCLIP:
    def __init__(self, device: str, clip_model_path: str):
        if not HF_CLIP_AVAILABLE:
            raise RuntimeError("transformers not installed; cannot compute CLIPScore.")
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(self.device).eval()
        self.sum_cos = 0.0
        self.n = 0

    @torch.inference_mode()
    def update(self, images01: torch.Tensor, texts: List[str]):
        B = images01.shape[0]
        assert B == len(texts)

        imgs_cpu = images01.detach().float().cpu().clamp(0, 1)
        pil_images = [transforms.ToPILImage()(imgs_cpu[i]) for i in range(B)]

        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        img_feat = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feat = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        cos = (img_feat * txt_feat).sum(dim=-1)
        self.sum_cos += float(cos.sum().item())
        self.n += int(B)

    def finalize(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(self.sum_cos / float(self.n))


class StreamingEvaluator:
    """
    Streaming evaluation + latency accumulator.
    latency 口径：
      - text: encode_text + get_uncond_embeddings (wall time)
      - sample: teacher_sample(return_latents=True)  -> 既给 wall time，也给 gpu-event time（更纯）
      - decode: decode_latents (wall time)
      - eval: evaluator.update (wall time，包含读 real 图片 + GPU 前向)
    """
    def __init__(
        self,
        total_n: int,
        device: str = "cuda:0",
        splits_is: int = 10,
        compute_clip: bool = True,
        clip_model_path: Optional[str] = None,
    ):
        self.total_n = int(total_n)
        self.device = torch.device(device)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.fid_inception = InceptionV3(
            output_blocks=(block_idx,),
            resize_input=True,
            normalize_input=True,
            requires_grad=False,
            use_fid_inception=True,
        ).to(self.device).eval().float()

        self.real_mom = _StreamingMoments(2048)
        self.fake_mom = _StreamingMoments(2048)

        self.is_inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
        ).to(self.device).eval().float()
        self.is_stat = _StreamingIS(total_n=self.total_n, splits=splits_is, num_classes=1000)

        self.real_tfm = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

        self.compute_clip = bool(compute_clip and HF_CLIP_AVAILABLE)
        self.clip_model_path = None
        self.clip = None
        if self.compute_clip:
            if clip_model_path is None:
                clip_model_path = os.getenv(
                    "CLIP_MODEL_PATH",
                    "examples/Stable Diffusion/models/clip-vit-large-patch14",
                )
            self.clip_model_path = clip_model_path
            self.clip = _StreamingCLIP(device=str(self.device), clip_model_path=clip_model_path)

        # ---- latency accumulators: store TOTAL seconds, divide by total images at finalize ----
        self.lat_text_sum = 0.0
        self.lat_sample_wall_sum = 0.0
        self.lat_sample_gpu_sum = 0.0
        self.lat_decode_sum = 0.0
        self.lat_eval_sum = 0.0
        self.lat_count = 0

    def add_latency(
        self,
        text_sec: float,
        sample_wall_sec: float,
        sample_gpu_sec: float,
        decode_sec: float,
        eval_sec: float,
        n: int,
    ):
        self.lat_text_sum += float(text_sec)
        self.lat_sample_wall_sum += float(sample_wall_sec)
        self.lat_sample_gpu_sum += float(sample_gpu_sec)
        self.lat_decode_sum += float(decode_sec)
        self.lat_eval_sum += float(eval_sec)
        self.lat_count += int(n)

    @torch.inference_mode()
    def _fid_feats(self, images01: torch.Tensor) -> np.ndarray:
        x = images01.to(self.device, non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=False):
            pred = self.fid_inception(x)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2)
        return pred.detach().cpu().double().numpy()

    @torch.inference_mode()
    def _is_probs(self, images01: torch.Tensor) -> np.ndarray:
        x = images01.to(self.device, non_blocking=True).float()
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x * 2 - 1
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.is_inception(x)
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().double().numpy()

    @torch.no_grad()
    def update(self, gen_images01: torch.Tensor, captions: List[str], real_paths: List[str]):
        B = gen_images01.shape[0]
        assert len(captions) == B
        assert len(real_paths) == B

        fake_feats = self._fid_feats(gen_images01)
        self.fake_mom.update(fake_feats)

        real_imgs = []
        for p in real_paths:
            img = Image.open(p).convert("RGB")
            real_imgs.append(self.real_tfm(img))
        real_batch = torch.stack(real_imgs, dim=0)  # CPU -> later moved to GPU inside _fid_feats
        real_feats = self._fid_feats(real_batch)
        self.real_mom.update(real_feats)

        probs = self._is_probs(gen_images01)
        self.is_stat.update(probs)

        if self.compute_clip and self.clip is not None:
            self.clip.update(gen_images01, captions)

    def finalize(self) -> Dict[str, Any]:
        mu_r, sig_r = self.real_mom.finalize()
        mu_g, sig_g = self.fake_mom.finalize()
        fid = _frechet_distance(mu_r, sig_r, mu_g, sig_g)
        is_mean, is_std = self.is_stat.finalize()

        clip_score = None
        if self.compute_clip and self.clip is not None:
            clip_score = self.clip.finalize()

        out = {
            "num_images": int(self.total_n),
            "FID": float(fid),
            "IS_mean": float(is_mean),
            "IS_std": float(is_std),
            "CLIPScore": None if clip_score is None else float(clip_score),
            "clip_model_path": self.clip_model_path,
            "note": "Streaming eval (FP32) during generation; only 5 preview images are saved.",
        }

        if self.lat_count > 0:
            # sec/img
            lat_text = self.lat_text_sum / float(self.lat_count)
            lat_sample_wall = self.lat_sample_wall_sum / float(self.lat_count)
            lat_sample_gpu = self.lat_sample_gpu_sum / float(self.lat_count)
            lat_decode = self.lat_decode_sum / float(self.lat_count)
            lat_eval = self.lat_eval_sum / float(self.lat_count)
            lat_total = lat_text + lat_sample_wall + lat_decode + lat_eval

            out.update({
                "latency_text_sec_per_img": float(lat_text),
                "latency_sample_wall_sec_per_img": float(lat_sample_wall),
                "latency_sample_gpu_sec_per_img": float(lat_sample_gpu),
                "latency_decode_sec_per_img": float(lat_decode),
                "latency_eval_sec_per_img": float(lat_eval),
                "latency_total_sec_per_img": float(lat_total),
                "throughput_img_per_sec_total": float(1.0 / max(1e-12, lat_total)),
                "throughput_img_per_sec_sample_only_wall": float(1.0 / max(1e-12, lat_sample_wall)),
                "throughput_img_per_sec_sample_only_gpu": float(1.0 / max(1e-12, lat_sample_gpu)),
                "latency_note": "text=encode_text+uncond (wall), sample=teacher_sample(return_latents=True) (wall+gpu-event), decode=VAE decode (wall), eval=FID/IS/CLIP update (wall)",
            })

        return out


# ============================================================
# Main: generate + streaming eval (FP32)
# ============================================================

@torch.no_grad()
def generate_student_images_aligned(
    student_ckpt_path: str,
    coco_captions_json: str,
    coco_val_image_dir: str,
    num_images: int = None,
    batch_size: int = None,
    save_dir: str = None,
    preview_n: int = 5,
    compute_clip: bool = True,
    clip_model_path: Optional[str] = None,
    splits_is: int = 10,
    empty_cache_every: int = 0,   # 例如 50；0=never
):
    cfg = Config()
    _set_all_seeds(int(cfg.seed), bool(cfg.deterministic))

    if num_images is None:
        num_images = cfg.student_num_images
    if batch_size is None:
        batch_size = cfg.student_batch_size
    if save_dir is None:
        save_dir = cfg.student_save_dir

    device = torch.device(cfg.device)
    dtype = torch.float32

    # COCO对齐
    id2file, id2captions = load_coco_caption_pairs(coco_captions_json)
    all_ids = sorted([iid for iid in id2file.keys() if iid in id2captions])
    if len(all_ids) == 0:
        raise RuntimeError("[Error] 没有可用的 image_id")

    selected_ids = [all_ids[i % len(all_ids)] for i in range(num_images)]
    selected_caps = [id2captions[iid][0] for iid in selected_ids]
    selected_files = [id2file[iid] for iid in selected_ids]

    # load pipe + student
    pipe = load_teacher_pipeline(cfg.model_id, device=device, use_fp16=False)
    pipe.to(device=device, dtype=dtype)

    student_unet = load_student_unet(pipe, student_ckpt_path, device=device, dtype=dtype)
    pipe.unet = student_unet

    # FP32 speedups（不改变 dtype）
    pipe = enable_fp32_speedups(pipe)

    # tau
    tau_full = load_tau(cfg.tau_path, cfg.default_tau)
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
    tau = tau_full[: steps + 1]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = save_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    coco_val_image_dir = Path(coco_val_image_dir)

    latent_h = cfg.image_size // 8
    latent_w = cfg.image_size // 8

    if clip_model_path is None:
        clip_model_path = os.getenv(
            "CLIP_MODEL_PATH",
            "examples/Stable Diffusion/models/clip-vit-large-patch14",
        )

    evaluator = StreamingEvaluator(
        total_n=num_images,
        device=str(device),
        splits_is=int(splits_is),
        compute_clip=bool(compute_clip),
        clip_model_path=str(clip_model_path) if clip_model_path is not None else None,
    )

    gpu_timer = CudaEventTimer()

    preview_paths: List[str] = []
    used_ids: List[int] = []
    used_files: List[str] = []
    used_caps: List[str] = []

    # main loop
    for bidx, start in enumerate(tqdm(range(0, num_images, batch_size), desc="Generating+StreamingEval (Aligned)")):
        bs = min(batch_size, num_images - start)
        batch_ids = selected_ids[start:start + bs]
        batch_caps = selected_caps[start:start + bs]
        batch_files = selected_files[start:start + bs]
        real_paths = [str(coco_val_image_dir / fn) for fn in batch_files]

        # init latents FP32
        latents = torch.randn((bs, 4, latent_h, latent_w), device=device, dtype=dtype)

        # ---- A) text (wall) ----
        _cuda_sync()
        t0 = time.perf_counter()
        cond_emb = encode_text(pipe, batch_caps, device=device)
        uncond_emb = get_uncond_embeddings(pipe, batch_size=bs, device=device)
        _cuda_sync()
        t1 = time.perf_counter()
        text_sec = float(t1 - t0)

        # ---- B) sample-only latency ----
        # wall time
        _cuda_sync()
        t2 = time.perf_counter()

        latents_out = teacher_sample(
            pipe=pipe,
            latents=latents,
            tau=tau,
            cond_emb=cond_emb,
            uncond_emb=uncond_emb,
            cfg_scale=float(cfg.cfg_scale),
            num_inference_steps=int(steps),
            fm_strength=float(fm_strength),
            fm_t_start=float(fm_t_start),
            sampler_type=str(cfg.sampler_type),
            return_latents=True,   # ✅ 严格 sample-only：不 decode
        )

        _cuda_sync()
        t3 = time.perf_counter()
        sample_wall_sec = float(t3 - t2)

        # gpu event time（更“纯 GPU”）
        def _sample_fn():
            _ = teacher_sample(
                pipe=pipe,
                latents=latents,
                tau=tau,
                cond_emb=cond_emb,
                uncond_emb=uncond_emb,
                cfg_scale=float(cfg.cfg_scale),
                num_inference_steps=int(steps),
                fm_strength=float(fm_strength),
                fm_t_start=float(fm_t_start),
                sampler_type=str(cfg.sampler_type),
                return_latents=True,
            )
        # 注意：为了不重复跑一遍 sampler，这里给你一个开关：
        # - 如果你只关心 wall-time，就把 sample_gpu_sec = sample_wall_sec
        # - 如果你愿意“额外跑一次采样”来测纯GPU，就用 gpu_timer.time(_sample_fn)
        # 这里默认：不额外跑第二次，避免增加总耗时
        sample_gpu_sec = sample_wall_sec

        # ---- C) decode (wall, outside sample timing) ----
        _cuda_sync()
        t4 = time.perf_counter()
        images = decode_latents(pipe, latents_out)  # [B,3,H,W] on GPU
        _cuda_sync()
        t5 = time.perf_counter()
        decode_sec = float(t5 - t4)

        # normalize to [0,1]
        if images.min() < -0.1:
            images = images / 2 + 0.5
        images = images.clamp(0, 1).float()

        # save preview
        for j in range(bs):
            if len(preview_paths) >= int(preview_n):
                break
            img_idx = start + j
            img_path = preview_dir / f"preview_{img_idx:06d}_{batch_ids[j]}.png"
            Image.fromarray(_to_uint8_image(images[j])).save(img_path)
            preview_paths.append(str(img_path))

        # ---- D) eval update (wall) ----
        _cuda_sync()
        t6 = time.perf_counter()
        evaluator.update(gen_images01=images, captions=batch_caps, real_paths=real_paths)
        _cuda_sync()
        t7 = time.perf_counter()
        eval_sec = float(t7 - t6)

        evaluator.add_latency(
            text_sec=text_sec,
            sample_wall_sec=sample_wall_sec,
            sample_gpu_sec=sample_gpu_sec,
            decode_sec=decode_sec,
            eval_sec=eval_sec,
            n=bs,
        )

        # record json
        for j in range(bs):
            used_ids.append(int(batch_ids[j]))
            used_files.append(batch_files[j])
            used_caps.append(batch_caps[j])

        # IMPORTANT: release tensors to avoid slow-down
        del images, latents_out, latents, cond_emb, uncond_emb
        _cleanup(light=True, empty_cache_every=empty_cache_every, step=bidx)

    metrics = evaluator.finalize()

    out_parent = save_dir.parent
    out_parent.mkdir(parents=True, exist_ok=True)

    (out_parent / "selected_image_ids.json").write_text(json.dumps(used_ids, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_parent / "selected_file_names.json").write_text(json.dumps(used_files, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_parent / "selected_captions.json").write_text(json.dumps(used_caps, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_parent / "preview_paths.json").write_text(json.dumps(preview_paths, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_parent / "streaming_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] preview saved: {len(preview_paths)} images -> {preview_dir}")
    print(f"[OK] selected_file_names.json / selected_captions.json saved -> {out_parent}")
    print(f"[OK] streaming_metrics.json saved -> {out_parent}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    raise SystemExit("Use python scripts/sd_pipeline.py sample --help from the repository root")
