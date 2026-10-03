"""
evaluation_metrics.py

评估指标计算模块（PyTorch / SciPy，GPU优先 + 防OOM）
实现：
- FID（pytorch-fid InceptionV3，官方口径）
- Inception Score (IS)（torchvision inception_v3）
- CLIP-Score（本地 HF Transformers CLIPModel；paths/tensor；分批防OOM）

GPU策略：
- device="cuda:0" 时：模型与前向都在GPU
- FID/IS：强制 FP32 + autocast disabled（稳定）
- CLIP：默认 fp16（省显存），可关闭
- 不再每个batch都 empty_cache（避免显存碎片化、变慢）
"""

from typing import Tuple, List, Union, Optional, Dict, Any
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from PIL import Image
from torchvision import models, transforms

from src.pytorch_fid import fid_score
from src.pytorch_fid.inception import InceptionV3

try:
    from transformers import CLIPModel, CLIPProcessor
    HF_CLIP_AVAILABLE = True
except Exception:
    HF_CLIP_AVAILABLE = False


# =========================================================
# Utils
# =========================================================

def _normalize_device_key(device: Union[str, torch.device]) -> str:
    return str(torch.device(device))


def _maybe_empty_cache(counter: int, every: int):
    """
    every<=0: never
    """
    if every is None or every <= 0:
        return
    if torch.cuda.is_available() and (counter % every == 0):
        torch.cuda.empty_cache()


# =========================================================
# Official pytorch-fid folder FID (optional)
# =========================================================

def calculate_fid_with_official_impl(
    real_dir: str,
    fake_dir: str,
    device: str = "cuda:0",
    batch_size: int = 16,
) -> float:
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=2048,
    )
    return float(fid)


# =========================================================
# Image loading (paths -> tensor)
# =========================================================

def load_images_from_paths(
    paths: List[str],
    image_size: int = 299,
) -> torch.Tensor:
    """
    Load a SMALL batch of images:
    - resize to image_size
    - return [N,3,H,W] float32 in [0,1]
    """
    if not isinstance(paths, list) or len(paths) == 0 or not all(isinstance(p, str) for p in paths):
        raise ValueError("load_images_from_paths expects non-empty List[str]")
    missing = [p for p in paths if not os.path.exists(p)]
    if len(missing) > 0:
        raise FileNotFoundError(f"Missing image paths (first 5): {missing[:5]}")

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(tfm(img))
    return torch.stack(imgs, dim=0)  # float32


# =========================================================
# FID Inception cache (pytorch-fid InceptionV3)
# =========================================================

_FID_INCEPTION_CACHE: Dict[str, InceptionV3] = {}


def get_fid_inception(device: Union[str, torch.device] = "cpu") -> InceptionV3:
    key = _normalize_device_key(device)
    if key not in _FID_INCEPTION_CACHE:
        dev = torch.device(key)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3(
            output_blocks=(block_idx,),
            resize_input=True,
            normalize_input=True,
            requires_grad=False,
            use_fid_inception=True,
        ).to(dev)
        model.eval()
        _FID_INCEPTION_CACHE[key] = model
    return _FID_INCEPTION_CACHE[key]


# =========================================================
# FID activations (2048-d)
# =========================================================

@torch.inference_mode()
def _get_activations(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    image_size_for_paths: int = 299,
    empty_cache_every: int = 0,  # ✅ 0=never; e.g. 50 for safety
    use_fp16: bool = True,  # ✅ 默认使用FP16加速
) -> np.ndarray:
    """
    Extract 2048-d features using FID Inception.
    - FP16 by default (faster, less memory)
    - autocast enabled for mixed precision
    - batch processing to prevent OOM
    """
    dev = torch.device(device)
    model = get_fid_inception(dev).eval()
    
    # ✅ 使用FP16模型
    if use_fp16 and dev.type == "cuda":
        model = model.half()

    feats_cpu: List[torch.Tensor] = []
    bcount = 0

    if isinstance(images, list):
        n = len(images)
        for i in range(0, n, batch_size):
            bcount += 1
            batch_paths = images[i:i + batch_size]
            batch = load_images_from_paths(batch_paths, image_size=image_size_for_paths)
            batch = batch.to(dev, non_blocking=True)
            
            # ✅ 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()

            # ✅ 启用autocast进行混合精度计算
            with torch.cuda.amp.autocast(enabled=use_fp16 and dev.type == "cuda"):
                pred = model(batch)[0]  # [B,2048,1,1]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)  # [B,2048]
            feats_cpu.append(pred.cpu())

            del batch, pred
            _maybe_empty_cache(bcount, empty_cache_every)

    elif isinstance(images, torch.Tensor):
        n = images.size(0)
        images = images.to(dev, non_blocking=True)
        for i in range(0, n, batch_size):
            bcount += 1
            batch = images[i:i + batch_size]
            
            # ✅ 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()

            # ✅ 启用autocast进行混合精度计算
            with torch.cuda.amp.autocast(enabled=use_fp16 and dev.type == "cuda"):
                pred = model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)
            feats_cpu.append(pred.cpu())

            del batch, pred
            _maybe_empty_cache(bcount, empty_cache_every)

        del images
        _maybe_empty_cache(bcount, empty_cache_every)

    else:
        raise ValueError("images must be torch.Tensor or List[str]")

    return torch.cat(feats_cpu, dim=0).numpy()


def _calculate_activation_statistics(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,  # ✅ 默认使用FP16加速
) -> Tuple[np.ndarray, np.ndarray]:
    feats = _get_activations(
        images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,  # ✅ 传递use_fp16参数
    )
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
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
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid_score(
    real_images: Union[torch.Tensor, List[str], None],
    generated_images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    ref_mu: Union[np.ndarray, str, None] = None,
    ref_sigma: Union[np.ndarray, str, None] = None,
    empty_cache_every: int = 0,  # ✅ 可设为 50 防碎片化
    use_fp16: bool = True,  # ✅ 默认使用FP16加速
) -> float:
    """
    FID(real, gen) using official FID Inception.
    If ref_mu/ref_sigma provided, real_images can be None.
    """
    if ref_mu is not None and ref_sigma is not None:
        mu_r = np.load(ref_mu) if isinstance(ref_mu, str) else ref_mu
        sigma_r = np.load(ref_sigma) if isinstance(ref_sigma, str) else ref_sigma
    else:
        if real_images is None:
            raise ValueError("Need real_images if ref_mu/ref_sigma not provided.")
        mu_r, sigma_r = _calculate_activation_statistics(
            real_images, device=device, batch_size=batch_size, empty_cache_every=empty_cache_every, use_fp16=use_fp16
        )

    mu_g, sigma_g = _calculate_activation_statistics(
        generated_images, device=device, batch_size=batch_size, empty_cache_every=empty_cache_every, use_fp16=use_fp16
    )

    return _calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# =========================================================
# Inception Score (torchvision inception_v3)
# =========================================================

_INCEPTION_CACHE: Dict[str, torch.nn.Module] = {}


def get_inception_logits_model(device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    key = _normalize_device_key(device)
    if key not in _INCEPTION_CACHE:
        dev = torch.device(key)
        inc = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
        ).to(dev)
        inc.eval()
        _INCEPTION_CACHE[key] = inc
    return _INCEPTION_CACHE[key]


@torch.inference_mode()
def _inception_probs(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    image_size_for_paths: int = 299,
    empty_cache_every: int = 0,
    use_fp16: bool = True,  # ✅ 默认使用FP16加速
) -> np.ndarray:
    dev = torch.device(device)
    inc = get_inception_logits_model(dev).eval()
    
    # ✅ 使用FP16模型
    if use_fp16 and dev.type == "cuda":
        inc = inc.half()

    probs_cpu: List[torch.Tensor] = []
    bcount = 0

    def _forward_logits(x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x * 2 - 1
        return inc(x)

    if isinstance(images, list):
        n = len(images)
        for i in range(0, n, batch_size):
            bcount += 1
            batch_paths = images[i:i + batch_size]
            batch = load_images_from_paths(batch_paths, image_size=image_size_for_paths).to(dev)
            
            # ✅ 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()
            
            # ✅ 启用autocast进行混合精度计算
            with torch.cuda.amp.autocast(enabled=use_fp16 and dev.type == "cuda"):
                logits = _forward_logits(batch)
                probs = torch.softmax(logits, dim=1)
            probs_cpu.append(probs.cpu())
            del batch, logits, probs
            _maybe_empty_cache(bcount, empty_cache_every)

    elif isinstance(images, torch.Tensor):
        n = images.size(0)
        images = images.to(dev, non_blocking=True)
        for i in range(0, n, batch_size):
            bcount += 1
            batch = images[i:i + batch_size]
            
            # ✅ 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()
            
            # ✅ 启用autocast进行混合精度计算
            with torch.cuda.amp.autocast(enabled=use_fp16 and dev.type == "cuda"):
                logits = _forward_logits(batch)
                probs = torch.softmax(logits, dim=1)
            probs_cpu.append(probs.cpu())
            del batch, logits, probs
            _maybe_empty_cache(bcount, empty_cache_every)

        del images
        _maybe_empty_cache(bcount, empty_cache_every)
    else:
        raise ValueError("images must be torch.Tensor or List[str]")

    return torch.cat(probs_cpu, dim=0).numpy()


def calculate_inception_score(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    splits: int = 10,
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,  # ✅ 默认使用FP16加速
) -> Tuple[float, float]:
    if isinstance(images, list):
        N = len(images)
    elif isinstance(images, torch.Tensor):
        N = images.size(0)
    else:
        raise ValueError("images must be torch.Tensor or List[str]")
    if N == 0:
        return float("nan"), float("nan")

    splits = min(int(splits), int(N))
    preds = _inception_probs(
        images, device=device, batch_size=batch_size, empty_cache_every=empty_cache_every, use_fp16=use_fp16
    )

    scores = []
    split_size = max(1, N // splits)
    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else N
        part = preds[start:end]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl)))

    return float(np.mean(scores)), float(np.std(scores))


# =========================================================
# CLIP-Score (HF Transformers local, GPU, batched)
# =========================================================

class CLIPScoreCalculator:
    def __init__(
        self,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        fp16: bool = True,  # ✅ 改为 True 以启用FP16加速
    ):
        if not HF_CLIP_AVAILABLE:
            raise RuntimeError("transformers not installed; cannot compute CLIPScore.")

        self.device = torch.device(device)
        if model_path is None:
            model_path = os.getenv(
                "CLIP_MODEL_PATH",
                "examples/Stable Diffusion/models/clip-vit-large-patch14",
            )
        self.model_path = model_path

        self.processor = CLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(self.model_path, local_files_only=True)
        self.model.eval().to(self.device)

        self.use_fp16 = bool(fp16 and self.device.type == "cuda")
        if self.use_fp16:
            self.model = self.model.half()

    @torch.inference_mode()
    def _encode_batch(self, pil_images: List[Image.Image], texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True)
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        # ✅ 修复：当使用 fp16 时，将所有输入张量都转换为 half 精度
        if self.use_fp16:
            for k in inputs:
                if inputs[k].is_floating_point():
                    inputs[k] = inputs[k].half()

        img_feat = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feat = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat * txt_feat).sum(dim=-1)

        del inputs, img_feat, txt_feat
        return sims.detach().cpu()

    @torch.inference_mode()
    def calculate_from_paths(
        self,
        image_paths: List[str],
        texts: List[str],
        batch_size: int = 16,
        empty_cache_every: int = 0,
    ) -> float:
        assert len(image_paths) == len(texts), "image_paths and texts length mismatch"
        if len(image_paths) == 0:
            raise ValueError("image_paths empty")

        sims_all: List[torch.Tensor] = []
        bcount = 0
        for i in range(0, len(image_paths), batch_size):
            bcount += 1
            batch_paths = image_paths[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            sims_all.append(self._encode_batch(pil_images, batch_texts))
            _maybe_empty_cache(bcount, empty_cache_every)

        sims = torch.cat(sims_all, dim=0)
        return float(sims.mean().item())

    @torch.inference_mode()
    def calculate_from_tensor(
        self,
        images: torch.Tensor,   # [N,3,H,W] in [0,1]
        texts: List[str],
        batch_size: int = 16,
        empty_cache_every: int = 0,
    ) -> float:
        assert images.size(0) == len(texts), "images and texts length mismatch"
        if images.numel() == 0:
            raise ValueError("images empty")

        sims_all: List[torch.Tensor] = []
        N = images.size(0)
        bcount = 0
        for i in range(0, N, batch_size):
            bcount += 1
            batch = images[i:i + batch_size].detach().cpu().clamp(0, 1)
            batch_texts = texts[i:i + batch_size]
            pil_images = [transforms.ToPILImage()(img) for img in batch]
            sims_all.append(self._encode_batch(pil_images, batch_texts))
            _maybe_empty_cache(bcount, empty_cache_every)

        sims = torch.cat(sims_all, dim=0)
        return float(sims.mean().item())


def calculate_clip_score(
    images_or_paths: Union[torch.Tensor, List[str]],
    texts: List[str],
    device: str = "cuda:0",
    batch_size: int = 16,
    clip_model_path: Optional[str] = None,
    fp16: bool = True,  # ✅ 改为 True 以启用FP16加速
    empty_cache_every: int = 0,
) -> float:
    scorer = CLIPScoreCalculator(device=device, model_path=clip_model_path, fp16=fp16)
    if isinstance(images_or_paths, torch.Tensor):
        return scorer.calculate_from_tensor(images_or_paths, texts, batch_size=batch_size, empty_cache_every=empty_cache_every)
    if isinstance(images_or_paths, list):
        return scorer.calculate_from_paths(images_or_paths, texts, batch_size=batch_size, empty_cache_every=empty_cache_every)
    raise TypeError("images_or_paths must be torch.Tensor or List[str]")


# =========================================================
# Self-test
# =========================================================

if __name__ == "__main__":
    imgs_g = torch.rand(16, 3, 256, 256)

    fid = calculate_fid_score(imgs_g, imgs_g, device="cuda:0" if torch.cuda.is_available() else "cpu", batch_size=8)
    is_m, is_s = calculate_inception_score(imgs_g, device="cuda:0" if torch.cuda.is_available() else "cpu", splits=4, batch_size=8)

    print("FID:", fid)
    print("IS:", is_m, "+/-", is_s)

    if HF_CLIP_AVAILABLE:
        texts = ["a cat"] * 16
        cs = calculate_clip_score(
            imgs_g.clamp(0, 1),
            texts,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            batch_size=8,
            fp16=torch.cuda.is_available(),
        )
        print("CLIPScore(HF):", cs)
    else:
        print("transformers not available; skip CLIPScore")
