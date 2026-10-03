"""
imagenet64_evaluation_metrics.py

专门用于ImageNet64数据集的评估指标计算模块
实现：
- FID（基于ImageNet64参考统计）
- Inception Score (IS)
- Precision and Recall
- KID (Kernel Inception Distance)
- 图像质量评估指标

针对ImageNet64数据集（64x64图像）进行了优化和适配
"""

from typing import Tuple, List, Union, Optional, Dict, Any
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# 导入InceptionV3模型
from src.pytorch_fid.inception import InceptionV3

# ImageNet64数据集特定的配置
IMAGENET64_SIZE = 64  # ImageNet64图像尺寸
IMAGENET64_REF_STATS_PATH = "examples/guided_diffusion/models"
IMAGENET64_MU_FILE = "imagenet64_mu.npy"
IMAGENET64_SIGMA_FILE = "imagenet64_sigma.npy"

# 支持的图像扩展名
IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

_REF_STATS_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_REF_STATS_LOGGED: set = set()

# =========================================================
# 工具函数
# =========================================================

def _normalize_device_key(device: Union[str, torch.device]) -> str:
    """标准化设备字符串"""
    return str(torch.device(device))

def _maybe_empty_cache(counter: int, every: int):
    """
    每隔every次清空缓存
    every<=0: 从不
    """
    if every is None or every <= 0:
        return
    if torch.cuda.is_available() and (counter % every == 0):
        torch.cuda.empty_cache()


def _autocast_context(device: torch.device, enabled: bool):
    """Use the non-deprecated AMP autocast API when CUDA autocast is needed."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.amp.autocast("cpu", enabled=False)


def _log_ref_stats_once(cache_key: str, mu: np.ndarray, sigma: np.ndarray, source: str) -> None:
    if cache_key in _REF_STATS_LOGGED:
        return
    _REF_STATS_LOGGED.add(cache_key)
    print(f"{source}:")
    print(f"  mu形状: {mu.shape}")
    print(f"  sigma形状: {sigma.shape}")

# =========================================================
# ImageNet64数据集加载器
# =========================================================

class ImageNet64Dataset(torch.utils.data.Dataset):
    """ImageNet64数据集加载器，专门处理64x64的图像"""
    
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        
        # 确保图像尺寸为64x64
        if img.size != (IMAGENET64_SIZE, IMAGENET64_SIZE):
            img = img.resize((IMAGENET64_SIZE, IMAGENET64_SIZE), Image.LANCZOS)
            
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def load_imagenet64_images(
    paths: List[str],
    image_size: int = IMAGENET64_SIZE,
) -> torch.Tensor:
    """
    加载ImageNet64图像
    - 确保图像尺寸为64x64
    - 返回 [N,3,H,W] float32 in [0,1]
    """
    if not isinstance(paths, list) or len(paths) == 0 or not all(isinstance(p, str) for p in paths):
        raise ValueError("load_imagenet64_images expects non-empty List[str]")
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
        # 确保图像尺寸为64x64
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.LANCZOS)
        imgs.append(tfm(img))
    return torch.stack(imgs, dim=0)  # float32

# =========================================================
# FID相关函数
# =========================================================

# FID Inception模型缓存
_FID_INCEPTION_CACHE: Dict[str, InceptionV3] = {}

def get_fid_inception(device: Union[str, torch.device] = "cpu") -> InceptionV3:
    """获取FID Inception模型（带缓存）"""
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

@torch.inference_mode()
def get_activations_for_imagenet64(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    计算ImageNet64图像的Inception特征
    - FP16 by default (faster, less memory)
    - batch processing to prevent OOM
    """
    dev = torch.device(device)
    model = get_fid_inception(dev).eval()
    
    # 使用FP16模型
    if use_fp16 and dev.type == "cuda":
        model = model.half()

    feats: List[torch.Tensor] = []
    bcount = 0

    if isinstance(images, list):
        n = len(images)
        for i in range(0, n, batch_size):
            bcount += 1
            batch_paths = images[i:i + batch_size]
            batch = load_imagenet64_images(batch_paths, image_size=IMAGENET64_SIZE)
            batch = batch.to(dev, non_blocking=True)
            
            # 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()

            # 启用autocast进行混合精度计算
            with _autocast_context(dev, enabled=use_fp16 and dev.type == "cuda"):
                pred = model(batch)[0]  # [B,2048,1,1]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)  # [B,2048]
            feats.append(pred if return_tensor else pred.cpu())

            del batch, pred
            _maybe_empty_cache(bcount, empty_cache_every)

    elif isinstance(images, torch.Tensor):
        n = images.size(0)
        images = images.to(dev, non_blocking=True)
        for i in range(0, n, batch_size):
            bcount += 1
            batch = images[i:i + batch_size]
            
            # 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()

            # 启用autocast进行混合精度计算
            with _autocast_context(dev, enabled=use_fp16 and dev.type == "cuda"):
                pred = model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)
            feats.append(pred if return_tensor else pred.cpu())

            del batch, pred
            _maybe_empty_cache(bcount, empty_cache_every)

        del images
        _maybe_empty_cache(bcount, empty_cache_every)

    else:
        raise ValueError("images must be torch.Tensor or List[str]")

    out = torch.cat(feats, dim=0)
    if return_tensor:
        return out
    return out.numpy()


def _torch_covariance(feats: torch.Tensor) -> torch.Tensor:
    feats = feats.float()
    n = feats.shape[0]
    if n <= 1:
        raise ValueError("Need at least 2 samples to estimate covariance.")
    centered = feats - feats.mean(dim=0, keepdim=True)
    return centered.t().matmul(centered) / float(n - 1)


def _matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat = 0.5 * (mat + mat.t())
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=eps)
    sqrt_evals = torch.sqrt(evals)
    return (evecs * sqrt_evals.unsqueeze(0)) @ evecs.t()


def calculate_activation_statistics_for_imagenet64_torch(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = get_activations_for_imagenet64(
        images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,
        return_tensor=True,
    )
    if not isinstance(feats, torch.Tensor):
        raise TypeError("Expected torch.Tensor activations when return_tensor=True.")
    feats = feats.float()
    mu = feats.mean(dim=0)
    sigma = _torch_covariance(feats)
    return mu, sigma

def calculate_activation_statistics_for_imagenet64(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算ImageNet64图像的激活统计信息"""
    feats = get_activations_for_imagenet64(
        images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,
    )
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """计算Fréchet距离"""
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


def calculate_frechet_distance_torch(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    mu1 = mu1.float()
    mu2 = mu2.float().to(mu1.device)
    sigma1 = sigma1.float()
    sigma2 = sigma2.float().to(sigma1.device)

    diff = mu1 - mu2
    sigma1 = sigma1 + torch.eye(sigma1.shape[0], device=sigma1.device, dtype=sigma1.dtype) * eps
    sigma2 = sigma2 + torch.eye(sigma2.shape[0], device=sigma2.device, dtype=sigma2.dtype) * eps

    sigma1_sqrt = _matrix_sqrt_psd(sigma1, eps=eps)
    middle = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    trace_covmean = torch.trace(_matrix_sqrt_psd(middle, eps=eps))

    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2.0 * trace_covmean
    return float(fid.item())

def load_imagenet64_reference_stats():
    """
    加载ImageNet64数据集的参考统计信息
    
    返回:
    -- mu    : ImageNet64数据集的Inception特征均值
    -- sigma : ImageNet64数据集的Inception特征协方差矩阵
    """
    mu_path = os.path.join(IMAGENET64_REF_STATS_PATH, IMAGENET64_MU_FILE)
    sigma_path = os.path.join(IMAGENET64_REF_STATS_PATH, IMAGENET64_SIGMA_FILE)
    cache_key = f"default::{mu_path}::{sigma_path}"

    if cache_key in _REF_STATS_CACHE:
        return _REF_STATS_CACHE[cache_key]
    
    if not os.path.exists(mu_path) or not os.path.exists(sigma_path):
        raise FileNotFoundError(f"ImageNet64参考统计文件未找到: {mu_path} 或 {sigma_path}")
    
    mu = np.load(mu_path)
    sigma = np.load(sigma_path)
    
    _REF_STATS_CACHE[cache_key] = (mu, sigma)
    _log_ref_stats_once(cache_key, mu, sigma, "已加载ImageNet64参考统计信息")
    return _REF_STATS_CACHE[cache_key]

def calculate_fid_for_imagenet64(
    generated_images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
    ref_mu: Optional[Union[str, np.ndarray]] = None,
    ref_sigma: Optional[Union[str, np.ndarray]] = None,
) -> float:
    """
    计算生成图像与ImageNet64数据集之间的FID分数
    
    参数:
    -- generated_images : 生成的图像张量或路径列表
    -- device            : 运行计算的设备
    -- batch_size        : 批次大小
    -- empty_cache_every : 每隔多少批次清空缓存
    -- use_fp16          : 是否使用FP16加速
    -- ref_mu            : 参考均值文件路径或数组（可选，默认使用硬编码路径）
    -- ref_sigma         : 参考协方差矩阵文件路径或数组（可选，默认使用硬编码路径）
    
    返回:
    -- fid_value         : 计算得到的FID分数
    """
    # 加载ImageNet64参考统计信息
    if ref_mu is not None and ref_sigma is not None:
        # 使用用户提供的参考统计信息
        cache_key = f"user::{str(ref_mu)}::{str(ref_sigma)}"
        if isinstance(ref_mu, str):
            mu_ref = np.load(ref_mu)
        else:
            mu_ref = ref_mu
            
        if isinstance(ref_sigma, str):
            sigma_ref = np.load(ref_sigma)
        else:
            sigma_ref = ref_sigma

        _log_ref_stats_once(cache_key, mu_ref, sigma_ref, "使用用户提供的参考统计信息")
    else:
        # 使用默认的ImageNet64参考统计信息
        mu_ref, sigma_ref = load_imagenet64_reference_stats()
    
    dev = torch.device(device)
    use_torch_fid = isinstance(generated_images, torch.Tensor) and dev.type == "cuda"

    if use_torch_fid:
        mu_gen_t, sigma_gen_t = calculate_activation_statistics_for_imagenet64_torch(
            generated_images,
            device=device,
            batch_size=batch_size,
            empty_cache_every=empty_cache_every,
            use_fp16=use_fp16,
        )
        mu_ref_t = torch.as_tensor(mu_ref, device=dev, dtype=torch.float32)
        sigma_ref_t = torch.as_tensor(sigma_ref, device=dev, dtype=torch.float32)
        return calculate_frechet_distance_torch(mu_gen_t, sigma_gen_t, mu_ref_t, sigma_ref_t)

    mu_gen, sigma_gen = calculate_activation_statistics_for_imagenet64(
        generated_images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,
    )
    return calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

# =========================================================
# Inception Score
# =========================================================

# Inception模型缓存
_INCEPTION_CACHE: Dict[str, torch.nn.Module] = {}

def get_inception_logits_model(device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    """获取Inception logits模型（带缓存）"""
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
def inception_probs_for_imagenet64(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
) -> np.ndarray:
    """计算ImageNet64图像的Inception概率"""
    dev = torch.device(device)
    inc = get_inception_logits_model(dev).eval()
    
    # 使用FP16模型
    if use_fp16 and dev.type == "cuda":
        inc = inc.half()

    probs_cpu: List[torch.Tensor] = []
    bcount = 0

    def _forward_logits(x: torch.Tensor) -> torch.Tensor:
        # 确保输入尺寸为299x299
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # 归一化到[-1,1]
        x = x * 2 - 1
        return inc(x)

    if isinstance(images, list):
        n = len(images)
        for i in range(0, n, batch_size):
            bcount += 1
            batch_paths = images[i:i + batch_size]
            batch = load_imagenet64_images(batch_paths, image_size=IMAGENET64_SIZE).to(dev)
            
            # 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()
            
            # 启用autocast进行混合精度计算
            with _autocast_context(dev, enabled=use_fp16 and dev.type == "cuda"):
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
            
            # 使用FP16
            if use_fp16 and dev.type == "cuda":
                batch = batch.half()
            
            # 启用autocast进行混合精度计算
            with _autocast_context(dev, enabled=use_fp16 and dev.type == "cuda"):
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

def calculate_inception_score_for_imagenet64(
    images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    splits: int = 10,
    batch_size: int = 32,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
) -> Tuple[float, float]:
    """
    计算ImageNet64图像的Inception Score
    
    返回:
    -- (mean, std) : IS的均值和标准差
    """
    if isinstance(images, list):
        N = len(images)
    elif isinstance(images, torch.Tensor):
        N = images.size(0)
    else:
        raise ValueError("images must be torch.Tensor or List[str]")
    if N == 0:
        return float("nan"), float("nan")

    splits = min(int(splits), int(N))
    preds = inception_probs_for_imagenet64(
        images, 
        device=device, 
        batch_size=batch_size, 
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16
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
# Precision and Recall for Distributions
# =========================================================

def calculate_manifold_features(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    k: int = 3,
    batch_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算流形特征，用于Precision和Recall计算
    
    参数:
    -- real_features      : 真实图像的特征
    -- generated_features : 生成图像的特征
    -- k                  : KNN的k值
    -- batch_size         : 批次大小
    
    返回:
    -- (precision_features, recall_features) : 用于计算Precision和Recall的特征
    """
    # 计算真实特征之间的距离
    n_real = real_features.shape[0]
    n_gen = generated_features.shape[0]
    
    # 计算真实特征的KNN
    real_precisions = np.zeros(n_real)
    real_recalls = np.zeros(n_real)
    
    # 分批计算以节省内存
    for i in range(0, n_real, batch_size):
        end = min(i + batch_size, n_real)
        batch_real = real_features[i:end]
        
        # 计算当前批次与所有真实特征的距离
        dists = np.linalg.norm(batch_real[:, None, :] - real_features[None, :, :], axis=2)
        # 排序，获取第k+1个最近的邻居（因为自身距离为0）
        sorted_indices = np.argsort(dists, axis=1)
        kth_indices = sorted_indices[:, k+1]
        kth_distances = np.take_along_axis(dists, kth_indices[:, None], axis=1).squeeze()
        
        # 计算当前批次与所有生成特征的距离
        dists_gen = np.linalg.norm(batch_real[:, None, :] - generated_features[None, :, :], axis=2)
        # 计算有多少生成特征在kth距离内
        precisions = np.sum(dists_gen < kth_distances[:, None], axis=1) / n_gen
        real_precisions[i:end] = precisions
    
    # 计算生成特征的KNN
    gen_precisions = np.zeros(n_gen)
    gen_recalls = np.zeros(n_gen)
    
    for i in range(0, n_gen, batch_size):
        end = min(i + batch_size, n_gen)
        batch_gen = generated_features[i:end]
        
        # 计算当前批次与所有生成特征的距离
        dists = np.linalg.norm(batch_gen[:, None, :] - generated_features[None, :, :], axis=2)
        # 排序，获取第k+1个最近的邻居（因为自身距离为0）
        sorted_indices = np.argsort(dists, axis=1)
        kth_indices = sorted_indices[:, k+1]
        kth_distances = np.take_along_axis(dists, kth_indices[:, None], axis=1).squeeze()
        
        # 计算当前批次与所有真实特征的距离
        dists_real = np.linalg.norm(batch_gen[:, None, :] - real_features[None, :, :], axis=2)
        # 计算有多少真实特征在kth距离内
        recalls = np.sum(dists_real < kth_distances[:, None], axis=1) / n_real
        gen_recalls[i:end] = recalls
    
    return real_precisions, gen_recalls

def calculate_precision_and_recall_for_imagenet64(
    real_images: Union[torch.Tensor, List[str]],
    generated_images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    k: int = 3,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
) -> Tuple[float, float]:
    """
    计算ImageNet64图像的Precision和Recall
    
    返回:
    -- (precision, recall) : Precision和Recall分数
    """
    # 计算真实图像和生成图像的特征
    real_features = get_activations_for_imagenet64(
        real_images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,
    )
    
    generated_features = get_activations_for_imagenet64(
        generated_images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,
    )
    
    # 计算Precision和Recall
    real_precisions, gen_recalls = calculate_manifold_features(
        real_features, generated_features, k=k
    )
    
    precision = float(np.mean(real_precisions))
    recall = float(np.mean(gen_recalls))
    
    return precision, recall

# =========================================================
# KID (Kernel Inception Distance)
# =========================================================

def polynomial_mmd(
    features_x: np.ndarray,
    features_y: np.ndarray,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
) -> float:
    """
    计算多项式核的最大均值差异(MMD)
    """
    if gamma is None:
        gamma = 1.0 / features_x.shape[1]
    
    # 计算核矩阵
    K_XX = (np.dot(features_x, features_x.T) * gamma + coef0) ** degree
    K_YY = (np.dot(features_y, features_y.T) * gamma + coef0) ** degree
    K_XY = (np.dot(features_x, features_y.T) * gamma + coef0) ** degree
    
    m = features_x.shape[0]
    n = features_y.shape[0]
    
    # 计算MMD
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    
    return mmd

def calculate_kid_for_imagenet64(
    real_images: Union[torch.Tensor, List[str]],
    generated_images: Union[torch.Tensor, List[str]],
    device: str = "cuda:0",
    batch_size: int = 32,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
) -> Tuple[float, float]:
    """
    计算ImageNet64图像的KID (Kernel Inception Distance)
    
    返回:
    -- (mean, std) : KID的均值和标准差
    """
    # 计算真实图像和生成图像的特征
    real_features = get_activations_for_imagenet64(
        real_images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,
    )
    
    generated_features = get_activations_for_imagenet64(
        generated_images,
        device=device,
        batch_size=batch_size,
        empty_cache_every=empty_cache_every,
        use_fp16=use_fp16,
    )
    
    # 随机采样子集计算KID
    n_real = real_features.shape[0]
    n_gen = generated_features.shape[0]
    
    subset_size = min(max_subset_size, min(n_real, n_gen))
    
    kid_values = []
    for _ in tqdm(range(num_subsets), desc="计算KID"):
        # 随机选择子集
        idx_real = np.random.choice(n_real, subset_size, replace=False)
        idx_gen = np.random.choice(n_gen, subset_size, replace=False)
        
        # 计算KID
        kid = polynomial_mmd(
            real_features[idx_real],
            generated_features[idx_gen],
            degree=degree,
            gamma=gamma,
            coef0=coef0,
        )
        
        kid_values.append(kid)
    
    return float(np.mean(kid_values)), float(np.std(kid_values))

# =========================================================
# 综合评估函数
# =========================================================

def evaluate_generated_images_for_imagenet64(
    generated_images: Union[torch.Tensor, List[str]],
    real_images: Optional[Union[torch.Tensor, List[str]]] = None,
    device: str = "cuda:0",
    batch_size: int = 32,
    calculate_fid: bool = True,
    calculate_is: bool = True,
    calculate_precision_recall: bool = True,
    calculate_kid: bool = True,
    is_splits: int = 10,
    pr_k: int = 3,
    kid_num_subsets: int = 100,
    kid_max_subset_size: int = 1000,
    empty_cache_every: int = 0,
    use_fp16: bool = True,
    ref_mu: Optional[Union[str, np.ndarray]] = None,
    ref_sigma: Optional[Union[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    对ImageNet64生成的图像进行综合评估
    
    参数:
    -- generated_images         : 生成的图像张量或路径列表
    -- real_images              : 真实图像张量或路径列表（用于Precision和Recall、KID）
    -- device                   : 运行计算的设备
    -- batch_size               : 批次大小
    -- calculate_fid            : 是否计算FID
    -- calculate_is             : 是否计算Inception Score
    -- calculate_precision_recall : 是否计算Precision和Recall
    -- calculate_kid            : 是否计算KID
    -- is_splits                : IS的分割数
    -- pr_k                     : Precision和Recall的k值
    -- kid_num_subsets          : KID的子集数量
    -- kid_max_subset_size      : KID的最大子集大小
    -- empty_cache_every        : 每隔多少批次清空缓存
    -- use_fp16                 : 是否使用FP16加速
    -- ref_mu                   : 参考均值文件路径或数组（可选，默认使用硬编码路径）
    -- ref_sigma                : 参考协方差矩阵文件路径或数组（可选，默认使用硬编码路径）
    
    返回:
    -- results : 包含所有评估指标的字典
    """
    results = {}
    
    # 计算FID
    if calculate_fid:
        print("计算FID...")
        fid = calculate_fid_for_imagenet64(
            generated_images,
            device=device,
            batch_size=batch_size,
            empty_cache_every=empty_cache_every,
            use_fp16=use_fp16,
            ref_mu=ref_mu,
            ref_sigma=ref_sigma,
        )
        results["fid"] = fid
        print(f"FID: {fid:.4f}")
    
    # 计算Inception Score
    if calculate_is:
        print("计算Inception Score...")
        is_mean, is_std = calculate_inception_score_for_imagenet64(
            generated_images,
            device=device,
            splits=is_splits,
            batch_size=batch_size,
            empty_cache_every=empty_cache_every,
            use_fp16=use_fp16,
        )
        results["is_mean"] = is_mean
        results["is_std"] = is_std
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # 计算Precision和Recall
    if calculate_precision_recall:
        if real_images is None:
            print("警告: 计算Precision和Recall需要真实图像，但未提供")
        else:
            print("计算Precision和Recall...")
            precision, recall = calculate_precision_and_recall_for_imagenet64(
                real_images,
                generated_images,
                device=device,
                batch_size=batch_size,
                k=pr_k,
                empty_cache_every=empty_cache_every,
                use_fp16=use_fp16,
            )
            results["precision"] = precision
            results["recall"] = recall
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # 计算KID
    if calculate_kid:
        if real_images is None:
            print("警告: 计算KID需要真实图像，但未提供")
        else:
            print("计算KID...")
            kid_mean, kid_std = calculate_kid_for_imagenet64(
                real_images,
                generated_images,
                device=device,
                batch_size=batch_size,
                num_subsets=kid_num_subsets,
                max_subset_size=kid_max_subset_size,
                empty_cache_every=empty_cache_every,
                use_fp16=use_fp16,
            )
            results["kid_mean"] = kid_mean
            results["kid_std"] = kid_std
            print(f"KID: {kid_mean:.4f} ± {kid_std:.4f}")
    
    return results

# =========================================================
# 主函数和示例用法
# =========================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估ImageNet64生成的图像")
    parser.add_argument("--generated_images", type=str, required=True, help="生成的图像路径")
    parser.add_argument("--real_images", type=str, help="真实图像路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备 (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--calculate_fid", action="store_true", default=True, help="计算FID")
    parser.add_argument("--calculate_is", action="store_true", default=True, help="计算Inception Score")
    parser.add_argument("--calculate_precision_recall", action="store_true", default=True, help="计算Precision和Recall")
    parser.add_argument("--calculate_kid", action="store_true", default=True, help="计算KID")
    parser.add_argument("--is_splits", type=int, default=10, help="IS的分割数")
    parser.add_argument("--pr_k", type=int, default=3, help="Precision和Recall的k值")
    parser.add_argument("--kid_num_subsets", type=int, default=100, help="KID的子集数量")
    parser.add_argument("--kid_max_subset_size", type=int, default=1000, help="KID的最大子集大小")
    parser.add_argument("--use_fp16", action="store_true", default=True, help="使用FP16加速")
    
    args = parser.parse_args()
    
    # 收集图像路径
    import glob
    generated_paths = sorted(glob.glob(os.path.join(args.generated_images, "*")))
    generated_paths = [p for p in generated_paths if any(p.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
    
    real_paths = None
    if args.real_images:
        real_paths = sorted(glob.glob(os.path.join(args.real_images, "*")))
        real_paths = [p for p in real_paths if any(p.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
    
    # 评估图像
    results = evaluate_generated_images_for_imagenet64(
        generated_paths,
        real_paths,
        device=args.device,
        batch_size=args.batch_size,
        calculate_fid=args.calculate_fid,
        calculate_is=args.calculate_is,
        calculate_precision_recall=args.calculate_precision_recall,
        calculate_kid=args.calculate_kid,
        is_splits=args.is_splits,
        pr_k=args.pr_k,
        kid_num_subsets=args.kid_num_subsets,
        kid_max_subset_size=args.kid_max_subset_size,
        use_fp16=args.use_fp16,
    )
    
    # 打印结果
    print("\n评估结果:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        elif isinstance(value, tuple) and len(value) == 2:
            print(f"{metric}: {value[0]:.4f} ± {value[1]:.4f}")
        else:
            print(f"{metric}: {value}")
