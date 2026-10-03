import os
from pathlib import Path
import math
import random
import pickle
import numpy as np
import warnings

# -------------------- 抑制警告信息 --------------------
# 抑制 transformers 库的 FutureWarning（关于 weights_only）
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message='.*You are using `torch.load` with `weights_only=False`.*')
warnings.filterwarnings('ignore', category=FutureWarning,
                       message='.*the default value for `weights_only` will be flipped to `True`.*')

import torch
import torch.cuda.amp as amp
import sys

# --- 1. 正确计算项目根目录和 Stable Diffusion 路径 ---
current_file = os.path.abspath(__file__)
distill_dir = os.path.dirname(os.path.dirname(current_file))           # sd_autodiffusion_distill
project_root = os.path.dirname(distill_dir)                            # AutoDiffusion-main
stable_diffusion_path = os.path.join(project_root, "examples", "Stable Diffusion")

# --- 2. 把 ldm 包的父目录加入 sys.path ---
if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)

ROOT = stable_diffusion_path

from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.openaimodel import UNetModel

from .train_config import Config
from .dataset_coco import COCODataset
from .teacher import (
    load_teacher_pipeline,
    encode_text,
    get_uncond_embeddings,
    encode_image_to_latent,
    teacher_eps,
    load_tau,
    sample_t_from_tau,
    set_seed,
    teacher_sample,  # 新增：通用采样函数
)
from .losses import distillation_loss, consistency_loss

# 关键：UNetAdapter + NAS policy 类型
from .adapters.unet import UNetAdapter
from .nas.search_space import SegmentPolicy

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ============================================================
# NAS: best_arch 构建 student（复用 run_nas_search 的做法）
# ============================================================
def _build_unet_from_teacher_with_best_arch(teacher_unet: UNetModel, arch_obj) -> UNetModel:
    """
    用 NAS 的 best_arch 覆盖 UNetModel 的 attention_resolutions / num_res_blocks
    其余参数尽量从 teacher_unet 拷贝。
    """
    import inspect
    from ldm.modules.diffusionmodules.openaimodel import UNetModel as _UNetModel

    sig = inspect.signature(_UNetModel.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]

    # best_arch 兼容：你的 UNetArchConfig 里一般有 attn_ds / num_res_blocks
    if not hasattr(arch_obj, "attn_ds") or not hasattr(arch_obj, "num_res_blocks"):
        raise ValueError("best_arch.pkl 里对象缺少 attn_ds 或 num_res_blocks 字段，无法构建 student。")

    overrides = {
        "attention_resolutions": tuple(int(x) for x in arch_obj.attn_ds),
        "num_res_blocks": int(arch_obj.num_res_blocks),
    }

    kwargs = {}
    for p in params:
        name = p.name
        if name in overrides:
            kwargs[name] = overrides[name]
            continue

        if hasattr(teacher_unet, name):
            kwargs[name] = getattr(teacher_unet, name)
            continue

        alt_map = {
            "in_channels": "in_channels",
            "out_channels": "out_channels",
            "model_channels": "model_channels",
            "channel_mult": "channel_mult",
            "dropout": "dropout",
            "num_heads": "num_heads",
            "num_head_channels": "num_head_channels",
            "use_checkpoint": "use_checkpoint",
            "use_scale_shift_norm": "use_scale_shift_norm",
            "resblock_updown": "resblock_updown",
            "use_new_attention_order": "use_new_attention_order",
        }
        alt = alt_map.get(name, None)
        if alt is not None and hasattr(teacher_unet, alt):
            kwargs[name] = getattr(teacher_unet, alt)
            continue

        if p.default is not inspect._empty:
            continue

        raise KeyError(f"[best_arch] Cannot infer required UNetModel arg '{name}' from teacher_unet.")

    return _UNetModel(**kwargs)


def _load_partial_weights_shape_match(dst_unet: nn.Module, src_unet: nn.Module) -> int:
    """
    best_arch 改结构后，shape 可能不完全一致；按 shape match 拷贝 teacher 权重，最大化“只微调 teacher 权重”的效果。
    """
    dst_sd = dst_unet.state_dict()
    src_sd = src_unet.state_dict()
    matched = 0
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v
            matched += 1
    dst_unet.load_state_dict(dst_sd, strict=False)
    return matched


# ============================================================
# NAS: g_star step_policies 注入（按 t 映射 step_idx）
# ============================================================
def _load_pickle_if_exists(p: str):
    if p is None:
        return None
    if not Path(p).is_file():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _tau_to_step_index(t_scalar: int, tau_list: list) -> int:
    """
    将离散 t(0..999) 映射到 tau 的 step_idx（0..K-1）
    - 优先 exact match
    - 否则取最近的 tau
    """
    t_scalar = int(t_scalar)
    tau_int = [int(x) for x in tau_list]
    if t_scalar in tau_int:
        return tau_int.index(t_scalar)
    diffs = [abs(int(x) - t_scalar) for x in tau_int]
    return int(np.argmin(diffs))


def _segment_policy_from_step_policy(step_pol) -> SegmentPolicy:
    """
    g_star.step_policies[i] 本身通常就是 SegmentPolicy。
    这里做一次显式转换/兜底，保证类型正确。
    """
    if isinstance(step_pol, SegmentPolicy):
        return step_pol
    # 兜底：当 step_pol 是 dict/tuple 等
    lm = getattr(step_pol, "layer_mask", None) or step_pol.get("layer_mask")
    cg = getattr(step_pol, "stage_cond_gain", None) or step_pol.get("stage_cond_gain")
    gs = getattr(step_pol, "stage_gate_scale", None) or step_pol.get("stage_gate_scale")
    return SegmentPolicy(layer_mask=tuple(bool(x) for x in lm),
                        stage_cond_gain=tuple(float(x) for x in cg),
                        stage_gate_scale=tuple(float(x) for x in gs))


def _set_adapter_policy_for_t(adapter: UNetAdapter, g_star_obj, tau_list: list, t_batch: torch.Tensor):
    """
    t_batch: LongTensor [B]，但我们会假设 batch 内 t 相同（per_batch 模式）
    """
    if g_star_obj is None:
        return

    step_pols = getattr(g_star_obj, "step_policies", None)
    if not step_pols:
        return

    # 取 batch 的第一个 t
    t0 = int(t_batch[0].item())
    step_idx = _tau_to_step_index(t0, tau_list)
    step_idx = max(0, min(step_idx, len(step_pols) - 1))

    pol = _segment_policy_from_step_policy(step_pols[step_idx])
    adapter.set_policy(pol)


# ============================================================
# 原逻辑：student 初始化（teacher 同构）
# ============================================================
def build_student_unet_from_teacher(teacher_unet: UNetModel, unet_config) -> UNetModel:
    student_unet = instantiate_from_config(unet_config)
    if isinstance(student_unet, nn.DataParallel):
        student_unet = student_unet.module

    # 同构时 strict=True
    student_unet.load_state_dict(teacher_unet.state_dict(), strict=True)
    print("[Student] UNet initialized from teacher weights (strict=True).")
    return student_unet


def _build_scheduler(optimizer, cfg):
    if getattr(cfg, "lr_scheduler_type", "none") == "cosine":
        warmup = int(getattr(cfg, "warmup_steps", 0))
        total = int(getattr(cfg, "max_train_steps", 1000))
        min_ratio = float(getattr(cfg, "cosine_min_lr_ratio", 0.1))

        def lr_lambda(step_idx):
            if total <= 0:
                return 1.0
            if step_idx < warmup and warmup > 0:
                return max(1e-8, step_idx / float(warmup))
            progress = (step_idx - warmup) / float(max(1, total - warmup))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None


def _atomic_save(obj, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _maybe_remove_old_step_dirs(ckpt_dir: Path, keep: int):
    if keep is None or keep <= 0:
        return
    step_dirs = sorted([p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_")],
                       key=lambda p: p.stat().st_mtime)
    if len(step_dirs) > keep:
        for p in step_dirs[: len(step_dirs) - keep]:
            try:
                for child in p.glob("**/*"):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                for child_dir in sorted([d for d in p.glob("**/*") if d.is_dir()], reverse=True):
                    child_dir.rmdir()
                p.rmdir()
            except Exception as e:
                print(f"[CKPT][WARN] 删除旧 step 目录失败 {p}: {e}")


def _sample_t(pipe, tau, batch_size: int):
    """
    支持 per_batch / per_sample 两种 tau 采样模式。
    """
    mode = str(getattr(Config, "tau_sample_mode", "per_batch")).lower()
    if mode == "per_sample":
        return sample_t_from_tau(pipe, tau, batch_size=batch_size)
    # per_batch: 一个 t 扩展到 batch
    t1 = sample_t_from_tau(pipe, tau, batch_size=1)
    return t1.expand(batch_size)


def train(student_unet=None, tau=None, channel_scales=None):
    Config.validate()
    cfg = Config

    set_seed(getattr(cfg, "seed", 42), getattr(cfg, "deterministic", True))

    if torch.cuda.is_available() and getattr(cfg, "allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = cfg.device
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / getattr(cfg, "csv_log_name", "train_log.csv")

    writer = None
    if getattr(cfg, "enable_tensorboard", False) and SummaryWriter is not None:
        tb_dir = Path(cfg.tensorboard_log_dir) if getattr(cfg, "tensorboard_log_dir", None) else (out_dir / "tb")
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    # 数据集
    full_dataset = COCODataset(
        root=cfg.coco_root,
        split="val2014",
        image_size=cfg.image_size,
    )

    def _worker_init_fn(worker_id: int):
        base_seed = int(getattr(cfg, "seed", 42))
        ws = base_seed + worker_id
        try:
            np.random.seed(ws)
        except Exception:
            pass
        random.seed(ws)
        torch.manual_seed(ws)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(ws)

    rng = torch.Generator()
    rng.manual_seed(int(getattr(cfg, "seed", 42)))

    total_len = len(full_dataset)
    val_len = max(1, int(total_len * cfg.val_split_ratio)) if cfg.val_split_ratio > 0 else 0
    if val_len > 0:
        indices = torch.randperm(total_len, generator=rng)
        val_indices = indices[:val_len].tolist()
        train_indices = indices[val_len:].tolist()
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    else:
        train_dataset = full_dataset
        val_dataset = None

    dl_kwargs = {
        "batch_size": cfg.train_batch_size,
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "worker_init_fn": _worker_init_fn,
        "generator": rng,
    }
    if cfg.num_workers > 0:
        dl_kwargs.update({
            "prefetch_factor": getattr(cfg, "prefetch_factor", 2),
            "persistent_workers": getattr(cfg, "persistent_workers", True),
        })
    train_dataloader = DataLoader(train_dataset, **dl_kwargs)

    val_dataloader = None
    if val_dataset is not None:
        val_kwargs = dl_kwargs.copy()
        val_kwargs.update({"shuffle": False, "num_workers": max(0, cfg.num_workers // 2)})
        val_dataloader = DataLoader(val_dataset, **val_kwargs)

    # Teacher
    pipe = load_teacher_pipeline(cfg.model_id, device=device, use_fp16=cfg.use_fp16)
    pipe = pipe.to(device)
    teacher_unet = pipe.unet

    # =============================
    # 自动加载 NAS：best_arch / g_star
    # =============================
    best_arch_obj = None
    g_star_obj = None

    if getattr(cfg, "use_best_arch", False):
        best_arch_obj = _load_pickle_if_exists(getattr(cfg, "best_arch_pkl", None))
        if best_arch_obj is not None:
            print(f"[NAS] Loaded best_arch from: {cfg.best_arch_pkl}")
        else:
            print(f"[NAS][Fallback] best_arch not found, use teacher-isomorphic student.")

    if getattr(cfg, "use_step_policies", False):
        g_star_obj = _load_pickle_if_exists(getattr(cfg, "g_star_pkl", None))
        if g_star_obj is not None:
            sp = getattr(g_star_obj, "step_policies", None)
            print(f"[NAS] Loaded g_star from: {cfg.g_star_pkl} | step_policies={0 if not sp else len(sp)}")
        else:
            print(f"[NAS][Fallback] g_star not found, will not inject step_policies.")

    # =============================
    # Student：优先用 best_arch 构建，再尽量从 teacher shape-match 拷贝权重
    # =============================
    if student_unet is None:
        if best_arch_obj is not None:
            student_unet = _build_unet_from_teacher_with_best_arch(teacher_unet, best_arch_obj)
            student_unet.to(device=device, dtype=pipe.dtype)
            matched = _load_partial_weights_shape_match(student_unet, teacher_unet)
            print(f"[Student] Built from best_arch + partial teacher weight copy. matched_keys={matched}")
        else:
            student_unet = build_student_unet_from_teacher(teacher_unet, pipe.unet_config)
            student_unet.to(device=device, dtype=pipe.dtype)
            print("[Student] Built teacher-isomorphic student.")
    else:
        student_unet.to(device=device, dtype=pipe.dtype)
        print("[Student] Using externally provided student_unet.")

    # =============================
    # 关键：用 UNetAdapter 包住 student，便于注入 step_policies
    # =============================
    student_adapter = UNetAdapter(student_unet)

    # 注意：训练参数来自 student_unet（adapter 只是包装，不新建参数）
    student_unet.train()

    # 可选 torch.compile
    if getattr(cfg, "enable_torch_compile", False) and hasattr(torch, "compile"):
        try:
            student_unet = torch.compile(student_unet, mode="max-autotune", fullgraph=False)
            print("[Perf] torch.compile enabled")
        except Exception as e:
            print(f"[Perf][WARN] torch.compile failed: {e}")

    use_channels_last = bool(getattr(cfg, "channels_last", True)) and torch.cuda.is_available()

    # tau
    if tau is None:
        tau = load_tau(cfg.tau_path, cfg.default_tau)
        print(f"[Tau] Loaded: {tau[:10]}... (len={len(tau)})")
    else:
        print(f"[Tau] Using external tau: {tau[:10]}... (len={len(tau)})")

    # 优化器/调度/AMP
    optimizer = torch.optim.AdamW(
        student_unet.parameters(),
        lr=cfg.lr,
        weight_decay=getattr(cfg, "weight_decay", 0.0),
    )
    scheduler = _build_scheduler(optimizer, cfg)
    scaler = amp.GradScaler(enabled=(torch.cuda.is_available() and cfg.use_fp16))

    ckpt_dir = out_dir
    best_ckpt = ckpt_dir / "best.ckpt"
    best_student_path = ckpt_dir / "best_student_unet.pth"
    update_steps = 0
    start_epoch = 0
    best_val = float("inf") if getattr(cfg, "best_mode", "min") == "min" else -float("inf")

    def save_checkpoint(tag: str = "best"):
        save_path = ckpt_dir / f"{tag}.ckpt"
        state_obj = {
            "model": student_unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "update_steps": update_steps,
            "epoch": epoch,
            "best_val": best_val,
        }
        if getattr(cfg, "atomic_save", True):
            _atomic_save(state_obj, save_path)
        else:
            torch.save(state_obj, save_path)

    max_train_steps = cfg.max_train_steps
    grad_accum = max(1, int(cfg.gradient_accumulation_steps))

    progress_bar = tqdm(total=max_train_steps, desc="Training", dynamic_ncols=True)
    progress_bar.update(update_steps)

    accum_counter = 0
    optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def run_validation():
        if val_dataloader is None:
            return None
        student_unet.eval()
        total = 0.0
        n = 0
        for i, batch in enumerate(val_dataloader):
            if i >= getattr(cfg, "val_batches", 10):
                break
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            if use_channels_last:
                pixel_values = pixel_values.to(memory_format=torch.channels_last)

            z0 = encode_image_to_latent(pipe, pixel_values)
            B = z0.shape[0]
            captions = batch["caption"]
            cond_emb = encode_text(pipe, captions, device=device)
            uncond_emb = get_uncond_embeddings(pipe, batch_size=B, device=device)

            t = _sample_t(pipe, tau, batch_size=B)

            # 注入 step policy（如果启用 & 存在）
            if g_star_obj is not None and getattr(cfg, "use_step_policies", False):
                _set_adapter_policy_for_t(student_adapter, g_star_obj, tau, t)

            alphas_cumprod = pipe.alphas_cumprod.to(device)
            alpha_t = alphas_cumprod[t].sqrt().view(B, 1, 1, 1)
            sigma_t = (1.0 - alphas_cumprod[t]).sqrt().view(B, 1, 1, 1)
            noise = torch.randn_like(z0)
            z_t = alpha_t * z0 + sigma_t * noise

            with amp.autocast(enabled=(torch.cuda.is_available() and cfg.use_fp16)):
                eps_teacher_cfg = teacher_eps(
                    pipe=pipe,
                    z_t=z_t,
                    t=t,
                    cond_emb=cond_emb,
                    uncond_emb=uncond_emb,
                    cfg_scale=cfg.cfg_scale,
                )
                # student 用 adapter.forward 以应用 mask/gain/gate
                eps_student_cfg = student_adapter.forward(z_t, t, context=cond_emb)

            loss_d = distillation_loss(eps_student_cfg.float(), eps_teacher_cfg.float())
            loss_c = consistency_loss(
                student_unet=student_unet,
                z0=z0,
                tau=tau,
                cond_emb=cond_emb,
                alphas_cumprod=alphas_cumprod,
            )
            loss = loss_d + cfg.lambda_consistency * loss_c
            total += float(loss.item())
            n += 1

        student_unet.train()
        return total / max(1, n)

    def append_csv(step, epoch_idx, loss_val, loss_d_val, loss_c_val, lr_val, val_loss_val):
        # 若不存在加表头
        if not csv_path.exists():
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("step,epoch,loss,loss_d,loss_c,lr,val_loss\n")
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{step},{epoch_idx},{loss_val:.6f},{loss_d_val:.6f},{loss_c_val:.6f},{lr_val:.8f},"
                    f"{'' if val_loss_val is None else f'{val_loss_val:.6f}'}\n")

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            for batch in train_dataloader:
                if update_steps >= max_train_steps:
                    break

                try:
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    if use_channels_last:
                        pixel_values = pixel_values.to(memory_format=torch.channels_last)
                    captions = batch["caption"]

                    # VAE encode
                    z0 = encode_image_to_latent(pipe, pixel_values)
                    B = z0.shape[0]

                    # text
                    cond_emb = encode_text(pipe, captions, device=device)
                    uncond_emb = get_uncond_embeddings(pipe, batch_size=B, device=device)

                    # sample t
                    t = _sample_t(pipe, tau, batch_size=B)

                    # 注入 step policy（如果启用 & 存在）
                    if g_star_obj is not None and getattr(cfg, "use_step_policies", False):
                        _set_adapter_policy_for_t(student_adapter, g_star_obj, tau, t)

                    # z_t
                    alphas_cumprod = pipe.alphas_cumprod.to(device)
                    alpha_t = alphas_cumprod[t].sqrt().view(B, 1, 1, 1)
                    sigma_t = (1.0 - alphas_cumprod[t]).sqrt().view(B, 1, 1, 1)
                    noise = torch.randn_like(z0)
                    z_t = alpha_t * z0 + sigma_t * noise

                    # forward
                    with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and cfg.use_fp16)):
                        eps_teacher_cfg = teacher_eps(
                            pipe=pipe,
                            z_t=z_t,
                            t=t,
                            cond_emb=cond_emb,
                            uncond_emb=uncond_emb,
                            cfg_scale=cfg.cfg_scale,
                        )
                        eps_student_cfg = student_adapter.forward(z_t, t, context=cond_emb)

                    loss_d = distillation_loss(eps_student_cfg.float(), eps_teacher_cfg.float())

                    loss_c = consistency_loss(
                        student_unet=student_unet,
                        z0=z0,
                        tau=tau,
                        cond_emb=cond_emb,
                        alphas_cumprod=alphas_cumprod,
                    )
                    total_loss = loss_d + cfg.lambda_consistency * loss_c

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print("[WARN] NaN/Inf loss, skip batch.")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    # grad accum
                    loss = total_loss / grad_accum
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    accum_counter += 1

                    if accum_counter % grad_accum == 0:
                        if getattr(cfg, "max_grad_norm", 0.0) and cfg.max_grad_norm > 0:
                            if scaler.is_enabled():
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(student_unet.parameters(), cfg.max_grad_norm)

                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                        optimizer.zero_grad(set_to_none=True)
                        accum_counter = 0
                        update_steps += 1
                        progress_bar.update(1)

                        current_lr = optimizer.param_groups[0]["lr"]

                        if update_steps % cfg.log_every == 0:
                            progress_bar.set_postfix({
                                "loss": float(total_loss.item()),
                                "loss_d": float(loss_d.item()),
                                "loss_c": float(loss_c.item()) if cfg.lambda_consistency > 0 else 0.0,
                                "lr": current_lr,
                            })
                            append_csv(update_steps, epoch, float(total_loss.item()),
                                       float(loss_d.item()),
                                       float(loss_c.item()),
                                       current_lr, None)
                            if writer is not None:
                                writer.add_scalar("train/loss", float(total_loss.item()), update_steps)
                                writer.add_scalar("train/loss_d", float(loss_d.item()), update_steps)
                                writer.add_scalar("train/loss_c", float(loss_c.item()), update_steps)
                                writer.add_scalar("train/lr", current_lr, update_steps)

                        if val_dataloader is not None and cfg.val_every > 0 and update_steps % cfg.val_every == 0:
                            val_loss = run_validation()
                            if val_loss is not None:
                                append_csv(update_steps, epoch, float(total_loss.item()),
                                           float(loss_d.item()),
                                           float(loss_c.item()),
                                           current_lr, float(val_loss))
                                if writer is not None:
                                    writer.add_scalar("val/loss", float(val_loss), update_steps)

                                is_better = (val_loss < best_val) if cfg.best_mode == "min" else (val_loss > best_val)
                                if is_better:
                                    best_val = float(val_loss)
                                    if getattr(cfg, "atomic_save", True):
                                        _atomic_save(student_unet.state_dict(), best_student_path)
                                        _atomic_save({
                                            "model": student_unet.state_dict(),
                                            "optimizer": optimizer.state_dict(),
                                            "scheduler": scheduler.state_dict() if scheduler is not None else None,
                                            "update_steps": update_steps,
                                            "epoch": epoch,
                                            "best_val": best_val,
                                        }, best_ckpt)
                                    else:
                                        torch.save(student_unet.state_dict(), best_student_path)
                                        torch.save({
                                            "model": student_unet.state_dict(),
                                            "optimizer": optimizer.state_dict(),
                                            "scheduler": scheduler.state_dict() if scheduler is not None else None,
                                            "update_steps": update_steps,
                                            "epoch": epoch,
                                            "best_val": best_val,
                                        }, best_ckpt)
                                    print(f"[BEST] step={update_steps} new best {cfg.best_metric}={best_val:.6f}")

                        if cfg.save_every and update_steps % cfg.save_every == 0:
                            step_dir = ckpt_dir / f"step_{update_steps}"
                            step_dir.mkdir(parents=True, exist_ok=True)
                            model_path = step_dir / "student_unet.pth"
                            if getattr(cfg, "atomic_save", True):
                                _atomic_save(student_unet.state_dict(), model_path)
                            else:
                                torch.save(student_unet.state_dict(), model_path)
                            _maybe_remove_old_step_dirs(ckpt_dir, getattr(cfg, "keep_last_n_step_ckpts", 3))

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("[WARN] CUDA OOM, skip batch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise

            if update_steps >= max_train_steps:
                break

        # flush last accum
        if accum_counter > 0 and update_steps < max_train_steps:
            if getattr(cfg, "max_grad_norm", 0.0) and cfg.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_unet.parameters(), cfg.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            update_steps += 1
            progress_bar.update(1)

        step_dir = ckpt_dir / f"step_{update_steps}"
        step_dir.mkdir(parents=True, exist_ok=True)
        model_path = step_dir / "student_unet.pth"
        if getattr(cfg, "atomic_save", True):
            _atomic_save(student_unet.state_dict(), model_path)
        else:
            torch.save(student_unet.state_dict(), model_path)

        _maybe_remove_old_step_dirs(ckpt_dir, getattr(cfg, "keep_last_n_step_ckpts", 3))
        print(f"[INFO] Saved student UNet checkpoint at {step_dir}")

    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user.")
        if student_unet is not None:
            if getattr(cfg, "atomic_save", True):
                _atomic_save(student_unet.state_dict(), model_path)
            else:
                torch.save(student_unet.state_dict(), model_path)
            _maybe_remove_old_step_dirs(ckpt_dir, getattr(cfg, "keep_last_n_step_ckpts", 3))
            print(f"[INFO] Saved student UNet checkpoint at {step_dir}")
        raise
    except Exception as e:
        print(f"[ERROR] Training exception: {e}")
        if student_unet is not None:
            if getattr(cfg, "atomic_save", True):
                _atomic_save(student_unet.state_dict(), model_path)
            else:
                torch.save(student_unet.state_dict(), model_path)
            _maybe_remove_old_step_dirs(ckpt_dir, getattr(cfg, "keep_last_n_step_ckpts", 3))
            print(f"[INFO] Saved student UNet checkpoint at {step_dir}")
        raise
    finally:
        progress_bar.close()
        if writer is not None:
            writer.flush()
            writer.close()
        print("[INFO] Training finished.")


if __name__ == "__main__":
    train()
