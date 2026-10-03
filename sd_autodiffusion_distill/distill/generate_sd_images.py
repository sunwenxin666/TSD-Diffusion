import argparse
import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
import time
from pytorch_lightning import seed_everything
import json
import gc
import random
from pathlib import Path

# 添加项目路径以便正确导入
# 获取AutoDiffusion-main的根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回退两级到AutoDiffusion-main根目录
stable_diffusion_path = os.path.join(project_root, "examples", "Stable Diffusion")
if stable_diffusion_path not in sys.path:
    sys.path.insert(0, stable_diffusion_path)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# 导入ACDMS采样相关函数（用于公平对比student模型）
from sd_autodiffusion_distill.distill.teacher import (
    load_teacher_pipeline,
    encode_text,
    get_uncond_embeddings,
    load_tau,
    teacher_acdms_sample,
)
from sd_autodiffusion_distill.distill.train_config import Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def load_model_from_config(config_path, ckpt, verbose=False):
    config = OmegaConf.load(config_path)
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

def load_coco_caption_pairs(coco_captions_json: str):
    """
    读取 COCO captions JSON，返回：
    - id2file: image_id -> file_name
    - id2captions: image_id -> [caption1, caption2, ...]
    """
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

@torch.no_grad()
def generate_images_in_batches(model, sampler, prompts, selected_ids, selected_files, batch_size, num_images,
                               image_size=512, ddim_steps=50, scale=7.5, device="cuda"):
    """
    分批生成图像，避免内存溢出，对齐COCO（使用DDIM/PLMS采样）
    """
    all_images = []
    all_prompts = []
    used_ids = []
    used_files = []
    used_caps = []

    num_batches = (num_images + batch_size - 1) // batch_size
    
    print(f"Generating {num_images} images in {num_batches} batches of size {batch_size}")
    
    latent_h = image_size // 8
    latent_w = image_size // 8

    with torch.no_grad():
        for start in trange(0, num_images, batch_size, desc="Generating (Aligned)"):
            bs = min(batch_size, num_images - start)
            batch_caps = prompts[start:start + bs]
            batch_ids = selected_ids[start:start + bs]
            batch_files = selected_files[start:start + bs]
            
            uc = None
            if scale != 1.0:
                uc = model.get_learned_conditioning(bs * [""])
            c = model.get_learned_conditioning(batch_caps)
            shape = [4, latent_h, latent_w]
            samples, _ = sampler.sample(S=ddim_steps, conditioning=c, batch_size=bs, shape=shape,
                                        unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=0.0)
            images = model.decode_first_stage(samples)
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images = images.cpu()
            images_np = images.permute(0, 2, 3, 1).numpy()
            
            for j in range(bs):
                iid = batch_ids[j]
                img_array = (images_np[j] * 255).astype(np.uint8)
                all_images.append(img_array)
                all_prompts.append(batch_caps[j])
                used_ids.append(int(iid))
                used_files.append(batch_files[j])
                used_caps.append(batch_caps[j])
            
            del images, images_np, samples
            torch.cuda.empty_cache()
            gc.collect()
    
    return all_images, all_prompts, used_ids, used_files, used_caps


@torch.no_grad()
def generate_images_with_acdms(pipe, prompts, selected_ids, selected_files, batch_size, num_images,
                               image_size=512, num_inference_steps=6, cfg_scale=4.5, device="cuda", use_fp16=False):
    """
    使用ACDMS采样分批生成图像，避免内存溢出，对齐COCO
    """
    all_images = []
    all_prompts = []
    used_ids = []
    used_files = []
    used_caps = []

    num_batches = (num_images + batch_size - 1) // batch_size
    
    print(f"Generating {num_images} images with ACDMS in {num_batches} batches of size {batch_size}")
    print(f"Using FP16: {use_fp16}")
    
    # 加载tau配置
    cfg = Config()
    tau_full = load_tau(cfg.tau_path, cfg.default_tau)
    tau = tau_full[: num_inference_steps + 1]
    
    latent_h = image_size // 8
    latent_w = image_size // 8

    dtype = torch.float16 if use_fp16 else torch.float32
    pipe.to(device=device, dtype=dtype)

    with torch.no_grad():
        for start in trange(0, num_images, batch_size, desc="Generating (ACDMS)"):
            bs = min(batch_size, num_images - start)
            batch_caps = prompts[start:start + bs]
            batch_ids = selected_ids[start:start + bs]
            batch_files = selected_files[start:start + bs]
            
            # 初始化latents
            latents = torch.randn((bs, 4, latent_h, latent_w), device=device, dtype=dtype)
            
            # 编码文本
            cond_emb = encode_text(pipe, batch_caps, device=device)
            uncond_emb = get_uncond_embeddings(pipe, batch_size=bs, device=device)
            
            # ACDMS采样
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available() and use_fp16):
                images = teacher_acdms_sample(
                    pipe=pipe,
                    latents=latents,
                    tau=tau,
                    cond_emb=cond_emb,
                    uncond_emb=uncond_emb,
                    cfg_scale=cfg_scale,
                    num_inference_steps=num_inference_steps,
                    fm_strength=cfg.acdms_fm_strength,
                    fm_t_start=cfg.acdms_fm_t_start,
                )
            
            # 归一化到[0, 1]
            if images.min() < -0.1:
                images = (images / 2 + 0.5)
            images = images.clamp(0, 1).cpu()
            images_np = images.permute(0, 2, 3, 1).numpy()
            
            for j in range(bs):
                iid = batch_ids[j]
                img_array = (images_np[j] * 255).astype(np.uint8)
                all_images.append(img_array)
                all_prompts.append(batch_caps[j])
                used_ids.append(int(iid))
                used_files.append(batch_files[j])
                used_caps.append(batch_caps[j])
            
            del images, images_np, latents, cond_emb, uncond_emb
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
    
    return all_images, all_prompts, used_ids, used_files, used_caps

def save_images(images, used_ids, output_dir):
    """保存生成的图像，使用image_id命名"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(images)} images to {output_dir}")
    
    generated_paths = []
    for i, (img_array, iid) in enumerate(zip(images, used_ids)):
        filepath = os.path.join(output_dir, f"{iid}.png")
        img = Image.fromarray(img_array)
        img.save(filepath)
        generated_paths.append(filepath)
        
        if (i + 1) % 100 == 0:
            print(f"Saved {i + 1} images...")
    
    print("All images saved successfully!")
    return generated_paths

def main():
    parser = argparse.ArgumentParser(description="Generate images with SD v1.4 aligned to COCO")
    
    # 模型参数
    parser.add_argument("--config", type=str,
                        default="examples/Stable Diffusion/configs/stable-diffusion/v1-inference.yaml",
                        help="Path to model config")
    parser.add_argument("--ckpt", type=str,
                        default="examples/Stable Diffusion/models/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
                        help="Path to model checkpoint")
    
    # 生成参数
    parser.add_argument("--num_images", type=int, default=30000,
                        help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size (both height and width)")
    parser.add_argument("--ddim_steps", type=int, default=4,
                        help="Number of sampling steps (for DDIM/PLMS)")
    parser.add_argument("--scale", type=float, default=4.5,
                        help="Guidance scale (CFG)")
    parser.add_argument("--use_acdms", action="store_true",
                        help="Use ACDMS sampling instead of DDIM/PLMS")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use FP16 precision for generation")
    
    # COCO参数
    parser.add_argument("--coco_captions_json", type=str,
                        default="examples/Stable Diffusion/path/to/dataset/captions_val2014_30K_samples.json",
                        help="COCO captions JSON file")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/sd_generated_aligned",
                        help="Directory to save generated images")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sampler", type=str, default="ddim",
                        choices=["ddim", "plms"],
                        help="Sampling method")
    
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    _set_all_seeds(int(args.seed), deterministic=True)
    
    # 加载COCO对齐
    print("Loading COCO captions...")
    id2file, id2captions = load_coco_caption_pairs(args.coco_captions_json)
    all_ids = sorted([iid for iid in id2file.keys() if iid in id2captions])
    if len(all_ids) < args.num_images:
        raise RuntimeError(f"[Error] 可用 image_id 数量 {len(all_ids)} < num_images {args.num_images}")
    selected_ids = all_ids[:args.num_images]
    selected_caps = [id2captions[iid][0] for iid in selected_ids]
    selected_files = [id2file[iid] for iid in selected_ids]
    
    print(f"Using {len(selected_caps)} prompts from COCO for generation")
    
    # 根据采样方法选择不同的生成流程
    if args.use_acdms:
        # 使用ACDMS采样（推荐，用于公平对比student模型）
        print("Using ACDMS sampling for generation...")
        print(f"Loading teacher pipeline with model_id: {args.config}")
        
        # 加载teacher pipeline（用于ACDMS采样）
        pipe = load_teacher_pipeline(args.ckpt, device=device, use_fp16=True)
        
        # 生成图像
        print("Starting image generation with ACDMS...")
        start_time = time.time()
        
        images, used_prompts, used_ids, used_files, used_caps = generate_images_with_acdms(
            pipe=pipe,
            prompts=selected_caps,
            selected_ids=selected_ids,
            selected_files=selected_files,
            batch_size=args.batch_size,
            num_images=args.num_images,
            image_size=args.image_size,
            num_inference_steps=args.ddim_steps,  # 这里复用ddim_steps作为num_inference_steps
            cfg_scale=args.scale,
            device=device,
            use_fp16=args.use_fp16
        )
    else:
        # 使用DDIM/PLMS采样
        print("Loading model...")
        model = load_model_from_config(args.config, args.ckpt)
        model.to(device)
        
        # 选择sampler
        if args.sampler == "ddim":
            sampler = DDIMSampler(model)
        elif args.sampler == "plms":
            sampler = PLMSSampler(model)
        else:
            raise ValueError("Invalid sampler")
        
        # 生成图像
        print("Starting image generation...")
        start_time = time.time()
        
        images, used_prompts, used_ids, used_files, used_caps = generate_images_in_batches(
            model=model,
            sampler=sampler,
            prompts=selected_caps,
            selected_ids=selected_ids,
            selected_files=selected_files,
            batch_size=args.batch_size,
            num_images=args.num_images,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            scale=args.scale,
            device=device
        )
    
    generation_time = time.time() - start_time
    print(f"Image generation completed in {generation_time:.2f} seconds")
    
    # 保存图像
    print("Saving images...")
    generated_paths = save_images(images, used_ids, args.output_dir)
    
    # 保存对齐信息
    out_parent = Path(args.output_dir).parent
    with open(out_parent / "generated_paths.json", 'w', encoding="utf-8") as f:
        json.dump(generated_paths, f, indent=2, ensure_ascii=False)
    with open(out_parent / "selected_image_ids.json", 'w', encoding="utf-8") as f:
        json.dump(used_ids, f, indent=2, ensure_ascii=False)
    with open(out_parent / "selected_file_names.json", 'w', encoding="utf-8") as f:
        json.dump(used_files, f, indent=2, ensure_ascii=False)
    with open(out_parent / "selected_captions.json", 'w', encoding="utf-8") as f:
        json.dump(used_caps, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON files saved to: {out_parent}")
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()