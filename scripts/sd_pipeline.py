"""Portable entry points for the retained SD1.5 implementation."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/Stable Diffusion"))
sys.path.insert(0, str(ROOT / "examples/Stable Diffusion/taming-transformers"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["search", "train", "sample", "evaluate"])
    parser.add_argument("--model", default=os.getenv("SD_MODEL_PATH"))
    parser.add_argument("--clip-model", default=os.getenv("CLIP_MODEL_PATH"))
    parser.add_argument("--coco-root", default=os.getenv("COCO_ROOT"))
    parser.add_argument("--prompts", default=os.getenv("COCO_PROMPTS_JSON"))
    parser.add_argument("--tau", default=os.getenv("TAU_PATH"))
    parser.add_argument("--nas-dir", default=os.getenv("NAS_DIR", str(ROOT / "artifacts/sd/search")))
    parser.add_argument("--train-dir", default=os.getenv("OUTPUT_DIR", str(ROOT / "artifacts/sd/train")))
    parser.add_argument("--sample-dir", default=os.getenv("SD_SAMPLE_DIR", str(ROOT / "artifacts/sd/samples")))
    parser.add_argument("--student-ckpt")
    parser.add_argument("--generated-paths", help="JSON list of saved generated images for offline evaluation")
    parser.add_argument("--num-images", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    resources = {"SD_MODEL_PATH": args.model, "CLIP_MODEL_PATH": args.clip_model,
                 "COCO_ROOT": args.coco_root, "COCO_PROMPTS_JSON": args.prompts,
                 "TAU_PATH": args.tau, "NAS_DIR": args.nas_dir,
                 "OUTPUT_DIR": args.train_dir, "SD_SAMPLE_DIR": args.sample_dir}
    for key, value in resources.items():
        if value is not None:
            os.environ[key] = str(Path(value).resolve())
    if args.stage in {"sample", "evaluate"} and not args.student_ckpt:
        parser.error("--student-ckpt is required for sample/evaluate")
    if args.num_images <= 0 or args.batch_size <= 0:
        parser.error("Sample count and batch size must be positive")
    for key in ("SD_MODEL_PATH", "CLIP_MODEL_PATH", "TAU_PATH"):
        value = os.environ.get(key)
        if not value or not Path(value).exists():
            parser.error(f"Set {key} to an existing external resource before running this stage")
    if args.stage in {"search", "train", "sample"}:
        for name in ("best_arch.pkl", "g_star.pkl") if args.stage != "search" else ():
            if not (Path(args.nas_dir) / name).is_file():
                parser.error(f"Missing searched artifact: {Path(args.nas_dir) / name}")
    if args.stage == "search":
        from sd_autodiffusion_distill.distill.run_nas_search import run_search
        run_search()
    elif args.stage == "train":
        from sd_autodiffusion_distill.distill.train_distill import train
        train()
    elif args.stage == "sample":
        from sd_autodiffusion_distill.distill.generate_student_images import generate_student_images_aligned
        if not args.coco_root or not args.prompts:
            parser.error("--coco-root and --prompts are required for sampling")
        generate_student_images_aligned(
            student_ckpt_path=args.student_ckpt, coco_captions_json=args.prompts,
            coco_val_image_dir=str(Path(args.coco_root) / "val2014"),
            num_images=args.num_images, batch_size=args.batch_size,
            save_dir=args.sample_dir, compute_clip=True, clip_model_path=args.clip_model)
    else:
        from sd_autodiffusion_distill.distill.evaluate_vs_coco import evaluate_vs_coco_aligned_official
        if not args.coco_root:
            parser.error("--coco-root is required for evaluation")
        if not args.generated_paths:
            parser.error("--generated-paths is required; streaming sampling does not save all images")
        metadata_dir = Path(args.sample_dir).parent
        evaluate_vs_coco_aligned_official(
            coco_val_image_dir=str(Path(args.coco_root) / "val2014"),
            generated_paths_json=args.generated_paths,
            selected_file_names_json=str(metadata_dir / "selected_file_names.json"),
            selected_captions_json=str(metadata_dir / "selected_captions.json"),
            student_ckpt_path=args.student_ckpt, clip_model_path=args.clip_model)

if __name__ == "__main__":
    main()
