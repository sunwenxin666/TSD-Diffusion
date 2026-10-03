# -*- coding: utf-8 -*-
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class COCODataset(Dataset):

    def __init__(self, root, split="val2014", image_size=512):
        """
        Args:
            root (str or Path): COCO 数据集根目录
            split (str): 使用的划分，通常为 "val2014"
            image_size (int): 图像缩放到的分辨率（例如 512）
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        # 1) 图片目录
        if split in ["val", "val2014"]:
            self.img_dir = self.root / "val2014"
        else:
            raise ValueError(f"Unsupported split for this dataset layout: {split}")

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # 2) 标注文件（30K captions）
        ann_path = self.root / "captions_val2014_30K_samples.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON 解码失败: {ann_path}, 错误: {e}")

        self.samples = []  # (file_name, caption)

        # 2.1) 解析 JSON
        if isinstance(data, dict) and "annotations" in data and "images" in data:
            images = data["images"]
            annotations = data["annotations"]
            id_to_filename = {img["id"]: img.get("file_name") for img in images}
            missing = 0
            for ann in annotations:
                image_id = ann.get("image_id")
                caption = ann.get("caption", "")
                file_name = id_to_filename.get(image_id)
                if not file_name:
                    missing += 1
                    continue
                self.samples.append((file_name, caption))
            if missing > 0:
                print(f"[COCODataset][WARN] {missing} annotations 缺失对应图片，已跳过")
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                file_name = item.get("file_name")
                caption = item.get("caption", "")
                if not file_name:
                    continue
                self.samples.append((file_name, caption))
        else:
            raise ValueError("Unsupported JSON format for captions_val2014_30K_samples.json")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples parsed from {ann_path}")

        print(f"[COCODataset] root={self.root}")
        print(f"[COCODataset] img_dir={self.img_dir}")
        print(f"[COCODataset] samples={len(self.samples)}")

        # 3) 预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 最多重试 5 次（跳过损坏样本）
        attempts = 0
        last_err = None
        while attempts < 5:
            file_name, caption = self.samples[idx]
            img_path = self.img_dir / file_name
            try:
                if not img_path.is_file():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                with Image.open(img_path) as im:
                    image = im.convert("RGB")
                image = self.image_transform(image)
                return {
                    "pixel_values": image,  # [3,H,W] in [-1,1]
                    "caption": caption,
                }
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                # 记录最后一次错误，并尝试下一个索引
                last_err = e
                attempts += 1
                idx = (idx + 1) % len(self.samples)
                continue
            except Exception as e:
                # 其它异常直接抛出
                raise e
        # 多次重试失败，抛出可读错误
        raise RuntimeError(f"加载图像失败（尝试 {attempts} 次）：{last_err}")
