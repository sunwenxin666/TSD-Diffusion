# sd_autodiffusion_distill/distill/utils/prompts.py

import json
import os
from typing import Any, Dict, List, Union, Tuple

# 获取项目根目录
# 注意：prompts.py 在 distill/utils/ 目录下，需要向上 4 层才能到达项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_PROMPTS_JSON = os.path.join(
    ROOT_DIR,
    "examples/Stable Diffusion/path/to/dataset/captions_val2014_30K_samples.json"
)

# 最终统一返回：List[List[str]] （每组是一张图的多个 caption）
PromptsType = List[List[str]]


def _as_groups_from_list(data: List[Any]) -> PromptsType:
    """data is list[str] or list[list[str]]"""
    if len(data) == 0:
        raise ValueError("Prompts list is empty.")

    if isinstance(data[0], str):
        # list[str] -> wrap each as group
        return [[s] for s in data if isinstance(s, str) and len(s) > 0]

    if isinstance(data[0], list):
        groups: PromptsType = []
        for g in data:
            if not isinstance(g, list):
                continue
            gg = [s for s in g if isinstance(s, str) and len(s) > 0]
            if len(gg) > 0:
                groups.append(gg)
        if len(groups) == 0:
            raise ValueError("Prompts list-of-lists contains no valid strings.")
        return groups

    raise ValueError(f"Unsupported list element type: {type(data[0])}")


def _as_groups_from_annotations_dict(d: Dict[str, Any]) -> PromptsType:
    """
    Handle common COCO-style dict:
      {"annotations":[{"caption":...,"image_id":...}, ...]}
    Return groups by image_id.
    """
    ann = d.get("annotations", None)
    if not isinstance(ann, list) or len(ann) == 0:
        raise ValueError("Dict has 'annotations' but it's empty or not a list.")

    groups_by_id: Dict[Any, List[str]] = {}
    for item in ann:
        if not isinstance(item, dict):
            continue
        cap = item.get("caption", None)
        img_id = item.get("image_id", None)
        if not isinstance(cap, str) or len(cap) == 0:
            continue
        # img_id might be missing in some files; fallback to None bucket
        key = img_id if img_id is not None else "__no_image_id__"
        groups_by_id.setdefault(key, []).append(cap)

    groups = [caps for caps in groups_by_id.values() if len(caps) > 0]
    if len(groups) == 0:
        raise ValueError("No valid captions found under annotations.")
    return groups


def _as_groups_from_generic_dict(d: Dict[str, Any]) -> PromptsType:
    """
    Handle other common dict formats:
      - {"captions":[...]} / {"prompts":[...]}  (list[str] or list[list[str]])
      - {"imgid":[cap1,cap2,...], "imgid2":[...], ...}  (dict values are list[str])
      - {"imgid":{"captions":[...]} ...}
    """
    # 1) direct keys
    for k in ("prompts", "captions", "texts"):
        if k in d:
            v = d[k]
            if isinstance(v, list):
                return _as_groups_from_list(v)

    # 2) COCO annotations
    if "annotations" in d:
        return _as_groups_from_annotations_dict(d)

    # 3) dict-of-lists or dict-of-dicts
    groups: PromptsType = []
    for _, v in d.items():
        if isinstance(v, list):
            # list[str] or list[list[str]]
            try:
                groups.extend(_as_groups_from_list(v))
            except Exception:
                continue
        elif isinstance(v, dict):
            # maybe {"captions":[...]}
            try:
                groups.extend(_as_groups_from_generic_dict(v))
            except Exception:
                continue

    if len(groups) == 0:
        raise ValueError(
            "Unsupported dict structure for prompts JSON. "
            "Expected keys like 'annotations' or 'captions/prompts', or dict values as lists."
        )
    return groups


def load_prompts_from_json(path: str = DEFAULT_PROMPTS_JSON) -> PromptsType:
    """
    Load prompts from JSON and normalize to List[List[str]].

    Supported JSON top-level types:
      - list[str]
      - list[list[str]]
      - dict (common COCO or custom grouped formats)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompts JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, list):
        groups = _as_groups_from_list(data)
    elif isinstance(data, dict):
        groups = _as_groups_from_generic_dict(data)
    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(data)}")

    # final sanity
    if len(groups) == 0:
        raise ValueError("No valid prompts loaded (empty after normalization).")

    return groups
