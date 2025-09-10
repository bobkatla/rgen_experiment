from __future__ import annotations
import os
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import CIFAR10

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _save_pil(img: Image.Image, path: str) -> None:
    # Always save as PNG to avoid JPEG artifacts
    img.save(path, format="PNG", compress_level=0)

def export_cifar10_as_images(
    root_dir: str,
    splits: Tuple[str, ...] = ("train", "test"),
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Download CIFAR-10 (if needed) and export to:
      {root_dir}/cifar10/{split}/{class_name}/{index:06d}.png
    Returns a dict with image counts per split.
    """
    out_root = os.path.join(root_dir, "cifar10")
    _ensure_dir(out_root)

    summary: Dict[str, int] = {}
    for split in splits:
        is_train = split == "train"
        ds = CIFAR10(root=out_root, train=is_train, download=True)
        classes: List[str] = ds.classes

        split_dir = os.path.join(out_root, split)
        _ensure_dir(split_dir)

        # Prepare class subdirs
        for cname in classes:
            _ensure_dir(os.path.join(split_dir, cname))

        n = len(ds)
        kept = 0
        bar = tqdm(range(n), desc=f"Export CIFAR-10 {split}", ncols=80)
        for i in bar:
            img, label = ds[i]  # img is PIL.Image
            cname = classes[label]
            out_path = os.path.join(split_dir, cname, f"{i:06d}.png")
            if (not overwrite) and os.path.exists(out_path):
                kept += 1
                continue
            _save_pil(img, out_path)
            kept += 1

        summary[split] = kept

    return summary
