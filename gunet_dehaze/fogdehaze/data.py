"""Paired hazy/clear dataset.

Expected layout (same as the official gUNet data):

    <root>/
        IN/   hazy images
        GT/   clear images

Hazy files are matched to clear files either by identical file name, or by
stripping trailing "_xxx" parts of the stem. That covers:
    * gUNet prepared data:   IN/0001.png        -> GT/0001.png
    * RESIDE ITS raw:        IN/1_1_0.90179.png -> GT/1.png
    * RESIDE OTS raw:        IN/0001_0.8_0.2.jpg -> GT/0001.jpg
    * our synthetic fog:     IN/cam_0042_3.png  -> GT/cam_0042.png
"""
import random
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .utils import IMG_EXTS, read_rgb, to_tensor


def _list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


HAZY_DIR_NAMES = ("IN", "hazy", "haze", "input", "Hazy")
CLEAR_DIR_NAMES = ("GT", "gt", "clear", "clean", "target", "Clear")


def _find_dir(root, names):
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


def build_pairs(root):
    root = Path(root)
    hazy_dir, clear_dir = _find_dir(root, HAZY_DIR_NAMES), _find_dir(root, CLEAR_DIR_NAMES)
    if hazy_dir is None or clear_dir is None:
        raise FileNotFoundError(f"{root} must contain a hazy folder {HAZY_DIR_NAMES} "
                                f"and a clear folder {CLEAR_DIR_NAMES}")

    clear_by_name = {p.name: p for p in _list_images(clear_dir)}
    clear_by_stem = {p.stem: p for p in clear_by_name.values()}

    pairs, unmatched = [], 0
    for hz in _list_images(hazy_dir):
        gt = clear_by_name.get(hz.name) or clear_by_stem.get(hz.stem)
        if gt is None:
            parts = hz.stem.split("_")
            for k in range(len(parts) - 1, 0, -1):
                gt = clear_by_stem.get("_".join(parts[:k]))
                if gt is not None:
                    break
        if gt is None:
            unmatched += 1
        else:
            pairs.append((hz, gt))

    if not pairs:
        raise RuntimeError(f"no hazy/clear pairs found in {root}")
    if unmatched:
        print(f"[data] warning: {unmatched} hazy images in {hazy_dir} had no matching clear image")
    return pairs


class PairDataset(Dataset):
    """mode='train': random crop + flip.  mode='valid': center crop.  mode='test': full image."""

    def __init__(self, root, mode="train", patch_size=256, hflip=True, max_items=None, seed=0):
        assert mode in ("train", "valid", "test")
        self.pairs = build_pairs(root)
        if max_items and len(self.pairs) > max_items:
            rng = random.Random(seed)
            self.pairs = rng.sample(self.pairs, max_items)
        self.mode = mode
        self.size = patch_size
        self.hflip = hflip

    def __len__(self):
        return len(self.pairs)

    def _ensure_min_size(self, imgs):
        h, w = imgs[0].shape[:2]
        if h >= self.size and w >= self.size:
            return imgs
        s = self.size / min(h, w)
        nh, nw = max(self.size, round(h * s)), max(self.size, round(w * s))
        return [cv2.resize(im, (nw, nh), interpolation=cv2.INTER_CUBIC) for im in imgs]

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        hz_path, gt_path = self.pairs[idx]
        hazy, clear = read_rgb(hz_path), read_rgb(gt_path)

        # some datasets (e.g. RESIDE ITS) store GT with a few extra border pixels
        if hazy.shape != clear.shape:
            clear = cv2.resize(clear, (hazy.shape[1], hazy.shape[0]), interpolation=cv2.INTER_AREA)

        if self.mode != "test":
            hazy, clear = self._ensure_min_size([hazy, clear])
            h, w = hazy.shape[:2]
            if self.mode == "train":
                y, x = random.randint(0, h - self.size), random.randint(0, w - self.size)
            else:
                y, x = (h - self.size) // 2, (w - self.size) // 2
            hazy = hazy[y:y + self.size, x:x + self.size]
            clear = clear[y:y + self.size, x:x + self.size]
            if self.mode == "train" and self.hflip and random.random() < 0.5:
                hazy, clear = np.flip(hazy, 1), np.flip(clear, 1)

        return {"source": to_tensor(hazy), "target": to_tensor(clear), "filename": hz_path.name}
