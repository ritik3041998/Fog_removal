"""Datasets for training.

PairedFogDataset  - foggy/clear pairs on disk (RESIDE OTS/ITS/SOTS, or output of synth_fog.py).
                    The clear image is matched by the part of the foggy filename before the
                    first "_", e.g. hazy/0001_0.8_0.2.jpg  ->  clear/0001.png  (RESIDE naming).
SyntheticFogDataset - folder of clear images only; fog is added on the fly, so every epoch
                    sees new fog.
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .synth_fog import add_fog, camera_effects

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _read_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def _random_crop_pair(a, b, size):
    h, w = a.shape[:2]
    if h < size or w < size:
        s = size / min(h, w)
        a = cv2.resize(a, (int(np.ceil(w * s)), int(np.ceil(h * s))))
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
        h, w = a.shape[:2]
    y, x = random.randint(0, h - size), random.randint(0, w - size)
    a, b = a[y:y + size, x:x + size], b[y:y + size, x:x + size]
    if random.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    return a, b


class PairedFogDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, crop=256, train=True, augment_camera=True):
        self.hazy_dir, self.clear_dir = hazy_dir, clear_dir
        self.crop, self.train, self.augment_camera = crop, train, augment_camera
        clear = {os.path.splitext(f)[0]: f for f in os.listdir(clear_dir) if f.lower().endswith(EXTS)}
        self.pairs = []
        for f in sorted(os.listdir(hazy_dir)):
            if not f.lower().endswith(EXTS):
                continue
            stem = os.path.splitext(f)[0]
            key = stem if stem in clear else stem.split("_")[0]   # same name, or RESIDE "0001_0.8_0.2"
            if key in clear:
                self.pairs.append((f, clear[key]))
        if not self.pairs:
            raise RuntimeError(f"no matching foggy/clear pairs in {hazy_dir} and {clear_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        hf, cf = self.pairs[i]
        hazy = _read_rgb(os.path.join(self.hazy_dir, hf))
        clear = _read_rgb(os.path.join(self.clear_dir, cf))
        if clear.shape[:2] != hazy.shape[:2]:     # RESIDE ITS/SOTS GT can be slightly larger
            clear = cv2.resize(clear, (hazy.shape[1], hazy.shape[0]))
        if self.train:
            hazy, clear = _random_crop_pair(hazy, clear, self.crop)
            if self.augment_camera and random.random() < 0.5:
                hazy = camera_effects(np.ascontiguousarray(hazy))
        return _to_tensor(hazy), _to_tensor(clear)


class SyntheticFogDataset(Dataset):
    def __init__(self, clear_dir, crop=256, repeats=1):
        self.clear_dir, self.crop = clear_dir, crop
        self.files = [f for f in sorted(os.listdir(clear_dir)) if f.lower().endswith(EXTS)] * repeats
        if not self.files:
            raise RuntimeError(f"no images in {clear_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        clear = _read_rgb(os.path.join(self.clear_dir, self.files[i]))
        clear, _ = _random_crop_pair(clear, clear, self.crop)
        clear = np.ascontiguousarray(clear)
        return _to_tensor(add_fog(clear)), _to_tensor(clear)
