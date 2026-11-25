"""Make realistic synthetic fog from clear images.

Models trained only on lab-made fog (uniform, pure white, noise-free) often do
badly on real fog. This generator adds the things real foggy camera footage has:
  * depth that increases towards the horizon (top of the frame) and is not uniform
  * patchy, non-uniform fog density (low-frequency noise)
  * coloured airlight (grey, slightly yellow from street lights, slightly blue)
  * camera effects: sensor noise, slight blur, JPEG compression

Physics:  I = J * t + A * (1 - t),   t = exp(-beta * depth)

Run it on its own to save foggy/clear pairs to disk:
    python -m fognet.synth_fog --clear data/clear --out data/synth --per-image 3
"""
import argparse
import os
import random

import cv2
import numpy as np


def _low_freq_noise(h, w, cells=4):
    g = np.random.rand(cells + 1, cells + 1).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def pseudo_depth(h, w):
    """Rough depth map in [0,1]: far at the top, near at the bottom, with random variation."""
    y = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None].repeat(w, 1)
    horizon = random.uniform(0.3, 0.7)              # where the horizon sits
    y = np.clip((y - (1 - horizon) * 0.5) / (0.5 + horizon * 0.5), 0, 1)
    mix = random.uniform(0.3, 0.8)
    d = mix * y + (1 - mix) * _low_freq_noise(h, w, random.choice([2, 3, 4, 6]))
    return d.clip(0, 1)


def add_fog(clear, depth=None):
    """clear: float32 HxWx3 RGB in [0,1]. Returns a foggy image (same format)."""
    h, w = clear.shape[:2]
    if depth is None:
        depth = pseudo_depth(h, w)
    beta = random.uniform(0.6, 2.8)                  # fog thickness
    density = 1.0 + random.uniform(-0.35, 0.35) * (_low_freq_noise(h, w, 3) * 2 - 1)  # patchiness
    t = np.exp(-beta * depth * density).clip(0.05, 1.0)[..., None]

    a = random.uniform(0.65, 1.0)
    tint = np.array([1.0, 1.0, 1.0], np.float32)
    r = random.random()
    if r < 0.2:      # sodium street lights: yellow/orange
        tint = np.array([1.0, random.uniform(0.88, 0.97), random.uniform(0.7, 0.88)], np.float32)
    elif r < 0.35:   # cold / bluish
        tint = np.array([random.uniform(0.88, 0.97), random.uniform(0.94, 1.0), 1.0], np.float32)
    A = (a * tint).reshape(1, 1, 3)

    foggy = clear * t + A * (1 - t)
    return camera_effects(foggy)


def camera_effects(img):
    """Sensor noise, slight blur, brightness change and JPEG compression."""
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.0))
    if random.random() < 0.5:
        img = img * random.uniform(0.75, 1.1)
    if random.random() < 0.5:
        img = img + np.random.normal(0, random.uniform(0.003, 0.02), img.shape).astype(np.float32)
    img = img.clip(0, 1)
    if random.random() < 0.5:
        q = random.randint(40, 90)
        u8 = (img * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    return img.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", required=True, help="folder of clear images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-image", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "hazy"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "clear"), exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in sorted(os.listdir(args.clear)) if f.lower().endswith(exts)]
    for f in files:
        stem = os.path.splitext(f)[0]
        bgr = cv2.imread(os.path.join(args.clear, f))
        if bgr is None:
            continue
        cv2.imwrite(os.path.join(args.out, "clear", stem + ".png"), bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for i in range(args.per_image):
            fog = (add_fog(rgb) * 255).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.out, "hazy", f"{stem}_{i}.png"),
                        cv2.cvtColor(fog, cv2.COLOR_RGB2BGR))
    print(f"wrote {len(files) * args.per_image} foggy images to {args.out}")


if __name__ == "__main__":
    main()
