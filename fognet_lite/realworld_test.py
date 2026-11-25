"""Test on real foggy photos that have no clear version (e.g. RTTS).

    python realworld_test.py --images %USERPROFILE%/fogdata/RTTS/images/test --model fognet.onnx

PSNR/SSIM need a clear image, so this reports no-reference measures instead:
  fog level  mean dark channel (fognet.dcp.fog_score). Lower = less haze left.
  contrast   std-dev of luminance (RMS contrast, 0-255). Higher = more visible detail.
  colour     Hasler-Suesstrunk colourfulness. Fog washes colour out; higher = more colour.
These can be fooled (e.g. over-darkening lowers "fog level" too), so always look at the grids
written to --out-dir as well.
"""
import argparse
import os

import cv2
import numpy as np

from fognet.dcp import DCPDehazer, fog_score
from pi_infer import OnnxDehazer

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def contrast(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())


def colourfulness(bgr):
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default="fognet.onnx")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--width", type=int, default=640, help="resize to this width (camera-like)")
    ap.add_argument("--grids", type=int, default=12, help="how many side-by-side images to save")
    ap.add_argument("--out-dir", default="realworld_samples")
    args = ap.parse_args()

    methods = {"input": lambda x: x, "DCP": DCPDehazer(smooth_A=0.0), "FogNet": OnnxDehazer(args.model)}
    files = [f for f in sorted(os.listdir(args.images)) if f.lower().endswith(EXTS)]
    if args.max:
        files = files[:args.max]
    os.makedirs(args.out_dir, exist_ok=True)
    every = max(1, len(files) // max(args.grids, 1))
    stats = {k: [] for k in methods}

    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(args.images, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
        outs = {k: fn(img) for k, fn in methods.items()}
        for k, o in outs.items():
            stats[k].append((fog_score(o), contrast(o), colourfulness(o)))
        if i % every == 0 and i // every < args.grids:
            grid = np.hstack([label(o, k) for k, o in outs.items()])
            cv2.imwrite(os.path.join(args.out_dir, os.path.splitext(f)[0] + ".jpg"), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"\n{len(files)} real foggy images from {args.images}")
    print(f"{'method':<10}{'fog level':>11}{'contrast':>10}{'colour':>9}")
    for k, s in stats.items():
        a = np.mean(s, axis=0)
        print(f"{k:<10}{a[0]:>11.3f}{a[1]:>10.1f}{a[2]:>9.1f}")
    print(f"\nside-by-side grids in {args.out_dir}/")


if __name__ == "__main__":
    main()
