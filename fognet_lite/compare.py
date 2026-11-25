"""Compare methods on a test set with ground truth (e.g. RESIDE SOTS-outdoor).

    python compare.py --hazy data/SOTS/outdoor/hazy --clear data/SOTS/outdoor/gt \
                       --model fognet.onnx --external results_online_repo/ --samples 6

Scores: PSNR and SSIM (higher is better) and time per image on this machine.
  input     - the foggy image itself (lower bound)
  DCP       - Dark Channel Prior
  FogNet    - our model (ONNX)
  external  - outputs already produced by the online repo; files must have the
              same name stem as the foggy images (any image extension)
Side-by-side samples are written to --out-dir (default eval_samples/).
"""
import argparse
import os
import time

import cv2
import numpy as np

from fognet.dcp import DCPDehazer

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = g(a), g(b)
    saa, sbb, sab = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(m.mean())


def find(folder, stem):
    for e in EXTS:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazy", required=True)
    ap.add_argument("--clear", required=True)
    ap.add_argument("--model", help="FogNet ONNX file")
    ap.add_argument("--external", help="folder of outputs from the online repo")
    ap.add_argument("--external-name", default="online repo")
    ap.add_argument("--max", type=int, default=0, help="limit number of images (0 = all)")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out-dir", default="eval_samples", help="where side-by-side samples go")
    args = ap.parse_args()

    methods = {"input": lambda x: x, "DCP": DCPDehazer(smooth_A=0.0)}
    if args.model:
        from pi_infer import OnnxDehazer
        methods["FogNet"] = OnnxDehazer(args.model)
    scores = {k: [] for k in list(methods) + ([args.external_name] if args.external else [])}
    times = {k: [] for k in methods}

    files = [f for f in sorted(os.listdir(args.hazy)) if f.lower().endswith(EXTS)]
    if args.max:
        files = files[:args.max]
    os.makedirs(args.out_dir, exist_ok=True)
    every = max(1, len(files) // max(args.samples, 1))

    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        gt_path = find(args.clear, stem) or find(args.clear, stem.split("_")[0])
        if gt_path is None:
            continue
        hazy, gt = cv2.imread(os.path.join(args.hazy, f)), cv2.imread(gt_path)
        if gt.shape != hazy.shape:
            gt = cv2.resize(gt, (hazy.shape[1], hazy.shape[0]))
        outs = {}
        for name, fn in methods.items():
            t0 = time.perf_counter()
            outs[name] = fn(hazy)
            times[name].append(time.perf_counter() - t0)
        if args.external:
            p = find(args.external, stem)
            if p is not None:
                ext = cv2.imread(p)
                outs[args.external_name] = cv2.resize(ext, (hazy.shape[1], hazy.shape[0]))
        for name, out in outs.items():
            scores[name].append((psnr(out, gt), ssim(out, gt)))
        if i % every == 0 and i // every < args.samples:
            tiles = list(outs.items()) + [("ground truth", gt)]
            row = []
            for name, img in tiles:
                img = img.copy()
                cv2.putText(img, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                row.append(img)
            cv2.imwrite(os.path.join(args.out_dir, stem + ".jpg"), np.hstack(row))

    print(f"\n{'method':<14}{'images':>7}{'PSNR':>9}{'SSIM':>9}{'ms/img':>9}")
    for name, s in scores.items():
        if not s:
            continue
        p, q = np.mean(s, axis=0)
        ms = f"{1000 * np.mean(times[name]):.1f}" if name in times else "-"
        print(f"{name:<14}{len(s):>7}{p:>9.2f}{q:>9.4f}{ms:>9}")
    print(f"\nsamples saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
