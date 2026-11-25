"""Test a pretrained dehazing model on real foggy images and/or paired benchmark data.

Models
  dehazeformer_mct  official DehazeFormer-MCT weights from the authors' HF demo
                    (pretrained/dehazeformer_demo/, trained on a mixed real+synthetic set)
  gunet             one of our gUNet checkpoints (--ckpt runs/.../best.pth)

python scripts/test_pretrained.py --source samples/online --out results/pretrained_online
python scripts/test_pretrained.py --pairs %USERPROFILE%/fogdata/SOTS/outdoor --limit 200
python scripts/test_pretrained.py --model gunet --ckpt runs/gunet_t/best.pth --pairs ...
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fogdehaze.data import build_pairs  # noqa: E402
from fogdehaze.models import build_gunet, load_gunet_checkpoint  # noqa: E402
from fogdehaze.utils import (IMG_EXTS, AverageMeter, dark_channel_fog_score, pad_to_multiple,  # noqa: E402
                             psnr, read_rgb, to_image, to_tensor, write_rgb)

DF_DIR = ROOT / "pretrained" / "dehazeformer_demo"


def load_model(name, ckpt):
    if name == "dehazeformer_mct":
        from fogdehaze.models.dehazeformer_mct import MCT
        model = MCT()
        # weights_only=True: tensors only, no arbitrary pickle code from downloaded files
        sd = torch.load(ckpt or DF_DIR / "dehazeformer.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(sd["state_dict"])
        return model, 1
    cfg = torch.load(ckpt, map_location="cpu", weights_only=True).get("config", {})
    model = build_gunet(cfg.get("model", "gunet_t"), cfg.get("pad_mode", "reflect"))
    load_gunet_checkpoint(model, ckpt)
    return model, model.patch_size


@torch.no_grad()
def run(model, multiple, x):
    h, w = x.shape[2:]
    if multiple > 1:
        x = pad_to_multiple(x, multiple)
    return model(x)[:, :, :h, :w].clamp(-1, 1)


@torch.no_grad()
def bench(model, multiple, device, h=480, w=640, n=10):
    x = torch.rand(1, 3, h, w, device=device) * 2 - 1
    for _ in range(3):
        run(model, multiple, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        run(model, multiple, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dehazeformer_mct", choices=["dehazeformer_mct", "gunet"])
    p.add_argument("--ckpt", help="checkpoint (required for gunet)")
    p.add_argument("--source", help="folder or image of real foggy photos (no ground truth)")
    p.add_argument("--pairs", help="paired dataset folder (hazy + GT) for PSNR/SSIM")
    p.add_argument("--limit", type=int, default=200, help="max pairs to score")
    p.add_argument("--out", default="results/pretrained", help="where to save images")
    p.add_argument("--max-side", type=int, default=1600, help="downscale larger photos")
    a = p.parse_args()
    if a.model == "gunet" and not a.ckpt:
        raise SystemExit("--ckpt is required for --model gunet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, multiple = load_model(a.model, a.ckpt)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[test] {a.model}: {n_params / 1e6:.3f} M params on {device}")
    print(f"[test] latency @640x480: GPU {bench(model, multiple, device):.1f} ms", end="")
    model_cpu = model.to("cpu")
    print(f", CPU {bench(model_cpu, multiple, torch.device('cpu'), n=3):.0f} ms")
    model.to(device)
    out_dir = Path(a.out)

    if a.source:
        src = Path(a.source)
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
        (out_dir / "real").mkdir(parents=True, exist_ok=True)
        print(f"\nReal foggy photos ({len(files)}):  fog score before -> after")
        for f in files:
            rgb = read_rgb(f)
            s = a.max_side / max(rgb.shape[:2])
            if s < 1:
                import cv2
                rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            out = to_image(run(model, multiple, to_tensor(rgb)[None].to(device)))
            print(f"  {f.name:28s} {dark_channel_fog_score(rgb):.2f} -> {dark_channel_fog_score(out):.2f}"
                  f"   ({rgb.shape[1]}x{rgb.shape[0]})")
            write_rgb(out_dir / "real" / f"{f.stem}_compare.jpg", np.hstack([rgb, out]))
        print(f"  side-by-side images: {out_dir / 'real'}")

    if a.pairs:
        pairs = build_pairs(a.pairs)
        if a.limit and len(pairs) > a.limit:
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, a.limit).astype(int)]
        (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
        P, S, P_in = AverageMeter(), AverageMeter(), AverageMeter()
        for i, (hz, gt) in enumerate(tqdm(pairs, desc="pairs")):
            hazy, clear = read_rgb(hz), read_rgb(gt)
            if hazy.shape != clear.shape:
                import cv2
                clear = cv2.resize(clear, (hazy.shape[1], hazy.shape[0]), interpolation=cv2.INTER_AREA)
            x, t = to_tensor(hazy)[None].to(device), to_tensor(clear)[None].to(device)
            y = run(model, multiple, x)
            h, w = x.shape[2:]
            r = max(1, round(min(h, w) / 256))
            s_val = ssim(F.adaptive_avg_pool2d(y * 0.5 + 0.5, (h // r, w // r)),
                         F.adaptive_avg_pool2d(t * 0.5 + 0.5, (h // r, w // r)), data_range=1).item()
            P.update(psnr(y, t).item())
            P_in.update(psnr(x, t).item())
            S.update(s_val)
            if i < 8:
                write_rgb(out_dir / "pairs" / f"{hz.stem}_hazy_out_gt.jpg",
                          np.hstack([hazy, to_image(y), clear]))
        print(f"\nPaired data {a.pairs}  ({len(pairs)} pairs)")
        print(f"  hazy input  PSNR {P_in.avg:.2f} dB")
        print(f"  dehazed     PSNR {P.avg:.2f} dB   SSIM {S.avg:.4f}")
        print(f"  examples (hazy | output | ground truth): {out_dir / 'pairs'}")


if __name__ == "__main__":
    main()
