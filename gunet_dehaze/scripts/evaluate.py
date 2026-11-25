"""Evaluate a dehazing checkpoint on a paired test set (full-resolution PSNR / SSIM).

python scripts/evaluate.py --model gunet_t --ckpt runs/gunet_t_ft/best.pth ^
    --data D:/fogdata/SOTS-OUT --save-dir results/sots_out
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fogdehaze.data import PairDataset  # noqa: E402
from fogdehaze.models import build_gunet, load_gunet_checkpoint  # noqa: E402
from fogdehaze.utils import AverageMeter, pad_to_multiple, psnr, to_image, write_rgb  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gunet_t")
    p.add_argument("--pad-mode", default="reflect", choices=["reflect", "zeros"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="folder with IN/ and GT/")
    p.add_argument("--save-dir", help="write dehazed images + per-image CSV here")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N pairs")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_gunet(a.model, a.pad_mode)
    load_gunet_checkpoint(model, a.ckpt)
    model.to(device).eval()

    ds = PairDataset(a.data, "test")
    if a.limit:
        ds.pairs = ds.pairs[: a.limit]
    save_dir = Path(a.save_dir) if a.save_dir else None
    if save_dir:
        (save_dir / "imgs").mkdir(parents=True, exist_ok=True)

    P, S, rows = AverageMeter(), AverageMeter(), []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="eval"):
            item = ds[i]
            src = item["source"].unsqueeze(0).to(device)
            tgt = item["target"].unsqueeze(0).to(device)
            h, w = src.shape[2:]
            out = model(pad_to_multiple(src, model.patch_size))[:, :, :h, :w].clamp(-1, 1)

            p_val = psnr(out, tgt).item()
            # SSIM protocol from the gUNet / DehazeFormer papers (downsample to ~256)
            r = max(1, round(min(h, w) / 256))
            o01, t01 = out * 0.5 + 0.5, tgt * 0.5 + 0.5
            s_val = ssim(F.adaptive_avg_pool2d(o01, (h // r, w // r)),
                         F.adaptive_avg_pool2d(t01, (h // r, w // r)), data_range=1).item()
            P.update(p_val)
            S.update(s_val)
            rows.append((item["filename"], f"{p_val:.2f}", f"{s_val:.4f}"))
            if save_dir:
                write_rgb(save_dir / "imgs" / (Path(item["filename"]).stem + ".png"), to_image(out))

    print(f"\n{len(ds)} images   PSNR {P.avg:.2f} dB   SSIM {S.avg:.4f}")
    if save_dir:
        with open(save_dir / "results.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["file", "psnr", "ssim"])
            wr.writerows(rows)
            wr.writerow(["MEAN", f"{P.avg:.2f}", f"{S.avg:.4f}"])
        print(f"results written to {save_dir}")


if __name__ == "__main__":
    main()
