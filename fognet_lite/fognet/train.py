r"""Train FogNet-Lite.

Examples
  # Setup used for the shipped model (data from download_data.py, D = %USERPROFILE%/fogdata):
  python -m fognet.train --paired D/RESIDE-6K/train/hazy D/RESIDE-6K/train/GT \
      --real D/NH-HAZE2/data/train_notest/hazy D/NH-HAZE2/data/train_notest/clean \
      --synthetic D/RESIDE-6K/train/GT --synthetic-repeats 1 \
      --val D/SOTS/SOTS/outdoor/hazy D/SOTS/SOTS/outdoor/gt --out runs/fognet

  # RESIDE OTS (outdoor, best match for real fog) + extra on-the-fly fog on your own clear photos
  python -m fognet.train --paired data/OTS/hazy data/OTS/clear --synthetic data/my_clear \
                         --val data/SOTS/outdoor/hazy data/SOTS/outdoor/gt --epochs 30

  # Only your own clear photos (fog generated automatically)
  python -m fognet.train --synthetic data/my_clear --epochs 50
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .dataset import PairedFogDataset, SyntheticFogDataset
from .metrics import psnr, ssim
from .model import FogNetLite, count_params


def loss_fn(pred, target):
    return F.l1_loss(pred, target) + 0.2 * (1 - ssim(pred, target))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    p = s = n = 0
    for hazy, clear in loader:
        hazy, clear = hazy.to(device), clear.to(device)
        out = model(hazy)
        p += psnr(out, clear).item()
        s += ssim(out, clear).item()
        n += 1
    model.train()
    return p / max(n, 1), s / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", nargs=2, action="append", metavar=("HAZY", "CLEAR"), default=[],
                    help="foggy/clear folder pair; can be given more than once")
    ap.add_argument("--real", nargs=2, action="append", metavar=("HAZY", "CLEAR"), default=[],
                    help="real-haze pairs (e.g. NH-HAZE2, O-HAZE); few images, so repeated each epoch")
    ap.add_argument("--real-repeats", type=int, default=10)
    ap.add_argument("--synthetic", action="append", default=[], help="folder of clear images")
    ap.add_argument("--synthetic-repeats", type=int, default=2)
    ap.add_argument("--val", nargs=2, metavar=("HAZY", "CLEAR"))
    ap.add_argument("--val-max", type=int, default=300, help="cap validation images to keep epochs fast")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--scale", type=int, default=4, help="K is estimated at 1/scale resolution")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume")
    args = ap.parse_args()

    sets = [PairedFogDataset(h, c, args.crop) for h, c in args.paired]
    for h, c in args.real:
        sets += [PairedFogDataset(h, c, args.crop)] * args.real_repeats
    sets += [SyntheticFogDataset(d, args.crop, args.synthetic_repeats) for d in args.synthetic]
    if not sets:
        ap.error("give at least one --paired, --real or --synthetic dataset")
    print(f"paired={sum(len(PairedFogDataset(h, c)) for h, c in args.paired)}  "
          f"real={sum(len(PairedFogDataset(h, c)) for h, c in args.real)} x{args.real_repeats}  "
          f"synthetic={sum(len(s) for s in sets if isinstance(s, SyntheticFogDataset))}")
    train_set = ConcatDataset(sets)

    val_loader = None
    if args.val:
        v = PairedFogDataset(*args.val, train=False)
        if len(v) > args.val_max:
            v = Subset(v, torch.linspace(0, len(v) - 1, args.val_max).long().tolist())
        val_loader = DataLoader(v, batch_size=1, num_workers=args.workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FogNetLite(width=args.width, scale=args.scale).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])
    print(f"device={device}  params={count_params(model):,}  train images={len(train_set)}")

    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader))
    os.makedirs(args.out, exist_ok=True)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        t0, total = time.time(), 0.0
        for hazy, clear in loader:
            hazy, clear = hazy.to(device, non_blocking=True), clear.to(device, non_blocking=True)
            loss = loss_fn(model(hazy), clear)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            total += loss.item()
        msg = f"epoch {epoch:3d}/{args.epochs}  loss {total / len(loader):.4f}  {time.time() - t0:.0f}s"
        ckpt = {"model": model.state_dict(), "width": args.width, "scale": args.scale, "epoch": epoch}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if val_loader is not None:
            vp, vs = validate(model, val_loader, device)
            msg += f"  val PSNR {vp:.2f}  SSIM {vs:.4f}"
            if vp > best:
                best = vp
                torch.save(ckpt, os.path.join(args.out, "best.pt"))
                msg += "  *best*"
        print(msg, flush=True)

    if val_loader is None:
        torch.save(ckpt, os.path.join(args.out, "best.pt"))
    print(f"done. checkpoints in {args.out}/")


if __name__ == "__main__":
    main()
