"""Train / fine-tune a gUNet dehazing model on a single GPU.

Examples
--------
# fine-tune the official pretrained gUNet-T on your data (recommended start)
python scripts/train.py --config configs/gunet_t_laptop.json ^
    --train-dir D:/fogdata/OTS --val-dir D:/fogdata/SOTS-OUT ^
    --init pretrained/reside-out/gunet_t.pth --out runs/gunet_t_ft

# resume an interrupted run
python scripts/train.py --config configs/gunet_t_laptop.json --out runs/gunet_t_ft --resume

# override any config key
python scripts/train.py --config configs/gunet_t_laptop.json --set batch_size=8 epochs=50
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fogdehaze.data import PairDataset  # noqa: E402
from fogdehaze.models import build_gunet, load_gunet_checkpoint  # noqa: E402
from fogdehaze.utils import AverageMeter, psnr  # noqa: E402

DEFAULTS = {
    "model": "gunet_t",
    "pad_mode": "reflect",
    "train_dir": None,
    "val_dir": None,
    "patch_size": 256,
    "batch_size": 16,
    "lr": 2e-4,
    "min_lr_ratio": 0.01,
    "weight_decay": 0.01,
    "epochs": 100,
    "samples_per_epoch": 4096,
    "warmup_epochs": 5,
    "frozen_bn_epochs": 10,
    "eval_every": 2,
    "ssim_weight": 0.0,
    "num_workers": 4,
    "amp": True,
    "max_val_items": 500,
    "hflip": True,
    "seed": 42,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, help="JSON config file")
    p.add_argument("--train-dir", type=str, nargs="+",
                   help="one or more folders with hazy + GT images; repeat a small folder "
                        "(e.g. real-haze NH-HAZE2) several times to sample it more often")
    p.add_argument("--val-dir", type=str)
    p.add_argument("--init", type=str, help="initialise weights from a checkpoint (e.g. official gUNet)")
    p.add_argument("--out", type=str, default="runs/exp", help="output directory")
    p.add_argument("--resume", action="store_true", help="resume from <out>/last.pth")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")
    a = p.parse_args()

    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown config key '{k}'")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v  # plain string, e.g. model=gunet_n
    if a.train_dir:
        cfg["train_dir"] = a.train_dir
    if a.val_dir:
        cfg["val_dir"] = a.val_dir
    if not cfg["train_dir"]:
        raise SystemExit("--train-dir is required (folder with IN/ and GT/)")
    return a, cfg


def make_lr_lambda(cfg, iters_per_epoch):
    warm = cfg["warmup_epochs"] * iters_per_epoch
    total = cfg["epochs"] * iters_per_epoch
    lo = cfg["min_lr_ratio"]

    def f(it):
        if it < warm:
            return lo + (1 - lo) * it / max(1, warm)
        t = (it - warm) / max(1, total - warm)
        return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * min(t, 1.0)))

    return f


def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        src = batch["source"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(src)
        meter.update(psnr(out.float().clamp(-1, 1), tgt).mean().item(), src.size(0))
    return meter.avg


def main():
    args, cfg = parse_args()
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[train] WARNING: CUDA not available, training on CPU will be very slow")
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---------------- data ----------------
    train_dirs = cfg["train_dir"] if isinstance(cfg["train_dir"], list) else [cfg["train_dir"]]
    parts = [PairDataset(d, "train", cfg["patch_size"], hflip=cfg["hflip"]) for d in train_dirs]
    for d, ds in zip(train_dirs, parts):
        print(f"[train]   {len(ds):6d} pairs from {d}")
    train_set = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    sampler = RandomSampler(train_set, num_samples=cfg["samples_per_epoch"])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], sampler=sampler,
                              num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                              persistent_workers=cfg["num_workers"] > 0)
    val_loader = None
    if cfg["val_dir"]:
        val_set = PairDataset(cfg["val_dir"], "valid", cfg["patch_size"], max_items=cfg["max_val_items"])
        val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                                num_workers=cfg["num_workers"], pin_memory=True)
    iters_per_epoch = len(train_loader)
    print(f"[train] {len(train_set)} training pairs, {iters_per_epoch} iters/epoch"
          + (f", {len(val_loader.dataset)} validation pairs" if val_loader else ", no validation set"))

    # ---------------- model / optim ----------------
    model = build_gunet(cfg["model"], cfg["pad_mode"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model {cfg['model']} ({n_params / 1e6:.3f} M params, pad_mode={cfg['pad_mode']})")

    if args.init and not args.resume:
        _, missing, unexpected = load_gunet_checkpoint(model, args.init, strict=False)
        print(f"[train] initialised from {args.init} (missing={len(missing)}, unexpected={len(unexpected)})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(cfg, iters_per_epoch))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ssim_fn = None
    if cfg["ssim_weight"] > 0:
        from pytorch_msssim import ssim as ssim_fn

    start_epoch, best_psnr = 0, -1.0
    last_path = out_dir / "last.pth"
    if args.resume:
        if not last_path.exists():
            raise SystemExit(f"nothing to resume: {last_path} not found")
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_psnr = ckpt["epoch"] + 1, ckpt["best_psnr"]
        print(f"[train] resumed from epoch {start_epoch}, best PSNR {best_psnr:.2f}")

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_psnr": best_psnr, "config": cfg}, path)

    # ---------------- loop ----------------
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        frozen_bn = epoch >= cfg["epochs"] - cfg["frozen_bn_epochs"]
        if frozen_bn:
            set_bn_eval(model)

        loss_meter, t0 = AverageMeter(), time.time()
        for batch in train_loader:
            src = batch["source"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(src)
            out = out.float()
            loss = F.l1_loss(out, tgt)
            if ssim_fn is not None:
                loss = loss + cfg["ssim_weight"] * (1 - ssim_fn(out.clamp(-1, 1) * 0.5 + 0.5,
                                                                tgt * 0.5 + 0.5, data_range=1.0))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_meter.update(loss.item(), src.size(0))

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", loss_meter.avg, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        msg = (f"epoch {epoch + 1:4d}/{cfg['epochs']}  loss {loss_meter.avg:.4f}  lr {lr:.2e}  "
               f"{dt:.0f}s ({iters_per_epoch / dt:.1f} it/s){'  [frozen BN]' if frozen_bn else ''}")

        is_eval = val_loader is not None and ((epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"])
        if is_eval:
            v = validate(model, val_loader, device, use_amp)
            writer.add_scalar("valid/psnr", v, epoch)
            msg += f"  val PSNR {v:.2f} dB"
            if v > best_psnr:
                best_psnr = v
                save(out_dir / "best.pth", epoch)
                msg += "  *best*"
        print(msg, flush=True)
        save(last_path, epoch)

    if val_loader is None:
        save(out_dir / "best.pth", cfg["epochs"] - 1)
    writer.close()
    print(f"[train] done. best val PSNR {best_psnr:.2f} dB -> {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
