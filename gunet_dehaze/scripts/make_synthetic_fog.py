"""Create hazy/clear training pairs from CLEAR images (e.g. your own camera footage).

Uses the atmospheric scattering model     I = J * t + A * (1 - t),   t = exp(-beta * d)
  J    clear image            A     airlight (fog brightness/colour)
  d    relative depth [0,1]   beta  fog density

Depth comes from MiDaS-small (--depth midas, downloads ~80 MB via torch.hub the
first time) or a cheap heuristic (--depth heuristic: far at the top of the frame,
near at the bottom - fine for road / street / landscape cameras).

Output layout matches the training loader:
    <out>/GT/<name>.png         clear
    <out>/IN/<name>_<k>.png     k-th hazy version

python scripts/make_synthetic_fog.py --src my_clear_frames/ --out D:/fogdata/mycam --per-image 4
# extract frames from a clear video first:
python scripts/make_synthetic_fog.py --src clear_drive.mp4 --every 15 --out D:/fogdata/mycam
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fogdehaze.utils import IMG_EXTS, read_rgb, write_rgb  # noqa: E402


def heuristic_depth(rgb):
    h, w = rgb.shape[:2]
    rows = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]   # top far, bottom near
    d = np.repeat(rows, w, axis=1)
    # bright, low-texture regions (sky) are treated as far away
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 5)
    sky = (gray > 0.6) & (tex < 0.02)
    d = np.where(sky, 1.0, d)
    return cv2.GaussianBlur(d, (0, 0), max(h, w) / 60)


class MidasDepth:
    def __init__(self):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.dev).eval()
        self.tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

    def __call__(self, rgb):
        torch = self.torch
        with torch.no_grad():
            inv = self.net(self.tf(rgb).to(self.dev))
            inv = torch.nn.functional.interpolate(inv[:, None], size=rgb.shape[:2], mode="bicubic",
                                                  align_corners=False)[0, 0].cpu().numpy()
        inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-6)   # MiDaS gives inverse depth
        return 1.0 - inv


def add_fog(rgb, depth, beta, airlight, tint):
    J = rgb.astype(np.float32) / 255.0
    d = 0.1 + 0.9 * depth                                  # keep some fog on near objects
    t = np.exp(-beta * d)[..., None]
    A = np.clip(airlight * np.array(tint, np.float32), 0, 1)
    I = J * t + A * (1 - t)
    I += np.random.normal(0, 0.008, I.shape).astype(np.float32)   # mild sensor noise
    return (np.clip(I, 0, 1) * 255).round().astype(np.uint8)


def iter_sources(src, every):
    src = Path(src)
    if src.is_file() and src.suffix.lower() not in IMG_EXTS:
        cap, i = cv2.VideoCapture(str(src)), 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f"{src.stem}_f{i:06d}", cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            i += 1
        return
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    for f in files:
        yield f.stem, read_rgb(f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="folder of clear images, one image, or a clear video")
    p.add_argument("--out", required=True)
    p.add_argument("--per-image", type=int, default=3, help="hazy versions per clear image")
    p.add_argument("--beta", type=float, nargs=2, default=[0.6, 2.8], help="fog density range")
    p.add_argument("--airlight", type=float, nargs=2, default=[0.7, 1.0])
    p.add_argument("--depth", choices=["heuristic", "midas"], default="heuristic")
    p.add_argument("--every", type=int, default=15, help="video: keep every N-th frame")
    p.add_argument("--max-side", type=int, default=1024, help="downscale larger images")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    (out / "GT").mkdir(parents=True, exist_ok=True)
    (out / "IN").mkdir(parents=True, exist_ok=True)
    depth_fn = MidasDepth() if a.depth == "midas" else heuristic_depth

    n = 0
    for name, rgb in iter_sources(a.src, a.every):
        s = a.max_side / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        depth = depth_fn(rgb)
        write_rgb(out / "GT" / f"{name}.png", rgb)
        for k in range(a.per_image):
            beta = random.uniform(*a.beta)
            air = random.uniform(*a.airlight)
            tint = [1.0 + random.uniform(-0.04, 0.04) for _ in range(3)]  # slightly blue/yellow fog
            write_rgb(out / "IN" / f"{name}_{k}.png", add_fog(rgb, depth, beta, air, tint))
        n += 1
        if n % 50 == 0:
            print(f"{n} images done")
    print(f"done: {n} clear images -> {n * a.per_image} hazy pairs in {out}")


if __name__ == "__main__":
    main()
