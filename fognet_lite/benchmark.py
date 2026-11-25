"""Speed test: FogNet-Lite (ONNX Runtime, CPU) and DCP at several resolutions and thread counts.

Run it on the Raspberry Pi to get the real numbers:
    python benchmark.py --model fognet.onnx
"""
import argparse
import platform
import time

import numpy as np

from fognet.dcp import DCPDehazer
from pi_infer import OnnxDehazer


def bench(fn, frame, n):
    for _ in range(3):
        fn(frame)
    t = time.perf_counter()
    for _ in range(n):
        fn(frame)
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fognet.onnx")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--sizes", nargs="+", default=["320x240", "640x480", "1280x720"])
    ap.add_argument("-n", type=int, default=30)
    args = ap.parse_args()

    print(f"{platform.machine()}  {platform.processor() or platform.platform()}")
    print(f"{'method':<18}{'size':>10}{'ms/frame':>10}{'FPS':>8}")
    rng = np.random.default_rng(0)
    for size in args.sizes:
        w, h = map(int, size.split("x"))
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        for t in args.threads:
            ms = bench(OnnxDehazer(args.model, threads=t), frame, args.n)
            print(f"{f'FogNet {t} thread':<18}{size:>10}{ms:>10.1f}{1000 / ms:>8.1f}")
        ms = bench(DCPDehazer(), frame, args.n)
        print(f"{'DCP':<18}{size:>10}{ms:>10.1f}{1000 / ms:>8.1f}")


if __name__ == "__main__":
    main()
