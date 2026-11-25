"""Sanity checks + size/speed table for the gUNet variants.

  1. Our export-friendly gUNet gives the same output as the official code
     (same weights), so official pretrained checkpoints are valid.
  2. Parameter count and latency for each variant at a given resolution.

python scripts/check_model.py --height 480 --width 640
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fogdehaze.models import VARIANTS, build_gunet  # noqa: E402


def check_equivalence():
    official_dir = ROOT / "third_party" / "gUNet"
    if not official_dir.exists():
        print("[check] third_party/gUNet not found, skipping equivalence test")
        return
    try:
        import timm  # noqa: F401  (official code imports timm)
    except ImportError:
        print("[check] `timm` not installed, skipping equivalence test (pip install timm)")
        return
    import importlib
    import types
    import warnings

    # the official models/__init__.py has a broken import, so register an empty
    # package and import only models/gunet.py (+ its norm_layer dependency)
    pkg = types.ModuleType("models")
    pkg.__path__ = [str(official_dir / "models")]
    sys.modules["models"] = pkg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        official_gunet_t = importlib.import_module("models.gunet").gunet_t

    torch.manual_seed(0)
    ref = official_gunet_t().eval()
    ours = build_gunet("gunet_t").eval()
    # randomise BN statistics so the test is not trivially passing
    for m in ref.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 1.5)
    ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.rand(2, 3, 128, 160) * 2 - 1
    with torch.no_grad():
        diff = (ref(x) - ours(x)).abs().max().item()
    status = "OK" if diff < 1e-4 else "MISMATCH"
    print(f"[check] official vs export-friendly gUNet-T: max diff {diff:.2e}  -> {status}")
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        del sys.modules[k]


@torch.no_grad()
def bench(model, x, n):
    for _ in range(2):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--variants", nargs="*", default=["gunet_p", "gunet_n", "gunet_t", "gunet_s"])
    a = p.parse_args()

    check_equivalence()

    cuda = torch.cuda.is_available()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"\n{'variant':10s} {'params':>9s} {'GMACs':>8s} {'CPU ms':>8s} {'GPU fp16 ms':>12s}   @ {a.width}x{a.height}")
    for name in a.variants:
        m = build_gunet(name).eval()
        params = sum(p.numel() for p in m.parameters())
        macs = estimate_macs(m, a.height, a.width)
        x = torch.rand(1, 3, a.height, a.width) * 2 - 1
        cpu_ms = bench(m, x, 3)
        gpu_ms = float("nan")
        if cuda:
            mg = build_gunet(name).eval().cuda().half()
            gpu_ms = bench(mg, x.cuda().half(), 20)
        print(f"{name:10s} {params / 1e6:8.3f}M {macs / 1e9:8.1f} {cpu_ms:8.0f} {gpu_ms:12.1f}")
    print("\nVariants in VARIANTS:", ", ".join(VARIANTS))


def estimate_macs(model, h, w):
    """Count conv multiply-accumulates with forward hooks (convs dominate this network)."""
    total = [0]

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        total[0] += out.numel() * (mod.in_channels // mod.groups) * k

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(1, 3, h, w))
    for hd in hs:
        hd.remove()
    return total[0]


if __name__ == "__main__":
    main()
