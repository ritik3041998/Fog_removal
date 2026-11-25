"""Export a trained gUNet to ONNX with a fixed input size and verify it.

The ONNX file is what goes to the Raspberry Pi:
  * CPU-only Pi  -> run directly with onnxruntime (scripts/infer.py --weights x.onnx)
  * Pi + AI HAT+ -> compile with the Hailo Dataflow Compiler (x86 Linux) into a .hef

python scripts/export_onnx.py --ckpt runs/gunet_t_ft/best.pth --height 480 --width 640
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fogdehaze.models import build_gunet, load_gunet_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", help="default: read from checkpoint config")
    p.add_argument("--pad-mode", choices=["reflect", "zeros"], help="default: read from checkpoint config")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out", help="output .onnx path (default: exports/<model>_<h>x<w>.onnx)")
    a = p.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    name = a.model or cfg.get("model", "gunet_t")
    pad_mode = a.pad_mode or cfg.get("pad_mode", "reflect")

    model = build_gunet(name, pad_mode)
    load_gunet_checkpoint(model, a.ckpt)
    model.eval()
    m = model.patch_size
    if a.height % m or a.width % m:
        raise SystemExit(f"height and width must be multiples of {m}")

    out = Path(a.out or f"exports/{name}_{a.height}x{a.width}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, a.height, a.width) * 2 - 1

    torch.onnx.export(model, dummy, str(out), input_names=["hazy"], output_names=["clear"],
                      opset_version=a.opset, do_constant_folding=True, dynamo=False)
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(out)))
    ops = sorted({n.op_type for n in onnx.load(str(out)).graph.node})
    print(f"[export] ONNX ops used: {', '.join(ops)}")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"hazy": dummy.numpy()})[0]
    print(f"[export] max |torch - onnx| = {np.abs(ref - got).max():.2e}")

    x = dummy.numpy()
    sess.run(None, {"hazy": x})
    t0, n = time.perf_counter(), 5
    for _ in range(n):
        sess.run(None, {"hazy": x})
    print(f"[export] onnxruntime CPU latency on this PC: {(time.perf_counter() - t0) / n * 1000:.0f} ms "
          f"(a Raspberry Pi 5 CPU will be several times slower - measure it on the Pi)")


if __name__ == "__main__":
    main()
