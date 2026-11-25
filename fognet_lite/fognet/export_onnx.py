"""Export a trained checkpoint to ONNX for the Raspberry Pi.

    python -m fognet.export_onnx --ckpt checkpoints/best.pt --out fognet.onnx

The ONNX file accepts any image size: input "input" (1,3,H,W) RGB float32 in [0,1].
"""
import argparse

import numpy as np
import torch

from .model import FogNetLite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="fognet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    model = FogNetLite(width=ck.get("width", 16), scale=ck.get("scale", 4))
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.rand(1, 3, 480, 640)
    torch.onnx.export(model, dummy, args.out, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
                      opset_version=args.opset, dynamo=False)

    # check that ONNX Runtime gives the same result as PyTorch, at a different size too
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for shape in [(1, 3, 480, 640), (1, 3, 240, 320)]:
        x = torch.rand(*shape)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        print(f"{shape}: max diff vs PyTorch = {np.abs(ref - got).max():.2e}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
