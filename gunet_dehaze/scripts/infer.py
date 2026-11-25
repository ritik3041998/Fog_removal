"""Dehaze an image, a folder, a video file or a live camera.

Works with a PyTorch checkpoint (.pth, for the laptop) or an exported ONNX model
(.onnx, for the Raspberry Pi / any CPU). Shows foggy | dehazed side by side with
a dark-channel fog score and FPS.

Examples
--------
python scripts/infer.py --weights runs/gunet_t_ft/best.pth --source 0              # webcam
python scripts/infer.py --weights runs/gunet_t_ft/best.pth --source foggy.mp4 --out clear.mp4
python scripts/infer.py --weights exports/gunet_t_480x640.onnx --source photos/ --out results/
python scripts/infer.py --weights model.onnx --source 0 --fog-threshold 0.25      # skip clear frames

Keys in the live window:  q / Esc = quit,  space = pause,  d = toggle dehazing.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fogdehaze.utils import IMG_EXTS, dark_channel_fog_score, read_rgb, write_rgb  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class TorchDehazer:
    def __init__(self, weights, model_name, pad_mode, size, half):
        import torch
        from fogdehaze.models import build_gunet, load_gunet_checkpoint

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if model_name is None and Path(weights).name == "dehazeformer.pth":
            model_name = "dehazeformer_mct"
        model_name = model_name or cfg.get("model", "gunet_t")
        if model_name == "dehazeformer_mct":
            # pretrained DehazeFormer-MCT: any input size, fp16 not supported by its grid_sample
            from fogdehaze.models.dehazeformer_mct import MCT
            self.model = MCT()
            self.model.load_state_dict(ckpt["state_dict"])
            self.multiple = 1
            half = False
        else:
            pad_mode = pad_mode or cfg.get("pad_mode", "reflect")
            self.model = build_gunet(model_name, pad_mode)
            load_gunet_checkpoint(self.model, weights)
            self.multiple = self.model.patch_size
        self.model.to(self.device).eval()
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.size = size  # (w, h) or None = native resolution
        print(f"[infer] PyTorch {model_name} on {self.device}{' fp16' if self.half else ''}")

    def __call__(self, rgb):
        torch = self.torch
        h0, w0 = rgb.shape[:2]
        if self.size:
            w, h = self.size
        else:
            w, h = (w0 // self.multiple) * self.multiple, (h0 // self.multiple) * self.multiple
        inp = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (w0, h0) else rgb
        x = torch.from_numpy(inp).to(self.device).permute(2, 0, 1)[None]
        x = x.half() if self.half else x.float()
        with torch.inference_mode():
            y = self.model(x / 127.5 - 1.0).clamp_(-1, 1)
        out = ((y[0].float() + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        return out if (w, h) == (w0, h0) else cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)


class OnnxDehazer:
    def __init__(self, weights, threads):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(weights), so, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        _, _, h, w = inp.shape
        if not isinstance(h, int) or not isinstance(w, int):
            raise SystemExit("ONNX model must have a fixed input size; re-export with --height/--width")
        self.size = (w, h)
        print(f"[infer] ONNX {Path(weights).name} {w}x{h} via {self.sess.get_providers()[0]}")

    def __call__(self, rgb):
        h0, w0 = rgb.shape[:2]
        inp = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        x = (inp.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.name: x})[0][0]
        out = ((np.clip(y, -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)


def label(img_bgr, text, y=28):
    cv2.putText(img_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def compose(src_rgb, out_rgb, score, fps, dehazed):
    left = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    label(left, f"input  fog={score:.2f}")
    label(right, f"{'dehazed' if dehazed else 'passthrough'}  {fps:.1f} FPS")
    return np.hstack([left, right])


def run_stream(dehaze, cap, args, writer_path):
    writer, enabled, paused = None, True, False
    fps_ema, frame = 0.0, None
    while True:
        if not paused:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_width and frame_bgr.shape[1] > args.max_width:
                s = args.max_width / frame_bgr.shape[1]
                frame_bgr = cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            score = dark_channel_fog_score(rgb)
            do_it = enabled and score >= args.fog_threshold
            t0 = time.perf_counter()
            out = dehaze(rgb) if do_it else rgb
            dt = time.perf_counter() - t0
            if do_it:
                fps_ema = 1 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
            frame = compose(rgb, out, score, fps_ema, do_it)

            if writer_path:
                if writer is None:
                    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_src, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

        if not args.no_show:
            cv2.imshow("fog removal  (q quit, space pause, d toggle)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                paused = not paused
            if k == ord("d"):
                enabled = not enabled
    cap.release()
    if writer:
        writer.release()
        print(f"[infer] saved {writer_path}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help=".pth checkpoint or .onnx model")
    p.add_argument("--source", default="0", help="camera index, image, folder or video file")
    p.add_argument("--out", help="output file (video) or folder (images)")
    p.add_argument("--model", help="gunet_t / gunet_n / ... or dehazeformer_mct "
                                   "(default: read from checkpoint)")
    p.add_argument("--pad-mode", choices=["reflect", "zeros"])
    p.add_argument("--size", help="inference size WxH for .pth, e.g. 640x480 (default: native)")
    p.add_argument("--max-width", type=int, default=1280, help="downscale wider video frames")
    p.add_argument("--fog-threshold", type=float, default=0.0,
                   help="only dehaze frames whose dark-channel fog score >= this")
    p.add_argument("--half", action="store_true", help="fp16 on GPU")
    p.add_argument("--threads", type=int, default=0, help="ONNX Runtime CPU threads")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    a = p.parse_args()

    if str(a.weights).lower().endswith(".onnx"):
        dehaze = OnnxDehazer(a.weights, a.threads)
    else:
        size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
        dehaze = TorchDehazer(a.weights, a.model, a.pad_mode, size, a.half)

    src = a.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not cap.isOpened():
            raise SystemExit(f"could not open camera {src}")
        run_stream(dehaze, cap, a, a.out)
        return

    path = Path(src)
    if path.suffix.lower() in VIDEO_EXTS:
        run_stream(dehaze, cv2.VideoCapture(str(path)), a, a.out)
        return

    files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMG_EXTS) if path.is_dir() else [path]
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        rgb = read_rgb(f)
        t0 = time.perf_counter()
        out = dehaze(rgb)
        dt = time.perf_counter() - t0
        print(f"{f.name}: fog score {dark_channel_fog_score(rgb):.2f} -> "
              f"{dark_channel_fog_score(out):.2f}  ({dt * 1000:.0f} ms)")
        if out_dir:
            write_rgb(out_dir / (f.stem + "_dehazed.png"), out)
            write_rgb(out_dir / (f.stem + "_compare.jpg"), np.hstack([rgb, out]))
        elif not a.no_show:
            cv2.imshow(f.name, compose(rgb, out, dark_channel_fog_score(rgb), 1 / dt, True))
            cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
