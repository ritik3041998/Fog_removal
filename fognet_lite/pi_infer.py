"""Real-time fog removal on a Raspberry Pi (also works on a PC).

Needs only:  numpy, opencv-python, onnxruntime   (no PyTorch)

  # USB webcam, show window with original | dehazed
  python pi_infer.py --model fognet.onnx --source 0 --show

  # Pi Camera Module (Bookworm / libcamera, needs python3-picamera2)
  python pi_infer.py --model fognet.onnx --source picam --show

  # Video file -> output video
  python pi_infer.py --model fognet.onnx --source foggy.mp4 --save out.mp4

  # Folder of images -> folder of results
  python pi_infer.py --model fognet.onnx --source foggy_images/ --save results/

  # No trained model yet: use Dark Channel Prior instead
  python pi_infer.py --dcp --source 0 --show

  --auto 0.35   only dehaze when the fog level is above 0.35 (saves CPU on clear days)
"""
import argparse
import os
import time

import cv2
import numpy as np

from fognet.dcp import DCPDehazer, fog_score

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class OnnxDehazer:
    def __init__(self, path, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, bgr_u8):
        x = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) * (1 / 255.0)
        x = x.transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.name: np.ascontiguousarray(x)})[0][0]
        y = (y.transpose(1, 2, 0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


class PiCam:
    def __init__(self, w, h):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": (w, h), "format": "RGB888"}))
        self.cam.start()

    def read(self):
        return True, self.cam.capture_array()   # "RGB888" in picamera2 is BGR byte order

    def release(self):
        self.cam.stop()


def open_source(src, w, h):
    if src == "picam":
        return PiCam(w, h)
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if src.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {src}")
    return cap


def run_folder(dehaze, src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(src)) if f.lower().endswith(EXTS)]
    times = []
    for f in files:
        img = cv2.imread(os.path.join(src, f))
        if img is None:
            continue
        t0 = time.perf_counter()
        out = dehaze(img)
        times.append(time.perf_counter() - t0)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(f)[0] + ".png"), out)
    if times:
        print(f"{len(times)} images, mean {1000 * np.mean(times):.1f} ms/image -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fognet.onnx")
    ap.add_argument("--dcp", action="store_true", help="use Dark Channel Prior instead of the model")
    ap.add_argument("--source", default="0", help="camera index, 'picam', video file, or image folder")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--auto", type=float, default=0.0, help="fog-level threshold; 0 = always dehaze")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", help="output video file, or output folder for an image folder")
    args = ap.parse_args()

    dehaze = DCPDehazer() if args.dcp else OnnxDehazer(args.model, args.threads)

    if os.path.isdir(args.source):
        return run_folder(dehaze, args.source, args.save or "results")

    cap = open_source(args.source, args.width, args.height)
    writer, fps, n = None, 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != args.width:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            score = fog_score(frame)
            foggy = score >= args.auto
            out = dehaze(frame) if foggy else frame
            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6) if n else 1 / max(dt, 1e-6)
            n += 1

            view = np.hstack([frame, out]) if args.show else out
            if args.show:
                label = f"{fps:5.1f} FPS  fog {score:.2f}  {'DEHAZE' if foggy else 'clear - skipped'}"
                cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            if args.show:
                cv2.imshow("fog removal  (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif n % 30 == 0:
                print(f"{fps:.1f} FPS  fog {score:.2f}", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
