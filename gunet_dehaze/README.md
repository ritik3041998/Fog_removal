# gUNet dehazing (fog removal) for live video on edge devices

This is the main dehazing model, based on [IDKiro/gUNet](https://github.com/IDKiro/gUNet) (MIT license).
It takes a foggy RGB frame and returns a clear frame. It is trained on a PC GPU,
exported to ONNX, and run on a Raspberry Pi 5 (CPU, or AI HAT+ via Hailo).

```
gunet_dehaze/
├── fogdehaze/models/gunet.py   gUNet, export-friendly (same weights as official)
├── fogdehaze/data.py           paired IN/ (hazy) + GT/ (clear) loader
├── fogdehaze/utils.py          image I/O, PSNR, dark-channel fog score
├── configs/                    training settings
├── scripts/train.py            train / fine-tune / resume
├── scripts/evaluate.py         full-resolution PSNR / SSIM on a test set
├── scripts/infer.py            image / folder / video / webcam, .pth or .onnx
├── scripts/export_onnx.py      fixed-size ONNX export + correctness check
├── scripts/make_synthetic_fog.py   clear images/video -> hazy training pairs
├── scripts/check_model.py      equivalence vs official code + speed table
└── third_party/gUNet/          official repo (reference)
```

## What was changed from the official gUNet

| Change | Why |
|---|---|
| `SKFusion` rewritten with 4-D ops (`sigmoid(a0-a1)` instead of a 5-D softmax) | Mathematically identical (max diff 1e-7, checked by `check_model.py`), but the graph exports cleanly to ONNX / Hailo / TFLite |
| `pad_mode="zeros"` option | Reflect padding adds 24 Pad/Slice/Reshape/Transpose chains to the ONNX graph. With zeros, the graph is only Conv, BN, Sigmoid, Mul, Add, DepthToSpace, GAP, ReLU, Split and Sub, which suits accelerators. |
| Smaller variants `gunet_n`, `gunet_p` | For a CPU-only Pi |
| Single-GPU training: AMP, warmup + cosine LR, L1 + SSIM loss, resume, TensorBoard | The official recipe assumes 4 GPUs and batch size 128 |
| No `timm` dependency | Simpler install |

## Setup (Windows training PC)

Keep the environment and datasets **outside OneDrive**, because they are many GB.

```powershell
python -m venv $env:USERPROFILE\.venvs\fogdehaze
$py = "$env:USERPROFILE\.venvs\fogdehaze\Scripts\python.exe"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
& $py -m pip install -r requirements.txt
cd gunet_dehaze
git clone https://github.com/IDKiro/gUNet third_party/gUNet   # optional: official code, used by check_model.py
& $py scripts/check_model.py          # should print "OK" + speed table
```

## Data

Each dataset is a folder with `IN/` (hazy) and `GT/` (clear) subfolders. Raw RESIDE names
(`0001_0.8_0.2.jpg` -> `0001.jpg`) are matched automatically.

| Use | Dataset |
|---|---|
| Train (outdoor) | RESIDE **OTS**, or a 10-20k image subset of it |
| Validate / test | **SOTS-outdoor** (500 pairs) |
| Real fog test | O-HAZE, NH-HAZE, RTTS (no GT, visual only) |
| Your camera | `make_synthetic_fog.py` on clear footage from the Pi camera |

```powershell
# turn clear camera footage into training pairs (MiDaS depth gives more realistic fog)
& $py scripts/make_synthetic_fog.py --src clear_drive.mp4 --every 15 --out $env:USERPROFILE\fogdata\mycam --depth midas
```

Official pretrained weights are linked from the gUNet README (Google Drive). Automated downloads
returned 404, so you may need to download them in a browser. If you get them, put them in
`pretrained/reside-out/gunet_t.pth` and use `configs/gunet_t_finetune.json`.

## Train

```powershell
# from scratch (pad_mode=zeros, Hailo-friendly). About 4 h for 100 epochs on an RTX 3050 laptop GPU.
& $py scripts/train.py --config configs/gunet_t_laptop.json `
    --train-dir $env:USERPROFILE\fogdata\OTS --val-dir $env:USERPROFILE\fogdata\SOTS-outdoor --out runs/gunet_t

# fine-tune official weights (or your own model) on your camera data
& $py scripts/train.py --config configs/gunet_t_finetune.json --init pretrained/reside-out/gunet_t.pth `
    --train-dir $env:USERPROFILE\fogdata\mycam --val-dir $env:USERPROFILE\fogdata\SOTS-outdoor --out runs/gunet_t_ft

& $py scripts/train.py --config configs/gunet_t_laptop.json --out runs/gunet_t --resume   # continue
& $py -m tensorboard.main --logdir runs                                                   # curves
```

Override any setting with `--set key=value`, for example `--set batch_size=8 epochs=50`.
If you run out of GPU memory, lower `batch_size` (8 still works with BatchNorm).

## Pretrained baseline: DehazeFormer-MCT (works today, no training)

The official gUNet weights are only on Google Drive, which refused automated download. The same
authors publish a pretrained **DehazeFormer-MCT** (1.23 M params, trained on a mixed set for real haze)
in their [Hugging Face demo](https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo). The weights are in
`pretrained/dehazeformer_demo/`, and the reviewed model code is in `fogdehaze/models/dehazeformer_mct.py`.

```powershell
& $py scripts/test_pretrained.py --source samples/online --pairs $env:USERPROFILE\fogdata\RESIDE-6K\train --limit 200
& $py scripts/test_pretrained.py --model gunet --ckpt runs/gunet_t/best.pth --pairs ...   # compare our model
```

On 200 RESIDE-6K pairs: hazy input 15.29 dB, dehazed **27.71 dB PSNR / 0.919 SSIM**. This may be optimistic,
because the demo's training mix may include RESIDE images. Its latency at 640x480 was 141 ms on the RTX 3050
and about 2 s on the laptop CPU. It uses window attention, so it is a quality reference rather than a Pi model.

## Evaluate, export, run

```powershell
& $py scripts/evaluate.py --ckpt runs/gunet_t/best.pth --data $env:USERPROFILE\fogdata\SOTS-outdoor --save-dir results/sots
& $py scripts/export_onnx.py --ckpt runs/gunet_t/best.pth --height 240 --width 320
& $py scripts/infer.py --weights runs/gunet_t/best.pth --source 0                 # laptop webcam, GPU
& $py scripts/infer.py --weights exports/gunet_t_240x320.onnx --source foggy.mp4 --out clear.mp4
```

`results/sots/imgs/<hazy stem>.png` can be compared against FogNet-Lite with
`python compare.py ... --external ../gunet_dehaze/results/sots` (from `../fognet_lite`).

`infer.py` also shows a **dark-channel fog score** (0 = clear, higher = foggier). Use
`--fog-threshold 0.3` to dehaze only foggy frames. Calibrate the threshold on your own camera.

## Size and speed (measured on the training laptop at 640x480)

| Variant | Params | GMACs | PC CPU | RTX 3050 fp16 |
|---|---|---|---|---|
| gunet_p | 0.24 M | 3.8 | 264 ms | 19 ms |
| gunet_n | 0.28 M | 5.6 | 511 ms | 32 ms |
| **gunet_t** | 0.85 M | 12.9 | 842 ms | 49 ms |
| gunet_s | 1.48 M | 22.8 | 1567 ms | 95 ms |

MACs scale with pixel count, so 320x240 costs about 1/4 of these numbers. On a Pi 5 **CPU**, expect
gUNet-T to manage only a few frames per second at 320x240. Use `gunet_n`/`gunet_p` there, or the
**AI HAT+**, which is the intended target for gUNet-T. Measure on the device.

## Raspberry Pi deployment

- **CPU only:** copy the `.onnx` file and `scripts/infer.py` + `fogdehaze/utils.py`, then
  `pip install onnxruntime opencv-python numpy`. Run with `--source 0`, or feed frames from `picamera2`.
- **AI HAT+ (Hailo-8 / 8L):** compile the ONNX on an x86 Linux PC with the Hailo Dataflow Compiler
  (parse, optimize/quantize to INT8 with about 500 calibration frames from your camera, compile to `.hef`),
  then run with HailoRT on the Pi. Use a `pad_mode=zeros` model.
