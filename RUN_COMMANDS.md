# Run commands: copy and paste into a terminal

Every command the project needs, in order. **Windows PowerShell** is shown first, and Linux / Raspberry Pi
equivalents are at the end. Lines starting with `#` are comments.

> Run each block from the folder named above it (`Fog_removal`, `gunet_dehaze` or `fognet_lite`).
> Keep the Python environment and the datasets **outside OneDrive**.

---

## 0. One-time setup

```powershell
# get the code (skip if you already have the folder)
git clone https://github.com/ritik3041998/Fog_removal.git
cd Fog_removal

# Python environment outside OneDrive
python -m venv $env:USERPROFILE\.venvs\fogdehaze
& $env:USERPROFILE\.venvs\fogdehaze\Scripts\Activate.ps1

# PyTorch with NVIDIA GPU support (CPU only: pip install torch torchvision)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r gunet_dehaze/requirements.txt
pip install -r fognet_lite/requirements.txt
pip install huggingface_hub pyarrow

# optional: official gUNet code, used only by check_model.py
git clone https://github.com/IDKiro/gUNet gunet_dehaze/third_party/gUNet
```

If PowerShell refuses to run `Activate.ps1`, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 1. Every new terminal

```powershell
& $env:USERPROFILE\.venvs\fogdehaze\Scripts\Activate.ps1
$D = "$env:USERPROFILE\fogdata"          # dataset folder used below
```

---

## 2. Test the pretrained model now (no training, no dataset)

From `gunet_dehaze`:

```powershell
cd gunet_dehaze
$W = "pretrained/dehazeformer_demo/dehazeformer.pth"

python scripts/infer.py --weights $W --source samples/online   --out results/demo        # 6 sample photos
python scripts/infer.py --weights $W --source samples/internet --out results/internet    # 23 internet photos
python scripts/infer.py --weights $W --source C:\path\to\my_foggy.jpg --out results/mine  # your own photo
python scripts/infer.py --weights $W --source C:\path\to\foggy.mp4 --out results/clear.mp4  # a video
python scripts/infer.py --weights $W --source 0                                         # live webcam
python scripts/infer.py --weights $W --source 0 --fog-threshold 0.3                     # dehaze only foggy frames
```

Live window keys: **q** quits, **space** pauses, **d** turns dehazing on or off. The results are saved as
`*_compare.jpg` (input on the left, dehazed on the right).

Download more real foggy photos (Wikimedia Commons, free licenses, author and license saved in `credits.csv`):
```powershell
python scripts/download_test_images.py --out samples/internet --per-query 3
python scripts/download_test_images.py --out samples/more --queries "foggy street" "fog highway night" --per-query 5
```

---

## 3. Download the datasets (~2 GB, once)

From `fognet_lite`:

```powershell
cd fognet_lite
python download_data.py --root $env:USERPROFILE\fogdata
```

This creates `RESIDE-6K\train\{hazy,GT}` (6000 training pairs), `SOTS\SOTS\outdoor\{hazy,gt}` (500 test pairs),
`NH-HAZE2\data\train_notest\{hazy,clean}` (70 real-haze pairs), `NH-HAZE2\data\test\{hazy,clean}` and `RTTS`.

---

## 4. Score the pretrained model on benchmarks

From `gunet_dehaze`:

```powershell
python scripts/test_pretrained.py --pairs $D\SOTS\SOTS\outdoor --limit 0 --out results/sots_outdoor     # synthetic fog
python scripts/test_pretrained.py --pairs $D\NH-HAZE2\data\test --limit 0 --out results/nhhaze2         # real fog
```

---

## 5. Main model (gUNet): train, evaluate, export

From `gunet_dehaze`. Training takes about 4 hours for 100 epochs on an RTX 3050. Don't run two trainings on the GPU at the same time.

```powershell
# check the model code + speed table (optional)
python scripts/check_model.py

# train: RESIDE-6K + real haze (NH-HAZE2 listed 3x so it is sampled more often), validate on SOTS-outdoor
python scripts/train.py --config configs/gunet_t_laptop.json `
    --train-dir $D\RESIDE-6K\train $D\NH-HAZE2\data\train_notest $D\NH-HAZE2\data\train_notest $D\NH-HAZE2\data\train_notest `
    --val-dir $D\SOTS\SOTS\outdoor --out runs/gunet_t

# continue after a stop or crash
python scripts/train.py --config configs/gunet_t_laptop.json --out runs/gunet_t --resume

# watch loss / PSNR curves at http://localhost:6006
tensorboard --logdir runs

# smaller model for a Pi without an AI accelerator
python scripts/train.py --config configs/gunet_n_pi_cpu.json --train-dir $D\RESIDE-6K\train --val-dir $D\SOTS\SOTS\outdoor --out runs/gunet_n

# evaluate (full-resolution PSNR / SSIM)
python scripts/evaluate.py --ckpt runs/gunet_t/best.pth --data $D\SOTS\SOTS\outdoor --save-dir results/gunet_sots
python scripts/evaluate.py --ckpt runs/gunet_t/best.pth --data $D\NH-HAZE2\data\test --save-dir results/gunet_nhhaze2

# compare against the pretrained DehazeFormer on the same data
python scripts/test_pretrained.py --model gunet --ckpt runs/gunet_t/best.pth --pairs $D\SOTS\SOTS\outdoor --limit 0

# run your trained model on photos / webcam
python scripts/infer.py --weights runs/gunet_t/best.pth --source samples/internet --out results/gunet_internet
python scripts/infer.py --weights runs/gunet_t/best.pth --source 0 --half

# export for the Raspberry Pi
python scripts/export_onnx.py --ckpt runs/gunet_t/best.pth --height 240 --width 320    # -> exports/gunet_t_240x320.onnx
python scripts/infer.py --weights exports/gunet_t_240x320.onnx --source samples/internet --out results/onnx_test

# make training pairs from your own clear camera video
python scripts/make_synthetic_fog.py --src C:\path\to\clear_video.mp4 --every 15 --out $D\mycam --depth midas
python scripts/train.py --config configs/gunet_t_finetune.json --init runs/gunet_t/best.pth `
    --train-dir $D\mycam $D\RESIDE-6K\train --val-dir $D\SOTS\SOTS\outdoor --out runs/gunet_t_mycam
```

---

## 6. Lightweight model (FogNet-Lite): train, test, export

From `fognet_lite`:

```powershell
cd ..\fognet_lite

python -m fognet.train --paired $D\RESIDE-6K\train\hazy $D\RESIDE-6K\train\GT `
    --real $D\NH-HAZE2\data\train_notest\hazy $D\NH-HAZE2\data\train_notest\clean `
    --synthetic $D\RESIDE-6K\train\GT --synthetic-repeats 1 `
    --val $D\SOTS\SOTS\outdoor\hazy $D\SOTS\SOTS\outdoor\gt --epochs 30 --workers 8 --out runs\fognet

python -m fognet.export_onnx --ckpt runs\fognet\best.pt --out fognet.onnx
python benchmark.py --model fognet.onnx                                                    # speed (run on the Pi too)

# synthetic fog with ground truth (add --external ..\gunet_dehaze\results\gunet_sots\imgs to compare gUNet)
python compare.py --hazy $D\SOTS\SOTS\outdoor\hazy --clear $D\SOTS\SOTS\outdoor\gt --model fognet.onnx
# real fog with ground truth
python compare.py --hazy $D\NH-HAZE2\data\test\hazy --clear $D\NH-HAZE2\data\test\clean --model fognet.onnx
# real fog without ground truth
python realworld_test.py --images $D\RTTS\images\test --model fognet.onnx
```

---

## 7. Raspberry Pi (64-bit Raspberry Pi OS)

Copy these files to the Pi: the `.onnx` model(s), `fognet_lite/pi_infer.py` + `fognet_lite/fognet/`, and
`gunet_dehaze/scripts/infer.py` + `gunet_dehaze/fogdehaze/`.

```bash
sudo apt install -y python3-picamera2 python3-opencv      # Pi Camera Module support
python3 -m venv --system-site-packages ~/fogenv && source ~/fogenv/bin/activate
pip install onnxruntime numpy

# FogNet-Lite (fastest on the Pi CPU)
python pi_infer.py --model fognet.onnx --source picam --show     # Pi Camera Module
python pi_infer.py --model fognet.onnx --source 0 --show         # USB webcam
python pi_infer.py --dcp --source picam --show                   # no model: Dark Channel Prior
python benchmark.py --model fognet.onnx                          # FPS on the Pi

# gUNet ONNX (USB webcam; needs the AI HAT+ for real-time speed)
python scripts/infer.py --weights gunet_t_240x320.onnx --source 0 --fog-threshold 0.3
```

---

## 8. Linux / macOS equivalents of the PC commands

```bash
python3 -m venv ~/.venvs/fogdehaze && source ~/.venvs/fogdehaze/bin/activate
D=~/fogdata
# then use the same commands with $D/... paths and "\" instead of the PowerShell backtick ` for line breaks
```

## 9. Save your changes to GitHub

```powershell
cd <path>\Fog_removal          # the repo root
git status
git add -A
git commit -m "describe what you changed"
git push
```
