# FogNet-Lite: our own real-time fog removal model for the Raspberry Pi

This is our own model, built alongside the online repo (gUNet, in `../gunet_dehaze/`).
Run every command below from inside this `fognet_lite/` folder.

## Why this model

| | FogNet-Lite (ours) | AOD-Net | gUNet-T (online repo) | DCP (no training) |
|---|---|---|---|---|
| Parameters | 6.7k | 1.7k | ~0.65M | 0 |
| Measured on the dev PC, 640x480, 4 threads | 6 ms | - | - | ~22 ms |
| Real time on a Pi CPU at 640x480 | expected yes (not yet measured on a Pi) | yes | no | borderline |
| Needs training | yes | yes | yes / pretrained | no |

A Pi is roughly 4-8x slower than a laptop CPU, so expect about 25-50 ms per frame for FogNet-Lite at
640x480 on a Pi 4 and less on a Pi 5. Measure with `pi_infer.py --show` (it prints FPS).
If it's too slow, use `--width 320 --height 240`, or train with `--scale 6`.

* **Architecture:** AOD-Net's formula `J = K·I − K + 1`, so the network only has to learn one map K.
* **Speed trick:** K changes smoothly across the image, so it is estimated at 1/4 resolution
  (`--scale 4`) and upsampled. The output itself is computed at full resolution, so edges stay sharp.
  On the dev PC this was 4x faster than 1/2 resolution (6 ms against 24 ms).
* **Depthwise-separable convs:** these allow 16 channels with large dilated kernels (big receptive
  field for patchy fog) at very low cost.
* **Real-fog training:** `fognet/synth_fog.py` adds non-uniform fog, fog that thickens towards the horizon,
  yellow (street lights) or blue tinted fog, and sensor noise, blur and JPEG compression. Real fog
  looks like this, but the standard RESIDE dataset doesn't include it.
* **Fallback:** the Dark Channel Prior (`fognet/dcp.py`) runs when no trained model exists. A `fog_score`
  measure lets the Pi skip dehazing on clear frames.

## 1. Training (laptop / Colab)

```bash
pip install -r requirements.txt

# Best: RESIDE OTS (outdoor) + your own clear photos with generated fog, validated on SOTS-outdoor
python -m fognet.train --paired data/OTS/hazy data/OTS/clear \
                       --synthetic data/my_clear_photos \
                       --val data/SOTS/outdoor/hazy data/SOTS/outdoor/gt \
                       --epochs 30 --out runs/fognet

# Without RESIDE: only clear photos (fog is generated during training)
python -m fognet.train --synthetic data/my_clear_photos --epochs 50 --out runs/fognet
```

Tip for real-world quality: add a few hundred clear photos from **your own camera and location** to
`--synthetic`. The model then learns your camera's colours and noise.

## 2. Export to ONNX

```bash
python -m fognet.export_onnx --ckpt runs/fognet/best.pt --out fognet.onnx
```

## 3. Run on the Raspberry Pi (64-bit Raspberry Pi OS, no PyTorch needed)

Copy `fognet.onnx`, `pi_infer.py`, `fognet/dcp.py` and `fognet/__init__.py` (keep the `fognet/` folder) to the Pi, then:

```bash
pip install -r requirements-pi.txt          # Pi Camera Module: sudo apt install python3-picamera2
python pi_infer.py --model fognet.onnx --source 0 --show          # USB webcam
python pi_infer.py --model fognet.onnx --source picam --show      # Pi Camera Module
python pi_infer.py --model fognet.onnx --source foggy.mp4 --save out.mp4
python pi_infer.py --model fognet.onnx --source 0 --auto 0.35     # only dehaze when foggy
python pi_infer.py --dcp --source 0 --show                        # no model yet
```

Lower `--width/--height` (e.g. 320x240) for more FPS.

## 4. Compare with DCP and the online repo

```bash
python compare.py --hazy data/SOTS/outdoor/hazy --clear data/SOTS/outdoor/gt \
                  --model fognet.onnx --external ../gunet_dehaze/results/sots --samples 6
```

This prints PSNR/SSIM and ms per image for: the foggy input, DCP, FogNet-Lite, and the online repo's
saved outputs. Side-by-side images go to `eval_samples/`.
