"""FogNet-Lite: a tiny dehazing CNN built for real-time use on a Raspberry Pi.

Based on AOD-Net (Li et al., ICCV 2017), which rewrites the atmospheric
scattering model  I = J*t + A*(1-t)  as

    J = K(I) * I - K(I) + b

so the network only has to predict one map K instead of t and A separately.

Changes for speed and real fog:
  * K is estimated at 1/4 resolution (``scale``) and upsampled. K varies
    smoothly with depth and fog density, so this loses very little quality but
    cuts compute by ~scale^2. The final formula is applied at full resolution,
    so output edges stay sharp.
  * Depthwise-separable convs with a little more width than AOD-Net's 3 channels,
    which helps with the non-uniform, coloured fog seen in real scenes.
  * Multi-scale concatenation like AOD-Net (small + large receptive fields).

About 6k parameters. ONNX export supports any input size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise-separable conv + ReLU."""

    def __init__(self, cin, cout, k, dilation=1):
        super().__init__()
        pad = dilation * (k // 2)
        self.dw = nn.Conv2d(cin, cin, k, padding=pad, dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class FogNetLite(nn.Module):
    def __init__(self, width=16, scale=4):
        super().__init__()
        self.scale = scale
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, padding=1), nn.ReLU(inplace=True))
        self.c1 = DSConv(w, w, 3)                  # fine detail
        self.c2 = DSConv(w, w, 5)                  # medium
        self.c3 = DSConv(2 * w, w, 5, dilation=2)  # large context
        self.c4 = DSConv(2 * w, w, 7, dilation=2)  # very large context (global fog level)
        self.head = nn.Conv2d(4 * w, 3, 3, padding=1)
        self.b = 1.0

    def estimate_k(self, x):
        x0 = self.stem(x)
        x1 = self.c1(x0)
        x2 = self.c2(x1)
        x3 = self.c3(torch.cat([x1, x2], 1))
        x4 = self.c4(torch.cat([x2, x3], 1))
        return F.relu(self.head(torch.cat([x1, x2, x3, x4], 1)))

    def forward(self, img):
        """img: (N,3,H,W) RGB in [0,1]. Returns the dehazed image in [0,1]."""
        if self.scale > 1:
            small = F.interpolate(img, scale_factor=1.0 / self.scale, mode="bilinear",
                                  align_corners=False, recompute_scale_factor=False)
            k = self.estimate_k(small)
            k = F.interpolate(k, size=img.shape[-2:], mode="bilinear", align_corners=False)
        else:
            k = self.estimate_k(img)
        out = k * img - k + self.b
        return out.clamp(0.0, 1.0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = FogNetLite()
    print(f"params: {count_params(m):,}")
    y = m(torch.rand(1, 3, 480, 640))
    print("out:", tuple(y.shape))
