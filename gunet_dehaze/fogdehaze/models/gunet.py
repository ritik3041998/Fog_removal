"""gUNet dehazing network, adapted for edge deployment.

Based on the official implementation of "Rethinking Performance Gains in Image
Dehazing Networks" (Song et al., 2022) - https://github.com/IDKiro/gUNet
Original code is MIT licensed (see third_party/gUNet/LICENSE).

Changes from the original:
  * No `timm` dependency (uses torch.nn.init.trunc_normal_).
  * SKFusion rewritten with 4-D tensors only. For two branches,
    softmax([a0, a1])[0] == sigmoid(a0 - a1), so the output is mathematically
    identical and the original pretrained weights load unchanged, but the graph
    no longer contains 5-D reshapes / softmax over a stacked axis, which edge
    compilers (Hailo DFC, TFLite, some ONNX runtimes) handle poorly.
  * `pad_mode` option: 'reflect' (original, required for official weights) or
    'zeros' (simpler graph for accelerators; train from scratch / fine-tune).
  * Extra smaller variants (`gunet_n`, `gunet_p`) for CPU-only Raspberry Pi use.

Parameter names are kept identical to the original so checkpoints from the
official repo load with `load_gunet_checkpoint`.
"""
import math

import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_


class ConvLayer(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid, pad_mode="reflect"):
        super().__init__()
        self.dim = dim
        self.net_depth = net_depth
        self.kernel_size = kernel_size

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=dim, padding_mode=pad_mode),
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True),
        )
        self.proj = nn.Conv2d(dim, dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.proj(self.Wv(x) * self.Wg(x))


class BasicBlock(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, norm_layer=nn.BatchNorm2d,
                 gate_act=nn.Sigmoid, pad_mode="reflect"):
        super().__init__()
        self.norm = norm_layer(dim)
        self.conv = ConvLayer(net_depth, dim, kernel_size, gate_act, pad_mode)

    def forward(self, x):
        return x + self.conv(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(self, net_depth, dim, depth, kernel_size=3, norm_layer=nn.BatchNorm2d,
                 gate_act=nn.Sigmoid, pad_mode="reflect"):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            BasicBlock(net_depth, dim, kernel_size, norm_layer, gate_act, pad_mode)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None, pad_mode="reflect"):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode=pad_mode)

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None, pad_mode="reflect"):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=pad_mode),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        return self.proj(x)


class SKFusion(nn.Module):
    """Selective-kernel fusion of (upsampled feature, skip feature).

    Equivalent to the original 5-D softmax formulation for height=2.
    """

    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        assert height == 2, "export-friendly SKFusion supports exactly 2 inputs"
        self.height = height
        self.dim = dim
        d = max(int(dim / reduction), 4)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim * height, 1, bias=False),
        )

    def forward(self, in_feats):
        x0, x1 = in_feats
        attn = self.mlp(x0 + x1)                       # B, 2C, 1, 1
        a0, a1 = torch.split(attn, self.dim, dim=1)    # matches view(B, 2, C, 1, 1)
        w0 = torch.sigmoid(a0 - a1)                    # == softmax over the 2 branches
        return x1 + w0 * (x0 - x1)                     # == w0*x0 + (1-w0)*x1


class gUNet(nn.Module):
    def __init__(self, kernel_size=5, base_dim=32, depths=(4, 4, 4, 4, 4, 4, 4),
                 norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, pad_mode="reflect"):
        super().__init__()
        depths = list(depths)
        assert len(depths) % 2 == 1
        stage_num = len(depths)
        half_num = stage_num // 2
        net_depth = sum(depths)
        embed_dims = [2 ** i * base_dim for i in range(half_num)]
        embed_dims = embed_dims + [2 ** half_num * base_dim] + embed_dims[::-1]

        # input H and W must be multiples of this
        self.patch_size = 2 ** (stage_num // 2)
        self.stage_num = stage_num
        self.half_num = half_num

        self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0],
                                 kernel_size=3, pad_mode=pad_mode)

        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for i in range(self.stage_num):
            self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth,
                                          kernel_size=kernel_size, norm_layer=norm_layer,
                                          gate_act=gate_act, pad_mode=pad_mode))
        for i in range(self.half_num):
            self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i],
                                         embed_dim=embed_dims[i + 1], pad_mode=pad_mode))
            self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i],
                                         embed_dim=embed_dims[i + 1], pad_mode=pad_mode))
            self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.fusions.append(SKFusion(embed_dims[i]))

        self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1],
                                    kernel_size=3, pad_mode=pad_mode)

    def forward(self, x):
        """x: B x 3 x H x W, RGB in [-1, 1]. Returns dehazed image in the same range."""
        feat = self.inconv(x)
        skips = []
        for i in range(self.half_num):
            feat = self.layers[i](feat)
            skips.append(self.skips[i](feat))
            feat = self.downs[i](feat)

        feat = self.layers[self.half_num](feat)

        for i in range(self.half_num - 1, -1, -1):
            feat = self.ups[i](feat)
            feat = self.fusions[i]([feat, skips[i]])
            feat = self.layers[self.stage_num - i - 1](feat)

        return self.outconv(feat) + x


# --------------------------------------------------------------------------
# Variants. t/s/b/d match the official repo (pretrained weights available).
# n/p are extra-small variants for CPU-only edge devices (train from scratch).
# --------------------------------------------------------------------------
VARIANTS = {
    "gunet_p": dict(kernel_size=5, base_dim=16, depths=[1, 1, 1, 2, 1, 1, 1]),
    "gunet_n": dict(kernel_size=5, base_dim=16, depths=[2, 2, 2, 2, 2, 2, 2]),
    "gunet_t": dict(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2]),
    "gunet_s": dict(kernel_size=5, base_dim=24, depths=[4, 4, 4, 8, 4, 4, 4]),
    "gunet_b": dict(kernel_size=5, base_dim=24, depths=[8, 8, 8, 16, 8, 8, 8]),
    "gunet_d": dict(kernel_size=5, base_dim=24, depths=[16, 16, 16, 32, 16, 16, 16]),
}


def build_gunet(name="gunet_t", pad_mode="reflect"):
    if name not in VARIANTS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(VARIANTS)}")
    return gUNet(pad_mode=pad_mode, **VARIANTS[name])


def load_gunet_checkpoint(model, path, strict=True):
    """Load either an official gUNet checkpoint or one saved by our train.py."""
    # weights_only=True: safe for checkpoints downloaded from the internet
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return ckpt, missing, unexpected
