"""PSNR / SSIM (torch, images in [0,1]) used for the training loss and evaluation."""
import torch
import torch.nn.functional as F


def psnr(pred, target, eps=1e-10):
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
    return (10 * torch.log10(1.0 / (mse + eps))).mean()


def _gauss_window(size=11, sigma=1.5, channels=3, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(channels, 1, size, size).contiguous()


def ssim(pred, target, size=11):
    c = pred.shape[1]
    w = _gauss_window(size, channels=c, device=pred.device)
    mu_x = F.conv2d(pred, w, groups=c)
    mu_y = F.conv2d(target, w, groups=c)
    sxx = F.conv2d(pred * pred, w, groups=c) - mu_x ** 2
    syy = F.conv2d(target * target, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(pred * target, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean()
