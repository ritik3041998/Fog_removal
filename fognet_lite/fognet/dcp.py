"""Dark Channel Prior dehazing (He et al., CVPR 2009) - fast version.

No training needed, and because it is based on a physical rule rather than on
training data, it holds up well on real fog. Used here as:
  * a baseline to compare FogNet-Lite against
  * a fallback on the Pi if no trained model is available

Speed-up: transmission is estimated on a downscaled copy and refined with a
guided filter, then upsampled; the final recovery is done at full resolution.
"""
import cv2
import numpy as np


def _dark_channel(img, patch):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(img.min(axis=2), k)


def _atmospheric_light(img, dark):
    n = max(1, dark.size // 1000)                 # brightest 0.1% of the dark channel
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    return img.reshape(-1, 3)[idx].mean(axis=0)


def _guided_filter(guide, src, r, eps):
    mean = lambda x: cv2.boxFilter(x, -1, (r, r))
    mi, ms = mean(guide), mean(src)
    a = (mean(guide * src) - mi * ms) / (mean(guide * guide) - mi * mi + eps)
    b = ms - a * mi
    return mean(a) * guide + mean(b)


def fog_score(bgr_u8, width=160):
    """Rough fog level in [0,1] (mean dark channel of a small copy).

    Clear outdoor scenes are usually < 0.25; foggy ones are usually > 0.4.
    Cheap enough (<1 ms) to run every frame on a Pi.
    """
    h, w = bgr_u8.shape[:2]
    small = cv2.resize(bgr_u8, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return float(_dark_channel(small.astype(np.float32) / 255.0, 7).mean())


class DCPDehazer:
    def __init__(self, omega=0.92, t_min=0.15, patch=15, work_width=320, smooth_A=0.9):
        self.omega, self.t_min, self.patch = omega, t_min, patch
        self.work_width, self.smooth_A = work_width, smooth_A
        self._A = None   # smoothed over video frames so brightness does not flicker

    def __call__(self, bgr_u8):
        img = bgr_u8.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        s = min(1.0, self.work_width / w)
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        patch = max(3, int(self.patch * s) | 1)

        A = _atmospheric_light(small, _dark_channel(small, patch))
        if self._A is not None:
            A = self.smooth_A * self._A + (1 - self.smooth_A) * A
        self._A = A

        t = 1 - self.omega * _dark_channel(small / np.maximum(A, 1e-3), patch)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        t = _guided_filter(gray, t.astype(np.float32), r=max(8, int(60 * s)), eps=1e-3)
        if s < 1:
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
        t = np.maximum(t, self.t_min)[..., None]
        out = (img - A) / t + A
        return (out.clip(0, 1) * 255).astype(np.uint8)
