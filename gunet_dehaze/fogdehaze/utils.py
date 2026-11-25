import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def read_rgb(path):
    """Read an image as uint8 RGB (H, W, 3). Handles non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path, img_rgb_u8):
    ext = str(path)[str(path).rfind("."):]
    ok, buf = cv2.imencode(ext, cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"could not encode image: {path}")
    buf.tofile(str(path))


def to_tensor(img_rgb_u8):
    """uint8 HWC RGB -> float CHW in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(img_rgb_u8)).permute(2, 0, 1).float()
    return t / 127.5 - 1.0


def to_image(t):
    """float CHW (or 1xCHW) in [-1, 1] -> uint8 HWC RGB."""
    if t.dim() == 4:
        t = t[0]
    t = ((t.detach().float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return t.permute(1, 2, 0).cpu().numpy()


def pad_to_multiple(x, multiple):
    """Reflect-pad a BCHW tensor so H and W are multiples of `multiple`."""
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x


def psnr(output, target):
    """Per-image PSNR for tensors in [-1, 1]. Returns a (B,) tensor."""
    mse = F.mse_loss(output * 0.5 + 0.5, target * 0.5 + 0.5, reduction="none").mean((1, 2, 3))
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def dark_channel_fog_score(img_rgb_u8, patch=15, size=160):
    """Cheap haze-density estimate in [0, 1] using the dark channel prior.

    Clear outdoor images have a dark channel close to 0; fog lifts it towards 1.
    Useful as a fog *detector* to decide whether to run the dehazing model.
    Thresholds must be calibrated on your own camera (roughly: <0.25 clear,
    0.25-0.45 light haze, >0.45 dense fog).
    """
    h, w = img_rgb_u8.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(img_rgb_u8, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    dark = small.min(axis=2)
    dark = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))
    return float(dark.mean()) / 255.0
