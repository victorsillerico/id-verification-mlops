import cv2
import numpy as np

rng = np.random.default_rng(0)   # fixed seed, so results are repeatable
MIN_BLUR = 100.0     # below this the image is too blurry
MAX_GLARE = 0.05     # more than 5% blown-out pixels is too much
MIN_SIDE = 600       # smallest allowed width or height, in pixels


def quality_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "glare": float((gray > 240).mean()),
        "height": img.shape[0],
        "width": img.shape[1],
    }

def quality_ok(q):
    return (
        q["blur"] >= MIN_BLUR
        and q["glare"] <= MAX_GLARE
        and min(q["height"], q["width"]) >= MIN_SIDE
    )

def recapture(img):
    """Photo of a screen: soft, bluish, fine pixel grid, a glare spot."""
    out = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    h, w = out.shape[:2]
    out += 8 * np.sin(np.arange(w) * 1.5)[None, :, None]   # vertical lines
    out[:, :, 2] *= 1.08                                   # more blue
    cx, cy = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4)
    yy, xx = np.ogrid[:h, :w]
    spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.15 * w) ** 2))
    out += 120 * spot[:, :, None]                          # bright blob
    return np.clip(out, 0, 255).astype(np.uint8)

def print_scan(img):
    """Printed copy photographed again: washed-out colors and grain."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = 0.5 * img + 0.5 * gray                           # half the color
    out = out * 0.7 + 60                                   # lower contrast
    out += rng.normal(0, 8, out.shape)                     # paper grain
    return np.clip(out, 0, 255).astype(np.uint8)

def local_edit(img):
    """Tampering: paste a patch from one place over another place."""
    out = img.copy()
    h, w = out.shape[:2]
    ph, pw = h // 8, w // 4
    y1, x1 = rng.integers(0, h - ph), rng.integers(0, w - pw)
    y2, x2 = rng.integers(0, h - ph), rng.integers(0, w - pw)
    out[y1:y1 + ph, x1:x1 + pw] = img[y2:y2 + ph, x2:x2 + pw]
    return out