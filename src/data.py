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