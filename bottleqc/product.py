"""Real product sprites built from a photographed bottle (RGBA cut-out).

From one photo we derive, without drawing anything synthetic on the product:
  * the original capped bottle,
  * an uncapped bottle (crown cap pixels removed, real glass neck kept),
  * wrong-colour caps (hue/saturation shift of the cap pixels only, so the
    real crimp ridges, highlights and texture are preserved).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DIR = ROOT / "assets" / "product"

# Target HSV (OpenCV ranges) for re-coloured caps. None = keep channel.
RECOLOR = {
    "red": (0, 205, None),
    "orange": (13, 210, None),
    "yellow": (26, 200, None),
    "green": (62, 175, None),
    "blue": (108, 190, None),
    "purple": (138, 175, None),
    "white": (None, 12, "bright"),
    "black": (None, 40, "dark"),
}


@dataclass
class ProductSprite:
    name: str
    rgba: np.ndarray        # H x W x 4 (BGRA), tightly cropped
    cap_mask: np.ndarray    # H x W bool
    cap_bottom: int         # first row below the cap
    native_color: str

    def variant(self, cap: str | None) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (BGRA image, cap mask or None) for a cap colour, or cap=None for missing cap."""
        if cap is None:
            img = self.rgba[self.cap_bottom:].copy()
            # soften the cut edge a little so it reads like the glass lip
            img[:2, :, 3] = (img[:2, :, 3] * np.array([[0.55], [0.85]])).astype(np.uint8)
            return img, None
        if cap == self.native_color:
            return self.rgba.copy(), self.cap_mask.copy()
        return _recolor(self.rgba, self.cap_mask, cap), self.cap_mask.copy()


def _recolor(rgba: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    out = rgba.copy()
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    h_t, s_t, v_mode = RECOLOR[color]
    m = mask
    s, v = hsv[..., 1], hsv[..., 2]
    # keep relative saturation texture: scale around the cap's own mean
    s_mean = max(1.0, float(s[m].mean()))
    if h_t is not None:
        hsv[..., 0][m] = h_t
    hsv[..., 1][m] = np.clip(s[m] / s_mean * s_t, 0, 255)
    if v_mode == "bright":
        hsv[..., 2][m] = np.clip(v[m] * 1.08 + 10, 0, 255)
    elif v_mode == "dark":
        hsv[..., 2][m] = np.clip(v[m] * 0.22, 0, 255)
    elif color in ("blue", "purple", "green"):
        hsv[..., 2][m] = np.clip(v[m] * 0.82, 0, 255)  # deeper pigments look darker
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out[..., :3][m] = bgr[m]
    return out


def _detect_cap(rgba: np.ndarray) -> tuple[np.ndarray, int, str]:
    """Find the cap as the coloured band at the top of the bottle silhouette."""
    from .color import classify_color

    alpha = rgba[..., 3] > 128
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2HSV)
    colored = alpha & (hsv[..., 1] > 35)
    rows = np.where(alpha.any(1))[0]
    top = rows[0]
    limit = top + int(0.15 * (rows[-1] - top))  # cap lives in the top 15%
    bottom = top
    for y in range(top, limit):
        frac = colored[y].sum() / max(1, alpha[y].sum())
        if frac > 0.5:
            bottom = y + 1
        elif bottom > top and frac < 0.25:
            break
    if bottom - top < 3:
        raise RuntimeError("Could not locate a coloured cap at the top of the product photo")
    mask = np.zeros_like(alpha)
    mask[top:bottom] = alpha[top:bottom]
    name, _ = classify_color(rgba[..., :3][top:bottom], mask[top:bottom].astype(np.uint8))
    return mask, bottom, name


@lru_cache(maxsize=None)
def load_products() -> tuple[ProductSprite, ...]:
    sprites = []
    for p in sorted(PRODUCT_DIR.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        ys, xs = np.where(img[..., 3] > 8)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        mask, bottom, color = _detect_cap(img)
        sprites.append(ProductSprite(p.stem, img, mask, bottom, color))
    if not sprites:
        raise RuntimeError(f"No RGBA product photos (.png with transparency) found in {PRODUCT_DIR}")
    return tuple(sprites)


def transform(img: np.ndarray, mask: np.ndarray | None, height: int, angle: float = 0.0):
    """Scale to `height` px and rotate slightly. Returns (BGRA, mask|None)."""
    s = height / img.shape[0]
    w = max(4, int(round(img.shape[1] * s)))
    img = cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if mask is not None:
        mask = cv2.resize(mask.astype(np.uint8), (w, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    if abs(angle) > 0.05:
        diag = int(np.hypot(w, height)) + 2
        M = cv2.getRotationMatrix2D((w / 2, height), angle, 1.0)  # pivot at the base
        M[0, 2] += (diag - w) / 2
        M[1, 2] += diag - height
        img = cv2.warpAffine(img, M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        if mask is not None:
            mask = cv2.warpAffine(mask.astype(np.uint8), M, (diag, diag), flags=cv2.INTER_NEAREST).astype(bool)
        ys, xs = np.where(img[..., 3] > 8)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        img = img[y0:y1, x0:x1]
        if mask is not None:
            mask = mask[y0:y1, x0:x1]
    return img, mask


def paste(dst: np.ndarray, sprite: np.ndarray, x: int, y: int, gain: float = 1.0):
    """Alpha-composite BGRA sprite onto BGR dst with top-left at (x, y). Returns visible box or None."""
    H, W = dst.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    sp = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sp[..., 3:4].astype(np.float32) / 255.0
    rgb = sp[..., :3].astype(np.float32) * gain
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(rgb * a + roi * (1 - a), 0, 255).astype(np.uint8)
    return x0, y0, x1, y1


def mask_box(mask: np.ndarray, x: int, y: int, W: int, H: int):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
