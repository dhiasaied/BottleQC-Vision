"""Cap colour classification in HSV space."""
from __future__ import annotations

import cv2
import numpy as np

COLOR_NAMES = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black"]


def classify_color(bgr_crop: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float]:
    """Return (colour name, confidence 0..1) for the dominant colour of a crop.

    `mask` (uint8, same HxW) restricts analysis to segmented cap pixels.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    if mask is not None and mask.size == bgr_crop.shape[0] * bgr_crop.shape[1]:
        sel = mask.reshape(-1) > 0
        if sel.sum() > 10:
            hsv = hsv[sel]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    labels = np.full(len(h), -1)
    black = v < 60
    white = (~black) & (s < 50) & (v > 150)
    chroma = (~black) & (~white) & (s >= 50)
    labels[black] = COLOR_NAMES.index("black")
    labels[white] = COLOR_NAMES.index("white")
    hc, sc, vc = h[chroma], s[chroma], v[chroma]
    # pink = magenta/red hue with pastel saturation and high brightness
    pink = (((hc >= 140) | (hc < 8)) & (sc < 140) & (vc > 130))
    bins = np.select(
        [pink, (hc < 8) | (hc >= 165), hc < 19, hc < 35, hc < 85, hc < 132, hc < 165],
        [COLOR_NAMES.index("pink"), 0, 1, 2, 3, 4, 5], default=0)
    labels[chroma] = bins
    valid = labels >= 0
    if valid.sum() == 0:
        return "unknown", 0.0
    counts = np.bincount(labels[valid], minlength=len(COLOR_NAMES))
    idx = int(counts.argmax())
    return COLOR_NAMES[idx], float(counts[idx] / valid.sum())


def find_cap_region(bgr_bottle: np.ndarray) -> tuple[np.ndarray, float]:
    """Heuristic for models without a 'cap' class (e.g. COCO yolo11n).

    Looks at the top of the bottle box and measures how much strongly
    coloured / very dark / very bright material sits on the neck.
    Returns (top crop, presence score 0..1).
    """
    h, w = bgr_bottle.shape[:2]
    # neck column only (central 40%), top 12% of the bottle
    top = bgr_bottle[: max(4, int(h * 0.12)), int(w * 0.30): max(int(w * 0.30) + 2, int(w * 0.70))]
    if top.size == 0:
        return top, 0.0
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    # coloured or very dark material; clear glass (bright, unsaturated) is not a cap
    capish = ((s > 40) & (v > 60)) | (v < 45)
    rows = np.sort(capish.mean(1))[::-1]
    k = max(2, len(rows) // 12)          # strongest rows ~ thickness of a crown cap
    return top, float(rows[:k].mean())
