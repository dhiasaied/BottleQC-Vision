"""Bottling-line scene compositor using the REAL product photo.

The product pixels always come from the photographed bottle in
assets/product/*.png (see bottleqc.product). Only the environment
(backlight panel, stainless conveyor) is generated, with photographic
noise/blur, or taken from real photos placed in assets/backgrounds/.

Used for:
  * a labelled YOLO dataset (classes: bottle, cap),
  * a demo conveyor video,
  * a live "simulated camera" source so the app runs with no hardware.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .product import ROOT, load_products, mask_box, paste, transform

BG_DIR = ROOT / "assets" / "backgrounds"
CAP_CHOICES = ["pink", "red", "orange", "yellow", "green", "blue", "purple", "white", "black"]


# ------------------------------------------------------------------ environment
def _brushed(w: int, h: int, rng: np.random.Generator, lo=110, hi=185) -> np.ndarray:
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    n = cv2.blur(n, (61, 1))                       # brushed streaks
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    shade = np.linspace(1.0, 0.72, h, dtype=np.float32)[:, None]
    g = (lo + (hi - lo) * (0.35 + 0.65 * n)) * shade
    return np.repeat(g[..., None], 3, 2)


@lru_cache(maxsize=8)
def _real_backgrounds():
    out = []
    if BG_DIR.exists():
        for p in sorted(BG_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                im = cv2.imread(str(p))
                if im is not None:
                    out.append(im)
    return tuple(out)


def render_environment(w: int, h: int, belt_y: int, offset: float, seed: int, tint=None) -> np.ndarray:
    """Backlit inspection panel + stainless conveyor. Deterministic per `seed` except belt motion."""
    rng = np.random.default_rng(seed)
    real = _real_backgrounds()
    if real and rng.random() < 0.6:
        bg = real[int(rng.integers(len(real)))]
        img = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        # backlight diffuser panel: bright, slightly uneven, soft vignette
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.2, 0.5)
        r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        base = rng.uniform(150, 235)
        panel = base * (1.0 - rng.uniform(0.15, 0.35) * r)
        panel += cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 25)
        tint = tint if tint is not None else rng.uniform(0.94, 1.04, 3)
        img = panel[..., None] * np.asarray(tint, np.float32)[None, None]
        # panel frame seams
        for x in range(int(rng.uniform(0, 300)), w, int(rng.uniform(280, 420))):
            img[:belt_y, max(0, x - 2):x + 2] *= 0.82
    # back guide rail (stainless)
    rail_y = belt_y - int(h * rng.uniform(0.18, 0.30))
    rail_h = max(6, int(h * 0.022))
    rail = _brushed(w, rail_h, rng, 120, 215)
    img[rail_y:rail_y + rail_h] = rail
    img[rail_y + rail_h:rail_y + rail_h + 3] *= 0.75
    # conveyor: modular plastic/steel slats moving with `offset`
    belt = _brushed(w, h - belt_y, rng, 70, 140)
    slat = max(18, int(h * 0.05))
    off = int(offset) % slat
    for x in range(-slat + off, w + slat, slat):
        belt[:, max(0, x):max(0, x + 2)] *= 0.62
        belt[:, max(0, x + 2):max(0, x + 4)] *= 1.12
    img[belt_y:] = belt
    img[belt_y:belt_y + 3] = img[belt_y:belt_y + 3] * 1.25
    # front rail
    fy = belt_y + int((h - belt_y) * 0.55)
    img[fy:fy + rail_h] = _brushed(w, rail_h, rng, 130, 220)
    return np.clip(img, 0, 255).astype(np.uint8)


def contact_shadow(img: np.ndarray, cx: int, base_y: int, width: int, strength=0.45):
    """Soft elliptical shadow under the bottle base (computed on a local ROI only)."""
    h, w = img.shape[:2]
    rx, ry = int(width * 0.62), max(3, int(width * 0.10))
    sigma = max(2.0, width * 0.08)
    pad = int(3 * sigma) + 2
    x0, x1 = max(0, cx - rx - pad), min(w, cx + rx + pad)
    y0, y1 = max(0, base_y - ry - pad), min(h, base_y + ry + pad)
    if x1 <= x0 or y1 <= y0:
        return
    sh = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.ellipse(sh, (cx - x0, base_y - y0), (rx, ry), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), sigma)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - strength * sh[..., None])).astype(np.uint8)


# ------------------------------------------------------------------ bottles
@dataclass
class Bottle:
    x: float                 # centre x (px)
    height: int              # rendered height of the full capped bottle (px)
    cap: str | None          # colour name, or None = missing cap
    angle: float = 0.0
    gain: float = 1.0
    uid: int = 0
    sprite_idx: int = 0
    _cache: tuple | None = None

    def sprite(self):
        if self._cache is None:
            prod = load_products()[self.sprite_idx % len(load_products())]
            img, mask = prod.variant(self.cap)
            # keep pixel scale identical whether or not the cap exists
            scale = self.height / prod.rgba.shape[0]
            target_h = max(8, int(round(img.shape[0] * scale)))
            self._cache = transform(img, mask, target_h, self.angle)
        return self._cache


def draw_bottle(img: np.ndarray, b: Bottle, base_y: int):
    """Composite a real-product bottle standing on base_y. Returns (bottle_box, cap_box|None)."""
    H, W = img.shape[:2]
    sp, mask = b.sprite()
    h, w = sp.shape[:2]
    x, y = int(round(b.x - w / 2)), base_y - h
    contact_shadow(img, int(b.x), base_y, w)
    box = paste(img, sp, x, y, b.gain)
    if box is None:
        return None, None
    ys, xs = np.where(sp[..., 3] > 60)
    bb = (max(0, x + xs.min()), max(0, y + ys.min()), min(W, x + xs.max() + 1), min(H, y + ys.max() + 1))
    cb = mask_box(mask, x, y, W, H) if mask is not None else None
    return bb, cb


def _pick_cap(rng: random.Random, native: str, missing_rate: float, wrong_rate: float):
    r = rng.random()
    if r < missing_rate:
        return None
    if r < missing_rate + wrong_rate:
        return rng.choice([c for c in CAP_CHOICES if c != native])
    return native


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    out = out * rng.uniform(0.75, 1.15) + rng.uniform(-18, 14)
    out += np.random.normal(0, rng.uniform(1.5, 6), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.3:  # conveyor motion blur
        k = rng.choice([3, 5, 7])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)
    if rng.random() < 0.5:  # camera JPEG artefacts
        q = rng.randint(55, 90)
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return out


def render_random_scene(w: int, h: int, rng: random.Random):
    """One training image + YOLO labels [(cls, cx, cy, bw, bh) normalised]."""
    products = load_products()
    belt_y = int(h * rng.uniform(0.80, 0.92))
    img = render_environment(w, h, belt_y, rng.uniform(0, 100), seed=rng.randrange(1 << 30))
    labels = []
    bh = int(h * rng.uniform(0.45, 0.85))
    x = rng.uniform(-0.2, 0.3) * bh * 0.35
    while x < w + bh * 0.2:
        idx = rng.randrange(len(products))
        b = Bottle(x=x, height=int(bh * rng.uniform(0.97, 1.03)),
                   cap=_pick_cap(rng, products[idx].native_color, 0.25, 0.35),
                   angle=rng.uniform(-2.5, 2.5), gain=rng.uniform(0.85, 1.08), sprite_idx=idx)
        width = b.sprite()[0].shape[1]
        b.x = x + width / 2
        bb, cb = draw_bottle(img, b, belt_y + int(h * 0.02))
        x += width + rng.uniform(0.02, 0.9) * width
        if bb is None:
            continue
        full_w = width
        for cls, box in ((0, bb), (1, cb)):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if cls == 0 and (x2 - x1) < 0.4 * full_w:
                continue  # mostly out of frame
            if cls == 1 and (x2 - x1) < 4:
                continue
            labels.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return augment(img, rng), labels


class ConveyorSimulator:
    """Infinite conveyor of the real product: bottles enter on the left and travel right."""

    def __init__(self, w=960, h=540, speed=6.0, seed=7, missing_rate=0.08, wrong_rate=0.10, expected=None):
        self.w, self.h = w, h
        self.speed = speed
        self.rng = random.Random(seed)
        self.products = load_products()
        self.expected = expected or self.products[0].native_color
        self.belt_y = int(h * 0.88)
        self.env_seed = seed
        self.offset = 0.0
        self.bottles: list[Bottle] = []
        self.missing_rate = missing_rate
        self.wrong_rate = wrong_rate
        self.next_gap = 0.0
        self.uid = 0
        self.bottle_h = int(h * 0.80)
        self._env_cache: dict[int, np.ndarray] = {}
        self._noise = [np.random.normal(0, 2.5, (h, w, 3)).astype(np.int16) for _ in range(4)]

    def _spawn(self):
        r = self.rng.random()
        if r < self.missing_rate:
            cap = None
        elif r < self.missing_rate + self.wrong_rate:
            cap = self.rng.choice([c for c in CAP_CHOICES if c != self.expected])
        else:
            cap = self.expected
        self.uid += 1
        b = Bottle(x=0, height=self.bottle_h, cap=cap, angle=self.rng.uniform(-1.2, 1.2),
                   gain=self.rng.uniform(0.95, 1.03), uid=self.uid, sprite_idx=self.uid % len(self.products))
        bw = b.sprite()[0].shape[1]
        b.x = -bw / 2 - 2
        self.bottles.append(b)
        self.next_gap = bw * self.rng.uniform(1.6, 2.6)

    def step(self) -> np.ndarray:
        self.offset += self.speed
        for b in self.bottles:
            b.x += self.speed
        self.bottles = [b for b in self.bottles if b.x < self.w + 200]
        if not self.bottles or min(b.x for b in self.bottles) >= self.next_gap - 60:
            self._spawn()
        slat = max(18, int(self.h * 0.05))
        phase = int(self.offset) % slat
        env = self._env_cache.get(phase)
        if env is None:
            env = self._env_cache[phase] = render_environment(self.w, self.h, self.belt_y, phase, seed=self.env_seed,
                                                              tint=(1.0, 1.0, 1.0))
        img = env.copy()
        for b in sorted(self.bottles, key=lambda b: b.x):
            draw_bottle(img, b, self.belt_y + int(self.h * 0.02))
        noise = self._noise[int(self.offset) % len(self._noise)]
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
