"""Generate a synthetic YOLO dataset of bottles and caps.

    python tools/make_dataset.py --train 800 --val 160
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bottleqc.scene import render_random_scene  # noqa: E402

CLASSES = ["bottle", "cap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "datasets" / "bottles"))
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=160)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    rng = random.Random(a.seed)
    for split, n in (("train", a.train), ("val", a.val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = render_random_scene(a.w, a.h, rng)
            name = f"{split}_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out / "labels" / split / f"{name}.txt", "w") as f:
                for cls, cx, cy, bw, bh in labels:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"{split}: {n} images")

    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print("wrote", yaml)


if __name__ == "__main__":
    main()
