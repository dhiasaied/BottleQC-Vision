"""Render a demo conveyor video (bottles with correct, wrong-colour and missing caps).

    python tools/make_demo_video.py --seconds 60 --out samples/demo_line.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bottleqc.scene import ConveyorSimulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "samples" / "demo_line.mp4"))
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--expected", default=None)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sim = ConveyorSimulator(seed=a.seed, expected=a.expected, missing_rate=0.1, wrong_rate=0.12)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (sim.w, sim.h))
    for _ in range(a.seconds * a.fps):
        vw.write(sim.step())
    vw.release()
    print("wrote", out, "bottles spawned:", sim.uid)


if __name__ == "__main__":
    main()
