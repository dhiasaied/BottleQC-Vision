"""Evaluate counting + cap QC against simulator ground truth.

    python tools/evaluate.py --frames 900                       # uses models/bottleqc_yolo11n.pt if present
    python tools/evaluate.py --model yolo11n.pt --frames 600    # COCO fallback (heuristic caps)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bottleqc.engine import Config, Engine  # noqa: E402
from bottleqc.scene import ConveyorSimulator  # noqa: E402
from bottleqc.storage import Storage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    eng = Engine(Storage(tmp / "eval.db"), Config(model_path=a.model))
    eng.load_model()
    eng.session_id = eng.storage.start_session("eval", Path(eng.cfg.model_path).name, eng.cfg.expected_color)
    (ROOT / "data" / "snapshots" / str(eng.session_id)).mkdir(parents=True, exist_ok=True)
    sim = ConveyorSimulator(seed=a.seed, missing_rate=0.2, wrong_rate=0.2, expected=eng.cfg.expected_color)
    print("model:", eng.model_info()["path"], "| caps via", eng.model_info()["cap_mode"])

    def truth(b):
        if b.cap is None:
            return "NO_CAP"
        return "OK" if b.cap == eng.cfg.expected_color else "WRONG_COLOR"

    crossed_gt, seen_events, rows = set(), 0, []
    line_x = eng.cfg.line_pos * sim.w
    t0 = time.time()
    for _ in range(a.frames):
        frame = sim.step()
        for b in sim.bottles:
            if b.x >= line_x and b.uid not in crossed_gt:
                crossed_gt.add(b.uid)
        eng.process(frame)
        for _, rec in list(eng.events)[seen_events:]:
            gt = min(sim.bottles, key=lambda b: abs(b.x - line_x))
            rows.append((gt.uid, truth(gt), rec["status"], gt.cap, rec["cap_color"]))
        seen_events = len(eng.events)
    dt = time.time() - t0

    correct = sum(1 for r in rows if r[1] == r[2])
    print(f"frames={a.frames}  {a.frames / dt:.1f} FPS")
    print(f"ground-truth bottles crossed: {len(crossed_gt)}   counted: {len(rows)}   "
          f"duplicates: {len(rows) - len({r[0] for r in rows})}")
    print(f"verdict accuracy: {correct}/{len(rows)} = {100 * correct / max(1, len(rows)):.1f}%")
    for r in rows:
        flag = "" if r[1] == r[2] else "   <-- mismatch"
        print(f"  bottle {r[0]:3d}: truth={r[1]:<12} pred={r[2]:<12} (cap {r[3]} -> {r[4]}){flag}")


if __name__ == "__main__":
    main()
