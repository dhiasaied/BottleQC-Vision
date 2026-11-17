"""Fine-tune YOLO11 on the bottle/cap dataset and copy the best weights to models/.

    python tools/train.py --epochs 25 --imgsz 512
    python tools/train.py --model yolo11n-seg.pt   # segmentation variant (needs polygon labels)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "bottles" / "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'cpu', '0' for first GPU; auto by default")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        workers=a.workers, project=str(ROOT / "runs"), name="bottleqc", exist_ok=True,
        patience=8, plots=True, fliplr=0.0, mosaic=0.7, hsv_h=0.0, hsv_s=0.3, cache="ram",  # no mirroring (real label), cap colour preserved
        verbose=True,
    )
    best = Path(model.trainer.best)
    dest = ROOT / "models" / "bottleqc_yolo11n.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print("saved", dest)
    metrics = YOLO(str(dest)).val(data=a.data, imgsz=a.imgsz, device=a.device, verbose=False)
    print(f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
