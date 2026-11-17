"""Real-time inspection engine.

Pipeline per frame:
  YOLO11 detect (+ optional segmentation masks) -> ByteTrack/BoT-SORT IDs
  -> associate caps to bottles -> HSV cap colour -> per-track voting
  -> verdict + count when the bottle crosses the counting line.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .color import classify_color, find_cap_region
from .scene import ConveyorSimulator
from .storage import Storage

log = logging.getLogger("bottleqc.engine")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STATUS_COLORS = {  # BGR for overlays
    "OK": (94, 197, 34),
    "NO_CAP": (68, 68, 239),
    "WRONG_COLOR": (11, 158, 245),
    "PENDING": (200, 200, 200),
}


@dataclass
class Config:
    model_path: str = ""
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"          # or botsort.yaml
    device: str = "cpu"
    expected_color: str = "pink"              # or "any" to only check presence
    line_pos: float = 0.55                   # fraction of frame width
    direction: str = "ltr"                   # ltr | rtl
    min_presence: float = 0.5                # share of frames where a cap must be seen
    heuristic_cap_threshold: float = 0.35   # fallback (no cap class): min capish share of the strongest neck rows
    loop_video: bool = True
    realtime: bool = True                    # pace file sources to their native FPS
    jpeg_quality: int = 80


# ---------------------------------------------------------------- sources
class Source:
    name = "source"
    fps = 25.0
    is_file = False

    def read(self):  # -> np.ndarray | None
        raise NotImplementedError

    def rewind(self) -> bool:
        return False

    def release(self):
        pass


class SyntheticSource(Source):
    def __init__(self, expected=None, seed=None):
        self.sim = ConveyorSimulator(seed=seed if seed is not None else int(time.time()), expected=expected)
        self.name = "synthetic"

    def read(self):
        return self.sim.step()


class CaptureSource(Source):
    def __init__(self, spec: str | int, is_file: bool):
        self.cap = cv2.VideoCapture(spec)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {spec}")
        if not is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.is_file = is_file
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not (1 <= self.fps <= 240):
            self.fps = 25.0
        self.name = str(spec) if not is_file else Path(str(spec)).name

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def rewind(self):
        if self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True
        return False

    def release(self):
        self.cap.release()


def open_source(kind: str, value: str, expected: str) -> Source:
    if kind == "synthetic":
        return SyntheticSource(expected=None if expected == "any" else expected)
    if kind == "camera":
        return CaptureSource(int(value or 0), is_file=False)
    if kind == "stream":
        return CaptureSource(value, is_file=False)
    if kind == "file":
        p = Path(value)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise RuntimeError(f"File not found: {value}")
        return CaptureSource(str(p), is_file=True)
    raise RuntimeError(f"Unknown source type: {kind}")


# ---------------------------------------------------------------- tracking
@dataclass
class Track:
    tid: int
    first_seen: int
    last_seen: int
    cx: float
    box: tuple = (0, 0, 0, 0)
    det_conf: float = 0.0
    present: list = field(default_factory=list)
    colors: list = field(default_factory=list)
    counted: bool = False
    verdict: str | None = None
    color: str | None = None

    def provisional(self, cfg: Config) -> tuple[str, str | None]:
        if self.verdict:
            return self.verdict, self.color
        if len(self.present) < 2:
            return "PENDING", None
        return decide(self.present, self.colors, cfg)


def decide(present: list, colors: list, cfg: Config) -> tuple[str, str | None]:
    if not present or np.mean(present) < cfg.min_presence:
        return "NO_CAP", None
    color = Counter(colors).most_common(1)[0][0] if colors else "unknown"
    if cfg.expected_color != "any" and color != cfg.expected_color:
        return "WRONG_COLOR", color
    return "OK", color


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, storage: Storage, cfg: Config | None = None):
        self.storage = storage
        self.cfg = cfg or Config()
        if not self.cfg.model_path:
            custom = ROOT / "models" / "bottleqc_yolo11n.pt"
            self.cfg.model_path = str(custom) if custom.exists() else "yolo11n.pt"
        self.model = None
        self.bottle_cls: int | None = None
        self.cap_cls: int | None = None
        self.model_lock = threading.Lock()

        self.source: Source | None = None
        self.source_desc = {"type": "synthetic", "value": ""}
        self.thread: threading.Thread | None = None
        self.stop_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.state = "idle"
        self.error: str | None = None
        self.session_id: int | None = None

        self.frame_lock = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_seq = 0

        self.events: deque = deque(maxlen=500)   # (seq, record)
        self.event_seq = 0
        self._reset_counters()

    # ---------------- model
    def load_model(self, path: str | None = None):
        from ultralytics import YOLO  # imported lazily so the API boots fast
        path = path or self.cfg.model_path
        with self.model_lock:
            model = YOLO(path)
            names = {int(k): str(v).lower() for k, v in model.names.items()}
            bottle = next((k for k, v in names.items() if v == "bottle"), None)
            cap = next((k for k, v in names.items() if "cap" in v or "lid" in v), None)
            if bottle is None:
                raise RuntimeError(f"Model {path} has no 'bottle' class (classes: {list(names.values())[:10]}...)")
            self.model, self.bottle_cls, self.cap_cls = model, bottle, cap
            self.cfg.model_path = path
            self.task = getattr(model, "task", "detect")
        log.info("model %s loaded: bottle=%s cap=%s task=%s", path, bottle, cap, self.task)

    def model_info(self):
        return {
            "path": Path(self.cfg.model_path).name, "task": getattr(self, "task", None),
            "classes": list(self.model.names.values()) if self.model else [],
            "cap_mode": "detector" if self.cap_cls is not None else "heuristic",
        }

    # ---------------- lifecycle
    def _reset_counters(self):
        self.tracks: dict[int, Track] = {}
        self.counts = Counter()
        self.color_counts = Counter()
        self.crossings: deque = deque(maxlen=5000)
        self.timeline: deque = deque(maxlen=720)   # (ts, total, defects)
        self.fps = 0.0
        self.infer_ms = 0.0
        self.frame_idx = 0
        self.started_at = None
        self._last_sample = 0.0
        self._reset_tracker = True

    def start(self, kind: str | None = None, value: str | None = None):
        if self.thread and self.thread.is_alive():
            self.stop()
        if kind:
            self.source_desc = {"type": kind, "value": value or ""}
        self.error = None
        try:
            if self.model is None:
                self.state = "loading"
                self.load_model()
            self.source = open_source(self.source_desc["type"], self.source_desc["value"], self.cfg.expected_color)
        except Exception as e:  # surface to UI
            self.state, self.error = "error", str(e)
            log.exception("start failed")
            raise
        self._reset_counters()
        self.started_at = time.time()
        self.session_id = self.storage.start_session(
            f'{self.source_desc["type"]}:{self.source.name}', Path(self.cfg.model_path).name, self.cfg.expected_color)
        (DATA / "snapshots" / str(self.session_id)).mkdir(parents=True, exist_ok=True)
        self.stop_evt.clear()
        self.pause_evt.clear()
        self.state = "running"
        self.thread = threading.Thread(target=self._run, name="inspection", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.thread = None
        if self.source:
            self.source.release()
            self.source = None
        if self.session_id:
            self.storage.end_session(self.session_id)
        if self.state != "error":
            self.state = "stopped"

    def pause(self, on: bool):
        if on:
            self.pause_evt.set()
            if self.state == "running":
                self.state = "paused"
        else:
            self.pause_evt.clear()
            if self.state == "paused":
                self.state = "running"

    # ---------------- main loop
    def _run(self):
        src = self.source
        period = 1.0 / src.fps if (src.is_file and self.cfg.realtime) else 0.0
        try:
            while not self.stop_evt.is_set():
                if self.pause_evt.is_set():
                    time.sleep(0.05)
                    continue
                t0 = time.perf_counter()
                frame = src.read()
                if frame is None:
                    if self.cfg.loop_video and src.rewind():
                        self._reset_tracker = True
                        self.tracks.clear()
                        continue
                    self.state = "finished"
                    break
                annotated = self.process(frame)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
                if ok:
                    with self.frame_lock:
                        self.jpeg = buf.tobytes()
                        self.frame_seq += 1
                        self.frame_lock.notify_all()
                dt = time.perf_counter() - t0
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, period, 1e-3)) if self.fps else 1.0 / max(dt, 1e-3)
                if period > dt:
                    time.sleep(period - dt)
        except Exception as e:
            log.exception("engine crashed")
            self.state, self.error = "error", str(e)
        finally:
            if self.session_id:
                self.storage.end_session(self.session_id)

    def process(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        H, W = frame.shape[:2]
        line_x = int(cfg.line_pos * W)
        self.frame_idx += 1
        t0 = time.perf_counter()
        with self.model_lock:
            res = self.model.track(frame, persist=not self._reset_tracker, conf=cfg.conf, iou=cfg.iou,
                                   imgsz=cfg.imgsz, tracker=cfg.tracker, device=cfg.device, verbose=False)[0]
        self._reset_tracker = False
        self.infer_ms = 0.8 * self.infer_ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.infer_ms else (time.perf_counter() - t0) * 1000

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)
        confs = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        masks = None
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()  # N x h x w (model resolution)

        cap_idx = [i for i in range(len(cls)) if self.cap_cls is not None and cls[i] == self.cap_cls]
        out = frame.copy()

        for i in range(len(cls)):
            if cls[i] != self.bottle_cls or ids is None:
                continue
            tid = int(ids[i])
            x1, y1, x2, y2 = xyxy[i]
            bw, bh = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid, self.frame_idx, self.frame_idx, cx)
            prev_cx = tr.cx
            tr.cx, tr.last_seen, tr.box, tr.det_conf = cx, self.frame_idx, (x1, y1, x2, y2), float(confs[i])

            # ---- cap analysis (skip partially visible bottles at frame edges)
            fully_visible = x1 > 2 and x2 < W - 2
            cap_box = None
            if fully_visible:
                present, color, ccf = False, None, 0.0
                if self.cap_cls is not None:
                    best = None
                    for j in cap_idx:
                        cx1, cy1, cx2, cy2 = xyxy[j]
                        ccx, ccy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                        if x1 <= ccx <= x2 and y1 - 0.1 * bh <= ccy <= y1 + 0.25 * bh:
                            if best is None or confs[j] > confs[best]:
                                best = j
                    if best is not None:
                        present = True
                        cap_box = xyxy[best].astype(int)
                        color, ccf = self._cap_color(frame, cap_box, masks[best] if masks is not None else None)
                else:
                    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                    top, score = find_cap_region(crop)
                    present = score >= cfg.heuristic_cap_threshold
                    if present:
                        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
                        m = (((hsv[..., 1] > 40) & (hsv[..., 2] > 60)) | (hsv[..., 2] < 45)).astype(np.uint8)
                        color, ccf = classify_color(top, m)
                        cap_box = np.array([x1, y1, x2, y1 + top.shape[0]], int)
                if not tr.counted:
                    tr.present.append(1 if present else 0)
                    if present and color and color != "unknown":
                        tr.colors.append(color)

            # ---- line crossing -> count once per track
            crossed = (prev_cx < line_x <= cx) if cfg.direction == "ltr" else (prev_cx > line_x >= cx)
            if crossed and not tr.counted and self.frame_idx > tr.first_seen:
                self._finalize(tr, frame)

            status, color = tr.provisional(cfg)
            self._draw_track(out, tr, status, color, cap_box)

        # prune stale tracks
        for tid in [t for t, tr in self.tracks.items() if self.frame_idx - tr.last_seen > 90]:
            del self.tracks[tid]

        self._draw_hud(out, line_x)
        now = time.time()
        if now - self._last_sample >= 2.0:
            self._last_sample = now
            self.timeline.append((now, self.counts["total"], self.counts["total"] - self.counts["OK"]))
        return out

    def _cap_color(self, frame, box, mask):
        x1, y1, x2, y2 = box
        # shrink to avoid background bleed
        dx, dy = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        x1, y1, x2, y2 = max(0, x1 + dx), max(0, y1 + dy), x2 - dx, y2 - dy
        crop = frame[y1:y2, x1:x2]
        m = None
        if mask is not None and crop.size:
            full = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (full[y1:y2, x1:x2] > 0.5).astype(np.uint8)
        return classify_color(crop, m)

    def _finalize(self, tr: Track, frame):
        cfg = self.cfg
        status, color = decide(tr.present, tr.colors, cfg)
        tr.counted, tr.verdict, tr.color = True, status, color
        color_conf = (Counter(tr.colors)[color] / len(tr.colors)) if (color and tr.colors) else 0.0
        self.counts["total"] += 1
        self.counts[status] += 1
        self.color_counts[color or "none"] += 1
        now = time.time()
        self.crossings.append(now)

        snap = None
        x1, y1, x2, y2 = map(int, tr.box)
        pad = 12
        crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if crop.size:
            snap = f"{self.session_id}/{self.counts['total']:06d}_{tr.tid}.jpg"
            cv2.imwrite(str(DATA / "snapshots" / snap), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rec = {
            "session_id": self.session_id, "ts": now, "track_id": tr.tid, "status": status,
            "cap_present": int(status != "NO_CAP"), "cap_color": color, "color_conf": round(color_conf, 3),
            "det_conf": round(tr.det_conf, 3), "observations": len(tr.present), "snapshot": snap,
        }
        rec["id"] = self.storage.add_inspection(rec)
        self.event_seq += 1
        self.events.append((self.event_seq, rec))

    # ---------------- drawing
    def _draw_track(self, img, tr: Track, status, color, cap_box):
        c = STATUS_COLORS[status]
        x1, y1, x2, y2 = map(int, tr.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        L = 14
        for (px, py, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + sx * L, py), c, 4)
            cv2.line(img, (px, py), (px, py + sy * L), c, 4)
        if cap_box is not None:
            cv2.rectangle(img, tuple(map(int, cap_box[:2])), tuple(map(int, cap_box[2:])), (255, 255, 255), 1, cv2.LINE_AA)
        label = f"#{tr.tid} {status.replace('_', ' ')}" + (f" · {color}" if color and status != 'NO_CAP' else "")
        label = label.replace("·", "-")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 8, y1 - 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), c, -1)
        cv2.putText(img, label, (x1 + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)

    def _draw_hud(self, img, line_x):
        H = img.shape[0]
        ov = img.copy()
        cv2.line(ov, (line_x, 0), (line_x, H), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)
        arrow = "->" if self.cfg.direction == "ltr" else "<-"
        cv2.putText(img, f"COUNT LINE {arrow}", (line_x + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        txt = f"TOTAL {self.counts['total']}  OK {self.counts['OK']}  DEFECT {self.counts['total'] - self.counts['OK']}  |  {self.fps:4.1f} FPS"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (8, H - th - 18), (16 + tw, H - 8), (20, 20, 20), -1)
        cv2.putText(img, txt, (12, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    # ---------------- API views
    def stats(self) -> dict:
        now = time.time()
        total = self.counts["total"]
        ok = self.counts["OK"]
        last_min = sum(1 for t in self.crossings if now - t <= 60)
        elapsed = (now - self.started_at) if self.started_at else 0
        return {
            "state": self.state, "error": self.error, "session_id": self.session_id,
            "source": self.source_desc, "elapsed": round(elapsed, 1),
            "total": total, "ok": ok, "defects": total - ok,
            "no_cap": self.counts["NO_CAP"], "wrong_color": self.counts["WRONG_COLOR"],
            "yield": round(100 * ok / total, 2) if total else None,
            "throughput_min": last_min,
            "avg_throughput_min": round(total / elapsed * 60, 1) if elapsed > 5 else None,
            "fps": round(self.fps, 1), "infer_ms": round(self.infer_ms, 1),
            "active_tracks": sum(1 for t in self.tracks.values() if self.frame_idx - t.last_seen < 3),
            "by_color": dict(self.color_counts),
            "timeline": list(self.timeline)[-180:],
            "model": self.model_info() if self.model else None,
            "config": asdict(self.cfg),
        }

    def events_since(self, seq: int):
        return [(s, r) for s, r in list(self.events) if s > seq]
