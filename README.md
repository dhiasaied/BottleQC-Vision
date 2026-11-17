<div align="center">

# BottleQC Vision

### AI-Powered Industrial Quality Control & Real-Time Bottle Inspection System

**Detect · Track · Count · Inspect · Report** — on a live bottling line, in the browser.

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11n-111F68?logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.163-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-storage-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack%20%7C%20BoT--SORT-6f42c1)](#bottle-tracking--counting-logic)
[![Reports](https://img.shields.io/badge/Reports-CSV%20%7C%20Excel%20%7C%20PDF-2ea44f)](#reports--exports)

</div>

---

BottleQC Vision watches bottles move along a conveyor and inspects every one of them in real time.
A **YOLO11** detector finds each bottle and its cap. **ByteTrack** or **BoT-SORT** keeps a stable
ID on every bottle. Each bottle is **counted exactly once** when it crosses a virtual counting
line, and at that moment it gets a verdict: **OK**, **Missing Cap**, or **Wrong Colour**.

Everything streams live to a web dashboard: annotated video, production KPIs, charts, a log
with a snapshot of every bottle, and one-click **CSV / Excel / PDF** reports.

The project is built around a real product: the **Safia 90 cl glass mineral-water bottle** with its
**pink crown cap**. Training images, the demo video and the built-in line simulator are all
composited from the real product photograph, not from drawings or 3D models.

> It runs with **no camera and no labelled dataset**: the built-in simulator streams an endless
> production line of real Safia bottles with randomly injected defects.

---

## Table of Contents

- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [YOLO11n Training](#yolo11n-training)
- [Video Processing Pipeline](#video-processing-pipeline)
- [Bottle Tracking & Counting Logic](#bottle-tracking--counting-logic)
- [Cap Colour Detection (Pink / Safia)](#cap-colour-detection-pink--safia)
- [Quality Classification](#quality-classification)
- [Dashboard Overview](#dashboard-overview)
- [Reports & Exports](#reports--exports)
- [API Documentation](#api-documentation)
- [Evaluation & Test Results](#evaluation--test-results)
- [Performance Considerations](#performance-considerations)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Key Features

| Feature | Details |
|---|---|
| **Object detection** | Ultralytics YOLO11 with two classes, `bottle` and `cap`, or any COCO model via the `bottle` class |
| **Multi-object tracking** | ByteTrack (default) or BoT-SORT, switchable live from the UI |
| **Exactly-once counting** | A track is counted when its centre crosses a vertical line; left→right or right→left |
| **Cap presence check** | A cap must be seen in at least a configurable share of a bottle's tracked frames (default 50%) |
| **Cap colour check** | HSV classifier with 9 colours, majority-voted over the bottle's whole track; **pink** is expected for Safia |
| **Segmentation-ready** | If the model returns masks, only the masked cap pixels are used for colour |
| **Fallback mode** | With no custom model, stock `yolo11n.pt` detects bottles and a neck-region colour heuristic finds caps |
| **Four input sources** | Built-in simulated line, video file (upload in the UI), USB/webcam, RTSP/HTTP stream |
| **Live dashboard** | MJPEG annotated stream, KPIs, yield ring, timeline, result and colour charts, filterable log with snapshots |
| **Defect alerts** | Toast notifications with a thumbnail for every defective bottle |
| **Persistence** | SQLite stores production sessions and per-bottle inspections; a JPEG snapshot is kept for each bottle |
| **Reporting** | CSV, Excel (charts plus colour-coded log) and PDF (KPIs, pie chart, defect gallery, full log) |
| **Hot reconfiguration** | Thresholds, counting line, direction, tracker, expected colour and model change without restarting |
| **Real-product data** | Dataset, demo video and simulator are composited from the photographed Safia bottle |

---

## Technologies Used

| Layer | Technology | Role |
|---|---|---|
| Detection | **Ultralytics YOLO11n** (`ultralytics` 8.4) | Bottle and cap detection; training and validation |
| Tracking | **ByteTrack / BoT-SORT** (bundled with Ultralytics, `lap`) | Persistent IDs across frames |
| Deep learning | **PyTorch** (CPU build in the reference environment) | Inference and training backend |
| Vision | **OpenCV** + **NumPy** | Capture, HSV analysis, compositing, overlays, JPEG encoding |
| Backend | **FastAPI** + **Uvicorn** | REST API, WebSocket, MJPEG streaming, static hosting |
| Uploads | **python-multipart** | Video file uploads |
| Storage | **SQLite** (stdlib `sqlite3`) | Sessions and inspections |
| Reports | **openpyxl** · **ReportLab** · `csv` | Excel, PDF and CSV exports |
| Frontend | Vanilla **HTML/CSS/JS** + **Chart.js 4.4.1** (CDN) | Dashboard UI and charts |
| Fonts | Inter + JetBrains Mono (Google Fonts) | UI typography, with system font fallbacks |

The reference environment is Windows 11, Python 3.13.5, ultralytics 8.4.215, FastAPI 0.163.1,
OpenCV 5.1.0, NumPy 2.5.3, openpyxl 3.1.5, ReportLab 5.0.1, and PyTorch 2.15.0+cpu.

---

## Project Architecture

```mermaid
flowchart LR
    subgraph Sources
        S1[Simulated line<br/>real Safia photo]
        S2[Video file]
        S3[USB / webcam]
        S4[RTSP / HTTP]
    end

    subgraph Engine["Inspection engine (background thread)"]
        D[YOLO11 detect<br/>bottle + cap] --> T[ByteTrack / BoT-SORT<br/>track IDs]
        T --> A[Cap ↔ bottle<br/>association]
        A --> C[HSV cap colour<br/>classifier]
        C --> V[Per-track voting<br/>presence + colour]
        V --> L{Counting line<br/>crossed?}
        L -- yes --> R[Verdict<br/>OK · NO_CAP · WRONG_COLOR]
    end

    Sources --> D
    R --> DB[(SQLite<br/>sessions + inspections)]
    R --> SN[(JPEG snapshots)]
    Engine --> MJ[MJPEG stream]
    Engine --> WS[WebSocket ticks<br/>every 0.5 s]

    subgraph API["FastAPI server"]
        MJ
        WS
        REST[REST API]
        EXP[CSV · XLSX · PDF export]
    end

    DB --> REST
    DB --> EXP
    API --> UI[Web dashboard]
```

**Threading model.** `run.py` starts Uvicorn. When the app starts, it loads the model. When you
press **Start**, the engine opens the source and runs the inspection loop on its own daemon
thread. The loop publishes the latest annotated JPEG, live counters and an event queue. The async
FastAPI handlers read those and stream them to the browser, so the UI never blocks inference.

---

## Installation

### Prerequisites

- Python 3.10+ (developed and tested on **3.13.5**)
- Disk space for the dependencies (PyTorch is the largest), plus about 75 MB for the generated
  dataset (600 + 120 images) and demo video
- Optional: an NVIDIA GPU with a CUDA build of PyTorch, for faster training. See
  [Performance](#performance-considerations).

### Steps

```bash
git clone <your-repo-url> bottle-qc-vision
cd bottle-qc-vision
```

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
ultralytics>=8.3.0
opencv-python>=4.9
numpy>=1.26
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
lap>=0.5.12
openpyxl>=3.1
reportlab>=4.0
```

On first use, Ultralytics downloads the pretrained `yolo11n.pt` weights automatically.

### Quick start

```bash
# 1. Build the labelled dataset (bottle + cap) and a demo video from the real product photo
python tools/make_dataset.py --train 600 --val 120
python tools/make_demo_video.py --seconds 40

# 2. (optional) Fine-tune YOLO11n  →  models/bottleqc_yolo11n.pt
python tools/train.py --epochs 8 --imgsz 640

# 3. Launch the app
python run.py
```

Then open **http://127.0.0.1:8000**.

> Step 2 is optional. If `models/bottleqc_yolo11n.pt` doesn't exist, the app starts on the stock
> COCO `yolo11n.pt` and uses the heuristic cap check.

---

## Configuration

### Server options (`run.py`)

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to expose it on the LAN. |
| `--port` | `8000` | HTTP port |
| `--autostart` | — | Start inspecting as soon as the model loads: `synthetic`, `file`, `camera` or `stream` |
| `--source` | `""` | Value for `--autostart`: a file path, camera index or stream URL |

```bash
python run.py --host 0.0.0.0 --port 8080 --autostart synthetic
python run.py --autostart file --source samples/demo_line.mp4
```

### Inspection settings (`bottleqc/engine.py → Config`)

All of these settings except `model_path`, `device` and `jpeg_quality` can be changed live from
**Settings** in the dashboard or through `POST /api/config`.

| Setting | Default | Range / values | Meaning |
|---|---|---|---|
| `model_path` | auto | `.pt` file | `models/bottleqc_yolo11n.pt` if it exists, otherwise `yolo11n.pt`. Hot-swap with `POST /api/model`. |
| `conf` | `0.35` | 0.05 – 0.95 | Detection confidence threshold |
| `iou` | `0.5` | 0.1 – 0.95 | NMS IoU threshold |
| `imgsz` | `640` | 160 – 1280 (UI: 320–800) | Inference resolution |
| `tracker` | `bytetrack.yaml` | `bytetrack.yaml`, `botsort.yaml` | Tracking algorithm |
| `device` | `cpu` | — | Inference device; edit in code |
| `expected_color` | `pink` | `any` or one of the 9 colours | Required cap colour. `any` checks presence only. |
| `line_pos` | `0.55` | 0.05 – 0.95 | Counting line position as a fraction of frame width |
| `direction` | `ltr` | `ltr`, `rtl` | Direction of product flow |
| `min_presence` | `0.5` | 0.05 – 1.0 | Share of tracked frames in which a cap must be seen |
| `heuristic_cap_threshold` | `0.35` | — | Fallback mode only: minimum cap-like share of the strongest neck rows |
| `loop_video` | `true` | bool | Restart video files at the end |
| `realtime` | `true` | bool | Pace video files to their native FPS |
| `jpeg_quality` | `80` | — | Quality of the MJPEG stream frames |

### Product & background assets

| Path | Purpose |
|---|---|
| `assets/product/*.png` | **RGBA** cut-out photos of the product. Each one is sampled. The cap colour is detected automatically from the photo. |
| `assets/backgrounds/*` | *(optional)* Photos of your real empty line (`.jpg/.jpeg/.png/.webp`). When present they replace the generated backlight panel 60% of the time. |

---

## Usage Guide

1. **Start the server**: `python run.py`, then open `http://127.0.0.1:8000`.
2. **Choose a source** in the control bar:
   - **Simulated production line**: an endless conveyor of real Safia bottles. 8% have a missing
     cap and 10% a wrong-colour cap.
   - **Video file**: pick a file from `samples/` or upload one with **Upload**
     (`.mp4 .avi .mov .mkv .webm .m4v`).
   - **USB / webcam**: the camera index (`0` is the default camera).
   - **RTSP / HTTP stream**: for example `rtsp://user:pass@host:554/stream`.
3. **Check the expected cap colour.** The chip shows **Pink** by default. Click it to open
   Settings.
4. Press **Start**. Bottles get coloured boxes as they are tracked:
   green = OK, red = Missing cap, amber = Wrong colour, grey = Pending (not enough observations yet).
   The yellow line is the counting line.
5. **Pause / Resume** freezes processing. **Stop** closes the session.
6. Open **Reports** to export the current session, any past session, or all sessions as CSV,
   Excel or PDF.

Every Start opens a new **production session** in the database. Counters, the log and the
timeline all belong to that session.

---

## YOLO11n Training

### Dataset: real product, synthetic scenes

`tools/make_dataset.py` composites the **real Safia photo** (`assets/product/safia.png`) into
conveyor scenes and writes pixel-accurate YOLO labels for two classes:

| id | class | |
|---|---|---|
| 0 | `bottle` | Full bottle silhouette, including the cap when present |
| 1 | `cap` | Crown cap only |

The product is never redrawn. The defects are created from the product's own pixels:

- **Missing cap**: the crown-cap rows are cut from the photo, leaving the real glass neck.
- **Wrong colour**: only the cap pixels are hue/saturation-shifted to red, orange, yellow, green,
  blue, purple, white or black. The real crimp ridges and highlights are preserved.

Scenes vary bottle size (45–85% of frame height), spacing, tilt (±2.5°) and exposure. Each scene
also gets a contact shadow, a generated backlit panel and stainless conveyor (or your real
backgrounds), sensor noise, blur, conveyor motion blur and JPEG artefacts. In training scenes,
25% of bottles have no cap and 35% have a wrong-colour cap.

```bash
python tools/make_dataset.py --train 600 --val 120   # defaults: 800 / 160, 960×540, seed 42
```

Output: `datasets/bottles/{images,labels}/{train,val}` and `datasets/bottles/data.yaml`.

### Training

```bash
python tools/train.py --epochs 8 --imgsz 640            # CPU (auto device)
python tools/train.py --epochs 8 --imgsz 640 --device 0 # first CUDA GPU
```

| Argument | Default | |
|---|---|---|
| `--data` | `datasets/bottles/data.yaml` | Dataset YAML |
| `--model` | `yolo11n.pt` | Base weights, e.g. `yolo11n-seg.pt` for segmentation (needs polygon labels) |
| `--epochs` | `25` | |
| `--imgsz` | `512` | **Use 640.** The crown cap is thin, and 640 keeps it detectable. |
| `--batch` | `16` | |
| `--device` | auto | `cpu`, `0`, … |
| `--workers` | `4` | |

Training settings fixed in the script:

- `patience=8`
- `fliplr=0.0`: the real label must not be mirrored
- `mosaic=0.7`
- `hsv_h=0.0`, `hsv_s=0.3`: cap colour must be preserved
- `cache="ram"`

Runs are written to `runs/bottleqc/`. When training finishes, the best weights are copied to
**`models/bottleqc_yolo11n.pt`**, which the app loads automatically on its next start, and a
validation pass prints mAP50 and mAP50-95.

> If a run is interrupted, the checkpoints remain in `runs/bottleqc/weights/` (`best.pt`,
> `last.pt`), but nothing is copied to `models/`. You can test a checkpoint with
> `tools/evaluate.py --model runs/bottleqc/weights/best.pt`. See
> [Evaluation](#evaluation--test-results) before promoting it.

### Training on your own line

1. Collect frames from your camera and label them with the same two classes, `bottle` and `cap`,
   in YOLO format.
2. Run `python tools/train.py --data path/to/data.yaml --epochs 100 --imgsz 640 --device 0`.
3. Select the weights in **Settings → Model**. Any `.pt` file in `models/` is listed.

---

## Video Processing Pipeline

Every frame goes through this sequence in `Engine.process()`:

```text
frame ─► model.track(conf, iou, imgsz, tracker, persist=True)
      ─► split detections: bottles (with track IDs) · caps
      ─► for each tracked bottle:
            ├─ update Track (centre x, box, confidence, last seen)
            ├─ if fully visible (not touching the left/right frame edge):
            │     find its cap → classify colour → append observation
            ├─ if centre crossed the counting line this frame → finalize verdict + count
            └─ draw box, corner brackets, label "#ID STATUS - colour"
      ─► prune tracks unseen for > 90 frames
      ─► draw counting line + HUD (TOTAL / OK / DEFECT / FPS)
      ─► every 2 s: append (time, total, defects) to the timeline
```

- **Pacing.** For video files with `realtime` on, the loop sleeps so playback never runs faster
  than the native FPS. If inference is slower than that, frames are processed one by one without
  skipping, so no bottle is missed.
- **Looping.** When a looping file restarts, the tracker and all live tracks are reset so IDs don't
  carry over.
- **Streaming.** Each annotated frame is JPEG-encoded once and shared by every MJPEG client and the
  `/api/frame.jpg` snapshot endpoint.

---

## Bottle Tracking & Counting Logic

| Rule | Implementation |
|---|---|
| **Identity** | Track IDs come from ByteTrack/BoT-SORT via `model.track(..., persist=True)` |
| **Crossing** | `ltr`: `prev_cx < line_x <= cx` · `rtl`: `prev_cx > line_x >= cx` |
| **Exactly once** | Each track has a `counted` flag. A counted track is never counted again. |
| **No instant counts** | A crossing is ignored on the frame where a track first appears |
| **Partial bottles** | Bottles touching the left or right edge add no cap observations, which avoids judging half-visible caps |
| **Observation freeze** | After a bottle is counted, its observations stop changing, so the verdict is final |
| **Stale tracks** | Removed after 90 frames without a detection |

At the crossing, the engine:

1. computes the verdict from all of the track's observations,
2. increments the session counters and the per-colour histogram,
3. saves a padded JPEG crop to `data/snapshots/<session>/<count>_<track>.jpg`,
4. inserts an inspection row into SQLite, and
5. pushes an event that the WebSocket sends to the dashboard.

---

## Cap Colour Detection (Pink / Safia)

### Locating the cap

| Mode | When | How |
|---|---|---|
| **Detector** | The model has a class whose name contains `cap` or `lid` | The best-scoring cap whose centre is inside the bottle's x-range and between 10% above and 25% below the bottle's top edge |
| **Heuristic** | COCO models such as `yolo11n.pt` | The central 40% of the top 12% of the bottle box (the neck). Pixels count as cap-like if they are saturated (`S>40, V>60`) or very dark (`V<45`). The cap is present if the strongest rows average ≥ `heuristic_cap_threshold` (0.35). |

In detector mode the cap box is shrunk by 15% on each side before analysis, to avoid background
bleed. With a segmentation model, only the mask pixels are used.

### Classifying the colour (`bottleqc/color.py`)

Each pixel is converted to OpenCV HSV (`H` 0–179, `S`/`V` 0–255) and assigned to one of 9 colours.
The cap's colour is the most common label, and the confidence is that label's share of the pixels.

| Colour | Rule (checked in this order) |
|---|---|
| black | `V < 60` |
| white | `S < 50` and `V > 150` |
| **pink** | chromatic (`S ≥ 50`), `H ≥ 140` or `H < 8`, `S < 140`, `V > 130` |
| red | `H < 8` or `H ≥ 165` |
| orange | `H < 19` |
| yellow | `H < 35` |
| green | `H < 85` |
| blue | `H < 132` |
| purple | `H < 165` |

The **pink** rule matches Safia's pastel crown cap, measured from the product photo at about
`H 164–167, S 79, V ~200`. A vivid red cap has `S ≥ 140` and therefore falls in **red**, not pink.

The product loader runs the same classifier on the photo's top band to find the product's native
cap colour. For `safia.png` it returns **pink**, which is why `expected_color` defaults to `pink`.

### Voting across the track

Each frame in which the bottle is fully visible adds a cap present / absent observation and, when
present, a colour. The colour is the **majority vote** of those observations. One bad frame (blur,
glare or occlusion) cannot flip the result.

---

## Quality Classification

The rule for each bottle, from `decide()` in `engine.py`:

```text
if share_of_frames_with_cap < min_presence          → NO_CAP       (Missing cap)
elif expected_color != "any"
     and majority_colour != expected_color          → WRONG_COLOR  (Wrong colour)
else                                                → OK
```

| Status | Dashboard label | Overlay colour | Meaning |
|---|---|---|---|
| `OK` | OK | green | Cap present and of the expected colour |
| `NO_CAP` | Missing cap | red | No cap detected in enough frames |
| `WRONG_COLOR` | Wrong colour | amber | Cap present but of a different colour, e.g. red on a pink line |
| `PENDING` | — | grey | Live only: fewer than 2 observations so far |

Before the line, each box shows a **provisional** verdict computed the same way. The final verdict
is fixed at the crossing.

**First-pass yield** = `OK / total × 100`. The dashboard ring turns green at ≥ 95%, amber at
≥ 85%, and red below that.

---

## Dashboard Overview

A single-page dark UI served at `/`, with no build step.

| Area | Contents |
|---|---|
| **Top bar** | Model name, task and cap mode (`detector` / `heuristic`) · state pill (Idle, Loading, Running, Paused, Stopped, Finished, Error) · session number · elapsed time · **Reports** and **Settings** buttons |
| **Control bar** | Source selector and its inputs, video upload, expected-cap chip, **Start / Pause-Resume / Stop** |
| **Live inspection** | Annotated MJPEG stream with FPS, inference time (ms) and active-track badges, plus a colour legend. When stopped, the last frame stays on screen. |
| **KPI cards** | Total counted (and average per minute) · Passed (% of output) · Defects (missing / wrong colour split, with a flash on each new defect) · First-pass yield (ring gauge, plus bottles in the last minute) |
| **Production timeline** | Cumulative count and cumulative defects, sampled every 2 s |
| **Quality results** | Doughnut chart of OK, Missing cap and Wrong colour |
| **Cap colours** | Bar chart of detected cap colours, each bar in its real colour |
| **Inspection log** | Live table with a snapshot thumbnail (click to enlarge), ID, time, track, result, colour, colour confidence, detection confidence and frame count. Filter tabs: All, Defects, Missing cap, Wrong colour. **Load more** pagination. |
| **Settings drawer** | Expected colour swatches · model selector · confidence, IoU, line position and presence threshold sliders · flow direction · tracker · inference size · loop video · real-time pacing. Changes apply live. |
| **Reports drawer** | Report scope (current, all, or a specific session) · CSV / Excel / PDF buttons · session history with yield per session |
| **Toasts** | Start/stop confirmations, errors, and per-defect alerts with a thumbnail |

The layout adapts to tablet and phone widths. Live data arrives over a WebSocket, which reconnects
automatically if the server restarts.

---

## Reports & Exports

`GET /api/reports/export?fmt=csv|xlsx|pdf` with the scope set by `session_id` or `scope=all`. You
can also limit the time window with `since` and `until` (Unix timestamps).

| Format | Contents |
|---|---|
| **CSV** | One row per inspection: `id, session_id, time, track_id, status, cap_present, cap_color, color_conf, det_conf, observations, snapshot`. UTF-8 with BOM, so it opens cleanly in Excel. |
| **Excel (.xlsx)** | **Summary** sheet: title, scope, KPIs (total, passed, defects, missing cap, wrong colour, yield, first and last item), a **pie chart** of results and a **bar chart** of cap colours. **Inspections** sheet: the full log with a frozen header and auto-filter; *Missing cap* rows are red and *Wrong colour* rows amber. |
| **PDF** | A4 report with the KPI table, period, a results **pie chart**, a **defect gallery** (up to 12 bottle snapshots), and the inspection log (up to the last 1,500 rows) with defects highlighted. |

File names follow `bottleqc_<session|all>_<YYYYmmdd_HHMM>.<ext>`.

---

## API Documentation

Base URL: `http://127.0.0.1:8000`. FastAPI also serves interactive docs at **`/docs`** (Swagger)
and **`/redoc`**.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | `{ok, state, model_loaded}` |
| `GET` | `/api/stats` | Live state, counters, yield, throughput, FPS, inference ms, colour histogram, timeline, model info, config |
| `GET` | `/api/config` | Current config, allowed colours, available models |
| `POST` | `/api/config` | Update settings live. The body holds any subset of `conf, iou, imgsz, tracker, expected_color, line_pos, direction, min_presence, loop_video, realtime`. |
| `POST` | `/api/model?name=<file.pt>` | Load other weights from `models/` or by name. Restarts the session if it's running. |
| `POST` | `/api/control` | `{"action": "start"\|"stop"\|"pause"\|"resume", "source": {"type": "synthetic"\|"file"\|"camera"\|"stream", "value": "..."}}` |
| `POST` | `/api/upload` | Multipart `file` upload of `.mp4 .avi .mov .mkv .webm .m4v` → `{path, name}` |
| `GET` | `/api/videos` | Videos available in `samples/` and `data/uploads/` |
| `GET` | `/api/inspections` | `session_id`, `status` (`OK\|NO_CAP\|WRONG_COLOR\|DEFECT`), `limit` (1–1000, default 50), `offset` |
| `GET` | `/api/sessions` | Session history with total and OK counts |
| `GET` | `/api/sessions/{id}/summary` | Totals, per-status and per-colour counts, yield, time span |
| `GET` | `/api/snapshots/{session}/{file}.jpg` | Bottle snapshot |
| `GET` | `/api/reports/export` | `fmt` (`csv\|xlsx\|pdf`), `session_id`, `scope` (`session\|all`), `since`, `until` |
| `GET` | `/api/stream.mjpg` | Annotated live MJPEG stream (`multipart/x-mixed-replace`) |
| `GET` | `/api/frame.jpg` | Latest annotated frame |
| `WS` | `/ws` | Every 0.5 s: `{"type": "tick", "stats": {...}, "events": [new inspections]}` |

### Examples

```bash
# start the simulated line
curl -X POST http://127.0.0.1:8000/api/control -H "Content-Type: application/json" -d "{\"action\":\"start\",\"source\":{\"type\":\"synthetic\",\"value\":\"\"}}"
```

```bash
# switch to presence-only checking and move the counting line
curl -X POST http://127.0.0.1:8000/api/config -H "Content-Type: application/json" -d "{\"expected_color\":\"any\",\"line_pos\":0.6}"
```

```bash
# last 20 defects of the current session
curl "http://127.0.0.1:8000/api/inspections?status=DEFECT&limit=20"
```

```bash
# PDF report for all sessions
curl -o report.pdf "http://127.0.0.1:8000/api/reports/export?fmt=pdf&scope=all"
```

### Inspection record

```json
{
  "id": 42, "session_id": 6, "ts": 1794929144.06, "track_id": 17,
  "status": "WRONG_COLOR", "cap_present": 1, "cap_color": "red",
  "color_conf": 0.97, "det_conf": 0.91, "observations": 38,
  "snapshot": "6/000042_17.jpg"
}
```

---

## Evaluation & Test Results

`tools/evaluate.py` runs the real engine (`Engine.process`) frame by frame on the line simulator
and checks every counted bottle against the simulator's ground truth. It reports throughput,
counts, duplicates and verdict accuracy.

```bash
python tools/evaluate.py --model yolo11n.pt --frames 600
python tools/evaluate.py --model runs/bottleqc/weights/best.pt --frames 600
```

The evaluation simulator injects 20% missing caps and 20% wrong-colour caps, with `--seed 21` by
default.

### Measured results (600 frames, seed 21, Windows, CPU-only PyTorch)

| Mode | Weights | Bottles crossed | Counted | Duplicates | Verdict accuracy | Speed |
|---|---|---|---|---|---|---|
| **Heuristic (COCO)** | `yolo11n.pt` | 12 | **12** | **0** | **12 / 12 = 100%** | 2.0 FPS ¹ |
| **Detector (custom)** | `runs/bottleqc/weights/best.pt`, 2 epochs | 12 | 14 | 2 | 12 / 14 = 85.7% | 9.3 FPS |

¹ Measured while a training job was running on the same CPU, so this is not comparable with the
9.3 FPS figure.

**Heuristic mode** classified every bottle correctly: missing caps, black and red wrong-colour
caps, and all pink OK caps.

The **2-epoch custom checkpoint** detected and coloured caps correctly, but two bottles changed
track ID mid-frame. In both cases the first ID was finalized as *Missing cap* before the cap was
associated, and the second ID was then counted as *OK*. This checkpoint therefore has **not** been
promoted to `models/`. It needs full training (8+ epochs) and a re-run of the evaluation first.

### Detector training metrics (synthetic validation set, 120 images, imgsz 640)

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 0.970 | 0.976 | 0.988 | 0.689 |
| 2 | 0.992 | 0.992 | **0.995** | **0.787** |

Training was planned for 8 epochs and stopped during epoch 3. These numbers come from
`runs/bottleqc/results.csv`. They measure performance on synthetic composites of the real product,
not on footage from a physical line.

---

## Performance Considerations

| Item | Observation |
|---|---|
| **Inference** | The reference setup uses a CPU-only PyTorch build. With nothing else running, the custom YOLO11n at 640 px ran at **9.3 FPS**, including tracking, colour analysis and JPEG encoding. |
| **Training** | On CPU at 640 px, epochs took about **8–12 minutes** for 600 images (epoch 1 took 725 s and epoch 2 about 494 s). A CUDA GPU would make this much faster: install a CUDA build of PyTorch and pass `--device 0`. |
| **Shared CPU** | Running training and the app at the same time slows both a lot. Live FPS fell to about 2–3 during training. |
| **Simulator** | Renders a 960×540 frame in about **21 ms**. The environment is cached for each belt phase, and shadows are computed on small regions only. |
| **Resolution vs. cap size** | The Safia crown cap is only about 2.3% of the bottle's height. Use `imgsz` 640 or more; at 416 the cap becomes too small to detect reliably. |
| **Real-time pacing** | When inference is slower than the source FPS, files are processed frame by frame (no dropped frames), so counting stays correct and playback just looks slower. For live cameras the capture buffer is 1 frame, to keep latency low. |
| **GPU inference** | `Config.device` is `"cpu"` by default. Change it in `bottleqc/engine.py`, for example to `"0"`, to run inference on a GPU. |
| **Storage** | Snapshots are about 5–12 KB each and are kept for every bottle. Clean up `data/snapshots/` periodically on long runs. |

---

## Project Structure

```text
bottle-qc-vision/
├── run.py                    # Entry point (Uvicorn, --host/--port/--autostart)
├── requirements.txt
├── README.md
├── bottleqc/
│   ├── engine.py             # Sources, YOLO11 tracking, cap association, voting, counting, overlays
│   ├── color.py              # HSV 9-colour classifier + heuristic neck-region cap finder
│   ├── product.py            # Real-photo sprites: cap detection, no-cap & recolour variants, compositing
│   ├── scene.py              # Scene compositor: dataset scenes, demo video, live ConveyorSimulator
│   ├── storage.py            # SQLite: sessions, inspections, summaries
│   ├── reports.py            # CSV / XLSX / PDF report builders
│   └── server.py             # FastAPI: REST, WebSocket, MJPEG, uploads, static UI
├── web/
│   ├── index.html            # Dashboard markup
│   ├── styles.css            # Dark industrial theme, responsive layout
│   └── app.js                # Live updates, charts, log, settings, reports
├── tools/
│   ├── make_dataset.py       # Real-product YOLO dataset (bottle, cap)
│   ├── make_demo_video.py    # Demo conveyor video → samples/demo_line.mp4
│   ├── train.py              # YOLO11 fine-tuning → models/bottleqc_yolo11n.pt
│   └── evaluate.py           # Ground-truth evaluation of counting + QC
├── assets/
│   ├── product/safia.png     # Real product photo (RGBA cut-out)
│   └── backgrounds/          # (optional) real line photos
├── models/                   # (generated) trained weights loaded by the app
├── samples/                  # (generated) demo videos
├── datasets/bottles/         # (generated) YOLO dataset + data.yaml
├── runs/                     # (generated) Ultralytics training runs
└── data/                     # (generated) bottleqc.db, snapshots/, uploads/
```

`.gitignore` excludes `.venv/`, `data/`, `runs/`, `datasets/`, `*.log`, `yolo11n.pt` and
`samples/*.mp4`. All of these can be regenerated with the commands above.

---

## Troubleshooting

| Problem | Cause & fix |
|---|---|
| **Every bottle is flagged *Missing cap*** | In heuristic mode, check that `expected_color` matches your product and that the cap is visible at the top of the bottle box. Try lowering `min_presence`, or train the detector so caps are found directly. |
| **Caps flagged *Wrong colour*** | Check which colour is detected in the log. Set the right expected colour in Settings, or choose **any** to check presence only. Strong coloured lighting can shift hues. |
| **`Model ... has no 'bottle' class`** | The weights must contain a class named `bottle`. Custom models also need a class containing `cap` or `lid` to use detector mode. |
| **`Cannot open source` / `File not found`** | Check the camera index, URL or file path. Relative file paths are resolved from the project root. |
| **Low FPS** | Stop other heavy jobs, especially training. Lower the inference size only if caps stay large enough, or use a GPU. |
| **Bottles counted twice** | This happens when the tracker switches IDs. Try **BoT-SORT**, raise `conf`, move the counting line to where bottles are fully visible and well separated, or train the detector longer. |
| **Charts are empty but numbers update** | Chart.js loads from the jsDelivr CDN, so the browser needs internet access. |
| **Dashboard stuck on *Idle*** | The page reconnects its WebSocket automatically. Check that `run.py` is still running and that only **one** server process is bound to the port. |
| **`No RGBA product photos ... found`** | `assets/product/` needs at least one `.png` with transparency. |
| **`Could not locate a coloured cap`** | The product photo's top 15% must contain a coloured cap band, with saturation above 35. |
| **Training is very slow** | It's expected on CPU (about 8–12 min/epoch at 640). Use fewer images or epochs for a test run, or a CUDA GPU with `--device 0`. |

---

## Future Improvements

None of the following is implemented yet:

- [] **Real-line dataset**: label real camera frames of the Safia line and fine-tune on them.
      The current model is trained on composites of one product photo.
- [] **Complete training** of the detector (8+ epochs), then promote it to `models/` and re-run
      `tools/evaluate.py`.
- [] **Track-switch handling**: merge a new ID with a recently lost one near the line, so
      a single bottle can't be counted twice.
- [] **Segmentation dataset**: have `make_dataset.py` also emit polygon labels for
      `yolo11n-seg`. The engine already uses masks when present.
- [] **More defect classes**: fill level, label presence and alignment, crooked or loose caps.
- [] **Line integration**: PLC or digital-output reject signals on defects.
- [] **GPU / export**: device selection in the UI; ONNX or TensorRT export for edge devices.
- [] **Multi-line & auth**: several cameras at once, user accounts and role-based access.
- [] **Offline UI**: bundle Chart.js and the fonts locally instead of loading them from a CDN.

---

<div align="center">

**BottleQC Vision**: computer vision, object tracking and real-time AI for industrial quality control.

Built with Ultralytics YOLO11 · FastAPI · OpenCV

</div>
