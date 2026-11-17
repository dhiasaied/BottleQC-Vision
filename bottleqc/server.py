"""FastAPI app: REST + WebSocket + MJPEG stream + static dashboard."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import reports
from .color import COLOR_NAMES
from .engine import DATA, ROOT, Config, Engine
from .storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bottleqc")

storage = Storage(DATA / "bottleqc.db")
engine = Engine(storage)
WEB = ROOT / "web"
UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, engine.load_model)
    except Exception:
        log.exception("model load failed")
    yield
    engine.stop()


app = FastAPI(title="BottleQC Vision", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ models
class SourceReq(BaseModel):
    type: str = Field(pattern="^(synthetic|file|camera|stream)$")
    value: str = ""


class ConfigReq(BaseModel):
    conf: float | None = Field(None, ge=0.05, le=0.95)
    iou: float | None = Field(None, ge=0.1, le=0.95)
    imgsz: int | None = Field(None, ge=160, le=1280)
    tracker: str | None = Field(None, pattern="^(bytetrack|botsort)\\.yaml$")
    expected_color: str | None = None
    line_pos: float | None = Field(None, ge=0.05, le=0.95)
    direction: str | None = Field(None, pattern="^(ltr|rtl)$")
    min_presence: float | None = Field(None, ge=0.05, le=1.0)
    loop_video: bool | None = None
    realtime: bool | None = None


class ControlReq(BaseModel):
    action: str = Field(pattern="^(start|stop|pause|resume)$")
    source: SourceReq | None = None


# ------------------------------------------------------------------ REST
@app.get("/api/health")
def health():
    return {"ok": True, "state": engine.state, "model_loaded": engine.model is not None}


@app.get("/api/stats")
def stats():
    return engine.stats()


@app.get("/api/config")
def get_config():
    return {"config": stats()["config"], "colors": ["any"] + COLOR_NAMES,
            "models": [p.name for p in (ROOT / "models").glob("*.pt")] + ["yolo11n.pt"]}


@app.post("/api/config")
def set_config(req: ConfigReq):
    data = req.model_dump(exclude_none=True)
    if "expected_color" in data and data["expected_color"] not in ["any"] + COLOR_NAMES:
        raise HTTPException(422, f"expected_color must be one of {['any'] + COLOR_NAMES}")
    names = {f.name for f in fields(Config)}
    for k, v in data.items():
        if k in names:
            setattr(engine.cfg, k, v)
    if "tracker" in data:
        engine._reset_tracker = True
    return {"config": engine.stats()["config"]}


@app.post("/api/model")
def set_model(name: str = Query(...)):
    path = ROOT / "models" / name if (ROOT / "models" / name).exists() else name
    if not re.fullmatch(r"[\w.\-]+\.pt", Path(str(path)).name):
        raise HTTPException(422, "invalid model name")
    was_running = engine.state in ("running", "paused")
    if was_running:
        engine.stop()
    try:
        engine.load_model(str(path))
    except Exception as e:
        raise HTTPException(400, str(e))
    if was_running:
        engine.start()
    return engine.model_info()


@app.post("/api/control")
async def control(req: ControlReq):
    loop = asyncio.get_running_loop()
    try:
        if req.action == "start":
            src = req.source
            await loop.run_in_executor(None, lambda: engine.start(src.type if src else None, src.value if src else None))
        elif req.action == "stop":
            await loop.run_in_executor(None, engine.stop)
        elif req.action == "pause":
            engine.pause(True)
        elif req.action == "resume":
            engine.pause(False)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"state": engine.state, "session_id": engine.session_id}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(415, "Unsupported video format")
    safe = re.sub(r"[^\w.\-]", "_", Path(file.filename).stem)[:60]
    dest = UPLOADS / f"{int(time.time())}_{safe}{ext}"
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"path": str(dest.relative_to(ROOT)).replace("\\", "/"), "name": file.filename}


@app.get("/api/videos")
def videos():
    out = []
    for d in (ROOT / "samples", UPLOADS):
        if d.exists():
            out += [{"name": p.name, "path": str(p.relative_to(ROOT)).replace("\\", "/")}
                    for p in sorted(d.iterdir()) if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}]
    return out


@app.get("/api/inspections")
def inspections(session_id: int | None = None, status: str | None = Query(None, pattern="^(OK|NO_CAP|WRONG_COLOR|DEFECT)$"),
                limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    sid = session_id if session_id is not None else engine.session_id
    return storage.inspections(session_id=sid, status=status, limit=limit, offset=offset)


@app.get("/api/sessions")
def sessions():
    return storage.sessions()


@app.get("/api/sessions/{sid}/summary")
def session_summary(sid: int):
    if not storage.session(sid):
        raise HTTPException(404, "session not found")
    return storage.summary(sid)


@app.get("/api/snapshots/{sid}/{name}")
def snapshot(sid: int, name: str):
    if not re.fullmatch(r"[\w\-]+\.jpg", name):
        raise HTTPException(404)
    p = DATA / "snapshots" / str(sid) / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/reports/export")
def export(fmt: str = Query("csv", pattern="^(csv|xlsx|pdf)$"), session_id: int | None = None,
           scope: str = Query("session", pattern="^(session|all)$"),
           since: float | None = None, until: float | None = None):
    sid = None if scope == "all" else (session_id or engine.session_id)
    fn = {"csv": reports.export_csv, "xlsx": reports.export_xlsx, "pdf": reports.export_pdf}[fmt]
    body = fn(storage, sid, since, until)
    media = {"csv": "text/csv", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             "pdf": "application/pdf"}[fmt]
    name = f"bottleqc_{'all' if sid is None else f'session{sid}'}_{datetime.now():%Y%m%d_%H%M}.{fmt}"
    return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------------------------------------------ streaming
@app.get("/api/stream.mjpg")
async def stream():
    boundary = "frame"

    async def gen():
        last = -1
        idle = 0
        while True:
            if engine.frame_seq != last and engine.jpeg:
                last = engine.frame_seq
                jpg = engine.jpeg
                yield (b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                idle = 0
            else:
                idle += 1
            await asyncio.sleep(0.02 if idle < 500 else 0.2)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/frame.jpg")
def frame():
    if not engine.jpeg:
        raise HTTPException(404, "no frame yet")
    return Response(engine.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    seq = engine.event_seq
    try:
        while True:
            evs = engine.events_since(seq)
            if evs:
                seq = evs[-1][0]
            await sock.send_json({"type": "tick", "stats": engine.stats(), "events": [r for _, r in evs]})
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError, ConnectionError):
        pass
    except Exception as e:  # client vanished mid-send (websockets ConnectionClosed etc.)
        log.debug("ws closed: %s", e)


# ------------------------------------------------------------------ static
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
