"""SQLite persistence for production sessions and per-bottle inspections."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    source TEXT,
    model TEXT,
    expected_color TEXT
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    ts REAL NOT NULL,
    track_id INTEGER,
    status TEXT NOT NULL,          -- OK | NO_CAP | WRONG_COLOR
    cap_present INTEGER NOT NULL,
    cap_color TEXT,
    color_conf REAL,
    det_conf REAL,
    observations INTEGER,
    snapshot TEXT
);
CREATE INDEX IF NOT EXISTS ix_insp_session ON inspections(session_id);
CREATE INDEX IF NOT EXISTS ix_insp_ts ON inspections(ts);
"""


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    # sessions -----------------------------------------------------------
    def start_session(self, source: str, model: str, expected: str) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO sessions(started_at, source, model, expected_color) VALUES (?,?,?,?)",
                (time.time(), source, model, expected))
            self.db.commit()
            return int(cur.lastrowid)

    def end_session(self, sid: int):
        with self.lock:
            self.db.execute("UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL", (time.time(), sid))
            self.db.commit()

    def sessions(self, limit=50):
        with self.lock:
            rows = self.db.execute(
                """SELECT s.*, COUNT(i.id) AS total,
                          SUM(CASE WHEN i.status='OK' THEN 1 ELSE 0 END) AS ok
                   FROM sessions s LEFT JOIN inspections i ON i.session_id = s.id
                   GROUP BY s.id ORDER BY s.id DESC LIMIT ?""", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def session(self, sid: int):
        with self.lock:
            r = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    # inspections ---------------------------------------------------------
    def add_inspection(self, rec: dict) -> int:
        with self.lock:
            cur = self.db.execute(
                """INSERT INTO inspections(session_id, ts, track_id, status, cap_present, cap_color,
                                           color_conf, det_conf, observations, snapshot)
                   VALUES (:session_id,:ts,:track_id,:status,:cap_present,:cap_color,
                           :color_conf,:det_conf,:observations,:snapshot)""", rec)
            self.db.commit()
            return int(cur.lastrowid)

    def inspections(self, session_id: int | None = None, status: str | None = None,
                    limit: int = 100, offset: int = 0, since: float | None = None, until: float | None = None):
        q, args = "SELECT * FROM inspections WHERE 1=1", []
        if session_id is not None:
            q += " AND session_id=?"; args.append(session_id)
        if status:
            if status == "DEFECT":
                q += " AND status!='OK'"
            else:
                q += " AND status=?"; args.append(status)
        if since is not None:
            q += " AND ts>=?"; args.append(since)
        if until is not None:
            q += " AND ts<=?"; args.append(until)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self.lock:
            return [dict(r) for r in self.db.execute(q, args).fetchall()]

    def summary(self, session_id: int | None = None, since: float | None = None, until: float | None = None):
        q, args = "FROM inspections WHERE 1=1", []
        if session_id is not None:
            q += " AND session_id=?"; args.append(session_id)
        if since is not None:
            q += " AND ts>=?"; args.append(since)
        if until is not None:
            q += " AND ts<=?"; args.append(until)
        with self.lock:
            by_status = {r[0]: r[1] for r in self.db.execute(f"SELECT status, COUNT(*) {q} GROUP BY status", args)}
            by_color = {r[0] or "none": r[1] for r in self.db.execute(f"SELECT cap_color, COUNT(*) {q} GROUP BY cap_color", args)}
            span = self.db.execute(f"SELECT MIN(ts), MAX(ts) {q}", args).fetchone()
        total = sum(by_status.values())
        ok = by_status.get("OK", 0)
        return {
            "total": total, "ok": ok, "defects": total - ok,
            "no_cap": by_status.get("NO_CAP", 0), "wrong_color": by_status.get("WRONG_COLOR", 0),
            "yield": round(100.0 * ok / total, 2) if total else None,
            "by_color": by_color, "first_ts": span[0], "last_ts": span[1],
        }
