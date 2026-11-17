"""Start the BottleQC Vision server.

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 8080 --host 0.0.0.0
    python run.py --autostart synthetic
"""
import argparse
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--autostart", choices=["synthetic", "file", "camera", "stream"], default=None)
    ap.add_argument("--source", default="", help="value for --autostart (path, camera index or URL)")
    a = ap.parse_args()

    if a.autostart:
        from bottleqc.server import engine

        def go():
            import time
            while engine.model is None:
                time.sleep(0.2)
            engine.start(a.autostart, a.source)
        threading.Thread(target=go, daemon=True).start()

    uvicorn.run("bottleqc.server:app", host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
