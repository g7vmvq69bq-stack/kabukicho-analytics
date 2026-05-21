"""
Kabukicho Urban Pulse Monitor — main recording loop.

Usage
-----
    python src/main.py                      # stream from config.py
    python src/main.py --duration 600       # stop after 10 minutes
    python src/main.py --no-display         # headless (no OpenCV window)
    python src/main.py --url <other_url>    # override stream URL

Press  q  in the video window to stop early.
Data is saved to data/kabukicho.db and data/kabukicho.csv.
"""

import argparse, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import config
from stream_reader import open_stream
from detector import Detector
from storage import DataStore
from event_engine import EventEngine


def run(stream_url: str, duration: float | None, show_display: bool):
    detector = Detector()
    store    = DataStore()
    events   = EventEngine(store._conn)
    cap      = open_stream(stream_url)

    window_start  = datetime.now(timezone.utc)
    window_frames = []
    frame_index   = 0
    start_time    = time.time()
    total_windows = 0

    print("[main] Recording started — press q to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[main] Stream ended or connection lost.")
                break

            if duration and (time.time() - start_time) >= duration:
                print(f"[main] Duration limit ({duration}s) reached.")
                break

            frame_index += 1
            if frame_index % config.FRAME_SKIP != 0:
                continue

            result = detector.detect(frame)
            window_frames.append(result)

            if show_display:
                cv2.imshow("Kabukicho Urban Pulse — press q to stop",
                           result["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[main] Stopped by user.")
                    break

            now = datetime.now(timezone.utc)
            if (now - window_start).total_seconds() >= config.AGGREGATION_WINDOW_SECONDS:
                row = store.save_window(window_start, window_frames)
                if row:
                    events.check(row)
                total_windows += 1
                window_start  = now
                window_frames = []

    finally:
        if window_frames:
            row = store.save_window(window_start, window_frames)
            if row:
                events.check(row)
        cap.release()
        cv2.destroyAllWindows()
        events.close()
        store.close()
        print(f"\n[main] Done — {total_windows + 1} windows saved.")
        print(f"[main] DB      : {config.DB_PATH}")
        print(f"[main] CSV     : {config.CSV_PATH}")
        print(f"[main] Events  : {config.DATA_DIR / 'kabukicho_events.csv'}")
        print("\nRun  python src/analysis.py  to generate charts.\n")


def main():
    parser = argparse.ArgumentParser(description="Kabukicho Urban Pulse Monitor")
    parser.add_argument("--url",        default=config.STREAM_URL)
    parser.add_argument("--duration",   type=float, default=None,
                        help="Stop after N seconds")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless — no OpenCV window")
    args = parser.parse_args()
    run(args.url, args.duration, not args.no_display)


if __name__ == "__main__":
    main()
