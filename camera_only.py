"""
camera_only.py - camera feed only mode (no motors, no tracking).

Usage:
  python3 camera_only.py

Optional:
  python3 camera_only.py --headless --port 5001

Open browser stream at:
  http://<pi-ip>:5001/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time

import cv2
from flask import Flask, Response


# Shared JPEG frame for MJPEG streaming
_frame_lock = threading.Lock()
_latest_jpeg: bytes | None = None

app = Flask(__name__)


def find_usb_camera() -> int:
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        if "usb" in os.path.realpath(path).lower():
            try:
                return int(os.path.basename(path).replace("video", ""))
            except ValueError:
                continue
    return 0


@app.route("/video")
def video() -> Response:
    def gen():
        while True:
            with _frame_lock:
                jpeg = _latest_jpeg
            if jpeg is None:
                time.sleep(0.01)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index() -> str:
    return (
        "<html><body style='margin:0;background:#000'>"
        "<img src='/video' style='width:100%;height:100vh;object-fit:contain'>"
        "</body></html>"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera feed only mode")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable local OpenCV preview window.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="HTTP port for MJPEG stream.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Force camera index (defaults to first detected USB camera).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height.",
    )
    return parser.parse_args()


def main() -> None:
    global _latest_jpeg

    args = parse_args()

    cam_idx = args.camera_index if args.camera_index is not None else find_usb_camera()

    print(f"[Camera] Opening /dev/video{cam_idx} ...")
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[FATAL] Cannot open camera.")
        sys.exit(1)
    print("[Camera] OK")

    show_preview = not args.headless
    if show_preview and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        show_preview = False
        print("[Preview] No display detected, running without local preview.")

    stream_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False),
        daemon=True,
    )
    stream_thread.start()

    print(f"[Stream] http://0.0.0.0:{args.port}/")
    print("[Info] Press q or ESC in preview window to quit.")

    prev_t = time.time()
    fps_ema = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            now = time.time()
            fps_ema = 0.1 * (1.0 / max(now - prev_t, 1e-6)) + 0.9 * fps_ema
            prev_t = now

            cv2.putText(
                frame,
                f"CAMERA ONLY | FPS {fps_ema:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_jpg:
                with _frame_lock:
                    _latest_jpeg = buf.tobytes()

            if show_preview:
                cv2.imshow("Camera Only", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
