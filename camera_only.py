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
from buildhat import Motor
from flask import Flask, Response


# Shared JPEG frame for MJPEG streaming
_frame_lock = threading.Lock()
_latest_jpeg: bytes | None = None

app = Flask(__name__)


class ManualTurretController:
    def __init__(self, pan_port_1: str, pan_port_2: str, tilt_port: str, motor_speed: int) -> None:
        self.pan_1 = Motor(pan_port_1)
        self.pan_2 = Motor(pan_port_2)
        self.tilt = Motor(tilt_port)
        self.motor_speed = motor_speed

    def pan(self, degrees: int) -> None:
        self.pan_1.run_for_degrees(degrees, speed=self.motor_speed, blocking=False)
        self.pan_2.run_for_degrees(degrees, speed=self.motor_speed, blocking=False)

    def tilt_move(self, degrees: int) -> None:
        self.tilt.run_for_degrees(degrees, speed=self.motor_speed, blocking=False)

    def stop(self) -> None:
        self.pan_1.stop()
        self.pan_2.stop()
        self.tilt.stop()


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
    parser.add_argument(
        "--pan-port-1",
        default="B",
        help="Build HAT port for first pan motor.",
    )
    parser.add_argument(
        "--pan-port-2",
        default="C",
        help="Build HAT port for second pan motor.",
    )
    parser.add_argument(
        "--tilt-port",
        default="A",
        help="Build HAT port for tilt motor.",
    )
    parser.add_argument(
        "--motor-speed",
        type=int,
        default=35,
        help="Motor speed for manual key control (0-100).",
    )
    parser.add_argument(
        "--pan-step-degrees",
        type=int,
        default=10,
        help="Pan step size per key press in degrees.",
    )
    parser.add_argument(
        "--tilt-step-degrees",
        type=int,
        default=8,
        help="Tilt step size per key press in degrees.",
    )
    parser.add_argument(
        "--pan-direction",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Set to -1 if left/right are reversed.",
    )
    parser.add_argument(
        "--tilt-direction",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="Set to -1 or 1 to match your tilt direction.",
    )
    return parser.parse_args()


def handle_keypress(key: int, turret: ManualTurretController, args: argparse.Namespace) -> bool:
    left_keys = {81, 2424832, 65361}
    up_keys = {82, 2490368, 65362}
    right_keys = {83, 2555904, 65363}
    down_keys = {84, 2621440, 65364}

    if key in left_keys:
        turret.pan(-args.pan_direction * args.pan_step_degrees)
        return True
    if key in right_keys:
        turret.pan(args.pan_direction * args.pan_step_degrees)
        return True
    if key in up_keys:
        turret.tilt_move(args.tilt_direction * args.tilt_step_degrees)
        return True
    if key in down_keys:
        turret.tilt_move(-args.tilt_direction * args.tilt_step_degrees)
        return True
    return False


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

    print("[Motors] Initialising manual control...")
    turret = ManualTurretController(
        pan_port_1=args.pan_port_1,
        pan_port_2=args.pan_port_2,
        tilt_port=args.tilt_port,
        motor_speed=args.motor_speed,
    )
    print(
        "[Motors] Ready "
        f"(Pan={args.pan_port_1}+{args.pan_port_2}, Tilt={args.tilt_port}, Speed={args.motor_speed})"
    )

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
    print("[Info] Arrow keys control turret in preview. Press q or ESC to quit.")

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
            cv2.putText(
                frame,
                "ARROWS: MOVE TURRET | Q/ESC: QUIT",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_jpg:
                with _frame_lock:
                    _latest_jpeg = buf.tobytes()

            if show_preview:
                cv2.imshow("Camera Only", frame)
                key = cv2.waitKeyEx(1)
                handle_keypress(key, turret, args)
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        turret.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
