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
from flask import Flask, Response, request


# Shared JPEG frame for MJPEG streaming
_frame_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_turret: ManualTurretController | None = None
_control_args: argparse.Namespace | None = None

app = Flask(__name__)


class ManualTurretController:
    def __init__(self, pan_port_1: str, pan_port_2: str, tilt_port: str, motor_speed: int) -> None:
        self.pan_1 = Motor(pan_port_1)
        self.pan_2 = Motor(pan_port_2)
        self.tilt = Motor(tilt_port)
        self.motor_speed = motor_speed
        self._lock = threading.Lock()

    def pan(self, degrees: int) -> None:
        with self._lock:
            self.pan_1.run_for_degrees(degrees, speed=self.motor_speed, blocking=False)
            self.pan_2.run_for_degrees(degrees, speed=self.motor_speed, blocking=False)

    def tilt_move(self, degrees: int) -> None:
        with self._lock:
            self.tilt.run_for_degrees(degrees, speed=self.motor_speed, blocking=False)

    def stop(self) -> None:
        with self._lock:
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
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>Camera Control</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>"
        "body{margin:0;background:#070a12;color:#dbe8ff;font-family:Segoe UI,Tahoma,sans-serif;}"
        ".wrap{display:grid;grid-template-rows:1fr auto;min-height:100vh;}"
        ".feed{display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at 20% 10%,#1c2948 0,#070a12 55%);}"
        "img{width:100%;height:100%;object-fit:contain;}"
        ".panel{padding:12px 14px 16px;background:#10192b;border-top:1px solid #223558;}"
        ".title{font-weight:700;margin-bottom:8px;}"
        ".hint{opacity:.86;margin-bottom:10px;font-size:14px;}"
        ".pad{display:grid;grid-template-columns:56px 56px 56px;grid-template-rows:56px 56px 56px;gap:8px;justify-content:center;}"
        "button{border:none;border-radius:10px;background:#1f3357;color:#e8f2ff;font-size:22px;font-weight:700;}"
        "button:active{background:#2b4a7f;}"
        "#status{margin-top:8px;font-size:13px;opacity:.9;text-align:center;}"
        "</style></head><body>"
        "<div class='wrap'>"
        "<div class='feed'><img src='/video' alt='Camera feed'></div>"
        "<div class='panel'>"
        "<div class='title'>Turret Control</div>"
        "<div class='hint'>Use arrow keys while this page is focused, or tap the buttons.</div>"
        "<div class='pad'>"
        "<div></div><button data-dir='up' aria-label='Up'>▲</button><div></div>"
        "<button data-dir='left' aria-label='Left'>◀</button><div></div><button data-dir='right' aria-label='Right'>▶</button>"
        "<div></div><button data-dir='down' aria-label='Down'>▼</button><div></div>"
        "</div>"
        "<div id='status'>Ready</div>"
        "</div></div>"
        "<script>"
        "const statusEl=document.getElementById('status');"
        "const keyMap={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'up',ArrowDown:'down'};"
        "let lastSend=0;"
        "function setStatus(msg){statusEl.textContent=msg;}"
        "function send(dir){"
        "  const now=Date.now();"
        "  if(now-lastSend<70){return;}"
        "  lastSend=now;"
        "  fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({direction:dir})})"
        "    .then(r=>{if(!r.ok){throw new Error('control failed');} setStatus('Sent: '+dir);})"
        "    .catch(()=>setStatus('Control request failed'));"
        "}"
        "window.addEventListener('keydown',e=>{"
        "  const dir=keyMap[e.key];"
        "  if(!dir){return;}"
        "  e.preventDefault();"
        "  send(dir);"
        "});"
        "document.querySelectorAll('button[data-dir]').forEach(btn=>{"
        "  btn.addEventListener('click',()=>send(btn.dataset.dir));"
        "});"
        "window.addEventListener('focus',()=>setStatus('Page focused - arrows active'));"
        "window.addEventListener('blur',()=>setStatus('Page not focused'))"
        "</script></body></html>"
    )


def apply_direction(direction: str, turret: ManualTurretController, args: argparse.Namespace) -> bool:
    if direction == "left":
        turret.pan(-args.pan_direction * args.pan_step_degrees)
        return True
    if direction == "right":
        turret.pan(args.pan_direction * args.pan_step_degrees)
        return True
    if direction == "up":
        turret.tilt_move(args.tilt_direction * args.tilt_step_degrees)
        return True
    if direction == "down":
        turret.tilt_move(-args.tilt_direction * args.tilt_step_degrees)
        return True
    return False


@app.route("/control", methods=["POST"])
def control() -> tuple[str, int]:
    if _turret is None or _control_args is None:
        return ("Controller not ready", 503)
    payload = request.get_json(silent=True) or {}
    direction = str(payload.get("direction", "")).lower()
    if not apply_direction(direction, _turret, _control_args):
        return ("Invalid direction", 400)
    return ("", 204)


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
        default=100,
        help="Motor speed for manual key control (0-100).",
    )
    parser.add_argument(
        "--pan-step-degrees",
        type=int,
        default=30,
        help="Pan step size per key press in degrees.",
    )
    parser.add_argument(
        "--tilt-step-degrees",
        type=int,
        default=30,
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
        default=1,
        help="Set to -1 or 1 to match your tilt direction.",
    )
    return parser.parse_args()


def handle_keypress(key: int, turret: ManualTurretController, args: argparse.Namespace) -> bool:
    left_keys = {81, 2424832, 65361}
    up_keys = {82, 2490368, 65362}
    right_keys = {83, 2555904, 65363}
    down_keys = {84, 2621440, 65364}

    if key in left_keys:
        return apply_direction("left", turret, args)
    if key in right_keys:
        return apply_direction("right", turret, args)
    if key in up_keys:
        return apply_direction("up", turret, args)
    if key in down_keys:
        return apply_direction("down", turret, args)
    return False


def main() -> None:
    global _latest_jpeg, _turret, _control_args

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
    _turret = turret
    _control_args = args
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
    print("[Info] Arrow keys work in preview window and in browser stream page.")
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
