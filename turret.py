"""
turret.py — USB webcam + YOLO + motors + live annotated stream.

Run on the Pi:
  python3 turret.py

Then open in browser on your laptop:
  http://10.47.206.127:5001/

Ports:
  A = Tilt   B = Pan (motor 1)   C = Pan (motor 2)   D = Shoot
"""

import glob
import os
import sys
import time
import threading
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
from buildhat import Motor

# ==============================================================================
# CONFIGURE
# ==============================================================================

PAN_PORT_1 = "B"
PAN_PORT_2 = "C"
TILT_PORT  = "A"
SHOOT_PORT = "D"

PAN_DIRECTION  = 1   # flip to -1 if reversed
TILT_DIRECTION = -1

# Motor speed when tracking (0-100)
MOTOR_SPEED = 100

# Pan proportional control
PAN_KP        = 0.10
PAN_MIN_SPEED = 5
PAN_MAX_SPEED = 15

# Tilt proportional control — |dy| px * TILT_KP = speed, clamped
TILT_KP        = 0.10
TILT_MIN_SPEED = 5
TILT_MAX_SPEED = 15

# Deadzone — within this many px on both axes = locked, shoot
DEADZONE_PX = 120

# Step mode â short pulse then pause for next frame
STEP_DURATION = 0.03

SHOOT_COOLDOWN = 2.0
SHOOT_DURATION = 0.3

CONFIDENCE           = 0.45
GRAVITY_OFFSET       = 0.0

OUTPUT_PORT = 5001

# ==============================================================================

# Shared annotated frame for the stream
_frame_lock   = threading.Lock()
_latest_jpeg  = None

app = Flask(__name__)

@app.route("/video")
def video():
    def gen():
        while True:
            with _frame_lock:
                j = _latest_jpeg
            if j is None:
                time.sleep(0.01)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return (
        "<html><body style='margin:0;background:#000'>"
        "<img src='/video' style='width:100%;height:100vh;object-fit:contain'>"
        "</body></html>"
    )


# ==============================================================================
# Drawing
# ==============================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

def text(img, t, pos, scale, color, thick=2):
    x, y = pos
    cv2.putText(img, t, (x-1,y-1), FONT, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, t, (x+1,y+1), FONT, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, t, pos,       FONT, scale, color,   thick,   cv2.LINE_AA)

def crosshair(img, cx, cy, size=20):
    cv2.line(img, (cx-size, cy), (cx+size, cy), (0,0,220), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy-size), (cx, cy+size), (0,0,220), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), size//2, (0,0,220), 1, cv2.LINE_AA)

def draw(frame, detections, primary, tx, ty, fps, locked):
    h, w = frame.shape[:2]

    for d in detections:
        cv2.rectangle(frame, (d['x1'],d['y1']), (d['x2'],d['y2']), (0,200,0), 2)
        text(frame, f"PERSON {d['conf']*100:.0f}%", (d['x1'], max(d['y1']-6,14)), 0.5, (0,200,0), 1)

    if primary:
        cv2.rectangle(frame, (primary['x1'],primary['y1']), (primary['x2'],primary['y2']), (0,0,220), 3)
        text(frame, f"TARGET {primary['conf']*100:.0f}%", (primary['x1'], max(primary['y1']-6,14)), 0.55, (0,0,220), 2)
        cx = (primary['x1']+primary['x2'])//2
        cy = primary['y1'] + int((primary['y2']-primary['y1']) * 0.15)
        cv2.circle(frame, (cx, cy), 6, (0,0,220), -1)
        cv2.line(frame, (cx, cy), (tx, ty), (0,0,220), 1, cv2.LINE_AA)

    crosshair(frame, tx, ty)

    text(frame, f"FPS: {fps:.1f}", (10, 28), 0.6, (220,220,0), 1)

    count = f"Persons: {len(detections)}"
    (tw,_),_ = cv2.getTextSize(count, FONT, 0.55, 1)
    text(frame, count, (w-tw-12, 28), 0.55, (255,255,255), 1)

    if not detections:
        status, col = "SCANNING...", (0,220,220)
    elif locked:
        status, col = "** LOCKED - SHOOT **", (0,0,220)
    else:
        status, col = "ACQUIRING", (255,255,255)

    (sw,_),_ = cv2.getTextSize(status, FONT, 0.75, 2)
    text(frame, status, ((w-sw)//2, h-14), 0.75, col, 2)

    if locked:
        shoot_s = "SHOOT"
        scale = 2.2
        (lw,lh),_ = cv2.getTextSize(shoot_s, FONT, scale, 4)
        text(frame, shoot_s, ((w-lw)//2, h//2+lh//2), scale, (0,0,220), 4)


# ==============================================================================
# Main loop
# ==============================================================================

def find_usb_camera():
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        if "usb" in os.path.realpath(path).lower():
            try:
                return int(os.path.basename(path).replace("video",""))
            except ValueError:
                continue
    return 0


def turret_loop():
    global _latest_jpeg

    # Motors
    print("[Motors] Initialising...")
    time.sleep(2)
    pan1  = Motor(PAN_PORT_1)
    pan2  = Motor(PAN_PORT_2)
    tilt  = Motor(TILT_PORT)
    shoot = Motor(SHOOT_PORT)
    print(f"[Motors] Pan={PAN_PORT_1}+{PAN_PORT_2}  Tilt={TILT_PORT}  Shoot={SHOOT_PORT}")

    # Camera
    idx = find_usb_camera()
    print(f"[Camera] Opening /dev/video{idx} ...")
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[FATAL] Cannot open camera.")
        sys.exit(1)
    print("[Camera] OK")

    print("[YOLO]  Loading NCNN model...")
    model = YOLO("/home/goon/yolov8n_ncnn_model")
    print("[YOLO]  Ready")
    print(f"[Stream] http://10.47.206.127:{OUTPUT_PORT}/")

    # Tracker state
    tracked = None
    lost_frames   = 0
    LOST_LIMIT    = 10
    TRACK_ALPHA   = 0.8
    MATCH_DIST    = 120

    target_visible = False
    last_shoot_t   = 0.0
    prev_t         = time.time()
    fps_ema        = 0.0
    frame_num      = 0

    # Motor state — track current direction to avoid spamming start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            fh, fw = frame.shape[:2]
            tx = fw // 2
            ty = int(fh // 2 - fh * GRAVITY_OFFSET)

            results = model(frame, verbose=False, conf=CONFIDENCE, classes=[0], imgsz=320)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    detections.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'conf':float(box.conf[0])})

            frame_num += 1
            now = time.time()
            fps_ema = 0.1*(1.0/max(now-prev_t,1e-6)) + 0.9*fps_ema
            prev_t  = now

            # Track
            best = None
            if detections:
                if tracked is None:
                    best = max(detections, key=lambda d:(d['x2']-d['x1'])*(d['y2']-d['y1']))
                else:
                    tcx = (tracked[0]+tracked[2])/2
                    tcy = (tracked[1]+tracked[3])/2
                    best = min(detections, key=lambda d:((d['x1']+d['x2'])//2-tcx)**2+((d['y1']+d['y2'])//2-tcy)**2)
                    bcx = (best['x1']+best['x2'])//2
                    bcy = (best['y1']+best['y2'])//2
                    if ((bcx-tcx)**2+(bcy-tcy)**2)**0.5 > MATCH_DIST:
                        tracked = None
                        best = max(detections, key=lambda d:(d['x2']-d['x1'])*(d['y2']-d['y1']))
                if tracked is None:
                    tracked = [float(best['x1']),float(best['y1']),float(best['x2']),float(best['y2'])]
                else:
                    for i,v in enumerate([best['x1'],best['y1'],best['x2'],best['y2']]):
                        tracked[i] += TRACK_ALPHA*(v-tracked[i])
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames >= LOST_LIMIT:
                    tracked = None

            primary = None
            if tracked is not None and best is not None:
                primary = {'x1':int(tracked[0]),'y1':int(tracked[1]),
                           'x2':int(tracked[2]),'y2':int(tracked[3]),
                           'conf':best['conf']}

            # Act
            if primary:
                if not target_visible:
                    print(f"[FOUND]  conf={primary['conf']*100:.0f}%")
                    target_visible = True

                cx = (primary['x1']+primary['x2'])//2
                cy = primary['y1'] + int((primary['y2']-primary['y1']) * 0.15)
                dx = cx - tx
                dy = cy - ty
                locked = abs(dx) <= DEADZONE_PX and abs(dy) <= DEADZONE_PX

                if locked:
                    pan1.stop(); pan2.stop(); tilt.stop()

                    if now - last_shoot_t >= SHOOT_COOLDOWN:
                        print(f"[LOCKED] FIRING  err=({dx:+d},{dy:+d})")
                        shoot.start(speed=100)
                        time.sleep(SHOOT_DURATION)
                        shoot.stop()
                        last_shoot_t = now
                else:
                    # Pan — pulse step
                    if abs(dx) > DEADZONE_PX:
                        mag = int(max(PAN_MIN_SPEED, min(PAN_MAX_SPEED, abs(dx) * PAN_KP)))
                        spd = PAN_DIRECTION * (1 if dx > 0 else -1) * mag
                        pan1.start(speed=spd)
                        pan2.start(speed=spd)

                    # Tilt — pulse step
                    if abs(dy) > DEADZONE_PX:
                        mag = int(max(TILT_MIN_SPEED, min(TILT_MAX_SPEED, abs(dy) * TILT_KP)))
                        sign = -1 if dy > 0 else 1
                        spd = TILT_DIRECTION * sign * mag
                        tilt.start(speed=spd)

                    time.sleep(STEP_DURATION)
                    pan1.stop(); pan2.stop(); tilt.stop()

                    pd = "R" if dx>0 else "L"
                    td = "D" if dy>0 else "U"
                    print(f"[AIM]  pan={pd} tilt={td}  err=({dx:+4d},{dy:+4d})  {primary['conf']*100:.0f}%")
            else:
                if target_visible:
                    print("[LOST]")
                    target_visible = False
                pan1.stop(); pan2.stop(); tilt.stop()
                if frame_num % 30 == 0:
                    print(f"[SCAN]  FPS={fps_ema:.1f}")

            # Annotate and push to stream
            draw(frame, detections, primary, tx, ty, fps_ema, locked if primary else False)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with _frame_lock:
                _latest_jpeg = jpeg.tobytes()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pan1.stop(); pan2.stop(); tilt.stop(); shoot.stop()
        print("[STOP] Done.")


def main():
    t = threading.Thread(target=turret_loop, daemon=True)
    t.start()
    print(f"[Flask] Starting stream on port {OUTPUT_PORT}...")
    app.run(host="0.0.0.0", port=OUTPUT_PORT, threaded=True)


if __name__ == "__main__":
    main()
