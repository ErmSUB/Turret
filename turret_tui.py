"""
turret_tui.py — USB webcam + YOLO + motors + framebuffer display.

Run on the Pi TTY (no X needed):
  python3 turret_tui.py

Ctrl+C to quit.

Ports:
  A = Tilt   B = Pan (motor 1)   C = Pan (motor 2)   D = Shoot
"""

import glob
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO
from buildhat import Motor

# ==============================================================================
# CONFIGURE
# ==============================================================================

PAN_PORT_1 = "B"
PAN_PORT_2 = "C"
TILT_PORT  = "A"
SHOOT_PORT = "D"

PAN_DIRECTION  = 1
TILT_DIRECTION = -1

MOTOR_SPEED = 100

# Pan proportional control
PAN_KP        = 0.5
PAN_MIN_SPEED = 20
PAN_MAX_SPEED = 40

# Tilt proportional control — |dy| px * TILT_KP = speed, clamped
TILT_KP        = 0.10
TILT_MIN_SPEED = 5
TILT_MAX_SPEED = 15

DEADZONE_PX = 120

# Step mode â short pulse then pause for next frame
STEP_DURATION = 0.05

SHOOT_COOLDOWN = 2.0
SHOOT_DURATION = 5.0

CONFIDENCE           = 0.45
GRAVITY_OFFSET       = 0.0

# ==============================================================================
# Framebuffer
# ==============================================================================

FB_DEV = "/dev/fb0"

def fb_init():
    w = int(open("/sys/class/graphics/fb0/virtual_size").read().split(",")[0])
    h = int(open("/sys/class/graphics/fb0/virtual_size").read().split(",")[1])
    bpp = int(open("/sys/class/graphics/fb0/bits_per_pixel").read().strip())
    stride = int(open("/sys/class/graphics/fb0/stride").read().strip())
    fb = open(FB_DEV, "wb")
    return fb, w, h, bpp, stride

def fb_write(fb, frame, fb_w, fb_h, stride):
    fh, fw = frame.shape[:2]
    # Scale frame to fit screen, centered
    scale = min(fb_w / fw, fb_h / fh)
    nw = int(fw * scale)
    nh = int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))

    # Create black canvas at screen size
    canvas = np.zeros((fb_h, fb_w, 3), dtype=np.uint8)
    y_off = (fb_h - nh) // 2
    x_off = (fb_w - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

    # BGR -> RGB565: RRRRRGGG GGGBBBBB
    b = canvas[:, :, 0].astype(np.uint16)
    g = canvas[:, :, 1].astype(np.uint16)
    r = canvas[:, :, 2].astype(np.uint16)
    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)

    # Pad rows to match stride if needed
    bytes_per_px = 2
    row_bytes = fb_w * bytes_per_px
    if stride > row_bytes:
        pad = stride - row_bytes
        raw = rgb565.astype("<u2").tobytes()
        padded = b""
        for y in range(fb_h):
            padded += raw[y * row_bytes:(y + 1) * row_bytes] + b"\x00" * pad
        fb.seek(0)
        fb.write(padded)
    else:
        fb.seek(0)
        fb.write(rgb565.astype("<u2").tobytes())
    fb.flush()

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
        text(frame, f"Trainee {primary['conf']*100:.0f}%", (primary['x1'], max(primary['y1']-6,14)), 0.55, (0,0,220), 2)
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
        status, col = "** LOCKED - ACTIVATING **", (0,0,220)
    else:
        status, col = "ACQUIRING", (255,255,255)

    (sw,_),_ = cv2.getTextSize(status, FONT, 0.75, 2)
    text(frame, status, ((w-sw)//2, h-14), 0.75, col, 2)

    if locked:
        shoot_s = "Activated"
        scale = 2.2
        (lw,lh),_ = cv2.getTextSize(shoot_s, FONT, scale, 4)
        text(frame, shoot_s, ((w-lw)//2, h//2+lh//2), scale, (0,0,220), 4)


# ==============================================================================
# Main
# ==============================================================================

def find_usb_camera():
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        if "usb" in os.path.realpath(path).lower():
            try:
                return int(os.path.basename(path).replace("video",""))
            except ValueError:
                continue
    return 0


def main():
    # Framebuffer
    print("[FB] Opening framebuffer...")
    fb, fb_w, fb_h, bpp, stride = fb_init()
    print(f"[FB] {fb_w}x{fb_h} @ {bpp}bpp, stride={stride}")

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
    print("[Display] Rendering to framebuffer. Ctrl+C to quit.")

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
                        shoot.run_for_rotations(1)
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

            # Draw and blast to framebuffer
            draw(frame, detections, primary, tx, ty, fps_ema, locked if primary else False)
            fb_write(fb, frame, fb_w, fb_h, stride)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pan1.stop(); pan2.stop(); tilt.stop(); shoot.stop()
        fb.close()
        print("\n[STOP] Done.")


if __name__ == "__main__":
    main()
