"""
turret_camera_test.py — Raspberry Pi 5 Turret System: Camera & Detection Test

Performs real-time person detection with bounding boxes and simulates turret
aiming commands as on-screen text overlay. No motor control — visuals only.

Supports:
  - Raspberry Pi AI Camera (IMX500) via picamera2 + MobileNet object detection
  - Laptop/USB webcam fallback via OpenCV + YOLOv8n (ultralytics)

Run:
  python turret_camera_test.py

Quit: press 'q'

Install dependencies:
  pip install opencv-python numpy ultralytics
  # On Raspberry Pi with AI Camera:
  pip install picamera2
  # or: sudo apt install python3-picamera2
"""

# ==============================================================================
# INSTALL INSTRUCTIONS (comment block)
# ==============================================================================
# pip install opencv-python numpy ultralytics
# For Raspberry Pi AI Camera support:
#   pip install picamera2
#   sudo apt install -y python3-libcamera python3-kms++
# ==============================================================================

# ==============================================================================
# CONFIGURABLE CONSTANTS — tune these for your setup
# ==============================================================================

# Pixel offset threshold for "aim is close enough" — below this, SHOOT is shown
THRESHOLD_PX = 30

# Aim point vertical offset above frame center, as a fraction of frame height.
# Aim point is offset above center to compensate for projectile drop.
# Adjust GRAVITY_OFFSET_PERCENT constant based on projectile type and range.
GRAVITY_OFFSET_PERCENT = 0.05

# Minimum detection confidence to show a bounding box (0.0–1.0)
CONFIDENCE_THRESHOLD = 0.45

# Target capture frame rate (actual FPS may vary by hardware)
TARGET_FPS = 30

# ==============================================================================

import os
import sys
import time
import abc
from typing import Optional

# Force xcb (X11) backend — OpenCV's Qt build does not include the Wayland plugin
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500
    from picamera2.devices.imx500.postprocess import softmax
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False


# ==============================================================================
# Data structures
# ==============================================================================

class Detection:
    """A single detected person."""
    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


# ==============================================================================
# Camera backends
# ==============================================================================

class CameraBackend(abc.ABC):
    """Abstract base class for camera backends."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Return a BGR numpy frame, or None on failure."""
        ...

    @abc.abstractmethod
    def detect_persons(self, frame: np.ndarray) -> list[Detection]:
        """Return list of Detection objects for persons found in frame."""
        ...

    @abc.abstractmethod
    def release(self) -> None:
        ...


# ── Raspberry Pi AI Camera (IMX500 + picamera2) ───────────────────────────────

class PiAICamera(CameraBackend):
    """
    Uses the Raspberry Pi AI Camera (Sony IMX500) with picamera2.
    Runs MobileNet SSD object detection on-chip for near-zero CPU overhead.
    """

    # COCO class index for 'person'
    _PERSON_CLASS = 0

    def __init__(self):
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 is not installed.")

        self._imx500 = IMX500("/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
        self._picam2 = Picamera2(self._imx500.camera_num)

        config = self._picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "BGR888"},
            controls={"FrameRate": TARGET_FPS},
        )
        self._picam2.configure(config)
        self._picam2.start()
        # Allow auto-exposure to settle
        time.sleep(2)

        self._frame_w = 1280
        self._frame_h = 720

    @property
    def name(self) -> str:
        return "Raspberry Pi AI Camera (IMX500 MobileNetSSD)"

    def read_frame(self) -> Optional[np.ndarray]:
        try:
            frame = self._picam2.capture_array("main")
            return frame  # already BGR888
        except Exception as e:
            print(f"[PiAICamera] Frame capture error: {e}")
            return None

    def detect_persons(self, frame: np.ndarray) -> list[Detection]:
        try:
            metadata = self._picam2.capture_metadata()
            np_outputs = self._imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is None:
                return []

            # IMX500 SSD output format: [boxes, classes, scores, count]
            boxes    = np_outputs[0][0]   # (N, 4) — y1,x1,y2,x2 normalised
            classes  = np_outputs[1][0]   # (N,)
            scores   = np_outputs[2][0]   # (N,)
            count    = int(np_outputs[3][0])

            h, w = frame.shape[:2]
            detections = []
            for i in range(count):
                if int(classes[i]) != self._PERSON_CLASS:
                    continue
                conf = float(scores[i])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                y1n, x1n, y2n, x2n = boxes[i]
                x1 = int(x1n * w); y1 = int(y1n * h)
                x2 = int(x2n * w); y2 = int(y2n * h)
                detections.append(Detection(x1, y1, x2, y2, conf))
            return detections
        except Exception as e:
            print(f"[PiAICamera] Detection error: {e}")
            return []

    def release(self) -> None:
        self._picam2.stop()
        self._picam2.close()


# ── Webcam / USB Camera (YOLOv8n via ultralytics) ────────────────────────────

class WebcamCamera(CameraBackend):
    """
    Uses a laptop or USB webcam via OpenCV.
    Runs YOLOv8n (ultralytics) for person detection on CPU/GPU.
    Falls back to OpenCV's built-in HOG person detector if ultralytics is unavailable.
    """

    def __init__(self, device_index: int = 0):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video device {device_index}.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if _YOLO_AVAILABLE:
            self._model = YOLO("yolov8n.pt")  # downloads on first run (~6 MB)
            self._hog = None
            self._backend_name = "Webcam — YOLOv8n (ultralytics)"
        else:
            self._model = None
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._backend_name = "Webcam — HOG Person Detector (built-in)"
            print("[WebcamCamera] ultralytics not found; using HOG detector.")
            print("  For better accuracy: pip install ultralytics")

    @property
    def name(self) -> str:
        return self._backend_name

    def read_frame(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def detect_persons(self, frame: np.ndarray) -> list[Detection]:
        if self._model is not None:
            return self._yolo_detect(frame)
        return self._hog_detect(frame)

    def _yolo_detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=[0])
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(Detection(x1, y1, x2, y2, conf))
        return detections

    def _hog_detect(self, frame: np.ndarray) -> list[Detection]:
        """OpenCV built-in HOG+SVM person detector — no model files needed."""
        # Detect at reduced scale for speed; winStride controls step size
        rects, weights = self._hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )
        detections = []
        for (x, y, w, h), weight in zip(rects, weights):
            conf = float(weight)
            if conf < CONFIDENCE_THRESHOLD:
                continue
            # HOG returns pixel coords directly
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            # Clip confidence to [0, 1] range for display (HOG scores can exceed 1)
            detections.append(Detection(x1, y1, x2, y2, min(conf, 1.0)))

        # Apply non-max suppression to remove overlapping boxes
        if len(detections) > 1:
            boxes  = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
            scores = np.array([d.confidence for d in detections], dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), CONFIDENCE_THRESHOLD, nms_threshold=0.4
            )
            detections = [detections[i] for i in (indices.flatten() if len(indices) else [])]

        return detections

    def release(self) -> None:
        self._cap.release()


# ==============================================================================
# Camera auto-detection
# ==============================================================================

def init_camera() -> CameraBackend:
    """Try Pi AI Camera first; fall back to webcam."""
    if _PICAMERA2_AVAILABLE:
        try:
            cam = PiAICamera()
            print(f"[Camera] Selected: {cam.name}")
            return cam
        except Exception as e:
            print(f"[Camera] Pi AI Camera unavailable ({e}), falling back to webcam.")

    try:
        cam = WebcamCamera(device_index=0)
        print(f"[Camera] Selected: {cam.name}")
        return cam
    except RuntimeError as e:
        print(f"[Camera] FATAL — no camera available: {e}")
        sys.exit(1)


# ==============================================================================
# Overlay rendering helpers
# ==============================================================================

# Colours (BGR)
GREEN  = (0,   200,   0)
RED    = (0,     0, 220)
YELLOW = (0,   220, 220)
WHITE  = (255, 255, 255)
BLACK  = (0,     0,   0)
CYAN   = (220, 220,   0)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = 0.55
_FONT_MED   = 0.75
_FONT_LARGE = 1.20
_THICK_S    = 1
_THICK_M    = 2
_THICK_L    = 3


def _put_text_outlined(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw text with a black outline for readability on any background."""
    x, y = pos
    # outline
    cv2.putText(img, text, (x - 1, y - 1), _FONT, font_scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 1, y + 1), _FONT, font_scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x - 1, y + 1), _FONT, font_scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 1, y - 1), _FONT, font_scale, BLACK, thickness + 2, cv2.LINE_AA)
    # foreground
    cv2.putText(img, text, pos, _FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_crosshair(img: np.ndarray, cx: int, cy: int, size: int = 18) -> None:
    """Draw a red targeting crosshair."""
    cv2.line(img, (cx - size, cy), (cx + size, cy), RED, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), RED, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), size // 2, RED, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2,         RED, -1)


def draw_box(
    img: np.ndarray,
    det: Detection,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, thickness, cv2.LINE_AA)
    _put_text_outlined(img, label, (det.x1, max(det.y1 - 6, 14)), _FONT_SMALL, color, _THICK_S)


# ==============================================================================
# Aim calculation
# ==============================================================================

def compute_aim(
    det: Detection,
    target_x: int,
    target_y: int,
) -> tuple[str, str, bool]:
    """
    Given the primary detection and the aim target point, return
    (h_cmd, v_cmd, locked) where locked=True means both axes are within threshold.
    """
    dx = det.cx - target_x  # positive → person is right of target
    dy = det.cy - target_y  # positive → person is below target

    if abs(dx) <= THRESHOLD_PX:
        h_cmd = ""
    elif dx > 0:
        h_cmd = "-> ROTATE RIGHT"
    else:
        h_cmd = "<- ROTATE LEFT"

    if abs(dy) <= THRESHOLD_PX:
        v_cmd = ""
    elif dy > 0:
        v_cmd = "v DEPRESS DOWN"
    else:
        v_cmd = "^ ELEVATE UP"

    locked = (h_cmd == "" and v_cmd == "")
    return h_cmd, v_cmd, locked


# ==============================================================================
# Main loop
# ==============================================================================

def main() -> None:
    camera = init_camera()
    print("[Turret Camera Test] Running — press 'q' to quit.")

    prev_time = time.time()
    fps_display = 0.0
    fps_alpha   = 0.1  # exponential moving average smoothing

    # ── Tracker state ─────────────────────────────────────────────────────────
    # Smoothed bounding box of the tracked primary target (floats).
    tracked: Optional[list[float]] = None   # [x1, y1, x2, y2]
    lost_frames = 0
    LOST_LIMIT  = 10   # frames without a match before resetting tracker
    TRACK_ALPHA = 0.4  # EMA weight for new detection (lower = smoother / slower)
    # Max distance (px) between tracked centre and new detection centre to
    # consider them the same person.
    MATCH_DIST  = 120

    while True:
        frame = camera.read_frame()
        if frame is None:
            print("[Main] Empty frame, skipping.")
            continue

        frame_h, frame_w = frame.shape[:2]

        # ── Aim target point ─────────────────────────────────────────────────
        target_x = frame_w // 2
        target_y = int(frame_h // 2 - frame_h * GRAVITY_OFFSET_PERCENT)

        # ── Detection ─────────────────────────────────────────────────────────
        detections = camera.detect_persons(frame)

        # ── Tracking: pick best detection and EMA-smooth its box ─────────────
        if detections:
            if tracked is None:
                # No existing track — initialise with the largest detection.
                best = max(detections, key=lambda d: d.area)
            else:
                # Prefer the detection closest to the current tracked centre.
                tcx = (tracked[0] + tracked[2]) / 2
                tcy = (tracked[1] + tracked[3]) / 2
                best = min(
                    detections,
                    key=lambda d: (d.cx - tcx) ** 2 + (d.cy - tcy) ** 2,
                )
                # If it's too far away, start fresh (different person appeared).
                dist = ((best.cx - tcx) ** 2 + (best.cy - tcy) ** 2) ** 0.5
                if dist > MATCH_DIST:
                    tracked = None
                    best = max(detections, key=lambda d: d.area)

            if tracked is None:
                tracked = [float(best.x1), float(best.y1),
                           float(best.x2), float(best.y2)]
            else:
                # EMA smooth toward new detection
                tracked[0] += TRACK_ALPHA * (best.x1 - tracked[0])
                tracked[1] += TRACK_ALPHA * (best.y1 - tracked[1])
                tracked[2] += TRACK_ALPHA * (best.x2 - tracked[2])
                tracked[3] += TRACK_ALPHA * (best.y2 - tracked[3])

            lost_frames = 0
        else:
            lost_frames += 1
            if lost_frames >= LOST_LIMIT:
                tracked = None

        # Build a Detection from the smoothed tracked box (for aim/draw).
        primary: Optional[Detection] = None
        if tracked is not None:
            primary = Detection(
                int(tracked[0]), int(tracked[1]),
                int(tracked[2]), int(tracked[3]),
                best.confidence if detections else 0.0,
            )

        # ── Draw bounding boxes ───────────────────────────────────────────────
        for det in detections:
            # All raw detections drawn in green (secondary)
            draw_box(frame, det, GREEN, f"PERSON  {det.confidence * 100:.0f}%", _THICK_M)

        if primary:
            # Smoothed tracked box in red (primary target)
            draw_box(frame, primary, RED, f"TARGET  {primary.confidence * 100:.0f}%", _THICK_M + 1)
            cv2.line(frame, (primary.cx, primary.cy), (target_x, target_y),
                     RED, 1, cv2.LINE_AA)

        # ── Crosshair at aim target ───────────────────────────────────────────
        draw_crosshair(frame, target_x, target_y)

        # ── Compute aim commands ──────────────────────────────────────────────
        h_cmd = v_cmd = ""
        locked = False
        if primary:
            h_cmd, v_cmd, locked = compute_aim(primary, target_x, target_y)

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.time()
        instant_fps = 1.0 / max(now - prev_time, 1e-6)
        fps_display = fps_alpha * instant_fps + (1 - fps_alpha) * fps_display
        prev_time = now

        # ══════════════════════════════════════════════════════════════════════
        # HUD OVERLAY
        # ══════════════════════════════════════════════════════════════════════

        # Top-left: camera type + FPS
        _put_text_outlined(frame, camera.name,             (10, 24),  _FONT_SMALL, WHITE, _THICK_S)
        _put_text_outlined(frame, f"FPS: {fps_display:.1f}", (10, 46), _FONT_SMALL, CYAN,  _THICK_S)

        # Top-right: persons detected count
        count_str = f"Persons: {len(detections)}"
        (tw, _), _ = cv2.getTextSize(count_str, _FONT, _FONT_SMALL, _THICK_S)
        _put_text_outlined(frame, count_str, (frame_w - tw - 12, 24), _FONT_SMALL, WHITE, _THICK_S)

        # Bottom-left: aim commands
        aim_parts = [p for p in [h_cmd, v_cmd] if p]
        aim_str   = "  ".join(aim_parts) if aim_parts else "ON TARGET"
        if not primary:
            aim_str = "NO TARGET"
        _put_text_outlined(frame, aim_str, (10, frame_h - 40), _FONT_MED, WHITE, _THICK_M)

        # Bottom-centre: status banner
        if not detections:
            status_str   = "SCANNING..."
            status_color = YELLOW
        elif locked:
            status_str   = "** TARGET LOCKED - SHOOT **"
            status_color = RED
        else:
            status_str   = "ACQUIRING TARGET"
            status_color = WHITE

        (sw, sh), _ = cv2.getTextSize(status_str, _FONT, _FONT_MED, _THICK_M)
        status_x = (frame_w - sw) // 2
        status_y = frame_h - 14
        _put_text_outlined(frame, status_str, (status_x, status_y), _FONT_MED, status_color, _THICK_M)

        # Large centred SHOOT indicator when locked
        if locked and primary:
            shoot_str = "SHOOT"
            (lw, lh), _ = cv2.getTextSize(shoot_str, _FONT, _FONT_LARGE * 1.8, _THICK_L + 1)
            shoot_x = (frame_w - lw) // 2
            shoot_y = frame_h // 2 + lh // 2
            _put_text_outlined(frame, shoot_str, (shoot_x, shoot_y),
                               _FONT_LARGE * 1.8, RED, _THICK_L + 1)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Turret Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("[Turret Camera Test] Exited cleanly.")


if __name__ == "__main__":
    main()
