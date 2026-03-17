"""
Turret auto-aim script for Raspberry Pi 5 + AI Camera + Build HAT (SPIKE motors).

How to configure (top of file):
- MOTOR PORTS: set PAN_MOTOR_PORT / TILT_MOTOR_PORT
- TURN DISTANCE: tune PAN_DEGREES_PER_PIXEL / TILT_DEGREES_PER_PIXEL
- DIRECTION: flip PAN_DIRECTION or TILT_DIRECTION if movement is reversed

Notes:
- This script processes camera frames in RAM only.
- It does not save photos to disk, so there is nothing to clean up.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
from buildhat import Motor
from picamera2 import Picamera2


@dataclass
class Config:
	# -------- Motor Ports (Build HAT) --------
	# Use one of: "A", "B", "C", "D"
	PAN_MOTOR_PORT: str = "A"
	TILT_MOTOR_PORT: Optional[str] = "B"  # Set to None if you only have pan

	# -------- Turn Tuning (make this obvious + easy to change) --------
	# Degrees moved for each pixel of target error from center.
	PAN_DEGREES_PER_PIXEL: float = 0.08
	TILT_DEGREES_PER_PIXEL: float = 0.06

	# Minimum/maximum correction step each move command can make.
	MIN_MOVE_DEGREES: int = 1
	MAX_MOVE_DEGREES: int = 20

	# Motor speed used for run_for_degrees (0-100)
	MOTOR_SPEED: int = 40

	# Reverse these if your turret moves the wrong way.
	PAN_DIRECTION: int = 1
	TILT_DIRECTION: int = 1

	# -------- Aiming Behaviour --------
	DEADZONE_PIXELS: int = 25
	COMMAND_COOLDOWN_SECONDS: float = 0.12

	# -------- Camera --------
	FRAME_WIDTH: int = 640
	FRAME_HEIGHT: int = 480
	SHOW_PREVIEW_WINDOW: bool = True


class TurretController:
	def __init__(self, cfg: Config) -> None:
		self.cfg = cfg
		self.pan = Motor(cfg.PAN_MOTOR_PORT)
		self.tilt = Motor(cfg.TILT_MOTOR_PORT) if cfg.TILT_MOTOR_PORT else None
		self.last_pan_command_time = 0.0
		self.last_tilt_command_time = 0.0

	def _scaled_move(self, pixel_error: int, degrees_per_pixel: float) -> int:
		raw_degrees = int(abs(pixel_error) * degrees_per_pixel)
		clamped = max(self.cfg.MIN_MOVE_DEGREES, raw_degrees)
		return min(self.cfg.MAX_MOVE_DEGREES, clamped)

	def aim(self, error_x: int, error_y: int) -> None:
		now = time.time()

		if abs(error_x) > self.cfg.DEADZONE_PIXELS:
			if now - self.last_pan_command_time >= self.cfg.COMMAND_COOLDOWN_SECONDS:
				pan_step = self._scaled_move(error_x, self.cfg.PAN_DEGREES_PER_PIXEL)
				pan_sign = 1 if error_x > 0 else -1
				pan_degrees = self.cfg.PAN_DIRECTION * pan_sign * pan_step
				self.pan.run_for_degrees(
					pan_degrees,
					speed=self.cfg.MOTOR_SPEED,
					blocking=False,
				)
				self.last_pan_command_time = now

		if self.tilt and abs(error_y) > self.cfg.DEADZONE_PIXELS:
			if now - self.last_tilt_command_time >= self.cfg.COMMAND_COOLDOWN_SECONDS:
				tilt_step = self._scaled_move(error_y, self.cfg.TILT_DEGREES_PER_PIXEL)
				# Image Y grows downward: positive error means target is lower than center.
				tilt_sign = -1 if error_y > 0 else 1
				tilt_degrees = self.cfg.TILT_DIRECTION * tilt_sign * tilt_step
				self.tilt.run_for_degrees(
					tilt_degrees,
					speed=self.cfg.MOTOR_SPEED,
					blocking=False,
				)
				self.last_tilt_command_time = now

	def stop(self) -> None:
		self.pan.stop()
		if self.tilt:
			self.tilt.stop()


def find_target_center(frame, face_detector: cv2.CascadeClassifier):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(40, 40),
	)
	if len(faces) == 0:
		return None

	# Track the largest face as the main target.
	x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
	cx = x + w // 2
	cy = y + h // 2
	return (cx, cy, (x, y, w, h))


def draw_overlay(frame, frame_center, target_info):
	cx, cy = frame_center
	cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
	cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)

	if target_info:
		tx, ty, (x, y, w, h) = target_info
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.circle(frame, (tx, ty), 5, (0, 255, 0), -1)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Auto-aim turret with Pi camera + Build HAT")
	parser.add_argument(
		"--headless",
		action="store_true",
		help="Run without OpenCV preview window (recommended over SSH).",
	)
	parser.add_argument(
		"--no-tilt",
		action="store_true",
		help="Disable tilt motor control and only use pan.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = Config()

	if args.no_tilt:
		cfg.TILT_MOTOR_PORT = None

	if args.headless:
		cfg.SHOW_PREVIEW_WINDOW = False
	else:
		has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
		if not has_display:
			cfg.SHOW_PREVIEW_WINDOW = False
			print("No display detected: running headless (no preview window).")

	cv2_data = getattr(cv2, "data", None)
	if cv2_data and hasattr(cv2_data, "haarcascades"):
		detector_path = cv2_data.haarcascades + "haarcascade_frontalface_default.xml"
	else:
		detector_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
	face_detector = cv2.CascadeClassifier(detector_path)
	if face_detector.empty():
		raise RuntimeError("Failed to load Haar cascade. Check OpenCV installation.")

	picam2 = Picamera2()
	config = picam2.create_preview_configuration(
		main={"size": (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT), "format": "RGB888"}
	)
	picam2.configure(config)
	picam2.start()
	time.sleep(0.5)

	turret = TurretController(cfg)
	frame_center = (cfg.FRAME_WIDTH // 2, cfg.FRAME_HEIGHT // 2)

	try:
		while True:
			frame = picam2.capture_array()

			target_info = find_target_center(frame, face_detector)
			if target_info:
				tx, ty, _ = target_info
				error_x = tx - frame_center[0]
				error_y = ty - frame_center[1]
				turret.aim(error_x, error_y)

			if cfg.SHOW_PREVIEW_WINDOW:
				draw_overlay(frame, frame_center, target_info)
				cv2.imshow("Turret Aim", frame)
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break
	except KeyboardInterrupt:
		pass
	finally:
		turret.stop()
		picam2.stop()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
