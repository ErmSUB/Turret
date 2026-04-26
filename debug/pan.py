"""pan.py — spin pan motor (port B) at full speed until Ctrl+C."""

import time
from buildhat import Motor

PAN_PORT = "B"
SPEED = 100  # -100..100

def main():
    print("[Motor] Initialising...")
    time.sleep(2)
    pan = Motor(PAN_PORT)
    print(f"[Motor] Pan on {PAN_PORT} @ speed={SPEED}. Ctrl+C to stop.")
    try:
        pan.start(speed=SPEED)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pan.stop()
        print("[Motor] Stopped.")

if __name__ == "__main__":
    main()
