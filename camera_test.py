"""Camera test client for the Traffic Violation Detection System.

Captures frames from a Dahua ITC352 (or any RTSP/USB) camera and sends them to
the running API's POST /api/v1/analyze endpoint, together with violation
metadata (speed / speed_limit) so the rule engine flags an over-speeding
violation in 'pending_human_review' state.

------------------------------------------------------------------------------
HOW TO RUN (on a machine on the SAME network as the camera)
------------------------------------------------------------------------------
1. Make sure the API + DB are running:
       docker-compose up --build
   (The API must be reachable at API_URL below, e.g. http://localhost:8000)

2. Install the client dependencies:
       pip install opencv-python requests

3. Edit the CONFIG section below: fill in your camera IP / user / password,
   and set TEST_SPEED / SPEED_LIMIT to the values you want to test.

4. Run:
       python camera_test.py                # single snapshot
       python camera_test.py --loop 5       # capture every 5 seconds
       python camera_test.py --image car.jpg  # test with a local image file
------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import base64
import sys
import time

import requests

try:
    import cv2  # opencv-python
except ImportError:  # pragma: no cover
    cv2 = None


# ============================================================================
# CONFIG  --  EDIT THESE PLACEHOLDERS
# ============================================================================
# --- Camera (Dahua ITC352) RTSP connection ---------------------------------
CAMERA_IP = "192.168.1.108"        # <-- your camera's IP address
CAMERA_USER = "admin"              # <-- your camera username
CAMERA_PASSWORD = "YOUR_PASSWORD"  # <-- your camera password
CAMERA_RTSP_PORT = 554
CAMERA_CHANNEL = 1
# subtype 0 = main (high-res) stream, 1 = sub (low-res) stream
CAMERA_SUBTYPE = 0

# Dahua standard RTSP URL template.
RTSP_URL = (
    f"rtsp://{CAMERA_USER}:{CAMERA_PASSWORD}@{CAMERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?channel={CAMERA_CHANNEL}&subtype={CAMERA_SUBTYPE}"
)

# --- API endpoint -----------------------------------------------------------
API_URL = "http://localhost:8000/api/v1/analyze"
REVIEW_URL = "http://localhost:8000/api/v1/review"

# --- Violation test metadata (mode B) ---------------------------------------
# These are sent to the rule engine. Since 80 > 60 a violation is flagged.
TEST_SPEED = 80.0       # measured speed (km/h)
SPEED_LIMIT = 60.0      # legal limit (km/h)
TEST_LOCATION = "Test Camera ITC352 - Lane 1"
# ============================================================================


def grab_frame_from_camera() -> "bytes":
    """Open the RTSP stream, grab one frame, return JPEG bytes."""
    if cv2 is None:
        sys.exit("OpenCV not installed. Run: pip install opencv-python")

    print(f"[camera] Connecting to RTSP stream at {CAMERA_IP} ...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(
            "[camera] ERROR: could not open the RTSP stream.\n"
            "  - Check CAMERA_IP / CAMERA_USER / CAMERA_PASSWORD.\n"
            "  - Confirm the camera and this machine are on the same network.\n"
            f"  - Tried URL: {RTSP_URL}"
        )

    # Read a few frames to let the stream stabilise, then keep the last one.
    frame = None
    for _ in range(5):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()

    if frame is None:
        sys.exit("[camera] ERROR: connected but failed to read a frame.")

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        sys.exit("[camera] ERROR: failed to encode frame to JPEG.")
    print("[camera] Frame captured.")
    return buf.tobytes()


def read_image_file(path: str) -> bytes:
    """Read a local image file as bytes (useful when no camera is available)."""
    with open(path, "rb") as fh:
        return fh.read()


def send_to_api(image_bytes: bytes) -> dict:
    """Base64-encode the frame and POST it to /api/v1/analyze."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "image_base64": b64,
        "metadata": {
            "speed": TEST_SPEED,
            "speed_limit": SPEED_LIMIT,
            "location": TEST_LOCATION,
        },
    }
    print(f"[api] POST {API_URL}  (speed={TEST_SPEED}, limit={SPEED_LIMIT})")
    resp = requests.post(API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def print_result(result: dict) -> None:
    """Pretty-print the analysis outcome."""
    print("\n=========== ANALYSIS RESULT ===========")
    vehicles = result.get("detected_vehicles", [])
    print(f"Detected vehicles : {len(vehicles)}")
    for i, v in enumerate(vehicles, 1):
        print(
            f"  [{i}] type={v['vehicle_type']:10s} "
            f"conf={v['detection_confidence']:.2f} "
            f"plate={v.get('license_plate')} "
            f"plate_conf={v.get('plate_confidence')}"
        )
    print(f"Primary plate     : {result.get('primary_license_plate')}")
    print(f"Violation detected: {result.get('violation_detected')}")

    violation = result.get("violation")
    if violation:
        print(f"Violation ID      : {violation['id']}")
        print(f"Status            : {violation['status']}")
        print(f"Type              : {violation['violation_type']}")
        print(
            f"  --> Review it:  GET  {REVIEW_URL}/{violation['id']}\n"
            f"  --> Approve  :  POST {REVIEW_URL}/{violation['id']} "
            f'{{"decision":"approved","reviewed_by":"operator","fine_amount":150}}'
        )
    print(f"Message           : {result.get('message')}")
    print("=======================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera test client for TVDS")
    parser.add_argument(
        "--image",
        help="Use a local image file instead of the live camera.",
        default=None,
    )
    parser.add_argument(
        "--loop",
        type=float,
        default=0,
        help="Capture every N seconds in a loop (0 = single shot).",
    )
    args = parser.parse_args()

    def one_cycle() -> None:
        if args.image:
            image_bytes = read_image_file(args.image)
        else:
            image_bytes = grab_frame_from_camera()
        try:
            result = send_to_api(image_bytes)
            print_result(result)
        except requests.RequestException as exc:
            print(f"[api] ERROR talking to the API: {exc}")

    if args.loop and args.loop > 0:
        print(f"[loop] Capturing every {args.loop}s. Press Ctrl+C to stop.")
        try:
            while True:
                one_cycle()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\n[loop] Stopped.")
    else:
        one_cycle()


if __name__ == "__main__":
    main()
