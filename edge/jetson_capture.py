"""
ROAD AI — Jetson capture script
Captures frames from the Jetson's camera and POSTs them to the laptop's
roadai.py server (which runs local, free inference). GPS coordinates from the
NEO-6M are attached to each frame. Run: python3 jetson_capture.py

Set LAPTOP_IP via env var to match wherever the laptop currently is
(e.g. its hotspot/Wi-Fi IP) — it changes depending on the network.
"""

import os
import time
import cv2
import requests
import gps_reader

LAPTOP_IP        = os.getenv("LAPTOP_IP", "172.20.10.3")
LAPTOP_PORT      = os.getenv("LAPTOP_PORT", "5000")
SERVER_URL       = f"http://{LAPTOP_IP}:{LAPTOP_PORT}/upload"
CAPTURE_INTERVAL = 1.0
JPEG_QUALITY     = 85
CAP_WIDTH        = 1280
CAP_HEIGHT       = 720


def open_camera():
    for index in (0, 1):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Camera opened on index {index}")
            return cap
        cap.release()
    raise RuntimeError("Could not open camera on index 0 or 1")


def capture_and_send(cap):
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        print("WARNING: failed to read frame — skipping")
        return

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        print("WARNING: JPEG encode failed — skipping")
        return

    jpeg_bytes = buf.tobytes()

    lat, lng = gps_reader.get_location()
    data_fields = {"frame": ("frame.jpg", jpeg_bytes, "image/jpeg")}
    extra = {}
    if lat is not None:
        extra = {"lat": str(lat), "lng": str(lng)}

    try:
        resp = requests.post(SERVER_URL, files=data_fields, data=extra, timeout=5)
        resp.raise_for_status()
        result  = resp.json()
        size_kb = len(jpeg_bytes) / 1024
        gps_str = f"GPS={lat},{lng}" if lat is not None else "GPS=no fix"
        print(f"[{time.strftime('%H:%M:%S')}] sent {size_kb:.1f} KB  {gps_str}  frame_id={result.get('frame_id')}")
    except requests.exceptions.ConnectionError:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: cannot reach {SERVER_URL} — retrying")
    except requests.exceptions.Timeout:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: request timed out — retrying")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")


def main():
    print(f"Sending frames to {SERVER_URL}")
    gps_reader.start()
    cap = open_camera()
    try:
        while True:
            start = time.monotonic()
            capture_and_send(cap)
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, CAPTURE_INTERVAL - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        gps_reader.stop()


if __name__ == "__main__":
    main()
