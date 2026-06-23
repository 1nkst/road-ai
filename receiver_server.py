"""
ROAD AI — Laptop receiver server
Accepts JPEG frames from the Raspberry Pi and saves them locally.
Run: python receiver_server.py
"""

import os
import time
from flask import Flask, request, jsonify

SAVE_DIR = "received_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload():
    if "frame" not in request.files:
        return jsonify({"status": "error", "message": "no frame field"}), 400

    frame_file = request.files["frame"]
    data = frame_file.read()

    if not data:
        return jsonify({"status": "error", "message": "empty file"}), 400

    frame_id = int(time.time() * 1000)
    filename = os.path.join(SAVE_DIR, f"frame_{frame_id}.jpg")

    with open(filename, "wb") as f:
        f.write(data)

    size_kb = len(data) / 1024
    print(f"[{time.strftime('%H:%M:%S')}] frame_id={frame_id}  size={size_kb:.1f} KB  saved → {filename}")

    return jsonify({"status": "ok", "frame_id": frame_id})


if __name__ == "__main__":
    print(f"Saving frames to: {os.path.abspath(SAVE_DIR)}/")
    print("Server listening on 0.0.0.0:5050 ...")
    app.run(host="0.0.0.0", port=5050, debug=False)
