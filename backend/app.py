import os
import time
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from GitHub Pages

# ── In-memory store (resets on redeploy — fine for prototype) ──
# Max 200 detections kept in memory at once
MAX_DETECTIONS = 200
detections_store = deque(maxlen=MAX_DETECTIONS)

# Simple session stats
stats = {
    "total_detections": 0,
    "session_start": time.time(),
}


# ── Health check ───────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({
        "service": "ROAD AI Backend",
        "status": "running",
        "uptime_seconds": round(time.time() - stats["session_start"]),
        "total_detections": stats["total_detections"],
    })


# ── Receive detections from roadai.py ─────────────────────────
# roadai.py will POST to this endpoint instead of keeping data local
@app.route("/detections", methods=["POST"])
def receive_detections():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    detections = data.get("detections", [])
    if not isinstance(detections, list):
        return jsonify({"error": "detections must be a list"}), 400

    timestamp = time.time()
    gps = data.get("gps", None)  # optional GPS: {"lat": 13.xx, "lng": 100.xx}

    for det in detections:
        det["timestamp"] = timestamp
        det["gps"] = gps
        detections_store.append(det)

    stats["total_detections"] += len(detections)

    return jsonify({"received": len(detections), "total": stats["total_detections"]}), 200


# ── Serve detections to the webapp ────────────────────────────
@app.route("/detections", methods=["GET"])
def get_detections():
    # Optional: return only detections from last N seconds
    since = request.args.get("since", type=float, default=None)

    result = list(detections_store)
    if since:
        result = [d for d in result if d.get("timestamp", 0) >= since]

    # Return most recent 50 for the live dashboard
    recent = result[-50:] if len(result) > 50 else result
    recent.reverse()  # newest first

    return jsonify({
        "detections": recent,
        "total": stats["total_detections"],
        "uptime_seconds": round(time.time() - stats["session_start"]),
    })


# ── Clear detections (useful for starting a new survey session) ──
@app.route("/detections", methods=["DELETE"])
def clear_detections():
    detections_store.clear()
    stats["total_detections"] = 0
    stats["session_start"] = time.time()
    return jsonify({"cleared": True})


# ── Stats endpoint ─────────────────────────────────────────────
@app.route("/stats")
def get_stats():
    return jsonify({
        "total_detections": stats["total_detections"],
        "stored_detections": len(detections_store),
        "uptime_seconds": round(time.time() - stats["session_start"]),
        "session_start": stats["session_start"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)