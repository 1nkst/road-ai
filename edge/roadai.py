import cv2
import numpy as np
import base64
import requests
import threading
import time
import os
from collections import deque
from datetime import datetime
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────
API_URL     = os.getenv("ROBOFLOW_API_URL",    "http://localhost:9001")
API_KEY     = os.getenv("ROBOFLOW_API_KEY",    "T4zuK4T4cMUowqMPvSVl")
WORKSPACE   = os.getenv("ROBOFLOW_WORKSPACE",  "cheemo")
WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID","custom-workflow-2")
CLOUD_URL   = os.getenv("CLOUD_API_URL",       None)
CAMERA_IDX  = int(os.getenv("CAMERA_INDEX",   0))

INFERENCE_WIDTH  = 640
INFERENCE_HEIGHT = 360
DISPLAY_WIDTH    = 1280
DISPLAY_HEIGHT   = 720

SX = DISPLAY_WIDTH  / INFERENCE_WIDTH
SY = DISPLAY_HEIGHT / INFERENCE_HEIGHT

PX_PER_CM_W = 27.06
PX_PER_CM_H = 29.58

# ── Cost table (THB per m²) ───────────────────────────────────
REPAIR_COST = {
    "pothole":             850,
    "longitudinal crack":  320,
    "longidutional crack": 320,  # handle typo in model
    "transverse crack":    320,
    "alligator crack":     480,
    "surface damage":      560,
    "default":             400,
}

# ── Shared state ──────────────────────────────────────────────
latest_frame       = None
latest_annotations = None
output_frame       = None
latest_detections  = []
detection_history  = deque(maxlen=500)   # rolling history for session log (flat detections)
damage_events      = deque(maxlen=300)   # grouped events w/ thumbnail for the history viewer
event_counter      = 0                   # monotonic id source for events
session_start      = time.time()
lock               = threading.Lock()
running            = True
detection_mode     = "auto"              # "auto" = continuous | "manual" = capture on demand

THUMB_WIDTH  = 360                       # stored thumbnail size for history viewer
THUMB_HEIGHT = 203

app = Flask(__name__)
CORS(app)


# ── Helpers ───────────────────────────────────────────────────
def encode_frame(frame):
    small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
    _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode("utf-8")


def make_thumbnail(frame, outputs):
    """Annotated low-res thumbnail (base64 data-URI) stored with each damage event."""
    annotated = draw_annotations(frame.copy(), outputs)
    thumb = cv2.resize(annotated, (THUMB_WIDTH, THUMB_HEIGHT))
    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def build_event(frame, outputs, detections, source):
    """Group a frame's detections into a single event with a thumbnail + id.
    Must be called while holding `lock` is NOT required (it locks internally)."""
    global event_counter
    if not detections:
        return None

    # Choose a representative detection (prefer the one with a real class over 'Segmentation')
    primary = next((d for d in detections if d.get("class") != "Segmentation"), detections[0])

    # Aggregate cost/area across the detections in this event
    total_cost = sum(d.get("cost_thb", 0) for d in detections)
    area_m2    = max((d.get("seg_area_m2") or d.get("area_m2") or 0) for d in detections)
    sev_rank   = {"low": 1, "medium": 2, "high": 3}
    worst_sev  = max((d.get("severity", "low") for d in detections), key=lambda s: sev_rank.get(s, 0))

    with lock:
        event_counter += 1
        eid = event_counter

    event = {
        "id":         eid,
        "class":      primary.get("class", "Unknown"),
        "source":     source,
        "timestamp":  time.time(),
        "confidence": primary.get("confidence", 0),
        "severity":   worst_sev,
        "cost_thb":   total_cost,
        "area_m2":    round(area_m2, 4),
        "thumbnail":  make_thumbnail(frame, outputs),
        "detections": detections,
    }
    return event


def get_cost_per_m2(class_name):
    key = class_name.lower()
    for k, v in REPAIR_COST.items():
        if k in key:
            return v
    return REPAIR_COST["default"]


def calculate_metrics(pred_bbox=None, pred_seg=None):
    metrics = {}
    if pred_bbox:
        w = pred_bbox["width"]
        h = pred_bbox["height"]
        metrics["bbox_width_cm"]  = round(w / PX_PER_CM_W, 2)
        metrics["bbox_height_cm"] = round(h / PX_PER_CM_H, 2)
        metrics["bbox_area_cm2"]  = round((w / PX_PER_CM_W) * (h / PX_PER_CM_H), 2)

    if pred_seg and "points" in pred_seg:
        points = np.array([[p["x"], p["y"]] for p in pred_seg["points"]], dtype=np.float32)
        area   = float(cv2.contourArea(points))
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        area_cm2 = round(area / (PX_PER_CM_W * PX_PER_CM_H), 2)
        area_m2  = round(area_cm2 / 10000, 4)
        metrics["seg_width_cm"]  = round(w / PX_PER_CM_W, 2)
        metrics["seg_height_cm"] = round(h / PX_PER_CM_H, 2)
        metrics["seg_area_cm2"]  = area_cm2
        metrics["seg_area_m2"]   = area_m2

    return metrics


def estimate_cost(detection):
    """Estimate repair cost in THB based on damage class and area."""
    cls      = detection.get("class", "")
    area_m2  = detection.get("seg_area_m2", None)

    if area_m2 is None:
        # Fall back to bbox area
        area_cm2 = detection.get("bbox_area_cm2", 0)
        area_m2  = area_cm2 / 10000

    cost_per_m2 = get_cost_per_m2(cls)
    total_cost  = round(area_m2 * cost_per_m2)
    return {"cost_thb": total_cost, "cost_per_m2": cost_per_m2, "area_m2": round(area_m2, 4)}


def severity(detection):
    """Return severity level based on area."""
    area_cm2 = detection.get("seg_area_cm2") or detection.get("bbox_area_cm2", 0)
    if area_cm2 < 100:   return "low"
    if area_cm2 < 500:   return "medium"
    return "high"


def upload_to_cloud(detections):
    if not CLOUD_URL or not detections:
        return
    try:
        requests.post(
            f"{CLOUD_URL}/detections",
            json={"detections": detections},
            timeout=3,
        )
    except Exception as e:
        print(f"[Cloud] {e}")


# ── Inference core (shared by auto worker + manual capture) ──
def infer_frame(frame, source="auto"):
    """Run inference on a single frame.
    Returns (outputs, detections) — outputs for drawing, detections with metrics."""
    url = f"{API_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    payload = {
        "api_key":  API_KEY,
        "inputs":   {"image": {"type": "base64", "value": encode_frame(frame)}},
        "use_cache": False,
    }
    response = requests.post(url, json=payload, timeout=10)
    if not response.ok:
        return None, []

    result  = response.json()
    outputs = result.get("outputs", [{}])[0]
    detections = []

    bbox_preds = outputs.get("predictions",         {}).get("predictions", [])
    seg_preds  = outputs.get("model_1_predictions", {}).get("predictions", [])

    for pred in seg_preds:
        for p in pred["points"]:
            p["x"] *= SX
            p["y"] *= SY
    for pred in bbox_preds:
        pred["x"]      *= SX
        pred["y"]      *= SY
        pred["width"]  *= SX
        pred["height"] *= SY

    bbox    = bbox_preds[0] if bbox_preds else None
    seg     = seg_preds[0]  if seg_preds  else None
    metrics = calculate_metrics(pred_bbox=bbox, pred_seg=seg)

    if bbox:
        det = {
            "class":      bbox["class"],
            "confidence": round(bbox["confidence"], 3),
            "type":       "bbox",
            "source":     source,
            "timestamp":  time.time(),
            **metrics,
        }
        det["severity"] = severity(det)
        det.update(estimate_cost(det))
        detections.append(det)

    if seg:
        det = {
            "class":      "Segmentation",
            "confidence": round(seg["confidence"], 3),
            "type":       "seg",
            "source":     source,
            "timestamp":  time.time(),
            **metrics,
        }
        det["severity"] = severity(det)
        det.update(estimate_cost(det))
        detections.append(det)

    return outputs, detections


# ── Inference worker (AUTO mode only) ─────────────────────────
def inference_worker():
    global latest_annotations, latest_detections

    while running:
        # In MANUAL mode the worker idles — detection only happens via /capture
        if detection_mode != "auto":
            time.sleep(0.1)
            continue

        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.005)
            continue

        try:
            outputs, detections = infer_frame(frame, source="auto")
            if outputs is None:
                continue

            event = build_event(frame, outputs, detections, "auto") if detections else None

            with lock:
                latest_annotations = outputs
                latest_detections  = detections
                if detections:
                    detection_history.extend(detections)
                if event:
                    damage_events.append(event)

            if detections and CLOUD_URL:
                threading.Thread(target=upload_to_cloud, args=(detections,), daemon=True).start()

        except Exception as e:
            print(f"[Inference] {e}")


# ── Annotation drawing ────────────────────────────────────────
def draw_annotations(frame, annotations):
    if not annotations:
        return frame

    for pred in annotations.get("model_1_predictions", {}).get("predictions", []):
        points  = np.array([[int(p["x"]), int(p["y"])] for p in pred["points"]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    for pred in annotations.get("predictions", {}).get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{pred['class']} {pred['confidence']:.0%}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


# ── Camera worker ─────────────────────────────────────────────
def camera_worker():
    global latest_frame, output_frame
    cap = cv2.VideoCapture(CAMERA_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            latest_frame = frame.copy()
            annotations  = latest_annotations
        display = draw_annotations(frame.copy(), annotations)
        with lock:
            output_frame = display.copy()
    cap.release()


# ── Flask routes ──────────────────────────────────────────────
def generate_stream():
    while True:
        with lock:
            frame = output_frame.copy() if output_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(1 / 30)


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def detections():
    with lock:
        dets = latest_detections.copy()
    return jsonify({"detections": dets})


@app.route("/history")
def history():
    """Full detection history for this session."""
    with lock:
        hist = list(detection_history)
    return jsonify({"history": hist, "count": len(hist)})


@app.route("/stats")
def stats():
    """Session summary stats for the webapp dashboard."""
    with lock:
        hist = list(detection_history)

    uptime = round(time.time() - session_start)
    total  = len(hist)

    # Count by class
    class_counts = {}
    for d in hist:
        cls = d.get("class", "Unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Total cost estimate
    total_cost = sum(d.get("cost_thb", 0) for d in hist)

    # Total damaged area
    total_area_m2 = sum(d.get("seg_area_m2", 0) or d.get("area_m2", 0) for d in hist)

    # Severity breakdown
    sev_counts = {"low": 0, "medium": 0, "high": 0}
    for d in hist:
        s = d.get("severity", "low")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    # Avg confidence
    confs = [d.get("confidence", 0) for d in hist]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0

    return jsonify({
        "uptime_seconds":  uptime,
        "total_detections": total,
        "class_counts":    class_counts,
        "severity_counts": sev_counts,
        "total_cost_thb":  total_cost,
        "total_area_m2":   round(total_area_m2, 4),
        "avg_confidence":  avg_conf,
        "session_start":   session_start,
    })


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear detection history — call when starting a new survey."""
    global session_start
    with lock:
        detection_history.clear()
        damage_events.clear()
        session_start = time.time()
    return jsonify({"cleared": True})


@app.route("/mode", methods=["GET", "POST"])
def mode():
    """Get or set detection mode: 'auto' (continuous) or 'manual' (capture on demand)."""
    global detection_mode, latest_annotations, latest_detections

    if request.method == "POST":
        new_mode = (request.get_json(silent=True) or {}).get("mode", "").lower()
        if new_mode not in ("auto", "manual"):
            return jsonify({"error": "mode must be 'auto' or 'manual'"}), 400
        with lock:
            detection_mode = new_mode
            # Clear stale overlays when entering manual mode
            if new_mode == "manual":
                latest_annotations = None
                latest_detections  = []
        print(f"[Mode] switched to {new_mode.upper()}")

    return jsonify({"mode": detection_mode})


@app.route("/events")
def events_list():
    """Damage history for the viewer — lightweight list WITHOUT thumbnails by default.
    Pass ?include_thumb=1 to embed thumbnails (heavier)."""
    include_thumb = request.args.get("include_thumb") == "1"
    with lock:
        evs = list(damage_events)

    items = []
    for e in evs:
        item = {
            "id":         e["id"],
            "class":      e["class"],
            "source":     e["source"],
            "timestamp":  e["timestamp"],
            "confidence": e["confidence"],
            "severity":   e["severity"],
            "cost_thb":   e["cost_thb"],
            "area_m2":    e["area_m2"],
            "num_detections": len(e["detections"]),
        }
        if include_thumb:
            item["thumbnail"] = e["thumbnail"]
        items.append(item)

    items.reverse()  # newest first
    return jsonify({"events": items, "count": len(items)})


@app.route("/event/<int:event_id>")
def event_detail(event_id):
    """Full detail for a single damage event, including thumbnail + all detections."""
    with lock:
        ev = next((e for e in damage_events if e["id"] == event_id), None)
    if ev is None:
        return jsonify({"error": "event not found"}), 404
    return jsonify(ev)


@app.route("/event/<int:event_id>/delete", methods=["POST"])
def event_delete(event_id):
    """Remove an event (e.g. operator flags it as a false positive)."""
    with lock:
        before = len(damage_events)
        kept = [e for e in damage_events if e["id"] != event_id]
        damage_events.clear()
        damage_events.extend(kept)
        removed = before - len(damage_events)
    return jsonify({"deleted": removed > 0, "id": event_id})


@app.route("/capture", methods=["POST"])
def capture():
    """MANUAL mode: freeze the current frame, run detection on it once,
    store results in history, and return the annotated frame + detections."""
    global latest_annotations, latest_detections

    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "no frame available from camera"}), 503

    try:
        outputs, detections = infer_frame(frame, source="manual")
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 502

    if outputs is None:
        return jsonify({"error": "inference server returned an error"}), 502

    # Store in history (auto-add to database as requested)
    event = build_event(frame, outputs, detections, "manual") if detections else None
    with lock:
        latest_annotations = outputs
        latest_detections  = detections
        if detections:
            detection_history.extend(detections)
        if event:
            damage_events.append(event)

    if detections and CLOUD_URL:
        threading.Thread(target=upload_to_cloud, args=(detections,), daemon=True).start()

    # Return the annotated captured frame so the UI can show the frozen result
    annotated = draw_annotations(frame.copy(), outputs)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "captured":   True,
        "detections": detections,
        "count":      len(detections),
        "event_id":   event["id"] if event else None,
        "frame":      f"data:image/jpeg;base64,{frame_b64}",
    })


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=camera_worker,    daemon=True).start()

    print("=" * 50)
    print("RoadAI running at http://localhost:5000")
    print(f"Docker inference: {API_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}")
    print(f"Camera index:     {CAMERA_IDX}")
    print(f"Cloud upload:     {'→ ' + CLOUD_URL if CLOUD_URL else 'disabled'}")
    print("Endpoints:")
    print("  /video_feed  — live stream")
    print("  /detections  — latest frame detections")
    print("  /history     — full session log")
    print("  /stats       — session summary")
    print("  /clear       — reset session (POST)")
    print("  /mode        — get/set AUTO|MANUAL mode (GET/POST)")
    print("  /capture     — manual capture & detect (POST)")
    print("  /events      — damage history list (GET)")
    print("  /event/<id>  — full event detail (GET)")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, threaded=True)