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

# AUTO-mode recording filters
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE",  0.60))  # ignore weak detections (< 60%)
DEDUP_COOLDOWN  = float(os.getenv("DEDUP_COOLDOWN",  5.0))   # secs to suppress same-class repeats

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
last_auto_record   = {}                  # {class_name: timestamp} for AUTO dedup cooldown

# ── Survey sessions ───────────────────────────────────────────
sessions           = []                  # all surveys (active + ended), newest appended
current_session    = None                # the active survey, or None
session_counter    = 0                   # monotonic id source for sessions

# Preset locations for local testing before GPS is wired up (Chiang Mai roads)
MOCK_LOCATIONS = [
    {"name": "ถนนห้วยแก้ว (Huay Kaew Rd)",        "lat": 18.7965, "lng": 98.9670},
    {"name": "ถนนนิมมานเหมินท์ (Nimman Rd)",       "lat": 18.7964, "lng": 98.9669},
    {"name": "ถนนช้างเผือก (Chang Phueak Rd)",     "lat": 18.8009, "lng": 98.9817},
    {"name": "ถนนสุเทพ (Suthep Rd)",               "lat": 18.7896, "lng": 98.9580},
    {"name": "ถนนเจริญเมือง (Charoen Muang Rd)",    "lat": 18.7869, "lng": 99.0089},
    {"name": "ถนนมหิดล (Mahidol Rd)",              "lat": 18.7693, "lng": 98.9870},
    {"name": "ถนนคันคลองชลประทาน (Canal Rd)",      "lat": 18.7833, "lng": 98.9520},
    {"name": "ถนนสมโภชเชียงใหม่ 700 ปี (700 Years Rd)", "lat": 18.8133, "lng": 98.9420},
]

THUMB_WIDTH  = 320                       # stored thumbnail size for history viewer
THUMB_HEIGHT = 180

app = Flask(__name__)
CORS(app, allow_headers=['Content-Type'], methods=['GET', 'POST', 'OPTIONS'])


# ── Helpers ───────────────────────────────────────────────────
def encode_frame(frame):
    small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
    _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode("utf-8")


def make_thumbnail(frame, outputs):
    """Annotated low-res thumbnail (base64 data-URI) stored with each damage event."""
    annotated = draw_annotations(frame.copy(), outputs)
    thumb = cv2.resize(annotated, (THUMB_WIDTH, THUMB_HEIGHT))
    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 42])
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


# ── Session helpers ───────────────────────────────────────────
def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two GPS points, in km."""
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return round(2 * R * math.asin(math.sqrt(a)), 4)


def advance_gps(session, step_m=10.0):
    """Simulate a GPS step ~step_m metres further along the survey path.
    Adds slight heading jitter so the track looks like a real road.
    Returns the new {lat, lng}. Replace with NEO-6M readings later.
    Caller holds `lock`."""
    import math, random
    path = session.setdefault("path", [])
    if not path:
        sl = session["start_location"]
        path.append({"lat": sl["lat"], "lng": sl["lng"]})
    last = path[-1]
    if last["lat"] is None:
        return {"lat": None, "lng": None}
    session["heading"] = session.get("heading", random.uniform(0, 2*math.pi)) + random.uniform(-0.22, 0.22)
    h = session["heading"]
    dlat = step_m * math.cos(h) / 111320.0
    dlng = step_m * math.sin(h) / (111320.0 * max(0.1, math.cos(math.radians(last["lat"]))))
    pt = {"lat": round(last["lat"] + dlat, 6), "lng": round(last["lng"] + dlng, 6)}
    path.append(pt)
    return pt


def record_event_to_session(event):
    """Append a damage event to the active session (if any) and stamp it with a
    simulated GPS position ~10 m along the track. Caller holds `lock`."""
    if current_session is not None and current_session["status"] == "active":
        pt = advance_gps(current_session, step_m=10.0)
        event["lat"] = pt["lat"]
        event["lng"] = pt["lng"]
        current_session["events"].append(event)


def summarize_session(s, include_thumbs=False):
    """Build a JSON-friendly summary of a session.
    By default omits per-event thumbnails (lightweight list view)."""
    events = s["events"]
    by_type = {}
    total_cost = 0.0
    total_area = 0.0
    for e in events:
        by_type[e["class"]] = by_type.get(e["class"], 0) + 1
        total_cost += e.get("cost_thb", 0) or 0
        total_area += e.get("area_m2", 0) or 0

    # GPS path + damage points for the map
    path = s.get("path", [])
    damage_points = [
        {"id": e["id"], "lat": e.get("lat"), "lng": e.get("lng"),
         "class": e["class"], "severity": e.get("severity", "low"),
         "cost_thb": e.get("cost_thb", 0), "confidence": e.get("confidence", 0)}
        for e in events if e.get("lat") is not None
    ]

    out = {
        "id":              s["id"],
        "status":          s["status"],
        "name":            s["name"],
        "start_time":      s["start_time"],
        "end_time":        s["end_time"],
        "start_location":  s["start_location"],
        "end_location":    s["end_location"],
        "distance_km":     s["distance_km"],
        "damage_summary":  by_type,
        "total_damage":    len(events),
        "total_cost":      round(total_cost, 2),
        "total_area_m2":   round(total_area, 4),
        "duration_sec":    round((s["end_time"] or time.time()) - s["start_time"], 1),
        "path":            path,
        "damage_points":   damage_points,
    }
    if include_thumbs:
        out["events"] = events
    return out


def filter_auto_detections(detections):
    """For AUTO mode: keep only detections that are
      (1) confident enough to be real road damage (>= MIN_CONFIDENCE), and
      (2) not a repeat of the same class seen within DEDUP_COOLDOWN seconds.
    Returns the subset of detections worth recording as a new damage event.
    Updates the per-class cooldown timestamps as a side effect."""
    now = time.time()
    kept = []
    seen_classes = set()

    for d in detections:
        cls  = d.get("class", "")
        conf = d.get("confidence", 0)

        # Segmentation entries ride along with their bbox; skip standalone seg here
        if cls == "Segmentation":
            continue

        # (1) confidence gate — ignore weak/uncertain guesses
        if conf < MIN_CONFIDENCE:
            continue

        # (2) dedup — same class recorded too recently → treat as same damage
        last = last_auto_record.get(cls, 0)
        if now - last < DEDUP_COOLDOWN:
            continue

        kept.append(d)
        seen_classes.add(cls)

    # Mark cooldown for everything we just accepted
    for cls in seen_classes:
        last_auto_record[cls] = now

    return kept


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

            # AUTO detections feed live stats unconditionally.
            # A damage EVENT is only recorded when the detection is confident
            # enough to be real road damage AND is a genuinely new damage
            # (not the same class seen within the last few seconds).
            new_damage = filter_auto_detections(detections) if detections else []
            event = build_event(frame, outputs, new_damage, "auto") if new_damage else None

            with lock:
                latest_annotations = outputs
                latest_detections  = detections
                if detections:
                    detection_history.extend(detections)
                if event and detection_mode == "auto":
                    damage_events.append(event)
                    record_event_to_session(event)

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
    try:
        with lock:
            hist = list(detection_history)

        uptime = round(time.time() - session_start)
        total  = len(hist)

        class_counts = {}
        for d in hist:
            cls = d.get("class", "Unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1

        total_cost    = sum(d.get("cost_thb", 0) or 0 for d in hist)
        total_area_m2 = sum(d.get("seg_area_m2", 0) or d.get("area_m2", 0) or 0 for d in hist)

        sev_counts = {"low": 0, "medium": 0, "high": 0}
        for d in hist:
            s = d.get("severity", "low")
            sev_counts[s] = sev_counts.get(s, 0) + 1

        confs    = [d.get("confidence", 0) for d in hist if d.get("confidence") is not None]
        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0

        return jsonify({
            "uptime_seconds":   uptime,
            "total_detections": total,
            "class_counts":     class_counts,
            "severity_counts":  sev_counts,
            "total_cost_thb":   total_cost,
            "total_area_m2":    round(total_area_m2, 4),
            "avg_confidence":   avg_conf,
            "session_start":    session_start,
        })
    except Exception as e:
        print(f"[Stats] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear detection history — call when starting a new survey."""
    global session_start
    with lock:
        detection_history.clear()
        damage_events.clear()
        last_auto_record.clear()
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
        # also remove from any session that holds it
        for s in sessions:
            s["events"] = [e for e in s["events"] if e["id"] != event_id]
    return jsonify({"deleted": removed > 0, "id": event_id})


# ── Session endpoints ─────────────────────────────────────────
@app.route("/mock-locations")
def mock_locations():
    """Preset locations for local testing before GPS is integrated."""
    return jsonify({"locations": MOCK_LOCATIONS})


@app.route("/session/current")
def session_current():
    """Status of the active survey, or null."""
    with lock:
        if current_session is None:
            return jsonify({"active": False, "session": None})
        return jsonify({"active": current_session["status"] == "active",
                        "session": summarize_session(current_session)})


@app.route("/session/start", methods=["POST"])
def session_start():
    """Begin a new survey. Body: {name, lat, lng}."""
    global current_session, session_counter
    body = request.get_json(silent=True) or {}
    name = body.get("name") or "Unnamed survey"
    lat  = body.get("lat")
    lng  = body.get("lng")

    with lock:
        # auto-end any session left active
        if current_session is not None and current_session["status"] == "active":
            current_session["status"]   = "ended"
            current_session["end_time"] = time.time()

        session_counter += 1
        import random, math
        current_session = {
            "id":             session_counter,
            "status":         "active",
            "name":           name,
            "start_time":     time.time(),
            "end_time":       None,
            "start_location": {"lat": lat, "lng": lng, "address": name},
            "end_location":   None,
            "distance_km":    0.0,
            "events":         [],
            "path":           ([{"lat": lat, "lng": lng}] if lat is not None else []),
            "heading":        random.uniform(0, 2*math.pi),
        }
        sessions.append(current_session)
        summary = summarize_session(current_session)

    print(f"[Session] START #{summary['id']} — {name}")
    return jsonify({"started": True, "session": summary})


@app.route("/session/end", methods=["POST"])
def session_end():
    """End the active survey. Body: {distance_km?, end_lat?, end_lng?, end_name?}.
    If distance_km is omitted but end coords are given, it's computed via haversine."""
    global current_session
    body = request.get_json(silent=True) or {}

    with lock:
        if current_session is None or current_session["status"] != "active":
            return jsonify({"error": "no active session"}), 400

        s = current_session
        s["status"]   = "ended"
        s["end_time"] = time.time()

        end_lat = body.get("end_lat")
        end_lng = body.get("end_lng")
        end_nm  = body.get("end_name") or s["name"]
        # default the end point to wherever the simulated track finished
        if end_lat is None and end_lng is None and s.get("path"):
            tail = s["path"][-1]
            end_lat, end_lng = tail["lat"], tail["lng"]
        if end_lat is not None and end_lng is not None:
            s["end_location"] = {"lat": end_lat, "lng": end_lng, "address": end_nm}

        if body.get("distance_km") is not None:
            try:
                s["distance_km"] = round(float(body["distance_km"]), 4)
            except (ValueError, TypeError):
                pass
        else:
            # sum the length of the GPS track
            path = s.get("path", [])
            dist = 0.0
            for a, b in zip(path, path[1:]):
                if a["lat"] is not None and b["lat"] is not None:
                    dist += haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
            if dist > 0:
                s["distance_km"] = round(dist, 4)

        summary = summarize_session(s)
        current_session = None

    print(f"[Session] END #{summary['id']} — {summary['total_damage']} damages, {summary['distance_km']} km")
    return jsonify({"ended": True, "session": summary})


@app.route("/sessions")
def sessions_list():
    """All surveys (newest first) as lightweight summaries for the map."""
    with lock:
        out = [summarize_session(s) for s in sessions]
    out.reverse()
    return jsonify({"sessions": out, "count": len(out)})


@app.route("/session/<int:session_id>")
def session_detail(session_id):
    """Full survey detail + damage breakdown by type (no thumbnails, lightweight)."""
    with lock:
        s = next((x for x in sessions if x["id"] == session_id), None)
        if s is None:
            return jsonify({"error": "session not found"}), 404
        return jsonify(summarize_session(s))


@app.route("/session/<int:session_id>/type/<damage_type>")
def session_type_events(session_id, damage_type):
    """All damage events of a given type within a session, with thumbnails."""
    with lock:
        s = next((x for x in sessions if x["id"] == session_id), None)
        if s is None:
            return jsonify({"error": "session not found"}), 404
        evs = [e for e in s["events"] if e["class"] == damage_type]
        evs = list(reversed(evs))
    return jsonify({"session_id": session_id, "type": damage_type,
                    "count": len(evs), "events": evs})


@app.route("/session/<int:session_id>/delete", methods=["POST"])
def session_delete(session_id):
    """Delete a whole survey."""
    global current_session
    with lock:
        before = len(sessions)
        sessions[:] = [x for x in sessions if x["id"] != session_id]
        if current_session is not None and current_session["id"] == session_id:
            current_session = None
        removed = before - len(sessions)
    return jsonify({"deleted": removed > 0, "id": session_id})


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
            record_event_to_session(event)

    if detections and CLOUD_URL:
        threading.Thread(target=upload_to_cloud, args=(detections,), daemon=True).start()

    # Return an annotated frame for the UI's frozen-result preview.
    # If we already built an event thumbnail, reuse it (avoids a 2nd render+encode).
    if event is not None:
        frame_uri = event["thumbnail"]
    else:
        # No detections → still show the frozen frame, but downscaled + light JPEG for speed
        preview = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_uri = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "captured":   True,
        "detections": detections,
        "count":      len(detections),
        "event_id":   event["id"] if event else None,
        "frame":      frame_uri,
    })


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
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
    print(f"AUTO filters:     min conf {MIN_CONFIDENCE:.0%} · dedup {DEDUP_COOLDOWN:.0f}s")
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
    print("  /sessions    — survey list for map (GET)")
    print("  /session/... — start|end|detail|type (GET/POST)")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, threaded=True)