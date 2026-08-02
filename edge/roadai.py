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

import gps_reader   # local NEO-6M reader; safe to import even without pyserial
import measurement  # pixel->cm area via camera calibration (homography)
import cost         # deterministic repair-cost estimate from the DRR manual rates

# onnxruntime is optional — only needed for the local ONNX backend. The Jetson
# edge unit uses the Ultralytics backend instead, so don't hard-require it.
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Load the .env sitting next to this file, regardless of the working directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Config ────────────────────────────────────────────────────
API_URL     = os.getenv("ROBOFLOW_API_URL",    "http://localhost:9001")
API_KEY     = os.getenv("ROBOFLOW_API_KEY",    "T4zuK4T4cMUowqMPvSVl")
WORKSPACE   = os.getenv("ROBOFLOW_WORKSPACE",  "cheemo")
WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID","custom-workflow-2")
CLOUD_URL   = os.getenv("CLOUD_API_URL",       None)

# A freshly (re)started self-hosted inference server (e.g. Roboflow's Jetson
# Docker image) can take minutes to load/optimize models on its FIRST request
# (TensorRT engine build on a Jetson Nano is slow). A short timeout here just
# causes an endless retry loop that never lets the model finish loading, so
# this defaults generously; serverless/warm requests return in ~1-2s regardless.
ROBOFLOW_TIMEOUT = float(os.getenv("ROBOFLOW_TIMEOUT", "120"))
CAMERA_IDX  = int(os.getenv("CAMERA_INDEX",   0))

# Frame source: "local" = USB webcam on this machine | "remote" = frames pushed
# by the Raspberry Pi via POST /upload. In remote mode the local camera is not opened.
FRAME_SOURCE = os.getenv("FRAME_SOURCE", "local").lower()

# Read GPS directly from a NEO-6M wired to THIS machine's UART (the Jetson edge
# unit). When off, GPS comes from the Pi's /upload payload instead (legacy).
USE_LOCAL_GPS = os.getenv("USE_LOCAL_GPS", "0").lower() in ("1", "true", "yes")

# Where annotated frames that CONTAIN a detection are written to disk.
# Frames with no detection are never written (keeps the disk from filling up).
SAVE_DIR = os.getenv("SAVE_DIR", "received_frames")
os.makedirs(SAVE_DIR, exist_ok=True)

# AUTO-mode recording filters
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE",  0.60))  # ignore weak detections (< 60%)
DEDUP_COOLDOWN  = float(os.getenv("DEDUP_COOLDOWN",  5.0))   # secs to suppress same-class repeats

# ── ONNX local inference ──────────────────────────────────────
USE_ONNX = os.getenv("USE_ONNX", "0").lower() in ("1", "true", "yes")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.onnx"
))
ONNX_CLASS_NAMES = ["longitudinal crack", "pothole"]
ONNX_CONF_THRESH = MIN_CONFIDENCE
ONNX_SESSION     = None

# ── Ultralytics local inference (YOLOv8 .pt on GPU — Jetson) ───
# Highest-priority backend when enabled: runs a YOLO .pt model directly on the
# GPU via the `ultralytics` package. Used on the Jetson Nano edge unit.
USE_ULTRA = os.getenv("USE_ULTRA", "0").lower() in ("1", "true", "yes")
ULTRA_MODEL_PATH = os.getenv("ULTRA_MODEL", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "YOLOv8_Small_RDD.pt"
))
ULTRA_MODEL = None

# ── Local in-process Roboflow inference (no serverless credits, no Docker) ──
# Runs the SAME Roboflow models (detection + segmentation) locally via the
# `inference` package. Free — uses this machine's compute, not serverless.
USE_LOCAL_INFERENCE = os.getenv("USE_LOCAL_INFERENCE", "0").lower() in ("1", "true", "yes")
LOCAL_DET_MODEL = os.getenv("LOCAL_DET_MODEL", "road-ai-yskqr/9")        # object detection
LOCAL_SEG_MODEL = os.getenv("LOCAL_SEG_MODEL", "road-ai-segmentation/2")  # instance segmentation
LOCAL_DET = None
LOCAL_SEG = None

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
latest_gps         = (None, None)         # (lat, lng) from Pi GPS or None
frame_seq          = 0                    # bumped on every new frame (lets the
last_inferred_seq  = -1                   # inference worker skip frames it already saw)
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


def push_frame(frame, gps=(None, None)):
    """Set a freshly-captured frame as the current one and render the annotated
    output frame for the live stream. Shared by the local camera_worker and the
    remote /upload endpoint, so both frame sources behave identically."""
    global latest_frame, output_frame, frame_seq, latest_gps
    with lock:
        latest_frame = frame.copy()
        frame_seq   += 1
        annotations  = latest_annotations
        if gps[0] is not None:
            latest_gps = gps
    display = draw_annotations(frame.copy(), annotations)
    with lock:
        output_frame = display.copy()


def _sanitize(text):
    """Make a string safe for use in a filename."""
    keep = "".join(c if c.isalnum() else "_" for c in str(text))
    return "_".join(p for p in keep.split("_") if p)  # collapse repeats


def save_detected_frame(frame, outputs, event_id, damage_type="unknown",
                        gps=(None, None), confidence=None):
    """Write a full-resolution annotated frame to disk — ONLY called when the
    frame contains at least one detection. Empty frames are never saved, so the
    disk only ever holds frames with real road damage.

    Filename encodes the metadata so each file is self-describing:
        <time>-<lng>-<lat>-<damagetype>-conf<NN>-id<eid>.jpg
    e.g. 20260617_143012-99.015567-18.757548-pothole-conf86-id42.jpg
    """
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    lat, lng = gps
    lat_str = f"{lat:.6f}" if lat is not None else "nofix"
    lng_str = f"{lng:.6f}" if lng is not None else "nofix"
    dmg     = _sanitize(damage_type)
    conf    = f"-conf{round(confidence*100)}" if confidence is not None else ""
    filename = os.path.join(
        SAVE_DIR, f"{ts}-{lng_str}-{lat_str}-{dmg}{conf}-id{event_id}.jpg"
    )
    annotated = draw_annotations(frame.copy(), outputs)
    cv2.imwrite(filename, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return filename


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
    # Persist the annotated frame to disk (only happens for frames WITH damage).
    try:
        event["image_path"] = save_detected_frame(
            frame, outputs, eid,
            damage_type=primary.get("class", "unknown"),
            gps=latest_gps,
            confidence=primary.get("confidence"),
        )
    except Exception as e:
        print(f"[Save] could not write frame for event {eid}: {e}")
        event["image_path"] = None
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
    """Append a damage event to the active session (if any) and stamp it with
    the real GPS position from the Pi (if available) or a simulated fallback."""
    if current_session is not None and current_session["status"] == "active":
        real_lat, real_lng = latest_gps
        if real_lat is not None:
            # Real GPS from NEO-6M — append to path and use directly
            pt = {"lat": real_lat, "lng": real_lng}
            current_session.setdefault("path", []).append(pt)
        else:
            # No GPS fix yet — fall back to random-walk simulation
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

    # Live distance: sum the GPS track length so far (updates while active).
    # If the session was ended with an explicit distance_km, prefer that.
    live_dist = 0.0
    for a, b in zip(path, path[1:]):
        if a["lat"] is not None and b["lat"] is not None:
            live_dist += haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
    live_dist = round(live_dist, 4)
    distance_km = s["distance_km"] if s["distance_km"] else live_dist

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
        "distance_km":     distance_km,
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
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        # Real-world area: uses the camera homography when calibrated, otherwise
        # falls back to the fixed PX_PER_CM scale (non-breaking).
        area_m2  = measurement.polygon_area_m2(pred_seg["points"], PX_PER_CM_W, PX_PER_CM_H)
        metrics["seg_width_cm"]  = round(w / PX_PER_CM_W, 2)
        metrics["seg_height_cm"] = round(h / PX_PER_CM_H, 2)
        metrics["seg_area_m2"]   = area_m2
        metrics["seg_area_cm2"]  = round(area_m2 * 10000, 2)
        metrics["measured_by"]   = "homography" if measurement.is_calibrated() else "fixed_scale"

    return metrics


def estimate_cost(detection):
    """Estimate repair cost in THB using the official DRR manual rates.
    Returns cost + the repair method and manual source for traceability."""
    cls     = detection.get("class", "")
    sev     = detection.get("severity", "medium")
    area_m2 = detection.get("seg_area_m2", None)
    if area_m2 is None:                         # no mask -> approximate from bbox
        area_m2 = detection.get("bbox_area_cm2", 0) / 10000

    result = cost.estimate_repair_cost(cls, sev, area_m2)
    result["area_m2"] = round(float(area_m2 or 0), 4)
    return result


def severity(detection):
    """Return severity level based on damaged area (cm²)."""
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


# ── ONNX inference helpers ────────────────────────────────────
def _preprocess(frame):
    """Resize + normalise a BGR frame to a (1,3,640,640) float32 tensor."""
    img = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_WIDTH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[np.newaxis]          # CHW → NCHW


def _parse_onnx_output(raw, orig_w, orig_h, conf_thresh):
    """Parse the YOLO11-seg ONNX output into bbox + mask predictions.

    YOLO11-seg exports two tensors:
      output0  shape (1, 4+nc+32, num_anchors)  — box + cls + mask coefficients
      output1  shape (1, 32, mh, mw)            — prototype masks

    Returns (bbox_preds, seg_preds) in the same dict format the rest of
    the pipeline already understands (same keys used by the old Roboflow path).
    """
    output0 = raw[0][0]          # (4+nc+32, num_anchors)
    proto   = raw[1][0]          # (32, mh, mw)

    nc      = len(ONNX_CLASS_NAMES)
    box_dim = 4
    mh, mw  = proto.shape[1], proto.shape[2]

    # Transpose so each row is one anchor: (num_anchors, 4+nc+32)
    preds = output0.T

    bbox_preds, seg_preds = [], []

    for pred in preds:
        cls_scores = pred[box_dim:box_dim + nc]
        conf       = float(cls_scores.max())
        if conf < conf_thresh:
            continue

        cls_id   = int(cls_scores.argmax())
        cx, cy, w, h = pred[:4]

        # Convert from INFERENCE_WIDTH-space to display resolution
        scale_x = orig_w / INFERENCE_WIDTH
        scale_y = orig_h / INFERENCE_WIDTH
        cx *= scale_x;  cy *= scale_y
        w  *= scale_x;  h  *= scale_y

        bbox_preds.append({
            "class":      ONNX_CLASS_NAMES[cls_id],
            "confidence": round(conf, 3),
            "x": float(cx), "y": float(cy),
            "width": float(w), "height": float(h),
        })

        # Build segmentation mask polygon for this anchor
        mask_coefs = pred[box_dim + nc:]            # (32,)
        mask_map   = (mask_coefs @ proto.reshape(32, -1)).reshape(mh, mw)
        mask_map   = 1 / (1 + np.exp(-mask_map))   # sigmoid
        mask_map   = (mask_map > 0.5).astype(np.uint8) * 255

        # Scale mask to display size and extract polygon
        mask_full  = cv2.resize(mask_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt    = max(contours, key=cv2.contourArea)
            points = [{"x": float(p[0][0]), "y": float(p[0][1])} for p in cnt]
            seg_preds.append({
                "class":      ONNX_CLASS_NAMES[cls_id],
                "confidence": round(conf, 3),
                "points":     points,
            })

    return bbox_preds, seg_preds


# ── Inference core (shared by auto worker + manual capture) ──
def infer_frame(frame, source="auto"):
    """Run inference on a single frame. Backend priority:
       1. Local in-process Roboflow models (no credits, no Docker)
       2. Ultralytics YOLO .pt on GPU (Jetson edge unit)
       3. Roboflow API (serverless / Docker)
       4. local ONNX runtime
    Returns (outputs, detections)."""
    if LOCAL_DET is not None:
        return _infer_local(frame, source)
    elif ULTRA_MODEL is not None:
        return _infer_ultralytics(frame, source)
    elif API_URL and ONNX_SESSION is None:
        return _infer_roboflow(frame, source)
    elif ONNX_SESSION is not None:
        return _infer_onnx(frame, source)
    else:
        print("[Inference] no inference backend available")
        return None, []


def _to_dict(p):
    """Roboflow prediction object -> plain dict (handles pydantic v1/v2)."""
    for m in ("model_dump", "dict"):
        if hasattr(p, m):
            try:
                return getattr(p, m)()
            except Exception:
                pass
    return dict(getattr(p, "__dict__", {}))


def _infer_local(frame, source="auto"):
    """Run the Roboflow detection + segmentation models LOCALLY, in-process via
    the `inference` package — no serverless credits, no Docker. Same output
    format as _infer_roboflow (predictions in the frame's pixel coordinates)."""
    det_res = LOCAL_DET.infer(frame)[0]
    seg_res = LOCAL_SEG.infer(frame)[0]

    bbox_preds = []
    for p in (getattr(det_res, "predictions", None) or []):
        d = _to_dict(p)
        bbox_preds.append({
            "class":      d.get("class") or d.get("class_name") or "damage",
            "confidence": round(float(d.get("confidence", 0)), 3),
            "x": float(d["x"]), "y": float(d["y"]),
            "width": float(d["width"]), "height": float(d["height"]),
        })

    seg_preds = []
    for p in (getattr(seg_res, "predictions", None) or []):
        d = _to_dict(p)
        pts = [{"x": float(pt["x"]), "y": float(pt["y"])} for pt in (d.get("points") or [])]
        if pts:
            seg_preds.append({
                "class":      d.get("class") or d.get("class_name") or "damage",
                "confidence": round(float(d.get("confidence", 0)), 3),
                "points":     pts,
            })

    outputs = {
        "predictions":         {"predictions": bbox_preds},
        "model_1_predictions": {"predictions": seg_preds},
    }
    detections = []
    bbox    = bbox_preds[0] if bbox_preds else None
    seg     = seg_preds[0]  if seg_preds  else None
    metrics = calculate_metrics(pred_bbox=bbox, pred_seg=seg)

    if bbox:
        det = {"class": bbox["class"], "confidence": bbox["confidence"], "type": "bbox",
               "source": source, "timestamp": time.time(), **metrics}
        det["severity"] = severity(det); det.update(estimate_cost(det)); detections.append(det)
    if seg:
        det = {"class": "Segmentation", "confidence": seg["confidence"], "type": "seg",
               "source": source, "timestamp": time.time(), **metrics}
        det["severity"] = severity(det); det.update(estimate_cost(det)); detections.append(det)

    return outputs, detections


def _infer_ultralytics(frame, source="auto"):
    """Run a YOLOv8 .pt model on the GPU via ultralytics.
    Detection model (bounding boxes) — area is approximated from the bbox.
    Returns ALL detections in the frame (not just the first)."""
    results = ULTRA_MODEL.predict(
        frame, conf=MIN_CONFIDENCE, verbose=False, device=0, imgsz=640
    )[0]

    bbox_preds = []
    detections = []

    if results.boxes is not None:
        for b in results.boxes:
            cls_id = int(b.cls)
            conf   = float(b.conf)
            # xywh: centre-x, centre-y, width, height in the frame's own pixels
            cx, cy, w, h = (float(v) for v in b.xywh[0])
            cls_name = ULTRA_MODEL.names.get(cls_id, f"class{cls_id}")

            bbox = {
                "class": cls_name, "confidence": round(conf, 3),
                "x": cx, "y": cy, "width": w, "height": h,
            }
            bbox_preds.append(bbox)

            metrics = calculate_metrics(pred_bbox=bbox)
            det = {
                "class":      cls_name,
                "confidence": round(conf, 3),
                "type":       "bbox",
                "source":     source,
                "timestamp":  time.time(),
                **metrics,
            }
            det["severity"] = severity(det)
            det.update(estimate_cost(det))
            detections.append(det)

    # No segmentation from a detect model — leave model_1_predictions empty.
    outputs = {
        "predictions":         {"predictions": bbox_preds},
        "model_1_predictions": {"predictions": []},
    }
    return outputs, detections


def _infer_roboflow(frame, source="auto"):
    """Call the Roboflow workflow endpoint (serverless cloud or local server,
    set by ROBOFLOW_API_URL). Returns (outputs, detections)."""
    url = f"{API_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    payload = {
        "api_key":  API_KEY,
        "inputs":   {"image": {"type": "base64", "value": encode_frame(frame)}},
        "use_cache": False,
    }
    response = requests.post(url, json=payload, timeout=ROBOFLOW_TIMEOUT)
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


def _infer_onnx(frame, source="auto"):
    """Run local ONNX inference."""
    tensor = _preprocess(frame)
    raw    = ONNX_SESSION.run(None, {ONNX_SESSION.get_inputs()[0].name: tensor})

    h, w = frame.shape[:2]
    bbox_preds, seg_preds = _parse_onnx_output(raw, w, h, ONNX_CONF_THRESH)

    outputs = {
        "predictions":         {"predictions": bbox_preds},
        "model_1_predictions": {"predictions": seg_preds},
    }

    detections = []
    bbox    = bbox_preds[0] if bbox_preds else None
    seg     = seg_preds[0]  if seg_preds  else None
    metrics = calculate_metrics(pred_bbox=bbox, pred_seg=seg)

    if bbox:
        det = {
            "class":      bbox["class"],
            "confidence": bbox["confidence"],
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
            "class":      seg["class"],
            "confidence": seg["confidence"],
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
    global latest_annotations, latest_detections, last_inferred_seq

    while running:
        # In MANUAL mode the worker idles — detection only happens via /capture
        if detection_mode != "auto":
            time.sleep(0.1)
            continue

        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            seq   = frame_seq
        if frame is None:
            time.sleep(0.005)
            continue
        # Skip frames we've already run inference on (avoids redundant API calls
        # when the same Pi frame sits in latest_frame between 1 fps uploads).
        if seq == last_inferred_seq:
            time.sleep(0.01)
            continue
        last_inferred_seq = seq

        try:
            outputs, detections = infer_frame(frame, source="auto")
            print(f"[Infer] seq={last_inferred_seq} detections={len(detections)}")
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
    """Local USB-webcam capture loop. Not started in remote (Pi) mode —
    there the frames arrive via POST /upload instead.

    On Linux (Jetson) we force the V4L2 backend + MJPG — the default GStreamer
    backend fails on many USB webcams ('Internal data stream error'). On Windows
    we let OpenCV pick its default backend."""
    import sys
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_IDX, backend)
    if sys.platform.startswith("linux"):
        # MJPG lets the webcam deliver 720p over USB without choking
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAMERA_IDX}")
        return
    print(f"Camera opened (index {CAMERA_IDX}, backend {'V4L2' if backend==cv2.CAP_V4L2 else 'default'})")

    fail = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            fail += 1
            if fail % 30 == 0:
                print(f"WARNING: camera read failed x{fail}")
            time.sleep(0.03)
            continue
        fail = 0
        # Tag the frame with the live local GPS fix (Jetson). If no module / no
        # fix, get_location() returns (None, None) and push_frame keeps the last.
        gps = gps_reader.get_location() if USE_LOCAL_GPS else (None, None)
        push_frame(frame, gps=gps)
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


@app.route("/snapshot")
def snapshot():
    """Return the current RAW camera frame as a JPEG (no annotations).
    Used by the measurement tool to grab a still for calibration/measurement."""
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "no frame available"}), 503
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/upload", methods=["POST"])
def upload():
    """Receive a JPEG frame pushed by the Raspberry Pi and feed it into the
    same pipeline the local camera would. Inference, event-building, disk-saving
    and streaming all happen exactly as in local mode."""
    if "frame" not in request.files:
        return jsonify({"status": "error", "message": "no 'frame' field"}), 400

    data = request.files["frame"].read()
    if not data:
        return jsonify({"status": "error", "message": "empty file"}), 400

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"status": "error", "message": "could not decode JPEG"}), 400

    # Match the display resolution the rest of the pipeline expects
    if (img.shape[1], img.shape[0]) != (DISPLAY_WIDTH, DISPLAY_HEIGHT):
        img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Read GPS coordinates sent by the Pi (optional — falls back to simulation)
    try:
        pi_lat = float(request.form.get("lat")) if request.form.get("lat") else None
        pi_lng = float(request.form.get("lng")) if request.form.get("lng") else None
    except (ValueError, TypeError):
        pi_lat, pi_lng = None, None

    push_frame(img, gps=(pi_lat, pi_lng))
    with lock:
        seq = frame_seq
    gps_str = f"GPS={pi_lat},{pi_lng}" if pi_lat is not None else "GPS=none"
    print(f"[{time.strftime('%H:%M:%S')}] frame #{seq}  {gps_str}  ({len(data)/1024:.1f} KB)")
    return jsonify({"status": "ok", "frame_id": seq})


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
@app.route("/gps")
def gps_location():
    """Latest GPS coordinates from the Pi. Returns {lat, lng, fix: true/false}."""
    with lock:
        lat, lng = latest_gps
    return jsonify({"fix": lat is not None, "lat": lat, "lng": lng})


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
def session_start_route():
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


def _load_local_inference():
    global LOCAL_DET, LOCAL_SEG
    if not USE_LOCAL_INFERENCE:
        return
    from inference import get_model
    print(f"Loading local models (downloads weights once, runs on THIS machine, no credits)...")
    LOCAL_DET = get_model(model_id=LOCAL_DET_MODEL, api_key=API_KEY)
    LOCAL_SEG = get_model(model_id=LOCAL_SEG_MODEL, api_key=API_KEY)
    print(f"Local in-process inference ready: {LOCAL_DET_MODEL} + {LOCAL_SEG_MODEL}")


def _load_ultralytics():
    global ULTRA_MODEL
    if USE_LOCAL_INFERENCE:
        return  # local Roboflow models take priority
    if not USE_ULTRA:
        return
    if not os.path.exists(ULTRA_MODEL_PATH):
        print(f"WARNING: Ultralytics model not found at {ULTRA_MODEL_PATH} — skipping")
        return
    from ultralytics import YOLO
    ULTRA_MODEL = YOLO(ULTRA_MODEL_PATH)
    # Warm up the GPU so the first real frame isn't slow
    warm = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    ULTRA_MODEL.predict(warm, verbose=False, device=0, imgsz=640)
    print(f"Ultralytics model loaded: {ULTRA_MODEL_PATH}")
    print(f"Classes: {ULTRA_MODEL.names}")


def _load_onnx():
    global ONNX_SESSION
    if USE_LOCAL_INFERENCE or USE_ULTRA:
        return  # a higher-priority backend is active
    if not USE_ONNX:
        print("ONNX inference disabled (USE_ONNX=0) — using Roboflow API")
        return
    if ort is None:
        print("WARNING: onnxruntime not installed — ONNX backend unavailable")
        return
    if os.path.exists(ONNX_MODEL_PATH):
        ONNX_SESSION = ort.InferenceSession(
            ONNX_MODEL_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print(f"ONNX model loaded: {ONNX_MODEL_PATH}")
        print(f"Provider in use:   {ONNX_SESSION.get_providers()[0]}")
    else:
        print(f"WARNING: ONNX model not found at {ONNX_MODEL_PATH} — inference disabled")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    _load_local_inference()
    _load_ultralytics()
    _load_onnx()

    if USE_LOCAL_GPS:
        gps_reader.start()

    threading.Thread(target=inference_worker, daemon=True).start()
    if FRAME_SOURCE == "remote":
        print("Frame source:     REMOTE — waiting for Pi frames at POST /upload")
    else:
        threading.Thread(target=camera_worker, daemon=True).start()

    print("=" * 50)
    print("RoadAI running at http://localhost:5000")
    print(f"Docker inference: {API_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}")
    print(f"Frame source:     {FRAME_SOURCE}  (camera index {CAMERA_IDX} if local)")
    print(f"Saving frames to: {os.path.abspath(SAVE_DIR)}  (only frames with damage)")
    print(f"Cloud upload:     {'→ ' + CLOUD_URL if CLOUD_URL else 'disabled'}")
    print(f"AUTO filters:     min conf {MIN_CONFIDENCE:.0%} · dedup {DEDUP_COOLDOWN:.0f}s")
    print("Endpoints:")
    print("  /upload      — receive a frame from the Pi (POST)")
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