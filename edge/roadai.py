import cv2
import numpy as np
import base64
import requests
import threading
import time
from flask import Flask, Response, jsonify

# Config
API_URL = "http://localhost:9001"
API_KEY = "T4zuK4T4cMUowqMPvSVl"
WORKSPACE = "cheemo"
WORKFLOW_ID = "custom-workflow-2"
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 360
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

SX = DISPLAY_WIDTH / INFERENCE_WIDTH
SY = DISPLAY_HEIGHT / INFERENCE_HEIGHT

# Calibration
PX_PER_CM_W = 27.06
PX_PER_CM_H = 29.58

# Shared state
latest_frame = None
latest_annotations = None
output_frame = None
latest_detections = []
lock = threading.Lock()
running = True

app = Flask(__name__)

def encode_frame(frame):
    small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
    _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode("utf-8")

def calculate_metrics(pred_bbox=None, pred_seg=None):
    metrics = {}

    if pred_bbox:
        w = pred_bbox["width"]
        h = pred_bbox["height"]
        metrics["bbox_width_cm"] = round(w / PX_PER_CM_W, 2)
        metrics["bbox_height_cm"] = round(h / PX_PER_CM_H, 2)
        metrics["bbox_area_cm2"] = round((w / PX_PER_CM_W) * (h / PX_PER_CM_H), 2)

    if pred_seg and "points" in pred_seg:
        points = np.array([[p["x"], p["y"]] for p in pred_seg["points"]], dtype=np.float32)
        area = float(cv2.contourArea(points))
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        metrics["seg_width_cm"] = round(w / PX_PER_CM_W, 2)
        metrics["seg_height_cm"] = round(h / PX_PER_CM_H, 2)
        metrics["seg_area_cm2"] = round(area / (PX_PER_CM_W * PX_PER_CM_H), 2)

    return metrics

def inference_worker():
    global latest_annotations, latest_detections
    url = f"{API_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}"

    while running:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.005)
            continue

        try:
            payload = {
                "api_key": API_KEY,
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": encode_frame(frame)
                    }
                },
                "use_cache": False
            }
            response = requests.post(url, json=payload, timeout=5)
            if response.ok:
                result = response.json()
                outputs = result.get("outputs", [{}])[0]
                detections = []

                bbox_preds = outputs.get("predictions", {}).get("predictions", [])
                seg_preds = outputs.get("model_1_predictions", {}).get("predictions", [])

                for pred in seg_preds:
                    for p in pred["points"]:
                        p["x"] = p["x"] * SX
                        p["y"] = p["y"] * SY

                for pred in bbox_preds:
                    pred["x"] = pred["x"] * SX
                    pred["y"] = pred["y"] * SY
                    pred["width"] = pred["width"] * SX
                    pred["height"] = pred["height"] * SY

                bbox = bbox_preds[0] if bbox_preds else None
                seg = seg_preds[0] if seg_preds else None
                metrics = calculate_metrics(pred_bbox=bbox, pred_seg=seg)

                if bbox:
                    detections.append({
                        "class": bbox["class"],
                        "confidence": bbox["confidence"],
                        "type": "bbox",
                        **metrics
                    })
                if seg:
                    detections.append({
                        "class": "Segmentation",
                        "confidence": seg["confidence"],
                        "type": "seg",
                        **metrics
                    })

                with lock:
                    latest_annotations = outputs
                    latest_detections = detections

        except Exception as e:
            print(f"Inference error: {e}")

def draw_annotations(frame, annotations):
    if not annotations:
        return frame

    for pred in annotations.get("model_1_predictions", {}).get("predictions", []):
        points = np.array([[int(p["x"]), int(p["y"])] for p in pred["points"]], dtype=np.int32)
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

def camera_worker():
    global latest_frame, output_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        with lock:
            latest_frame = frame.copy()
            annotations = latest_annotations

        display = draw_annotations(frame.copy(), annotations)

        with lock:
            output_frame = display.copy()

    cap.release()

def generate_stream():
    while True:
        with lock:
            frame = output_frame.copy() if output_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1/30)

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    with lock:
        dets = latest_detections.copy()
    return jsonify({"detections": dets})

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    print("RoadAI running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)