"""Smoke test: confirm the Roboflow serverless workflow endpoint responds.
Run: py -3.12 smoke_test_roboflow.py
"""
import os, base64, requests, cv2, numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join("edge", ".env"))
API_URL = os.getenv("ROBOFLOW_API_URL")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WS, WF  = os.getenv("ROBOFLOW_WORKSPACE", "cheemo"), os.getenv("ROBOFLOW_WORKFLOW_ID", "custom-workflow-2")

img = np.full((360, 640, 3), 128, np.uint8)              # neutral grey test frame
_, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf).decode()

url = f"{API_URL}/{WS}/workflows/{WF}"
print(f"POST {url}")
r = requests.post(url, json={
    "api_key": API_KEY,
    "inputs": {"image": {"type": "base64", "value": b64}},
    "use_cache": False,
}, timeout=60)

print("HTTP", r.status_code)
if r.ok:
    outputs = r.json().get("outputs", [{}])
    print("output entry count:", len(outputs))
    print("output keys:", list(outputs[0].keys()) if outputs else "none")
else:
    print("ERROR body:", r.text[:600])
