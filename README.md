# ROAD AI

AI-powered road damage detection, measurement, and repair-cost estimation for
Thai road maintenance. Runs as a self-contained edge unit on an NVIDIA Jetson
Nano (camera + GPS + AI + web dashboard).

## What it does
- Detects road damage (pothole, longitudinal / transverse / alligator crack)
- Measures real-world size via camera homography calibration
- Estimates repair cost from the Department of Rural Roads (กรมทางหลวงชนบท) manual
- Pins damage on a live GPS map and serves a bilingual (TH/EN) dashboard

## Folder layout
```
edge/               Device code (runs on the Jetson)
  roadai.py           Flask server: capture + inference + GPS + dashboard API
  cost.py             Deterministic repair-cost calculator
  cost_rates.json     Official per-unit rates extracted from the DRR manual
  measurement.py      Pixel -> cm area via homography calibration
  gps_reader.py       NEO-6M GPS reader (UART)
  .env                Device configuration
index.html          Survey dashboard (open in a browser -> nano.local:5000)
measure_tool.html   Optical measurement tool (for physics experiments)
presentation.html   Standalone slide/screenshot mockup (not part of the app)
models/             Trained model files (best.pt / best.onnx)
ROADAI_Train_Export.ipynb   Colab notebook for (re)training + ONNX export
docs/               Project docs, competition scripts, experiment plans
archive/            Legacy code kept for reference (old Pi scripts, etc.)
```

## Running it
The Jetson auto-starts `roadai.py` on boot (systemd service `roadai.service`).
1. Power on the Jetson, connect it to Wi-Fi / phone hotspot
2. Put your laptop/phone on the same network
3. Open `index.html` -> it connects to `http://nano.local:5000`

Manage the service:  `sudo systemctl {status,restart} roadai` · logs: `journalctl -u roadai -f`

## Hardware
- NVIDIA Jetson Nano B01 4GB (Ubuntu 20.04 image), Intel AC8265 Wi-Fi
- USB camera, u-blox NEO-6M GPS
