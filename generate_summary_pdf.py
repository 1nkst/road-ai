"""
Generates ROADAI_Project_Summary.pdf — comprehensive AI-readable project brief.
Run: python generate_summary_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "ROADAI_Project_Summary.pdf"

# ── Colour palette ────────────────────────────────────────────
C_BG        = colors.HexColor("#0a0a0a")
C_ACCENT    = colors.HexColor("#c8f135")
C_SURFACE   = colors.HexColor("#161616")
C_MUTED     = colors.HexColor("#888888")
C_WHITE     = colors.HexColor("#f0f0f0")
C_DANGER    = colors.HexColor("#ff4d4d")
C_WARN      = colors.HexColor("#f5a623")
C_INFO      = colors.HexColor("#4da6ff")
C_PURPLE    = colors.HexColor("#6366f1")
C_GREEN     = colors.HexColor("#22c55e")

W, H = A4

# ── Styles ────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    base = styles["Normal"]
    return ParagraphStyle(name, parent=base, **kw)

TITLE     = S("Title2",   fontSize=28, textColor=C_ACCENT,  spaceAfter=4,  fontName="Helvetica-Bold", alignment=TA_CENTER)
SUBTITLE  = S("Sub",      fontSize=12, textColor=C_WHITE,   spaceAfter=2,  fontName="Helvetica",      alignment=TA_CENTER)
BYLINE    = S("Byline",   fontSize=9,  textColor=C_MUTED,   spaceAfter=2,  fontName="Helvetica",      alignment=TA_CENTER)
H1        = S("H1",       fontSize=16, textColor=C_ACCENT,  spaceBefore=14,spaceAfter=4,  fontName="Helvetica-Bold")
H2        = S("H2",       fontSize=12, textColor=C_WHITE,   spaceBefore=10,spaceAfter=3,  fontName="Helvetica-Bold")
H3        = S("H3",       fontSize=10, textColor=C_INFO,    spaceBefore=7, spaceAfter=2,  fontName="Helvetica-Bold")
BODY      = S("Body",     fontSize=9,  textColor=C_WHITE,   spaceAfter=4,  fontName="Helvetica",      leading=14, alignment=TA_JUSTIFY)
MONO      = S("Mono",     fontSize=8,  textColor=C_ACCENT,  spaceAfter=2,  fontName="Courier",        leading=12)
MONO_BODY = S("MonoBody", fontSize=8,  textColor=C_WHITE,   spaceAfter=1,  fontName="Courier",        leading=12)
LABEL     = S("Label",    fontSize=7,  textColor=C_MUTED,   spaceAfter=1,  fontName="Helvetica",      leading=10)
BULLET    = S("Bullet",   fontSize=9,  textColor=C_WHITE,   spaceAfter=3,  fontName="Helvetica",      leading=13, leftIndent=12, firstLineIndent=-8)
TAG_GREEN = S("TagG",     fontSize=8,  textColor=C_GREEN,   fontName="Helvetica-Bold")
TAG_WARN  = S("TagW",     fontSize=8,  textColor=C_WARN,    fontName="Helvetica-Bold")
TAG_RED   = S("TagR",     fontSize=8,  textColor=C_DANGER,  fontName="Helvetica-Bold")

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#2a2a2a"), spaceAfter=6, spaceBefore=2)

def sp(n=6):
    return Spacer(1, n)

def bullet(text):
    return Paragraph(f"•  {text}", BULLET)

def kv_table(rows, col_widths=None):
    """Two-column key→value table."""
    cw = col_widths or [55*mm, 105*mm]
    data = [[Paragraph(k, LABEL), Paragraph(v, BODY)] for k, v in rows]
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_SURFACE),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_SURFACE, colors.HexColor("#111111")]),
        ("TEXTCOLOR", (0,0), (0,-1), C_MUTED),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LINEBELOW", (0,0), (-1,-2), 0.3, colors.HexColor("#2a2a2a")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#2a2a2a")),
    ]))
    return t

def section_box(title, content_paragraphs, accent=C_ACCENT):
    """A titled box that groups related paragraphs."""
    header = Table([[Paragraph(title, ParagraphStyle("bh", parent=H2, textColor=accent, spaceBefore=0, spaceAfter=0, fontName="Helvetica-Bold", fontSize=11))]],
                   colWidths=[160*mm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#1a1a1a")),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LINEABOVE", (0,0), (-1,-1), 1.5, accent),
    ]))
    return [header] + content_paragraphs + [sp(4)]

def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm,  bottomMargin=18*mm,
        title="ROAD AI — Project Summary",
        author="Yanannathan Mueanhong (Cheemo)",
    )

    story = []

    # ═══════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════
    cover_tbl = Table([[
        Paragraph("ROAD AI", TITLE),
    ]], colWidths=[160*mm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#0d0d0d")),
        ("TOPPADDING",  (0,0), (-1,-1), 28),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("BOX", (0,0), (-1,-1), 2, C_ACCENT),
    ]))
    story.append(sp(30))
    story.append(cover_tbl)
    story.append(sp(8))
    story.append(Paragraph("AI-POWERED ROAD DAMAGE DETECTION &amp; REPORTING SYSTEM", SUBTITLE))
    story.append(sp(4))
    story.append(Paragraph("Comprehensive Technical &amp; Conceptual Brief — AI Model Reference Document", BYLINE))
    story.append(sp(20))
    story.append(hr())
    story.append(sp(6))

    meta = [
        ("Project Name",    "ROAD AI — ระบบตรวจจับความเสียหายถนนด้วย AI"),
        ("Lead Developer",  "Yanannathan Mueanhong (Cheemo), M.6, Varee Chiang Mai School"),
        ("Co-Developer",    "Traisit Wuttichaikijtcharoen, M.6, Varee Chiang Mai School"),
        ("Advisor",         "Sorawat  |  Director: Waree"),
        ("Competitions",    "NSC 2026 (28th National Software Contest, Category 23, code: 28P23N00470)  •  KMUTNB TIA 26th Innovation Award  •  CEDT Innovation Summit 2026"),
        ("Current Status",  "Working prototype — laptop inference + Pi streaming established; ONNX offline deployment in progress"),
        ("Document Date",   "June 2026"),
    ]
    story.append(kv_table(meta))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 1. PROJECT OVERVIEW & PROBLEM STATEMENT
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("1. PROJECT OVERVIEW &amp; PROBLEM STATEMENT", H1))
    story.append(hr())

    story.append(Paragraph(
        "Thailand's road infrastructure suffers from widespread, under-documented damage. Local government agencies — กรมทางหลวง (Department of Highways), เทศบาล (Municipalities), and อบต. (Subdistrict Administrative Organisations) — rely on manual visual inspections that are slow, subjective, inconsistently documented, and expensive. Damage is frequently missed, incorrectly classified, or recorded on paper forms that are later lost or mis-filed.",
        BODY))
    story.append(Paragraph(
        "ROAD AI automates the entire inspection pipeline: a camera-equipped Raspberry Pi 4 mounted on a survey vehicle captures road footage while driving; an AI model detects and classifies damage in real time; the system calculates the physical area of each defect, estimates repair cost in Thai Baht, pins the damage to a GPS map, and ultimately generates the official Thai government Form 1 repair-request document — all without human annotation.",
        BODY))

    story.append(sp(4))
    story += section_box("Target Users &amp; Deployment Context", [
        bullet("Thai Highway Department field engineers conducting road surveys"),
        bullet("Municipal maintenance crews (เทศบาล / อบต.) with limited technical resources"),
        bullet("Road inspection agencies requiring legally compliant Form 1 documentation"),
        bullet("Vehicles: pickup trucks, motorcycles, or any platform with a 12V power supply"),
        bullet("Environment: open roads, rural highways, urban streets in Chiang Mai and beyond"),
        bullet("Connectivity: unreliable 4G in rural areas — system must work fully OFFLINE"),
    ], accent=C_INFO)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 2. SYSTEM ARCHITECTURE
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("2. SYSTEM ARCHITECTURE", H1))
    story.append(hr())
    story.append(Paragraph(
        "The system is split across two physical devices that communicate over a local Wi-Fi network (or direct hotspot). Each device has a distinct, well-defined role:",
        BODY))

    arch_data = [
        [Paragraph("Layer", H3), Paragraph("Device", H3), Paragraph("Role", H3)],
        [Paragraph("Capture", BODY), Paragraph("Raspberry Pi 4", MONO), Paragraph("USB webcam → OpenCV → JPEG frame → POST to laptop every 1 second", BODY)],
        [Paragraph("Inference", BODY), Paragraph("Laptop (current)\nPi 4 (target)", MONO), Paragraph("Flask server receives frame, runs YOLO11 instance segmentation, returns detections", BODY)],
        [Paragraph("Storage", BODY), Paragraph("Laptop / Pi SD", MONO), Paragraph("Only frames WITH detections saved to disk. No-detection frames discarded immediately.", BODY)],
        [Paragraph("Presentation", BODY), Paragraph("Laptop browser", MONO), Paragraph("Single-page HTML5 dashboard: live feed, map, history, session management", BODY)],
        [Paragraph("Cloud (optional)", BODY), Paragraph("Remote server", MONO), Paragraph("Detections POSTed to cloud backend for long-term storage and cross-device access", BODY)],
    ]
    arch_t = Table(arch_data, colWidths=[30*mm, 38*mm, 92*mm])
    arch_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",  (0,0), (-1,0), C_ACCENT),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_SURFACE, colors.HexColor("#111111")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#2a2a2a")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(arch_t)
    story.append(sp(8))

    story.append(Paragraph("2.1  Data Flow — Frame by Frame", H2))
    steps = [
        ("Step 1 — Capture",    "Pi's camera_worker thread reads frames from USB webcam at up to 30 fps using OpenCV VideoCapture. Resolution: 1280×720 for display, downscaled to 640×360 for inference."),
        ("Step 2 — Encode",     "Frame JPEG-encoded (quality 85) and POSTed via HTTP multipart/form-data to laptop server at POST /upload (port 5050) or processed locally by the inference_worker thread."),
        ("Step 3 — Infer",      "YOLO11 instance segmentation model runs on the frame. Returns bounding boxes (class + confidence + x/y/w/h) AND segmentation polygon (xy point list) for each detected damage."),
        ("Step 4 — Scale",      "Predictions scaled from inference resolution (640×360) back to display resolution (1280×720) using SX=2.0 and SY=2.0 scale factors."),
        ("Step 5 — Metrics",    "Physical dimensions calculated from pixel measurements using calibrated constants: PX_PER_CM_W = 27.06, PX_PER_CM_H = 29.58. Segmentation polygon area computed via OpenCV contourArea."),
        ("Step 6 — Filter",     "AUTO mode: detections below 60% confidence discarded. Same-class detections within 5-second cooldown window treated as the same damage instance (deduplication)."),
        ("Step 7 — Event",      "Qualifying detections grouped into a Damage Event with: unique ID, class, severity, cost_thb, area_m2, GPS coordinates (real or simulated), annotated thumbnail (320×180 JPEG)."),
        ("Step 8 — Store",      "Event appended to in-memory deque (maxlen=300 events, maxlen=500 flat detections). Annotated frame saved to disk ONLY if detections present — empty frames discarded."),
        ("Step 9 — Stream",     "output_frame (annotated) served as MJPEG stream at /video_feed. Frontend polls /detections, /events, /stats every second."),
        ("Step 10 — Map",       "Each event stamped with GPS lat/lng. On survey map, events rendered as colored pins (red=high, orange=medium, yellow=low severity) along the survey path polyline."),
    ]
    for title, desc in steps:
        story.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", H3),
            Paragraph(desc, BODY),
        ]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 3. AI MODEL — YOLO11
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("3. AI MODEL — YOLO11 INSTANCE SEGMENTATION", H1))
    story.append(hr())

    story.append(Paragraph(
        "The core intelligence of ROAD AI is a YOLO11 instance segmentation model trained and hosted via Roboflow. Unlike simple object detection (bounding boxes only), instance segmentation produces a precise polygon outline of each damage area, enabling accurate physical area measurement in square centimetres and square metres.",
        BODY))

    model_rows = [
        ("Model architecture",  "YOLO11 Instance Segmentation (Ultralytics)"),
        ("Training platform",   "Roboflow (cloud training + hosted inference API)"),
        ("Inference endpoint",  "POST http://localhost:9001/{workspace}/workflows/{workflow_id}"),
        ("Input format",        "Base64-encoded JPEG, 640×360 pixels"),
        ("Output format",       "JSON: bounding box predictions + segmentation polygon point lists"),
        ("Current accuracy",    "83.1% mAP@50"),
        ("Classes trained",     "2 currently: Pothole, Longitudinal Crack"),
        ("Classes planned",     "6 total: + Transverse Crack, Alligator Crack, Surface Damage, Edge Break"),
        ("Confidence threshold","60% minimum (configurable via .env MIN_CONFIDENCE)"),
        ("Dedup cooldown",      "5 seconds per class (configurable via .env DEDUP_COOLDOWN)"),
        ("Inference resolution","640×360 (upscaled ×2 for 1280×720 display rendering)"),
        ("Offline target",      "ONNX export for on-device Pi inference without internet"),
    ]
    story.append(kv_table(model_rows))
    story.append(sp(8))

    story.append(Paragraph("3.1  Roboflow Workflow (Current Cloud Pipeline)", H2))
    story.append(Paragraph(
        "The current implementation calls a Roboflow custom workflow that chains two model outputs: (1) a bounding box detection model ('predictions') and (2) an instance segmentation model ('model_1_predictions'). Both outputs are returned in a single API call. This architecture allows the system to simultaneously draw tight bounding boxes for labelling and segmentation masks for area measurement.",
        BODY))

    story.append(Paragraph("3.2  ONNX Offline Plan (Critical Next Step)", H2))
    story.append(Paragraph(
        "The single biggest known risk to field deployment is internet dependency. Rural Thai roads frequently have no reliable 4G signal. The planned solution is to export the trained YOLO11 model from Roboflow as an ONNX file and run inference directly on the Raspberry Pi 4 using the ONNX Runtime library. This eliminates the Roboflow API call entirely and reduces per-frame latency from ~200-400ms (cloud round-trip) to an estimated 80-200ms (on-device, no network).",
        BODY))
    story += [
        bullet("Export: Roboflow UI → Versions → Export → ONNX format → download best.onnx"),
        bullet("Runtime: pip install onnxruntime (CPU) or onnxruntime-gpu (if Pi has accelerator)"),
        bullet("Replace infer_frame() in roadai.py with local ONNX session.run() call"),
        bullet("Expected Pi 4 throughput: 3-8 FPS on CPU (sufficient for 1-frame/sec survey capture)"),
        bullet("Fallback: if Pi FPS insufficient, keep inference on laptop, Pi streams only"),
    ]

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 4. DAMAGE CLASSIFICATION & COST ESTIMATION
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("4. DAMAGE CLASSIFICATION, METRICS &amp; COST ESTIMATION", H1))
    story.append(hr())

    story.append(Paragraph(
        "Every detected damage instance is automatically classified by type, measured physically, assigned a severity level, and given a Thai Baht repair cost estimate. This transforms raw AI detections into actionable maintenance data.",
        BODY))

    story.append(Paragraph("4.1  Damage Classes &amp; Repair Costs", H2))
    cost_data = [
        [Paragraph("Damage Class", H3), Paragraph("THB / m²", H3), Paragraph("Notes", H3)],
        [Paragraph("Pothole", BODY),             Paragraph("850", MONO), Paragraph("Most expensive — requires asphalt excavation and fill", BODY)],
        [Paragraph("Longitudinal Crack", BODY),  Paragraph("320", MONO), Paragraph("Surface sealing sufficient in most cases", BODY)],
        [Paragraph("Transverse Crack", BODY),    Paragraph("320", MONO), Paragraph("Similar treatment to longitudinal; planned class", BODY)],
        [Paragraph("Alligator Crack", BODY),     Paragraph("480", MONO), Paragraph("Structural fatigue; requires base repair", BODY)],
        [Paragraph("Surface Damage", BODY),      Paragraph("560", MONO), Paragraph("Ravelling / rutting; overlay treatment", BODY)],
        [Paragraph("Default (unknown)", BODY),   Paragraph("400", MONO), Paragraph("Fallback when class not in table", BODY)],
    ]
    cost_t = Table(cost_data, colWidths=[48*mm, 22*mm, 90*mm])
    cost_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",  (0,0), (-1,0), C_ACCENT),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_SURFACE, colors.HexColor("#111111")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#2a2a2a")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(cost_t)
    story.append(sp(8))

    story.append(Paragraph("4.2  Physical Measurement Pipeline", H2))
    story.append(Paragraph(
        "Pixel-to-centimetre conversion uses two calibration constants derived from a known reference object filmed at standard camera height:",
        BODY))
    story.append(Paragraph("PX_PER_CM_W = 27.06  (pixels per centimetre, horizontal)", MONO_BODY))
    story.append(Paragraph("PX_PER_CM_H = 29.58  (pixels per centimetre, vertical)", MONO_BODY))
    story.append(sp(4))
    story.append(Paragraph(
        "Bounding box area (fallback): width_px / PX_PER_CM_W  ×  height_px / PX_PER_CM_H = area_cm2. "
        "Segmentation area (preferred): OpenCV contourArea() on the polygon point list → area_px2 / (PX_PER_CM_W × PX_PER_CM_H) = area_cm2. "
        "Both converted to m2 by dividing by 10,000. Cost = area_m2 × cost_per_m2.",
        BODY))

    story.append(Paragraph("4.3  Severity Classification", H2))
    sev_data = [
        [Paragraph("Severity", H3), Paragraph("Area Threshold", H3), Paragraph("Map Pin Colour", H3), Paragraph("Meaning", H3)],
        [Paragraph("Low",    TAG_GREEN), Paragraph("< 100 cm²",  BODY), Paragraph("Yellow", BODY), Paragraph("Minor surface defect; monitor only",         BODY)],
        [Paragraph("Medium", TAG_WARN),  Paragraph("100–499 cm²",BODY), Paragraph("Orange", BODY), Paragraph("Significant damage; schedule for repair",     BODY)],
        [Paragraph("High",   TAG_RED),   Paragraph(">= 500 cm²", BODY), Paragraph("Red",    BODY), Paragraph("Urgent; safety risk; immediate action needed",BODY)],
    ]
    sev_t = Table(sev_data, colWidths=[22*mm, 30*mm, 30*mm, 78*mm])
    sev_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",  (0,0), (-1,0), C_ACCENT),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_SURFACE, colors.HexColor("#111111")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#2a2a2a")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(sev_t)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 5. BACKEND — roadai.py
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("5. BACKEND — roadai.py", H1))
    story.append(hr())
    story.append(Paragraph(
        "The backend is a multi-threaded Python Flask application (Python 3.12, Flask 3.x) that runs on the laptop at port 5000. It manages three concurrent threads and exposes a comprehensive REST API to the frontend.",
        BODY))

    story.append(Paragraph("5.1  Thread Architecture", H2))
    threads = [
        ("camera_worker",     "Reads frames from USB webcam via OpenCV at 30 fps (1280×720). Applies current annotations overlay and writes to output_frame for streaming. Runs independently of inference pace."),
        ("inference_worker",  "In AUTO mode: continuously takes latest_frame, sends to Roboflow API (or local ONNX), receives predictions, filters them, builds damage events, updates shared state. Idles in MANUAL mode."),
        ("Flask main thread", "Serves all HTTP endpoints. generate_stream() yields MJPEG frames from output_frame at 30 fps. All shared state access protected by a single threading.Lock()."),
    ]
    for name, desc in threads:
        story.append(KeepTogether([
            Paragraph(f"<b>{name}</b>", H3),
            Paragraph(desc, BODY),
        ]))

    story.append(Paragraph("5.2  Detection Modes", H2))
    story.append(Paragraph(
        "<b>AUTO mode</b> — inference_worker runs continuously. Each qualifying detection (confidence >= 60%, not a duplicate within 5s) creates a Damage Event automatically. Designed for driving surveys where the operator focuses on the road.",
        BODY))
    story.append(Paragraph(
        "<b>MANUAL mode</b> — inference_worker idles. The operator clicks 'CAPTURE &amp; DETECT' in the UI, which calls POST /capture. The current frame is frozen, inference runs once, results returned immediately with an annotated preview image.",
        BODY))

    story.append(Paragraph("5.3  Session (Survey) Management", H2))
    story.append(Paragraph(
        "Surveys are wrapped in Sessions. An operator starts a session (POST /session/start with name + GPS start point), drives the route, then ends it (POST /session/end). The system tracks: GPS path (array of lat/lng points), all damage events with their positions, total distance (km), cumulative cost, duration. Multiple sessions are stored in memory and displayed as distinct coloured paths on the map.",
        BODY))

    story.append(Paragraph("5.4  API Endpoints Reference", H2))
    api_data = [
        [Paragraph("Endpoint", H3), Paragraph("Method", H3), Paragraph("Purpose", H3)],
        [Paragraph("/video_feed",              MONO_BODY), Paragraph("GET",  BODY), Paragraph("MJPEG live camera stream (30 fps)",                                   BODY)],
        [Paragraph("/detections",              MONO_BODY), Paragraph("GET",  BODY), Paragraph("Latest frame's detection list with metrics",                           BODY)],
        [Paragraph("/events",                  MONO_BODY), Paragraph("GET",  BODY), Paragraph("Damage event history (lightweight, no thumbnails by default)",         BODY)],
        [Paragraph("/events?include_thumb=1",  MONO_BODY), Paragraph("GET",  BODY), Paragraph("Same as above but with base64 annotated thumbnails embedded",          BODY)],
        [Paragraph("/event/<id>",              MONO_BODY), Paragraph("GET",  BODY), Paragraph("Full event detail including all sub-detections + thumbnail",           BODY)],
        [Paragraph("/event/<id>/delete",       MONO_BODY), Paragraph("POST", BODY), Paragraph("Remove a false positive from history and all sessions",                BODY)],
        [Paragraph("/stats",                   MONO_BODY), Paragraph("GET",  BODY), Paragraph("Session summary: total detections, class counts, cost, area, uptime",  BODY)],
        [Paragraph("/history",                 MONO_BODY), Paragraph("GET",  BODY), Paragraph("Full flat detection history (all frames, not grouped into events)",     BODY)],
        [Paragraph("/mode",                    MONO_BODY), Paragraph("GET",  BODY), Paragraph("Get current detection mode (auto/manual)",                             BODY)],
        [Paragraph("/mode",                    MONO_BODY), Paragraph("POST", BODY), Paragraph("Set detection mode: {mode: 'auto'} or {mode: 'manual'}",               BODY)],
        [Paragraph("/capture",                 MONO_BODY), Paragraph("POST", BODY), Paragraph("Manual capture: freeze frame, run inference, return annotated result",  BODY)],
        [Paragraph("/clear",                   MONO_BODY), Paragraph("POST", BODY), Paragraph("Reset session detection history and counters",                         BODY)],
        [Paragraph("/session/start",           MONO_BODY), Paragraph("POST", BODY), Paragraph("Begin new survey: {name, lat, lng}",                                   BODY)],
        [Paragraph("/session/end",             MONO_BODY), Paragraph("POST", BODY), Paragraph("End active survey, compute distance, finalise session",                 BODY)],
        [Paragraph("/session/current",         MONO_BODY), Paragraph("GET",  BODY), Paragraph("Status of the currently active survey",                                BODY)],
        [Paragraph("/sessions",                MONO_BODY), Paragraph("GET",  BODY), Paragraph("All surveys (newest first) as map-ready summaries",                    BODY)],
        [Paragraph("/session/<id>",            MONO_BODY), Paragraph("GET",  BODY), Paragraph("Full detail for one survey",                                           BODY)],
        [Paragraph("/session/<id>/type/<type>",MONO_BODY), Paragraph("GET",  BODY), Paragraph("All events of a specific damage type within a survey (with thumbs)",   BODY)],
        [Paragraph("/session/<id>/delete",     MONO_BODY), Paragraph("POST", BODY), Paragraph("Delete an entire survey from memory",                                  BODY)],
        [Paragraph("/mock-locations",          MONO_BODY), Paragraph("GET",  BODY), Paragraph("8 preset Chiang Mai GPS points for testing without real GPS hardware",  BODY)],
    ]
    api_t = Table(api_data, colWidths=[52*mm, 16*mm, 92*mm])
    api_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",  (0,0), (-1,0), C_ACCENT),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_SURFACE, colors.HexColor("#111111")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#2a2a2a")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(api_t)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 6. FRONTEND — index.html
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("6. FRONTEND — index.html", H1))
    story.append(hr())
    story.append(Paragraph(
        "The frontend is a single self-contained HTML5 file (no build step, no npm, no framework — runs directly in a browser). It has a dark cyberpunk aesthetic (background #0a0a0a, accent #c8f135 lime-green) and is built with vanilla JavaScript + Leaflet.js for mapping. It polls the Flask backend every second for live data.",
        BODY))

    views = [
        ("Survey Map View",
         "The main dashboard. A full-viewport Leaflet.js map shows all completed and active survey routes as coloured polylines (each session gets a unique colour: cyan, magenta, lime, orange...). Start and end points marked with ▶ and ■ flag icons. Each damage event shown as a coloured circle marker sized by severity. Clicking a marker shows a popup with class, confidence, cost, and severity. A live sidebar shows current session stats. Operator can start/end surveys, pick GPS start location from a dropdown of preset Chiang Mai road coordinates."),
        ("Live Detection View",
         "Two-panel layout. Left: MJPEG stream from /video_feed with animated corner brackets and a REC indicator. Right sidebar: real-time metrics from the latest detection — damage class, confidence percentage, bounding box dimensions (cm × cm), segmentation area (cm² and m²), estimated repair cost (THB), severity badge. Metrics update every second. Mode toggle (AUTO / MANUAL) in the header."),
        ("Manual Capture View",
         "When in MANUAL mode, a large 'CAPTURE & DETECT' button triggers POST /capture. The UI freezes the annotated frame as a static preview image and displays the full detection breakdown. Designed for stationary inspections where the operator wants deliberate control over when inference fires."),
        ("Damage History View",
         "A scrollable grid of all damage events from the current session. Each card shows: annotated thumbnail (320×180), damage class badge, severity indicator, confidence %, area (m²), and cost (฿). Cards filterable by severity or class. Operator can delete false positives (calls POST /event/<id>/delete). Newest events appear first."),
        ("Session Detail View",
         "Activated by clicking a session on the map. Shows the survey's summary statistics: total events, total cost (THB), total area (m²), distance driven (km), duration. A breakdown table shows count by damage type. Each damage type row is clickable → opens Gallery View."),
        ("Damage Gallery View",
         "Shows all photos of one damage type within one session. Full annotated thumbnails in a grid. Useful for preparing documentation or checking a specific damage category across an entire route."),
    ]
    for vname, vdesc in views:
        story.append(KeepTogether([
            Paragraph(vname, H2),
            Paragraph(vdesc, BODY),
            sp(4),
        ]))

    story.append(Paragraph("6.1  Offline Demo Mode", H2))
    story.append(Paragraph(
        "If the frontend cannot reach the Flask backend (HTTP error or no connection), it automatically switches to Demo Mode. A warning pill appears in the header. All five views remain fully functional, populated with realistic fake data: multiple sessions, multiple damage types, GPS paths across Chiang Mai roads, cost breakdowns. This ensures the system can be demonstrated at competitions or meetings without requiring a live camera connection.",
        BODY))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 7. HARDWARE
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("7. HARDWARE STACK", H1))
    story.append(hr())

    hw_rows = [
        ("Edge compute",        "Raspberry Pi 4 Model B (4GB RAM recommended) — Ubuntu 22.04 LTS aarch64, Python 3.11"),
        ("Camera",              "Logitech USB webcam (1080p capable, used at 1280×720 for balance of quality vs. bandwidth). OpenCV VideoCapture index 0 or 1."),
        ("Dev machine",         "Windows 11 laptop — Python 3.12, Flask 5050/5000, VS Code with Remote-SSH extension connected to Pi"),
        ("GPS module",          "NEO-6M UART GPS (acquired, not yet wired) — will replace GPS simulation in advance_gps()"),
        ("Connectivity",        "4G LTE USB dongle (status: not yet tested) — for cloud upload in field. System designed to work fully offline without it."),
        ("Power",               "Plan: 12V→5V USB-C stepdown from vehicle power or portable power bank (min 10,000 mAh for 4+ hour survey)"),
        ("Mounting",            "Planned: dashcam-style mount on windscreen or external roof mount for clear downward road view"),
        ("Network (dev)",       "Both Pi and laptop on same Wi-Fi network. Pi IP: 192.168.1.109. Laptop IP found via ipconfig/ifconfig."),
    ]
    story.append(kv_table(hw_rows))
    story.append(sp(8))

    story.append(Paragraph("7.1  Pi Streaming Pipeline (Current Implementation)", H2))
    story.append(Paragraph(
        "pi_capture.py runs on the Pi and sends frames to the laptop. It opens the webcam with OpenCV, captures one frame per second, JPEG-encodes it (quality 85), and POSTs it to http://<LAPTOP_IP>:5050/upload as multipart form data. receiver_server.py on the laptop accepts these frames. Connection errors are caught gracefully — the Pi logs the error and retries on the next cycle without crashing. If webcam index 0 fails, the script automatically tries index 1.",
        BODY))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 8. CURRENT STATE & KNOWN ISSUES
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("8. CURRENT STATE &amp; KNOWN ISSUES", H1))
    story.append(hr())

    story.append(Paragraph("What is Working Right Now", H2))
    working = [
        "Raspberry Pi 4 set up with Ubuntu 22.04 LTS, SSH accessible from laptop",
        "USB webcam streaming from Pi to laptop via pi_capture.py → receiver_server.py over local Wi-Fi",
        "YOLO11 inference pipeline in roadai.py functional on laptop webcam (Roboflow cloud mode)",
        "83.1% mAP@50 on 2 trained classes (Pothole, Longitudinal Crack)",
        "Full 5-view HTML dashboard working in browser with live data and demo mode",
        "Session management, GPS simulation, coloured route map, damage history, gallery view",
        "Cost estimation (THB) and severity classification for all detected events",
        "Annotated thumbnail generation and storage for each damage event",
        "False positive deletion, session deletion, auto/manual mode switching",
        "VS Code Remote-SSH workflow established for coding directly on Pi from laptop",
    ]
    for w in working:
        story.append(bullet(w))

    story.append(sp(6))
    story.append(Paragraph("Known Issues &amp; Gaps", H2))
    issues = [
        ("CRITICAL — Offline inference",  "System still calls Roboflow cloud API. Fails without internet. Fix: ONNX export + on-device inference. This is the #1 priority."),
        ("CRITICAL — No real GPS",        "GPS position is simulated with advance_gps() (random walk ~10m steps). NEO-6M module acquired but not wired. Positions on map are approximate/fake."),
        ("HIGH — No database",            "All detections and sessions stored in Python deque in RAM. Lost on restart. Fix: SQLite with SQLAlchemy for persistence across reboots."),
        ("HIGH — Form 1 not implemented", "Thai government repair request form (Form 1 / แบบฟอร์มที่ 1) not yet generated. This is a key deliverable for the target agencies."),
        ("MEDIUM — Map tile dependency",  "Leaflet uses CARTO CDN tiles. Black map if CDN blocked. Fix: local tile server or offline tile cache."),
        ("MEDIUM — 4G not tested",        "4G dongle cloud upload path unverified in field conditions."),
        ("LOW — Pi FPS unknown",          "Inference speed on Pi hardware not benchmarked. May need FPS optimisation or relay model to laptop."),
        ("LOW — Single camera angle",     "One webcam captures one view. Wider coverage may require multiple cameras or panoramic lens."),
    ]
    issues_data = [[Paragraph(s, TAG_WARN if "CRITICAL" not in s and "HIGH" not in s else TAG_RED if "CRITICAL" in s else TAG_WARN),
                    Paragraph(d, BODY)] for s, d in issues]
    issues_t = Table(issues_data, colWidths=[42*mm, 118*mm])
    issues_t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_SURFACE, colors.HexColor("#111111")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#2a2a2a")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(issues_t)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 9. ROADMAP
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("9. DEVELOPMENT ROADMAP", H1))
    story.append(hr())

    roadmap = [
        ("1", "ONNX Export & Offline Inference",
         "Export YOLO11 model from Roboflow to ONNX. Integrate onnxruntime into roadai.py. Replace infer_frame() HTTP call with local session.run(). Benchmark FPS on Pi. Target: fully offline inference with no cloud dependency.",
         C_DANGER),
        ("2", "Real GPS Integration (NEO-6M)",
         "Wire NEO-6M GPS module to Pi via UART (TX→Pi RX, RX→Pi TX, 3.3V, GND). Install gpsd + pynmea2. Replace advance_gps() simulation with real NMEA sentence parsing. Add ~10m GPS jitter filtering (Kalman or moving average).",
         C_DANGER),
        ("3", "SQLite Database Persistence",
         "Replace in-memory deques with SQLAlchemy + SQLite. Schema: sessions table, events table, detections table. Surveys survive reboots. Add export endpoints for JSON and CSV. Critical for real-world deployments.",
         C_WARN),
        ("4", "Form 1 PDF Generation",
         "Thai government repair request form (แบบฟอร์มที่ 1). Pre-fill from session data: road name, date, GPS coordinates, damage type table, estimated costs, total area. Generate as PDF using reportlab. Operator reviews and submits digitally.",
         C_WARN),
        ("5", "Expand Model Classes",
         "Train YOLO11 on 4 additional damage types: Transverse Crack, Alligator Crack, Surface Damage (ravelling/rutting), Edge Break. Source training data from Thai road surveys. Target: 6-class model at >= 80% mAP@50.",
         C_INFO),
        ("6", "4G Cloud Upload & Sync",
         "Test 4G dongle in field. Implement background sync queue: detections POST to cloud backend when connectivity available. Handle intermittent connection gracefully (queue, retry, deduplicate).",
         C_INFO),
        ("7", "Field Trial — Chiang Mai Roads",
         "Mount hardware in vehicle. Survey 3-5 real roads around Chiang Mai (Huay Kaew Rd, Nimman Rd, Suthep Rd target routes). Collect ground truth vs. AI detection comparison. Document accuracy, missed detections, false positives.",
         C_GREEN),
        ("8", "Competition Submissions",
         "NSC 2026 submission (Category 23) — deadline tracked. KMUTNB TIA 26th preparation. CEDT Innovation Summit 2026 poster and demo materials. Prepare 3-minute pitch video demonstrating end-to-end pipeline from driving survey to Form 1 output.",
         C_GREEN),
    ]

    for num, title, desc, col in roadmap:
        header = Table([[
            Paragraph(f"{num}", ParagraphStyle("rn", parent=H1, textColor=col, fontSize=14, fontName="Helvetica-Bold", spaceBefore=0, spaceAfter=0)),
            Paragraph(title, ParagraphStyle("rt", parent=H2, textColor=C_WHITE, spaceBefore=0, spaceAfter=0)),
        ]], colWidths=[12*mm, 148*mm])
        header.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#1a1a1a")),
            ("LINEABOVE", (0,0), (-1,-1), 1.5, col),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING",  (0,0), (-1,-1), 8),
            ("TOPPADDING",   (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([header, Paragraph(desc, BODY), sp(4)]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 10. TECH STACK SUMMARY
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("10. TECHNOLOGY STACK SUMMARY", H1))
    story.append(hr())

    stack_data = [
        [Paragraph("Component", H3), Paragraph("Technology", H3), Paragraph("Version / Notes", H3)],
        [Paragraph("AI Model",           BODY), Paragraph("YOLO11 Instance Segmentation", MONO_BODY), Paragraph("via Ultralytics + Roboflow; ONNX target", BODY)],
        [Paragraph("Model Training",     BODY), Paragraph("Roboflow", MONO_BODY),                     Paragraph("Cloud training platform; 83.1% mAP@50", BODY)],
        [Paragraph("Backend language",   BODY), Paragraph("Python 3.12 (laptop) / 3.11 (Pi)", MONO_BODY), Paragraph("Flask 3.x, Flask-CORS, python-dotenv", BODY)],
        [Paragraph("Camera library",     BODY), Paragraph("OpenCV (cv2)", MONO_BODY),                 Paragraph("VideoCapture, imencode, drawing, contourArea", BODY)],
        [Paragraph("HTTP framework",     BODY), Paragraph("Flask", MONO_BODY),                        Paragraph("MJPEG streaming, REST API, CORS", BODY)],
        [Paragraph("Pi capture script",  BODY), Paragraph("Python requests + OpenCV", MONO_BODY),      Paragraph("pi_capture.py — 1 fps, multipart POST", BODY)],
        [Paragraph("Frontend",           BODY), Paragraph("HTML5 / Vanilla JS / CSS3", MONO_BODY),    Paragraph("No framework; single file; no build step", BODY)],
        [Paragraph("Mapping",            BODY), Paragraph("Leaflet.js 1.9.4", MONO_BODY),             Paragraph("CARTO tiles; polylines + markers + popups", BODY)],
        [Paragraph("GPS (current)",      BODY), Paragraph("Simulated (advance_gps)", MONO_BODY),       Paragraph("Random walk ~10m steps; NEO-6M target", BODY)],
        [Paragraph("GPS (target)",       BODY), Paragraph("NEO-6M + gpsd + pynmea2", MONO_BODY),      Paragraph("UART NMEA sentences parsed on Pi", BODY)],
        [Paragraph("Storage (current)",  BODY), Paragraph("Python deque (in-memory)", MONO_BODY),     Paragraph("maxlen=300 events, lost on restart", BODY)],
        [Paragraph("Storage (target)",   BODY), Paragraph("SQLite + SQLAlchemy", MONO_BODY),           Paragraph("Persistent across reboots; exportable", BODY)],
        [Paragraph("PDF generation",     BODY), Paragraph("reportlab (planned)", MONO_BODY),           Paragraph("Form 1 Thai government repair request", BODY)],
        [Paragraph("Edge hardware",      BODY), Paragraph("Raspberry Pi 4 (4GB)", MONO_BODY),          Paragraph("Ubuntu 22.04 LTS aarch64; Pi capture + (future) inference", BODY)],
        [Paragraph("Development IDE",    BODY), Paragraph("VS Code + Remote-SSH", MONO_BODY),          Paragraph("Code on Pi from laptop; integrated terminal", BODY)],
        [Paragraph("Connectivity",       BODY), Paragraph("Local Wi-Fi (dev) / 4G dongle (field)", MONO_BODY), Paragraph("Offline ONNX mode removes cloud dependency", BODY)],
    ]
    stack_t = Table(stack_data, colWidths=[38*mm, 50*mm, 72*mm])
    stack_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",  (0,0), (-1,0), C_ACCENT),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_SURFACE, colors.HexColor("#111111")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#2a2a2a")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(stack_t)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 11. VISION & IMPACT
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("11. VISION, IMPACT &amp; FUTURE DIRECTION", H1))
    story.append(hr())

    story.append(Paragraph("Long-Term Vision", H2))
    story.append(Paragraph(
        "ROAD AI is designed to scale from a student competition prototype to a production tool used daily by Thai road maintenance agencies. The long-term vision is a network of Pi-equipped survey vehicles — motorcycles, pickup trucks, road maintenance crew vehicles — each streaming inspection data to a central dashboard. Municipal engineers see damage accumulate on a live map in real time, prioritise repairs by severity and cost, and generate Form 1 documents with one click instead of hours of manual paperwork.",
        BODY))

    story.append(Paragraph("Impact Metrics (Target)", H2))
    impacts = [
        ("Survey speed",        "Manual inspection: ~2 km/hour. ROAD AI: 40+ km/hour (highway speed)"),
        ("Documentation time",  "Manual Form 1: 2-4 hours per survey. ROAD AI: instant auto-generation"),
        ("Missed damage",       "Human surveys miss ~30% of early-stage cracks. AI detects consistently regardless of inspector fatigue"),
        ("Cost savings",        "Early detection of cracks (฿320/m²) prevents progression to potholes (฿850/m²)"),
        ("Coverage",            "One Pi kit costs ~฿3,000–฿5,000. Scalable to every municipality at low cost"),
        ("Data quality",        "GPS-tagged, timestamped, photo-evidenced records replace handwritten notes"),
    ]
    story.append(kv_table(impacts))
    story.append(sp(8))

    story.append(Paragraph("Design Philosophy", H2))
    principles = [
        "Offline-first: must work on rural roads with no internet. Cloud is optional, not required.",
        "Low operator burden: AUTO mode requires no interaction while driving. Zero training for basic use.",
        "Thai context: cost tables, severity thresholds, and Form 1 output designed for Thai government requirements.",
        "Modular inference: swapping Roboflow → ONNX → any future model requires changing one function (infer_frame).",
        "Graceful degradation: if camera fails, if GPS fails, if cloud fails — each component fails independently without crashing the system.",
        "Transparent AI: every detection shows confidence %, class name, and physical measurements so operators can judge and override.",
    ]
    for p in principles:
        story.append(bullet(p))

    story.append(sp(8))
    story.append(hr())
    story.append(sp(4))
    story.append(Paragraph(
        "This document was generated automatically from the ROAD AI codebase on June 2026. "
        "It is intended as a complete context brief for AI models assisting with development of this project. "
        "All technical details are current as of the document date. "
        "For the most up-to-date state, refer to edge/roadai.py, index.html, pi_side/pi_capture.py, and receiver_server.py.",
        LABEL))

    doc.build(story)
    print(f"Generated: {OUTPUT}")

if __name__ == "__main__":
    build()
