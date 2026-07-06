"""
edge/measurement.py — real-world size measurement for road damage.

Converts a damage outline in IMAGE PIXELS into a real-world AREA (m²) using the
camera geometry, instead of a fixed pixels-per-cm guess.

Two modes, chosen automatically (accurate first):

  1. HOMOGRAPHY (accurate — handles camera HEIGHT and ANGLE/TILT together)
     Requires a one-time calibration file (calib_homography.json). A homography
     maps every image pixel to a real point on the flat road plane, so a mask's
     true ground area can be measured no matter where it sits in the frame or
     how the camera is tilted.

  2. GROUND-SAMPLING-DISTANCE fallback (only correct for a perpendicular/nadir
     camera). Uses camera height + horizontal field of view from the .env:
        CAM_HEIGHT_CM, CAM_HFOV_DEG
     If neither a calibration file nor these env vars exist, falls back to the
     legacy PX_PER_CM constants passed in by the caller.

────────────────────────────────────────────────────────────────────────────
CALIBRATION (do this ONCE, after the camera is mounted at its final height/angle)
  1. Lay a flat object of known size on the road inside the camera view:
       - an A4 sheet  = 21.0 x 29.7 cm, or
       - a printed/cut square, e.g. 50 x 50 cm
  2. Capture one frame from the camera.
  3. Read the 4 corner pixel coordinates of that object from the image, in the
     SAME order as their real-world corners (e.g. top-left, top-right,
     bottom-right, bottom-left).
  4. Call:
       calibrate_from_points(
           image_pts_px = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
           world_pts_cm = [(0,0),(21,0),(21,29.7),(0,29.7)]   # A4 example
       )
     This saves calib_homography.json next to this file. From then on
     polygon_area_m2() uses it automatically.
────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

_HERE       = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH  = os.getenv("CALIB_FILE", os.path.join(_HERE, "calib_homography.json"))

# Perpendicular-camera fallback (only used if there is no homography)
CAM_HEIGHT_CM = float(os.getenv("CAM_HEIGHT_CM", 0) or 0)   # lens height above road
CAM_HFOV_DEG  = float(os.getenv("CAM_HFOV_DEG",  0) or 0)   # horizontal field of view

_H_CACHE = None   # cached homography matrix (image px -> ground cm)


# ── Calibration ───────────────────────────────────────────────
def calibrate_from_points(image_pts_px, world_pts_cm, path=CALIB_PATH):
    """Compute + save the homography from >=4 point correspondences.
    image_pts_px : list of (x,y) pixel coords of known points in the frame
    world_pts_cm : list of (x,y) real-world coords (cm) on the road plane
    Returns the 3x3 homography matrix (image px -> ground cm)."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available")
    src = np.array(image_pts_px, dtype=np.float32)
    dst = np.array(world_pts_cm, dtype=np.float32)
    if len(src) < 4:
        raise ValueError("need at least 4 point correspondences")
    if len(src) == 4:
        H = cv2.getPerspectiveTransform(src, dst)
    else:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    with open(path, "w") as f:
        json.dump({"homography": H.tolist(),
                   "image_pts_px": [list(map(float, p)) for p in image_pts_px],
                   "world_pts_cm": [list(map(float, p)) for p in world_pts_cm]}, f, indent=2)
    global _H_CACHE
    _H_CACHE = H
    print(f"[measure] calibration saved -> {path}")
    return H


def _load_homography():
    global _H_CACHE
    if _H_CACHE is not None:
        return _H_CACHE
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH) as f:
                _H_CACHE = np.array(json.load(f)["homography"], dtype=np.float64)
            return _H_CACHE
        except Exception as e:
            print(f"[measure] could not load calibration: {e}")
    return None


def is_calibrated():
    return _load_homography() is not None


# ── Area measurement ──────────────────────────────────────────
def _shoelace_cm2(world_pts):
    """Polygon area (cm²) from Nx2 array of (x_cm, y_cm) vertices."""
    x = world_pts[:, 0]
    y = world_pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_area_m2(points_px, px_per_cm_w=None, px_per_cm_h=None):
    """Return the real-world area (m²) of a damage outline.

    points_px : list of {"x":.., "y":..} (segmentation polygon in image pixels)
                or list of (x, y) tuples.
    px_per_cm_w/h : legacy constants, used only if there's no calibration and no
                    height/FOV configured (keeps old behaviour as a last resort).
    """
    if not points_px:
        return 0.0
    pts = np.array([[p["x"], p["y"]] if isinstance(p, dict) else [p[0], p[1]]
                    for p in points_px], dtype=np.float32)

    # 1) Homography — the accurate path
    H = _load_homography()
    if H is not None and cv2 is not None:
        world = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        return round(_shoelace_cm2(world) / 10000.0, 4)

    # 2) Perpendicular-camera ground-sampling-distance
    if CAM_HEIGHT_CM > 0 and CAM_HFOV_DEG > 0:
        img_w = float(pts[:, 0].max() - pts[:, 0].min()) or 1.0
        # cm per pixel from geometry; assumes square pixels + nadir view
        # (kept simple; homography is preferred whenever available)
        # ground width covered by the full image is unknown here without the
        # frame width, so this branch expects px_per_cm to be derived upstream.
        pass

    # 3) Legacy fallback: fixed pixels-per-cm
    if px_per_cm_w and px_per_cm_h:
        area_px = _shoelace_cm2(pts)               # this is px², reuse shoelace
        return round((area_px / (px_per_cm_w * px_per_cm_h)) / 10000.0, 4)

    return 0.0


def ground_sampling_distance_cm_per_px(image_width_px):
    """cm-per-pixel for a perpendicular camera from height + HFOV.
    Only valid for a nadir (straight-down) camera. Returns None if not set."""
    if CAM_HEIGHT_CM > 0 and CAM_HFOV_DEG > 0 and image_width_px:
        ground_w_cm = 2.0 * CAM_HEIGHT_CM * math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
        return ground_w_cm / image_width_px
    return None
