"""
edge/cost.py — deterministic road-repair cost estimation.

Uses the official per-unit repair costs extracted from the Department of Rural
Roads maintenance manual (edge/cost_rates.json). The AI's job was to read the
manual ONCE and produce that rate table; at runtime the cost is a simple,
reproducible, auditable calculation:

    cost = damaged_area (m²) × per-unit rate (THB/m²) for the repair method

The repair method is chosen from the detected damage class + severity via the
mapping in cost_rates.json (easily edited without touching code).

Every estimate carries its source (repair method + rate + manual page), so a
cost can always be traced back to the manual line it came from.
"""

import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
RATES_PATH = os.getenv("COST_RATES", os.path.join(_HERE, "cost_rates.json"))

_RATES = None


def _load():
    global _RATES
    if _RATES is None:
        with open(RATES_PATH, encoding="utf-8") as f:
            _RATES = json.load(f)
    return _RATES


def _match_class(cls):
    """Map a detected class string to a mapping key (substring, case-insensitive)."""
    mapping = _load()["damage_to_repair_mapping"]
    key = (cls or "").lower()
    for k in mapping:
        if k == "_comment" or k == "default":
            continue
        if k in key or key in k:
            return k
    return "default"


def estimate_repair_cost(damage_class, severity, area_m2):
    """Return a cost estimate dict for one damage detection.

    damage_class : e.g. "pothole", "Longitudinal Crack", "หลุมบ่อ"
    severity     : "low" | "medium" | "high"
    area_m2      : damaged area in square metres

    Returns cost + repair method + source, plus PHYSICAL QUANTITIES:
      repair_volume_m3, repair_mass_kg  (material needed to patch the layer)
    """
    rates  = _load()
    area   = float(area_m2 or 0)
    sev    = (severity or "medium").lower()

    cls_key    = _match_class(damage_class)
    method_key = rates["damage_to_repair_mapping"][cls_key].get(sev) \
                 or rates["damage_to_repair_mapping"]["default"].get(sev, "skin_patching")
    method     = rates["per_unit_repair_cost"].get(method_key, {})

    rate = method.get("thb")
    unit = method.get("unit", "m2")

    if rate is None:              # e.g. crack_sealing has no fixed rate configured
        cost = 0.0
    elif unit == "m2":
        cost = area * rate
    else:                         # per-metre methods would need a length, not area
        cost = 0.0

    # ── Physical quantities (physics layer) ──────────────────────────────
    pq        = rates.get("physical_quantities", {})
    thickness = pq.get("repair_layer_thickness_m", {}).get(method_key, 0.0)
    density   = pq.get("material_density_kg_m3", {}).get("default", 2200)
    repair_volume_m3 = area * thickness           # V = area × layer thickness
    repair_mass_kg   = repair_volume_m3 * density  # m = ρV

    src = rates.get("_source", {})
    return {
        "cost_thb":         round(cost),
        "repair_method":    method.get("en", method_key),
        "repair_method_th": method.get("th", ""),
        "rate_thb":         rate,
        "rate_unit":        unit,
        "cost_source":      f"{src.get('agency','')} — {src.get('chapter','')} (p.{method.get('page','?')})",
        "repair_volume_m3": round(repair_volume_m3, 5),
        "repair_mass_kg":   round(repair_mass_kg, 2),
    }
