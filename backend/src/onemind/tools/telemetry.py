"""Remote monitoring data plane - device telemetry time series.

Only the Remote Monitoring specialist holds these tools.

Trend and threshold arithmetic happens here, in Python, not in the model. A 4B
model asked to average twenty-one floats will produce a plausible number rather
than the correct one; asked to interpret a computed mean it does fine. Keeping
the numerics in the tool is what makes the answers trustworthy.
"""

from __future__ import annotations

from typing import Any

from . import store
from .base import obj_schema, tool, tools


def _devices_for(patient_id: str = "", metric: str = "") -> list[dict[str, Any]]:
    rows = store.devices()
    if patient_id:
        rows = [d for d in rows if d["patient_id"] == str(patient_id).strip()]
    if metric:
        needle = metric.strip().lower()
        rows = [d for d in rows if needle in d["metric"].lower()]
    return rows


@tool(
    tools,
    name="telemetry_series",
    description=(
        "Fetch recent device readings for a patient, with computed mean, min, "
        "max, and the direction of the recent trend. Use this for questions "
        "about vitals readings over time."
    ),
    parameters=obj_schema(
        {
            "patient_id": {"type": "string", "description": "Patient identifier"},
            "metric": {
                "type": "string",
                "description": "e.g. blood_pressure_systolic, spo2, heart_rate, blood_glucose",
            },
            "days": {"type": "integer", "description": "Lookback window, default 14"},
        }
    ),
)
def telemetry_series(patient_id: str = "", metric: str = "", days: int = 14) -> dict[str, Any]:
    targets = _devices_for(patient_id, metric)
    if not targets:
        # Metric names are schema and safe to return - they help the model
        # retry with a valid one. The patient ids they used to be paired with
        # were not: that turned a miss into a roster of who is monitored.
        return {
            "found": False,
            "detail": "no monitored device matches that patient and metric",
            "available_metrics": sorted({d["metric"] for d in store.devices()}),
        }

    window = max(1, min(days, 21))
    out = []
    for device in targets:
        readings = [r for r in store.telemetry() if r["device_id"] == device["device_id"]]
        readings.sort(key=lambda r: r["recorded_at"])
        readings = readings[-window:]
        if not readings:
            continue

        values = [r["value"] for r in readings]
        half = max(1, len(values) // 2)
        earlier = sum(values[:half]) / half
        later = sum(values[-half:]) / half
        delta = later - earlier
        # 3% of the earlier mean is the noise floor these fixtures sit above.
        if abs(delta) < 0.03 * max(abs(earlier), 1):
            direction = "flat"
        else:
            direction = "rising" if delta > 0 else "falling"

        out.append(
            {
                "device_id": device["device_id"],
                "patient_id": device["patient_id"],
                "metric": device["metric"],
                "unit": device["unit"],
                "alert_threshold": device["alert_threshold"],
                "alert_direction": device["alert_direction"],
                "reading_count": len(values),
                "window_days": window,
                "mean": round(sum(values) / len(values), 1),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
                "latest": readings[-1]["value"],
                "latest_recorded_at": readings[-1]["recorded_at"],
                "trend": direction,
                "trend_delta": round(delta, 1),
                "readings": [
                    {"recorded_at": r["recorded_at"], "value": r["value"]} for r in readings
                ],
            }
        )
    return {"found": bool(out), "count": len(out), "series": out}


@tool(
    tools,
    name="evaluate_thresholds",
    description=(
        "Check recent readings against the configured alert threshold and "
        "report which readings breached it and how sustained the breach was."
    ),
    parameters=obj_schema(
        {
            "patient_id": {"type": "string", "description": "Patient identifier"},
            "metric": {"type": "string", "description": "Metric name"},
            "days": {"type": "integer", "description": "Lookback window, default 7"},
        }
    ),
)
def evaluate_thresholds(patient_id: str = "", metric: str = "", days: int = 7) -> dict[str, Any]:
    series = telemetry_series(patient_id=patient_id, metric=metric, days=days)
    if not series.get("found"):
        return series

    out = []
    for entry in series["series"]:
        threshold = entry["alert_threshold"]
        direction = entry["alert_direction"]

        # Direction is not cosmetic. SpO2 alerts on a floor breach; blood
        # pressure alerts on a ceiling breach. Testing everything against an
        # upper bound reports healthy oxygen saturation as an alert and misses
        # genuine hypoxaemia entirely.
        def breached(value: float, _t: float = threshold, _d: str = direction) -> bool:
            return value < _t if _d == "below" else value > _t

        breaches = [r for r in entry["readings"] if breached(r["value"])]

        # Longest run of consecutive breaching readings - one spike is noise,
        # several in a row is a clinical signal.
        longest = run = 0
        for reading in entry["readings"]:
            run = run + 1 if breached(reading["value"]) else 0
            longest = max(longest, run)

        extreme = entry["min"] if direction == "below" else entry["max"]

        out.append(
            {
                "device_id": entry["device_id"],
                "patient_id": entry["patient_id"],
                "metric": entry["metric"],
                "unit": entry["unit"],
                "threshold": threshold,
                "alert_direction": direction,
                "breach_rule": (
                    f"value {'<' if direction == 'below' else '>'} {threshold} {entry['unit']}"
                ),
                "readings_evaluated": entry["reading_count"],
                "breach_count": len(breaches),
                "longest_consecutive_breach": longest,
                "worst_observed": extreme,
                "trend": entry["trend"],
                "alert": longest >= 3 or len(breaches) > entry["reading_count"] / 2,
                "breaching_readings": breaches[:10],
            }
        )
    return {"found": True, "count": len(out), "evaluations": out}
