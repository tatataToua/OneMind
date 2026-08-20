"""Generate the synthetic demo corpus.

Everything under fixtures/ is produced by this script from a fixed seed. No real
patient data has ever been near this repository, and the generator is committed
so that claim is checkable rather than asserted.

Run:  python fixtures/generate.py
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 20260820
HERE = Path(__file__).parent

FIRST = ["Dana", "Marcus", "Priya", "Elena", "Samuel", "Nia", "Tobias", "Rosa"]
LAST = ["Whitfield", "Okafor", "Raman", "Vasquez", "Lindqvist", "Boateng", "Ferreira", "Kaur"]

CONDITIONS = [
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I10", "Essential (primary) hypertension"),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
    ("N18.3", "Chronic kidney disease, stage 3"),
    ("I48.91", "Unspecified atrial fibrillation"),
]

MEDS = [
    ("Metformin", "500 mg", "twice daily"),
    ("Lisinopril", "10 mg", "once daily"),
    ("Atorvastatin", "20 mg", "at bedtime"),
    ("Insulin glargine", "18 units", "at bedtime"),
    ("Apixaban", "5 mg", "twice daily"),
    ("Tiotropium", "18 mcg", "once daily"),
]

LABS = [
    ("Hemoglobin A1c", "%", 5.4, 11.2, 5.7),
    ("eGFR", "mL/min/1.73m2", 22.0, 98.0, 60.0),
    ("LDL cholesterol", "mg/dL", 60.0, 190.0, 100.0),
    ("Potassium", "mmol/L", 3.1, 5.9, 5.1),
]

DENIALS = [
    ("CO-197", "Precertification/authorization absent"),
    ("CO-16", "Claim lacks information needed for adjudication"),
    ("CO-11", "Diagnosis inconsistent with the procedure"),
    ("CO-29", "Time limit for filing has expired"),
    ("PR-204", "Service not covered under the patient's current benefit plan"),
]

CPT = [
    ("99214", "Office visit, established patient, moderate complexity"),
    ("95250", "Continuous glucose monitoring, sensor placement and training"),
    ("E0784", "External ambulatory insulin infusion pump"),
    ("93000", "Electrocardiogram, routine ECG with interpretation"),
    ("99457", "Remote physiologic monitoring treatment management, first 20 minutes"),
]


def build(rng: random.Random) -> dict[str, object]:
    today = date(2026, 8, 20)
    patients = []
    for i in range(6):
        pid = f"{12345 + i * 111}"
        dob = date(1948 + rng.randint(0, 45), rng.randint(1, 12), rng.randint(1, 28))
        conds = rng.sample(CONDITIONS, k=rng.randint(1, 3))
        meds = rng.sample(MEDS, k=rng.randint(2, 4))
        labs = []
        for name, unit, lo, hi, ref in rng.sample(LABS, k=rng.randint(2, 4)):
            for back in (240, 120, 20):
                labs.append(
                    {
                        "name": name,
                        "value": round(rng.uniform(lo, hi), 1),
                        "unit": unit,
                        "reference_upper": ref,
                        "effective_date": (today - timedelta(days=back)).isoformat(),
                    }
                )
        patients.append(
            {
                "patient_id": pid,
                "mrn": f"MRN-{rng.randint(100000, 999999)}",
                "ssn": f"{rng.randint(100,899)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}",
                "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "birth_date": dob.isoformat(),
                "gender": rng.choice(["female", "male"]),
                "phone": f"(555) {rng.randint(200,999)}-{rng.randint(1000,9999)}",
                "conditions": [{"code": c, "display": d} for c, d in conds],
                "medications": [
                    {"name": n, "dose": d, "frequency": f} for n, d, f in meds
                ],
                "labs": labs,
                "encounters": [
                    {
                        "encounter_id": f"ENC-{rng.randint(1000,9999)}",
                        "date": (today - timedelta(days=rng.randint(5, 300))).isoformat(),
                        "type": rng.choice(["ambulatory", "telehealth", "emergency"]),
                        "reason": rng.choice([d for _, d in conds]),
                    }
                    for _ in range(rng.randint(1, 3))
                ],
            }
        )

    claims = []
    for i in range(14):
        patient = rng.choice(patients)
        code, code_desc = rng.choice(CPT)
        denied = rng.random() < 0.5
        denial = rng.choice(DENIALS) if denied else None
        claims.append(
            {
                "claim_id": f"CLM-{8840 + i * 3}",
                "patient_id": patient["patient_id"],
                "date_of_service": (today - timedelta(days=rng.randint(10, 200))).isoformat(),
                "cpt_code": code,
                "cpt_description": code_desc,
                "icd10_code": patient["conditions"][0]["code"],
                "billed_amount": round(rng.uniform(120, 4200), 2),
                "allowed_amount": 0.0 if denied else round(rng.uniform(80, 3000), 2),
                "status": "denied" if denied else "paid",
                "denial_code": denial[0] if denial else None,
                "denial_reason": denial[1] if denial else None,
                "payer": rng.choice(["Meridian Health", "Cascade Mutual", "Northwind Care"]),
            }
        )

    devices = []
    series = []
    # Direction matters clinically: hypertension is a ceiling breach, hypoxaemia
    # is a floor breach. A single "upper threshold" field silently inverts the
    # meaning of every SpO2 alert.
    metrics = [
        ("blood_pressure_systolic", "mmHg", 118, 172, 140, "above"),
        ("spo2", "%", 86, 99, 92, "below"),
        ("heart_rate", "bpm", 52, 118, 100, "above"),
        ("blood_glucose", "mg/dL", 70, 310, 180, "above"),
    ]
    # One patient per metric, so the demo always has a floor-breach case to show.
    for patient, (metric, unit, lo, hi, threshold, direction) in zip(patients, metrics):
        device_id = f"DEV-{rng.randint(1000, 9999)}"
        devices.append(
            {
                "device_id": device_id,
                "patient_id": patient["patient_id"],
                "metric": metric,
                "unit": unit,
                "alert_threshold": threshold,
                "alert_direction": direction,
            }
        )
        drift = rng.random() < 0.6
        for day in range(21, 0, -1):
            base = rng.uniform(lo, hi)
            if drift and day <= 4:
                # Drift toward the breach side, whichever side that is.
                push = (5 - day) * rng.uniform(3, 9)
                if direction == "above":
                    base = min(hi, base + push)
                else:
                    base = max(lo, base - push * 0.35)
            stamp = datetime(2026, 8, 20, 8, 0) - timedelta(days=day)
            series.append(
                {
                    "device_id": device_id,
                    "patient_id": patient["patient_id"],
                    "metric": metric,
                    "unit": unit,
                    "value": round(base, 1),
                    "recorded_at": stamp.isoformat(),
                }
            )

    return {
        "patients": patients,
        "claims": claims,
        "devices": devices,
        "telemetry": series,
        "codesets": {
            "icd10": [{"code": c, "display": d} for c, d in CONDITIONS],
            "cpt": [{"code": c, "display": d} for c, d in CPT],
            "denial_codes": [{"code": c, "display": d} for c, d in DENIALS],
        },
    }


def main() -> None:
    rng = random.Random(SEED)
    data = build(rng)
    for name, payload in data.items():
        path = HERE / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        count = len(payload) if isinstance(payload, list) else len(payload.keys())
        print(f"wrote {path.name} ({count} entries)")


if __name__ == "__main__":
    main()
