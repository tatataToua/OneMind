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

N_PATIENTS = 18
# Patient 0 (id 12345, "Samuel Ferreira") is pinned by identity below - several
# tests and the codebase's own example prompts treat that record as real, so
# its demographics and its N18.3 condition must survive any reshuffle here.
TWIN_INDEX = 9  # shares patient 0's name; see _pin_name_collision.

FIRST = ["Dana", "Marcus", "Priya", "Elena", "Samuel", "Nia", "Tobias", "Rosa"]
LAST = ["Whitfield", "Okafor", "Raman", "Vasquez", "Lindqvist", "Boateng", "Ferreira", "Kaur"]

CONDITIONS = [
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I10", "Essential (primary) hypertension"),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
    ("N18.3", "Chronic kidney disease, stage 3"),
    ("I48.91", "Unspecified atrial fibrillation"),
    ("E78.5", "Hyperlipidemia, unspecified"),
    ("E66.9", "Obesity, unspecified"),
    ("J45.909", "Unspecified asthma, uncomplicated"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
    ("E03.9", "Hypothyroidism, unspecified"),
    ("I50.32", "Chronic diastolic (congestive) heart failure"),
    ("M17.9", "Osteoarthritis of knee, unspecified"),
]

MEDS = [
    ("Metformin", "500 mg", "twice daily"),
    ("Insulin glargine", "18 units", "at bedtime"),
    ("Empagliflozin", "10 mg", "once daily"),
    ("Lisinopril", "10 mg", "once daily"),
    ("Losartan", "50 mg", "once daily"),
    ("Metoprolol succinate", "50 mg", "once daily"),
    ("Tiotropium", "18 mcg", "once daily"),
    ("Albuterol", "90 mcg, 2 puffs", "every 4-6 hours as needed"),
    ("Fluticasone/salmeterol", "250/50 mcg", "twice daily"),
    ("Apixaban", "5 mg", "twice daily"),
    ("Atorvastatin", "20 mg", "at bedtime"),
    ("Omeprazole", "20 mg", "once daily"),
    ("Levothyroxine", "75 mcg", "once daily"),
    ("Furosemide", "20 mg", "once daily"),
    ("Acetaminophen", "500 mg", "every 6 hours as needed"),
]

# Which medications a condition plausibly puts someone on. A condition absent
# here (E66.9 - obesity) is realistic too: not every diagnosis is pharmacologic.
CONDITION_MEDS = {
    "E11.9": ["Metformin", "Insulin glargine", "Empagliflozin"],
    "I10": ["Lisinopril", "Losartan", "Metoprolol succinate"],
    "J44.9": ["Tiotropium", "Albuterol", "Fluticasone/salmeterol"],
    "N18.3": ["Lisinopril"],
    "I48.91": ["Apixaban", "Metoprolol succinate"],
    "E78.5": ["Atorvastatin"],
    "J45.909": ["Albuterol", "Fluticasone/salmeterol"],
    "K21.9": ["Omeprazole"],
    "E03.9": ["Levothyroxine"],
    "I50.32": ["Furosemide", "Metoprolol succinate", "Lisinopril"],
    "M17.9": ["Acetaminophen"],
}

# (name, unit, low, high, threshold, direction). direction marks which side of
# the threshold is abnormal - eGFR is a floor breach (low = worse kidney
# function), the rest are ceiling breaches. A single "reference_upper" field
# used to be applied to all of them, which silently mis-flagged eGFR.
LABS = [
    ("Hemoglobin A1c", "%", 5.4, 11.2, 5.7, "above"),
    ("eGFR", "mL/min/1.73m2", 22.0, 98.0, 60.0, "below"),
    ("LDL cholesterol", "mg/dL", 60.0, 190.0, 100.0, "above"),
    ("Potassium", "mmol/L", 3.1, 5.9, 5.1, "above"),
    ("Creatinine", "mg/dL", 0.6, 3.5, 1.3, "above"),
    ("TSH", "mIU/L", 0.1, 15.0, 4.5, "above"),
    ("NT-proBNP", "pg/mL", 50.0, 3000.0, 450.0, "above"),
]

# Which labs a condition is actually monitored with. Conditions absent here
# (hypertension, COPD, AFib, obesity, asthma, GERD, osteoarthritis) don't drive
# a routine lab in this model - real primary care doesn't order one for every
# diagnosis either.
CONDITION_LABS = {
    "E11.9": ["Hemoglobin A1c"],
    "N18.3": ["eGFR", "Creatinine", "Potassium"],
    "E78.5": ["LDL cholesterol"],
    "E03.9": ["TSH"],
    "I50.32": ["NT-proBNP"],
}
# Fallback for patients whose conditions map to none of the above - routine
# screening labs a primary care visit orders regardless of diagnosis.
GENERAL_LABS = ["Hemoglobin A1c", "LDL cholesterol"]

# Denial codes, each classified by what would actually resolve it.
#
# `coding_related` is the field that matters: it separates denials a coder can
# fix by re-billing from denials no amount of recoding will touch. Asked why a
# claim was denied and whether the diagnosis matches, a system that cannot tell
# those apart will happily suggest a recode for a missing prior authorisation.
#
# The categories follow what the CARC text actually says. A maintained
# CARC/RARC set would carry effective dates and revision history; this is a
# ten-row stand-in, and `docs/decisions.md` says so.
DENIALS = [
    ("CO-197", "Precertification/authorization absent", "authorization", False),
    ("CO-16", "Claim lacks information needed for adjudication", "submission", False),
    ("CO-11", "Diagnosis inconsistent with the procedure", "coding", True),
    ("CO-29", "Time limit for filing has expired", "timely_filing", False),
    ("PR-204", "Service not covered under the patient's current benefit plan", "benefit", False),
    ("CO-50", "Non-covered services because not deemed a medical necessity", "coding", True),
    ("CO-45", "Charge exceeds the fee schedule/maximum allowable amount", "pricing", False),
    (
        "CO-97",
        "Benefit for this service is included in another already-adjudicated service",
        "bundling",
        True,
    ),
    ("PR-1", "Deductible amount", "patient_responsibility", False),
    ("PR-2", "Coinsurance amount", "patient_responsibility", False),
]

CPT = [
    ("99213", "Office visit, established patient, low complexity"),
    ("99214", "Office visit, established patient, moderate complexity"),
    ("99215", "Office visit, established patient, high complexity"),
    ("95250", "Continuous glucose monitoring, sensor placement and training"),
    ("83036", "Hemoglobin A1c"),
    ("E0784", "External ambulatory insulin infusion pump"),
    ("93000", "Electrocardiogram, routine ECG with interpretation"),
    ("94060", "Bronchodilator responsiveness, spirometry"),
    ("71046", "Radiologic exam, chest, 2 views"),
    ("E0470", "Respiratory assist device, bilevel, without backup rate"),
    ("80053", "Comprehensive metabolic panel"),
    ("85025", "Complete blood count, automated"),
    ("99457", "Remote physiologic monitoring treatment management, first 20 minutes"),
    ("99490", "Chronic care management, first 20 minutes"),
    ("G0438", "Annual wellness visit, initial"),
]

# Which billing codes a condition plausibly generates a claim for.
CONDITION_CPT = {
    "E11.9": ["95250", "83036", "E0784"],
    "I10": ["99213", "99214", "99215"],
    "J44.9": ["94060", "71046", "E0470"],
    "N18.3": ["80053"],
    "I48.91": ["93000"],
    "E78.5": ["80053"],
}
# Always in the running: routine E/M and care-management codes any patient
# might be billed for regardless of diagnosis.
GENERAL_CPT = ["99213", "99214", "99215", "99490", "G0438", "85025"]

PAYERS = ["Meridian Health", "Cascade Mutual", "Northwind Care", "Harborview Preferred", "Bluecrest Assurance"]

# (metric, unit, low, high, alert_threshold, alert_direction). Direction
# matters clinically: hypertension is a ceiling breach, hypoxaemia is a floor
# breach. A single "upper threshold" field would silently invert SpO2 alerts.
METRICS = [
    ("blood_pressure_systolic", "mmHg", 118, 172, 140, "above"),
    ("spo2", "%", 86, 99, 92, "below"),
    ("heart_rate", "bpm", 52, 118, 100, "above"),
    ("blood_glucose", "mg/dL", 70, 310, 180, "above"),
]

# Which condition puts a patient on which kind of remote monitoring. Only a
# subset of conditions has a matching device in this model - not every chronic
# diagnosis is remotely monitored in practice.
CONDITION_METRIC = {
    "E11.9": "blood_glucose",
    "I10": "blood_pressure_systolic",
    "J44.9": "spo2",
    "I48.91": "heart_rate",
}


def _pin_patient_zero_identity(patients: list[dict]) -> None:
    """Lock patient 0's identity to the record the rest of the codebase treats
    as real.

    `fhir.py`'s tool description, `examples.py`'s sample prompts, and several
    tests refer to patient 12345 / Samuel Ferreira / MRN-672113 as a concrete
    record. Regenerating the corpus must not quietly turn those into lies.
    Applied after the draw, like the name-collision pin below, so identity
    fields don't perturb anyone else's RNG draws. Patient 0's *condition* is
    pinned earlier, in `build()`, because it has to be in place before that
    patient's labs and medications are derived from it.
    """
    anchor = patients[0]
    assert anchor["patient_id"] == "12345"
    anchor["mrn"] = "MRN-672113"
    anchor["ssn"] = "541-63-1736"
    anchor["name"] = "Samuel Ferreira"
    anchor["birth_date"] = "1979-10-22"
    anchor["gender"] = "male"
    anchor["phone"] = "(555) 493-1882"


def _pin_name_collision(patients: list[dict]) -> None:
    """Force two patients to share a name, on purpose.

    Drawing names from an 8x8 space leaves collisions to luck, which means the
    corpus was clean by chance rather than by design. Luck is the wrong basis
    for this: real populations contain people who share a name, duplicate-
    record rates inside a single health system are commonly cited near 10%,
    and the US has no national patient identifier to fall back on. Identity
    ambiguity is precisely the case a clinical agent must refuse to resolve by
    guessing, so the corpus should always contain one.

    Applied after the draw so it cannot perturb the RNG sequence - every other
    field, MRN, SSN, claim id and device id stays byte-identical.
    """
    original, twin = patients[0], patients[TWIN_INDEX]
    twin["name"] = original["name"]

    # The twins must stay separable by a second identifier, or the refusal
    # becomes a dead end rather than a prompt for more information.
    assert original["mrn"] != twin["mrn"], "name twins need distinct MRNs"
    assert original["birth_date"] != twin["birth_date"], "name twins need distinct DOBs"


def _append_coding_mismatch(
    claims: list[dict],
    patients: list[dict],
    claim_i: int,
    cpt_lookup: dict[str, str],
    today: date,
) -> None:
    """Force one claim whose billed diagnosis is not on the patient's chart.

    Every generated claim draws `icd10_code` from the patient's own condition
    list, so the billed diagnosis always matches the problem list by
    construction. That makes the reconciler's `mismatch` verdict unreachable
    outside unit tests - a reviewer asking "show me one that does *not* match"
    would have nothing to open.

    Two claims are added, as a matched pair for the same patient, so the
    interesting comparison is the diagnosis rather than the patient:

      - a mismatch, denied CO-11 (coding), where re-billing is the real remedy
      - a match, denied CO-29 (timely filing), where recoding would change
        nothing at all

    Appended after the loop with literal values and no `rng` draws, following
    the pins above, so every previously generated claim id, amount and payer
    stays byte-identical.
    """
    patient = patients[0]
    on_chart = {c["code"] for c in patient["conditions"]}
    off_chart = next(code for code, _ in CONDITIONS if code not in on_chart)
    matching = sorted(on_chart)[0]

    for offset, (icd10, denial_code) in enumerate(((off_chart, "CO-11"), (matching, "CO-29"))):
        reason, category, _ = next(
            (text, cat, flag) for code, text, cat, flag in DENIALS if code == denial_code
        )
        assert category, f"{denial_code} must be classified"
        claims.append(
            {
                "claim_id": f"CLM-{8840 + (claim_i + offset) * 3}",
                "patient_id": patient["patient_id"],
                "date_of_service": (today - timedelta(days=45 + offset)).isoformat(),
                "cpt_code": "99214",
                "cpt_description": cpt_lookup["99214"],
                "icd10_code": icd10,
                "billed_amount": 412.5 + offset,
                "allowed_amount": 0.0,
                "status": "denied",
                "denial_code": denial_code,
                "denial_reason": reason,
                "payer": PAYERS[0],
            }
        )


def _labs_for(rng: random.Random, cond_codes: list[str], today: date) -> list[dict]:
    lookup = {name: (unit, lo, hi, ref, direction) for name, unit, lo, hi, ref, direction in LABS}
    names = sorted({name for code in cond_codes for name in CONDITION_LABS.get(code, [])})
    if not names:
        names = GENERAL_LABS

    labs = []
    for name in names:
        unit, lo, hi, ref, direction = lookup[name]
        for back in (240, 120, 20):
            labs.append(
                {
                    "name": name,
                    "value": round(rng.uniform(lo, hi), 1),
                    "unit": unit,
                    "reference_threshold": ref,
                    "reference_direction": direction,
                    "effective_date": (today - timedelta(days=back)).isoformat(),
                }
            )
    return labs


def _meds_for(rng: random.Random, cond_codes: list[str]) -> list[dict]:
    lookup = {name: (name, dose, freq) for name, dose, freq in MEDS}
    names = sorted({name for code in cond_codes for name in CONDITION_MEDS.get(code, [])})
    if not names:
        return []
    chosen = rng.sample(names, k=rng.randint(1, min(4, len(names))))
    return [{"name": n, "dose": d, "frequency": f} for n, d, f in (lookup[name] for name in chosen)]


def build(rng: random.Random) -> dict[str, object]:
    today = date(2026, 8, 20)
    patients = []
    for i in range(N_PATIENTS):
        pid = f"{12345 + i * 111}"
        dob = date(1948 + rng.randint(0, 45), rng.randint(1, 12), rng.randint(1, 28))
        conds = rng.sample(CONDITIONS, k=rng.randint(1, 3))
        if i == 0 and not any(code == "N18.3" for code, _ in conds):
            # See `test_answer_grounding.py::test_a_supported_answer_raises_nothing` -
            # this record's CKD diagnosis is asserted as real evidence.
            conds.append(("N18.3", "Chronic kidney disease, stage 3"))
        cond_codes = [c for c, _ in conds]
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
                "medications": _meds_for(rng, cond_codes),
                "labs": _labs_for(rng, cond_codes, today),
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

    _pin_patient_zero_identity(patients)
    _pin_name_collision(patients)

    cpt_lookup = dict(CPT)
    claims = []
    claim_i = 0
    for patient in patients:
        cond_codes = [c["code"] for c in patient["conditions"]]
        pool = sorted({code for cc in cond_codes for code in CONDITION_CPT.get(cc, [])}) + GENERAL_CPT
        for _ in range(rng.randint(1, 4)):
            cpt_code = rng.choice(pool)
            denied = rng.random() < 0.4
            denial = rng.choice(DENIALS) if denied else None
            claims.append(
                {
                    "claim_id": f"CLM-{8840 + claim_i * 3}",
                    "patient_id": patient["patient_id"],
                    "date_of_service": (today - timedelta(days=rng.randint(10, 200))).isoformat(),
                    "cpt_code": cpt_code,
                    "cpt_description": cpt_lookup[cpt_code],
                    "icd10_code": rng.choice(cond_codes),
                    "billed_amount": round(rng.uniform(120, 4200), 2),
                    "allowed_amount": 0.0 if denied else round(rng.uniform(80, 3000), 2),
                    "status": "denied" if denied else "paid",
                    "denial_code": denial[0] if denial else None,
                    "denial_reason": denial[1] if denial else None,
                    "payer": rng.choice(PAYERS),
                }
            )
            claim_i += 1

    _append_coding_mismatch(claims, patients, claim_i, cpt_lookup, today)

    metrics_by_name = {name: (unit, lo, hi, threshold, direction) for name, unit, lo, hi, threshold, direction in METRICS}
    # One patient per metric is forced to carry the demo drift/breach case, so
    # the corpus always has a floor-breach (SpO2) and ceiling-breach example to
    # show regardless of who else qualifies by condition.
    forced: dict[str, str] = {}
    for patient in patients:
        for code in (c["code"] for c in patient["conditions"]):
            metric = CONDITION_METRIC.get(code)
            if metric and metric not in forced:
                forced[metric] = patient["patient_id"]

    devices = []
    series = []
    for patient in patients:
        cond_codes = [c["code"] for c in patient["conditions"]]
        candidate_metrics = sorted({CONDITION_METRIC[c] for c in cond_codes if c in CONDITION_METRIC})
        for metric in candidate_metrics:
            is_forced = forced.get(metric) == patient["patient_id"]
            # Not every eligible patient is remotely monitored in real life;
            # the forced demo case always gets a device, everyone else is a
            # coin flip.
            if not is_forced and rng.random() > 0.5:
                continue

            unit, lo, hi, threshold, direction = metrics_by_name[metric]
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
            drift = is_forced or rng.random() < 0.3
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
            # `category` and `coding_related` ride along with the code set
            # rather than living in a second file: this is the same kind of
            # versioned reference data, and `validate_code` already searches
            # here. The reconciler reads the classification; `validate_code`
            # ignores the extra keys.
            "denial_codes": [
                {"code": c, "display": d, "category": cat, "coding_related": flag}
                for c, d, cat, flag in DENIALS
            ],
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
