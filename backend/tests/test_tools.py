"""Tool layer tests - the deterministic half of the system."""

from __future__ import annotations

import pytest

from onemind.tools import store, tools
from onemind.tools.claims import claim_lookup, denial_summary, validate_code
from onemind.tools.fhir import fhir_get_resource, fhir_search_patient
from onemind.tools.policy import policy_search
from onemind.tools.telemetry import evaluate_thresholds, telemetry_series


def test_every_declared_tool_is_registered() -> None:
    """A specialist declaring a tool that does not exist should fail loudly at
    import, not at demo time."""
    from onemind.orchestrator.registry import registry

    for spec in registry.all():
        for name in spec.tool_names:
            assert name in tools, f"{spec.key} declares unknown tool {name}"


def test_specialists_do_not_share_tools() -> None:
    """The whole routing design rests on disjoint data planes. If two
    specialists share a tool, they can answer the same question and routing
    becomes a coin flip."""
    from onemind.orchestrator.registry import registry

    seen: dict[str, str] = {}
    for spec in registry.all():
        for name in spec.tool_names:
            assert name not in seen, f"{name} shared by {seen[name]} and {spec.key}"
            seen[name] = spec.key


# -- clinical ---------------------------------------------------------------


def test_patient_lookup_by_id_and_mrn() -> None:
    patient = store.patients()[0]
    by_id = fhir_search_patient(patient["patient_id"])
    by_mrn = fhir_search_patient(patient["mrn"])
    assert by_id["found"] and by_mrn["found"]
    assert by_id["patient_id"] == by_mrn["patient_id"]


def test_unknown_patient_reports_what_exists() -> None:
    """A specialist that cannot find a record should be able to say what it
    could have found, rather than inventing one."""
    result = fhir_search_patient("99999")
    assert result["found"] is False
    assert result["known_patient_ids"]


def test_labs_are_newest_first_and_flag_abnormal() -> None:
    patient = store.patients()[0]
    result = fhir_get_resource(patient["patient_id"], "labs")
    dates = [row["effective_date"] for row in result["items"]]
    assert dates == sorted(dates, reverse=True)
    for row in result["items"]:
        assert row["abnormal"] == (row["value"] > row["reference_upper"])


# -- revenue cycle ----------------------------------------------------------


def test_claim_lookup_by_id() -> None:
    claim = store.claims()[0]
    result = claim_lookup(claim_id=claim["claim_id"])
    assert result["found"]
    assert result["claims"][0]["claim_id"] == claim["claim_id"]


def test_claim_lookup_requires_an_argument() -> None:
    assert claim_lookup()["found"] is False


def test_code_validation_round_trip() -> None:
    known = store.codesets()["cpt"][0]
    assert validate_code(known["code"])["valid"] is True
    assert validate_code(known["code"].lower())["valid"] is True
    assert validate_code("NOT-A-CODE")["valid"] is False


def test_denial_summary_arithmetic() -> None:
    summary = denial_summary()
    rows = store.claims()
    denied = [c for c in rows if c["status"] == "denied"]
    assert summary["denied_claims"] == len(denied)
    assert summary["denial_rate_pct"] == pytest.approx(100 * len(denied) / len(rows), abs=0.05)


# -- compliance -------------------------------------------------------------


def test_policy_search_finds_the_deciding_section() -> None:
    """The BAA question is the one that exposed a recall problem: the decisive
    section is 'When a BAA is NOT required', and a top-3 window missed it."""
    result = policy_search("Do we need a BAA for a vendor processing only de-identified data?")
    sections = [row["section"] for row in result["results"]]
    assert any("NOT required" in section for section in sections)


def test_policy_search_returns_citations() -> None:
    result = policy_search("how long must we retain audit logs")
    assert result["results"]
    assert all(row["citation"] for row in result["results"])


def test_policy_search_handles_a_stopword_only_query() -> None:
    assert policy_search("the and of")["count"] == 0


# -- remote monitoring ------------------------------------------------------


def test_threshold_direction_is_honoured() -> None:
    """SpO2 alerts on a floor breach. Testing everything against an upper bound
    reports healthy oxygen saturation as an alert and misses real hypoxaemia -
    the model caught this before the tests did."""
    result = evaluate_thresholds()
    by_metric = {row["metric"]: row for row in result["evaluations"]}

    spo2 = by_metric["spo2"]
    assert spo2["alert_direction"] == "below"
    for reading in spo2["breaching_readings"]:
        assert reading["value"] < spo2["threshold"]

    for metric, row in by_metric.items():
        if row["alert_direction"] != "above":
            continue
        for reading in row["breaching_readings"]:
            assert reading["value"] > row["threshold"], metric


def test_series_reports_trend_and_window() -> None:
    device = store.devices()[0]
    result = telemetry_series(patient_id=device["patient_id"], days=7)
    entry = result["series"][0]
    assert entry["reading_count"] <= 7
    assert entry["trend"] in {"rising", "falling", "flat"}
    assert entry["min"] <= entry["mean"] <= entry["max"]


def test_unknown_device_lists_alternatives() -> None:
    result = telemetry_series(patient_id="99999")
    assert result["found"] is False
    assert result["available"]
