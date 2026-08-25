"""Tool layer tests - the deterministic half of the system."""

from __future__ import annotations

from typing import Any

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


def _uniquely_named() -> dict[str, Any]:
    names = [p["name"] for p in store.patients()]
    return next(p for p in store.patients() if names.count(p["name"]) == 1)


def _name_twins() -> list[dict[str, Any]]:
    names = [p["name"] for p in store.patients()]
    shared = next(n for n in names if names.count(n) > 1)
    return [p for p in store.patients() if p["name"] == shared]


def test_corpus_contains_a_name_collision() -> None:
    """The generator pins this deliberately. If a future regeneration produces
    six unique names, the ambiguity path stops being exercised by real data and
    this suite would start passing for the wrong reason."""
    twins = _name_twins()
    assert len(twins) == 2
    assert twins[0]["mrn"] != twins[1]["mrn"]
    assert twins[0]["birth_date"] != twins[1]["birth_date"]


def test_patient_lookup_by_id_mrn_and_name() -> None:
    """Clinicians ask for people by name, so name resolves like any other key -
    as long as the name is unambiguous."""
    patient = _uniquely_named()
    by_id = fhir_search_patient(patient_id=patient["patient_id"])
    by_mrn = fhir_search_patient(mrn=patient["mrn"])
    by_name = fhir_search_patient(name=patient["name"])
    assert by_id["found"] and by_mrn["found"] and by_name["found"]
    assert by_id["patient_id"] == by_mrn["patient_id"] == by_name["patient_id"]


def test_name_lookup_is_case_insensitive() -> None:
    patient = _uniquely_named()
    assert fhir_search_patient(name=patient["name"].upper())["found"]
    assert fhir_search_patient(name=patient["name"].lower())["found"]


def test_shared_name_in_the_real_corpus_refuses() -> None:
    twins = _name_twins()
    result = fhir_search_patient(name=twins[0]["name"])
    assert result["found"] is False and result["ambiguous"] is True
    assert result["match_count"] == 2
    assert "medications" not in result


def test_birth_date_separates_name_twins() -> None:
    """The second identifier is what turns a refusal into an answer - the shape
    FHIR's Patient/$match takes, resolving identity from demographics."""
    for twin in _name_twins():
        result = fhir_search_patient(name=twin["name"], birth_date=twin["birth_date"])
        assert result["found"] is True
        assert result["patient_id"] == twin["patient_id"]
        assert result["mrn"] == twin["mrn"]


def test_birth_date_does_not_rescue_a_wrong_one() -> None:
    twins = _name_twins()
    result = fhir_search_patient(name=twins[0]["name"], birth_date="1900-01-01")
    assert result["found"] is False and result["ambiguous"] is True


def test_unique_key_ignores_a_mismatched_birth_date() -> None:
    """Narrowing applies to ambiguity only. An MRN that already identifies one
    person should not be second-guessed by a stray demographic."""
    patient = _uniquely_named()
    result = fhir_search_patient(mrn=patient["mrn"], birth_date="1900-01-01")
    assert result["found"] is True
    assert result["patient_id"] == patient["patient_id"]


def test_resource_fetch_separates_name_twins_by_birth_date() -> None:
    twin = _name_twins()[1]
    result = fhir_get_resource("medications", name=twin["name"], birth_date=twin["birth_date"])
    assert result["found"] is True
    assert result["patient_id"] == twin["patient_id"]


def test_shared_name_refuses_instead_of_picking_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two people can share a name. Answering for whichever record sorted first
    is a confident answer about the wrong patient - the worst failure here."""
    a, b = store.patients()[0], store.patients()[1]
    twins = [{**a}, {**b, "name": a["name"]}]
    monkeypatch.setattr(store, "patients", lambda: twins)

    result = fhir_search_patient(name=a["name"])
    assert result["found"] is False
    assert result["ambiguous"] is True
    assert result["match_count"] == 2
    assert "medications" not in result

    # The unique keys still resolve, and they resolve to different people.
    assert fhir_search_patient(mrn=a["mrn"])["patient_id"] == a["patient_id"]
    assert fhir_search_patient(mrn=b["mrn"])["patient_id"] == b["patient_id"]


def test_ambiguous_name_does_not_name_the_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refusing must not become its own disclosure: 'did you mean MRN-x or
    MRN-y?' tells the caller two people they never asked about exist."""
    a, b = store.patients()[0], store.patients()[1]
    twins = [{**a}, {**b, "name": a["name"]}]
    monkeypatch.setattr(store, "patients", lambda: twins)

    blob = str(fhir_search_patient(name=a["name"]))
    for patient in twins:
        assert patient["mrn"] not in blob
        assert patient["patient_id"] not in blob


def test_resource_fetch_refuses_a_shared_name(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = store.patients()[0], store.patients()[1]
    twins = [{**a}, {**b, "name": a["name"]}]
    monkeypatch.setattr(store, "patients", lambda: twins)

    result = fhir_get_resource("medications", name=a["name"])
    assert result["found"] is False and result["ambiguous"] is True
    assert "items" not in result


def test_unknown_patient_does_not_enumerate_the_store() -> None:
    """A miss must not confirm or deny what else we hold. The old behaviour
    returned every patient id, and the model recited them into the answer."""
    result = fhir_search_patient(patient_id="99999")
    assert result["found"] is False
    assert "known_patient_ids" not in result
    known = {p["patient_id"] for p in store.patients()} | {p["name"] for p in store.patients()}
    assert not any(value in str(result) for value in known)


def test_labs_are_newest_first_and_flag_abnormal() -> None:
    patient = store.patients()[0]
    result = fhir_get_resource("labs", patient_id=patient["patient_id"])
    dates = [row["effective_date"] for row in result["items"]]
    assert dates == sorted(dates, reverse=True)
    for row in result["items"]:
        expected = (
            row["value"] < row["reference_threshold"]
            if row["reference_direction"] == "below"
            else row["value"] > row["reference_threshold"]
        )
        assert row["abnormal"] == expected


# -- revenue cycle ----------------------------------------------------------


def test_claim_lookup_by_id() -> None:
    claim = store.claims()[0]
    result = claim_lookup(claim_id=claim["claim_id"])
    assert result["found"]
    assert result["claims"][0]["claim_id"] == claim["claim_id"]


def test_claim_lookup_requires_an_argument() -> None:
    assert claim_lookup()["found"] is False


def test_unknown_claim_does_not_sample_the_ledger() -> None:
    result = claim_lookup(claim_id="CLM-000000")
    assert result["found"] is False
    assert "known_claim_ids" not in result
    assert not any(c["claim_id"] in str(result) for c in store.claims())


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
    the model caught this before the tests did.

    Evaluated per patient, because an unscoped call now returns configuration
    rather than everyone's readings. That is the point of the change, and the
    direction logic is just as checkable one chart at a time.
    """
    by_metric: dict[str, dict[str, Any]] = {}
    for patient_id in sorted({d["patient_id"] for d in store.devices()}):
        result = evaluate_thresholds(patient_id=patient_id)
        for row in result.get("evaluations", []):
            by_metric.setdefault(row["metric"], row)

    assert "spo2" in by_metric, "fixtures must monitor SpO2 somewhere"
    spo2 = by_metric["spo2"]
    assert spo2["alert_direction"] == "below"
    for reading in spo2["breaching_readings"]:
        assert reading["value"] < spo2["threshold"]

    for metric, row in by_metric.items():
        if row["alert_direction"] != "above":
            continue
        for reading in row["breaching_readings"]:
            assert reading["value"] > row["threshold"], metric


def test_an_unscoped_question_gets_configuration_not_a_patient() -> None:
    """Asked "what is the alert threshold for SpO2" - a question about a setting
    with no patient in it - the answer used to name the first device that
    matched, its owner, and their latest reading."""
    result = telemetry_series(metric="spo2")
    assert result["found"] is True
    assert result["scope"] == "configuration"

    row = next(r for r in result["thresholds"] if r["metric"] == "spo2")
    assert row["alert_threshold"] == 92
    assert row["alert_direction"] == "below"
    assert row["device_count"] >= 1

    blob = str(result)
    assert "DEV-" not in blob, "a device id identifies a patient by proxy"
    assert not any(d["patient_id"] in blob for d in store.devices())
    assert "readings" not in result


def test_breaches_are_not_evaluated_without_a_patient() -> None:
    """A breach is a fact about a person. The threshold still comes back."""
    result = evaluate_thresholds(metric="spo2")
    assert result["scope"] == "configuration"
    assert "evaluations" not in result


def test_series_reports_trend_and_window() -> None:
    device = store.devices()[0]
    result = telemetry_series(patient_id=device["patient_id"], days=7)
    entry = result["series"][0]
    assert entry["reading_count"] <= 7
    assert entry["trend"] in {"rising", "falling", "flat"}
    assert entry["min"] <= entry["mean"] <= entry["max"]


def test_an_unscoped_miss_lists_metrics_but_not_patients() -> None:
    """Naming the metrics helps the model retry. Naming who is monitored is a
    roster, and the model will repeat it into the answer."""
    result = telemetry_series(metric="glucose")
    assert result["found"] is False
    assert result["available_metrics"]
    monitored = {d["patient_id"] for d in store.devices()}
    assert not any(pid in str(result) for pid in monitored)


def test_a_patient_with_no_devices_is_told_so_plainly() -> None:
    """The metric hint follows the question.

    Observed live: asked whether patient 12456 had breached an SpO2 threshold,
    the answer said no device matched *and* that "the available metrics for
    this patient" included spo2 - because the list was every metric in the
    store, sitting beside a detail that said "that patient". 12456 has no
    devices at all.
    """
    without = next(
        p["patient_id"]
        for p in store.patients()
        if not any(d["patient_id"] == p["patient_id"] for d in store.devices())
    )
    result = telemetry_series(patient_id=without, metric="spo2")
    assert result["found"] is False
    assert result["available_metrics"] == []
    assert "no monitored devices" in result["detail"]


def test_a_patient_hint_lists_only_their_own_metrics() -> None:
    """And a miss must not report what the rest of the population is monitored for."""
    device = store.devices()[0]
    theirs = {d["metric"] for d in store.devices() if d["patient_id"] == device["patient_id"]}
    everything = {d["metric"] for d in store.devices()}
    assert everything - theirs, "fixture needs a metric this patient does not have"

    result = telemetry_series(patient_id=device["patient_id"], metric="glucose")
    assert result["found"] is False
    assert set(result["available_metrics"]) == theirs
