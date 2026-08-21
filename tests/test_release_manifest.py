"""Tests for the machine-readable release evidence generator."""
from pathlib import Path

import release_manifest


def test_parse_pytest_summary():
    result = release_manifest.parse_pytest_summary("137 passed, 1 warning in 9.05s\n", 0)
    assert result["result"] == "passed"
    assert result["passed"] == 137
    assert result["failed"] == 0


def test_demo_evidence_is_reproducible_and_explicitly_synthetic():
    first = release_manifest.collect_demo()
    second = release_manifest.collect_demo()
    assert first == second
    assert first["dataset_label"] == "synthetic"
    assert first["seed"] == release_manifest.DEFAULT_SEED
    assert first["well_count"] == 40
    envelope = first["historical_economic_scenario_envelope"]
    assert envelope["low"] < envelope["midpoint"] < envelope["high"]
    assert "not forecast" in envelope["label"]
    assert "not KPI" in envelope["label"]


def test_manifest_labels_do_not_claim_field_validation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(release_manifest, "collect_git", lambda root: {
        "available": True, "commit": "abc", "dirty": False,
    })
    tests = {"command": "pytest", "result": "passed", "exit_code": 0,
             "passed": 3, "failed": 0, "errors": 0, "summary": "3 passed"}
    manifest = release_manifest.build_manifest(tmp_path, tests, "2026-01-01T00:00:00+00:00")
    assert manifest["evidence_labels"]["software_verification"] == "passed"
    assert manifest["evidence_labels"]["model_validation"] == "not validated on independent field data"
    assert manifest["demo"]["interpretation_label"] == "scenario illustration; not field data"
    assert manifest["competition_demo"]["scenario_count"] == 5
    assert manifest["pilot_unit_economics"]["currency_claim"].startswith("none until")
    assert manifest["pilot_unit_economics"]["customer_checklist"] == (
        "reports/customer-input-checklist.md"
    )
    assert set(manifest["competition_demo"]["labels"]) == {
        "synthetic", "illustrative", "not field validated",
    }


def test_reports_are_valid_json_and_markdown(tmp_path: Path):
    tests = {"command": "pytest", "result": "passed", "exit_code": 0,
             "passed": 4, "failed": 0, "errors": 0, "summary": "4 passed"}
    manifest = {
        "schema_version": "1.0", "project": "GALIT", "version": "0.1.0",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "git": {"commit": "abc", "dirty": True},
        "runtime": {"python": "3.12", "implementation": "CPython"},
        "software_verification": tests,
        "demo": release_manifest.collect_demo(),
        "evidence_labels": {},
    }
    json_path, md_path = release_manifest.write_reports(tmp_path, manifest)
    assert '"project": "GALIT"' in json_path.read_text(encoding="utf-8")
    markdown = md_path.read_text(encoding="utf-8")
    assert "software verification, not model validation" in markdown
    assert "**synthetic**" in markdown
