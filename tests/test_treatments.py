from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import api
from galit import (
    TreatmentConflictError,
    TreatmentRepository,
    TreatmentStatus,
    compare_reagents,
    new_treatment,
    treatment_summary,
)


NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


def values(**overrides):
    base = {
        "well_id": "w-1", "well_name": "Well 1", "event_at": NOW,
        "complication_type": "halite", "description": "Pressure increased",
        "reagent_name": "A", "reagent_id": None, "dosage": 25,
        "dosage_unit": "mg/l", "cost": 100, "currency": "BYN",
        "treatment_type": "inhibitor", "well_group": "late-stage",
    }
    return base | overrides


def assessed(reagent: str, success: bool, days: float, **overrides):
    record = new_treatment(now=NOW, **values(reagent_name=reagent, **overrides))
    record = record.transition(TreatmentStatus.IN_PROGRESS, now=NOW)
    record = record.transition(TreatmentStatus.COMPLETED, now=NOW)
    return record.transition(
        TreatmentStatus.ASSESSED, now=NOW, actual_result="Observed result",
        result_metrics={"oil_rate_gain": 2.0}, success=success,
        effect_duration_days=days, recurrence=False,
    )


def test_repository_lifecycle_revision_pagination_and_archive(tmp_path):
    repository = TreatmentRepository(tmp_path / "journal.json")
    created = repository.create(new_treatment(now=NOW, **values()))
    edited = repository.update(created.edit(description="Edited", now=NOW), expected_revision=1)
    assert edited.revision == 2
    progressed = repository.update(
        edited.transition(TreatmentStatus.IN_PROGRESS, now=NOW), expected_revision=2)
    assert progressed.status is TreatmentStatus.IN_PROGRESS and progressed.revision == 3
    with pytest.raises(TreatmentConflictError):
        repository.update(progressed.edit(comment="stale", now=NOW), expected_revision=2)
    archived = repository.archive(created.id, expected_revision=3, now=NOW)
    assert archived.archived and archived.revision == 4
    assert repository.list() == []
    assert repository.list(include_archived=True, offset=0, limit=1) == [archived]


def test_validation_recurrence_date_and_assessed_immutability():
    with pytest.raises(ValueError, match="recurrence_date requires"):
        new_treatment(now=NOW, **values(status="completed", recurrence=False,
                                        recurrence_date=NOW))
    record = assessed("A", True, 10)
    with pytest.raises(TreatmentConflictError):
        record.edit(comment="cannot edit")


def test_summary_separates_currency_and_compare_uses_explicit_cohort():
    records = [assessed("A", True, 10, currency="BYN"),
               assessed("A", False, 5, currency="USD"),
               assessed("B", True, 15), assessed("B", True, 20)]
    summary = treatment_summary(records)
    costs = summary["groups"][0]["costs_by_currency"]
    assert set(costs) == {"BYN", "USD"}
    comparison = compare_reagents(
        records, "A", "B", min_sample_size=2,
        complication_type="halite", well_group="late-stage",
    )
    assert comparison["status"] == "available"
    assert comparison["reagent_a"]["n"] == comparison["reagent_b"]["n"] == 2
    assert comparison["relative_uplift"] == 1.0
    assert "причин" in comparison["warning"].lower()


def test_api_full_path_summary_compare_conflict_and_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "TREATMENTS", TreatmentRepository(tmp_path / "api-journal.json"))
    client = TestClient(api.app)
    payload = values()
    payload["event_at"] = NOW.isoformat()
    created = client.post("/api/v1/treatments", json=payload)
    assert created.status_code == 201
    item = created.json()
    record_id = item["id"]
    for status in ("in_progress", "completed"):
        response = client.patch(
            f"/api/v1/treatments/{record_id}",
            json={"revision": item["revision"], "status": status},
        )
        assert response.status_code == 200
        item = response.json()
    assessed_response = client.patch(
        f"/api/v1/treatments/{record_id}", json={
            "revision": item["revision"], "status": "assessed",
            "actual_result": "Flow restored", "result_metrics": {"gain": 2},
            "success": True, "effect_duration_days": 30, "recurrence": False,
        },
    )
    assert assessed_response.status_code == 200
    item = assessed_response.json()
    assert item["status"] == "assessed"
    assert client.patch(
        f"/api/v1/treatments/{record_id}",
        json={"revision": item["revision"] - 1, "comment": "stale"},
    ).status_code == 409
    summary = client.get("/api/v1/treatments/analytics/summary")
    assert summary.status_code == 200
    assert summary.json()["groups"][0]["assessed_observations"] == 1
    compare = client.get("/api/v1/treatments/analytics/compare", params={
        "reagent_a": "A", "reagent_b": "B", "complication_type": "halite",
        "well_group": "late-stage", "min_sample_size": 2,
    })
    assert compare.status_code == 200 and compare.json()["status"] == "insufficient_data"
    archived = client.delete(
        f"/api/v1/treatments/{record_id}", params={"revision": item["revision"]})
    assert archived.status_code == 200 and archived.json()["archived"] is True
    assert client.get("/api/v1/treatments").json() == []
