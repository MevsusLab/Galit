from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

import api
import galit

NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


def record(**changes):
    values = dict(
        well_id="w-1", well_name="Well 1", field_name="Field A", cluster="C1", site="S1",
        event_at=NOW, complication_type="wax", description="Wash", reagent_name="R1",
        reagent_id=None, dosage=2, dosage_unit="l/m3", cost=120, currency="BYN",
        treatment_type="wash", rate_before_m3_day=10, source="test",
        status=galit.TreatmentStatus.COMPLETED,
    )
    values.update(changes)
    return galit.new_treatment(now=NOW, **values)


def test_effect_math_thresholds_zero_and_nullable():
    item = record(rate_after_m3_day=12, effect_duration_days=20)
    effect = galit.treatment_effect(item)
    assert effect["rate_change_m3_day"] == 2
    assert effect["rate_change_percent"] == 20
    assert effect["incremental_production_m3"] == 40
    assert effect["cost_per_incremental_m3"] == 3
    assert effect["classification"] == "effective"

    assert galit.treatment_effect(record(rate_before_m3_day=0, rate_after_m3_day=1))[
        "rate_change_percent"
    ] is None
    assert galit.treatment_effect(record())["classification"] == "insufficient_data"
    with pytest.raises(ValueError):
        record(rate_after_m3_day=-1)


def test_analytics_robust_interval_minimum_sample_and_excessive():
    rows = []
    for index, days in enumerate((20, 21, 22, 23, 200)):
        rows.append(record(
            well_id=f"w-{index}", event_at=NOW + timedelta(days=index),
            rate_after_m3_day=12, effect_duration_days=days,
            recurrence=True, recurrence_date=NOW + timedelta(days=index + days),
        ))
    first = record(well_id="repeat", rate_after_m3_day=10, effect_duration_days=2)
    second = record(well_id="repeat", event_at=NOW + timedelta(days=5))
    result = galit.treatment_analytics([*rows, first, second], min_sample_size=5)
    assert result["comparisons"][0]["n"] >= 5
    assert result["interval_recommendations"][0]["median_interval_days"] == 21.5
    assert result["potentially_excessive"]
    assert "причин" in result["warning"]


def test_csv_roundtrip_and_utf8_validation():
    source = record(rate_after_m3_day=12)
    loaded = galit.treatments_from_csv(galit.treatments_to_csv([source]))
    assert loaded[0].well_name == source.well_name
    assert loaded[0].rate_after_m3_day == 12
    with pytest.raises(ValueError, match="UTF-8"):
        galit.treatments_from_csv(b"\xff")


def test_api_effectiveness_nullable_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "TREATMENTS", galit.TreatmentRepository(tmp_path / "treatments.json"))
    client = TestClient(api.app)
    payload = record().to_dict()
    for key in ("id", "status", "result_metrics", "success", "effect_duration_days", "recurrence",
                "recurrence_date", "actual_result", "revision", "archived", "archived_at",
                "created_at", "updated_at"):
        payload.pop(key, None)
    response = client.post("/api/v1/treatments", json=payload)
    assert response.status_code == 201
    assert response.json()["rate_after_m3_day"] is None
    analytics = client.get("/api/v1/treatments/analytics/effectiveness")
    assert analytics.status_code == 200
    assert analytics.json()["effects"][0]["classification"] == "insufficient_data"
    assert client.get("/api/v1/treatments/missing").status_code == 404
    payload["rate_before_m3_day"] = -1
    assert client.post("/api/v1/treatments", json=payload).status_code == 422
