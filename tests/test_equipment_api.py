from datetime import datetime, timezone
from fastapi.testclient import TestClient
import api
import galit


def test_equipment_api_happy_errors_and_portfolio(tmp_path, monkeypatch):
    repo=galit.EquipmentRepository(tmp_path/"equipment.json")
    monkeypatch.setattr(api, "EQUIPMENT", repo)
    client=TestClient(api.app)
    metadata={"well":"API-W","lift_type":"ESP","runtime_days":10,"nominal_current_a":100,
              "temperature_limit_c":100,"vibration_limit_mm_s":10}
    assert client.post("/api/v1/equipment", json=metadata).status_code == 201
    snapshot={"id":"api-1","well":"API-W","lift_type":"ESP","timestamp":"2026-08-23T12:00:00+00:00",
              "current_a":150,"nominal_current_a":100,"intake_pressure":5,"discharge_pressure":10,
              "baseline_intake_pressure":5,"baseline_discharge_pressure":10,"motor_temperature_c":95,
              "temperature_limit_c":100,"vibration_mm_s":9,"vibration_limit_mm_s":10}
    assert client.post("/api/v1/equipment/telemetry", json=snapshot).status_code == 201
    assert client.post("/api/v1/equipment/telemetry", json=snapshot).status_code == 201
    assert client.get("/api/v1/equipment/forecast/API-W").json()["baseline_failure_risk"] is not None
    assert len(client.get("/api/v1/equipment/forecast").json()) == 1
    assert client.get("/api/v1/equipment/forecast/missing").status_code == 404
    bad=snapshot | {"id":"bad", "timestamp":"2026-08-23T12:00:00"}
    assert client.post("/api/v1/equipment/telemetry", json=bad).status_code == 422
