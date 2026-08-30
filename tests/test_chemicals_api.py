from datetime import datetime, timezone
from fastapi.testclient import TestClient
import api
import galit

NOW = "2026-08-23T12:00:00+00:00"


def setup(tmp_path, monkeypatch):
    repo = galit.ChemicalRepository(tmp_path / "chemicals.json")
    monkeypatch.setattr(api, "CHEMICALS", repo)
    return TestClient(api.app), repo


def product():
    return {"id":"p1", "name":"P", "manufacturer":"M", "hazards":["halite"],
            "density_kg_l":1.2, "price_per_kg":2.5, "currency":"BYN"}


def test_catalog_evidence_recommendation_and_validation(tmp_path, monkeypatch):
    client, _ = setup(tmp_path, monkeypatch)
    assert client.put("/api/v1/chemicals/products/p1", json=product()).status_code == 200
    assert client.get("/api/v1/chemicals/products").json()[0]["id"] == "p1"
    bad = {"id":"e1", "product_id":"p1", "hazard":"halite", "points":[{"dose":25,"unit":"g/m3","effective":True}], "validated":True, "conditions":"lab"}
    assert client.put("/api/v1/chemicals/envelopes/e1", json=bad).status_code == 400
    good = bad | {"validation_reference":"report-1"}
    assert client.put("/api/v1/chemicals/envelopes/e1", json=good).status_code == 200
    rec = client.post("/api/v1/chemicals/recommendations", json={"hazards":["halite"],"treated_fluid_m3_day":100,"oil_m3_day":0})
    assert rec.status_code == 200
    assert rec.json()[0]["dose_kg_m3"] == "0.025" and rec.json()[0]["cost_per_m3_oil"] is None
    stale = good | {"conditions":"field", "revision":1, "expected_revision":0}
    assert client.put("/api/v1/chemicals/envelopes/e1", json=stale).status_code == 409


def test_inventory_reservations_forecast_shortage_and_errors(tmp_path, monkeypatch):
    client, repo = setup(tmp_path, monkeypatch)
    client.put("/api/v1/chemicals/products/p1", json=product())
    lot = {"id":"l1","product_id":"p1","received_at":NOW,"expires_on":"2027-01-01",
           "quantity":10,"unit":"l","idempotency_key":"receipt-1"}
    assert client.post("/api/v1/chemicals/lots", json=lot).status_code == 201
    assert client.post("/api/v1/chemicals/lots", json=lot).status_code == 201
    assert client.get("/api/v1/chemicals/stock/p1", params={"as_of":"2026-08-23"}).json()["available_kg"] == "12.00"
    consumed = {"product_id":"p1","quantity":3,"unit":"kg","occurred_at":NOW,"idempotency_key":"use-1","reference":"job"}
    assert client.post("/api/v1/chemicals/consume", json=consumed).status_code == 201
    reservation = {"product_id":"p1","quantity":2,"unit":"kg","required_on":"2026-09-01","idempotency_key":"r1"}
    created = client.post("/api/v1/chemicals/reservations", json=reservation)
    assert created.status_code == 201 and created.json()["allocations"][0]["lot_id"] == "l1"
    rid = created.json()["id"]
    assert client.post(f"/api/v1/chemicals/reservations/{rid}/release", json={"revision":1}).status_code == 200
    forecast = client.post("/api/v1/chemicals/forecasts", json={"product_id":"p1","as_of":"2026-08-23","horizon_days":10})
    assert forecast.status_code == 200 and forecast.json()["status"] == "available"
    shortage = client.post("/api/v1/chemicals/shortages", json={"product_id":"p1","as_of":"2026-08-23","lead_time_days":20,"safety_stock_days":2})
    assert shortage.status_code == 200 and shortage.json()["status"] == "available"
    assert client.post("/api/v1/chemicals/lots", json=lot | {"id":"missing","product_id":"none","idempotency_key":"x"}).status_code == 404
    assert client.post("/api/v1/chemicals/consume", json=consumed | {"quantity":999,"idempotency_key":"too-much"}).status_code == 409
    assert len(client.get("/api/v1/chemicals/transactions").json()) == len(repo.list_transactions())


def test_append_only_transaction_and_http_422(tmp_path, monkeypatch):
    client, _ = setup(tmp_path, monkeypatch)
    client.put("/api/v1/chemicals/products/p1", json=product())
    client.post("/api/v1/chemicals/lots", json={"id":"l1","product_id":"p1","received_at":NOW,"expires_on":"2027-01-01","quantity":10,"idempotency_key":"r"})
    tx = {"id":"t1","idempotency_key":"a1","product_id":"p1","lot_id":"l1","kind":"adjustment","quantity":2,"occurred_at":NOW,"reference":"count"}
    assert client.post("/api/v1/chemicals/transactions", json=tx).status_code == 201
    assert client.post("/api/v1/chemicals/transactions", json=tx).status_code == 201
    assert client.post("/api/v1/chemicals/transactions", json=tx | {"kind":"receipt"}).status_code == 422
    assert client.post("/api/v1/chemicals/consume", json={"product_id":"p1","quantity":1,"occurred_at":"2026-08-23T12:00:00","idempotency_key":"x","reference":"x"}).status_code == 422


def test_existing_health_contract_regression(tmp_path, monkeypatch):
    client, _ = setup(tmp_path, monkeypatch)
    response = client.get("/api/v1/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"
