from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

PAYLOAD = {
    "water_a": {"name": "formation", "ions_mg_l": {"Na": 20000, "Cl": 32000,
        "Ca": 4000, "Ba": 1000, "HCO3": 50, "SO4": 0}, "ph": 6.2, "t_c": 25, "p_pa": 5e6},
    "water_b": {"name": "injection", "ions_mg_l": {"Na": 1000, "Cl": 1500,
        "Ca": 100, "Ba": 0, "HCO3": 300, "SO4": 3000}, "ph": 7, "t_c": 25, "p_pa": 1e5},
    "fractions_b": [0, .5, 1],
    "profile": [{"depth_m": 0, "t_c": 25, "p_pa": 1e5},
                {"depth_m": 1000, "t_c": 35, "p_pa": 8e6}],
}


def test_endpoint_contract_and_no_implicit_dose():
    response = client.post("/api/v1/water-compatibility", json=PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == "water-compatibility-screening/1.0"
    assert len(body["ratios"]) == 3
    assert set(body["ratios"][0]["minerals"]) == {"calcite", "barite", "gypsum", "halite"}
    assert {"dangerous_ratio", "unsafe_intervals", "deposition_locations", "units", "assumptions", "warnings"} <= set(body)
    assert body["inhibitor"]["dose_mg_l"] is None
    assert body["inhibitor"]["status"] == "laboratory_test_required"


def test_endpoint_default_grid_and_validated_curve():
    payload = copy.deepcopy(PAYLOAD)
    payload.pop("fractions_b")
    first = client.post("/api/v1/water-compatibility", json=payload)
    assert first.status_code == 200 and len(first.json()["ratios"]) == 101
    mineral = max(first.json()["ratios"][50]["minerals"].values(),
                  key=lambda x: -1e99 if x["saturation_index"] is None else x["saturation_index"])["mineral"]
    payload["fractions_b"] = [.5]
    payload["dose_response"] = {"product": "Product X", "mineral": mineral, "validated": True,
        "validation_reference": "lab-report-1", "points": [
            {"dose_mg_l": 10, "maximum_supported_si": -10},
            {"dose_mg_l": 20, "maximum_supported_si": 100}]}
    body = client.post("/api/v1/water-compatibility", json=payload).json()
    assert body["inhibitor"]["dose_mg_l"] == 20


def test_endpoint_rejects_invalid_arrays_curve_and_nonfinite_values():
    payload = copy.deepcopy(PAYLOAD)
    payload["fractions_b"] = [.5, .4]
    assert client.post("/api/v1/water-compatibility", json=payload).status_code == 400
    payload = copy.deepcopy(PAYLOAD)
    payload["dose_response"] = {"product": "X", "mineral": "barite", "validated": False,
        "validation_reference": "r", "points": [{"dose_mg_l": 1, "maximum_supported_si": 1},
        {"dose_mg_l": 2, "maximum_supported_si": 2}]}
    assert client.post("/api/v1/water-compatibility", json=payload).status_code == 400
    payload = copy.deepcopy(PAYLOAD)
    payload["water_a"]["ions_mg_l"]["Na"] = "NaN"
    assert client.post("/api/v1/water-compatibility", json=payload).status_code == 422
