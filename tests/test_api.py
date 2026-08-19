"""Контрактные тесты REST API: запрос-ответ, консистентность с ядром, ошибки.

Полезная нагрузка взята из example_one_well.py (Речицкая 123), чтобы
тест работал на реалистичных промысловых данных.
"""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from api import app
from galit import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WaterAnalysis,
    WaxProperties,
    WellCase,
    WellGeometry,
    diagnose,
)

client = TestClient(app)

WELL = {
    "name": "Речицкая 123",
    "geometry": {
        "depth_m": 3200.0,
        "tubing_id_m": 0.062,
        "inclination_deg": 15.0,
    },
    "rate": {
        "q_oil_m3d": 8.0,
        "q_water_m3d": 72.0,
        "gor_m3m3": 65.0,
    },
    "fluid": {
        "gamma_oil": 0.86,
        "gamma_gas": 0.78,
        "salinity_ppm": 290_000.0,
    },
    "thermal": {
        "t_surface_c": 8.0,
        "geothermal_grad": 0.033,
        "u_to": 15.0,
        "production_days": 400.0,
    },
    "water": {
        "ions_mg_l": {
            "Na": 95_000.0,
            "Cl": 205_000.0,
            "Ca": 28_000.0,
            "Mg": 3_100.0,
            "K": 1_800.0,
            "HCO3": 130.0,
            "SO4": 250.0,
        },
        "ph": 6.0,
        "t_c": 40.0,
        "p_pa": 5e6,
    },
    "wax": {
        "wat_stock_tank_c": 34.0,
        "wax_content_pct": 6.5,
    },
    "co2_mol_frac": 0.012,
    "inhibitor_efficiency": 0.0,
    "lift_type": "ЭЦН",
    "p_wellhead_pa": 1.4e6,
}


def _reference_case() -> WellCase:
    return WellCase(
        name=WELL["name"],
        geometry=WellGeometry(**WELL["geometry"]),
        rate=ProductionRate(**WELL["rate"]),
        fluid=FluidProperties(**WELL["fluid"]),
        thermal=ThermalParams(**WELL["thermal"]),
        water=WaterAnalysis(**WELL["water"]),
        wax=WaxProperties(**WELL["wax"]),
        co2_mol_frac=WELL["co2_mol_frac"],
        inhibitor_efficiency=WELL["inhibitor_efficiency"],
        lift_type=WELL["lift_type"],
        p_wellhead_pa=WELL["p_wellhead_pa"],
    )


class TestDiagnoseEndpoint:

    def test_happy_path_contract(self):
        resp = client.post("/api/v1/diagnose", json=WELL)
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "integrated_risk", "dominant", "wax_onset_m",
            "recommendation", "warnings",
        }
        assert 0.0 <= body["integrated_risk"] <= 1.0
        assert body["dominant"] in {"halite", "calcite", "wax", "corrosion"}
        assert isinstance(body["recommendation"], str) and body["recommendation"]
        assert isinstance(body["warnings"], list)

    def test_matches_core_direct_call(self):
        """Слой отображения не искажает результат ядра."""
        reference = diagnose(_reference_case())
        body = client.post("/api/v1/diagnose", json=WELL).json()
        assert body["integrated_risk"] == pytest.approx(reference.integrated_risk)
        assert body["dominant"] == reference.dominant
        assert body["wax_onset_m"] == pytest.approx(reference.wax_onset_m)
        assert body["recommendation"] == reference.recommendation

    def test_legacy_contract_keeps_null_wax_onset(self):
        no_wax = copy.deepcopy(WELL)
        no_wax["wax"]["wat_stock_tank_c"] = -50.0
        body = client.post("/api/v1/diagnose", json=no_wax).json()
        assert set(body) == {
            "integrated_risk", "dominant", "wax_onset_m",
            "recommendation", "warnings",
        }
        assert body["wax_onset_m"] is None

    def test_uncertainty_opt_in_adds_policy_and_intervals(self):
        resp = client.post(
            "/api/v1/diagnose?include_uncertainty=true&uncertainty_seed=7&uncertainty_samples=20",
            json=WELL,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert {"policy", "uncertainty"} <= set(body)
        assert body["policy"]["id"] == "galit-baseline"
        assert body["uncertainty"]["seed"] == 7
        assert body["uncertainty"]["samples"] == 20
        assert set(body["uncertainty"]["integrated_risk"]) == {"p05", "p50", "p95"}

    def test_defaults_applied_when_optional_fields_omitted(self):
        minimal = copy.deepcopy(WELL)
        for key in ("co2_mol_frac", "inhibitor_efficiency", "lift_type", "p_wellhead_pa"):
            minimal.pop(key)
        resp = client.post("/api/v1/diagnose", json=minimal)
        assert resp.status_code == 200

    def test_production_mode_rejects_omitted_critical_defaults_with_400(self):
        minimal = copy.deepcopy(WELL)
        minimal.pop("co2_mol_frac")
        minimal.pop("inhibitor_efficiency")
        resp = client.post("/api/v1/diagnose?production_mode=true", json=minimal)
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert isinstance(detail["reasons"], list)
        assert "co2_mol_frac" in str(detail)

    def test_production_mode_preserves_422_for_invalid_payload(self):
        bad = copy.deepcopy(WELL)
        del bad["thermal"]["u_to"]
        resp = client.post("/api/v1/diagnose?production_mode=true", json=bad)
        assert resp.status_code == 422

    def test_unknown_ion_rejected(self):
        bad = copy.deepcopy(WELL)
        bad["water"]["ions_mg_l"]["Uranium"] = 1.0
        resp = client.post("/api/v1/diagnose", json=bad)
        assert resp.status_code == 422
        assert "Uranium" in resp.text

    def test_missing_required_field_rejected(self):
        bad = copy.deepcopy(WELL)
        del bad["thermal"]["u_to"]
        resp = client.post("/api/v1/diagnose", json=bad)
        assert resp.status_code == 422

    @pytest.mark.parametrize("field,value", [
        ("depth_m", -1.0),
        ("tubing_id_m", 0.0),
        ("inclination_deg", 120.0),
    ])
    def test_unphysical_geometry_rejected(self, field: str, value: float):
        bad = copy.deepcopy(WELL)
        bad["geometry"][field] = value
        resp = client.post("/api/v1/diagnose", json=bad)
        assert resp.status_code == 422


class TestHealthEndpoint:

    def test_health(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
