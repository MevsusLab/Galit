"""Контрактные тесты политики риска и сценарной неопределённости."""
from __future__ import annotations

import copy

import pytest

from galit import RiskPolicy, UncertaintyConfig, diagnose
from galit.integrated import MECHANISMS, WellCase
from galit.scale import WaterAnalysis
from galit.wax import WaxProperties
from galit.wellbore import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
)


def _case() -> WellCase:
    return WellCase(
        name="risk-policy-uncertainty",
        geometry=WellGeometry(depth_m=1800.0, tubing_id_m=0.062),
        rate=ProductionRate(q_oil_m3d=10.0, q_water_m3d=50.0, gor_m3m3=60.0),
        fluid=FluidProperties(),
        thermal=ThermalParams(),
        water=WaterAnalysis(
            {"Na": 95_000.0, "Cl": 205_000.0, "Ca": 28_000.0,
             "Mg": 3_100.0, "HCO3": 130.0},
            ph=6.0, t_c=60.0, p_pa=8e6,
        ),
        wax=WaxProperties(42.0),
        co2_mol_frac=0.02,
        p_wellhead_pa=1.2e6,
    )


def test_default_policy_leaves_baseline_unchanged():
    implicit = diagnose(_case())
    explicit = diagnose(_case(), risk_policy=RiskPolicy())
    assert explicit.integrated_risk == pytest.approx(implicit.integrated_risk)
    assert explicit.dominant == implicit.dominant
    assert explicit.severity == implicit.severity
    assert explicit.recommendation == implicit.recommendation


def test_custom_weights_predictably_affect_risk_and_dominant():
    baseline = diagnose(_case())
    target = max(MECHANISMS, key=lambda key: baseline.severity[key])
    weights = {key: float(key == target) for key in MECHANISMS}
    custom = diagnose(
        _case(),
        risk_policy=RiskPolicy(policy_id="test", version="1", weights=weights),
    )
    assert custom.integrated_risk == pytest.approx(baseline.severity[target])
    assert custom.dominant == target
    assert custom.mechanism_weights == weights


@pytest.mark.parametrize("kwargs", [
    {"weights": {"halite": 1.0}},
    {"weights": {"halite": 0.3, "calcite": 0.3, "wax": 0.3, "corrosion": 0.3}},
    {"weights": {"halite": -0.1, "calcite": 0.2, "wax": 0.4, "corrosion": 0.5}},
    {"severity_warn": 0.8, "severity_critical": 0.2},
    {"policy_id": ""},
])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RiskPolicy(**kwargs)


def test_uncertainty_is_seeded_and_reproducible():
    config = UncertaintyConfig(seed=1234, samples=20)
    first = diagnose(_case(), uncertainty=config).uncertainty
    second = diagnose(_case(), uncertainty=config).uncertainty
    assert first == second


def test_uncertainty_percentiles_are_ordered_and_bounded():
    uncertainty = diagnose(
        _case(), uncertainty=UncertaintyConfig(seed=17, samples=20)
    ).uncertainty
    assert uncertainty is not None
    intervals = [uncertainty.integrated_risk, *uncertainty.mechanisms.values()]
    for interval in intervals:
        assert interval is not None
        assert 0.0 <= interval.p05 <= interval.p50 <= interval.p95 <= 1.0
    onset = uncertainty.wax_onset_m
    assert onset is not None
    if onset.p05 is not None:
        assert 0.0 <= onset.p05 <= onset.p50 <= onset.p95 <= _case().geometry.depth_m * 1.2
    assert 0.0 <= uncertainty.probability_of_deposition <= 1.0


def test_uncertainty_does_not_mutate_original_case():
    case = _case()
    original = copy.deepcopy(case)
    diagnose(case, uncertainty=UncertaintyConfig(seed=9, samples=20))
    assert case == original
