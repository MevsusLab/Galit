"""Регрессионный baseline первой безопасной партии физических исправлений."""
from __future__ import annotations

import pytest

from galit import diagnose
from galit.integrated import WellCase
from galit.scale import (
    WaterAnalysis,
    calcite_degas_ph_screening,
    stiff_davis_index,
)
from galit.wax import (
    WaxProperties,
    wax_deposition_severity,
    wax_onset_depth,
)
from galit.wellbore import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
)


def test_wax_no_deposition():
    depths = [0.0, 500.0, 1000.0]
    temps = [40.0, 45.0, 50.0]
    onset, wat = wax_onset_depth(
        depths, temps, [1e5] * 3, WaxProperties(20.0), pb_pa=1e5,
    )
    assert onset is None
    assert wax_deposition_severity(depths, temps, wat, onset, WaxProperties(20.0)) == 0.0


def test_wax_partial_zone_uses_onset_fraction_from_wellhead():
    depths = [0.0, 500.0, 1000.0]
    temps = [10.0, 20.0, 30.0]
    wax = WaxProperties(20.06, wax_content_pct=10.0)  # WAT=20 C при 0.1 МПа
    onset, wat_profile = wax_onset_depth(
        depths, temps, [1e5] * 3, wax, pb_pa=1e5,
    )
    assert onset == pytest.approx(500.0)
    severity = wax_deposition_severity(depths, temps, wat_profile, onset, wax)
    # zone_frac=onset/total=0.5; старая инвертированная геометрия давала
    # тот же результат только в середине, поэтому отдельно фиксируем края ниже.
    assert 0.0 < severity < 0.5


def test_wax_whole_well_deposition_is_full_zone():
    depths = [0.0, 500.0, 1000.0]
    temps = [5.0, 10.0, 15.0]
    wax = WaxProperties(30.06, wax_content_pct=10.0)
    onset, wat_profile = wax_onset_depth(
        depths, temps, [1e5] * 3, wax, pb_pa=1e5,
    )
    assert onset == 1000.0
    assert wax_deposition_severity(depths, temps, wat_profile, onset, wax) > 0.5


def test_calcite_decompression_with_co2_does_not_lower_ph_or_index():
    ions = {"Na": 20_000.0, "Cl": 35_000.0, "Ca": 2_000.0, "HCO3": 400.0}
    initial = WaterAnalysis(ions, ph=6.0, t_c=35.0, p_pa=10e6)
    ph_wh, warning = calcite_degas_ph_screening(
        initial.ph, t_c=35.0, p_initial_pa=10e6,
        p_local_pa=1e6, co2_mol_frac=0.02,
    )
    wellhead = WaterAnalysis(ions, ph=ph_wh, t_c=35.0, p_pa=1e6)
    assert ph_wh >= initial.ph
    assert stiff_davis_index(wellhead) >= stiff_davis_index(initial)
    assert warning is not None and "screening" in warning


def _case() -> WellCase:
    return WellCase(
        name="regression-baseline",
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


def test_diagnosis_regression_contract_and_corrosion_profile():
    result = diagnose(_case())
    corr = result.corrosion
    # Существующий публичный контракт сохранён.
    assert {"rate_mm_yr", "category", "limiting"} <= corr.keys()
    assert len(corr["profile"]) == len(result.depths) == len(result.temps) == len(result.pressures)
    assert corr["rate_mm_yr"] == max(node["rate_mm_yr"] for node in corr["profile"])
    assert corr["depth_of_max_m"] in result.depths
    assert all(
        node["depth_m"] == z and node["t_c"] == t and node["p_pa"] == p
        for node, z, t, p in zip(corr["profile"], result.depths, result.temps, result.pressures)
    )


def test_diagnosis_propagates_checked_stiff_davis_and_screening_warnings():
    result = diagnose(_case())
    assert any("Stiff-Davis" in warning for warning in result.warnings)
    assert any("screening" in warning for warning in result.warnings)
    assert result.scale["ph_wellhead"] >= result.scale["ph_initial"]
