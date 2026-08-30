from __future__ import annotations

import copy
import math

import pytest

from galit import (
    CompatibilityWater, DoseResponseCurve, DoseResponsePoint, ProfilePoint,
    default_mix_fractions, evaluate_compatibility, mix_waters,
)


def waters():
    a = CompatibilityWater({"Na": 20000, "Cl": 32000, "Ca": 4000, "Ba": 1000,
                            "HCO3": 50, "SO4": 0}, 6.2, 25, 5e6, "formation")
    b = CompatibilityWater({"Na": 1000, "Cl": 1500, "Ca": 100, "Ba": 0,
                            "HCO3": 300, "SO4": 3000}, 7.0, 25, 1e5, "injection")
    return a, b


def test_default_grid_is_inclusive_and_deterministic():
    grid = default_mix_fractions()
    assert len(grid) == 101 and grid[0] == 0 and grid[-1] == 1
    assert grid == default_mix_fractions()


def test_mix_is_typed_linear_and_does_not_mutate_inputs():
    ions_a = {"Na": 100.0, "Cl": 200.0}
    a = CompatibilityWater(ions_a, 6, 20, 1e5)
    b = CompatibilityWater({"Na": 300.0, "Cl": 400.0}, 8, 40, 3e5)
    mixed = mix_waters(a, b, .25)
    assert mixed.ions_mg_l["Na"] == pytest.approx(150)
    assert mixed.t_c == pytest.approx(25)
    assert mixed.ph == pytest.approx(-math.log10(.75e-6 + .25e-8))
    ions_a["Na"] = 999
    assert a.ions_mg_l["Na"] == 100
    with pytest.raises(TypeError):
        a.ions_mg_l["Na"] = 1


def test_missing_chemistry_is_unavailable_not_fabricated():
    a = CompatibilityWater({"Na": 1000}, 7, 25, 1e5)
    result = evaluate_compatibility(a, a, [0, 1])
    row = result.ratios[0]
    assert row.minerals["barite"].saturation_index is None
    assert row.minerals["gypsum"].saturation_index is None
    assert row.minerals["halite"].saturation_index is None


def test_missing_ion_in_either_water_is_not_treated_as_measured_zero():
    a = CompatibilityWater({"Na": 1000, "Cl": 2000}, 7, 25, 1e5)
    b = CompatibilityWater({"Na": 500}, 7, 25, 1e5)
    result = evaluate_compatibility(a, b, [0, .5, 1])
    assert result.ratios[0].minerals["halite"].saturation_index is not None
    assert result.ratios[1].minerals["halite"].saturation_index is None
    assert result.ratios[2].minerals["halite"].saturation_index is None


def test_all_minerals_dangerous_ratio_intervals_and_locations():
    a, b = waters()
    result = evaluate_compatibility(a, b, [0, .25, .5, .75, 1], [
        ProfilePoint(0, 25, 1e5), ProfilePoint(1000, 35, 8e6)
    ])
    assert set(result.ratios[2].minerals) == {"calcite", "barite", "gypsum", "halite"}
    assert result.dangerous_fraction_b in {0, .25, .5, .75, 1}
    assert result.dangerous_ratio_a_to_b is not None
    assert result.unsafe_intervals
    assert len(result.deposition_locations) == 4
    assert all(item.flow_direction == "bottom_to_surface" for item in result.deposition_locations)


def test_dose_is_null_without_curve_and_uses_only_validated_curve():
    a, b = waters()
    no_curve = evaluate_compatibility(a, b, [.5])
    assert no_curve.inhibitor.dose_mg_l is None
    assert no_curve.inhibitor.status == "laboratory_test_required"
    controlling = max(no_curve.ratios[0].minerals.values(),
                      key=lambda item: -math.inf if item.saturation_index is None else item.saturation_index)
    curve = DoseResponseCurve("Product X", controlling.mineral,
        (DoseResponsePoint(10, -10), DoseResponsePoint(20, 100)), True, "lab-report-1")
    result = evaluate_compatibility(a, b, [.5], dose_response=curve)
    assert result.inhibitor.dose_mg_l == 20
    with pytest.raises(ValueError, match="validated"):
        DoseResponseCurve("X", "barite", (DoseResponsePoint(1, 1), DoseResponsePoint(2, 2)), False, "r")
    with pytest.raises(ValueError, match="monotonic"):
        DoseResponseCurve("X", "barite", (DoseResponsePoint(1, 2), DoseResponsePoint(2, 1)), True, "r")


def test_validation_rejects_nonfinite_and_unbounded_or_unsorted_arrays():
    a, b = waters()
    with pytest.raises(ValueError, match="finite"):
        CompatibilityWater({"Na": float("nan")}, 7, 25, 1e5)
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_compatibility(a, b, [.5, .4])
    with pytest.raises(ValueError, match="1001"):
        evaluate_compatibility(a, b, [i / 1001 for i in range(1002)])
