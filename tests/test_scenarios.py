from copy import deepcopy
import math

import pytest

import galit
from galit.synthetic import make_fund


def case():
    return make_fund(1, seed=7)[0]


def test_direct_changes_combination_and_baseline_immutability():
    baseline = case()
    original = deepcopy(baseline)
    result = galit.compare_scenario(baseline, galit.ScenarioChanges(
        oil_rate_relative_change=-.1, water_rate_delta_m3_day=2,
        wellhead_pressure_relative_change=.05, surface_temperature_delta_c=2,
    ))
    assert result.status is galit.ScenarioStatus.PARTIAL
    assert result.after.forecast_oil_rate_m3_day == pytest.approx(original.rate.q_oil_m3d * .9)
    assert baseline.rate.q_oil_m3d == original.rate.q_oil_m3d
    assert baseline.p_wellhead_pa == original.p_wellhead_pa
    assert set(result.delta["severity"]) == {"halite", "calcite", "wax", "corrosion"}
    assert "economics" in result.missing_inputs


def test_actions_without_override_are_partial_and_do_not_invent_effect():
    baseline = case()
    result = galit.compare_scenario(baseline, galit.ScenarioChanges(
        inhibitor_dosage_delta_mg_l=25, wash_treatment=True, operating_mode="ШГН",
    ))
    assert result.status is galit.ScenarioStatus.PARTIAL
    assert result.after.parameters["inhibitor_efficiency_fraction"] == baseline.inhibitor_efficiency
    assert any("no hidden coefficient" in warning for warning in result.warnings)
    assert any("not a diagnostic input" in warning for warning in result.warnings)


def test_explicit_override_and_economics_are_consistent():
    baseline = case()
    result = galit.compare_scenario(
        baseline,
        galit.ScenarioChanges(
            inhibitor_dosage_delta_mg_l=20,
            effect_override=galit.EffectOverride(inhibitor_efficiency=.8, source="supplier test"),
        ),
        galit.ScenarioEconomics(
            horizon_days=30, event_probability=.5, treatment_efficiency=.6,
            product_price_per_m3=100, operating_loss_per_day=10,
            treatment_cost=500, currency="BYN",
        ),
    )
    assert result.status is galit.ScenarioStatus.AVAILABLE
    assert result.after.parameters["inhibitor_efficiency_fraction"] == .8
    b = result.economics.breakdown
    assert b.net_expected_effect == pytest.approx(b.potential_avoided_damage - b.total_treatment_cost)
    assert b.roi_ratio == pytest.approx(b.net_expected_effect / b.total_treatment_cost)


def test_missing_probability_never_uses_screening_risk_as_probability():
    result = galit.compare_scenario(case(), galit.ScenarioChanges(surface_temperature_delta_c=1),
        galit.ScenarioEconomics(horizon_days=30, treatment_efficiency=.5))
    assert result.economics.breakdown.expected_damage_without_treatment is None
    assert "event_probability" in result.missing_inputs
    assert any("not calibrated probabilities" in item for item in result.assumptions)


@pytest.mark.parametrize("factory", [
    lambda: galit.ScenarioChanges(oil_rate_delta_m3_day=1, oil_rate_relative_change=.1),
    lambda: galit.ScenarioChanges(surface_temperature_delta_c=math.inf),
    lambda: galit.ScenarioChanges(inhibitor_dosage_delta_mg_l=-1),
    lambda: galit.EffectOverride(inhibitor_efficiency=.5, source=None),
])
def test_invalid_nonfinite_ranges_and_incompatible_changes(factory):
    with pytest.raises(ValueError):
        factory()


def test_resulting_negative_value_rejected():
    with pytest.raises(ValueError, match="oil rate"):
        galit.compare_scenario(case(), galit.ScenarioChanges(oil_rate_relative_change=-2))
