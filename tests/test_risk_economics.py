"""Per-well risk economics tests."""
import math

import pytest

import galit


def explicit(**overrides):
    values = dict(
        event_probability=1.0, horizon_days=31, treatment_efficiency=1.0,
        event_downtime_days=0, treatment_downtime_days=0,
        oil_rate_m3_day=1, product_price_per_m3=1000,
        operating_loss_per_day=0, treatment_cost=8000, currency="BYN",
    )
    values.update(overrides)
    return galit.RiskEconomicsInput(**values)


def test_documented_8000_31000_23000_example_is_explicit():
    result = galit.calculate_risk_economics(explicit())
    b = result.breakdown
    assert result.status is galit.RiskEconomicsStatus.AVAILABLE
    assert b.expected_production_loss_m3 == 31
    assert b.expected_damage_without_treatment == 31_000
    assert b.recommended_treatment_cost == 8_000
    assert b.potential_avoided_damage == 31_000
    assert b.net_expected_effect == 23_000
    assert b.roi_ratio == pytest.approx(2.875)


def test_incomplete_inputs_are_partial_and_not_fabricated():
    result = galit.calculate_risk_economics(explicit(
        event_probability=None, product_price_per_m3=None,
        operating_loss_per_day=None, treatment_cost=None, currency=None,
    ))
    assert result.status is galit.RiskEconomicsStatus.UNAVAILABLE
    assert not result.data_sufficient
    assert result.breakdown.expected_production_loss_money is None
    assert result.breakdown.net_expected_effect is None


def test_zero_denominators_do_not_create_infinite_roi_or_payback():
    result = galit.calculate_risk_economics(explicit(
        oil_rate_m3_day=0, product_price_per_m3=0, treatment_cost=0,
    ))
    assert result.breakdown.net_expected_effect == 0
    assert result.breakdown.roi_ratio is None
    assert result.breakdown.payback_ratio is None


@pytest.mark.parametrize("field,value", [
    ("horizon_days", 0), ("treatment_cost", -1),
    ("event_probability", 1.1), ("oil_rate_m3_day", math.inf),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValueError):
        explicit(**{field: value})
