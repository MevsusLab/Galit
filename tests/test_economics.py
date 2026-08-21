"""Unit economics and pilot break-even tests."""
import math

import pytest

from galit.economics import (
    Assumption,
    PilotOutcomeMix,
    PilotUnitEconomicsInput,
    compute_pilot_break_even,
    pilot_sensitivity,
)


def asm(name: str, value: float, low=None, high=None) -> Assumption:
    return Assumption(name, value, "BYN", "customer input", "уточнить", low, high)


def inputs(**overrides) -> PilotUnitEconomicsInput:
    values = {
        "pilot_cost": asm("Стоимость пилота", 100_000),
        "treatment_value": asm("Ценность обработки", 10_000, 5_000, 20_000),
        "failure_value": asm("Ценность отказа", 50_000, 25_000, 100_000),
        "downtime_day_value": asm("Ценность суток", 2_000, 1_000, 4_000),
        "saved_tonne_value": asm("Ценность тонны", 500, 250, 1_000),
    }
    values.update(overrides)
    return PilotUnitEconomicsInput(**values)


def test_break_even_separate_channels_and_mixed():
    result = compute_pilot_break_even(
        inputs(), PilotOutcomeMix(2, 1, 5, 40),
    )
    assert result.treatments_only == 10
    assert result.failures_only == 2
    assert result.downtime_days_only == 50
    assert result.saved_tonnes_only == 200
    assert result.mixed_value == 100_000
    assert result.mixed_gap == 0
    assert result.mixed_break_even
    assert result.mixed_break_even_share == 1


def test_zero_cost_is_immediately_break_even():
    result = compute_pilot_break_even(
        inputs(pilot_cost=asm("Стоимость пилота", 0)), PilotOutcomeMix(),
    )
    assert result.treatments_only == 0
    assert result.mixed_break_even
    assert result.mixed_break_even_share == 0


def test_zero_unit_value_is_explicitly_unavailable():
    result = compute_pilot_break_even(
        inputs(treatment_value=asm("Ценность обработки", 0)), PilotOutcomeMix(),
    )
    assert result.treatments_only is None
    assert result.mixed_break_even_share is None
    assert not result.mixed_break_even


@pytest.mark.parametrize("bad", [-1.0, math.inf, -math.inf, math.nan])
def test_invalid_input_values_rejected(bad):
    with pytest.raises(ValueError):
        compute_pilot_break_even(inputs(pilot_cost=asm("Стоимость пилота", bad)))


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
def test_invalid_outcome_values_rejected(bad):
    with pytest.raises(ValueError):
        compute_pilot_break_even(inputs(), PilotOutcomeMix(saved_tonnes=bad))


def test_compact_sensitivity_uses_assumption_ranges():
    rows = pilot_sensitivity(inputs(), PilotOutcomeMix(2, 1, 5, 40))
    assert len(rows) == 4
    assert all(low is not None and high is not None for _, low, high in rows)
