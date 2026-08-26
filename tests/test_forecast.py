from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from galit.forecast import (
    CorrosionIntegrityInput,
    ForecastCalibrationEvidence,
    ForecastConfig,
    ForecastHistory,
    ForecastMechanism,
    ForecastSnapshot,
    ObservedForecastEvent,
    ForecastStatus,
    forecast_well,
)
from galit.integrated import DiagnosisResult, WellCase
from galit.scale import WaterAnalysis
from galit.wax import WaxProperties
from galit.wellbore import FluidProperties, ProductionRate, ThermalParams, WellGeometry

UTC = timezone.utc
AS_OF = datetime(2026, 1, 1, tzinfo=UTC)


def case() -> WellCase:
    return WellCase("A", WellGeometry(1500, .062), ProductionRate(100, 50, 60),
        FluidProperties(), ThermalParams(), WaterAnalysis({"Na": 1, "Cl": 1}, 6, 40, 1e6),
        WaxProperties(30))


def diagnosis(**severity) -> DiagnosisResult:
    values = {"wax": .5, "halite": .4, "calcite": .3, "corrosion": .5}
    values.update(severity)
    return DiagnosisResult("A", [], [], [], [], severity=values,
                           corrosion={"rate_mm_yr": 1.0})


def snapshots(metric: str, values: list[float], step=10):
    return ForecastHistory(tuple(ForecastSnapshot("A", AS_OF-timedelta(days=step*(len(values)-i)),
        **{metric: value}) for i, value in enumerate(values)))


def by_mechanism(result, mechanism):
    return next(event for event in result.events if event.mechanism is mechanism)


def test_no_history_has_no_fake_wax_scale_dates_or_probabilities():
    result = forecast_well(diagnosis(), case(), as_of=AS_OF)
    for mechanism in (ForecastMechanism.WAX, ForecastMechanism.HALITE, ForecastMechanism.CALCITE):
        event = by_mechanism(result, mechanism)
        assert event.status is ForecastStatus.SCREENING
        assert event.horizon_start_days is event.horizon_end_days is None
        assert event.probability is None and event.risk_band is not None
    assert by_mechanism(result, ForecastMechanism.PRODUCTION_DECLINE).status is ForecastStatus.UNAVAILABLE


def test_corrosion_needs_wall_parameters_then_screening_is_scenario():
    missing = by_mechanism(forecast_well(diagnosis(), case(), as_of=AS_OF), ForecastMechanism.CORROSION)
    assert missing.status is ForecastStatus.UNAVAILABLE and missing.horizon_start_days is None
    integrity = CorrosionIntegrityInput(5, 4, AS_OF)
    event = by_mechanism(forecast_well(diagnosis(), case(), as_of=AS_OF,
        corrosion_integrity=integrity), ForecastMechanism.CORROSION)
    assert event.status is ForecastStatus.SCREENING and event.horizon_start_days is not None
    assert event.probability is None and "scenario" in event.method


def test_production_history_crossing_and_deterministic_dates_ids_sorting():
    history = snapshots("oil_rate_m3_day", [100, 96, 92, 88, 84], 10)
    first = forecast_well(diagnosis(), case(), history=history, as_of=AS_OF)
    second = forecast_well(diagnosis(), case(), history=history, as_of=AS_OF)
    event = by_mechanism(first, ForecastMechanism.PRODUCTION_DECLINE)
    assert event.status is ForecastStatus.SCREENING
    assert 0 <= event.horizon_start_days <= event.horizon_end_days
    assert event.horizon_start_date <= event.horizon_end_date
    assert [(e.id, e.horizon_start_date) for e in first.events] == [(e.id, e.horizon_start_date) for e in second.events]
    starts = [e.horizon_start_days if e.horizon_start_days is not None else float("inf") for e in first.events]
    assert starts == sorted(starts)


@pytest.mark.parametrize("values", [[100, 100, 100, 100], [80, 85, 90, 95]])
def test_flat_or_improving_production_has_no_event(values):
    event = by_mechanism(forecast_well(diagnosis(), case(),
        history=snapshots("oil_rate_m3_day", values), as_of=AS_OF),
        ForecastMechanism.PRODUCTION_DECLINE)
    assert event.status is ForecastStatus.UNAVAILABLE and event.horizon_start_days is None


def test_insufficient_span_and_invalid_history():
    short = snapshots("oil_rate_m3_day", [100, 99, 98], 1)
    assert by_mechanism(forecast_well(diagnosis(), case(), history=short, as_of=AS_OF),
        ForecastMechanism.PRODUCTION_DECLINE).status is ForecastStatus.UNAVAILABLE
    future = ForecastHistory((ForecastSnapshot("A", AS_OF+timedelta(days=1), oil_rate_m3_day=10),))
    with pytest.raises(ValueError, match="after as_of"):
        forecast_well(diagnosis(), case(), history=future, as_of=AS_OF)
    with pytest.raises(ValueError):
        ForecastSnapshot("A", AS_OF, wax_severity=1.1)


def test_duplicate_timestamps_are_aggregated_and_reported():
    rows = [ForecastSnapshot("A", AS_OF-timedelta(days=30-i*10), oil_rate_m3_day=100-i*5)
            for i in range(4)]
    rows.append(ForecastSnapshot("A", rows[-1].timestamp, oil_rate_m3_day=80))
    event = by_mechanism(forecast_well(diagnosis(), case(),
        history=ForecastHistory(tuple(rows)), as_of=AS_OF), ForecastMechanism.PRODUCTION_DECLINE)
    assert any("duplicate" in limitation for limitation in event.limitations)


def test_calibrated_is_gated_by_valid_metrics_and_probability_is_artifact_only():
    base_history = snapshots("wax_severity", [.2, .3, .4, .5], 10)
    history = ForecastHistory(base_history.snapshots, dataset_id="d")
    invalid = ForecastCalibrationEvidence("x", "d", "calibrated-not-field-validated", 100, .1,
        (ForecastMechanism.WAX,), {ForecastMechanism.WAX: .76})
    event = by_mechanism(forecast_well(diagnosis(), case(), history=history, as_of=AS_OF,
        calibration=invalid), ForecastMechanism.WAX)
    assert event.status is ForecastStatus.SCREENING and event.probability is None
    valid = replace(
        invalid, validation_status="holdout-validated", probability_horizon_days=90,
        endpoints={ForecastMechanism.WAX: "wax_threshold_within_90_days"},
    )
    event = by_mechanism(forecast_well(diagnosis(), case(), history=history, as_of=AS_OF,
        calibration=valid), ForecastMechanism.WAX)
    assert event.status is ForecastStatus.CALIBRATED and event.probability == .76
    assert event.production_ready


def test_future_outcomes_integrity_and_nonfinite_severity_are_rejected():
    future_event = ObservedForecastEvent(
        "future-1", "A", AS_OF + timedelta(days=1), ForecastMechanism.WAX
    )
    with pytest.raises(ValueError, match="observed event.*after as_of"):
        forecast_well(diagnosis(), case(),
                      history=ForecastHistory(events=(future_event,)), as_of=AS_OF)
    with pytest.raises(ValueError, match="integrity measurement is after as_of"):
        forecast_well(diagnosis(), case(), as_of=AS_OF,
                      corrosion_integrity=CorrosionIntegrityInput(
                          5, 4, AS_OF + timedelta(days=1)))
    with pytest.raises(ValueError, match="severity wax"):
        forecast_well(diagnosis(wax=float("nan")), case(), as_of=AS_OF)


def test_calibration_requires_matching_history_dataset():
    history = snapshots("wax_severity", [.2, .3, .4, .5], 10)
    evidence = ForecastCalibrationEvidence(
        "x", "different", "holdout-validated", 100, .1,
        (ForecastMechanism.WAX,), {ForecastMechanism.WAX: .76},
    )
    event = by_mechanism(forecast_well(
        diagnosis(), case(), history=history, as_of=AS_OF, calibration=evidence,
    ), ForecastMechanism.WAX)
    assert event.status is ForecastStatus.SCREENING and event.probability is None


def test_config_and_integrity_validation():
    with pytest.raises(ValueError):
        ForecastConfig(min_history_points=2)
    with pytest.raises(ValueError):
        CorrosionIntegrityInput(4, 4, AS_OF)
    with pytest.raises(ValueError):
        forecast_well(diagnosis(), replace(case(), name="B"))
