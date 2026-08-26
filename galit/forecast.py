"""Honest, deterministic time-to-event forecasts for GALIT.

This module deliberately separates a current physical diagnosis from a temporal
forecast.  A severity score is not a failure probability and, without repeated
observations, it is not a clock.  Screening results are scenario estimates;
``calibrated`` is gated by explicit holdout evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from .integrated import DiagnosisResult, WellCase


class ForecastStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    SCREENING = "screening"
    CALIBRATED = "calibrated"


class ForecastMechanism(str, Enum):
    WAX = "wax"
    CORROSION = "corrosion"
    HALITE = "halite"
    CALCITE = "calcite"
    PRODUCTION_DECLINE = "production_decline"


class LikelihoodCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class ForecastSnapshot:
    """One leakage-safe, pre-event observation of a well.

    All rates are daily field observations. Severity fields use GALIT's public
    0..1 scale. ``corrosion_wall_loss_mm`` is a cumulative measured wall loss,
    not the modelled instantaneous corrosion rate.
    """

    well: str
    timestamp: datetime
    wax_severity: float | None = None
    halite_severity: float | None = None
    calcite_severity: float | None = None
    corrosion_wall_loss_mm: float | None = None
    oil_rate_m3_day: float | None = None
    quality: str = "good"
    source: str = "measured"
    regime_id: str | None = None

    def __post_init__(self) -> None:
        if not self.well.strip():
            raise ValueError("ForecastSnapshot.well must be non-empty")
        _aware(self.timestamp, "ForecastSnapshot.timestamp")
        if self.quality not in {"good", "questionable", "bad"}:
            raise ValueError("snapshot quality must be good, questionable, or bad")
        if self.source not in {"measured", "derived", "laboratory"}:
            raise ValueError("snapshot source must be measured, derived, or laboratory")
        for name in ("wax_severity", "halite_severity", "calcite_severity"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name in ("corrosion_wall_loss_mm", "oil_rate_m3_day"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ObservedForecastEvent:
    """Observed outcome contract for later calibration/evaluation."""

    event_id: str
    well: str
    timestamp: datetime
    mechanism: ForecastMechanism
    outcome: bool = True
    source: str = "measured"

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.well.strip():
            raise ValueError("observed event id and well must be non-empty")
        _aware(self.timestamp, "ObservedForecastEvent.timestamp")
        if self.source not in {"measured", "laboratory"}:
            raise ValueError("observed event source must be measured or laboratory")


@dataclass(frozen=True)
class ForecastHistory:
    """Typed temporal input; snapshots and outcomes remain separate."""

    snapshots: tuple[ForecastSnapshot, ...] = ()
    events: tuple[ObservedForecastEvent, ...] = ()
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True)
class CorrosionIntegrityInput:
    """Measured wall reserve used by the corrosion screening scenario."""

    current_wall_thickness_mm: float
    minimum_allowable_wall_thickness_mm: float
    measured_at: datetime
    source: str = "measured"

    def __post_init__(self) -> None:
        _aware(self.measured_at, "CorrosionIntegrityInput.measured_at")
        for name in ("current_wall_thickness_mm", "minimum_allowable_wall_thickness_mm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.current_wall_thickness_mm <= self.minimum_allowable_wall_thickness_mm:
            raise ValueError("current wall thickness must exceed the minimum allowable thickness")
        if self.source not in {"measured", "laboratory"}:
            raise ValueError("wall thickness source must be measured or laboratory")

    @property
    def available_wall_loss_mm(self) -> float:
        return self.current_wall_thickness_mm - self.minimum_allowable_wall_thickness_mm


@dataclass(frozen=True)
class ForecastCalibrationEvidence:
    """Evidence gate for calibrated labels and optional supplied probabilities.

    This is an evaluation artifact contract, not a fitted model. Probabilities
    are consumed only when explicitly supplied by the validated artifact.
    """

    artifact_id: str
    dataset_id: str
    validation_status: str
    holdout_n: int
    brier_score: float
    mechanisms: tuple[ForecastMechanism, ...]
    probabilities: Mapping[ForecastMechanism, float] = field(default_factory=dict)
    synthetic: bool = False
    probability_horizon_days: int | None = None
    endpoints: Mapping[ForecastMechanism, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id.strip() or not self.dataset_id.strip():
            raise ValueError("calibration artifact_id and dataset_id must be non-empty")
        if self.holdout_n < 0:
            raise ValueError("calibration holdout_n must be non-negative")
        if not math.isfinite(self.brier_score) or not 0.0 <= self.brier_score <= 1.0:
            raise ValueError("calibration brier_score must be finite and within [0, 1]")
        mechanisms = tuple(ForecastMechanism(item) for item in self.mechanisms)
        if len(set(mechanisms)) != len(mechanisms):
            raise ValueError("calibration mechanisms must be unique")
        object.__setattr__(self, "mechanisms", mechanisms)
        probabilities = {
            ForecastMechanism(key): float(value) for key, value in self.probabilities.items()
        }
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in probabilities.values()):
            raise ValueError("calibrated probabilities must be finite and within [0, 1]")
        if set(probabilities) - set(mechanisms):
            raise ValueError("probabilities may only reference declared mechanisms")
        if self.probability_horizon_days is not None and self.probability_horizon_days < 1:
            raise ValueError("probability_horizon_days must be positive")
        endpoints = {ForecastMechanism(key): str(value) for key, value in self.endpoints.items()}
        if set(endpoints) - set(mechanisms):
            raise ValueError("endpoints may only reference declared mechanisms")
        object.__setattr__(self, "probabilities", MappingProxyType(probabilities))
        object.__setattr__(self, "endpoints", MappingProxyType(endpoints))


@dataclass(frozen=True)
class TemporalEvidence:
    """Machine-readable evidence behind a temporal estimate (additive contract)."""

    points: int = 0
    span_days: float | None = None
    trend_consistency: float | None = None
    quality: str = "insufficient"
    regime_id: str | None = None
    regime_compatible: bool = True


@dataclass(frozen=True)
class ForecastCalibrationMetadata:
    artifact_id: str | None = None
    dataset_id: str | None = None
    validation_status: str = "not_supplied"
    holdout_n: int | None = None
    brier_score: float | None = None
    endpoint: str | None = None
    horizon_days: int | None = None
    matched: bool = False


@dataclass(frozen=True)
class ForecastConfig:
    wax_critical_severity: float = 0.60
    halite_deposition_severity: float = 0.60
    calcite_deposition_severity: float = 0.60
    production_decline_fraction: float = 0.20
    min_history_points: int = 4
    min_history_span_days: float = 21.0
    max_horizon_days: float = 365.0
    minimum_trend_consistency: float = 0.70
    scenario_slope_quantiles: tuple[float, float] = (0.25, 0.75)
    corrosion_rate_uncertainty_fraction: float = 0.30
    min_calibration_holdout_n: int = 30
    max_calibration_brier: float = 0.25
    strict_history: bool = True

    def __post_init__(self) -> None:
        for name in ("wax_critical_severity", "halite_deposition_severity",
                     "calcite_deposition_severity", "production_decline_fraction",
                     "minimum_trend_consistency", "corrosion_rate_uncertainty_fraction",
                     "max_calibration_brier"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.min_history_points < 3 or self.min_calibration_holdout_n < 1:
            raise ValueError("history/calibration minimum counts are invalid")
        if (not math.isfinite(self.min_history_span_days) or
                not math.isfinite(self.max_horizon_days) or
                self.min_history_span_days <= 0 or self.max_horizon_days <= 0):
            raise ValueError("history span and horizon must be finite and positive")
        low, high = self.scenario_slope_quantiles
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("scenario slope quantiles must satisfy 0 <= low <= high <= 1")


@dataclass(frozen=True)
class ForecastEvent:
    id: str
    well: str
    mechanism: ForecastMechanism
    title: str
    status: ForecastStatus
    horizon_start_days: float | None
    horizon_end_days: float | None
    horizon_start_date: date | None
    horizon_end_date: date | None
    probability: float | None
    risk_band: tuple[float, float] | None
    likelihood: LikelihoodCategory
    current_risk: float | None
    threshold: float | None
    basis: str
    method: str
    assumptions: tuple[str, ...]
    required_inputs: tuple[str, ...]
    limitations: tuple[str, ...]
    production_ready: bool
    actionable: bool
    probability_endpoint: str | None = None
    probability_horizon_days: int | None = None
    evidence: TemporalEvidence = field(default_factory=TemporalEvidence)
    calibration: ForecastCalibrationMetadata = field(default_factory=ForecastCalibrationMetadata)

    def __post_init__(self) -> None:
        if self.status is ForecastStatus.SCREENING and self.probability is not None:
            raise ValueError("screening forecasts cannot contain an exact probability")
        if self.status is ForecastStatus.UNAVAILABLE and self.production_ready:
            raise ValueError("unavailable forecasts cannot be production-ready")
        if (self.horizon_start_days is None) != (self.horizon_end_days is None):
            raise ValueError("both horizon bounds must be supplied together")
        if self.horizon_start_days is not None:
            if not (0.0 <= self.horizon_start_days <= self.horizon_end_days):
                raise ValueError("invalid horizon bounds")
        for name in ("horizon_start_days", "horizon_end_days", "probability", "current_risk", "threshold"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.probability is not None and (not self.probability_endpoint or self.probability_horizon_days is None):
            raise ValueError("probability requires a matched endpoint and horizon")
        if self.probability is not None and not self.calibration.matched:
            raise ValueError("probability requires matched validated calibration metadata")
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be within [0, 1]")
        if self.risk_band is not None and (len(self.risk_band) != 2 or
                any(not math.isfinite(value) for value in self.risk_band) or
                not 0.0 <= self.risk_band[0] <= self.risk_band[1] <= 1.0):
            raise ValueError("risk_band must be finite, ordered, and within [0, 1]")


@dataclass(frozen=True)
class ForecastSummary:
    total: int
    unavailable: int
    screening: int
    calibrated: int
    actionable: int
    earliest_event_id: str | None


@dataclass(frozen=True)
class WellForecast:
    well: str
    as_of: datetime | None
    events: tuple[ForecastEvent, ...]
    summary: ForecastSummary

    @property
    def timeline(self) -> tuple[ForecastEvent, ...]:
        return self.events


@dataclass(frozen=True)
class _Trend:
    current: float
    slope_per_day: float
    low_slope: float
    high_slope: float
    span_days: float
    points: int
    consistency: float
    regime_id: str | None = None
    limitations: tuple[str, ...] = ()


_TITLES = {
    ForecastMechanism.WAX: "Wax critical / АСПО",
    ForecastMechanism.CORROSION: "Corrosion threshold exceedance",
    ForecastMechanism.HALITE: "Halite deposition",
    ForecastMechanism.CALCITE: "Calcite deposition",
    ForecastMechanism.PRODUCTION_DECLINE: "Production decline",
}
_METRICS = {
    ForecastMechanism.WAX: "wax_severity",
    ForecastMechanism.CORROSION: "corrosion_wall_loss_mm",
    ForecastMechanism.HALITE: "halite_severity",
    ForecastMechanism.CALCITE: "calcite_severity",
    ForecastMechanism.PRODUCTION_DECLINE: "oil_rate_m3_day",
}


def forecast_well(
    diagnosis: DiagnosisResult,
    case: WellCase,
    *,
    history: ForecastHistory | None = None,
    as_of: datetime | None = None,
    config: ForecastConfig | None = None,
    corrosion_integrity: CorrosionIntegrityInput | None = None,
    calibration: ForecastCalibrationEvidence | None = None,
) -> WellForecast:
    """Build five honest event contracts without changing diagnostic physics."""
    cfg = config or ForecastConfig()
    if diagnosis.well != case.name:
        raise ValueError("diagnosis.well and case.name must match")
    if as_of is not None:
        as_of = _aware(as_of, "as_of").astimezone(timezone.utc)
    snapshots, history_notes = _validated_history(history, case.name, as_of, cfg)
    if corrosion_integrity is not None and as_of is not None:
        measured_at = corrosion_integrity.measured_at.astimezone(timezone.utc)
        if measured_at > as_of:
            raise ValueError("corrosion integrity measurement is after as_of")
    diagnosis_ready = bool(diagnosis.quality.production_ready)
    calibration_valid, calibration_reason = _valid_calibration(calibration, history, cfg)
    if not diagnosis_ready:
        calibration_valid = False
        calibration_reason = "diagnosis.quality.production_ready is false"

    events: list[ForecastEvent] = []
    thresholds = {
        ForecastMechanism.WAX: cfg.wax_critical_severity,
        ForecastMechanism.HALITE: cfg.halite_deposition_severity,
        ForecastMechanism.CALCITE: cfg.calcite_deposition_severity,
    }
    for mechanism in (ForecastMechanism.WAX, ForecastMechanism.HALITE, ForecastMechanism.CALCITE):
        current = _severity(diagnosis, mechanism)
        trend, reason = _trend_for(snapshots, _METRICS[mechanism], cfg, increasing=True)
        if trend is not None:
            events.append(_trend_event(case.name, mechanism, current, thresholds[mechanism], trend,
                                       as_of, cfg, calibration, calibration_valid, history_notes))
        else:
            limitations = list(history_notes)
            if reason:
                limitations.append(reason)
            limitations.append("No deposition-rate evidence; current severity is not a time-to-deposition clock.")
            events.append(_event(case.name, mechanism, ForecastStatus.SCREENING,
                current=current, threshold=thresholds[mechanism], as_of=as_of,
                likelihood=_likelihood(current), risk_band=_risk_band(current),
                basis="Current GALIT physical severity only.",
                method="scenario screening; no temporal extrapolation",
                assumptions=("Current operating conditions remain representative.",),
                required=(f"Repeated {_METRICS[mechanism]} observations", "Observed deposition/event outcomes"),
                limitations=limitations, actionable=current >= thresholds[mechanism]))

    events.append(_corrosion_event(diagnosis, case, snapshots, as_of, cfg,
                                    corrosion_integrity, calibration, calibration_valid,
                                    calibration_reason, history_notes))
    events.append(_production_event(diagnosis, case, snapshots, as_of, cfg,
                                    calibration, calibration_valid, calibration_reason,
                                    history_notes))
    events.sort(key=_sort_key)
    counts = {status: sum(event.status is status for event in events) for status in ForecastStatus}
    earliest = next((event.id for event in events if event.horizon_start_days is not None), None)
    summary = ForecastSummary(len(events), counts[ForecastStatus.UNAVAILABLE],
                              counts[ForecastStatus.SCREENING], counts[ForecastStatus.CALIBRATED],
                              sum(event.actionable for event in events), earliest)
    return WellForecast(case.name, as_of, tuple(events), summary)


def _trend_event(well: str, mechanism: ForecastMechanism, fallback_current: float,
                 threshold: float, trend: _Trend, as_of: datetime | None,
                 cfg: ForecastConfig, calibration: ForecastCalibrationEvidence | None,
                 calibration_valid: bool, history_notes: Sequence[str]) -> ForecastEvent:
    current = trend.current
    if current >= threshold:
        bounds = (0.0, 0.0)
    else:
        crossings = [(threshold-current)/s for s in (trend.high_slope, trend.low_slope) if s > 0]
        if len(crossings) != 2 or crossings[0] > cfg.max_horizon_days:
            return _event(well, mechanism, ForecastStatus.UNAVAILABLE, current=current,
                threshold=threshold, as_of=as_of, likelihood=_likelihood(current),
                basis="Robust trend did not establish a stable threshold crossing in the configured horizon.",
                method="Theil-Sen median pairwise slope",
                required=("Longer stable time series",),
                limitations=(*history_notes, *trend.limitations,
                             "No event is asserted outside the configured horizon."))
        bounds = (max(0.0, crossings[0]), min(cfg.max_horizon_days, crossings[1]))
    status, probability = _calibrated_fields(mechanism, calibration, calibration_valid)
    return _event(well, mechanism, status, current=current, threshold=threshold,
        bounds=bounds, as_of=as_of, probability=probability,
        likelihood=_likelihood(current), risk_band=None if status is ForecastStatus.CALIBRATED else _risk_band(current),
        basis=f"{trend.points} observations over {trend.span_days:.1f} days; robust slope {trend.slope_per_day:.6g}/day.",
        method="Theil-Sen trend with pairwise-slope scenario bounds",
        assumptions=("The historical trend persists through the forecast window.",
                     "No unrecorded treatment or operating-regime change occurs."),
        required=() if status is ForecastStatus.CALIBRATED else ("Holdout-validated event calibration artifact",),
        limitations=(*history_notes, *trend.limitations,
                     "Trend crossing is a scenario, not a failure date."),
        production_ready=status is ForecastStatus.CALIBRATED, actionable=True,
        evidence=TemporalEvidence(trend.points, trend.span_days, trend.consistency, "good",
                                  trend.regime_id, True),
        calibration_evidence=calibration)


def _corrosion_event(diagnosis: DiagnosisResult, case: WellCase,
                     snapshots: Sequence[ForecastSnapshot], as_of: datetime | None,
                     cfg: ForecastConfig, integrity: CorrosionIntegrityInput | None,
                     calibration: ForecastCalibrationEvidence | None, calibration_valid: bool,
                     calibration_reason: str, history_notes: Sequence[str]) -> ForecastEvent:
    mechanism = ForecastMechanism.CORROSION
    current = _severity(diagnosis, mechanism)
    rate = diagnosis.corrosion.get("rate_mm_yr")
    trend, trend_reason = _trend_for(snapshots, _METRICS[mechanism], cfg, increasing=True)
    if integrity is None:
        return _event(case.name, mechanism, ForecastStatus.UNAVAILABLE, current=current,
            threshold=None, as_of=as_of, likelihood=_likelihood(current),
            basis="Instantaneous modelled corrosion rate is available, but wall reserve is not.",
            method="not calculated", required=("Measured current wall thickness",
                "Minimum allowable wall thickness"),
            limitations=(*history_notes, "A corrosion rate alone cannot determine threshold date."))
    reserve = integrity.available_wall_loss_mm
    if trend is not None:
        threshold = trend.current + reserve
        return _trend_event(case.name, mechanism, current, threshold, trend, as_of, cfg,
                            calibration, calibration_valid, history_notes)
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        return _event(case.name, mechanism, ForecastStatus.UNAVAILABLE, current=current,
            threshold=reserve, as_of=as_of, likelihood=_likelihood(current),
            basis="No positive finite corrosion rate.", method="not calculated",
            required=("Measured wall-loss history",), limitations=(*history_notes, trend_reason))
    days = reserve / float(rate) * 365.25
    fraction = cfg.corrosion_rate_uncertainty_fraction
    low = days / (1.0 + fraction)
    high = days / max(1.0 - fraction, 1e-9)
    if low > cfg.max_horizon_days:
        return _event(case.name, mechanism, ForecastStatus.UNAVAILABLE, current=current,
            threshold=reserve, as_of=as_of, likelihood=_likelihood(current),
            basis="Wall-loss scenario does not cross the threshold in the configured horizon.",
            method="constant-rate wall-loss screening", required=("Repeated wall-thickness measurements",),
            limitations=(*history_notes, trend_reason, "No event is asserted beyond the horizon."))
    bounds = (max(0.0, low), min(cfg.max_horizon_days, high))
    status, probability = _calibrated_fields(mechanism, calibration, calibration_valid)
    return _event(case.name, mechanism, status, current=current, threshold=reserve,
        bounds=bounds, as_of=as_of, probability=probability, likelihood=_likelihood(current),
        risk_band=None if status is ForecastStatus.CALIBRATED else _risk_band(current),
        basis=f"Measured wall reserve {reserve:.3g} mm and modelled rate {float(rate):.3g} mm/year.",
        method="constant-rate wall-loss scenario",
        assumptions=(f"Corrosion rate remains constant; scenario uncertainty ±{fraction:.0%}.",
                     "No localized pitting allowance is inferred."),
        required=() if status is ForecastStatus.CALIBRATED else ("Repeated wall-thickness measurements",
                     "Holdout-validated event calibration artifact"),
        limitations=(*history_notes, trend_reason,
                     "Screening rate is not a remaining-life certification.",
                     calibration_reason if not calibration_valid else ""),
        production_ready=status is ForecastStatus.CALIBRATED, actionable=True,
        calibration_evidence=calibration)


def _production_event(diagnosis: DiagnosisResult, case: WellCase,
                      snapshots: Sequence[ForecastSnapshot], as_of: datetime | None,
                      cfg: ForecastConfig, calibration: ForecastCalibrationEvidence | None,
                      calibration_valid: bool, calibration_reason: str,
                      history_notes: Sequence[str]) -> ForecastEvent:
    mechanism = ForecastMechanism.PRODUCTION_DECLINE
    values = _metric_points(snapshots, _METRICS[mechanism])
    baseline = _median([value for _, value in values[:max(1, len(values)//3)]]) if values else None
    trend, reason = _trend_for(snapshots, _METRICS[mechanism], cfg, increasing=False)
    if trend is None or baseline is None:
        return _event(case.name, mechanism, ForecastStatus.UNAVAILABLE,
            current=values[-1][1] if values else case.rate.q_oil_m3d,
            threshold=None if baseline is None else baseline*(1-cfg.production_decline_fraction),
            as_of=as_of, likelihood=LikelihoodCategory.NOT_ASSESSED,
            basis="Production decline requires a quality-controlled rate history.",
            method="not calculated", required=(f"At least {cfg.min_history_points} oil-rate observations",
                f"History span of at least {cfg.min_history_span_days:g} days"),
            limitations=(*history_notes, reason, calibration_reason if not calibration_valid else ""))
    threshold = baseline * (1.0-cfg.production_decline_fraction)
    current = trend.current
    if current <= threshold:
        bounds = (0.0, 0.0)
    else:
        slopes = [trend.low_slope, trend.high_slope]
        crossings = [(threshold-current)/s for s in slopes if s < 0]
        if len(crossings) != 2:
            return _event(case.name, mechanism, ForecastStatus.UNAVAILABLE, current=current,
                threshold=threshold, as_of=as_of, likelihood=LikelihoodCategory.NOT_ASSESSED,
                basis="Decline direction is not robust across scenario slopes.",
                method="Theil-Sen median pairwise slope", required=("Longer stable oil-rate history",),
                limitations=(*history_notes, *trend.limitations))
        bounds = tuple(sorted(crossings))
        if bounds[0] > cfg.max_horizon_days:
            return _event(case.name, mechanism, ForecastStatus.UNAVAILABLE, current=current,
                threshold=threshold, as_of=as_of, likelihood=LikelihoodCategory.NOT_ASSESSED,
                basis="No decline-threshold crossing in the configured horizon.",
                method="Theil-Sen median pairwise slope", required=(),
                limitations=(*history_notes, "No event is asserted beyond the horizon."))
        bounds = (max(0.0, bounds[0]), min(cfg.max_horizon_days, bounds[1]))
    status, probability = _calibrated_fields(mechanism, calibration, calibration_valid)
    return _event(case.name, mechanism, status, current=current, threshold=threshold,
        bounds=bounds, as_of=as_of, probability=probability,
        likelihood=LikelihoodCategory.NOT_ASSESSED,
        risk_band=None if status is ForecastStatus.CALIBRATED else (0.0, 1.0),
        basis=f"{trend.points} oil-rate observations over {trend.span_days:.1f} days; baseline {baseline:.3g} m3/day.",
        method="Theil-Sen oil-rate trend with pairwise-slope scenario bounds",
        assumptions=("No choke, pump, workover, allocation, or downtime regime change is hidden in the series.",),
        required=() if status is ForecastStatus.CALIBRATED else ("Holdout-validated decline calibration artifact",),
        limitations=(*history_notes, *trend.limitations,
                     "Trend continuation is a scenario; it is not a guaranteed decline date."),
        production_ready=status is ForecastStatus.CALIBRATED, actionable=True,
        evidence=TemporalEvidence(trend.points, trend.span_days, trend.consistency, "good",
                                  trend.regime_id, True),
        calibration_evidence=calibration)


def _trend_for(snapshots: Sequence[ForecastSnapshot], metric: str,
               cfg: ForecastConfig, *, increasing: bool) -> tuple[_Trend | None, str]:
    metric_rows = [row for row in snapshots if row.quality == "good" and getattr(row, metric) is not None]
    regimes = {row.regime_id for row in metric_rows}
    if len(regimes) > 1:
        return None, f"{metric} crosses incompatible operating regimes; trend blocked."
    points = _metric_points(snapshots, metric)
    if len(points) < cfg.min_history_points:
        return None, f"Insufficient valid {metric} history ({len(points)}/{cfg.min_history_points} points)."
    span = (points[-1][0]-points[0][0]).total_seconds()/86400.0
    if span < cfg.min_history_span_days:
        return None, f"{metric} history span {span:.1f} days is below {cfg.min_history_span_days:g} days."
    slopes = []
    for i, (left_time, left_value) in enumerate(points):
        for right_time, right_value in points[i+1:]:
            days = (right_time-left_time).total_seconds()/86400.0
            if days > 0:
                slopes.append((right_value-left_value)/days)
    if not slopes:
        return None, f"No distinct timestamps for {metric}."
    slope = _median(slopes)
    directional = [value > 0 if increasing else value < 0 for value in slopes]
    consistency = sum(directional)/len(directional)
    if (increasing and slope <= 0) or (not increasing and slope >= 0):
        return None, f"{metric} trend is flat or improving."
    if consistency < cfg.minimum_trend_consistency:
        return None, f"{metric} trend consistency {consistency:.0%} is below {cfg.minimum_trend_consistency:.0%}."
    low = _quantile(slopes, cfg.scenario_slope_quantiles[0])
    high = _quantile(slopes, cfg.scenario_slope_quantiles[1])
    if increasing and low <= 0:
        return None, f"{metric} lower scenario slope is non-positive."
    if not increasing and high >= 0:
        return None, f"{metric} upper scenario slope is non-negative."
    notes = ()
    if any(snapshot.quality == "questionable" and getattr(snapshot, metric) is not None
           for snapshot in snapshots):
        notes = ("Questionable-quality observations were excluded.",)
    regime_id = next(iter(regimes)) if regimes else None
    return _Trend(points[-1][1], slope, low, high, span, len(points), consistency,
                  regime_id, notes), ""


def _metric_points(snapshots: Sequence[ForecastSnapshot], metric: str) -> list[tuple[datetime, float]]:
    grouped: dict[datetime, list[float]] = {}
    for snapshot in snapshots:
        value = getattr(snapshot, metric)
        if snapshot.quality != "good" or value is None:
            continue
        grouped.setdefault(snapshot.timestamp.astimezone(timezone.utc), []).append(float(value))
    return [(timestamp, _median(values)) for timestamp, values in sorted(grouped.items())]


def _validated_history(history: ForecastHistory | None, well: str,
                       as_of: datetime | None, cfg: ForecastConfig) -> tuple[list[ForecastSnapshot], tuple[str, ...]]:
    if history is None:
        return [], ()
    valid: list[ForecastSnapshot] = []
    notes: list[str] = []
    errors: list[str] = []
    for snapshot in history.snapshots:
        if snapshot.well != well:
            errors.append(f"history contains well {snapshot.well!r}, expected {well!r}")
            continue
        timestamp = snapshot.timestamp.astimezone(timezone.utc)
        if as_of is not None and timestamp > as_of:
            errors.append(f"history snapshot {timestamp.isoformat()} is after as_of")
            continue
        valid.append(snapshot)
    event_ids: set[str] = set()
    for event in history.events:
        if event.well != well:
            errors.append(f"observed events contain well {event.well!r}, expected {well!r}")
        if as_of is not None and event.timestamp.astimezone(timezone.utc) > as_of:
            errors.append(f"observed event {event.event_id!r} is after as_of")
        if event.event_id in event_ids:
            errors.append(f"duplicate observed event_id {event.event_id!r}")
        event_ids.add(event.event_id)
    duplicate_count = len(valid)-len({item.timestamp.astimezone(timezone.utc) for item in valid})
    if duplicate_count:
        notes.append(f"{duplicate_count} duplicate timestamp row(s) aggregated by median per metric.")
    if errors and cfg.strict_history:
        raise ValueError("Invalid forecast history: " + "; ".join(errors))
    notes.extend(errors)
    valid.sort(key=lambda item: item.timestamp.astimezone(timezone.utc))
    return valid, tuple(notes)


def _valid_calibration(evidence: ForecastCalibrationEvidence | None,
                       history: ForecastHistory | None,
                       cfg: ForecastConfig) -> tuple[bool, str]:
    if evidence is None:
        return False, "No forecast calibration evidence supplied."
    reasons = []
    if history is None or not history.dataset_id:
        reasons.append("history dataset_id missing")
    elif history.dataset_id != evidence.dataset_id:
        reasons.append("calibration dataset_id does not match history dataset_id")
    if evidence.validation_status != "holdout-validated":
        reasons.append("validation_status is not holdout-validated")
    if evidence.synthetic:
        reasons.append("synthetic evidence cannot gate calibrated forecasts")
    if evidence.holdout_n < cfg.min_calibration_holdout_n:
        reasons.append(f"holdout_n below {cfg.min_calibration_holdout_n}")
    if not math.isfinite(evidence.brier_score) or not 0 <= evidence.brier_score <= cfg.max_calibration_brier:
        reasons.append(f"Brier score unavailable or above {cfg.max_calibration_brier}")
    return not reasons, "; ".join(reasons)


def _calibrated_fields(mechanism: ForecastMechanism,
                       evidence: ForecastCalibrationEvidence | None,
                       valid: bool) -> tuple[ForecastStatus, float | None]:
    if valid and evidence is not None and mechanism in evidence.mechanisms:
        probability = evidence.probabilities.get(mechanism)
        endpoint = evidence.endpoints.get(mechanism)
        if probability is not None and (not endpoint or evidence.probability_horizon_days is None):
            return ForecastStatus.SCREENING, None
        return ForecastStatus.CALIBRATED, probability
    return ForecastStatus.SCREENING, None


def _event(well: str, mechanism: ForecastMechanism, status: ForecastStatus, *,
           current: float | None, threshold: float | None, as_of: datetime | None,
           basis: str, method: str, assumptions: Iterable[str] = (),
           required: Iterable[str] = (), limitations: Iterable[str] = (),
           bounds: tuple[float, float] | None = None, probability: float | None = None,
           risk_band: tuple[float, float] | None = None,
           likelihood: LikelihoodCategory = LikelihoodCategory.NOT_ASSESSED,
           production_ready: bool = False, actionable: bool = False,
           evidence: TemporalEvidence | None = None,
           calibration_evidence: ForecastCalibrationEvidence | None = None) -> ForecastEvent:
    start, end = bounds if bounds is not None else (None, None)
    start_date = end_date = None
    if as_of is not None and start is not None:
        start_date = (as_of + timedelta(days=math.floor(start))).date()
        end_date = (as_of + timedelta(days=math.ceil(end))).date()
    clean_limitations = tuple(item for item in limitations if item)
    stable_id = str(uuid5(NAMESPACE_URL, f"galit:forecast:v1:{well}:{mechanism.value}"))
    endpoint = calibration_evidence.endpoints.get(mechanism) if calibration_evidence else None
    probability_horizon = calibration_evidence.probability_horizon_days if calibration_evidence else None
    matched = bool(probability is not None and endpoint and probability_horizon is not None)
    calibration_meta = ForecastCalibrationMetadata(
        artifact_id=calibration_evidence.artifact_id if calibration_evidence else None,
        dataset_id=calibration_evidence.dataset_id if calibration_evidence else None,
        validation_status=calibration_evidence.validation_status if calibration_evidence else "not_supplied",
        holdout_n=calibration_evidence.holdout_n if calibration_evidence else None,
        brier_score=calibration_evidence.brier_score if calibration_evidence else None,
        endpoint=endpoint, horizon_days=probability_horizon, matched=matched,
    )
    return ForecastEvent(stable_id, well, mechanism, _TITLES[mechanism], status,
                         start, end, start_date, end_date, probability, risk_band,
                         likelihood, current, threshold, basis, method,
                         tuple(assumptions), tuple(required), clean_limitations,
                         production_ready and matched, actionable,
                         endpoint if matched else None, probability_horizon if matched else None,
                         evidence or TemporalEvidence(), calibration_meta)


def _severity(diagnosis: DiagnosisResult, mechanism: ForecastMechanism) -> float:
    value = float(diagnosis.severity.get(mechanism.value, 0.0))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"diagnosis severity {mechanism.value} must be finite and within [0, 1]")
    return value


def _likelihood(value: float | None) -> LikelihoodCategory:
    if value is None:
        return LikelihoodCategory.NOT_ASSESSED
    if value < 0.35:
        return LikelihoodCategory.LOW
    if value < 0.60:
        return LikelihoodCategory.MEDIUM
    return LikelihoodCategory.HIGH


def _risk_band(value: float | None) -> tuple[float, float] | None:
    if value is None:
        return None
    return max(0.0, value-0.10), min(1.0, value+0.10)


def _sort_key(event: ForecastEvent) -> tuple[float, str, str]:
    horizon = event.horizon_start_days if event.horizon_start_days is not None else math.inf
    return horizon, event.mechanism.value, event.id


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered)//2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle-1]+ordered[middle])/2.0


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q*(len(ordered)-1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position-low
    return ordered[low]*(1-fraction)+ordered[high]*fraction


__all__ = [
    "CorrosionIntegrityInput", "ForecastCalibrationEvidence", "ForecastConfig",
    "ForecastCalibrationMetadata", "ForecastEvent", "ForecastHistory", "ForecastMechanism", "ForecastSnapshot",
    "ForecastStatus", "ForecastSummary", "LikelihoodCategory",
    "ObservedForecastEvent", "TemporalEvidence", "WellForecast", "forecast_well",
]
