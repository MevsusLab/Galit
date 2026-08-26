"""Auditable, immutable what-if comparisons built on GALIT's existing engines.

The module changes only inputs that the diagnostic model actually consumes.
Operational actions (dosage, wash, operating mode) have no invented physical
response: their effects must be supplied explicitly through ``EffectOverride``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import math
from typing import Any

from .forecast import WellForecast, forecast_well
from .integrated import DiagnosisResult, WellCase, diagnose
from .risk_economics import RiskEconomicsInput, RiskEconomicsResult, calculate_risk_economics


class ScenarioStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class EffectOverride:
    """Explicit, user-sourced effects for actions absent from diagnostic physics."""

    inhibitor_efficiency: float | None = None
    oil_rate_delta_m3_day: float | None = None
    oil_rate_relative_change: float | None = None
    water_rate_delta_m3_day: float | None = None
    water_rate_relative_change: float | None = None
    source: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("inhibitor_efficiency",):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name in ("oil_rate_delta_m3_day", "oil_rate_relative_change",
                     "water_rate_delta_m3_day", "water_rate_relative_change"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if any(getattr(self, name) is not None for name in (
            "inhibitor_efficiency", "oil_rate_delta_m3_day", "oil_rate_relative_change",
            "water_rate_delta_m3_day", "water_rate_relative_change",
        )) and not (self.source and self.source.strip()):
            raise ValueError("effect_override.source is required when an effect is supplied")


@dataclass(frozen=True)
class ScenarioChanges:
    """Requested changes; relative values are fractions (e.g. -0.10 means -10%)."""

    oil_rate_delta_m3_day: float | None = None
    oil_rate_relative_change: float | None = None
    water_rate_delta_m3_day: float | None = None
    water_rate_relative_change: float | None = None
    wellhead_pressure_delta_pa: float | None = None
    wellhead_pressure_relative_change: float | None = None
    surface_temperature_delta_c: float | None = None
    inhibitor_dosage_delta_mg_l: float | None = None
    wash_treatment: bool = False
    operating_mode: str | None = None
    effect_override: EffectOverride | None = None

    def __post_init__(self) -> None:
        pairs = (
            ("oil_rate_delta_m3_day", "oil_rate_relative_change"),
            ("water_rate_delta_m3_day", "water_rate_relative_change"),
            ("wellhead_pressure_delta_pa", "wellhead_pressure_relative_change"),
        )
        for absolute, relative in pairs:
            if getattr(self, absolute) is not None and getattr(self, relative) is not None:
                raise ValueError(f"{absolute} and {relative} are mutually exclusive")
        for name in ("oil_rate_delta_m3_day", "oil_rate_relative_change",
                     "water_rate_delta_m3_day", "water_rate_relative_change",
                     "wellhead_pressure_delta_pa", "wellhead_pressure_relative_change",
                     "surface_temperature_delta_c", "inhibitor_dosage_delta_mg_l"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.inhibitor_dosage_delta_mg_l is not None and self.inhibitor_dosage_delta_mg_l < 0:
            raise ValueError("inhibitor_dosage_delta_mg_l must be non-negative")
        if self.operating_mode is not None and not self.operating_mode.strip():
            raise ValueError("operating_mode must be non-empty when supplied")


@dataclass(frozen=True)
class ScenarioEconomics:
    horizon_days: float
    event_probability: float | None = None
    treatment_efficiency: float | None = None
    event_downtime_days: float = 0.0
    treatment_downtime_days: float = 0.0
    product_price_per_m3: float | None = None
    operating_loss_per_day: float | None = None
    treatment_cost: float | None = None
    currency: str | None = None
    production_loss_fraction: float = 1.0
    probability_source: str = "explicit_input"

    def __post_init__(self) -> None:
        if not math.isfinite(self.horizon_days) or self.horizon_days <= 0:
            raise ValueError("horizon_days must be finite and positive")
        if self.treatment_efficiency is not None and (
            not math.isfinite(self.treatment_efficiency) or not 0 <= self.treatment_efficiency <= 1
        ):
            raise ValueError("treatment_efficiency must be finite and within [0, 1]")


@dataclass(frozen=True)
class ScenarioSnapshot:
    parameters: dict[str, Any]
    integrated_risk: float
    severity: dict[str, float]
    dominant: str
    forecast_oil_rate_m3_day: float
    forecast_status: str


@dataclass(frozen=True)
class ScenarioComparison:
    status: ScenarioStatus
    well: str
    before: ScenarioSnapshot
    after: ScenarioSnapshot
    delta: dict[str, Any]
    economics: RiskEconomicsResult | None
    applied_changes: tuple[dict[str, Any], ...]
    formulas: dict[str, str]
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    audit_trail: dict[str, Any]


def _changed(base: float, absolute: float | None, relative: float | None, name: str) -> float:
    value = base + absolute if absolute is not None else base * (1 + relative) if relative is not None else base
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"resulting {name} must be finite and non-negative")
    return value


def _parameters(case: WellCase) -> dict[str, Any]:
    return {
        "q_oil_m3_day": case.rate.q_oil_m3d,
        "q_water_m3_day": case.rate.q_water_m3d,
        "p_wellhead_pa": case.p_wellhead_pa,
        "t_surface_c": case.thermal.t_surface_c,
        "inhibitor_efficiency_fraction": case.inhibitor_efficiency,
        "lift_type": case.lift_type,
    }


def _snapshot(case: WellCase, result: DiagnosisResult, forecast: WellForecast) -> ScenarioSnapshot:
    production = next((e for e in forecast.events if e.mechanism.value == "production_decline"), None)
    return ScenarioSnapshot(
        parameters=_parameters(case), integrated_risk=result.integrated_risk,
        severity=dict(result.severity), dominant=result.dominant,
        forecast_oil_rate_m3_day=case.rate.q_oil_m3d,
        forecast_status=production.status.value if production else "unavailable",
    )


def compare_scenario(case: WellCase, changes: ScenarioChanges,
                     economics: ScenarioEconomics | None = None) -> ScenarioComparison:
    """Compare diagnostics without mutating ``case`` and preserve a full audit trail."""
    warnings: list[str] = []
    missing: list[str] = []
    assumptions: list[str] = [
        "Risk values are GALIT screening severity scores, not calibrated probabilities.",
        "Forecast oil rate is the scenario operating input unless temporal field history is supplied elsewhere.",
        "The comparison is associative, not a causal guarantee.",
    ]
    applied: list[dict[str, Any]] = []

    oil = _changed(case.rate.q_oil_m3d, changes.oil_rate_delta_m3_day,
                   changes.oil_rate_relative_change, "oil rate")
    water = _changed(case.rate.q_water_m3d, changes.water_rate_delta_m3_day,
                     changes.water_rate_relative_change, "water rate")
    pressure = _changed(case.p_wellhead_pa, changes.wellhead_pressure_delta_pa,
                        changes.wellhead_pressure_relative_change, "wellhead pressure")
    temperature = case.thermal.t_surface_c + (changes.surface_temperature_delta_c or 0.0)
    if not math.isfinite(temperature) or not -100 <= temperature <= 200:
        raise ValueError("resulting surface temperature must be finite and within [-100, 200] C")

    for name, value, unit in (
        ("oil_rate", oil, "m3/day"), ("water_rate", water, "m3/day"),
        ("wellhead_pressure", pressure, "Pa"), ("surface_temperature", temperature, "degC"),
    ):
        if value != _parameters(case).get({"oil_rate":"q_oil_m3_day", "water_rate":"q_water_m3_day",
            "wellhead_pressure":"p_wellhead_pa", "surface_temperature":"t_surface_c"}[name]):
            applied.append({"field": name, "result": value, "unit": unit, "source": "direct_change"})

    override = changes.effect_override
    if override:
        oil = _changed(oil, override.oil_rate_delta_m3_day, override.oil_rate_relative_change, "oil rate")
        water = _changed(water, override.water_rate_delta_m3_day, override.water_rate_relative_change, "water rate")
        assumptions.extend(override.assumptions)
    inhibitor_efficiency = (override.inhibitor_efficiency if override and
                            override.inhibitor_efficiency is not None else case.inhibitor_efficiency)

    operational = []
    if changes.inhibitor_dosage_delta_mg_l is not None:
        operational.append("inhibitor dosage")
        applied.append({"field": "inhibitor_dosage_delta", "result": changes.inhibitor_dosage_delta_mg_l,
                        "unit": "mg/L", "source": "requested_action"})
        if not override or override.inhibitor_efficiency is None:
            missing.append("changes.effect_override.inhibitor_efficiency")
            warnings.append("Inhibitor dosage is not a diagnostic input; no risk effect was applied without an explicit efficiency override.")
    if changes.wash_treatment:
        operational.append("wash treatment")
        applied.append({"field": "wash_treatment", "result": True, "unit": "boolean", "source": "requested_action"})
        if not override:
            missing.append("changes.effect_override")
            warnings.append("Wash treatment has no universal physical effect in GALIT; no hidden coefficient was applied.")
    lift = case.lift_type
    if changes.operating_mode is not None:
        operational.append("operating mode")
        lift = changes.operating_mode.strip()
        applied.append({"field": "operating_mode", "result": lift, "unit": "category", "source": "requested_action"})
        if not override:
            missing.append("changes.effect_override")
            warnings.append("Operating mode is not consumed by diagnostic physics; only the label changed without an explicit effect override.")

    after_case = replace(case, rate=replace(case.rate, q_oil_m3d=oil, q_water_m3d=water),
                         thermal=replace(case.thermal, t_surface_c=temperature),
                         p_wellhead_pa=pressure, inhibitor_efficiency=inhibitor_efficiency,
                         lift_type=lift)
    before_d = diagnose(case)
    after_d = diagnose(after_case)
    before_f = forecast_well(before_d, case)
    after_f = forecast_well(after_d, after_case)

    econ_result = None
    if economics is not None:
        efficiency = economics.treatment_efficiency
        if efficiency is None:
            missing.append("economics.treatment_efficiency")
            efficiency = 0.0
            warnings.append("Avoided damage and net effect require explicit treatment_efficiency; zero is used only to keep partial arithmetic auditable.")
        econ_result = calculate_risk_economics(RiskEconomicsInput(
            event_probability=economics.event_probability, horizon_days=economics.horizon_days,
            treatment_efficiency=efficiency, event_downtime_days=economics.event_downtime_days,
            treatment_downtime_days=economics.treatment_downtime_days,
            oil_rate_m3_day=case.rate.q_oil_m3d,
            product_price_per_m3=economics.product_price_per_m3,
            operating_loss_per_day=economics.operating_loss_per_day,
            treatment_cost=economics.treatment_cost, currency=economics.currency,
            production_loss_fraction=economics.production_loss_fraction,
            probability_source=economics.probability_source,
        ))
        missing.extend(econ_result.missing_inputs)
    else:
        missing.append("economics")
        warnings.append("Economics unavailable: no explicit rates, probability, effectiveness, cost and currency were supplied.")

    missing = list(dict.fromkeys(missing))
    status = ScenarioStatus.AVAILABLE if not missing else ScenarioStatus.PARTIAL
    if not applied and economics is None:
        status = ScenarioStatus.UNAVAILABLE
    before = _snapshot(case, before_d, before_f)
    after = _snapshot(after_case, after_d, after_f)
    delta = {
        "integrated_risk": after.integrated_risk - before.integrated_risk,
        "severity": {key: after.severity[key] - before.severity[key] for key in before.severity},
        "forecast_oil_rate_m3_day": after.forecast_oil_rate_m3_day - before.forecast_oil_rate_m3_day,
    }
    formulas = {
        "relative_change": "result = baseline × (1 + relative_change_fraction)",
        "absolute_change": "result = baseline + delta_in_declared_unit",
        "risk_delta": "after screening severity − before screening severity",
        "production_delta": "after scenario oil-rate input − baseline oil-rate input",
    }
    if econ_result:
        formulas.update(econ_result.formulas)
    return ScenarioComparison(
        status=status, well=case.name, before=before, after=after, delta=delta,
        economics=econ_result, applied_changes=tuple(applied), formulas=formulas,
        assumptions=tuple(assumptions), warnings=tuple(warnings),
        missing_inputs=tuple(missing), audit_trail={
            "baseline_unchanged": _parameters(case), "resulting_parameters": _parameters(after_case),
            "requested_changes": asdict(changes), "operational_actions": operational,
        },
    )
