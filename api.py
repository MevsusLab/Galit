"""REST API ГАЛИТ: обёртка расчётного ядра для корпоративных дашбордов.

Контракт:
    POST /api/v1/diagnose   -- диагностика одной скважины (JSON -> JSON)
    GET  /api/v1/health     -- проверка живости сервиса

Запуск для локальной отладки:
    python api.py

Продакшен (несколько воркеров, за реверс-прокси):
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

Спецификация OpenAPI и Swagger UI доступны на /docs.

Модели запроса -- строгое зеркало WellCase из galit/integrated.py:
поля вложенных групп соответствуют полям дата-классов ядра, остальные
параметры дата-классов (roughness_m, k_earth, pour_point_c, ...)
остаются со значениями по умолчанию.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from datetime import date, datetime
import os
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator

import galit
from galit import (
    DEFAULT_MASTER_PLAN_POLICY,
    DataProvenance,
    DataQualityError,
    DiagnosedWell,
    FluidProperties,
    ProductionRate,
    ThermalParams,
    UncertaintyConfig,
    WaterAnalysis,
    WaxProperties,
    WellCase,
    WellGeometry,
    diagnose,
    generate_master_plan,
)
from galit.calibration import ParameterSet
from galit.scale import MW as KNOWN_IONS


def _load_server_calibration():
    """Load only the operator-selected startup artifact; requests never supply paths."""
    configured = os.environ.get("GALIT_CALIBRATION_ARTIFACT", "").strip()
    if not configured:
        return None
    path = Path(configured).resolve(strict=True)
    allowed_root = Path(os.environ.get("GALIT_CALIBRATION_ROOT", "calibration-artifacts")).resolve()
    if path != allowed_root and allowed_root not in path.parents:
        raise RuntimeError(f"Calibration artifact must be inside allow-listed root {allowed_root}")
    return ParameterSet.load(path)


SERVER_PARAMETER_SET = _load_server_calibration()
SERVER_RUNTIME_CALIBRATION = SERVER_PARAMETER_SET.to_runtime() if SERVER_PARAMETER_SET else None


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be positive")
    return value


def parse_cors_origins(raw: str | None) -> list[str]:
    """Parse an explicit origin allow-list; empty means deny cross-origin requests."""
    origins = [item.strip().rstrip("/") for item in (raw or "").split(",") if item.strip()]
    if "*" in origins:
        raise RuntimeError("GALIT_CORS_ORIGINS wildcard is forbidden")
    if any(not origin.startswith(("http://", "https://")) for origin in origins):
        raise RuntimeError("GALIT_CORS_ORIGINS entries must be absolute http(s) origins")
    return list(dict.fromkeys(origins))


MAX_BATCH_SIZE = _positive_int_env("GALIT_MAX_BATCH_SIZE", 25)
CORS_ORIGINS = parse_cors_origins(os.environ.get("GALIT_CORS_ORIGINS"))
AUDIT_LOGGER = logging.getLogger("galit.audit")
EVIDENCE_LABELS = {
    "model_validation": "not validated on independent field data",
    "allowed_use": "screening and decision-support only",
    "api_positioning": "integration prototype; not production-ready",
}
TREATMENT_STORAGE_PATH = Path(os.environ.get("GALIT_TREATMENT_STORAGE", "data/treatments.json"))
TREATMENTS = galit.TreatmentRepository(TREATMENT_STORAGE_PATH)
PASSPORT_STORAGE_PATH = Path(os.environ.get("GALIT_PASSPORT_STORE", "data/well_passports.json"))
PASSPORTS = galit.PassportRepository(PASSPORT_STORAGE_PATH)


# --------------------------------------------------------------------------
# Модели запроса (зеркало WellCase)
# --------------------------------------------------------------------------

class GeometryIn(BaseModel):
    """Конструкция скважины."""

    depth_m: float = Field(gt=0.0, description="Глубина по стволу до забоя, м")
    tubing_id_m: float = Field(gt=0.0, description="Внутренний диаметр НКТ, м")
    inclination_deg: float = Field(
        default=0.0, ge=0.0, le=90.0,
        description="Средний угол от вертикали, град",
    )


class RateIn(BaseModel):
    """Дебиты в поверхностных условиях, м3/сут и м3/м3."""

    q_oil_m3d: float = Field(ge=0.0, description="Дебит нефти, м3/сут")
    q_water_m3d: float = Field(ge=0.0, description="Дебит воды, м3/сут")
    gor_m3m3: float = Field(ge=0.0, description="Газовый фактор, м3/м3")


class FluidIn(BaseModel):
    """Свойства флюидов (относительные плотности, минерализация)."""

    gamma_oil: float = Field(gt=0.0, description="Отн. плотность нефти по воде")
    gamma_gas: float = Field(gt=0.0, description="Отн. плотность газа по воздуху")
    salinity_ppm: float = Field(ge=0.0, description="Минерализация воды, мг/л")


class ThermalIn(BaseModel):
    """Теплофизика для модели Ramey."""

    t_surface_c: float = Field(description="Температура пород у поверхности, C")
    geothermal_grad: float = Field(description="Геотермический градиент, К/м")
    u_to: float = Field(gt=0.0, description="Коэф. теплопередачи, Вт/(м2*К)")
    production_days: float = Field(
        gt=0.0, description="Непрерывная наработка скважины, сут",
    )


class WaterIn(BaseModel):
    """Химанализ пластовой воды: ионы в мг/л, C, Па."""

    ions_mg_l: dict[str, Annotated[float, Field(ge=0.0)]] = Field(
        description="Концентрации ионов, мг/л: Na, Cl, Ca, Mg, K, HCO3, SO4, ...",
    )
    ph: float = Field(ge=0.0, le=14.0)
    t_c: float = Field(description="Температура отбора пробы, C")
    p_pa: float = Field(ge=0.0, description="Давление отбора пробы, Па")

    @field_validator("ions_mg_l")
    @classmethod
    def _known_ions_only(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(KNOWN_IONS))
        if unknown:
            raise ValueError(
                f"неизвестные ионы: {', '.join(unknown)}; "
                f"поддерживаются: {', '.join(sorted(KNOWN_IONS))}"
            )
        return value


class WaxIn(BaseModel):
    """Парафинистость нефти (лабораторный замер)."""

    wat_stock_tank_c: float = Field(
        description="WAT дегазированной нефти при атмосферном давлении, C",
    )
    wax_content_pct: float = Field(
        default=5.0, ge=0.0, le=100.0, description="Содержание парафина, % масс.",
    )


class WellCaseIn(BaseModel):
    """Полное описание скважины для расчёта (JSON-зеркало WellCase)."""

    name: str = Field(min_length=1, description="Название скважины")
    geometry: GeometryIn
    rate: RateIn
    fluid: FluidIn
    thermal: ThermalIn
    water: WaterIn
    wax: WaxIn
    co2_mol_frac: float = Field(
        default=0.02, ge=0.0, le=1.0, description="Доля CO2 в попутном газе",
    )
    inhibitor_efficiency: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0 = без ингибитора, 0.9 = 90 % защиты",
    )
    lift_type: str = Field(
        default="ЭЦН", description="Способ эксплуатации: ЭЦН | ШГН | фонтан",
    )
    p_wellhead_pa: float = Field(
        default=1.2e6, gt=0.0, description="Буферное давление, Па",
    )


class ForecastSnapshotIn(BaseModel):
    well: str = Field(min_length=1)
    timestamp: datetime
    wax_severity: float | None = Field(default=None, ge=0, le=1)
    halite_severity: float | None = Field(default=None, ge=0, le=1)
    calcite_severity: float | None = Field(default=None, ge=0, le=1)
    corrosion_wall_loss_mm: float | None = Field(default=None, ge=0)
    oil_rate_m3_day: float | None = Field(default=None, ge=0)
    quality: str = Field(default="good", pattern="^(good|questionable|bad)$")
    source: str = Field(default="measured", pattern="^(measured|derived|laboratory)$")
    regime_id: str | None = None

    @field_validator("timestamp")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value


class ObservedForecastEventIn(BaseModel):
    event_id: str = Field(min_length=1)
    well: str = Field(min_length=1)
    timestamp: datetime
    mechanism: galit.ForecastMechanism
    outcome: bool = True
    source: str = Field(default="measured", pattern="^(measured|laboratory)$")

    @field_validator("timestamp")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value


class ForecastConfigIn(BaseModel):
    wax_critical_severity: float = Field(default=.60, ge=0, le=1)
    halite_deposition_severity: float = Field(default=.60, ge=0, le=1)
    calcite_deposition_severity: float = Field(default=.60, ge=0, le=1)
    production_decline_fraction: float = Field(default=.20, ge=0, le=1)
    min_history_points: int = Field(default=4, ge=3)
    min_history_span_days: float = Field(default=21, gt=0)
    max_horizon_days: float = Field(default=365, gt=0, le=3650)
    minimum_trend_consistency: float = Field(default=.70, ge=0, le=1)
    corrosion_rate_uncertainty_fraction: float = Field(default=.30, ge=0, le=1)
    strict_history: bool = True


class CorrosionIntegrityIn(BaseModel):
    current_wall_thickness_mm: float = Field(gt=0)
    minimum_allowable_wall_thickness_mm: float = Field(ge=0)
    measured_at: datetime
    source: str = Field(default="measured", pattern="^(measured|laboratory)$")

    @field_validator("measured_at")
    @classmethod
    def aware_measured_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("measured_at must include a timezone offset")
        return value


class ForecastCalibrationIn(BaseModel):
    artifact_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    validation_status: str
    holdout_n: int = Field(ge=0)
    brier_score: float = Field(ge=0, le=1)
    mechanisms: list[galit.ForecastMechanism]
    probabilities: dict[galit.ForecastMechanism, Annotated[float, Field(ge=0, le=1)]] = Field(default_factory=dict)
    synthetic: bool = False
    probability_horizon_days: int | None = Field(default=None, ge=1)
    endpoints: dict[galit.ForecastMechanism, str] = Field(default_factory=dict)


class ForecastRequest(BaseModel):
    well: WellCaseIn
    as_of: datetime
    history_snapshots: list[ForecastSnapshotIn] = Field(default_factory=list, max_length=10000)
    observed_events: list[ObservedForecastEventIn] = Field(default_factory=list, max_length=10000)
    dataset_id: str | None = None
    config: ForecastConfigIn | None = None
    corrosion_integrity: CorrosionIntegrityIn | None = None
    calibration_evidence: ForecastCalibrationIn | None = None

    @field_validator("as_of")
    @classmethod
    def aware_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone offset")
        return value


class ForecastEventOut(BaseModel):
    id: str
    well: str
    mechanism: str
    title: str
    status: str
    horizon_start_days: float | None
    horizon_end_days: float | None
    horizon_start_date: date | None
    horizon_end_date: date | None
    probability: float | None
    risk_band: tuple[float, float] | None
    likelihood: str
    current_risk: float | None
    threshold: float | None
    basis: str
    method: str
    assumptions: tuple[str, ...]
    required_inputs: tuple[str, ...]
    limitations: tuple[str, ...]
    production_ready: bool
    actionable: bool
    probability_endpoint: str | None
    probability_horizon_days: int | None
    evidence: dict[str, Any]
    calibration: dict[str, Any]


class ForecastResponse(BaseModel):
    well: str
    as_of: datetime
    summary: dict[str, Any]
    events: list[ForecastEventOut]
    methodology: dict[str, str]
    advisory_notice: str


class RiskEconomicsIn(BaseModel):
    """Explicit single-currency inputs; no prices are inferred by the server."""

    event_probability: float | None = Field(default=None, ge=0, le=1)
    horizon_days: float = Field(gt=0, le=3650)
    treatment_efficiency: float = Field(ge=0, le=1)
    event_downtime_days: float = Field(ge=0)
    treatment_downtime_days: float = Field(ge=0)
    oil_rate_m3_day: float | None = Field(default=None, ge=0)
    product_price_per_m3: float | None = Field(default=None, ge=0)
    operating_loss_per_day: float | None = Field(default=None, ge=0)
    treatment_cost: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, min_length=1, max_length=12)
    production_loss_fraction: float = Field(default=1.0, ge=0, le=1)
    probability_source: str = Field(default="explicit_input", min_length=1)


class RiskEconomicsRequest(BaseModel):
    well: WellCaseIn
    economics: RiskEconomicsIn
    use_forecast_probability: bool = False
    as_of: datetime | None = None
    history_snapshots: list[ForecastSnapshotIn] = Field(default_factory=list, max_length=10000)
    dataset_id: str | None = None
    calibration_evidence: ForecastCalibrationIn | None = None

    @field_validator("as_of")
    @classmethod
    def aware_optional_as_of(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("as_of must include a timezone offset")
        return value


class RiskEconomicsResponse(BaseModel):
    well: str
    status: str
    currency: str | None
    data_sufficient: bool
    missing_inputs: tuple[str, ...]
    assumptions: tuple[str, ...]
    formulas: dict[str, str]
    breakdown: dict[str, float | None]
    limitations: tuple[str, ...]
    forecast_link: dict[str, Any]


class EffectOverrideIn(BaseModel):
    inhibitor_efficiency: float | None = Field(default=None, ge=0, le=1)
    oil_rate_delta_m3_day: float | None = None
    oil_rate_relative_change: float | None = None
    water_rate_delta_m3_day: float | None = None
    water_rate_relative_change: float | None = None
    source: str | None = None
    assumptions: tuple[str, ...] = ()


class ScenarioChangesIn(BaseModel):
    oil_rate_delta_m3_day: float | None = None
    oil_rate_relative_change: float | None = None
    water_rate_delta_m3_day: float | None = None
    water_rate_relative_change: float | None = None
    wellhead_pressure_delta_pa: float | None = None
    wellhead_pressure_relative_change: float | None = None
    surface_temperature_delta_c: float | None = None
    inhibitor_dosage_delta_mg_l: float | None = Field(default=None, ge=0)
    wash_treatment: bool = False
    operating_mode: str | None = None
    effect_override: EffectOverrideIn | None = None


class ScenarioEconomicsIn(BaseModel):
    horizon_days: float = Field(gt=0, le=3650)
    event_probability: float | None = Field(default=None, ge=0, le=1)
    treatment_efficiency: float | None = Field(default=None, ge=0, le=1)
    event_downtime_days: float = Field(default=0, ge=0)
    treatment_downtime_days: float = Field(default=0, ge=0)
    product_price_per_m3: float | None = Field(default=None, ge=0)
    operating_loss_per_day: float | None = Field(default=None, ge=0)
    treatment_cost: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, min_length=1, max_length=12)
    production_loss_fraction: float = Field(default=1, ge=0, le=1)
    probability_source: str = Field(default="explicit_input", min_length=1)


class ScenarioCompareRequest(BaseModel):
    well: WellCaseIn
    changes: ScenarioChangesIn
    economics: ScenarioEconomicsIn | None = None


class ScenarioCompareResponse(BaseModel):
    status: str
    well: str
    before: dict[str, Any]
    after: dict[str, Any]
    delta: dict[str, Any]
    economics: dict[str, Any] | None
    applied_changes: tuple[dict[str, Any], ...]
    formulas: dict[str, str]
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    audit_trail: dict[str, Any]


class TreatmentCreateIn(BaseModel):
    well_id: str = Field(min_length=1)
    well_name: str = Field(min_length=1)
    event_at: datetime
    complication_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    reagent_name: str = Field(min_length=1)
    reagent_id: str | None = None
    dosage: float = Field(ge=0)
    dosage_unit: str = Field(min_length=1)
    cost: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    treatment_type: str = Field(min_length=1)
    baseline_risk: float | None = Field(default=None, ge=0, le=1)
    baseline_state: str | None = None
    expected_result: str | None = None
    comment: str | None = None
    source: str = Field(default="manual", min_length=1)
    well_group: str | None = None

    @field_validator("event_at")
    @classmethod
    def aware_event_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_at must include a timezone offset")
        return value


class TreatmentUpdateIn(BaseModel):
    revision: int = Field(ge=1, description="Optimistic concurrency revision from the latest response")
    status: galit.TreatmentStatus | None = None
    well_id: str | None = Field(default=None, min_length=1)
    well_name: str | None = Field(default=None, min_length=1)
    event_at: datetime | None = None
    complication_type: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    reagent_name: str | None = Field(default=None, min_length=1)
    reagent_id: str | None = None
    dosage: float | None = Field(default=None, ge=0)
    dosage_unit: str | None = Field(default=None, min_length=1)
    cost: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    treatment_type: str | None = Field(default=None, min_length=1)
    baseline_risk: float | None = Field(default=None, ge=0, le=1)
    baseline_state: str | None = None
    expected_result: str | None = None
    actual_result: str | None = None
    result_metrics: dict[str, float] | None = None
    success: bool | None = None
    effect_duration_days: float | None = Field(default=None, ge=0)
    recurrence: bool | None = None
    recurrence_date: datetime | None = None
    comment: str | None = None
    source: str | None = Field(default=None, min_length=1)
    well_group: str | None = None

    @field_validator("event_at", "recurrence_date")
    @classmethod
    def aware_treatment_dates(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("treatment dates must include a timezone offset")
        return value


class TreatmentOut(BaseModel):
    id: str
    well_id: str
    well_name: str
    event_at: datetime
    complication_type: str
    description: str
    reagent_name: str
    reagent_id: str | None
    dosage: float
    dosage_unit: str
    cost: float
    currency: str
    treatment_type: str
    status: galit.TreatmentStatus
    baseline_risk: float | None
    baseline_state: str | None
    expected_result: str | None
    actual_result: str | None
    result_metrics: dict[str, float]
    success: bool | None
    effect_duration_days: float | None
    recurrence: bool | None
    recurrence_date: datetime | None
    comment: str | None
    source: str
    well_group: str | None
    revision: int
    archived: bool
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


class PassportEventCreateIn(BaseModel):
    well_id: str = Field(min_length=1, max_length=200)
    well_name: str = Field(min_length=1, max_length=200)
    event_type: galit.PassportEventType
    event_at: datetime
    title: str = Field(min_length=1, max_length=500)
    data: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = Field(default=None, max_length=10000)
    source: str = Field(default="api", min_length=1, max_length=100)

    @field_validator("event_at")
    @classmethod
    def aware_passport_date(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_at must include a timezone offset")
        return value


class PassportEventUpdateIn(BaseModel):
    revision: int = Field(ge=1)
    well_id: str | None = Field(default=None, min_length=1, max_length=200)
    well_name: str | None = Field(default=None, min_length=1, max_length=200)
    event_type: galit.PassportEventType | None = None
    event_at: datetime | None = None
    title: str | None = Field(default=None, min_length=1, max_length=500)
    data: dict[str, Any] | None = None
    notes: str | None = Field(default=None, max_length=10000)
    source: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("event_at")
    @classmethod
    def aware_optional_passport_date(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("event_at must include a timezone offset")
        return value


# --------------------------------------------------------------------------
# Response assembly
# --------------------------------------------------------------------------

LEGACY_FIELDS = {"integrated_risk", "dominant", "wax_onset_m", "recommendation", "warnings"}


def _to_well_case(payload: WellCaseIn) -> WellCase:
    """Map the API model and inferred provenance without accepting client paths."""
    sources: dict[str, str] = {}
    for api_field in ("co2_mol_frac", "inhibitor_efficiency", "p_wellhead_pa"):
        if api_field not in payload.model_fields_set:
            sources[api_field] = "default"
    if "wax_content_pct" not in payload.wax.model_fields_set:
        sources["wax.wax_content_pct"] = "default"
    return WellCase(
        name=payload.name,
        geometry=WellGeometry(**payload.geometry.model_dump()),
        rate=ProductionRate(**payload.rate.model_dump()),
        fluid=FluidProperties(**payload.fluid.model_dump()),
        thermal=ThermalParams(**payload.thermal.model_dump()),
        water=WaterAnalysis(**payload.water.model_dump()),
        wax=WaxProperties(**payload.wax.model_dump()),
        co2_mol_frac=payload.co2_mol_frac,
        inhibitor_efficiency=payload.inhibitor_efficiency,
        lift_type=payload.lift_type,
        p_wellhead_pa=payload.p_wellhead_pa,
        provenance=DataProvenance(sources=sources),
    )


def _quality_dict(quality: Any) -> dict[str, Any]:
    return {
        "grade": quality.grade,
        "completeness": quality.completeness,
        "production_ready": quality.production_ready,
        "missing_fields": quality.missing_fields,
        "defaulted_fields": quality.defaulted_fields,
        "synthetic_fields": quality.synthetic_fields,
        "reasons": quality.reasons,
    }


def _result_dict(result: Any, *, include_metadata: bool, include_profiles: bool,
                 include_uncertainty: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "integrated_risk": result.integrated_risk,
        "dominant": result.dominant,
        "wax_onset_m": result.wax_onset_m,
        "recommendation": result.recommendation,
        "warnings": result.warnings,
    }
    if include_metadata:
        body.update({
            "quality": _quality_dict(result.quality),
            "severity": result.severity,
            "contributions": {
                key: result.severity[key] * result.mechanism_weights[key]
                for key in result.severity
            },
            "applicability_warnings": result.warnings,
            "policy": {"id": result.policy_id, "version": result.policy_version,
                       "weights": result.mechanism_weights},
            "calibration": {"id": result.calibration_id,
                            "version": result.calibration_version,
                            "status": result.calibration_status},
            "evidence_labels": EVIDENCE_LABELS,
        })
    if include_profiles:
        body["profiles"] = {
            "depth_m": result.depths,
            "temperature_c": result.temps,
            "pressure_pa": result.pressures,
            "wat_c": result.wat_profile,
        }
    if include_uncertainty:
        body["uncertainty"] = {
            "method": result.uncertainty.method,
            "samples": result.uncertainty.samples,
            "seed": result.uncertainty.seed,
            "confidence_label": result.uncertainty.confidence_label,
            "integrated_risk": vars(result.uncertainty.integrated_risk),
            "mechanisms": {key: vars(value) for key, value in result.uncertainty.mechanisms.items()},
            "wax_onset_m": vars(result.uncertainty.wax_onset_m),
            "probability_of_deposition": result.uncertainty.probability_of_deposition,
            "warnings": result.uncertainty.warnings,
        }
    return body


async def _calculate(payload: WellCaseIn, production_mode: bool,
                     include_uncertainty: bool, uncertainty_seed: int,
                     uncertainty_samples: int) -> Any:
    config = UncertaintyConfig(seed=uncertainty_seed, samples=uncertainty_samples) \
        if include_uncertainty else None
    return await run_in_threadpool(
        diagnose, _to_well_case(payload), production_mode, None, config,
        SERVER_RUNTIME_CALIBRATION,
    )


# --------------------------------------------------------------------------
# Application and endpoints
# --------------------------------------------------------------------------

app = FastAPI(
    title="GALIT API",
    description=("Versioned integration-prototype API for well complication screening. "
                 "It is decision support, not automatic control and not production-ready. "
                 "Authentication/authorization claims are explicitly roadmap items."),
    version=galit.__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied.strip()[:128] if supplied.strip() else str(uuid.uuid4())
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        AUDIT_LOGGER.exception(json.dumps({
            "event": "api_request", "request_id": request_id,
            "method": request.method, "path": request.url.path, "status": 500,
        }, ensure_ascii=False))
        raise
    response.headers["X-Request-ID"] = request_id
    AUDIT_LOGGER.info(json.dumps({
        "event": "api_request", "request_id": request_id,
        "method": request.method, "path": request.url.path, "status": status_code,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }, ensure_ascii=False))
    return response


@app.get("/api/v1/health", summary="Liveness check",
         description="Process liveness only; safe for container health checks.")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": galit.__version__}


@app.get("/api/v1/readiness", summary="Readiness and safe runtime metadata",
         description="Returns non-secret runtime configuration and evidence labels.")
async def readiness() -> dict[str, Any]:
    runtime = SERVER_RUNTIME_CALIBRATION
    return {
        "status": "ready",
        "version": galit.__version__,
        "max_batch_size": MAX_BATCH_SIZE,
        "cors": {"mode": "allow-list" if CORS_ORIGINS else "deny", "origin_count": len(CORS_ORIGINS)},
        "calibration": {
            "id": runtime.calibration_id if runtime else "baseline",
            "version": runtime.artifact_version if runtime else "baseline",
            "status": runtime.validation_status if runtime else "baseline",
        },
        "authentication": {"status": "not implemented", "roadmap": True},
        "evidence_labels": EVIDENCE_LABELS,
    }


@app.post(
    "/api/v1/diagnose", response_model=None, summary="Diagnose one well",
    description=("Keeps the historical five-field response by default. Set include_metadata=true "
                 "for quality, severity, policy/calibration and evidence labels; profiles are opt-in."),
    responses={400: {"description": "Calculation or production-mode data-quality rejection"},
               422: {"description": "Request schema validation error"}},
)
async def diagnose_well(
    payload: WellCaseIn,
    production_mode: bool = False,
    include_metadata: bool = False,
    include_profiles: bool = False,
    include_uncertainty: bool = False,
    uncertainty_seed: int = 0,
    uncertainty_samples: int = Query(default=100, ge=20, le=1000),
) -> dict[str, Any]:
    try:
        result = await _calculate(payload, production_mode, include_uncertainty,
                                  uncertainty_seed, uncertainty_samples)
    except DataQualityError as exc:
        raise HTTPException(400, detail={"message": str(exc), "reasons": exc.reasons}) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return _result_dict(result, include_metadata=include_metadata or include_uncertainty,
                        include_profiles=include_profiles,
                        include_uncertainty=include_uncertainty)


@app.post(
    "/api/v1/forecast", response_model=ForecastResponse, summary="Forecast one well",
    description=("Honest time-to-event contract. Undated screening/unavailable events remain undated; "
                 "exact probabilities are returned only from valid holdout calibration evidence."),
    responses={400: {"description": "Forecast domain validation error"},
               422: {"description": "Request schema validation error"}},
)
async def forecast_endpoint(payload: ForecastRequest) -> ForecastResponse:
    try:
        case = _to_well_case(payload.well)
        diagnosis = await run_in_threadpool(
            diagnose, case, False, None, None, SERVER_RUNTIME_CALIBRATION,
        )
        history = galit.ForecastHistory(
            snapshots=tuple(galit.ForecastSnapshot(**item.model_dump())
                            for item in payload.history_snapshots),
            events=tuple(galit.ObservedForecastEvent(**item.model_dump())
                         for item in payload.observed_events),
            dataset_id=payload.dataset_id,
        )
        config = galit.ForecastConfig(**payload.config.model_dump()) if payload.config else None
        integrity = (galit.CorrosionIntegrityInput(**payload.corrosion_integrity.model_dump())
                     if payload.corrosion_integrity else None)
        calibration = (galit.ForecastCalibrationEvidence(
            **payload.calibration_evidence.model_dump()) if payload.calibration_evidence else None)
        forecast = await run_in_threadpool(
            galit.forecast_well, diagnosis, case, history=history, as_of=payload.as_of,
            config=config, corrosion_integrity=integrity, calibration=calibration,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return ForecastResponse(
        well=forecast.well,
        as_of=payload.as_of,
        summary=asdict(forecast.summary),
        events=[ForecastEventOut(**asdict(event)) for event in forecast.events],
        methodology={
            "module": "galit.forecast.v1",
            "temporal_method": "Theil-Sen trend with scenario bounds where sufficient history exists",
            "calibration_gate": "holdout-validated evidence only",
        },
        advisory_notice=("Screening windows are scenarios, not guaranteed event dates or failure probabilities. "
                         "Unavailable events intentionally contain no fabricated dates."),
    )


@app.post(
    "/api/v1/risk-economics", response_model=RiskEconomicsResponse,
    summary="Calculate auditable risk economics for one well",
    description=("All monetary inputs are explicit and interpreted in one supplied currency. "
                 "The endpoint never supplies prices or exchange rates. A validated forecast "
                 "probability may be used only when explicitly requested and available."),
)
async def risk_economics_endpoint(payload: RiskEconomicsRequest) -> RiskEconomicsResponse:
    case = _to_well_case(payload.well)
    economics = payload.economics
    probability = economics.event_probability
    probability_source = economics.probability_source
    forecast_link: dict[str, Any] = {"requested": payload.use_forecast_probability,
                                     "used": False, "reason": "not requested"}
    if payload.use_forecast_probability:
        if payload.as_of is None:
            raise HTTPException(400, detail="as_of is required when use_forecast_probability=true")
        diagnosis = await run_in_threadpool(
            diagnose, case, False, None, None, SERVER_RUNTIME_CALIBRATION,
        )
        history = galit.ForecastHistory(
            snapshots=tuple(galit.ForecastSnapshot(**item.model_dump())
                            for item in payload.history_snapshots),
            dataset_id=payload.dataset_id,
        )
        calibration = (galit.ForecastCalibrationEvidence(
            **payload.calibration_evidence.model_dump()) if payload.calibration_evidence else None)
        forecast = await run_in_threadpool(
            galit.forecast_well, diagnosis, case, history=history, as_of=payload.as_of,
            calibration=calibration,
        )
        candidates = [event for event in forecast.events if event.probability is not None]
        if candidates:
            selected = max(candidates, key=lambda event: float(event.probability or 0.0))
            probability = selected.probability
            probability_source = f"forecast:{selected.mechanism.value}:{selected.id}"
            forecast_link = {"requested": True, "used": True, "event_id": selected.id,
                             "mechanism": selected.mechanism.value,
                             "probability_horizon_days": selected.probability_horizon_days}
        else:
            forecast_link = {"requested": True, "used": False,
                             "reason": "no calibrated forecast probability available"}

    oil_rate = economics.oil_rate_m3_day
    if oil_rate is None:
        oil_rate = case.rate.q_oil_m3d
    try:
        result = await run_in_threadpool(
            galit.calculate_risk_economics,
            galit.RiskEconomicsInput(
                event_probability=probability,
                horizon_days=economics.horizon_days,
                treatment_efficiency=economics.treatment_efficiency,
                event_downtime_days=economics.event_downtime_days,
                treatment_downtime_days=economics.treatment_downtime_days,
                oil_rate_m3_day=oil_rate,
                product_price_per_m3=economics.product_price_per_m3,
                operating_loss_per_day=economics.operating_loss_per_day,
                treatment_cost=economics.treatment_cost,
                currency=economics.currency,
                production_loss_fraction=economics.production_loss_fraction,
                probability_source=probability_source,
            ),
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return RiskEconomicsResponse(
        well=case.name,
        status=result.status.value,
        currency=result.currency,
        data_sufficient=result.data_sufficient,
        missing_inputs=result.missing_inputs,
        assumptions=result.assumptions,
        formulas=result.formulas,
        breakdown=asdict(result.breakdown),
        limitations=result.limitations,
        forecast_link=forecast_link,
    )


@app.post(
    "/api/v1/scenarios/compare", response_model=ScenarioCompareResponse,
    summary="Compare an auditable what-if scenario",
    description=("Additive before/after contract. Screening scores are not probabilities; "
                 "operational actions affect physics only through explicit sourced overrides."),
)
async def scenario_compare_endpoint(payload: ScenarioCompareRequest) -> ScenarioCompareResponse:
    try:
        changes_data = payload.changes.model_dump()
        override_data = changes_data.pop("effect_override")
        override = galit.EffectOverride(**override_data) if override_data else None
        changes = galit.ScenarioChanges(**changes_data, effect_override=override)
        economics = (galit.ScenarioEconomics(**payload.economics.model_dump())
                     if payload.economics else None)
        result = await run_in_threadpool(
            galit.compare_scenario, _to_well_case(payload.well), changes, economics,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    data = asdict(result)
    data["status"] = result.status.value
    if result.economics is not None:
        data["economics"]["status"] = result.economics.status.value
    return ScenarioCompareResponse(**data)


def _treatment_error(exc: Exception) -> HTTPException:
    if isinstance(exc, galit.TreatmentNotFoundError):
        return HTTPException(404, detail=str(exc))
    if isinstance(exc, galit.TreatmentConflictError):
        return HTTPException(409, detail={"message": str(exc), "type": "conflict"})
    if isinstance(exc, galit.TreatmentStorageError):
        return HTTPException(503, detail={"message": str(exc), "type": "storage_error"})
    return HTTPException(400, detail=str(exc))


@app.post("/api/v1/treatments", response_model=TreatmentOut, status_code=201,
          summary="Create a planned treatment journal record")
async def create_treatment(payload: TreatmentCreateIn) -> TreatmentOut:
    try:
        record = await run_in_threadpool(galit.new_treatment, **payload.model_dump())
        record = await run_in_threadpool(TREATMENTS.create, record)
        return TreatmentOut(**record.to_dict())
    except (ValueError, galit.TreatmentStorageError) as exc:
        raise _treatment_error(exc) from exc


@app.get("/api/v1/treatments", response_model=list[TreatmentOut],
         summary="List treatment journal records")
async def list_treatments(
    well: str | None = None, status: galit.TreatmentStatus | None = None,
    complication_type: str | None = None, reagent: str | None = None,
    currency: str | None = None, well_group: str | None = None,
    include_archived: bool = False, offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[TreatmentOut]:
    try:
        records = await run_in_threadpool(
            TREATMENTS.list, well=well, status=status, complication_type=complication_type,
            reagent=reagent, currency=currency, well_group=well_group,
            include_archived=include_archived, offset=offset, limit=limit,
        )
        return [TreatmentOut(**record.to_dict()) for record in records]
    except galit.TreatmentStorageError as exc:
        raise _treatment_error(exc) from exc


@app.get("/api/v1/treatments/{record_id}", response_model=TreatmentOut,
         summary="Get one treatment journal record")
async def get_treatment(record_id: str) -> TreatmentOut:
    try:
        return TreatmentOut(**(await run_in_threadpool(TREATMENTS.get, record_id)).to_dict())
    except (galit.TreatmentNotFoundError, galit.TreatmentStorageError) as exc:
        raise _treatment_error(exc) from exc


@app.patch("/api/v1/treatments/{record_id}", response_model=TreatmentOut,
           summary="Edit a record before assessment or advance its lifecycle")
async def update_treatment(record_id: str, payload: TreatmentUpdateIn) -> TreatmentOut:
    try:
        current = await run_in_threadpool(TREATMENTS.get, record_id)
        changes = payload.model_dump(exclude={"revision", "status"}, exclude_unset=True)
        updated = (current.transition(payload.status, **changes)
                   if payload.status is not None else current.edit(**changes))
        updated = await run_in_threadpool(
            TREATMENTS.update, updated, expected_revision=payload.revision)
        return TreatmentOut(**updated.to_dict())
    except (ValueError, galit.TreatmentConflictError, galit.TreatmentNotFoundError,
            galit.TreatmentStorageError) as exc:
        raise _treatment_error(exc) from exc


@app.delete("/api/v1/treatments/{record_id}", response_model=TreatmentOut,
            summary="Soft archive a treatment record")
async def archive_treatment(
    record_id: str, revision: int = Query(ge=1),
) -> TreatmentOut:
    try:
        archived = await run_in_threadpool(
            TREATMENTS.archive, record_id, expected_revision=revision)
        return TreatmentOut(**archived.to_dict())
    except (galit.TreatmentConflictError, galit.TreatmentNotFoundError,
            galit.TreatmentStorageError) as exc:
        raise _treatment_error(exc) from exc


@app.get("/api/v1/treatments/analytics/summary", response_model=None,
         summary="Aggregate assessed treatment effects")
async def treatment_analytics(
    group_by: str = Query(default="reagent", pattern="^(well|complication_type|reagent|well_group)$"),
    well: str | None = None, complication_type: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    try:
        records = await run_in_threadpool(
            TREATMENTS.list, well=well, complication_type=complication_type, currency=currency,
        )
        return await run_in_threadpool(galit.treatment_summary, records, group_by)
    except (ValueError, galit.TreatmentStorageError) as exc:
        raise _treatment_error(exc) from exc


@app.get("/api/v1/treatments/analytics/compare", response_model=None,
         summary="Compare reagents on an explicit observational metric")
async def treatment_compare(
    reagent_a: str, reagent_b: str,
    metric: str = Query(default="success_rate", pattern="^(success_rate|mean_effect_days)$"),
    min_sample_size: int = Query(default=galit.DEFAULT_MIN_SAMPLE_SIZE, ge=2, le=1000),
    complication_type: str = Query(min_length=1),
    well_group: str = Query(min_length=1),
) -> dict[str, Any]:
    try:
        records = await run_in_threadpool(TREATMENTS.list)
        return await run_in_threadpool(
            galit.compare_reagents, records, reagent_a, reagent_b, metric=metric,
            min_sample_size=min_sample_size, complication_type=complication_type,
            well_group=well_group,
        )
    except (ValueError, galit.TreatmentStorageError) as exc:
        raise _treatment_error(exc) from exc


def _passport_error(exc: Exception) -> HTTPException:
    if isinstance(exc, galit.PassportNotFoundError):
        return HTTPException(404, detail=str(exc))
    if isinstance(exc, galit.PassportConflictError):
        return HTTPException(409, detail={"message": str(exc), "type": "conflict"})
    if isinstance(exc, galit.PassportStorageError):
        return HTTPException(503, detail={"message": str(exc), "type": "storage_error"})
    return HTTPException(400, detail=str(exc))


@app.post("/api/v1/passport/events", status_code=201, response_model=None,
          summary="Add an event to a digital well passport")
async def create_passport_event(payload: PassportEventCreateIn) -> dict[str, Any]:
    if payload.event_type in {galit.PassportEventType.DEPOSIT_PHOTO, galit.PassportEventType.LAB_REPORT}:
        raise HTTPException(400, detail="attachment events must be uploaded through /api/v1/passport/attachments")
    try:
        event = await run_in_threadpool(galit.new_passport_event, **payload.model_dump())
        return (await run_in_threadpool(PASSPORTS.create, event)).to_dict()
    except (ValueError, galit.PassportStorageError) as exc:
        raise _passport_error(exc) from exc


@app.get("/api/v1/passport/events", response_model=None, summary="Filter passport events")
async def list_passport_events(
    well: str | None = None, event_type: galit.PassportEventType | None = None,
    date_from: datetime | None = None, date_to: datetime | None = None,
    offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    try:
        rows = await run_in_threadpool(PASSPORTS.list, well=well, event_type=event_type,
                                       date_from=date_from, date_to=date_to, offset=offset, limit=limit)
        return [row.to_dict() for row in rows]
    except (ValueError, galit.PassportStorageError) as exc:
        raise _passport_error(exc) from exc


@app.get("/api/v1/passport/events/{event_id}", response_model=None, summary="Get one passport event")
async def get_passport_event(event_id: str) -> dict[str, Any]:
    try:
        return (await run_in_threadpool(PASSPORTS.get, event_id)).to_dict()
    except (galit.PassportNotFoundError, galit.PassportStorageError) as exc:
        raise _passport_error(exc) from exc


@app.patch("/api/v1/passport/events/{event_id}", response_model=None, summary="Update a passport event")
async def update_passport_event(event_id: str, payload: PassportEventUpdateIn) -> dict[str, Any]:
    try:
        current = await run_in_threadpool(PASSPORTS.get, event_id)
        changes = payload.model_dump(exclude={"revision"}, exclude_unset=True)
        updated = current.edit(**changes)
        return (await run_in_threadpool(PASSPORTS.update, updated,
                                        expected_revision=payload.revision)).to_dict()
    except (ValueError, galit.PassportNotFoundError, galit.PassportConflictError,
            galit.PassportStorageError) as exc:
        raise _passport_error(exc) from exc


@app.delete("/api/v1/passport/events/{event_id}", response_model=None, summary="Delete a passport event")
async def delete_passport_event(event_id: str, revision: int = Query(ge=1)) -> dict[str, Any]:
    try:
        return (await run_in_threadpool(PASSPORTS.delete, event_id,
                                        expected_revision=revision)).to_dict()
    except (galit.PassportNotFoundError, galit.PassportConflictError,
            galit.PassportStorageError) as exc:
        raise _passport_error(exc) from exc


@app.post("/api/v1/passport/attachments", status_code=201, response_model=None,
          summary="Upload a deposit photo or laboratory report")
async def upload_passport_attachment(request: Request) -> dict[str, Any]:
    filename = request.headers.get("x-file-name", "")
    mime_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    kind = request.query_params.get("event_type", "")
    try:
        event_type = galit.PassportEventType(kind)
        if event_type not in {galit.PassportEventType.DEPOSIT_PHOTO, galit.PassportEventType.LAB_REPORT}:
            raise ValueError("event_type must be deposit_photo or lab_report")
        content = await request.body()
        if len(content) > galit.MAX_ATTACHMENT_SIZE:
            raise HTTPException(413, detail=f"attachment exceeds {galit.MAX_ATTACHMENT_SIZE} bytes")
        attachment = await run_in_threadpool(PASSPORTS.save_attachment, filename, mime_type, content)
        event = galit.new_passport_event(
            well_id=request.query_params.get("well_id", ""),
            well_name=request.query_params.get("well_name", ""), event_type=event_type,
            event_at=datetime.now().astimezone(), title=request.query_params.get("title", filename),
            data={}, notes=request.query_params.get("notes") or None, source="api-upload",
            attachment=attachment,
        )
        return (await run_in_threadpool(PASSPORTS.create, event)).to_dict()
    except HTTPException:
        raise
    except (ValueError, galit.PassportStorageError) as exc:
        raise _passport_error(exc) from exc


@app.get("/api/v1/passport/{well}", response_model=None,
         summary="Get unified passport timeline and summary including treatments")
async def get_passport(well: str, event_type: galit.PassportEventType | None = None,
                       date_from: datetime | None = None, date_to: datetime | None = None,
                       limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
    try:
        events = await run_in_threadpool(PASSPORTS.list, well=well, event_type=event_type,
                                         date_from=date_from, date_to=date_to, limit=limit)
        treatments = await run_in_threadpool(TREATMENTS.list, well=well, limit=limit)
        return {"well": well, "schema_version": galit.PASSPORT_SCHEMA_VERSION,
                "summary": galit.passport_summary(events, treatments),
                "timeline": galit.passport_timeline(events, treatments)}
    except (ValueError, galit.PassportStorageError, galit.TreatmentStorageError) as exc:
        raise _passport_error(exc) from exc


@app.post(
    "/api/v1/diagnose/bulk", response_model=None, summary="Diagnose a bounded well batch",
    description=("Each item is validated and calculated independently. Item errors do not abort "
                 "the batch. An empty/non-array envelope is 422; exceeding the configured limit is 413."),
    responses={413: {"description": "Batch exceeds GALIT_MAX_BATCH_SIZE"},
               422: {"description": "Envelope must be a non-empty JSON array"}},
)
async def diagnose_bulk(
    payload: list[Any],
    production_mode: bool = False,
    include_profiles: bool = False,
) -> dict[str, Any]:
    if not payload:
        raise HTTPException(422, detail="bulk payload must contain at least one item")
    if len(payload) > MAX_BATCH_SIZE:
        raise HTTPException(413, detail={"message": "batch size exceeds configured limit",
                                         "max_batch_size": MAX_BATCH_SIZE, "received": len(payload)})
    items: list[dict[str, Any]] = []
    succeeded = 0
    for index, raw in enumerate(payload):
        try:
            model = WellCaseIn.model_validate(raw)
            result = await _calculate(model, production_mode, False, 0, 100)
            items.append({"index": index, "status": "success", "name": model.name,
                          "result": _result_dict(result, include_metadata=True,
                                                 include_profiles=include_profiles,
                                                 include_uncertainty=False)})
            succeeded += 1
        except ValidationError as exc:
            items.append({"index": index, "status": "error", "error": {
                "type": "validation_error", "status_code": 422,
                "details": exc.errors(include_url=False, include_input=False)}})
        except DataQualityError as exc:
            items.append({"index": index, "status": "error", "error": {
                "type": "data_quality_error", "status_code": 422,
                "message": str(exc), "reasons": exc.reasons}})
        except (ValueError, KeyError) as exc:
            items.append({"index": index, "status": "error", "error": {
                "type": "calculation_error", "status_code": 422, "message": str(exc)}})
    return {"count": len(items), "succeeded": succeeded, "failed": len(items) - succeeded,
            "max_batch_size": MAX_BATCH_SIZE, "items": items,
            "policy": {"id": "server-selected", "request_overrides": False},
            "evidence_labels": EVIDENCE_LABELS}


@app.post(
    "/api/v1/master-plan", response_model=None, summary="Build a master plan for a well batch",
    description=("Partial-success policy: every well is validated and calculated independently; "
                 "successful wells form the plan and item-level errors are returned in errors. "
                 "Loss values are screening estimates for prioritisation, not failure forecasts."),
    responses={413: {"description": "Batch exceeds GALIT_MAX_BATCH_SIZE"},
               422: {"description": "Payload must be a non-empty JSON array"}},
)
async def master_plan(
    payload: list[Any],
    plan_date: date | None = None,
    min_risk: float = Query(default=0.10, ge=0.0, le=1.0),
    limit: int | None = Query(default=None, ge=0),
    production_mode: bool = False,
) -> dict[str, Any]:
    if not payload:
        raise HTTPException(422, detail="master-plan payload must contain at least one item")
    if len(payload) > MAX_BATCH_SIZE:
        raise HTTPException(413, detail={"message": "batch size exceeds configured limit",
                                         "max_batch_size": MAX_BATCH_SIZE, "received": len(payload)})
    diagnosed: list[DiagnosedWell] = []
    errors: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        try:
            model = WellCaseIn.model_validate(raw)
            case = _to_well_case(model)
            result = await _calculate(model, production_mode, False, 0, 100)
            diagnosed.append(DiagnosedWell(case, result))
        except ValidationError as exc:
            errors.append({"index": index, "name": raw.get("name") if isinstance(raw, dict) else None,
                           "type": "validation_error",
                           "details": exc.errors(include_url=False, include_input=False)})
        except DataQualityError as exc:
            errors.append({"index": index, "name": raw.get("name") if isinstance(raw, dict) else None,
                           "type": "data_quality_error", "message": str(exc),
                           "reasons": exc.reasons})
        except (ValueError, KeyError) as exc:
            errors.append({"index": index, "name": raw.get("name") if isinstance(raw, dict) else None,
                           "type": "calculation_error", "message": str(exc)})

    policy = replace(DEFAULT_MASTER_PLAN_POLICY, low_risk_cutoff=min_risk)
    plan = generate_master_plan(
        diagnosed, generated_at=plan_date, include_low_risk=False, limit=limit, policy=policy,
    )
    tasks = [asdict(task) for task in plan.tasks]
    loss_methodology = ({"method": tasks[0]["possible_oil_loss"]["method"],
                         "limitations": tasks[0]["possible_oil_loss"]["limitations"]}
                        if tasks else {
                            "method": "Дебит нефти × сценарный диапазон или integrated risk score.",
                            "limitations": "Screening-оценка для ранжирования, не прогноз потери.",
                        })
    return {
        "policy": {"mode": "partial", "id": plan.policy_id, "version": plan.policy_version,
                   "min_risk": min_risk},
        "generated_at": plan.generated_at,
        "plan_date": plan.plan_date,
        "summary": asdict(plan.summary),
        "tasks": tasks,
        "errors": errors,
        "loss_methodology": loss_methodology,
        "advisory_notice": plan.advisory_notice,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
