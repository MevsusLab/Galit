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
    DataProvenance,
    DataQualityError,
    FluidProperties,
    ProductionRate,
    ThermalParams,
    UncertaintyConfig,
    WaterAnalysis,
    WaxProperties,
    WellCase,
    WellGeometry,
    diagnose,
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
    allow_methods=["GET", "POST"],
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
