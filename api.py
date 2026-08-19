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

from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

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
from galit.scale import MW as KNOWN_IONS


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
# Модель ответа (зеркало DiagnosisResult)
# --------------------------------------------------------------------------

class IntervalOut(BaseModel):
    p05: float | None
    p50: float | None
    p95: float | None


class UncertaintyOut(BaseModel):
    method: str
    samples: int
    seed: int
    confidence_label: str
    integrated_risk: IntervalOut
    mechanisms: dict[str, IntervalOut]
    wax_onset_m: IntervalOut
    probability_of_deposition: float
    warnings: list[str]


class DiagnosisOut(BaseModel):
    """Legacy-поля плюс opt-in сценарная неопределённость."""

    integrated_risk: float = Field(description="Интегральный риск, 0..1")
    dominant: str = Field(description="Доминирующий механизм: halite | calcite | wax | corrosion")
    wax_onset_m: float | None = Field(
        default=None, description="Глубина начала АСПО от устья, м (None -- отложений нет)",
    )
    recommendation: str = Field(description="Рекомендация по технологии")
    warnings: list[str] = Field(default_factory=list, description="Предупреждения расчёта")
    policy: dict[str, object] | None = None
    uncertainty: UncertaintyOut | None = None


# --------------------------------------------------------------------------
# Приложение
# --------------------------------------------------------------------------

app = FastAPI(
    title="ГАЛИТ API",
    description="Интегрированная диагностика осложнений добычи: "
                "галит, кальцит, АСПО, CO2-коррозия.",
    version=galit.__version__,
)


def _to_well_case(payload: WellCaseIn) -> WellCase:
    """Отображение JSON-модели и автоматически выведенного provenance."""
    sources: dict[str, str] = {}
    optional_paths = {
        "co2_mol_frac": "co2_mol_frac",
        "inhibitor_efficiency": "inhibitor_efficiency",
        "p_wellhead_pa": "p_wellhead_pa",
    }
    for api_field, core_path in optional_paths.items():
        if api_field not in payload.model_fields_set:
            sources[core_path] = "default"
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


@app.get("/api/v1/health", summary="Проверка живости сервиса")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": galit.__version__}


@app.post(
    "/api/v1/diagnose",
    response_model=None,
    summary="Диагностика одной скважины",
)
async def diagnose_well(
    payload: WellCaseIn,
    production_mode: bool = False,
    include_uncertainty: bool = False,
    uncertainty_seed: int = 0,
    uncertainty_samples: int = 100,
) -> DiagnosisOut:
    """По умолчанию точный legacy JSON; расширение включается явно."""
    case = _to_well_case(payload)
    try:
        config = UncertaintyConfig(seed=uncertainty_seed, samples=uncertainty_samples) \
            if include_uncertainty else None
        result = await run_in_threadpool(
            diagnose, case, production_mode, None, config
        )
    except DataQualityError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "reasons": exc.reasons},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:  # например, непредвиденный неизвестный ион
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    output = DiagnosisOut(
        integrated_risk=result.integrated_risk,
        dominant=result.dominant,
        wax_onset_m=result.wax_onset_m,
        recommendation=result.recommendation,
        warnings=result.warnings,
    )
    if include_uncertainty:
        output.policy = {
            "id": result.policy_id,
            "version": result.policy_version,
            "weights": result.mechanism_weights,
        }
        output.uncertainty = UncertaintyOut.model_validate(result.uncertainty, from_attributes=True)
    if include_uncertainty:
        return output.model_dump()  # type: ignore[return-value]
    # Legacy contract has exactly five keys and keeps an explicit null onset.
    return output.model_dump(
        include={
            "integrated_risk", "dominant", "wax_onset_m",
            "recommendation", "warnings",
        }
    )  # type: ignore[return-value]


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
