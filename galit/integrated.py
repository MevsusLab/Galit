"""Интегрированная оценка риска осложнений и подбор технологии.

Это ядро продукта: не четыре разрозненные модели, а единая шкала,
на которой галит, кальцит, АСПО и коррозия сравниваются между собой
и конкурируют за ограниченный бюджет обработок.

Почему интеграция важна, а не косметика:
  * механизмы КОНКУРИРУЮТ за одну и ту же операцию (одна поездка
    бригады, один простой скважины) -- значит их надо ранжировать
    на общей шкале, иначе четыре модели дадут четыре несравнимых
    "риска 0.7";
  * механизмы ВЗАИМОДЕЙСТВУЮТ: отложения солей и АСПО сужают
    проходное сечение -> растёт скорость потока -> растёт
    массоперенос -> ускоряется CO2-коррозия. Обратно: плёнка FeCO3
    защищает от коррозии, но её разрушают скребки;
  * технология борьбы с одним механизмом может усугубить другой
    (например, кислотная обработка от карбонатов повышает коррозию).

Ни одного исследования, покрывающего все четыре механизма сразу,
в открытой литературе не существует -- это и есть незанятая ниша.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
from types import MappingProxyType
from typing import Mapping

from .corrosion import (
    CorrosionConditions,
    corrosion_rate_1995,
    corrosion_severity,
)
from .scale import (
    WaterAnalysis,
    calcite_degas_ph_screening,
    scale_risk_profile,
    stiff_davis_index_checked,
)
from .wax import (
    WaxProperties,
    recommend_wax_treatment,
    wax_deposition_severity,
    wax_onset_depth,
)
from .wellbore import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
    pressure_profile,
    temperature_profile,
)
from .fluids import (
    bubble_point_standing,
    oil_fvf_standing,
    solution_gor_standing,
)


MECHANISMS = ("halite", "calcite", "wax", "corrosion")


@dataclass(frozen=True)
class RiskPolicy:
    """Версионированная политика интегрального риска."""

    policy_id: str = "galit-baseline"
    version: str = "1.0"
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "halite": 0.30, "calcite": 0.15, "wax": 0.30, "corrosion": 0.25,
    })
    severity_warn: float = 0.35
    severity_critical: float = 0.60
    risk_warn: float = 0.35
    risk_critical: float = 0.60
    normalize_weights: bool = False

    def __post_init__(self) -> None:
        weights = dict(self.weights)
        if set(weights) != set(MECHANISMS):
            raise ValueError("RiskPolicy.weights должен содержать ровно четыре механизма")
        if any(not math.isfinite(v) or v < 0.0 for v in weights.values()):
            raise ValueError("Веса механизмов должны быть конечными и неотрицательными")
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("Сумма весов должна быть положительной")
        if self.normalize_weights:
            weights = {key: value / total for key, value in weights.items()}
        elif not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Сумма весов RiskPolicy должна равняться 1")
        for low, high, name in (
            (self.severity_warn, self.severity_critical, "severity"),
            (self.risk_warn, self.risk_critical, "risk"),
        ):
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(f"Пороги {name} должны удовлетворять 0 <= warn <= critical <= 1")
        if not self.policy_id or not self.version:
            raise ValueError("RiskPolicy требует непустые policy_id и version")
        object.__setattr__(self, "weights", MappingProxyType(weights))


DEFAULT_RISK_POLICY = RiskPolicy()
# Legacy alias: прежние ключи/значения доступны, но случайная мутация запрещена.
MECHANISM_WEIGHTS = DEFAULT_RISK_POLICY.weights


@dataclass(frozen=True)
class UncertaintyConfig:
    """Настройки воспроизводимого сценарного ensemble (не калибровка)."""

    seed: int = 0
    samples: int = 100
    relative_range: float = 0.05
    absolute_ranges: Mapping[str, float] = field(default_factory=lambda: {
        "thermal.t_surface_c": 1.0,
        "wax.wat_stock_tank_c": 1.0,
        "water.ph": 0.1,
    })
    site_relative_range: float = 0.05

    def __post_init__(self) -> None:
        if self.samples < 20:
            raise ValueError("UncertaintyConfig.samples должен быть не меньше 20")
        if self.relative_range < 0.0 or self.site_relative_range < 0.0:
            raise ValueError("Диапазоны неопределённости не могут быть отрицательными")
        ranges = dict(self.absolute_ranges)
        if any(not math.isfinite(v) or v < 0.0 for v in ranges.values()):
            raise ValueError("Абсолютные диапазоны должны быть конечными и неотрицательными")
        object.__setattr__(self, "absolute_ranges", MappingProxyType(ranges))


@dataclass(frozen=True)
class ScenarioInterval:
    p05: float | None
    p50: float | None
    p95: float | None


@dataclass
class UncertaintyResult:
    method: str = "scenario/sensitivity ensemble"
    samples: int = 0
    seed: int = 0
    confidence_label: str = "scenario"
    integrated_risk: ScenarioInterval | None = None
    mechanisms: dict[str, ScenarioInterval] = field(default_factory=dict)
    wax_onset_m: ScenarioInterval | None = None
    probability_of_deposition: float = 0.0
    warnings: list[str] = field(default_factory=list)

# Поля, влияющие на промышленное решение по каждому механизму. Группы
# намеренно заданы в терминах публичной модели WellCase.
CRITICAL_FIELDS = {
    "wellbore": {
        "geometry.depth_m", "geometry.tubing_id_m",
        "rate.q_oil_m3d", "rate.q_water_m3d", "rate.gor_m3m3",
        "thermal.t_surface_c", "thermal.geothermal_grad", "thermal.u_to",
        "p_wellhead_pa",
    },
    "halite_calcite": {
        "water.ions_mg_l", "water.ph", "water.t_c", "water.p_pa",
        "fluid.salinity_ppm",
    },
    "wax": {"wax.wat_stock_tank_c", "wax.wax_content_pct"},
    "corrosion": {"co2_mol_frac", "inhibitor_efficiency"},
}
QUALITY_FIELDS = set().union(*CRITICAL_FIELDS.values())
VALID_SOURCES = {"measured", "derived", "default", "synthetic"}


class DataQualityError(ValueError):
    """Недостаточно фактических данных для промышленного прогноза."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("Промышленный расчёт отклонён: " + "; ".join(self.reasons))


@dataclass
class DataProvenance:
    """Происхождение входов; пустая карта сохраняет legacy-поведение.

    Для старых конструкторов явно переданные значения считаются measured.
    Загрузчики, которые сами подставляют данные, обязаны записывать source.
    """

    sources: dict[str, str] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    defaulted_fields: list[str] = field(default_factory=list)
    synthetic_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = sorted({v for v in self.sources.values() if v not in VALID_SOURCES})
        if unknown:
            raise ValueError("Неизвестные источники данных: " + ", ".join(unknown))
        self.defaulted_fields = sorted(set(self.defaulted_fields) | {
            k for k, v in self.sources.items() if v == "default"
        })
        self.synthetic_fields = sorted(set(self.synthetic_fields) | {
            k for k, v in self.sources.items() if v == "synthetic"
        })
        self.missing_fields = sorted(set(self.missing_fields))

    def source(self, field_name: str) -> str:
        return self.sources.get(field_name, "measured")


@dataclass
class DataQuality:
    """Итоговая совместимая оценка полноты и пригодности данных."""

    sources: dict[str, str] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    defaulted_fields: list[str] = field(default_factory=list)
    synthetic_fields: list[str] = field(default_factory=list)
    completeness: float = 1.0
    grade: str = "A"
    production_ready: bool = True
    reasons: list[str] = field(default_factory=list)


def assess_quality(provenance: DataProvenance) -> DataQuality:
    """Оценить качество без изменения численного расчёта."""
    scores = {"measured": 1.0, "derived": 0.8, "default": 0.5, "synthetic": 0.25}
    missing = set(provenance.missing_fields)
    completeness = sum(
        0.0 if name in missing else scores[provenance.source(name)]
        for name in QUALITY_FIELDS
    ) / len(QUALITY_FIELDS)
    if completeness >= 0.90:
        grade = "A"
    elif completeness >= 0.75:
        grade = "B"
    elif completeness >= 0.60:
        grade = "C"
    else:
        grade = "D"

    reasons: list[str] = []
    for group, fields in CRITICAL_FIELDS.items():
        bad = sorted(
            f for f in fields
            if f in missing or provenance.source(f) in {"default", "synthetic"}
        )
        if bad:
            reasons.append(f"{group}: нет фактических данных ({', '.join(bad)})")
    return DataQuality(
        sources=dict(provenance.sources),
        missing_fields=sorted(missing),
        defaulted_fields=list(provenance.defaulted_fields),
        synthetic_fields=list(provenance.synthetic_fields),
        completeness=max(0.0, min(completeness, 1.0)),
        grade=grade,
        production_ready=not reasons,
        reasons=reasons,
    )


@dataclass
class WellCase:
    """Полное описание скважины для расчёта."""
    name: str
    geometry: WellGeometry
    rate: ProductionRate
    fluid: FluidProperties
    thermal: ThermalParams
    water: WaterAnalysis
    wax: WaxProperties
    co2_mol_frac: float = 0.02
    inhibitor_efficiency: float = 0.0
    lift_type: str = "ЭЦН"          # ЭЦН | ШГН | фонтан
    p_wellhead_pa: float = 1.2e6
    provenance: DataProvenance = field(default_factory=DataProvenance)


@dataclass
class DiagnosisResult:
    """Результат интегрированной диагностики."""
    well: str
    depths: list[float]
    temps: list[float]
    pressures: list[float]
    wat_profile: list[float]
    scale: dict[str, float] = field(default_factory=dict)
    corrosion: dict[str, float] = field(default_factory=dict)
    severity: dict[str, float] = field(default_factory=dict)
    wax_onset_m: float | None = None
    integrated_risk: float = 0.0
    dominant: str = ""
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)
    quality: DataQuality = field(default_factory=DataQuality)
    policy_id: str = DEFAULT_RISK_POLICY.policy_id
    policy_version: str = DEFAULT_RISK_POLICY.version
    mechanism_weights: dict[str, float] = field(default_factory=lambda: dict(MECHANISM_WEIGHTS))
    uncertainty: UncertaintyResult | None = None


def _halite_severity(si: float) -> float:
    """Нормировка индекса насыщения галита в шкалу 0..1.

    SI < -0.3  -- недосыщение, риска нет
    SI = 0     -- равновесие, начало кристаллизации
    SI > 0.3   -- сильное пересыщение
    """
    if si < -0.3:
        return 0.0
    return min(max((si + 0.3) / 0.6, 0.0), 1.0)


def _calcite_severity(sdsi: float) -> float:
    """Нормировка индекса Стиффа-Дэвиса в шкалу 0..1."""
    if sdsi <= 0.0:
        return 0.0
    return min(sdsi / 1.5, 1.0)


def diagnose(
    case: WellCase,
    production_mode: bool = False,
    risk_policy: RiskPolicy | None = None,
    uncertainty: UncertaintyConfig | None = None,
) -> DiagnosisResult:
    """Полный расчёт; uncertainty opt-in сохраняет быстрый legacy-контракт."""
    policy = risk_policy or DEFAULT_RISK_POLICY
    quality = assess_quality(case.provenance)
    if production_mode and not quality.production_ready:
        raise DataQualityError(quality.reasons)
    warnings: list[str] = []
    if not quality.production_ready:
        warnings.append(
            f"Screening: качество данных {quality.grade}, полнота "
            f"{quality.completeness:.0%}; промышленный прогноз недоступен: "
            + "; ".join(quality.reasons)
        )

    # --- 1. профили в стволе ---
    depths, temps, warn_t = temperature_profile(
        case.geometry, case.rate, case.fluid, case.thermal
    )
    if warn_t:
        warnings.append(warn_t)

    pressures = pressure_profile(
        case.geometry, case.rate, case.fluid, depths, temps,
        p_wellhead_pa=case.p_wellhead_pa,
    )

    # --- 2. АСПО ---
    pb = bubble_point_standing(
        case.rate.gor_m3m3, case.fluid.gamma_gas,
        case.thermal.t_surface_c + case.thermal.geothermal_grad * case.geometry.depth_m,
        case.fluid.gamma_oil,
    )
    onset, wat_profile = wax_onset_depth(depths, temps, pressures, case.wax, pb)
    sev_wax = wax_deposition_severity(depths, temps, wat_profile, onset, case.wax)

    # --- 3. соли: устье, с явным screening дегазации CO2 ---
    ph_wh, warn_degas = calcite_degas_ph_screening(
        ph_initial=case.water.ph,
        t_c=temps[0],
        p_initial_pa=case.water.p_pa,
        p_local_pa=pressures[0],
        co2_mol_frac=case.co2_mol_frac,
    )
    if warn_degas:
        warnings.append(warn_degas)
    water_wh = WaterAnalysis(
        ions_mg_l=case.water.ions_mg_l,
        ph=ph_wh,
        t_c=temps[0],
        p_pa=pressures[0],
    )
    scale = scale_risk_profile(water_wh)
    scale["ph_initial"] = case.water.ph
    scale["ph_wellhead"] = ph_wh
    sdsi_checked, sdsi_warnings = stiff_davis_index_checked(water_wh)
    scale["si_calcite"] = sdsi_checked
    warnings.extend(sdsi_warnings)
    sev_halite = _halite_severity(scale["si_halite"])
    sev_calcite = _calcite_severity(scale["si_calcite"])

    if water_wh.tds_mg_l > 10_000:
        warnings.append(
            f"TDS {water_wh.tds_mg_l / 1000:.0f} г/л -- LSI неприменим, "
            f"используется Stiff-Davis (ASTM D4582)."
        )

    # --- 4. коррозия: профиль по согласованным узлам depths/T/P ---
    area = 3.14159 * (case.geometry.tubing_id_m / 2.0) ** 2

    def corrosion_profile(diameter_m: float) -> list[dict[str, float]]:
        area_local = 3.14159 * (diameter_m / 2.0) ** 2
        profile: list[dict[str, float]] = []
        for depth, t_c, p_pa in zip(depths, temps, pressures):
            pb_local = bubble_point_standing(
                case.rate.gor_m3m3, case.fluid.gamma_gas, t_c,
                case.fluid.gamma_oil,
            )
            rs_local = solution_gor_standing(
                p_pa, pb_local, case.rate.gor_m3m3, case.fluid.gamma_gas,
                t_c, case.fluid.gamma_oil,
            )
            bo_local = oil_fvf_standing(
                rs_local, case.fluid.gamma_gas, t_c, case.fluid.gamma_oil,
            )
            q_liq_local = (
                case.rate.q_oil_m3d * bo_local + case.rate.q_water_m3d
            ) / 86400.0
            velocity = q_liq_local / max(area_local, 1e-9)
            cond = CorrosionConditions(
                t_c=t_c,
                p_total_pa=p_pa,
                co2_mol_frac=case.co2_mol_frac,
                velocity_m_s=velocity,
                diameter_m=diameter_m,
                ph_actual=case.water.ph,
                watercut=case.rate.watercut,
                inhibitor_efficiency=case.inhibitor_efficiency,
            )
            node = corrosion_rate_1995(cond)
            node.update({
                "depth_m": depth,
                "t_c": t_c,
                "p_pa": p_pa,
                "velocity_m_s": velocity,
            })
            profile.append(node)
        return profile

    corr_profile = corrosion_profile(case.geometry.tubing_id_m)
    corr_max = max(corr_profile, key=lambda node: node["rate_mm_yr"])
    corr = dict(corr_max)
    corr["depth_of_max_m"] = corr_max["depth_m"]
    corr["profile"] = corr_profile
    corr_label, sev_corr = corrosion_severity(corr["rate_mm_yr"])
    corr["category"] = corr_label

    # --- 5. взаимодействие: сужение применяется ко всему профилю ---
    blockage = max(sev_halite, sev_wax)
    if blockage > 0.3:
        shrink = 1.0 - 0.3 * blockage
        corr_profile_adj = corrosion_profile(case.geometry.tubing_id_m * shrink)
        corr_adj = max(corr_profile_adj, key=lambda node: node["rate_mm_yr"])
        corr["profile_with_blockage"] = corr_profile_adj
        if corr_adj["rate_mm_yr"] > corr["rate_mm_yr"]:
            corr["rate_with_blockage"] = corr_adj["rate_mm_yr"]
            corr["depth_of_max_with_blockage_m"] = corr_adj["depth_m"]
            _, sev_corr_adj = corrosion_severity(corr_adj["rate_mm_yr"])
            sev_corr = max(sev_corr, sev_corr_adj)
            warnings.append(
                f"Сужение сечения отложениями ускоряет максимум коррозии: "
                f"{corr['rate_mm_yr']:.3f} -> {corr_adj['rate_mm_yr']:.3f} мм/год."
            )

    severity = {
        "halite": sev_halite,
        "calcite": sev_calcite,
        "wax": sev_wax,
        "corrosion": sev_corr,
    }

    # --- 6. интегральный риск ---
    integrated = sum(policy.weights[k] * v for k, v in severity.items())
    dominant = max(severity, key=lambda k: severity[k] * policy.weights[k])

    rec = _recommend(dominant, severity, onset, case, corr)

    result = DiagnosisResult(
        well=case.name,
        depths=depths,
        temps=temps,
        pressures=pressures,
        wat_profile=wat_profile,
        scale=scale,
        corrosion=corr,
        severity=severity,
        wax_onset_m=onset,
        integrated_risk=integrated,
        dominant=dominant,
        recommendation=rec,
        warnings=warnings,
        quality=quality,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        mechanism_weights=dict(policy.weights),
    )
    if uncertainty is not None:
        result.uncertainty = _estimate_uncertainty(case, policy, uncertainty, quality)
        result.warnings.extend(result.uncertainty.warnings)
    return result


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _interval(values: list[float]) -> ScenarioInterval:
    return ScenarioInterval(
        _percentile(values, 0.05), _percentile(values, 0.50), _percentile(values, 0.95)
    )


def _perturb_case(case: WellCase, rng: random.Random, config: UncertaintyConfig,
                  quality: DataQuality) -> WellCase:
    # Screening/default/synthetic inputs receive deliberately wider scenario ranges.
    quality_factor = 1.0 if quality.production_ready else 1.0 + config.site_relative_range / max(config.relative_range, 1e-12)

    def rel(value: float, path: str, floor: float = 0.0) -> float:
        width = config.relative_range * quality_factor
        absolute = config.absolute_ranges.get(path, 0.0) * quality_factor
        return max(floor, value * (1.0 + rng.uniform(-width, width)) + rng.uniform(-absolute, absolute))

    ions = {key: rel(value, "water.ions_mg_l") for key, value in case.water.ions_mg_l.items()}
    return replace(
        case,
        geometry=replace(case.geometry,
            depth_m=rel(case.geometry.depth_m, "geometry.depth_m", 1.0),
            tubing_id_m=rel(case.geometry.tubing_id_m, "geometry.tubing_id_m", 1e-4)),
        rate=replace(case.rate,
            q_oil_m3d=rel(case.rate.q_oil_m3d, "rate.q_oil_m3d"),
            q_water_m3d=rel(case.rate.q_water_m3d, "rate.q_water_m3d"),
            gor_m3m3=rel(case.rate.gor_m3m3, "rate.gor_m3m3")),
        fluid=replace(case.fluid,
            gamma_oil=rel(case.fluid.gamma_oil, "fluid.gamma_oil", 0.01),
            gamma_gas=rel(case.fluid.gamma_gas, "fluid.gamma_gas", 0.01),
            salinity_ppm=rel(case.fluid.salinity_ppm, "fluid.salinity_ppm")),
        thermal=replace(case.thermal,
            t_surface_c=rel(case.thermal.t_surface_c, "thermal.t_surface_c"),
            geothermal_grad=rel(case.thermal.geothermal_grad, "thermal.geothermal_grad", 1e-6),
            u_to=rel(case.thermal.u_to, "thermal.u_to", 1e-6)),
        water=replace(case.water, ions_mg_l=ions,
            ph=min(14.0, rel(case.water.ph, "water.ph")),
            t_c=rel(case.water.t_c, "water.t_c"),
            p_pa=rel(case.water.p_pa, "water.p_pa")),
        wax=replace(case.wax,
            wat_stock_tank_c=rel(case.wax.wat_stock_tank_c, "wax.wat_stock_tank_c"),
            wax_content_pct=min(100.0, rel(case.wax.wax_content_pct, "wax.wax_content_pct"))),
        co2_mol_frac=min(1.0, rel(case.co2_mol_frac, "co2_mol_frac")),
        inhibitor_efficiency=min(1.0, rel(case.inhibitor_efficiency, "inhibitor_efficiency")),
        p_wellhead_pa=rel(case.p_wellhead_pa, "p_wellhead_pa", 1.0),
    )


def _estimate_uncertainty(case: WellCase, policy: RiskPolicy,
                          config: UncertaintyConfig, quality: DataQuality) -> UncertaintyResult:
    rng = random.Random(config.seed)
    runs = [
        diagnose(_perturb_case(case, rng, config, quality), risk_policy=policy)
        for _ in range(config.samples)
    ]
    onsets = [run.wax_onset_m for run in runs if run.wax_onset_m is not None]
    warning = (
        "Интервалы являются сценарными/sensitivity, не статистически "
        "откалиброванными доверительными интервалами."
    )
    confidence = "screening/low" if not quality.production_ready else "scenario/medium"
    if not quality.production_ready:
        warning += " Default/synthetic provenance расширяет диапазоны сценариев."
    return UncertaintyResult(
        samples=config.samples,
        seed=config.seed,
        confidence_label=confidence,
        integrated_risk=_interval([run.integrated_risk for run in runs]),
        mechanisms={key: _interval([run.severity[key] for run in runs]) for key in MECHANISMS},
        wax_onset_m=_interval(onsets) if onsets else ScenarioInterval(None, None, None),
        probability_of_deposition=len(onsets) / config.samples,
        warnings=[warning],
    )


def _recommend(dominant: str, severity: dict[str, float],
               onset: float | None, case: WellCase,
               corr: dict[str, float]) -> str:
    """Подбор технологии из арсенала НГДУ «Речицанефть».

    Учитывает конфликты: скребок разрушает защитную плёнку FeCO3,
    кислотная обработка усиливает коррозию.
    """
    sev = severity[dominant]
    if sev < 0.1:
        return "мероприятия не требуются, плановый контроль"

    parts: list[str] = []

    if dominant == "wax":
        max_scraper = 1500.0 if case.lift_type == "ЭЦН" else case.geometry.depth_m
        parts.append(recommend_wax_treatment(onset, sev, max_scraper))
        if severity["corrosion"] > 0.4 and "скребок" in parts[0]:
            parts.append(
                "ВНИМАНИЕ: скребкование разрушает плёнку FeCO3 -- "
                "требуется одновременная подача ингибитора коррозии"
            )
    elif dominant == "halite":
        if sev > 0.6:
            parts.append("постоянная подача пресной воды в затруб (разбавление)")
            parts.append("контроль: разбавление на 5 % опускает SI галита к равновесию")
        else:
            parts.append("периодические промывки пресной водой")
    elif dominant == "calcite":
        parts.append("ингибитор солеотложения (фосфонатный ряд)")
        if severity["corrosion"] > 0.4:
            parts.append(
                "кислотную обработку не применять без ингибитора коррозии"
            )
    elif dominant == "corrosion":
        rate = corr.get("rate_mm_yr", 0.0)
        parts.append(f"ингибиторная защита, целевая скорость < 0.1 мм/год "
                     f"(текущая {rate:.3f})")
        if corr.get("limiting") == "массоперенос":
            parts.append("лимитирует массоперенос -- рассмотреть снижение скорости потока")
        if case.lift_type == "ШГН":
            parts.append("контроль клапанных пар на эрозионную коррозию")

    # сопутствующие механизмы
    others = [k for k, v in severity.items() if v > 0.35 and k != dominant]
    if others:
        names = {"halite": "галит", "calcite": "кальцит",
                 "wax": "АСПО", "corrosion": "коррозия"}
        parts.append("сопутствующие: " + ", ".join(names[o] for o in others))

    return "; ".join(parts)


def rank_wells(cases: list[WellCase]) -> list[DiagnosisResult]:
    """Ранжирование фонда по интегральному риску."""
    results = [diagnose(c) for c in cases]
    return sorted(results, key=lambda r: r.integrated_risk, reverse=True)
