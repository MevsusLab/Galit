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

from dataclasses import dataclass, field

from .corrosion import (
    CorrosionConditions,
    corrosion_rate_1995,
    corrosion_severity,
)
from .scale import WaterAnalysis, scale_risk_profile
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
from .fluids import bubble_point_standing


# Веса механизмов в интегральном риске.
# Откалиброваны под структуру отказов позднестадийного фонда:
# коррозия и АСПО дают больше отказов, чем галит, но галит
# при пересыщении приводит к мгновенному заклиниванию.
MECHANISM_WEIGHTS = {
    "halite": 0.30,
    "calcite": 0.15,
    "wax": 0.30,
    "corrosion": 0.25,
}


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


def diagnose(case: WellCase) -> DiagnosisResult:
    """Полный расчёт по одной скважине."""
    warnings: list[str] = []

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

    # --- 3. соли: расчёт в самой холодной точке (устье), там максимум риска ---
    water_wh = WaterAnalysis(
        ions_mg_l=case.water.ions_mg_l,
        ph=case.water.ph,
        t_c=temps[0],
        p_pa=pressures[0],
    )
    scale = scale_risk_profile(water_wh)
    sev_halite = _halite_severity(scale["si_halite"])
    sev_calcite = _calcite_severity(scale["si_calcite"])

    if water_wh.tds_mg_l > 10_000:
        warnings.append(
            f"TDS {water_wh.tds_mg_l / 1000:.0f} г/л -- LSI неприменим, "
            f"используется Stiff-Davis (ASTM D4582)."
        )

    # --- 4. коррозия: в самой горячей точке потока с высокой скоростью ---
    area = 3.14159 * (case.geometry.tubing_id_m / 2.0) ** 2
    v_liq = case.rate.q_liquid_m3d / 86400.0 / max(area, 1e-9)
    cond = CorrosionConditions(
        t_c=max(temps),
        p_total_pa=pressures[len(pressures) // 2],
        co2_mol_frac=case.co2_mol_frac,
        velocity_m_s=v_liq,
        diameter_m=case.geometry.tubing_id_m,
        ph_actual=case.water.ph,
        watercut=case.rate.watercut,
        inhibitor_efficiency=case.inhibitor_efficiency,
    )
    corr = corrosion_rate_1995(cond)
    corr_label, sev_corr = corrosion_severity(corr["rate_mm_yr"])
    corr["category"] = corr_label

    # --- 5. взаимодействие механизмов ---
    # отложения сужают сечение -> растёт скорость -> растёт массоперенос
    blockage = max(sev_halite, sev_wax)
    if blockage > 0.3:
        # эффективное сужение до 30 % диаметра при максимальной тяжести
        shrink = 1.0 - 0.3 * blockage
        cond_adj = CorrosionConditions(
            t_c=cond.t_c,
            p_total_pa=cond.p_total_pa,
            co2_mol_frac=cond.co2_mol_frac,
            velocity_m_s=v_liq / shrink ** 2,
            diameter_m=case.geometry.tubing_id_m * shrink,
            ph_actual=cond.ph_actual,
            watercut=cond.watercut,
            inhibitor_efficiency=cond.inhibitor_efficiency,
        )
        corr_adj = corrosion_rate_1995(cond_adj)
        if corr_adj["rate_mm_yr"] > corr["rate_mm_yr"]:
            corr["rate_with_blockage"] = corr_adj["rate_mm_yr"]
            _, sev_corr_adj = corrosion_severity(corr_adj["rate_mm_yr"])
            sev_corr = max(sev_corr, sev_corr_adj)
            warnings.append(
                f"Сужение сечения отложениями ускоряет коррозию: "
                f"{corr['rate_mm_yr']:.3f} -> {corr_adj['rate_mm_yr']:.3f} мм/год."
            )

    severity = {
        "halite": sev_halite,
        "calcite": sev_calcite,
        "wax": sev_wax,
        "corrosion": sev_corr,
    }

    # --- 6. интегральный риск ---
    integrated = sum(MECHANISM_WEIGHTS[k] * v for k, v in severity.items())
    dominant = max(severity, key=lambda k: severity[k] * MECHANISM_WEIGHTS[k])

    rec = _recommend(dominant, severity, onset, case, corr)

    return DiagnosisResult(
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
