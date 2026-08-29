"""Генератор синтетического фонда скважин.

ЧЕСТНОЕ ПРЕДУПРЕЖДЕНИЕ
----------------------
Это НЕ данные Белоруснефти и не претендует ими быть. Это физически
непротиворечивый синтетический фонд, построенный по опубликованным
характеристикам месторождений Припятского прогиба:

  * глубины 2000-4100 м (скв. 333 Осташковичская -- 4001 м);
  * пластовые воды плотностью 1,18-1,25 г/см3, хлоркальциевый тип;
  * два вида солей: галит и карбонатная;
  * обводнённость 60-99 % на позднестадийном фонде;
  * способы эксплуатации: ЭЦН, ШГН, фонтан;
  * средний дебит ~9 т/сут (2,013 млн т на ~600 механизированных скв.).

Назначение -- показать работу пайплайна и ранжирование фонда
до получения реальных данных. При передаче данных заказчика
генератор заменяется загрузчиком без изменения остального кода.
"""
from __future__ import annotations

import random

from .integrated import WellCase
from .scale import WaterAnalysis
from .wax import WaxProperties
from .wellbore import FluidProperties, ProductionRate, ThermalParams, WellGeometry

FIELDS = [
    "Речицкое", "Вишанское", "Осташковичское", "Южно-Осташковичское",
    "Золотухинское", "Барсуковское", "Москвичевское",
]

# Условные центры учебных площадок. Координаты синтетические и нужны только
# для демонстрации интерфейса карты, а не для навигации или полевых работ.
FIELD_CENTERS = {
    "Речицкое": (52.42, 30.35), "Вишанское": (52.36, 30.22),
    "Осташковичское": (52.28, 29.98), "Южно-Осташковичское": (52.20, 30.02),
    "Золотухинское": (52.50, 29.92), "Барсуковское": (52.57, 30.12),
    "Москвичевское": (52.47, 30.55),
}

LIFT_TYPES = ["ЭЦН", "ЭЦН", "ЭЦН", "ШГН", "ШГН", "фонтан"]


def _brine_composition(density_g_cm3: float, rng: random.Random) -> dict[str, float]:
    """Ионный состав рассола хлоркальциевого типа по плотности.

    Пропорции характерны для глубоких рассолов Припятского прогиба:
    преобладание Na-Cl с высоким содержанием Ca, малой щёлочностью
    и низким сульфатом (сульфат-редукция).
    """
    # общая минерализация из плотности (эмпирическая связь)
    tds = (density_g_cm3 - 1.0) / 0.0007 * 1000.0  # мг/л
    tds *= rng.uniform(0.95, 1.05)

    f_na = rng.uniform(0.26, 0.32)
    f_ca = rng.uniform(0.055, 0.085)
    f_mg = rng.uniform(0.006, 0.012)
    f_k = rng.uniform(0.004, 0.010)
    f_cl = 1.0 - (f_na + f_ca + f_mg + f_k) - 0.002

    return {
        "Na": tds * f_na,
        "Ca": tds * f_ca,
        "Mg": tds * f_mg,
        "K": tds * f_k,
        "Cl": tds * f_cl,
        "HCO3": rng.uniform(40.0, 320.0),
        "SO4": rng.uniform(20.0, 600.0),
    }


def make_well(idx: int, rng: random.Random) -> WellCase:
    """Одна синтетическая скважина."""
    field_name = rng.choice(FIELDS)
    lift = rng.choice(LIFT_TYPES)

    depth = rng.uniform(2000.0, 4100.0)
    incl = rng.uniform(0.0, 35.0)
    tubing_id = rng.choice([0.050, 0.062, 0.062, 0.076])

    watercut = rng.uniform(0.55, 0.985)
    q_liquid = rng.uniform(15.0, 130.0)
    q_oil = q_liquid * (1.0 - watercut)
    q_water = q_liquid - q_oil

    # Плотность до 1,29 г/см3: промысловый факт -- отложения галита
    # связаны с добычей воды плотностью 1,2 г/см3 И ВЫШЕ. Верхний край
    # диапазона -- как раз тот фонд, ради которого продукт и делается.
    density = rng.uniform(1.17, 1.29)
    salinity = (density - 1.0) / 0.0007 * 1000.0

    geom = WellGeometry(depth_m=depth, tubing_id_m=tubing_id,
                        inclination_deg=incl)
    rate = ProductionRate(q_oil_m3d=q_oil, q_water_m3d=q_water,
                          gor_m3m3=rng.uniform(30.0, 140.0))
    fluid = FluidProperties(
        gamma_oil=rng.uniform(0.83, 0.89),
        gamma_gas=rng.uniform(0.68, 0.85),
        salinity_ppm=salinity,
    )
    thermal = ThermalParams(
        t_surface_c=8.0,
        geothermal_grad=rng.uniform(0.028, 0.038),
        u_to=rng.uniform(8.0, 25.0),
        production_days=rng.uniform(120.0, 2500.0),
    )
    water = WaterAnalysis(
        ions_mg_l=_brine_composition(density, rng),
        ph=rng.uniform(5.4, 6.8),
        t_c=40.0,
        p_pa=5e6,
    )
    wax = WaxProperties(
        wat_stock_tank_c=rng.uniform(22.0, 48.0),
        wax_content_pct=rng.uniform(1.5, 11.0),
    )

    return WellCase(
        name=f"{field_name} {rng.randint(100, 999)}",
        geometry=geom,
        rate=rate,
        fluid=fluid,
        thermal=thermal,
        water=water,
        wax=wax,
        # ТРЕБУЕТ УТОЧНЕНИЯ У ЗАКАЗЧИКА. Содержание CO2 в попутном газе
        # Припятского прогиба в открытых источниках не найдено. Диапазон
        # 0,2-2 % взят как типичный для нефтяного попутного газа.
        # Параметр критичен: скорость коррозии ~ fCO2^0.58, и завышенный
        # CO2 делает коррозию единственным "победителем" в ранжировании.
        co2_mol_frac=rng.uniform(0.002, 0.02),
        inhibitor_efficiency=rng.choice([0.0, 0.0, 0.0, 0.85, 0.9]),
        lift_type=lift,
        p_wellhead_pa=rng.uniform(0.8e6, 2.2e6),
        latitude=FIELD_CENTERS[field_name][0] + rng.uniform(-0.025, 0.025),
        longitude=FIELD_CENTERS[field_name][1] + rng.uniform(-0.04, 0.04),
        cluster=f"Куст {1 + idx % 6}",
        site=field_name,
    )


def make_fund(n: int = 40, seed: int = 20260806) -> list[WellCase]:
    """Синтетический фонд из n скважин с фиксированным seed."""
    rng = random.Random(seed)
    return [make_well(i, rng) for i in range(n)]
