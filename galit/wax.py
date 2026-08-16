"""АСПО: температура насыщения парафином и глубина начала отложений.

Ключевая физика (подтверждена промысловыми исследованиями):
глубина начала интенсивного образования АСПО соответствует ПЕРЕСЕЧЕНИЮ
кривой распределения температуры потока и кривой температуры насыщения
нефти парафином (WAT). Выше этой точки поток холоднее WAT -- парафин
кристаллизуется на стенке НКТ.

Это отличает физически осмысленную модель от слепого градиентного
бустинга: мы предсказываем не только ФАКТ осложнения, но и ГЛУБИНУ,
а глубина определяет выбор технологии (скребок достаёт не везде,
греющий кабель имеет ограниченную длину, растворитель закачивается
на конкретную отметку).

О ЗАВИСИМОСТИ WAT ОТ ДАВЛЕНИЯ
-----------------------------
Published корреляции WAT(P) с коэффициентами НЕ СУЩЕСТВУЕТ -- есть
только термодинамические SLE-модели и наборы экспериментальных точек.
Реализована кусочная модель по качественной картине, надёжно
подтверждённой в литературе:

  * ниже давления насыщения Pb: рост P растворяет лёгкие фракции
    обратно в нефть, они работают растворителем парафина -> WAT падает.
    dWAT/dP < 0, порядок -0.6 K/МПа (оценка по Fuel 2023: депрессия
    5.8 C при росте 0 -> 9.65 МПа для 20 wt% парафина).
  * выше Pb: только сжатие -> WAT растёт.
    dWAT/dP > 0, порядок +0.2 K/МПа (оценка по ряду опубликованных пар
    точек; в алмазной наковальне для растительных масел ~0.125 K/МПа).

Таким образом WAT МИНИМАЛЬНА вблизи Pb. Градиенты -- порядковые
оценки, а не published коэффициенты; в продуктиве они должны
калиброваться по лабораторным замерам заказчика (ДСК/ВТ-микроскопия).
"""
from __future__ import annotations

from dataclasses import dataclass

# Порядковые оценки градиентов, K/Па
DWAT_DP_BELOW_PB = -0.6e-6   # K/Па  (= -0.6 K/МПа)
DWAT_DP_ABOVE_PB = 0.2e-6    # K/Па  (= +0.2 K/МПа)


@dataclass
class WaxProperties:
    """Параметры парафинистости нефти.

    wat_stock_tank_c -- WAT дегазированной нефти при атмосферном давлении
                        (лабораторный замер, ДСК или вискозиметрия)
    wax_content_pct  -- содержание парафина, % масс.
    """
    wat_stock_tank_c: float
    wax_content_pct: float = 5.0
    pour_point_c: float | None = None


def wat_at_pressure(wax: WaxProperties, p_pa: float, pb_pa: float) -> float:
    """Температура насыщения парафином при давлении p, C.

    Кусочно-линейная модель с изломом в точке насыщения.
    WAT минимальна при Pb.
    """
    # WAT в точке насыщения: от stock tank идём вниз по мере
    # растворения газа (давление растёт 0 -> Pb)
    wat_at_pb = wax.wat_stock_tank_c + DWAT_DP_BELOW_PB * pb_pa

    if p_pa <= pb_pa:
        # ниже Pb: линейно от stock tank к wat_at_pb
        return wax.wat_stock_tank_c + DWAT_DP_BELOW_PB * p_pa
    # выше Pb: растёт от минимума
    return wat_at_pb + DWAT_DP_ABOVE_PB * (p_pa - pb_pa)


def wax_onset_depth(
    depths: list[float],
    temps: list[float],
    pressures: list[float],
    wax: WaxProperties,
    pb_pa: float,
) -> tuple[float | None, list[float]]:
    """Глубина начала отложений АСПО, м от устья.

    Ищет пересечение T_поток(z) и WAT(P(z)) сверху вниз.
    Возвращает (глубина или None, профиль WAT).

    None означает, что весь ствол либо горячее WAT (отложений нет),
    либо холоднее (отложения по всему стволу -- возвращается 0).
    """
    wat_profile = [wat_at_pressure(wax, p, pb_pa) for p in pressures]

    delta = [t - w for t, w in zip(temps, wat_profile)]

    # если на устье поток уже теплее WAT -- отложений нет нигде выше
    if delta[0] > 0:
        return None, wat_profile

    # ищем первую снизу вверх смену знака
    onset = None
    for i in range(len(delta) - 1, 0, -1):
        if delta[i] > 0 >= delta[i - 1]:
            # линейная интерполяция точки пересечения
            d0, d1 = delta[i - 1], delta[i]
            z0, z1 = depths[i - 1], depths[i]
            frac = -d0 / (d1 - d0) if abs(d1 - d0) > 1e-12 else 0.0
            onset = z0 + frac * (z1 - z0)
            break

    if onset is None and delta[-1] <= 0:
        # весь ствол холоднее WAT
        onset = 0.0
    return onset, wat_profile


def wax_deposition_severity(
    depths: list[float],
    temps: list[float],
    wat_profile: list[float],
    onset_depth: float | None,
    wax: WaxProperties,
) -> float:
    """Безразмерная интенсивность отложений, 0..1.

    Учитывает три фактора:
      * протяжённость зоны отложений (доля ствола выше onset);
      * средний температурный напор (T_wat - T_поток) в этой зоне --
        движущая сила кристаллизации;
      * содержание парафина в нефти.
    """
    if onset_depth is None:
        return 0.0
    total = depths[-1] if depths else 1.0
    zone_frac = max(0.0, min((total - onset_depth) / max(total, 1e-6), 1.0))

    driving = [
        max(w - t, 0.0)
        for z, t, w in zip(depths, temps, wat_profile)
        if z <= onset_depth
    ]
    mean_dt = sum(driving) / len(driving) if driving else 0.0

    # нормировки: напор 20 K и 10 % парафина принимаем за «сильное»
    dt_term = min(mean_dt / 20.0, 1.0)
    wax_term = min(wax.wax_content_pct / 10.0, 1.0)

    return min(zone_frac * (0.6 * dt_term + 0.4 * wax_term), 1.0)


def recommend_wax_treatment(onset_depth: float | None, severity: float,
                            max_scraper_depth_m: float = 1500.0) -> str:
    """Подбор технологии борьбы под глубину и интенсивность.

    Арсенал соответствует реально применяемому в НГДУ «Речицанефть»:
    скребки собственной конструкции (раздвижной, фрез-скребок на
    проволоке), скребки-центраторы на штангах для ШГН, растворители
    СГБ и КР-01 собственного производства, электропрогрев греющим
    кабелем (внедрён с декабря 2014).
    """
    if onset_depth is None or severity < 0.05:
        return "не требуется"
    if severity < 0.2:
        return "профилактика: скребкование по графику"
    if onset_depth <= max_scraper_depth_m:
        if severity < 0.5:
            return f"скребок (зона с {onset_depth:.0f} м) + КР-01 периодически"
        return f"греющий кабель до {onset_depth:.0f} м + КР-01"
    if severity < 0.5:
        return f"растворитель КР-01/СГБ, закачка на {onset_depth:.0f} м"
    return (
        f"глубокая зона ({onset_depth:.0f} м, вне досягаемости скребка): "
        f"растворитель СГБ + пересмотр режима насоса"
    )
