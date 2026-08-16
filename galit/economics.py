"""Расчёт экономического эффекта с явными допущениями.

Принцип: НИ ОДНО число не зашивается молча. Каждый параметр имеет
источник ("промысловый факт", "оценка", "требует уточнения у заказчика")
и участвует в анализе чувствительности.

Причина такой строгости: внутренний фильтр Белоруснефти -- Совет
по цифровизации -- отбирает проекты по ожидаемому экономическому
эффекту. Один красивый ROI без допущений там не пройдёт, а таблица
чувствительности по двум неизвестным пройдёт.

Опорный промысловый факт для калибровки правдоподобия: НГДУ
"Речицанефть" за 10 лет сократило объём обработок теплоносителем
в 3,5 раза за счёт оптимизации, новых технологий и нефтехимии.
То есть заказчик уже умеет считать эффект именно в этой метрике
и относится к ней серьёзно.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Assumption:
    """Параметр расчёта с прослеживаемым источником."""
    name: str
    value: float
    unit: str
    source: str
    confidence: str        # "факт" | "оценка" | "уточнить"
    low: float | None = None
    high: float | None = None

    @property
    def range(self) -> tuple[float, float]:
        lo = self.low if self.low is not None else self.value * 0.5
        hi = self.high if self.high is not None else self.value * 2.0
        return lo, hi


# --------------------------------------------------------------------------
# Базовый набор допущений
# --------------------------------------------------------------------------

def default_assumptions() -> dict[str, Assumption]:
    """Допущения по умолчанию для НГДУ «Речицанефть».

    Значения, помеченные "уточнить", -- это ровно тот список вопросов,
    который нужно задать заказчику письмом. Сам факт, что мы знаем,
    ЧТО спрашивать, работает на нас.
    """
    return {
        "n_wells_mech": Assumption(
            "Механизированный фонд", 601, "скв.",
            "Фонд на 01.01.2008: 651 добывающая, из них 601 механизированных",
            "уточнить", 500, 700,
        ),
        "n_complicated": Assumption(
            "Доля осложнённого фонда", 0.45, "-",
            "Оценка: обводнённость 60-99 %, 4 механизма осложнений",
            "уточнить", 0.25, 0.65,
        ),
        "treatments_per_well_year": Assumption(
            "Обработок на скважину в год", 6.0, "1/год",
            "Оценка по практике периодических обработок растворителем",
            "уточнить", 3.0, 12.0,
        ),
        "cost_reagent": Assumption(
            "Реагент на обработку", 900.0, "BYN",
            "Оценка; СГБ/КР-01 собственного производства, дешевле закупных",
            "уточнить", 400.0, 2000.0,
        ),
        "cost_crew": Assumption(
            "Бригадо-выезд", 2500.0, "BYN",
            "Оценка стоимости выезда бригады с техникой",
            "уточнить", 1200.0, 5000.0,
        ),
        "downtime_hours": Assumption(
            "Простой на обработку", 8.0, "ч",
            "Оценка",
            "уточнить", 4.0, 24.0,
        ),
        "well_rate_t_day": Assumption(
            "Средний дебит по нефти", 9.2, "т/сут",
            "2,013 млн т / 601 механизированная скв. / 365 сут",
            "факт", 5.0, 15.0,
        ),
        "oil_price": Assumption(
            "Цена нефти", 1450.0, "BYN/т",
            "Оценка при ~60 USD/барр и курсе ~3,3 BYN/USD",
            "уточнить", 1000.0, 2000.0,
        ),
        "cost_tkrs": Assumption(
            "Стоимость ТКРС", 45000.0, "BYN",
            "Оценка текущего/капитального ремонта скважины",
            "уточнить", 25000.0, 90000.0,
        ),
        "tkrs_downtime_days": Assumption(
            "Простой при ТКРС", 5.0, "сут",
            "Оценка",
            "уточнить", 3.0, 12.0,
        ),
        "mtbf_days": Assumption(
            "Наработка на отказ", 400.0, "сут",
            "Публично доступна только цифра 209,5 сут (конец 1990-х). "
            "Текущее значение -- обязательный вопрос заказчику",
            "уточнить", 250.0, 600.0,
        ),
        # --- эффекты от внедрения ---
        "treatment_reduction": Assumption(
            "Сокращение числа обработок", 0.25, "-",
            "Консервативная цель: адресность вместо графика. "
            "Ориентир правдоподобия: НГДУ сократило обработки "
            "теплоносителем в 3,5 раза (то есть на 71 %) за 10 лет",
            "оценка", 0.10, 0.45,
        ),
        "failure_reduction": Assumption(
            "Снижение числа отказов", 0.08, "-",
            "Ориентир: PDO (Оман) -2,75 % на >400 ESP-скважинах; "
            "ADIPEC 2025 +145 сут наработки по предотвращённым случаям",
            "оценка", 0.03, 0.15,
        ),
        "rate_loss_recovered": Assumption(
            "Восстановленный дебит", 0.015, "-",
            "Отложения сужают сечение НКТ и вызывают утечки клапанных "
            "пар ШГН -> падение дебита. Оценка доли возврата",
            "оценка", 0.005, 0.03,
        ),
    }


# --------------------------------------------------------------------------
# Расчёт
# --------------------------------------------------------------------------

@dataclass
class EconomicResult:
    """Результат расчёта с разложением по источникам эффекта."""
    saving_treatments: float = 0.0
    saving_failures: float = 0.0
    saving_production: float = 0.0
    total: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)


def compute_effect(a: dict[str, Assumption]) -> EconomicResult:
    """Годовой экономический эффект, BYN/год.

    Три независимых канала:
      1) сокращение числа обработок (реагент + бригада + простой);
      2) снижение числа отказов (ТКРС + простой);
      3) восстановление дебита, потерянного из-за отложений.
    """
    v = {k: x.value for k, x in a.items()}

    n_compl = v["n_wells_mech"] * v["n_complicated"]
    hourly_loss = v["well_rate_t_day"] / 24.0 * v["oil_price"]

    # --- 1. обработки ---
    treatments = n_compl * v["treatments_per_well_year"]
    cost_one = (
        v["cost_reagent"]
        + v["cost_crew"]
        + v["downtime_hours"] * hourly_loss
    )
    saving_tr = treatments * v["treatment_reduction"] * cost_one

    # --- 2. отказы ---
    failures = v["n_wells_mech"] * 365.0 / max(v["mtbf_days"], 1.0)
    cost_fail = (
        v["cost_tkrs"]
        + v["tkrs_downtime_days"] * v["well_rate_t_day"] * v["oil_price"]
    )
    saving_fail = failures * v["failure_reduction"] * cost_fail

    # --- 3. дебит ---
    annual_oil_compl = n_compl * v["well_rate_t_day"] * 365.0
    saving_prod = annual_oil_compl * v["rate_loss_recovered"] * v["oil_price"]

    total = saving_tr + saving_fail + saving_prod
    return EconomicResult(
        saving_treatments=saving_tr,
        saving_failures=saving_fail,
        saving_production=saving_prod,
        total=total,
        breakdown={
            "обработки": saving_tr,
            "отказы": saving_fail,
            "дебит": saving_prod,
        },
        detail={
            "осложнённых скважин": n_compl,
            "обработок в год": treatments,
            "стоимость одной обработки": cost_one,
            "отказов в год": failures,
            "стоимость одного отказа": cost_fail,
            "предотвращено обработок": treatments * v["treatment_reduction"],
            "предотвращено отказов": failures * v["failure_reduction"],
            "доп. добыча, т/год": annual_oil_compl * v["rate_loss_recovered"],
        },
    )


def sensitivity(a: dict[str, Assumption], keys: list[str],
                n_steps: int = 5) -> dict[str, list[tuple[float, float]]]:
    """Одномерная чувствительность по каждому параметру из keys.

    Возвращает {параметр: [(значение, эффект), ...]}.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for key in keys:
        lo, hi = a[key].range
        pairs = []
        for i in range(n_steps):
            val = lo + (hi - lo) * i / (n_steps - 1)
            mod = dict(a)
            mod[key] = replace(a[key], value=val)
            pairs.append((val, compute_effect(mod).total))
        out[key] = pairs
    return out


def tornado(a: dict[str, Assumption]) -> list[tuple[str, float, float, float]]:
    """Ранжирование параметров по влиянию на итог (tornado-диаграмма).

    Возвращает [(имя, эффект_при_low, эффект_при_high, размах), ...]
    отсортированный по убыванию размаха.
    """
    base = compute_effect(a).total
    rows = []
    for key, asm in a.items():
        lo, hi = asm.range
        mod_lo = dict(a)
        mod_lo[key] = replace(asm, value=lo)
        mod_hi = dict(a)
        mod_hi[key] = replace(asm, value=hi)
        e_lo = compute_effect(mod_lo).total
        e_hi = compute_effect(mod_hi).total
        rows.append((asm.name, e_lo, e_hi, abs(e_hi - e_lo)))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows


def scenario_bounds(a: dict[str, Assumption]) -> dict[str, float]:
    """Консервативный / базовый / оптимистичный сценарии.

    Консервативный: все ЭФФЕКТЫ по нижней границе (не все параметры --
    иначе получится бессмысленная комбинация худшего фонда с худшими
    эффектами).
    """
    effect_keys = ["treatment_reduction", "failure_reduction",
                   "rate_loss_recovered"]
    cons = dict(a)
    opt = dict(a)
    for k in effect_keys:
        cons[k] = replace(a[k], value=a[k].range[0])
        opt[k] = replace(a[k], value=a[k].range[1])
    return {
        "консервативный": compute_effect(cons).total,
        "базовый": compute_effect(a).total,
        "оптимистичный": compute_effect(opt).total,
    }


def unknowns(a: dict[str, Assumption]) -> list[Assumption]:
    """Параметры, требующие уточнения у заказчика.

    Этот список -- готовый текст запроса. Отсортирован по влиянию
    на результат, чтобы спрашивать в первую очередь важное.
    """
    ranked = {name: span for name, _, _, span in tornado(a)}
    todo = [x for x in a.values() if x.confidence == "уточнить"]
    todo.sort(key=lambda x: ranked.get(x.name, 0.0), reverse=True)
    return todo
