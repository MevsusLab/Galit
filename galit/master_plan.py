"""Чистый доменный генератор «План мастера на сегодня».

Модуль не выполняет диагностику и не меняет физические модели. Он переводит
готовые :class:`DiagnosisResult` и исходные :class:`WellCase` в безопасный,
детерминированный план работ. План является инженерным screening/advisory:
любое промышленное воздействие требует проверки данных, наряда-допуска и
утверждённого регламента предприятия.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
import hashlib
import math
from typing import Iterable

from .integrated import DiagnosisResult, ScenarioInterval, WellCase


DOMINANT_LABELS = {
    "halite": "галит",
    "calcite": "кальцит",
    "wax": "АСПО",
    "corrosion": "CO2-коррозия",
}


@dataclass(frozen=True)
class MasterPlanPolicy:
    """Детерминированные пороги диспетчеризации, не новая модель риска."""

    policy_id: str = "galit-master-plan"
    version: str = "1.0"
    low_risk_cutoff: float = 0.10
    medium_risk: float = 0.35
    elevated_risk: float = 0.45
    high_risk: float = 0.60
    immediate_risk: float = 0.75

    def __post_init__(self) -> None:
        values = (
            self.low_risk_cutoff, self.medium_risk, self.elevated_risk,
            self.high_risk, self.immediate_risk,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Пороги плана должны быть конечными")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Пороги плана должны быть в диапазоне 0..1")
        if not self.medium_risk <= self.elevated_risk <= self.high_risk <= self.immediate_risk:
            raise ValueError("Пороги приоритета должны возрастать")
        if not self.policy_id or not self.version:
            raise ValueError("Политика плана требует id и version")


DEFAULT_MASTER_PLAN_POLICY = MasterPlanPolicy()


@dataclass(frozen=True)
class DiagnosedWell:
    """Связка исходного кейса с уже рассчитанным результатом диагностики."""

    case: WellCase
    diagnosis: DiagnosisResult

    def __post_init__(self) -> None:
        if _normalise_well(self.case.name) != _normalise_well(self.diagnosis.well):
            raise ValueError("Имена скважины в WellCase и DiagnosisResult не совпадают")


@dataclass(frozen=True)
class OilLossEstimate:
    """Грубая оценка добычи под риском, м3/сут, а не прогноз фактической потери."""

    lower_m3d: float | None
    central_m3d: float | None
    upper_m3d: float | None
    method: str
    limitations: str


@dataclass(frozen=True)
class MasterTask:
    id: str
    well: str
    priority: int
    level: str
    dominant: str
    dominant_label: str
    risk: float
    risk_interval: ScenarioInterval | None
    possible_oil_loss: OilLossEstimate
    response_deadline: str
    recommended_action: str
    pre_trip_checklist: tuple[str, ...]
    materials: tuple[str, ...]
    equipment: tuple[str, ...]
    priority_reasons: tuple[str, ...]
    quality_warnings: tuple[str, ...]
    production_ready: bool
    safe_to_act: bool


@dataclass(frozen=True)
class MasterPlanSummary:
    diagnosed_wells: int
    unique_wells: int
    task_count: int
    filtered_low_risk: int
    excluded_by_limit: int
    blocked_tasks: int
    production_ready_tasks: int
    possible_oil_loss_lower_m3d: float | None
    possible_oil_loss_central_m3d: float | None
    possible_oil_loss_upper_m3d: float | None
    counts_by_level: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MasterPlan:
    generated_at: datetime
    plan_date: date
    policy_id: str
    policy_version: str
    tasks: tuple[MasterTask, ...]
    summary: MasterPlanSummary
    advisory_notice: str = (
        "План предназначен для приоритизации. Промышленное воздействие — только "
        "после проверки данных, оценки рисков, наряда-допуска и утверждения ответственным лицом."
    )


def _normalise_well(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def _stable_id(well: str, dominant: str, policy: MasterPlanPolicy) -> str:
    identity = f"{_normalise_well(well)}|{dominant}|{policy.policy_id}|{policy.version}"
    return "mp-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _generated_at(value: date | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _risk_interval(result: DiagnosisResult) -> ScenarioInterval | None:
    return result.uncertainty.integrated_risk if result.uncertainty else None


def _bounded_risk(value: float | None, fallback: float) -> float:
    candidate = fallback if value is None or not math.isfinite(value) else value
    return min(1.0, max(0.0, candidate))


def _loss_estimate(case: WellCase, result: DiagnosisResult) -> OilLossEstimate:
    """Оценить добычу под риском без представления score как вероятности отказа."""
    rate = case.rate.q_oil_m3d
    if not math.isfinite(rate) or rate < 0.0:
        return OilLossEstimate(
            None, None, None,
            "Оценка недоступна: некорректный дебит нефти.",
            "Нужен подтверждённый неотрицательный дебит нефти.",
        )
    risk = _bounded_risk(result.integrated_risk, 0.0)
    interval = _risk_interval(result)
    if interval is not None:
        low = _bounded_risk(interval.p05, risk)
        mid = _bounded_risk(interval.p50, risk)
        high = _bounded_risk(interval.p95, risk)
        low, mid, high = sorted((low, mid, high))
        method = (
            "Дебит нефти × p05/p50/p95 сценарного диапазона интегрального risk score."
        )
    else:
        low, mid, high = max(0.0, risk * 0.5), risk, min(1.0, risk * 1.5)
        method = (
            "Дебит нефти × интегральный risk score; границы screening 0.5×..1.5× score."
        )
    return OilLossEstimate(
        round(rate * low, 1), round(rate * mid, 1), round(rate * high, 1),
        method,
        "Это диапазон добычи под риском для ранжирования, не вероятность отказа и не "
        "прогноз фактической потери. Точность ограничена качеством входов и отсутствием "
        "полевой калибровки связи risk score с потерями.",
    )


def _priority(risk: float, policy: MasterPlanPolicy) -> tuple[int, str, str]:
    if risk >= policy.immediate_risk:
        return 1, "критический", "немедленно"
    if risk >= policy.high_risk:
        return 2, "высокий", "24ч"
    if risk >= policy.elevated_risk:
        return 3, "повышенный", "48ч"
    if risk >= policy.medium_risk:
        return 4, "средний", "72ч"
    return 5, "низкий", "планово"


def _playbook(dominant: str) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    common = (
        "Подтвердить актуальные режимные данные и историю осложнений",
        "Проверить наряд-допуск, HSE-риски и утверждённую технологическую карту",
        "Согласовать окно работ и критерии остановки с ответственным лицом",
    )
    if dominant == "halite":
        return (
            "Подтвердить солеотложение пробой/анализом; подобрать промывку или ингибирование по утверждённому регламенту и реальным калибровочным данным.",
            common + ("Отобрать свежую пробу воды и проверить совместимость жидкостей",),
            ("Пробоотборная тара", "Промывочная жидкость — после проверки совместимости"),
            ("Комплект пробоотбора", "Средства контроля давления и расхода"),
        )
    if dominant == "calcite":
        return (
            "Подтвердить карбонатный состав отложений; выбрать механическую/химическую обработку только по утверждённому регламенту с оценкой коррозионного риска.",
            common + ("Проверить анализ воды, pH и риск несовместимости/коррозии",),
            ("Пробоотборная тара", "Реагент — только после лабораторного подбора"),
            ("Комплект пробоотбора", "Коррозионный контроль", "Средства контроля давления"),
        )
    if dominant == "wax":
        return (
            "Подтвердить АСПО и доступную глубину; выбрать очистку/тепловое воздействие или реагент по утверждённому регламенту без назначения непроверенной дозировки.",
            common + ("Проверить ограничения лифта и риск повреждения защитной плёнки",),
            ("Тара для пробы отложений", "Реагент — только после лабораторного подбора"),
            ("Средства контроля глубины и нагрузки", "Разрешённый комплект очистки"),
        )
    if dominant == "corrosion":
        return (
            "Провести подтверждающий коррозионный контроль и проверку герметичности; подбор ингибитора и режима — только по утверждённому регламенту и калибровочным данным.",
            common + ("Проверить последние замеры толщины, купоны и признаки утечки",),
            ("Коррозионные купоны/пробоотборная тара", "Ингибитор — только после утверждённого подбора"),
            ("Толщиномер", "Газоанализатор", "Средства контроля герметичности"),
        )
    return (
        "Уточнить механизм осложнения до выбора воздействия.", common,
        ("Пробоотборная тара",), ("Диагностический комплект",),
    )


def _quality_warnings(result: DiagnosisResult) -> tuple[str, ...]:
    warnings = list(result.quality.reasons)
    warnings.extend(result.warnings)
    if not result.quality.production_ready:
        warnings.insert(0, f"Данные не production-ready: grade {result.quality.grade}")
    return tuple(dict.fromkeys(warnings))


def _make_task(item: DiagnosedWell, policy: MasterPlanPolicy) -> MasterTask:
    result, case = item.diagnosis, item.case
    risk = _bounded_risk(result.integrated_risk, 0.0)
    priority, level, deadline = _priority(risk, policy)
    action, checklist, materials, equipment = _playbook(result.dominant)
    ready = bool(result.quality.production_ready)
    if not ready:
        action = (
            "Не выполнять воздействие: сначала закрыть предупреждения качества, "
            "получить фактические данные и повторить диагностику. " + action
        )
    contribution = result.severity.get(result.dominant)
    reasons = [f"Интегральный риск {risk:.2f} ({level})"]
    if contribution is not None:
        reasons.append(f"Доминирующий механизм {DOMINANT_LABELS.get(result.dominant, result.dominant)}: severity {contribution:.2f}")
    if not ready:
        reasons.append("Промышленное действие заблокировано качеством данных")
    return MasterTask(
        id=_stable_id(result.well, result.dominant, policy),
        well=result.well.strip(),
        priority=priority,
        level=level,
        dominant=result.dominant,
        dominant_label=DOMINANT_LABELS.get(result.dominant, result.dominant or "не определено"),
        risk=risk,
        risk_interval=_risk_interval(result),
        possible_oil_loss=_loss_estimate(case, result),
        response_deadline=deadline,
        recommended_action=action,
        pre_trip_checklist=checklist,
        materials=materials,
        equipment=equipment,
        priority_reasons=tuple(reasons),
        quality_warnings=_quality_warnings(result),
        production_ready=ready,
        safe_to_act=ready,
    )


def generate_master_plan(
    diagnosed_wells: Iterable[DiagnosedWell | tuple[WellCase, DiagnosisResult]],
    *,
    generated_at: date | datetime | None = None,
    include_low_risk: bool = False,
    limit: int | None = None,
    policy: MasterPlanPolicy = DEFAULT_MASTER_PLAN_POLICY,
) -> MasterPlan:
    """Сформировать отсортированный план, дедуплицируя имена скважин.

    При дубле (регистр/пробелы не учитываются) сохраняется результат с большим
    интегральным риском; при равенстве — лексикографически минимальный механизм.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit не может быть отрицательным")
    items = [item if isinstance(item, DiagnosedWell) else DiagnosedWell(*item) for item in diagnosed_wells]
    unique: dict[str, DiagnosedWell] = {}
    for item in items:
        key = _normalise_well(item.diagnosis.well)
        current = unique.get(key)
        candidate_key = (-_bounded_risk(item.diagnosis.integrated_risk, 0.0), item.diagnosis.dominant)
        current_key = None if current is None else (
            -_bounded_risk(current.diagnosis.integrated_risk, 0.0), current.diagnosis.dominant
        )
        if current is None or candidate_key < current_key:
            unique[key] = item

    tasks = [_make_task(item, policy) for item in unique.values()]
    filtered = 0
    if not include_low_risk:
        filtered = sum(task.risk < policy.low_risk_cutoff for task in tasks)
        tasks = [task for task in tasks if task.risk >= policy.low_risk_cutoff]
    tasks.sort(key=lambda task: (task.priority, -task.risk, _normalise_well(task.well), task.id))
    excluded = 0
    if limit is not None:
        excluded = max(0, len(tasks) - limit)
        tasks = tasks[:limit]

    available = [task.possible_oil_loss for task in tasks if task.possible_oil_loss.central_m3d is not None]
    def total(attribute: str) -> float | None:
        return round(sum(getattr(loss, attribute) for loss in available), 1) if available else None

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.level] = counts.get(task.level, 0) + 1
    stamp = _generated_at(generated_at)
    summary = MasterPlanSummary(
        diagnosed_wells=len(items), unique_wells=len(unique), task_count=len(tasks),
        filtered_low_risk=filtered, excluded_by_limit=excluded,
        blocked_tasks=sum(not task.safe_to_act for task in tasks),
        production_ready_tasks=sum(task.production_ready for task in tasks),
        possible_oil_loss_lower_m3d=total("lower_m3d"),
        possible_oil_loss_central_m3d=total("central_m3d"),
        possible_oil_loss_upper_m3d=total("upper_m3d"),
        counts_by_level=counts,
    )
    return MasterPlan(
        generated_at=stamp, plan_date=stamp.date(), policy_id=policy.policy_id,
        policy_version=policy.version, tasks=tuple(tasks), summary=summary,
    )


__all__ = [
    "DEFAULT_MASTER_PLAN_POLICY", "DiagnosedWell", "MasterPlan",
    "MasterPlanPolicy", "MasterPlanSummary", "MasterTask", "OilLossEstimate",
    "generate_master_plan",
]
