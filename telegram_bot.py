"""ГАЛИТ | Telegram-бот мастера добычи: экспресс-диагностика АСПО.

Альтернативный интерфейс к расчётному ядру galit для полевого персонала:
мастер отправляет параметры скважины одной командой и получает
глубину начала отложения АСПО и рекомендацию по химической обработке.

Запуск:
    set GALIT_BOT_TOKEN=123456:ABC...   (токен от @BotFather)
    python telegram_bot.py

Примеры команды /aspo:
    /aspo 3200 62 8 72 65 34
        позиционно: глубина, м · НКТ, мм · дебит нефти, м3/сут ·
        дебит воды, м3/сут · газовый фактор, м3/м3 · WAT, °C
    /aspo 3200 62 8 72 65 34 способ=ШГН парафин=6.5
        то же + именованные параметры (список -- /help)

Ответ строго структурирован, без эмодзи и украшательств.
"""
from __future__ import annotations

import asyncio
import csv
import html
import io
import logging
import math
from collections import OrderedDict, deque
import os
import shlex
import sys
import unicodedata
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from dotenv import load_dotenv

import galit
from galit import (
    DataProvenance,
    DiagnosedWell,
    WellCase,
    diagnose,
    forecast_well,
    generate_master_plan,
)
from galit.wax import recommend_wax_treatment
from galit.wellbore import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
)
from galit.scale import WaterAnalysis
from galit.wax import WaxProperties

# Секреты загружаются только при запуске main(), но не при импорте в тестах.
BOT_TOKEN = ""

# ==========================================================================
# Разбор параметров команды (чистые функции -- тестируются pytest)
# ==========================================================================

# Именованные параметры: алиасы (RU/EN, регистр не важен) -> ключ.
KEY_ALIASES = {
    "глубина": "depth_m", "depth": "depth_m", "h": "depth_m",
    "нкт": "tubing_mm", "tubing": "tubing_mm", "d": "tubing_mm",
    "нефть": "q_oil_m3d", "qoil": "q_oil_m3d", "q_oil": "q_oil_m3d",
    "вода": "q_water_m3d", "qwater": "q_water_m3d", "q_water": "q_water_m3d",
    "гф": "gor_m3m3", "gor": "gor_m3m3",
    "ват": "wat_c", "wat": "wat_c",
    "температура": "t_surface_c", "t_surface": "t_surface_c",
    "градиент": "geothermal_grad", "grad": "geothermal_grad",
    "парафин": "wax_pct", "wax_pct": "wax_pct",
    "способ": "lift_type", "lift": "lift_type",
    "co2": "co2_mol_frac",
    "буферное": "p_wellhead_mpa", "wellhead": "p_wellhead_mpa",
    "скважина": "name", "well": "name", "name": "name",
    "широта": "latitude", "lat": "latitude",
    "долгота": "longitude", "lon": "longitude", "lng": "longitude",
}

# Позиционные аргументы /aspo в порядке следования
POSITIONAL_KEYS = ["depth_m", "tubing_mm", "q_oil_m3d",
                   "q_water_m3d", "gor_m3m3", "wat_c"]

LIFT_TYPES = {"эцн": "ЭЦН", "шгн": "ШГН", "фонтан": "фонтан", "esp": "ЭЦН"}

# Диапазоны физической допустимости: ключ -> (мин, макс, подпись для сообщений)
RANGES: dict[str, tuple[float, float, str]] = {
    "depth_m": (100.0, 8000.0, "глубина, м (100–8000)"),
    "tubing_mm": (20.0, 200.0, "внутренний диаметр НКТ, мм (20–200)"),
    "q_oil_m3d": (0.0, 1000.0, "дебит нефти, м3/сут (0–1000)"),
    "q_water_m3d": (0.0, 1000.0, "дебит воды, м3/сут (0–1000)"),
    "gor_m3m3": (0.0, 2000.0, "газовый фактор, м3/м3 (0–2000)"),
    "wat_c": (-20.0, 90.0, "WAT, °C (от -20 до 90)"),
    "t_surface_c": (-10.0, 40.0, "температура у поверхности, °C"),
    "geothermal_grad": (0.01, 0.1, "геотермический градиент, К/м"),
    "wax_pct": (0.0, 20.0, "содержание парафина, % масс."),
    "co2_mol_frac": (0.0, 0.5, "доля CO2 (0–0.5)"),
    "p_wellhead_mpa": (0.2, 25.0, "буферное давление, МПа"),
    "latitude": (-90.0, 90.0, "широта WGS84"),
    "longitude": (-180.0, 180.0, "долгота WGS84"),
}

# Значения по умолчанию, отсутствующие в ядре WellCase
DEFAULTS: dict[str, float | str] = {
    "t_surface_c": 8.0,
    "geothermal_grad": 0.033,
    "wax_pct": 5.0,
    "co2_mol_frac": 0.02,
    "p_wellhead_mpa": 1.2,
    "lift_type": "ЭЦН",
}

# Типовой солевой состав пластовой воды Припятского прогиба (мг/л).
# На глубину начала АСПО и подбор обработки не влияет -- нужен только
# для расчёта сопутствующих механизмов (соли, коррозия).
TYPICAL_BRINE = {
    "Na": 95_000.0, "Cl": 205_000.0, "Ca": 28_000.0, "Mg": 3_100.0,
    "K": 1_800.0, "HCO3": 130.0, "SO4": 250.0,
}

REQUIRED_KEYS = ["depth_m", "tubing_mm", "q_oil_m3d",
                 "q_water_m3d", "gor_m3m3", "wat_c"]


class RecentDiagnosisStore:
    """Bounded in-memory per-chat history; stores only domain objects for this process."""

    def __init__(self, per_chat_limit: int = 20, chat_limit: int = 500):
        self.per_chat_limit = per_chat_limit
        self.chat_limit = chat_limit
        self._chats: OrderedDict[int, deque[DiagnosedWell]] = OrderedDict()

    def add(self, chat_id: int, item: DiagnosedWell) -> None:
        history = self._chats.setdefault(chat_id, deque(maxlen=self.per_chat_limit))
        history.append(item)
        self._chats.move_to_end(chat_id)
        while len(self._chats) > self.chat_limit:
            self._chats.popitem(last=False)

    def get(self, chat_id: int) -> list[DiagnosedWell]:
        return list(self._chats.get(chat_id, ()))

    def clear(self, chat_id: int) -> int:
        history = self._chats.pop(chat_id, ())
        return len(history)


RECENT_DIAGNOSES = RecentDiagnosisStore()
TELEGRAM_TEXT_LIMIT = 4096
TREATMENTS = galit.TreatmentRepository(os.environ.get("GALIT_TREATMENT_STORAGE", "data/treatments.json"))
PASSPORTS = galit.PassportRepository(os.environ.get("GALIT_PASSPORT_STORE", "data/well_passports.json"))
EQUIPMENT = galit.EquipmentRepository(os.environ.get("GALIT_EQUIPMENT_STORAGE", "data/equipment.json"))
WATERCUT = galit.WatercutRepository(os.environ.get("GALIT_WATERCUT_STORAGE", "data/watercut.json"))
TWIN_EVENTS = galit.ManualEventRepository(os.environ.get("GALIT_TWIN_EVENT_STORAGE", "data/digital_twin_events.json"))
SMART_MAP = galit.SmartMapRepository(os.environ.get("GALIT_SMART_MAP_STORAGE", "data/smart_map.json"))
CHEMICALS = galit.ChemicalRepository(os.environ.get("GALIT_CHEMICAL_STORAGE", "data/chemicals.json"))


def smart_map_service() -> galit.SmartMapService:
    return galit.SmartMapService(SMART_MAP)


def twin_service() -> galit.DigitalTwinService:
    return galit.build_default_service(watercut=WATERCUT, equipment=EQUIPMENT,
        treatments=TREATMENTS, passports=PASSPORTS, manual=TWIN_EVENTS)


PASSPORT_ALIASES = {
    "well": "well", "скважина": "well", "type": "event_type", "тип": "event_type",
    "title": "title", "заголовок": "title", "text": "text", "текст": "text",
    "oil": "oil_rate_m3d", "нефть": "oil_rate_m3d",
    "water": "water_rate_m3d", "вода": "water_rate_m3d",
    "gas": "gas_rate_m3d", "газ": "gas_rate_m3d", "limit": "limit", "лимит": "limit",
}


def parse_passport_command(text: str) -> tuple[dict[str, str], list[str]]:
    try:
        tokens = shlex.split(text, posix=True)[1:]
    except ValueError as exc:
        return {}, ["ошибка кавычек: " + str(exc)]
    values: dict[str, str] = {}
    errors: list[str] = []
    for token in tokens:
        raw_key, separator, raw = token.partition("=")
        key = PASSPORT_ALIASES.get(raw_key.casefold())
        if not separator or key is None or not raw.strip():
            errors.append(f"неверный параметр: {token}")
        elif key in values:
            errors.append(f"параметр задан повторно: {raw_key}")
        else:
            values[key] = raw.strip()
    return values, errors


def format_passport_summary(well: str, events: list[galit.PassportEvent],
                            treatments: list[galit.TreatmentRecord]) -> list[str]:
    summary = galit.passport_summary(events, treatments)
    latest_rate = summary["latest_rate"] or {}
    latest_risk = summary["latest_risk"] or {}
    blocks = [
        f"<b>Цифровой паспорт · {html.escape(well)}</b>",
        f"Событий: {summary['event_count']} · обработок: {summary['treatment_count']} · оценено: {summary['assessed_treatments']}",
        "Последний дебит: " + (f"нефть {latest_rate.get('oil_rate_m3d', '—')} · вода {latest_rate.get('water_rate_m3d', '—')} м³/сут" if latest_rate else "—"),
        "Последний риск: " + (f"{latest_risk.get('integrated_risk', '—')}" if latest_risk else "—"),
    ]
    return _forecast_chunks(blocks)


def format_passport_history(events: list[galit.PassportEvent],
                            treatments: list[galit.TreatmentRecord]) -> list[str]:
    rows = galit.passport_timeline(events, treatments)
    if not rows:
        return ["История паспорта пуста."]
    blocks = ["<b>История паспорта</b>"]
    for row in rows:
        blocks.append(f"{html.escape(row['event_at'][:10])} · {html.escape(row['event_type'])}\n{html.escape(row['title'])}")
    return _forecast_chunks(blocks)


TREATMENT_ALIASES = {
    "well": "well", "скважина": "well", "event": "complication_type",
    "complication": "complication_type", "осложнение": "complication_type",
    "description": "description", "описание": "description",
    "reagent": "reagent", "реагент": "reagent", "a": "reagent_a", "b": "reagent_b",
    "dose": "dosage", "доза": "dosage", "unit": "dosage_unit", "ед": "dosage_unit",
    "cost": "cost", "стоимость": "cost", "currency": "currency", "валюта": "currency",
    "type": "treatment_type", "обработка": "treatment_type",
    "expected": "expected_result", "ожидание": "expected_result", "id": "id",
    "revision": "revision", "ревизия": "revision", "status": "status", "статус": "status",
    "result": "actual_result", "actual_result": "actual_result", "результат": "actual_result",
    "metric": "metric", "показатель": "metric", "value": "metric_value",
    "значение": "metric_value", "success": "success", "успех": "success",
    "days": "effect_duration_days", "duration": "effect_duration_days", "дни": "effect_duration_days",
    "recurrence": "recurrence", "повтор": "recurrence", "recurrence_date": "recurrence_date",
    "дата_повтора": "recurrence_date", "group": "well_group", "well_group": "well_group",
    "группа": "well_group", "field": "field_name", "месторождение": "field_name",
    "cluster": "cluster", "куст": "cluster", "site": "site", "участок": "site",
    "before": "rate_before_m3_day", "дебит_до": "rate_before_m3_day",
    "after": "rate_after_m3_day", "дебит_после": "rate_after_m3_day",
    "limit": "limit", "лимит": "limit", "min_n": "min_sample_size",
}


def treatment_tokens(text: str) -> tuple[list[str], list[str]]:
    """Tokenize a command with shell-style quoted key=value values."""
    try:
        return shlex.split(text, posix=True)[1:], []
    except ValueError as exc:
        return [], ["ошибка кавычек: " + html.escape(str(exc))]


def parse_treatment_args(args: list[str], required: set[str]) -> tuple[dict[str, str], list[str]]:
    """Parse RU/EN key=value arguments; values may contain spaces when quoted."""
    values: dict[str, str] = {}
    errors: list[str] = []
    for token in args:
        raw_key, separator, raw = token.partition("=")
        key = TREATMENT_ALIASES.get(raw_key.casefold())
        if not separator or key is None or not raw.strip():
            errors.append(f"неверный параметр: {html.escape(token)}")
        elif key in values:
            errors.append(f"параметр задан повторно: {html.escape(raw_key)}")
        else:
            values[key] = raw.strip()
    missing = sorted(required - values.keys())
    if missing:
        errors.append("не заданы: " + ", ".join(missing))
    return values, errors


def parse_treatment_command(text: str, required: set[str] | None = None) -> tuple[dict[str, str], list[str]]:
    tokens, errors = treatment_tokens(text)
    if errors:
        return {}, errors
    return parse_treatment_args(tokens, required or set())


def parse_bool(value: str, field: str) -> bool:
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "да", "y"}:
        return True
    if normalized in {"0", "false", "no", "нет", "n"}:
        return False
    raise ValueError(f"{field}: используйте да/нет или true/false")


def parse_utc_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_treatment(repo: galit.TreatmentRepository, record_id: str) -> galit.TreatmentRecord:
    """Allow an unambiguous ID prefix from list output without hiding ambiguity."""
    try:
        return repo.get(record_id)
    except galit.TreatmentNotFoundError:
        matches = [item for item in repo.list(include_archived=True) if item.id.startswith(record_id)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise galit.TreatmentConflictError("ID неоднозначен; укажите больше символов")
        raise


def _effect_label(classification: str) -> str:
    return {"effective": "эффективна", "limited_effect": "ограниченный эффект",
            "ineffective": "неэффективна", "insufficient_data": "недостаточно данных"}.get(
                classification, classification)


def _format_effect(item: galit.TreatmentRecord) -> str:
    effect = galit.treatment_effect(item)
    delta = effect["rate_change_m3_day"]
    percent = effect["rate_change_percent"]
    change = "недостаточно данных"
    if delta is not None:
        change = f"{delta:+g} м³/сут" + ("" if percent is None else f" ({percent:+.1f}%)")
    return (f"Оценка: <b>{html.escape(_effect_label(str(effect['classification'])))}</b>\n"
            f"Изменение дебита: {change}\n{html.escape(str(effect['explanation']))}")


def format_treatment_card(item: galit.TreatmentRecord) -> list[str]:
    e = html.escape
    metrics = ", ".join(f"{e(k)}={v:g}" for k, v in item.result_metrics.items()) or "—"
    result = e(item.actual_result) if item.actual_result else "—"
    recurrence = "—" if item.recurrence is None else ("да" if item.recurrence else "нет")
    if item.recurrence_date:
        recurrence += f" ({item.recurrence_date.date().isoformat()})"
    before = "—" if item.rate_before_m3_day is None else f"{item.rate_before_m3_day:g} м³/сут"
    after = "—" if item.rate_after_m3_day is None else f"{item.rate_after_m3_day:g} м³/сут"
    blocks = [
        f"<b>Мероприятие {e(item.id[:8])}</b>",
        f"Скважина: {e(item.well_name)}\nОсложнение: {e(item.complication_type)}\n"
        f"Группа: {e(item.well_group or '—')}\nСтатус: <b>{item.status.value}</b> · revision={item.revision}",
        f"Описание: {e(item.description)}\nРеагент: {e(item.reagent_name)}\n"
        f"Доза: {item.dosage:g} {e(item.dosage_unit)}\nСтоимость: {item.cost:g} {e(item.currency)}",
        f"Дебит до: {before}\nДебит после: {after}\n{_format_effect(item)}",
        f"Ожидание: {e(item.expected_result or '—')}\nФакт: {result}\nМетрики: {metrics}\n"
        f"Успех: {'—' if item.success is None else ('да' if item.success else 'нет')}\n"
        f"Длительность: {'—' if item.effect_duration_days is None else f'{item.effect_duration_days:g} сут'}\n"
        f"Повтор: {recurrence}",
    ]
    return _forecast_chunks(blocks)


def format_treatments(records: list[galit.TreatmentRecord]) -> list[str]:
    if not records:
        return ["Записей журнала не найдено."]
    blocks = ["<b>Мероприятия</b>"]
    for item in records:
        effect = galit.treatment_effect(item)
        delta = effect["rate_change_m3_day"]
        change = "дебит: недостаточно данных" if delta is None else f"Δ дебита {delta:+g} м³/сут"
        blocks.append(f"<code>{html.escape(item.id[:8])}</code> · {html.escape(item.well_name)} · "
                      f"{html.escape(item.reagent_name)} · {item.status.value} · rev={item.revision}\n"
                      f"{html.escape(_effect_label(str(effect['classification'])))} · {change}")
    return _forecast_chunks(blocks)


def format_treatment_comparison(result: dict[str, object]) -> list[str]:
    e = html.escape
    comp = result["comparability"]
    a, b = result["reagent_a"], result["reagent_b"]
    def shown(row: dict[str, object]) -> str:
        value = row["value"]
        rendered = "—" if value is None else (f"{value:.0%}" if result["metric"] == "success_rate" else f"{value:g} сут")
        return f"{e(str(row['name']))}: n={row['n']}, значение={rendered}"
    uplift = result["relative_uplift"]
    blocks = [
        "<b>A/B-сравнение реагентов</b>",
        f"Когорта: complication={e(str(comp['complication_type']))}; well_group={e(str(comp['well_group']))}",
        shown(a), shown(b),
        f"uplift B к A: {'—' if uplift is None else f'{uplift:+.1%}'}\n"
        f"confidence: {e(str(result['confidence']))}\nstatus: {e(str(result['status']))}",
    ]
    if result.get("reason"):
        blocks.append("Недостаточно данных: " + e(str(result["reason"])))
    blocks.append(e(str(result["warning"])))
    return _forecast_chunks(blocks)


def format_treatment_stats(records: list[galit.TreatmentRecord],
                           min_sample_size: int = galit.DEFAULT_MIN_SAMPLE_SIZE) -> list[str]:
    summary = galit.treatment_summary(records, "reagent")
    analytics = galit.treatment_analytics(records, min_sample_size=min_sample_size)
    blocks = ["<b>Контроль обработок</b>",
              f"Записей: {analytics['records']} · с измеримым эффектом: {analytics['assessed_effects']} · min_n={min_sample_size}"]
    if summary["status"] == "insufficient_data":
        blocks.append("Эффективность реагентов: <b>недостаточно данных</b>.")
    for row in summary["groups"]:
        success = "—" if row["success_rate"] is None else f"{row['success_rate']:.0%}"
        costs = ", ".join(f"{currency} {values['total']:g}" for currency, values in row["costs_by_currency"].items()) or "—"
        blocks.append(f"{html.escape(row['group'])}: n={row['assessed_observations']}; "
                      f"успех={success}; confidence={row['confidence']}; стоимость по валютам: {costs}")
    ineffective = analytics["ineffective_treatment_ids"]
    excessive = analytics["potentially_excessive"]
    blocks.append("Неэффективные: " + (", ".join(html.escape(value[:8]) for value in ineffective) if ineffective else "не выявлены"))
    blocks.append("Потенциально избыточные: " + (", ".join(html.escape(row["treatment_id"][:8]) for row in excessive) if excessive else "не выявлены"))
    blocks.append("<b>Сопоставимые реагенты/технологии</b>")
    comparisons = analytics["comparisons"]
    if not comparisons:
        blocks.append("Недостаточно данных: нет записей с дебитом до и после.")
    for row in comparisons:
        context = (f"{html.escape(row['field_name'])} · {html.escape(row['complication_type'])} · "
                   f"{html.escape(row['treatment_type'])} · {html.escape(row['reagent_name'])}")
        if row["status"] == "available":
            blocks.append(f"{context}: n={row['n']}; median Δ={row['median_rate_change_percent']:+.1f}%")
        else:
            blocks.append(f"{context}: n={row['n']}; <b>недостаточно данных</b> (нужно {min_sample_size})")
    blocks.append("<b>Оптимальный median-интервал</b>")
    intervals = analytics["interval_recommendations"]
    if not intervals:
        blocks.append("Недостаточно данных: интервалы повторов не зарегистрированы.")
    for row in intervals:
        context = f"{html.escape(row['field_name'])} · {html.escape(row['complication_type'])}"
        if row["status"] == "available":
            blocks.append(f"{context}: n={row['n']}; median={row['median_interval_days']:g} сут")
        else:
            blocks.append(f"{context}: n={row['n']}; <b>недостаточно данных</b> (нужно {min_sample_size})")
    blocks.append(html.escape(str(analytics["warning"])))
    return _forecast_chunks(blocks)


def _normalized_well_name(value: str) -> str:
    """Unicode/case/whitespace-insensitive well name used only for lookup."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def select_forecast_diagnosis(items: list[DiagnosedWell], query: str = "") -> DiagnosedWell:
    """Select latest diagnosis or a unique normalized exact well-name match."""
    if not items:
        raise LookupError("История пуста. Сначала рассчитайте скважину через /aspo или пошаговый ввод.")
    normalized = _normalized_well_name(query)
    if not normalized:
        return items[-1]
    matches = [item for item in items if _normalized_well_name(item.case.name) == normalized]
    if not matches:
        raise LookupError(f"Скважина «{query.strip()}» не найдена в истории этого чата.")
    names = {item.case.name for item in matches}
    if len(names) > 1:
        raise LookupError("Имя скважины неоднозначно после нормализации: " + ", ".join(sorted(names)))
    return matches[-1]


def _forecast_chunks(blocks: list[str]) -> list[str]:
    """Join escaped/self-contained blocks and split oversized blocks at safe boundaries."""
    limit = TELEGRAM_TEXT_LIMIT - 100
    normalized: list[str] = []
    for block in blocks:
        remaining = block
        while len(remaining) > limit:
            cut = remaining.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = remaining.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            normalized.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            normalized.append(remaining)
    chunks: list[str] = []
    current = ""
    for block in normalized:
        candidate = block if not current else current + "\n\n" + block
        if len(candidate) > limit and current:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def format_forecast_messages(item: DiagnosedWell) -> list[str]:
    """Compact honest forecast: no as_of/history means no invented calendar dates."""
    forecast = forecast_well(item.diagnosis, item.case)
    blocks = [
        f"<b>Прогноз во времени · {html.escape(forecast.well)}</b>",
        "Временная история не передана: дата появляется только при реальном расчётном окне.",
    ]
    for event in forecast.events:
        if event.horizon_start_date is not None and event.horizon_end_date is not None:
            if event.horizon_start_date == event.horizon_end_date:
                date_text = event.horizon_start_date.isoformat()
            else:
                date_text = f"{event.horizon_start_date.isoformat()}–{event.horizon_end_date.isoformat()}"
        elif event.horizon_start_days is not None and event.horizon_end_days is not None:
            date_text = f"окно {event.horizon_start_days:.0f}–{event.horizon_end_days:.0f} сут.; календарная дата недоступна"
        else:
            date_text = "дата недоступна"
        if event.probability is not None:
            likelihood = f"вероятность {event.probability:.0%}"
        elif event.risk_band is not None:
            likelihood = f"likelihood {event.likelihood.value}; risk band {event.risk_band[0]:.2f}–{event.risk_band[1]:.2f}"
        else:
            likelihood = f"likelihood {event.likelihood.value}; risk band недоступен"
        required = ", ".join(event.required_inputs) if event.required_inputs else "нет"
        blocks.append(
            f"<b>{html.escape(event.title)}</b>\n"
            f"status: {event.status.value}\n"
            f"{html.escape(date_text)}\n"
            f"{html.escape(likelihood)}\n"
            f"Основание: {html.escape(event.basis)}\n"
            f"Нужно: {html.escape(required)}"
        )
    blocks.append("Оценка не заменяет промысловые исследования и инженерную проверку; screening-окна не являются гарантированными датами отказа.")
    return _forecast_chunks(blocks)


def parse_economics_args(args: list[str]) -> tuple[dict[str, float | str], list[str]]:
    """Parse explicit /economics inputs; no monetary defaults are applied."""
    aliases = {
        "вероятность": "event_probability", "probability": "event_probability",
        "горизонт": "horizon_days", "horizon": "horizon_days",
        "эффективность": "treatment_efficiency", "efficiency": "treatment_efficiency",
        "простой_события": "event_downtime_days", "event_days": "event_downtime_days",
        "простой_обработки": "treatment_downtime_days", "treatment_days": "treatment_downtime_days",
        "цена": "product_price_per_m3", "price": "product_price_per_m3",
        "оперпотери": "operating_loss_per_day", "operating": "operating_loss_per_day",
        "стоимость": "treatment_cost", "cost": "treatment_cost",
        "валюта": "currency", "currency": "currency",
        "доля_потери": "production_loss_fraction", "loss_fraction": "production_loss_fraction",
        "скважина": "well", "well": "well",
    }
    values: dict[str, float | str] = {}
    errors: list[str] = []
    for token in args:
        key_raw, separator, raw = token.partition("=")
        key = aliases.get(key_raw.casefold())
        if not separator or key is None:
            errors.append(f"неизвестный параметр: {html.escape(token)}")
            continue
        if key in {"currency", "well"}:
            if not raw.strip():
                errors.append(f"{html.escape(key_raw)}: пустое значение")
            else:
                values[key] = raw.strip()
            continue
        value = _to_float(raw)
        if value is None:
            errors.append(f"{html.escape(key_raw)}: не число")
        elif value < 0:
            errors.append(f"{html.escape(key_raw)}: значение не может быть отрицательным")
        else:
            values[key] = value
    required = ("event_probability", "horizon_days", "treatment_efficiency",
                "event_downtime_days", "treatment_downtime_days", "product_price_per_m3",
                "operating_loss_per_day", "treatment_cost", "currency")
    missing = [name for name in required if name not in values]
    if missing:
        errors.append("не заданы обязательные экономические входы: " + ", ".join(missing))
    for name in ("event_probability", "treatment_efficiency", "production_loss_fraction"):
        if name in values and float(values[name]) > 1:
            errors.append(f"{name}: требуется значение 0…1")
    if "horizon_days" in values and float(values["horizon_days"]) <= 0:
        errors.append("horizon_days должен быть больше нуля")
    return values, errors


def format_economics_messages(item: DiagnosedWell,
                              values: dict[str, float | str]) -> list[str]:
    """Render an escaped, bounded report from explicit single-currency inputs."""
    result = galit.calculate_risk_economics(galit.RiskEconomicsInput(
        event_probability=float(values["event_probability"]),
        horizon_days=float(values["horizon_days"]),
        treatment_efficiency=float(values["treatment_efficiency"]),
        event_downtime_days=float(values["event_downtime_days"]),
        treatment_downtime_days=float(values["treatment_downtime_days"]),
        oil_rate_m3_day=item.case.rate.q_oil_m3d,
        product_price_per_m3=float(values["product_price_per_m3"]),
        operating_loss_per_day=float(values["operating_loss_per_day"]),
        treatment_cost=float(values["treatment_cost"]),
        currency=str(values["currency"]),
        production_loss_fraction=float(values.get("production_loss_fraction", 1.0)),
        probability_source="telegram_explicit_input",
    ))
    b = result.breakdown
    currency = html.escape(result.currency or "—")
    def money(value: float | None) -> str:
        return "недоступно" if value is None else f"{value:,.2f} {currency}".replace(",", " ")
    blocks = [
        f"<b>Экономика риска · {html.escape(item.case.name)}</b>",
        f"Статус: {result.status.value} · валюта: {currency}",
        f"Ожидаемая потеря добычи: {'недоступно' if b.expected_production_loss_m3 is None else f'{b.expected_production_loss_m3:.2f} м³'}\n"
        f"В деньгах: {money(b.expected_production_loss_money)}\n"
        f"Ожидаемая стоимость простоя: {money(b.expected_event_downtime_cost)}",
        f"Обработка: {money(b.recommended_treatment_cost)}\n"
        f"Простой обработки: {money(b.treatment_downtime_cost)}\n"
        f"Полная стоимость мероприятия: {money(b.total_treatment_cost)}",
        f"Возможный предотвращённый ущерб: {money(b.potential_avoided_damage)}\n"
        f"Чистый ожидаемый эффект: {money(b.net_expected_effect)}\n"
        f"ROI: {'недоступно' if b.roi_ratio is None else f'{b.roi_ratio:.2f}'} · "
        f"payback ratio: {'недоступно' if b.payback_ratio is None else f'{b.payback_ratio:.2f}'}",
        "Допущения: постоянные дебит/цена на горизонте; эффективность — доля предотвращаемого ущерба; конвертация валют не выполняется.",
    ]
    return _forecast_chunks(blocks)


def parse_scenario_args(args: list[str]) -> tuple[dict[str, float | str | bool], list[str]]:
    """Parse /scenario key=value changes; economic values remain optional."""
    aliases = {
        "нефть_%": "oil_rate_relative_change", "oil_pct": "oil_rate_relative_change",
        "нефть": "oil_rate_delta_m3_day", "oil_delta": "oil_rate_delta_m3_day",
        "вода_%": "water_rate_relative_change", "water_pct": "water_rate_relative_change",
        "вода": "water_rate_delta_m3_day", "water_delta": "water_rate_delta_m3_day",
        "давление": "wellhead_pressure_delta_pa", "pressure_pa": "wellhead_pressure_delta_pa",
        "температура": "surface_temperature_delta_c", "temperature": "surface_temperature_delta_c",
        "ингибитор": "inhibitor_dosage_delta_mg_l", "inhibitor": "inhibitor_dosage_delta_mg_l",
        "промывка": "wash_treatment", "wash": "wash_treatment",
        "режим": "operating_mode", "mode": "operating_mode",
        "эффект_ингибитора": "inhibitor_efficiency", "inhibitor_effect": "inhibitor_efficiency",
        "источник": "source", "source": "source", "скважина": "well", "well": "well",
        "вероятность": "event_probability", "probability": "event_probability",
        "горизонт": "horizon_days", "horizon": "horizon_days",
        "эффективность": "treatment_efficiency", "efficiency": "treatment_efficiency",
        "цена": "product_price_per_m3", "price": "product_price_per_m3",
        "оперпотери": "operating_loss_per_day", "operating": "operating_loss_per_day",
        "стоимость": "treatment_cost", "cost": "treatment_cost",
        "валюта": "currency", "currency": "currency",
    }
    values: dict[str, float | str | bool] = {}
    errors: list[str] = []
    for token in args:
        raw_key, sep, raw = token.partition("=")
        key = aliases.get(raw_key.casefold())
        if not sep or key is None:
            errors.append(f"неизвестный параметр: {html.escape(token)}")
            continue
        if key in {"operating_mode", "source", "well", "currency"}:
            values[key] = raw.strip()
        elif key == "wash_treatment":
            if raw.casefold() not in {"1", "0", "да", "нет", "yes", "no", "true", "false"}:
                errors.append("промывка: используйте да/нет")
            else:
                values[key] = raw.casefold() in {"1", "да", "yes", "true"}
        else:
            number = _to_float(raw)
            if number is None or not math.isfinite(number):
                errors.append(f"{html.escape(raw_key)}: требуется конечное число")
            else:
                values[key] = number / 100 if key.endswith("relative_change") else number
    if not any(key in values for key in aliases.values() if key not in {
        "well", "source", "currency", "event_probability", "horizon_days",
        "treatment_efficiency", "product_price_per_m3", "operating_loss_per_day", "treatment_cost"
    }):
        errors.append("не задано ни одного изменения")
    return values, errors


def format_scenario_messages(item: DiagnosedWell, values: dict[str, float | str | bool]) -> list[str]:
    change_keys = {key for key in galit.ScenarioChanges.__dataclass_fields__ if key != "effect_override"}
    override = None
    if "inhibitor_efficiency" in values:
        override = galit.EffectOverride(
            inhibitor_efficiency=float(values["inhibitor_efficiency"]),
            source=str(values.get("source", "")) or None,
        )
    changes = galit.ScenarioChanges(
        **{key: values[key] for key in change_keys if key in values}, effect_override=override,
    )
    economics = None
    if "horizon_days" in values:
        economics = galit.ScenarioEconomics(
            horizon_days=float(values["horizon_days"]),
            event_probability=float(values["event_probability"]) if "event_probability" in values else None,
            treatment_efficiency=float(values["treatment_efficiency"]) if "treatment_efficiency" in values else None,
            product_price_per_m3=float(values["product_price_per_m3"]) if "product_price_per_m3" in values else None,
            operating_loss_per_day=float(values["operating_loss_per_day"]) if "operating_loss_per_day" in values else None,
            treatment_cost=float(values["treatment_cost"]) if "treatment_cost" in values else None,
            currency=str(values["currency"]) if "currency" in values else None,
        )
    result = galit.compare_scenario(item.case, changes, economics)
    delta = result.delta
    blocks = [
        f"<b>Сценарий · {html.escape(result.well)}</b>",
        f"Статус: {result.status.value}\nРиск: {result.before.integrated_risk:.3f} → {result.after.integrated_risk:.3f} "
        f"(Δ {delta['integrated_risk']:+.3f})\nДебит нефти: {result.before.forecast_oil_rate_m3_day:.2f} → "
        f"{result.after.forecast_oil_rate_m3_day:.2f} м³/сут",
    ]
    if result.economics:
        b = result.economics.breakdown
        unit = html.escape(result.economics.currency or "—")
        blocks.append(f"Стоимость: {'—' if b.total_treatment_cost is None else f'{b.total_treatment_cost:.2f} {unit}'}\n"
                      f"Предотвращённый ущерб: {'—' if b.potential_avoided_damage is None else f'{b.potential_avoided_damage:.2f} {unit}'}\n"
                      f"Чистый эффект: {'—' if b.net_expected_effect is None else f'{b.net_expected_effect:.2f} {unit}'}\n"
                      f"ROI: {'—' if b.roi_ratio is None else f'{b.roi_ratio:.2f}'}")
    if result.missing_inputs:
        blocks.append("Не хватает: " + html.escape(", ".join(result.missing_inputs)))
    if result.warnings:
        blocks.append("Предупреждения:\n" + "\n".join("• " + html.escape(x) for x in result.warnings))
    blocks.append("Screening score не является вероятностью; причинный эффект не обещается.")
    return _forecast_chunks(blocks)


def format_equipment_messages(rows: list[galit.EquipmentForecast], *, title: str = "Оборудование") -> list[str]:
    if not rows:
        return ["Данных по оборудованию недостаточно. Загрузите metadata и телеметрию через сайт/API."]
    blocks = [f"<b>{html.escape(title)}</b>"]
    for row in rows[:20]:
        risk = "—" if row.baseline_failure_risk is None else f"{row.baseline_failure_risk:.2f}"
        rul = "недоступен" if row.rul_days is None else f"{row.rul_days[0]}–{row.rul_days[1]} сут"
        causes = ", ".join(html.escape(x.label) for x in row.causes[:3]) or "данных недостаточно"
        window = "—" if row.maintenance_window_start is None else (
            f"{row.maintenance_window_start.date().isoformat()}–{row.maintenance_window_end.date().isoformat()}")
        blocks.append(f"<b>{html.escape(row.well)}</b> · {html.escape(row.lift_type)}\n"
                      f"baseline-риск {risk} · {html.escape(row.risk_level)} · RUL {rul}\n"
                      f"Причины/аномалии: {causes}\nОбслуживание: {window} · {html.escape(row.urgency)}\n"
                      f"Качество: {row.data_completeness:.0%} · confidence {html.escape(row.confidence)}")
    blocks.append(html.escape(galit.EQUIPMENT_DISCLAIMER))
    return _forecast_chunks(blocks)


def parse_equipment_query(text: str) -> str:
    try:
        tokens = shlex.split(text, posix=True)[1:]
    except ValueError as exc:
        raise ValueError("ошибка кавычек: " + str(exc)) from exc
    if len(tokens) > 1:
        raise ValueError("укажите не более одного имени скважины")
    return tokens[0].strip() if tokens else ""


def format_plan_messages(items: list[DiagnosedWell], top: int = 10) -> list[str]:
    """Build escaped compact HTML chunks within Telegram's message limit."""
    if not items:
        return ["История пуста. Сначала выполните /aspo или пошаговый расчёт."]
    plan = generate_master_plan(items, limit=top)
    loss = plan.summary.possible_oil_loss_central_m3d
    lines = [
        "<b>План мастера</b>",
        f"Задач: {plan.summary.task_count} · заблокировано: {plan.summary.blocked_tasks} · "
        f"потеря под риском: {'—' if loss is None else f'{loss:.1f} м³/сут'}",
    ]
    for index, task in enumerate(plan.tasks, 1):
        action = task.recommended_action
        if len(action) > 300:
            action = action[:297] + "…"
        status = "можно планировать" if task.safe_to_act else "БЛОК: верифицировать данные"
        central = task.possible_oil_loss.central_m3d
        lines.append(
            f"\n<b>{index}. {html.escape(task.well)}</b> · {html.escape(task.response_deadline)}\n"
            f"{html.escape(task.dominant_label)} · риск {task.risk:.2f} · "
            f"потеря {'—' if central is None else f'{central:.1f} м³/сут'}\n"
            f"{html.escape(status)}\n{html.escape(action)}"
        )
    chunks: list[str] = []
    current = ""
    for block in lines:
        candidate = block if not current else current + "\n" + block
        if len(candidate) > TELEGRAM_TEXT_LIMIT - 100 and current:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def format_map_messages(items: list[DiagnosedWell]) -> list[str]:
    """Text-only map summary for field use; no unsupported interactive attachment."""
    if not items:
        return ["История пуста. Сначала выполните /aspo, при необходимости задав lat= и lon=."]
    data = galit.prepare_field_map(items)
    summary = data.summary
    blocks = [
        "<b>Карта месторождения · сводка</b>",
        f"На карте: {summary.mapped_wells} из {summary.total_wells} · "
        f"без координат: {summary.missing_coordinates} · некорректных: {summary.invalid_coordinates}",
        "Статусы: " + " · ".join(
            f"{html.escape(label)} {summary.counts_by_status[label]}"
            for label in ("норма", "растущий риск", "критический")
        ),
        "Потеря под риском по отображённым: " + (
            "—" if summary.possible_oil_loss_m3d is None
            else f"{summary.possible_oil_loss_m3d:.1f} м³/сут"
        ),
    ]
    for point in sorted(data.points, key=lambda row: (-row.risk, row.well.casefold()))[:10]:
        loss = "—" if point.possible_oil_loss_m3d is None else f"{point.possible_oil_loss_m3d:.1f} м³/сут"
        blocks.append(
            f"<b>{html.escape(point.well)}</b> · {html.escape(point.status_label)} · риск {point.risk:.2f}\n"
            f"{point.latitude:.5f}, {point.longitude:.5f} · потеря {loss}"
        )
    if not data.points:
        blocks.append("Нет скважин с полной валидной парой latitude/longitude. Добавьте lat= и lon= в /aspo.")
    blocks.append("Размер маркера в веб-карте соответствует screening-оценке добычи под риском, не прогнозу фактической потери.")
    return _forecast_chunks(blocks)


def _to_float(raw: str) -> float | None:
    """Число с допуском запятой в качестве десятичного разделителя."""
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def parse_args(args: list[str]) -> tuple[dict[str, float | str], list[str]]:
    """Разбор аргументов /aspo.

    Поддерживаются позиционные значения (6 основных параметров по порядку)
    и пары «ключ=значение» в любой комбинации. Возвращает
    (параметры, список ошибок).
    """
    params: dict[str, float | str] = {}
    errors: list[str] = []
    positional: list[float] = []

    for token in args:
        if "=" in token:
            key_raw, _, value_raw = token.partition("=")
            key = KEY_ALIASES.get(key_raw.strip().lower())
            if key is None:
                errors.append(f"неизвестный параметр: «{html.escape(key_raw)}»")
                continue
            value_raw = value_raw.strip()
            if key == "lift_type":
                lift = LIFT_TYPES.get(value_raw.lower())
                if lift is None:
                    errors.append("способ эксплуатации: ЭЦН | ШГН | фонтан")
                else:
                    params["lift_type"] = lift
                continue
            if key == "name":
                params["name"] = value_raw or "Без имени"
                continue
            value = _to_float(value_raw)
            if value is None:
                errors.append(f"«{html.escape(key_raw)}»: не число")
                continue
            params[key] = value
        else:
            value = _to_float(token)
            if value is None:
                errors.append(f"лишний аргумент: «{html.escape(token)}»")
                continue
            positional.append(value)

    # непомеченные значения распределяем по основным параметрам по порядку
    for value, key in zip(positional, POSITIONAL_KEYS):
        params[key] = value

    missing = [k for k in REQUIRED_KEYS if k not in params]
    if missing:
        labels = {k: RANGES[k][2] for k in missing}
        errors.append("не задано: " + "; ".join(labels[k] for k in missing))
    if params.get("q_oil_m3d", 0.0) == 0.0 and params.get("q_water_m3d", 0.0) == 0.0:
        errors.append("дебит жидкости равен нулю — задайте нефть и/или воду")

    for key, (lo, hi, label) in RANGES.items():
        if key in params and not (lo <= float(params[key]) <= hi):
            errors.append(f"{label}: получено {params[key]}")
    return params, errors


def build_case(params: dict[str, float | str]) -> WellCase:
    """Сборка WellCase из разобранных параметров с типовыми значениями."""
    merged = {**DEFAULTS, **params}  # type: ignore[dict-item]
    return WellCase(
        name=str(merged.get("name", "Скважина мастера")),
        geometry=WellGeometry(
            depth_m=float(merged["depth_m"]),
            tubing_id_m=float(merged["tubing_mm"]) / 1000.0,
        ),
        rate=ProductionRate(
            q_oil_m3d=float(merged["q_oil_m3d"]),
            q_water_m3d=float(merged["q_water_m3d"]),
            gor_m3m3=float(merged["gor_m3m3"]),
        ),
        fluid=FluidProperties(),
        thermal=ThermalParams(
            t_surface_c=float(merged["t_surface_c"]),
            geothermal_grad=float(merged["geothermal_grad"]),
        ),
        water=WaterAnalysis(
            ions_mg_l=dict(TYPICAL_BRINE),
            ph=6.0,
            t_c=40.0,
            p_pa=5.0e6,
        ),
        wax=WaxProperties(
            wat_stock_tank_c=float(merged["wat_c"]),
            wax_content_pct=float(merged["wax_pct"]),
        ),
        co2_mol_frac=float(merged["co2_mol_frac"]),
        lift_type=str(merged["lift_type"]),
        p_wellhead_pa=float(merged["p_wellhead_mpa"]) * 1e6,
        latitude=float(merged["latitude"]) if "latitude" in merged else None,
        longitude=float(merged["longitude"]) if "longitude" in merged else None,
        provenance=DataProvenance(sources={
            "water.ions_mg_l": "synthetic",
            "water.ph": "default", "water.t_c": "default", "water.p_pa": "default",
            "fluid.salinity_ppm": "default", "thermal.u_to": "default",
            "inhibitor_efficiency": "default",
            **{
                path: "default" for key, path in {
                    "t_surface_c": "thermal.t_surface_c",
                    "geothermal_grad": "thermal.geothermal_grad",
                    "wax_pct": "wax.wax_content_pct",
                    "co2_mol_frac": "co2_mol_frac",
                    "p_wellhead_mpa": "p_wellhead_pa",
                }.items() if key not in params
            },
        }),
    )


def wax_treatment(result: galit.DiagnosisResult, case: WellCase) -> str:
    """Подбор технологии борьбы с АСПО под глубину начала отложений.

    Скребок на проволоке с поверхности достаёт ограниченно (~1500 м);
    на ШГН скребки-центраторы идут вместе со штангами до забоя.
    """
    max_scraper = 1500.0 if case.lift_type == "ЭЦН" else case.geometry.depth_m
    return recommend_wax_treatment(
        result.wax_onset_m, result.severity["wax"], max_scraper
    )


def _level(value: float) -> str:
    """Понятный уровень для нормированной шкалы 0…1."""
    if value < 0.25:
        return "низкий"
    if value < 0.50:
        return "умеренный"
    if value < 0.75:
        return "высокий"
    return "критический"


def _format_rate(value: float) -> str:
    """Показать малую скорость без ложного округления до нуля."""
    if value == 0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.2f}"


def _warning_summary(result: galit.DiagnosisResult) -> list[str]:
    """Свести технические предупреждения к нескольким безопасным выводам."""
    warnings = [str(w) for w in result.warnings]
    lowered = " ".join(warnings).lower()
    summary: list[str] = []

    incomplete = (not result.quality.production_ready or
                  any(key in lowered for key in ("screening", "типов", "default",
                                                  "synthetic", "фактическ")))
    if incomplete:
        summary.append("Неполные/типовые данные: результат только для предварительной оценки.")

    rate = result.corrosion.get("rate_mm_yr")
    anomalous_corrosion = (
        isinstance(rate, (int, float)) and float(rate) >= 0.25
    ) or any(key in lowered for key in ("аномальн", "коррози"))
    if anomalous_corrosion:
        detail = (f" ({_format_rate(float(rate))} мм/год)"
                  if isinstance(rate, (int, float)) else "")
        summary.append("Аномально высокая коррозия" + detail +
                       ": требуется инструментальная проверка.")

    if any(key in lowered for key in ("неприменим", "stiff-davis", "tds", "дегазац")):
        summary.append("Расчёт солеотложений ограничен областью применимости моделей.")
    if any(key in lowered for key in ("температур", "сужение", "профил")):
        summary.append("Профиль ствола и влияние отложений требуют сверки с промысловыми данными.")

    if warnings and not summary:
        summary.append("Есть модельные допущения; проверьте исходные данные перед решением.")
    return summary[:4]


def format_report(result: galit.DiagnosisResult, case: WellCase,
                  treatment: str) -> str:
    """Компактный HTML-отчёт для быстрого управленческого просмотра."""
    e = html.escape
    risk = float(result.integrated_risk)
    wax = float(result.severity["wax"])
    lines = [
        "<b>ГАЛИТ · Предварительная оценка</b>",
        f"Скважина: <b>{e(result.well)}</b>",
        "",
        "<b>Итог</b>",
        f"Риск: <b>{risk:.2f} · {_level(risk)}</b>",
        f"Доминирующий фактор: {e(_mech_ru(result.dominant))}",
    ]

    rate = result.corrosion.get("rate_mm_yr")
    if isinstance(rate, (int, float)):
        lines.append(f"Коррозия: <b>{_format_rate(float(rate))} мм/год</b>")

    lines += ["", "<b>АСПО</b>"]
    if result.wax_onset_m is None:
        lines.append("Начало отложений: <b>не прогнозируется</b>")
    else:
        onset = result.wax_onset_m
        depth = case.geometry.depth_m
        zone_pct = 100.0 * onset / depth if depth else 0.0
        lines += [
            f"Начало отложений: <b>{onset:.0f} м</b> от устья",
            f"Зона: 0–{onset:.0f} м · {zone_pct:.0f}% ствола",
        ]
    lines.append(f"Тяжесть: <b>{wax:.2f} · {_level(wax)}</b>")

    lines += [
        "",
        "<b>Рекомендуемое действие</b>",
        f"Основное действие: {e(treatment)}",
    ]
    if result.recommendation and result.recommendation.strip() != treatment.strip():
        lines.append(f"Дополнительно: {e(result.recommendation)}")

    quality_level = ("достаточная" if result.quality.production_ready else
                     "ограниченная")
    lines += [
        "",
        "<b>Надёжность данных</b>",
        f"{quality_level.capitalize()} · класс {e(result.quality.grade)} · "
        f"полнота {result.quality.completeness:.0%}",
    ]

    summary = _warning_summary(result)
    if summary:
        lines += ["", "<b>Краткие ограничения</b>"]
        lines += [f"• {e(item)}" for item in summary]
    lines += ["", "Оценка не заменяет промысловые исследования и инженерную проверку."]
    return "\n".join(lines)


def _mech_ru(mech: str) -> str:
    return {"halite": "галит", "calcite": "кальцит", "wax": "АСПО",
            "corrosion": "коррозия"}.get(mech, mech)


COMPATIBILITY_ION_KEYS = {"Na", "Cl", "Ca", "Ba", "SO4", "HCO3", "CO3"}
COMPATIBILITY_ALIASES = {
    "name": "name", "имя": "name", "вода": "name",
    "ph": "ph", "t": "t_c", "temp": "t_c", "температура": "t_c",
    "p": "p_mpa", "pressure": "p_mpa", "давление": "p_mpa",
    **{key.casefold(): key for key in COMPATIBILITY_ION_KEYS},
}
COMPATIBILITY_REQUIRED = {"Na", "Cl", "Ca", "Ba", "SO4", "ph", "t_c", "p_mpa"}


def parse_compatibility_water(text: str, default_name: str = "Вода") -> galit.CompatibilityWater:
    """Parse one measured water from a compact semicolon-delimited key=value row."""
    values: dict[str, str] = {}
    for part in (item.strip() for item in text.strip().split(";") if item.strip()):
        if "=" not in part:
            if "name" not in values:
                values["name"] = part
                continue
            raise ValueError(f"ожидалось поле key=value: {part}")
        raw_key, raw_value = (item.strip() for item in part.split("=", 1))
        key = COMPATIBILITY_ALIASES.get(raw_key.casefold())
        if key is None:
            raise ValueError(f"неизвестное поле: {raw_key}")
        if key in values:
            raise ValueError(f"поле задано повторно: {raw_key}")
        values[key] = raw_value
    missing = sorted(COMPATIBILITY_REQUIRED - values.keys())
    if not ({"HCO3", "CO3"} & values.keys()):
        missing.append("HCO3 или CO3")
    if missing:
        raise ValueError("нет измеренных полей: " + ", ".join(missing))
    ions: dict[str, float] = {}
    for key in COMPATIBILITY_ION_KEYS & values.keys():
        try:
            ions[key] = float(values[key].replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"{key}: требуется число") from exc
    try:
        ph = float(values["ph"].replace(",", "."))
        t_c = float(values["t_c"].replace(",", "."))
        p_pa = float(values["p_mpa"].replace(",", ".")) * 1e6
    except ValueError as exc:
        raise ValueError("pH, T и P должны быть числами") from exc
    return galit.CompatibilityWater(ions, ph, t_c, p_pa, values.get("name", default_name))


def parse_compatibility_rows(rows: list[dict[str, object]]) -> tuple[galit.CompatibilityWater, galit.CompatibilityWater]:
    """Parse exactly two tabular water rows (CSV/XLSX column names equal compact keys)."""
    if len(rows) != 2:
        raise ValueError("документ должен содержать ровно две строки воды")
    parsed = []
    for index, row in enumerate(rows, 1):
        parts = [f"{key}={value}" for key, value in row.items()
                 if value is not None and str(value).strip() and str(value).lower() != "nan"]
        parsed.append(parse_compatibility_water(";".join(parts), f"Вода {index}"))
    return parsed[0], parsed[1]


def parse_compatibility_document(data: bytes, filename: str) -> tuple[
        tuple[galit.CompatibilityWater, galit.CompatibilityWater],
        tuple[galit.ProfilePoint, ...] | None, galit.DoseResponseCurve | None]:
    """Read two waters and optional measured profile/validated curve from CSV/XLSX."""
    suffix = Path(filename).suffix.casefold()
    profile = None
    curve = None
    if suffix == ".csv":
        text = data.decode("utf-8-sig")
        sample = text[:2048]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        waters = parse_compatibility_rows(list(csv.DictReader(io.StringIO(text), dialect=dialect)))
        return waters, profile, curve
    if suffix != ".xlsx":
        raise ValueError("поддерживаются только CSV и XLSX")
    import pandas as pd
    book = pd.read_excel(io.BytesIO(data), sheet_name=None)
    water_sheet = next((frame for name, frame in book.items() if name.casefold() in {"waters", "воды"}), None)
    if water_sheet is None:
        water_sheet = next(iter(book.values()), None)
    if water_sheet is None:
        raise ValueError("XLSX не содержит листов")
    waters = parse_compatibility_rows(water_sheet.to_dict("records"))
    profile_sheet = next((frame for name, frame in book.items() if name.casefold() in {"profile", "профиль"}), None)
    if profile_sheet is not None:
        profile = tuple(galit.ProfilePoint(float(row["depth_m"]), float(row["t_c"]),
                                           float(row["p_mpa"]) * 1e6)
                        for row in profile_sheet.to_dict("records"))
    dose_sheet = next((frame for name, frame in book.items() if name.casefold() in {"dose_response", "доза"}), None)
    if dose_sheet is not None:
        records = dose_sheet.to_dict("records")
        if records:
            first = records[0]
            validated = str(first.get("validated", "")).casefold() in {"true", "1", "yes", "да"}
            points = tuple(galit.DoseResponsePoint(float(row["dose_mg_l"]),
                                                   float(row["maximum_supported_si"])) for row in records)
            curve = galit.DoseResponseCurve(str(first["product"]), str(first["mineral"]), points,
                                             validated, str(first["validation_reference"]))
    return waters, profile, curve


def format_compatibility_report(result: galit.CompatibilityResult,
                                water_a: galit.CompatibilityWater,
                                water_b: galit.CompatibilityWater) -> list[str]:
    e = html.escape
    dangerous = next((row for row in result.ratios
                      if row.fraction_b == result.dangerous_fraction_b), result.ratios[0])
    labels = {"calcite": "кальцит", "barite": "барит", "gypsum": "гипс", "halite": "галит"}
    mineral_lines = []
    for key in galit.compatibility.SUPPORTED_MINERALS:
        item = dangerous.minerals[key]
        if item.saturation_index is None:
            shown = "не рассчитан"
        else:
            shown = f"SI {item.saturation_index:.2f} · " + ("есть риск" if item.supersaturated else "риск не выявлен")
        mineral_lines.append(f"{labels[key].capitalize()}: {shown}")
    intervals = "; ".join(f"B {x.start_fraction_b:.0%}–{x.end_fraction_b:.0%}"
                          for x in result.unsafe_intervals) or "не выявлены"
    blocks = [
        "<b>Совместимость двух измеренных вод</b>",
        f"A: {e(water_a.name)} · B: {e(water_b.name)}",
        "Совместимость: <b>" + ("неблагоприятная" if result.unsafe_intervals else "риск по screening не выявлен") + "</b>",
        "<b>Риски при наиболее опасной смеси</b>\n" + "\n".join(mineral_lines),
        f"Опасное A:B: <b>{e(result.dangerous_ratio_a_to_b or '—')}</b> "
        f"(B {result.dangerous_fraction_b:.0%})" if result.dangerous_fraction_b is not None else "Опасное A:B: —",
        "Небезопасные интервалы: " + intervals,
    ]
    if result.deposition_locations:
        locations = []
        for item in result.deposition_locations:
            first = "не выявлена" if item.first_supersaturation_depth_m is None else f"с {item.first_supersaturation_depth_m:.0f} м"
            peak = "—" if item.maximum_risk_depth_m is None else f"максимум около {item.maximum_risk_depth_m:.0f} м"
            locations.append(f"{labels[item.mineral].capitalize()}: {first}; {peak}")
        blocks.append("<b>Вероятная зона по измеренному профилю</b>\n" + "\n".join(locations))
    else:
        blocks.append("Зона/глубина: не оценена — измеренный профиль T/P не предоставлен.")
    if result.inhibitor.dose_mg_l is None:
        blocks.append("Доза ингибитора: не назначена. Требуется лабораторная/валидированная vendor-кривая dose-response для продукта и контролирующей соли.")
    else:
        blocks.append(f"Доза ингибитора: <b>{result.inhibitor.dose_mg_l:g} мг/л</b> · "
                      f"{e(result.inhibitor.product or '')} · основание {e(result.inhibitor.validation_reference or '')}")
    blocks.append("Screening показывает термодинамическую склонность, а не скорость или массу осадка; решение подтвердить лабораторно.")
    return _forecast_chunks(blocks)


CHEMICAL_ALIASES = {
    "id": "product_id", "product": "product_id", "продукт": "product_id",
    "hazard": "hazards", "hazards": "hazards", "риск": "hazards", "риски": "hazards",
    "fluid": "treated_fluid_m3_day", "жидкость": "treated_fluid_m3_day",
    "oil": "oil_m3_day", "нефть": "oil_m3_day", "quantity": "quantity_kg",
    "qty": "quantity_kg", "количество": "quantity_kg", "required": "required_on",
    "дата": "required_on", "lead": "lead_time_days", "поставка": "lead_time_days",
    "safety": "safety_stock_days", "резерв": "safety_stock_days",
    "kind": "kind", "тип": "kind", "lot": "lot_id", "партия": "lot_id",
    "reference": "reference", "основание": "reference", "expires": "expires_on",
    "годен": "expires_on", "cost": "unit_cost", "стоимость": "unit_cost",
    "currency": "currency", "валюта": "currency", "date": "occurred_on",
}

CHEMICAL_KINDS = {
    "receipt": "receipt", "приход": "receipt", "consumption": "consumption",
    "расход": "consumption", "adjustment": "adjustment", "корректировка": "adjustment",
    "expiry": "expiry", "списание": "expiry", "release": "release", "возврат": "release",
}


def parse_chemical_command(text: str) -> tuple[dict[str, str], list[str]]:
    """Strict shell-style key=value parser shared by chemical commands."""
    try:
        tokens = shlex.split(text, posix=True)[1:]
    except ValueError as exc:
        return {}, ["ошибка кавычек: " + str(exc)]
    values: dict[str, str] = {}
    errors: list[str] = []
    for token in tokens:
        raw_key, separator, raw = token.partition("=")
        key = CHEMICAL_ALIASES.get(raw_key.casefold())
        if not separator or key is None or not raw.strip():
            errors.append("неверный параметр: " + html.escape(token))
        elif key in values:
            errors.append("параметр задан повторно: " + html.escape(raw_key))
        else:
            values[key] = raw.strip()
    return values, errors


def _positive_decimal(raw: str, field: str) -> Decimal:
    try:
        value = Decimal(raw.replace(",", "."))
    except Exception as exc:
        raise ValueError(f"{field}: требуется число") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field}: требуется положительное конечное число")
    return value


def _nonnegative_int(raw: str, field: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field}: требуется целое число") from exc
    if value < 0:
        raise ValueError(f"{field}: не может быть отрицательным")
    return value


def resolve_chemical_product(repo: galit.ChemicalRepository, query: str) -> galit.ChemicalProduct:
    """Resolve exact ID/name or an unambiguous ID prefix without guessing."""
    normalized = query.strip().casefold()
    products = repo.list_products()
    exact = [x for x in products if x.id.casefold() == normalized or x.name.casefold() == normalized]
    matches = exact or [x for x in products if x.id.casefold().startswith(normalized)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise galit.ChemicalConflictError("реагент указан неоднозначно")
    raise galit.ChemicalNotFoundError(f"product {query} not found")


def format_reagents(products: list[galit.ChemicalProduct]) -> list[str]:
    if not products:
        return ["Каталог реагентов пуст. Доступность и дозы не определены."]
    blocks = ["<b>Реагенты</b>"]
    for item in products:
        price = ("не задана" if item.price_per_kg is None else
                 f"{item.price_per_kg} {html.escape(item.currency or '—')}/кг")
        blocks.append(
            f"<code>{html.escape(item.id)}</code> · <b>{html.escape(item.name)}</b>\n"
            f"Производитель: {html.escape(item.manufacturer)} · статус: {'активен' if item.active else 'неактивен'}\n"
            f"Риски: {html.escape(', '.join(item.hazards))} · цена: {price}"
        )
    return _forecast_chunks(blocks)


def format_recommendations(rows: list[galit.ChemicalRecommendation]) -> list[str]:
    blocks = ["<b>Подбор реагента по валидированным данным</b>"]
    for row in rows:
        if row.status != "available":
            blocks.append("Недоступно: " + html.escape(row.reason or "основание не определено"))
            continue
        cost = "не определена" if row.daily_cost is None else f"{row.daily_cost} {html.escape(row.currency or '—')}/сут"
        oil_cost = "не определена" if row.cost_per_m3_oil is None else f"{row.cost_per_m3_oil} {html.escape(row.currency or '—')}/м³ нефти"
        blocks.append(
            f"<b>{html.escape(row.product_name or row.product_id or '—')}</b> · <code>{html.escape(row.product_id or '—')}</code>\n"
            f"Доза: {row.dose_kg_m3} кг/м³ · расход: {row.daily_consumption_kg} кг/сут\n"
            f"Стоимость: {cost} · {oil_cost}\n"
            f"Подтверждения: {html.escape(', '.join(row.evidence_ids))}"
        )
    blocks.append("Доза выводится только из валидированной ссылочной dose-response кривой; инженерная проверка обязательна.")
    return _forecast_chunks(blocks)


def format_stock(product: galit.ChemicalProduct, stock: dict[str, object]) -> list[str]:
    blocks = [
        f"<b>Остаток · {html.escape(product.name)}</b>",
        f"Доступно без активных резервов: <b>{html.escape(str(stock['available_kg']))} кг</b> · на {html.escape(str(stock['as_of']))}\n"
        f"revision={stock['revision']}",
    ]
    lots = stock.get("lots", [])
    if lots:
        for lot in lots:
            blocks.append(f"Партия <code>{html.escape(str(lot['lot_id']))}</code> · {html.escape(str(lot['available_kg']))} кг · годна до {html.escape(str(lot['expires_on']))}")
    else:
        blocks.append("Годных доступных партий нет.")
    return _forecast_chunks(blocks)


def chemical_shortage_rows(repo: galit.ChemicalRepository, *, lead_time_days: int,
                            safety_stock_days: int, as_of: date) -> list[dict[str, object]]:
    """Build shortage rows; lack of consumption history remains explicitly unavailable."""
    rows: list[dict[str, object]] = []
    horizon = lead_time_days + safety_stock_days
    for product in repo.list_products():
        stock = repo.stock(product.id, as_of=as_of)
        history_by_day: dict[date, Decimal] = {}
        for tx in repo.list_transactions(product.id):
            if tx.kind == "consumption" and tx.occurred_at.date() <= as_of:
                day = tx.occurred_at.date()
                history_by_day[day] = history_by_day.get(day, Decimal(0)) + tx.quantity_kg
        forecast = galit.deterministic_consumption_forecast(
            sorted(history_by_day.items()), horizon_days=max(1, horizon), as_of=as_of,
        )
        if forecast["status"] != "available":
            rows.append({"product": product, "status": "unavailable", "reason": forecast["reason"], "stock": stock})
            continue
        report = galit.shortage_report(
            stock["available_kg"], forecast["daily_kg"], lead_time_days=lead_time_days,
            safety_stock_days=safety_stock_days, as_of=as_of,
        )
        rows.append({"product": product, "status": "available", "forecast": forecast, "report": report})
    return rows


def format_shortages(rows: list[dict[str, object]], lead: int, safety: int) -> list[str]:
    blocks = [f"<b>Дефициты</b> · поставка {lead} сут · страховой запас {safety} сут"]
    if not rows:
        blocks.append("Каталог реагентов пуст; дефицит не определён.")
    for row in rows:
        product = row["product"]
        if row["status"] != "available":
            blocks.append(f"<b>{html.escape(product.name)}</b>: не определено — нет истории расхода. Остаток {html.escape(str(row['stock']['available_kg']))} кг.")
            continue
        report = row["report"]
        risk = "ДЕФИЦИТ" if report["risk"] else "достаточно"
        cover = "не определён" if report["days_cover"] is None else f"{report['days_cover']} сут"
        blocks.append(
            f"<b>{html.escape(product.name)}</b>: {risk}\n"
            f"доступно {report['available_kg']} кг · требуется {report['required_kg']} кг · дефицит {report['shortage_kg']} кг · покрытие {cover}"
        )
    return _forecast_chunks(blocks)


def build_reservation_preview(values: dict[str, str], repo: galit.ChemicalRepository,
                              *, idempotency_key: str) -> dict[str, object]:
    required = {"product_id", "quantity_kg", "required_on"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError("не заданы: " + ", ".join(missing))
    product = resolve_chemical_product(repo, values["product_id"])
    quantity = _positive_decimal(values["quantity_kg"], "quantity_kg")
    required_on = date.fromisoformat(values["required_on"])
    if required_on < date.today():
        raise ValueError("required_on не может быть в прошлом")
    stock = repo.stock(product.id, as_of=required_on)
    if Decimal(str(stock["available_kg"])) < quantity:
        raise galit.ChemicalConflictError("insufficient non-expired FEFO stock")
    return {"action": "reserve", "product_id": product.id, "product_name": product.name,
            "quantity_kg": str(quantity), "required_on": required_on.isoformat(),
            "idempotency_key": idempotency_key, "expected_revision": stock["revision"]}


def build_transaction_preview(values: dict[str, str], repo: galit.ChemicalRepository,
                              *, idempotency_key: str) -> dict[str, object]:
    required = {"product_id", "quantity_kg", "kind", "reference"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError("не заданы: " + ", ".join(missing))
    product = resolve_chemical_product(repo, values["product_id"])
    kind = CHEMICAL_KINDS.get(values["kind"].casefold())
    if kind is None:
        raise ValueError("kind: receipt | consumption | adjustment | expiry | release")
    quantity = _positive_decimal(values["quantity_kg"], "quantity_kg")
    occurred_on = date.fromisoformat(values.get("occurred_on", date.today().isoformat()))
    preview: dict[str, object] = {
        "action": "transaction", "kind": kind, "product_id": product.id,
        "product_name": product.name, "quantity_kg": str(quantity),
        "occurred_on": occurred_on.isoformat(), "reference": values["reference"],
        "idempotency_key": idempotency_key, "expected_revision": repo.revision,
    }
    if kind == "receipt":
        if "lot_id" not in values or "expires_on" not in values:
            raise ValueError("для receipt требуются lot= и expires=")
        expires_on = date.fromisoformat(values["expires_on"])
        if expires_on < occurred_on:
            raise ValueError("expires_on не может предшествовать дате прихода")
        preview.update(lot_id=values["lot_id"], expires_on=expires_on.isoformat(),
                       unit_cost=values.get("unit_cost"), currency=values.get("currency"))
    elif kind != "consumption":
        if "lot_id" not in values:
            raise ValueError("для adjustment/expiry/release требуется lot=")
        preview["lot_id"] = values["lot_id"]
    return preview


def format_mutation_preview(preview: dict[str, object]) -> str:
    if preview["action"] == "reserve":
        body = (f"<b>Предпросмотр резерва</b>\nРеагент: {html.escape(str(preview['product_name']))}\n"
                f"Количество: {preview['quantity_kg']} кг · требуется: {preview['required_on']}")
    else:
        body = (f"<b>Предпросмотр складской операции</b>\nРеагент: {html.escape(str(preview['product_name']))}\n"
                f"Тип: {preview['kind']} · количество: {preview['quantity_kg']} кг\n"
                f"Основание: {html.escape(str(preview['reference']))}")
    return body + "\n\nДанные ещё не изменены. Ответьте <b>Подтвердить</b> или <b>Отмена</b>."


def execute_chemical_preview(repo: galit.ChemicalRepository,
                             preview: dict[str, object]) -> object:
    """Execute a validated preview against its captured repository revision."""
    revision = int(preview["expected_revision"])
    if preview["action"] == "reserve":
        return repo.reserve(
            str(preview["product_id"]), preview["quantity_kg"], date.fromisoformat(str(preview["required_on"])),
            idempotency_key=str(preview["idempotency_key"]), expected_revision=revision,
        )
    when = datetime.combine(date.fromisoformat(str(preview["occurred_on"])), datetime.min.time(), tzinfo=timezone.utc)
    kind = str(preview["kind"])
    if kind == "receipt":
        lot = galit.StockLot(
            str(preview["lot_id"]), str(preview["product_id"]), when,
            date.fromisoformat(str(preview["expires_on"])), Decimal(str(preview["quantity_kg"])),
            Decimal(str(preview["unit_cost"])) if preview.get("unit_cost") else None,
            str(preview["currency"]) if preview.get("currency") else None,
        )
        return repo.add_lot(lot, idempotency_key=str(preview["idempotency_key"]), expected_revision=revision)
    if kind == "consumption":
        return repo.consume(
            str(preview["product_id"]), preview["quantity_kg"], when,
            idempotency_key=str(preview["idempotency_key"]), reference=str(preview["reference"]),
            expected_revision=revision,
        )
    tx = galit.StockTransaction(
        str(uuid4()), str(preview["idempotency_key"]), str(preview["product_id"]),
        str(preview["lot_id"]), kind, Decimal(str(preview["quantity_kg"])), when,
        str(preview["reference"]),
    )
    return repo.append_transaction(tx, expected_revision=revision)


MENU_NEW = "🔎 Новый расчёт"
MENU_HELP = "ℹ️ Справка"
MENU_EXAMPLE = "📋 Пример"
CANCEL = "Отмена"

MAIN_MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=MENU_NEW)],
        [KeyboardButton(text=MENU_HELP), KeyboardButton(text=MENU_EXAMPLE)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Выберите действие",
)
CANCEL_MENU = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=CANCEL)]], resize_keyboard=True,
)

START_TEXT = (
    "<b>ГАЛИТ · предварительная оценка АСПО</b>\n"
    "Показывает начало отложений, уровень риска и рекомендуемое действие.\n\n"
    "<b>Шесть шагов</b>\n"
    "1. Глубина, м\n2. Внутренний диаметр НКТ, мм\n"
    "3. Дебит нефти, м³/сут\n4. Дебит воды, м³/сут\n"
    "5. Газовый фактор, м³/м³\n6. WAT, °C\n\n"
    "Чтобы начать, нажмите «🔎 Новый расчёт».\n"
    "Последние расчёты: /plan; карта: /map; прогноз: /forecast [скважина]; сценарий: /scenario; экономика: /economics; очистка — /plan_clear."
)
HELP_TEXT = (
    "<b>Справка</b>\n\n"
    "ГАЛИТ даёт предварительную оценку.\n\n"
    "<b>Пошагово:</b> 6 значений: глубина (м), НКТ (мм), нефть/вода "
    "(м³/сут), газовый фактор (м³/м³), WAT (°C).\n"
    "<code>/aspo 3200 62 8 72 65 34</code>\n"
    "Доп.: скважина=, способ=, парафин=, lat=, lon=.\n\n"
    "<b>Двойник:</b> <code>/twin [скважина]</code>, "
    "<code>/timeline [скважина] [days]</code>, <code>/changes [скважина]</code>.\n"
    "<b>Совместимость вод:</b> /compatibility — ввести ровно две измеренные воды строками или CSV/XLSX.\n"
    "<b>Команды:</b> /plan, /map, /areas, /hotspots [days], /cluster &lt;name&gt;, "
    "/spread [mechanism] [days], /forecast, /watercut, /breakthroughs, "
    "/injectors, /equipment, /failures, /maintenance, /scenario, /economics, /plan_clear.\n"
    "Паспорт: /passport, /passport_history, /passport_rate.\n"
    "Обработки: /treatments, /treatment_add, /treatment_result, "
    "/treatment_stats, /treatment_compare. before=/after= — м³/сут; "
    "валюты и единицы не смешиваются; аргументы key=value.\n"
    "Реагенты: /reagents, /reagent, /stock, /shortages; изменения склада: /reserve, /transaction — только через подтверждение."
)
EXAMPLE_TEXT = (
    "<b>Пример исходных данных</b>\n\n"
    "Глубина: 3200 м\nНКТ: 62 мм\nНефть: 8 м3/сут\n"
    "Вода: 72 м3/сут\nГазовый фактор: 65 м3/м3\nWAT: 34 °C\n\n"
    "Команда: <code>/aspo 3200 62 8 72 65 34</code>"
)

FSM_FIELDS = (
    ("depth_m", "Введите глубину скважины, м (100–8000):"),
    ("tubing_mm", "Введите внутренний диаметр НКТ, мм (20–200):"),
    ("q_oil_m3d", "Введите дебит нефти, м3/сут (0–1000):"),
    ("q_water_m3d", "Введите дебит воды, м3/сут (0–1000):"),
    ("gor_m3m3", "Введите газовый фактор, м3/м3 (0–2000):"),
    ("wat_c", "Введите WAT, °C (от -20 до 90):"),
)


class Calculation(StatesGroup):
    collecting = State()


class CompatibilityCalculation(StatesGroup):
    collecting = State()


class ChemicalMutation(StatesGroup):
    confirming = State()


COMPATIBILITY_PROMPT = (
    "Отправьте воду {label} одной строкой (все концентрации — измеренные, мг/л):\n"
    "<code>name=Пластовая; Na=20000; Cl=32000; Ca=4000; Ba=1000; "
    "SO4=0; HCO3=50; pH=6.2; T=25; P=5</code>\n"
    "P — МПа. Нужны Na, Cl, Ca, Ba, SO4, HCO3 или CO3, pH, T, P. "
    "Пропуски типовой водой не заполняются. Вместо строк можно отправить CSV/XLSX "
    "с ровно двумя строками; заголовки как в примере."
)


def validate_fsm_value(key: str, raw: str) -> tuple[float | None, str | None]:
    """Проверить одно значение пошагового ввода через общие диапазоны."""
    value = _to_float(raw.strip())
    if value is None:
        return None, "Введите число. Допускается десятичная запятая."
    lo, hi, label = RANGES[key]
    if not lo <= value <= hi:
        return None, f"Значение вне диапазона: {label}."
    return value, None


async def send_calculation(message: Message, params: dict[str, float | str]) -> None:
    """Выполнить расчёт и отправить отчёт для команды и FSM."""
    case = build_case(params)
    try:
        result = await asyncio.to_thread(diagnose, case)
    except (ValueError, KeyError) as exc:
        await message.answer(f"Расчёт не выполнен: {html.escape(str(exc))}",
                             reply_markup=MAIN_MENU)
        return
    if message.chat:
        RECENT_DIAGNOSES.add(message.chat.id, DiagnosedWell(case, result))
    await message.answer(
        format_report(result, case, wax_treatment(result, case)),
        disable_web_page_preview=True, reply_markup=MAIN_MENU,
    )


# ==========================================================================
# Обработчики aiogram
# ==========================================================================

dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(START_TEXT, reply_markup=MAIN_MENU)


def format_watercut_diagnosis(item: galit.WatercutDiagnosis) -> list[str]:
    e=html.escape; current="—" if item.current_water_cut is None else f"{item.current_water_cut:.1%}"; change="—" if item.absolute_change_pp is None else f"{item.absolute_change_pp:+.1%}"
    blocks=[f"<b>Обводнение · {e(item.well)}</b>",f"Текущая: {current} · изменение: {change}\nСтатус: <b>{e(item.severity)}</b> · confidence: {e(item.confidence)}"]
    if item.oil_forecast: blocks.append("Baseline-прогноз нефти: " + "; ".join(f"{x.days} сут {x.low_m3d:g}–{x.high_m3d:g} м³/сут" for x in item.oil_forecast))
    if item.candidate_injectors: blocks.append("Возможные нагнетатели:\n"+"\n".join(f"• {e(x.injector)} · score {x.score:.2f} · {x.distance_km:.1f} км · lag {x.lag_days or '—'} сут · {e(x.confidence)}" for x in item.candidate_injectors[:5]))
    else: blocks.append("Нагнетатели: недостаточно данных/нет кандидатов выше порога.")
    blocks += ["Проверить замеры и режим; затем пробы/ионный fingerprint, трассер и ГДИС.",e(galit.WATERCUT_DISCLAIMER)]
    return _forecast_chunks(blocks)


def _watercut_results() -> list[galit.WatercutDiagnosis]:
    metadata=WATERCUT.list_metadata(); production=WATERCUT.list_production(); injection=WATERCUT.list_injection(); injectors=[x for x in metadata if x.role=="injector"]
    return [galit.diagnose_watercut(x,production,injectors,injection) for x in metadata if x.role=="producer"]


@dp.message(Command("watercut"))
async def cmd_watercut(message: Message) -> None:
    well=" ".join((message.text or "").split()[1:]).strip()
    try:
        rows=await asyncio.to_thread(_watercut_results); item=next((x for x in rows if x.well.casefold()==well.casefold()),None)
        if not well: await message.answer("Укажите скважину: <code>/watercut Добывающая 139</code>"); return
        if item is None: await message.answer("Скважина не найдена или недостаточно данных."); return
        for chunk in format_watercut_diagnosis(item): await message.answer(chunk)
    except galit.WatercutStorageError: await message.answer("История обводнения временно недоступна.")


@dp.message(Command("breakthroughs"))
async def cmd_breakthroughs(message: Message) -> None:
    try:
        rows=[x for x in await asyncio.to_thread(_watercut_results) if x.severity in {"growing","critical"}]
        blocks=["<b>Возможное влияние/прорыв · screening</b>"]+[f"{html.escape(x.well)} · {x.severity} · WC {x.current_water_cut:.1%} · {x.confidence}" for x in rows]
        blocks.append(html.escape(galit.WATERCUT_DISCLAIMER))
        for chunk in _forecast_chunks(blocks): await message.answer(chunk)
    except galit.WatercutStorageError: await message.answer("История обводнения временно недоступна.")


@dp.message(Command("injectors"))
async def cmd_injectors(message: Message) -> None:
    await cmd_watercut(message)


def parse_smart_map_command(text: str) -> tuple[str | None, int, list[str]]:
    """Strict mechanism/days parser shared by /hotspots and /spread."""
    parts=(text or "").split()[1:]; mechanism=None; days=30; errors=[]
    for value in parts:
        if value.isdigit():
            days=int(value)
            if not 1<=days<=1095: errors.append("days должен быть 1–1095")
        elif value in galit.SMART_MAP_MECHANISMS: mechanism=value
        else: errors.append("неизвестный аргумент: "+html.escape(value))
    return mechanism,days,errors


def format_area_report(groups: list[dict], title="Участки риска") -> list[str]:
    blocks=[f"<b>{html.escape(title)}</b>"]
    for x in groups[:10]:
        losses=", ".join(f"{v['total']:g} {html.escape(v['currency'])}/{html.escape(v['unit'])}" for v in x["economic_loss_buckets"]) or "—"
        blocks.append(f"{html.escape(x.get('field') or 'field не задан')} · {html.escape(x['name'])}: n={x['count']}, critical={x['critical']}, avg={x['avg_risk']:.2f}\nМеханизмы: {html.escape(', '.join(x['dominant_mechanisms']))}; coverage={x['coverage']:.0%}; потери: {losses}")
    if not groups: blocks.append("Датированных данных недостаточно.")
    blocks.append(html.escape(galit.SMART_MAP_DISCLAIMER)); return _forecast_chunks(blocks)


def format_hotspot_report(zones: list[dict]) -> list[str]:
    blocks=["<b>Системные зоны одновременного ухудшения</b>"]
    for z in zones[:10]:
        wells=", ".join(html.escape(x["display_name"]) for x in z["member_wells"][:12])
        blocks.append(f"Зона {html.escape(z['zone_id'][-8:])}: {len(z['member_wells'])} скв., radius={z['radius_km']} км, confidence={z['confidence']}\n{wells}\nМеханизмы: {html.escape(', '.join(z['common_mechanisms']))}; coverage={z['coverage']:.0%}")
    if not zones: blocks.append("Зоны выше порога не выявлены или данных недостаточно.")
    blocks.append(html.escape(galit.SMART_MAP_DISCLAIMER)); return _forecast_chunks(blocks)


@dp.message(Command("areas"))
async def cmd_areas(message: Message) -> None:
    try:
        for chunk in format_area_report(await asyncio.to_thread(smart_map_service().groups)): await message.answer(chunk)
    except (ValueError,galit.SmartMapStorageError) as exc: await message.answer(html.escape(str(exc)))


@dp.message(Command("hotspots"))
async def cmd_hotspots(message: Message) -> None:
    mechanism,days,errors=parse_smart_map_command(message.text or "")
    if errors: await message.answer("Параметры не приняты: "+"; ".join(errors)); return
    try:
        rows=await asyncio.to_thread(smart_map_service().hotspots,days=days,mechanism=mechanism or "integrated")
        for chunk in format_hotspot_report(rows): await message.answer(chunk)
    except (ValueError,galit.SmartMapStorageError) as exc: await message.answer(html.escape(str(exc)))


@dp.message(Command("cluster"))
async def cmd_cluster(message: Message) -> None:
    name=" ".join((message.text or "").split()[1:]).strip()
    if not name: await message.answer("Укажите: <code>/cluster название куста</code>"); return
    try:
        matches=[x for x in await asyncio.to_thread(smart_map_service().groups) if x["name"].casefold()==name.casefold()]
        if len({x.get("field") for x in matches})>1: await message.answer("Название неоднозначно в разных fields; уточните через /areas."); return
        for chunk in format_area_report(matches,"Куст "+name): await message.answer(chunk)
    except (ValueError,galit.SmartMapStorageError) as exc: await message.answer(html.escape(str(exc)))


@dp.message(Command("spread"))
async def cmd_spread(message: Message) -> None:
    mechanism,days,errors=parse_smart_map_command(message.text or "")
    if errors: await message.answer("Параметры не приняты: "+"; ".join(errors)); return
    try:
        row=await asyncio.to_thread(smart_map_service().spread,mechanism=mechanism or "integrated",date_from=datetime.now(timezone.utc)-timedelta(days=days))
        if not row["available"]: text="Распространение не оценено: "+row["reason"]
        else: text=f"<b>Направление screening</b>\nАзимут: {row['bearing_deg']}°; скорость: {row['speed_km_day_range'][0]}–{row['speed_km_day_range'][1]} км/сут; frames={row['frames']}; confidence={row['confidence']}"
        await message.answer(text+"\n"+html.escape(galit.SMART_MAP_DISCLAIMER))
    except (ValueError,galit.SmartMapStorageError) as exc: await message.answer(html.escape(str(exc)))


@dp.message(Command("help"))
@dp.message(F.text == MENU_HELP)
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT, disable_web_page_preview=True,
                         reply_markup=MAIN_MENU)


@dp.message(F.text == MENU_EXAMPLE)
async def show_example(message: Message) -> None:
    await message.answer(EXAMPLE_TEXT, reply_markup=MAIN_MENU)


@dp.message(Command("cancel"))
@dp.message(F.text.casefold() == CANCEL.casefold())
async def cancel_calculation(message: Message, state: FSMContext) -> None:
    active = await state.get_state()
    await state.clear()
    text = "Расчёт отменён." if active else "Активного расчёта нет."
    await message.answer(text, reply_markup=MAIN_MENU)


@dp.message(F.text == MENU_NEW)
async def start_calculation(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(Calculation.collecting)
    await state.update_data(step=0, params={})
    await message.answer(FSM_FIELDS[0][1], reply_markup=CANCEL_MENU)


def _treatment_error(exc: Exception) -> str:
    if isinstance(exc, galit.TreatmentNotFoundError):
        return "Мероприятие не найдено. Проверьте ID."
    if isinstance(exc, galit.TreatmentConflictError):
        return "Конфликт изменения: запись уже обновлена. Откройте /treatment и повторите с актуальной revision."
    if isinstance(exc, galit.TreatmentStorageError):
        return "Журнал временно недоступен. Данные не изменены."
    return "Параметры не приняты: " + str(exc)


async def _answer_treatment_error(message: Message, exc: Exception) -> None:
    await message.answer(html.escape(_treatment_error(exc)))


@dp.message(Command("reagents"))
async def cmd_reagents(message: Message) -> None:
    try:
        for chunk in format_reagents(await asyncio.to_thread(CHEMICALS.list_products)):
            await message.answer(chunk)
    except galit.ChemicalStorageError:
        await message.answer("Каталог реагентов временно недоступен.")


@dp.message(Command("reagent"))
async def cmd_reagent(message: Message) -> None:
    values, errors = parse_chemical_command(message.text or "")
    required = {"hazards", "treated_fluid_m3_day", "oil_m3_day"}
    missing = sorted(required - values.keys())
    if errors or missing:
        suffix = errors + (["не заданы: " + ", ".join(missing)] if missing else [])
        await message.answer("Формат: <code>/reagent hazards=aspo,scale fluid=80 oil=8</code>\n· " + "\n· ".join(suffix))
        return
    try:
        hazards = tuple(x.strip().casefold() for x in values["hazards"].split(",") if x.strip())
        if not hazards:
            raise ValueError("hazards: укажите хотя бы один риск")
        fluid = _positive_decimal(values["treated_fluid_m3_day"], "fluid")
        oil = _positive_decimal(values["oil_m3_day"], "oil")
        products, envelopes = await asyncio.gather(
            asyncio.to_thread(CHEMICALS.list_products), asyncio.to_thread(CHEMICALS.list_envelopes),
        )
        rows = galit.recommend_products(products, envelopes, hazards, fluid, oil)
        for chunk in format_recommendations(rows):
            await message.answer(chunk)
    except (ValueError, galit.ChemicalStorageError) as exc:
        await message.answer("Подбор недоступен: " + html.escape(str(exc)))


@dp.message(Command("stock"))
async def cmd_stock(message: Message) -> None:
    values, errors = parse_chemical_command(message.text or "")
    if errors or "product_id" not in values:
        await message.answer("Формат: <code>/stock product=ID-или-название</code>")
        return
    try:
        product = await asyncio.to_thread(resolve_chemical_product, CHEMICALS, values["product_id"])
        stock = await asyncio.to_thread(CHEMICALS.stock, product.id)
        for chunk in format_stock(product, stock):
            await message.answer(chunk)
    except galit.ChemicalNotFoundError:
        await message.answer("Реагент не найден. Проверьте /reagents.")
    except (galit.ChemicalConflictError, galit.ChemicalStorageError) as exc:
        await message.answer("Остаток недоступен: " + html.escape(str(exc)))


@dp.message(Command("shortages"))
async def cmd_shortages(message: Message) -> None:
    values, errors = parse_chemical_command(message.text or "")
    if errors or not {"lead_time_days", "safety_stock_days"}.issubset(values):
        await message.answer("Формат: <code>/shortages lead=14 safety=7</code>. Оба горизонта обязательны.")
        return
    try:
        lead = _nonnegative_int(values["lead_time_days"], "lead")
        safety = _nonnegative_int(values["safety_stock_days"], "safety")
        rows = await asyncio.to_thread(
            chemical_shortage_rows, CHEMICALS, lead_time_days=lead,
            safety_stock_days=safety, as_of=date.today(),
        )
        for chunk in format_shortages(rows, lead, safety):
            await message.answer(chunk)
    except (ValueError, galit.ChemicalStorageError) as exc:
        await message.answer("Дефициты не рассчитаны: " + html.escape(str(exc)))


@dp.message(Command("reserve"))
async def cmd_reserve(message: Message, state: FSMContext) -> None:
    values, errors = parse_chemical_command(message.text or "")
    if errors:
        await message.answer("Параметры не приняты: " + "; ".join(errors))
        return
    try:
        preview = await asyncio.to_thread(
            build_reservation_preview, values, CHEMICALS,
            idempotency_key=f"telegram-reserve-{uuid4().hex}",
        )
        await state.set_state(ChemicalMutation.confirming)
        await state.update_data(chemical_preview=preview)
        await message.answer(format_mutation_preview(preview), reply_markup=CANCEL_MENU)
    except (ValueError, galit.ChemicalNotFoundError, galit.ChemicalConflictError,
            galit.ChemicalStorageError) as exc:
        await message.answer("Резерв не подготовлен: " + html.escape(str(exc)))


@dp.message(Command("transaction"))
async def cmd_transaction(message: Message, state: FSMContext) -> None:
    values, errors = parse_chemical_command(message.text or "")
    if errors:
        await message.answer("Параметры не приняты: " + "; ".join(errors))
        return
    try:
        preview = await asyncio.to_thread(
            build_transaction_preview, values, CHEMICALS,
            idempotency_key=f"telegram-transaction-{uuid4().hex}",
        )
        await state.set_state(ChemicalMutation.confirming)
        await state.update_data(chemical_preview=preview)
        await message.answer(format_mutation_preview(preview), reply_markup=CANCEL_MENU)
    except (ValueError, galit.ChemicalNotFoundError, galit.ChemicalConflictError,
            galit.ChemicalStorageError) as exc:
        await message.answer("Операция не подготовлена: " + html.escape(str(exc)))


@dp.message(ChemicalMutation.confirming, F.text.casefold() == "подтвердить")
async def confirm_chemical_mutation(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    preview = data.get("chemical_preview")
    if not preview:
        await state.clear()
        await message.answer("Предпросмотр отсутствует; данные не изменены.", reply_markup=MAIN_MENU)
        return
    try:
        result = await asyncio.to_thread(execute_chemical_preview, CHEMICALS, preview)
        if preview["action"] == "reserve":
            text = f"Резерв подтверждён: <code>{html.escape(result.id)}</code> · revision={result.revision}."
        else:
            count = len(result) if isinstance(result, list) else 1
            text = f"Складская операция подтверждена · записей: {count}."
        await state.clear()
        await message.answer(text, reply_markup=MAIN_MENU)
    except (ValueError, galit.ChemicalNotFoundError, galit.ChemicalConflictError,
            galit.ChemicalStorageError) as exc:
        await state.clear()
        await message.answer("Изменение не выполнено: " + html.escape(str(exc)), reply_markup=MAIN_MENU)


@dp.message(ChemicalMutation.confirming)
async def reject_chemical_mutation(message: Message, state: FSMContext) -> None:
    if (message.text or "").casefold() in {CANCEL.casefold(), "/cancel"}:
        await state.clear()
        await message.answer("Изменение отменено. Данные не изменены.", reply_markup=MAIN_MENU)
    else:
        await message.answer("Ответьте <b>Подтвердить</b> или <b>Отмена</b>.", reply_markup=CANCEL_MENU)


@dp.message(Command("treatment_add"))
async def cmd_treatment_add(message: Message) -> None:
    required = {"well", "complication_type", "description", "reagent", "dosage",
                "dosage_unit", "cost", "currency", "treatment_type"}
    values, errors = parse_treatment_command(message.text or "", required)
    if errors:
        await message.answer("Параметры не приняты:\n· " + "\n· ".join(errors) +
                             "\nПример: <code>/treatment_add well=\"Скважина 12\" event=АСПО description=\"Промывка НКТ\" reagent=R1 dose=2 unit=l/m3 cost=100 currency=BYN type=промывка group=\"куст 1\"</code>")
        return
    try:
        record = galit.new_treatment(
            well_id=values["well"], well_name=values["well"], event_at=datetime.now(timezone.utc),
            complication_type=values["complication_type"], description=values["description"],
            reagent_name=values["reagent"], reagent_id=None, dosage=float(values["dosage"].replace(",", ".")),
            dosage_unit=values["dosage_unit"], cost=float(values["cost"].replace(",", ".")),
            currency=values["currency"], treatment_type=values["treatment_type"],
            expected_result=values.get("expected_result"), source="telegram", well_group=values.get("well_group"),
            field_name=values.get("field_name"), cluster=values.get("cluster"), site=values.get("site"),
            rate_before_m3_day=(float(values["rate_before_m3_day"].replace(",", "."))
                                if values.get("rate_before_m3_day") else None),
        )
        saved = await asyncio.to_thread(TREATMENTS.create, record)
        await message.answer(f"План сохранён: <code>{html.escape(saved.id)}</code> · revision={saved.revision}.\n"
                             f"Для явного запуска: <code>/treatment_start id={html.escape(saved.id)} revision={saved.revision}</code>")
    except (ValueError, galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


async def _transition_treatment(message: Message, target: galit.TreatmentStatus) -> None:
    values, errors = parse_treatment_command(message.text or "", {"id", "revision"})
    if errors:
        await message.answer("Параметры не приняты:\n· " + "\n· ".join(errors))
        return
    try:
        revision = int(values["revision"])
        record = await asyncio.to_thread(resolve_treatment, TREATMENTS, values["id"])
        if record.revision != revision:
            raise galit.TreatmentConflictError("stale revision")
        updated = record.transition(target)
        saved = await asyncio.to_thread(TREATMENTS.update, updated, expected_revision=revision)
        await message.answer(f"Статус подтверждён: <b>{saved.status.value}</b> · revision={saved.revision}.")
    except (ValueError, galit.TreatmentNotFoundError, galit.TreatmentConflictError,
            galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


@dp.message(Command("treatment_start"))
async def cmd_treatment_start(message: Message) -> None:
    await _transition_treatment(message, galit.TreatmentStatus.IN_PROGRESS)


@dp.message(Command("treatment_complete"))
async def cmd_treatment_complete(message: Message) -> None:
    await _transition_treatment(message, galit.TreatmentStatus.COMPLETED)


@dp.message(Command("treatment_result"))
async def cmd_treatment_result(message: Message) -> None:
    required = {"id", "revision", "actual_result", "metric", "metric_value", "success",
                "effect_duration_days", "recurrence"}
    values, errors = parse_treatment_command(message.text or "", required)
    if errors:
        await message.answer("Параметры не приняты:\n· " + "\n· ".join(errors))
        return
    try:
        revision = int(values["revision"])
        record = await asyncio.to_thread(resolve_treatment, TREATMENTS, values["id"])
        if record.revision != revision:
            raise galit.TreatmentConflictError("stale revision")
        if record.status is not galit.TreatmentStatus.COMPLETED:
            raise ValueError("сначала явно выполните /treatment_start, затем /treatment_complete")
        recurrence = parse_bool(values["recurrence"], "recurrence")
        recurrence_date = parse_utc_date(values["recurrence_date"]) if values.get("recurrence_date") else None
        if recurrence and recurrence_date is None:
            raise ValueError("при recurrence=true требуется recurrence_date=YYYY-MM-DD")
        updated = record.transition(
            galit.TreatmentStatus.ASSESSED, actual_result=values["actual_result"],
            result_metrics={values["metric"]: float(values["metric_value"].replace(",", "."))},
            success=parse_bool(values["success"], "success"),
            effect_duration_days=float(values["effect_duration_days"].replace(",", ".")),
            recurrence=recurrence, recurrence_date=recurrence_date,
            rate_after_m3_day=(float(values["rate_after_m3_day"].replace(",", "."))
                               if values.get("rate_after_m3_day") else record.rate_after_m3_day),
        )
        saved = await asyncio.to_thread(TREATMENTS.update, updated, expected_revision=revision)
        await message.answer(f"Фактический результат зафиксирован: <b>assessed</b> · revision={saved.revision}.")
    except (ValueError, galit.TreatmentNotFoundError, galit.TreatmentConflictError,
            galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


@dp.message(Command("treatment"))
async def cmd_treatment(message: Message) -> None:
    values, errors = parse_treatment_command(message.text or "", {"id"})
    if errors:
        await message.answer("Укажите <code>/treatment id=...</code>")
        return
    try:
        record = await asyncio.to_thread(resolve_treatment, TREATMENTS, values["id"])
        for chunk in format_treatment_card(record):
            await message.answer(chunk)
    except (galit.TreatmentNotFoundError, galit.TreatmentConflictError, galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


@dp.message(Command("treatments"))
async def cmd_treatments(message: Message) -> None:
    values, errors = parse_treatment_command(message.text or "")
    allowed = {"well", "status", "complication_type", "reagent", "currency", "well_group", "limit"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        errors.append("фильтр не поддерживается: " + ", ".join(unknown))
    if errors:
        await message.answer("Параметры не приняты:\n· " + "\n· ".join(errors))
        return
    try:
        limit = int(values.get("limit", "10"))
        records = await asyncio.to_thread(
            TREATMENTS.list, well=values.get("well"),
            status=galit.TreatmentStatus(values["status"]) if values.get("status") else None,
            complication_type=values.get("complication_type"), reagent=values.get("reagent"),
            currency=values.get("currency"), well_group=values.get("well_group"), limit=min(limit, 50),
        )
        for chunk in format_treatments(records):
            await message.answer(chunk)
    except (ValueError, galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


@dp.message(Command("treatment_compare"))
async def cmd_treatment_compare(message: Message) -> None:
    required = {"reagent_a", "reagent_b", "complication_type", "well_group"}
    values, errors = parse_treatment_command(message.text or "", required)
    if errors:
        await message.answer("Пример: <code>/treatment_compare a=R1 b=R2 event=АСПО group=\"куст 1\" metric=success_rate</code>\n· " + "\n· ".join(errors))
        return
    try:
        records = await asyncio.to_thread(TREATMENTS.list)
        result = galit.compare_reagents(
            records, values["reagent_a"], values["reagent_b"],
            metric=values.get("metric", "success_rate"),
            min_sample_size=int(values.get("min_sample_size", galit.DEFAULT_MIN_SAMPLE_SIZE)),
            complication_type=values["complication_type"], well_group=values["well_group"],
        )
        for chunk in format_treatment_comparison(result):
            await message.answer(chunk)
    except (ValueError, galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


@dp.message(Command("treatment_cancel"))
async def cmd_treatment_cancel(message: Message) -> None:
    values, errors = parse_treatment_command(message.text or "", {"id", "revision"})
    if errors:
        await message.answer("Укажите <code>/treatment_cancel id=... revision=...</code>")
        return
    try:
        record = await asyncio.to_thread(resolve_treatment, TREATMENTS, values["id"])
        saved = await asyncio.to_thread(TREATMENTS.archive, record.id, expected_revision=int(values["revision"]))
        await message.answer(f"Мероприятие отменено и архивировано · revision={saved.revision}.")
    except (ValueError, galit.TreatmentNotFoundError, galit.TreatmentConflictError,
            galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


@dp.message(Command("treatment_stats"))
async def cmd_treatment_stats(message: Message) -> None:
    values, errors = parse_treatment_command(message.text or "")
    allowed = {"well", "min_sample_size"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        errors.append("параметр не поддерживается: " + ", ".join(unknown))
    if errors:
        await message.answer("Параметры не приняты:\n· " + "\n· ".join(errors))
        return
    try:
        min_n = int(values.get("min_sample_size", galit.DEFAULT_MIN_SAMPLE_SIZE))
        records = await asyncio.to_thread(TREATMENTS.list, well=values.get("well"))
        for chunk in format_treatment_stats(records, min_sample_size=min_n):
            await message.answer(chunk)
    except (ValueError, galit.TreatmentStorageError) as exc:
        await _answer_treatment_error(message, exc)


@dp.message(Command("passport"))
async def cmd_passport(message: Message) -> None:
    values, errors = parse_passport_command(message.text or "")
    well = values.get("well", "")
    if errors or not well:
        await message.answer("Формат: <code>/passport well=\"Скважина 12\"</code>")
        return
    try:
        events = await asyncio.to_thread(PASSPORTS.list, well=well)
        treatments = await asyncio.to_thread(TREATMENTS.list, well=well)
        for chunk in format_passport_summary(well, events, treatments):
            await message.answer(chunk)
    except (galit.PassportStorageError, galit.TreatmentStorageError) as exc:
        await message.answer("Паспорт временно недоступен: " + html.escape(str(exc)))


@dp.message(Command("passport_history"))
async def cmd_passport_history(message: Message) -> None:
    values, errors = parse_passport_command(message.text or "")
    well = values.get("well", "")
    if errors or not well:
        await message.answer("Формат: <code>/passport_history well=Имя limit=10</code>")
        return
    try:
        limit = min(max(int(values.get("limit", "10")), 1), 50)
        events = await asyncio.to_thread(PASSPORTS.list, well=well, limit=limit)
        treatments = await asyncio.to_thread(TREATMENTS.list, well=well, limit=limit)
        for chunk in format_passport_history(events, treatments):
            await message.answer(chunk)
    except (ValueError, galit.PassportStorageError, galit.TreatmentStorageError) as exc:
        await message.answer("Параметры не приняты: " + html.escape(str(exc)))


@dp.message(Command("passport_add"))
async def cmd_passport_add(message: Message) -> None:
    values, errors = parse_passport_command(message.text or "")
    required = {"well", "event_type", "title", "text"}
    missing = sorted(required - values.keys())
    if errors or missing:
        await message.answer("Формат: <code>/passport_add well=Имя type=complication title=Заголовок text=Описание</code>")
        return
    try:
        kind = galit.PassportEventType(values["event_type"])
        if kind in {galit.PassportEventType.DEPOSIT_PHOTO, galit.PassportEventType.LAB_REPORT,
                    galit.PassportEventType.RATE_CHANGE}:
            raise ValueError("для вложений используйте сайт/API, для дебита — /passport_rate")
        data_key = "complication_type" if kind is galit.PassportEventType.COMPLICATION else "description"
        event = galit.new_passport_event(
            well_id=values["well"], well_name=values["well"], event_type=kind,
            event_at=datetime.now(timezone.utc), title=values["title"],
            data={data_key: values["text"]}, notes=values["text"], source="telegram",
        )
        await asyncio.to_thread(PASSPORTS.create, event)
        await message.answer("Событие добавлено в паспорт.")
    except (ValueError, galit.PassportStorageError) as exc:
        await message.answer("Событие не сохранено: " + html.escape(str(exc)))


@dp.message(Command("passport_rate"))
async def cmd_passport_rate(message: Message) -> None:
    values, errors = parse_passport_command(message.text or "")
    well = values.get("well", "")
    rate_keys = ("oil_rate_m3d", "water_rate_m3d", "gas_rate_m3d")
    if errors or not well or not any(key in values for key in rate_keys):
        await message.answer("Формат: <code>/passport_rate well=Имя oil=10 water=20 gas=0</code>")
        return
    try:
        data = {key: float(values[key].replace(",", ".")) for key in rate_keys if key in values}
        event = galit.new_passport_event(
            well_id=well, well_name=well, event_type=galit.PassportEventType.RATE_CHANGE,
            event_at=datetime.now(timezone.utc), title="Снимок дебита", data=data,
            source="telegram",
        )
        await asyncio.to_thread(PASSPORTS.create, event)
        await message.answer("Снимок дебита добавлен в паспорт.")
    except (ValueError, galit.PassportStorageError) as exc:
        await message.answer("Снимок не сохранён: " + html.escape(str(exc)))


def parse_twin_command(text: str, *, allow_days: bool = False) -> tuple[str, int, list[str]]:
    """Strict parser for /twin, /timeline and /changes."""
    try: tokens = shlex.split(text, posix=True)[1:]
    except ValueError as exc: return "", 30, ["ошибка кавычек: " + str(exc)]
    days = 30
    if allow_days and tokens and tokens[-1].isdigit():
        days = int(tokens.pop())
        if not 1 <= days <= 3650: return "", days, ["days должен быть 1..3650"]
    well = " ".join(tokens).strip()
    return well, days, ([] if well else ["укажите скважину"])


def format_twin_snapshot(snapshot: galit.TwinSnapshot, events: list[dict] | None = None) -> list[str]:
    e = html.escape; blocks = [f"<b>Цифровой двойник · {e(snapshot.well.display_name)}</b>",
        f"Состояние: <b>{e(snapshot.state)}</b> · health: " + ("—" if snapshot.health_score is None else f"{snapshot.health_score:.0%}"),
        "Драйверы: " + ("; ".join(e(x) for x in snapshot.drivers) or "недостаточно данных"),
        "Missing: " + (", ".join(e(x) for x in snapshot.missing_sources) or "нет") +
        " · stale: " + (", ".join(e(x) for x in snapshot.stale_sources) or "нет")]
    for item in (events or [])[:5]: blocks.append(f"{e(item['occurred_at'][:10])} · {e(item['title'])} [{e(item['category'])}]")
    blocks.append(e(galit.ASSOCIATION_DISCLAIMER)); return _forecast_chunks(blocks)


def format_twin_changes(changes: list[galit.ChangeExplanation]) -> list[str]:
    blocks = ["<b>Почему изменилось состояние</b>"]
    if not changes: blocks.append("Недостаточно датированных событий для сопоставления.")
    for item in changes[:10]: blocks.append(f"{html.escape(item.occurred_at.date().isoformat())} · {html.escape(item.title)}\n{html.escape(item.statement)} · confidence={item.confidence}")
    blocks.append(html.escape(galit.ASSOCIATION_DISCLAIMER)); return _forecast_chunks(blocks)


@dp.message(Command("twin"))
async def cmd_twin(message: Message) -> None:
    well, _, errors = parse_twin_command(message.text or "")
    if errors: await message.answer("Формат: <code>/twin Скважина 12</code>"); return
    try:
        service=twin_service(); snapshot=await asyncio.to_thread(service.snapshot,well); timeline=await asyncio.to_thread(service.timeline,well,limit=5)
        for chunk in format_twin_snapshot(snapshot,timeline["items"]): await message.answer(chunk)
    except (galit.TwinNotFoundError,galit.TwinAmbiguousError,galit.TwinStorageError) as exc: await message.answer(html.escape(str(exc)))


@dp.message(Command("timeline"))
async def cmd_timeline(message: Message) -> None:
    well, days, errors = parse_twin_command(message.text or "",allow_days=True)
    if errors: await message.answer("Формат: <code>/timeline Скважина 12 30</code>"); return
    try:
        result=await asyncio.to_thread(twin_service().timeline,well,date_from=datetime.now(timezone.utc)-__import__('datetime').timedelta(days=days),limit=20)
        snapshot=await asyncio.to_thread(twin_service().snapshot,well)
        for chunk in format_twin_snapshot(snapshot,result["items"]): await message.answer(chunk)
    except (galit.TwinNotFoundError,galit.TwinAmbiguousError,galit.TwinStorageError) as exc: await message.answer(html.escape(str(exc)))


@dp.message(Command("changes"))
async def cmd_changes(message: Message) -> None:
    well, _, errors = parse_twin_command(message.text or "")
    if errors: await message.answer("Формат: <code>/changes Скважина 12</code>"); return
    try:
        rows=await asyncio.to_thread(twin_service().changes,well)
        for chunk in format_twin_changes(rows): await message.answer(chunk)
    except (galit.TwinNotFoundError,galit.TwinAmbiguousError,galit.TwinStorageError) as exc: await message.answer(html.escape(str(exc)))


@dp.message(Command("plan"))
async def cmd_plan(message: Message) -> None:
    chat_id = message.chat.id if message.chat else 0
    for chunk in format_plan_messages(RECENT_DIAGNOSES.get(chat_id)):
        await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)


@dp.message(Command("map"))
async def cmd_map(message: Message) -> None:
    chat_id = message.chat.id if message.chat else 0
    for chunk in format_map_messages(RECENT_DIAGNOSES.get(chat_id)):
        await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)


@dp.message(Command("equipment"))
async def cmd_equipment(message: Message) -> None:
    try:
        query = parse_equipment_query(message.text or "")
        rows = ([await asyncio.to_thread(EQUIPMENT.forecast, query)] if query
                else await asyncio.to_thread(EQUIPMENT.portfolio))
        for chunk in format_equipment_messages(rows):
            await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)
    except (ValueError, galit.EquipmentNotFoundError, galit.EquipmentStorageError) as exc:
        await message.answer(html.escape(str(exc)), reply_markup=MAIN_MENU)


@dp.message(Command("failures"))
async def cmd_failures(message: Message) -> None:
    try:
        rows = [row for row in await asyncio.to_thread(EQUIPMENT.portfolio)
                if row.risk_level in {"warning", "critical"}]
        for chunk in format_equipment_messages(rows, title="Риски отказов ЭЦН/ШГН"):
            await message.answer(chunk, reply_markup=MAIN_MENU)
    except galit.EquipmentStorageError as exc:
        await message.answer(html.escape(str(exc)))


@dp.message(Command("maintenance"))
async def cmd_maintenance(message: Message) -> None:
    try:
        rows = [row for row in await asyncio.to_thread(EQUIPMENT.portfolio)
                if row.risk_level in {"warning", "critical"}]
        for chunk in format_equipment_messages(rows, title="Приоритеты обслуживания"):
            await message.answer(chunk, reply_markup=MAIN_MENU)
    except galit.EquipmentStorageError as exc:
        await message.answer(html.escape(str(exc)))


@dp.message(Command("forecast"))
async def cmd_forecast(message: Message) -> None:
    chat_id = message.chat.id if message.chat else 0
    query = (message.text or "").partition(" ")[2].strip()
    try:
        item = select_forecast_diagnosis(RECENT_DIAGNOSES.get(chat_id), query)
    except LookupError as exc:
        await message.answer(html.escape(str(exc)), reply_markup=MAIN_MENU)
        return
    for chunk in format_forecast_messages(item):
        await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)


@dp.message(Command("scenario"))
async def cmd_scenario(message: Message) -> None:
    args = (message.text or "").split()[1:]
    values, errors = parse_scenario_args(args)
    if errors:
        example = "/scenario oil_pct=-10 temperature=2 wash=yes well=Речицкая-123"
        await message.answer("Параметры сценария не приняты:\n· " + "\n· ".join(errors) +
                             "\n\nПример:\n<code>" + example + "</code>", reply_markup=MAIN_MENU)
        return
    chat_id = message.chat.id if message.chat else 0
    try:
        item = select_forecast_diagnosis(RECENT_DIAGNOSES.get(chat_id), str(values.get("well", "")))
        chunks = format_scenario_messages(item, values)
    except (LookupError, ValueError) as exc:
        await message.answer(html.escape(str(exc)), reply_markup=MAIN_MENU)
        return
    for chunk in chunks:
        await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)


@dp.message(Command("economics"))
async def cmd_economics(message: Message) -> None:
    args = (message.text or "").split()[1:]
    values, errors = parse_economics_args(args)
    if errors:
        example = ("/economics probability=1 horizon=31 efficiency=1 event_days=0 "
                   "treatment_days=0 price=1000 operating=0 cost=8000 currency=BYN")
        await message.answer("Параметры экономики не приняты:\n· " + "\n· ".join(errors) +
                             "\n\nПример:\n<code>" + example + "</code>", reply_markup=MAIN_MENU)
        return
    chat_id = message.chat.id if message.chat else 0
    try:
        item = select_forecast_diagnosis(
            RECENT_DIAGNOSES.get(chat_id), str(values.get("well", "")),
        )
        chunks = format_economics_messages(item, values)
    except (LookupError, ValueError) as exc:
        await message.answer(html.escape(str(exc)), reply_markup=MAIN_MENU)
        return
    for chunk in chunks:
        await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)


@dp.message(Command("plan_clear"))
async def cmd_plan_clear(message: Message) -> None:
    chat_id = message.chat.id if message.chat else 0
    removed = RECENT_DIAGNOSES.clear(chat_id)
    await message.answer(f"История плана очищена: {removed} расч.", reply_markup=MAIN_MENU)


async def _send_compatibility(message: Message, water_a: galit.CompatibilityWater,
                              water_b: galit.CompatibilityWater, *, profile=None,
                              dose_response=None) -> None:
    result = await asyncio.to_thread(galit.evaluate_compatibility, water_a, water_b,
                                     profile=profile, dose_response=dose_response)
    for chunk in format_compatibility_report(result, water_a, water_b):
        await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)


@dp.message(Command("compatibility"))
async def cmd_compatibility(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(CompatibilityCalculation.collecting)
    await state.update_data(compatibility_waters=[])
    await message.answer(COMPATIBILITY_PROMPT.format(label="A"), reply_markup=CANCEL_MENU)


@dp.message(CompatibilityCalculation.collecting, F.document)
async def collect_compatibility_document(message: Message, state: FSMContext) -> None:
    try:
        if not message.document or message.document.file_size and message.document.file_size > 5_000_000:
            raise ValueError("файл отсутствует или больше 5 МБ")
        buffer = io.BytesIO()
        await message.bot.download(message.document, destination=buffer)
        (water_a, water_b), profile, curve = parse_compatibility_document(
            buffer.getvalue(), message.document.file_name or "")
        await state.clear()
        await _send_compatibility(message, water_a, water_b, profile=profile, dose_response=curve)
    except (ValueError, KeyError, UnicodeError, ImportError) as exc:
        await message.answer("Документ не принят: " + html.escape(str(exc)), reply_markup=CANCEL_MENU)


@dp.message(CompatibilityCalculation.collecting)
async def collect_compatibility_water(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    waters = list(data.get("compatibility_waters", []))
    try:
        water = parse_compatibility_water(message.text or "", "Вода A" if not waters else "Вода B")
    except ValueError as exc:
        await message.answer("Вода не принята: " + html.escape(str(exc)) + "\n\n" +
                             COMPATIBILITY_PROMPT.format(label="A" if not waters else "B"),
                             reply_markup=CANCEL_MENU)
        return
    waters.append(water)
    if len(waters) == 1:
        await state.update_data(compatibility_waters=waters)
        await message.answer(COMPATIBILITY_PROMPT.format(label="B"), reply_markup=CANCEL_MENU)
        return
    await state.clear()
    await _send_compatibility(message, waters[0], waters[1])


@dp.message(Command("aspo"))
async def cmd_aspo(message: Message) -> None:
    args = (message.text or "").split()[1:]
    if not args:
        await message.answer("Формат: /aspo глубина НКТ нефть вода ГФ WAT\n"
                             "Пример: /aspo 3200 62 8 72 65 34",
                             reply_markup=MAIN_MENU)
        return
    params, errors = parse_args(args)
    if errors:
        await message.answer(
            "Параметры не приняты:\n· " + "\n· ".join(errors) +
            "\n\nПодробнее — /help", reply_markup=MAIN_MENU,
        )
        return
    await send_calculation(message, params)


@dp.message(Calculation.collecting)
async def collect_calculation_value(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    step = int(data.get("step", 0))
    key, _ = FSM_FIELDS[step]
    value, error = validate_fsm_value(key, message.text or "")
    if error:
        await message.answer(error, reply_markup=CANCEL_MENU)
        return

    params = dict(data.get("params", {}))
    params[key] = value
    step += 1
    if step < len(FSM_FIELDS):
        await state.update_data(step=step, params=params)
        await message.answer(FSM_FIELDS[step][1], reply_markup=CANCEL_MENU)
        return

    await state.clear()
    _, errors = parse_args([str(params[key]) for key in REQUIRED_KEYS])
    if errors:
        await message.answer("Параметры не приняты:\n· " + "\n· ".join(errors),
                             reply_markup=MAIN_MENU)
        return
    await message.answer("Данные приняты. Выполняю расчёт…",
                         reply_markup=ReplyKeyboardRemove())
    await send_calculation(message, params)


@dp.message()
async def fallback(message: Message) -> None:
    await message.answer(
        "Выберите действие в меню. Доступны /aspo, /forecast, /plan, /plan_clear; "
        "описание — /help.", reply_markup=MAIN_MENU,
    )


async def main() -> None:
    load_dotenv()
    bot_token = os.environ.get("GALIT_BOT_TOKEN", "").strip()
    if not bot_token:
        print("Не задан токен бота. Получите его у @BotFather и выполните:\n"
              "    set GALIT_BOT_TOKEN=123456:ABC...\n"
              "(Windows) или export GALIT_BOT_TOKEN=... (Linux/macOS)")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = Bot(token=bot_token,
              default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await bot.delete_webhook(drop_pending_updates=True)
    print("ГАЛИТ-бот запущен. Остановка — Ctrl+C.")
    try:
        await dp.start_polling(bot)
    except TelegramUnauthorizedError:
        print("Токен отклонён Telegram (401). Проверьте GALIT_BOT_TOKEN у @BotFather.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
