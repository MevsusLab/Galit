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
import html
import logging
import math
from collections import OrderedDict, deque
import os
import shlex
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

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
    "группа": "well_group", "limit": "limit", "лимит": "limit", "min_n": "min_sample_size",
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


def format_treatment_card(item: galit.TreatmentRecord) -> list[str]:
    e = html.escape
    metrics = ", ".join(f"{e(k)}={v:g}" for k, v in item.result_metrics.items()) or "—"
    result = e(item.actual_result) if item.actual_result else "—"
    recurrence = "—" if item.recurrence is None else ("да" if item.recurrence else "нет")
    if item.recurrence_date:
        recurrence += f" ({item.recurrence_date.date().isoformat()})"
    blocks = [
        f"<b>Мероприятие {e(item.id[:8])}</b>",
        f"Скважина: {e(item.well_name)}\nОсложнение: {e(item.complication_type)}\n"
        f"Группа: {e(item.well_group or '—')}\nСтатус: <b>{item.status.value}</b> · revision={item.revision}",
        f"Описание: {e(item.description)}\nРеагент: {e(item.reagent_name)}\n"
        f"Доза: {item.dosage:g} {e(item.dosage_unit)}\nСтоимость: {item.cost:g} {e(item.currency)}",
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
        effect = (f"успех={'да' if item.success else 'нет'}; {item.effect_duration_days:g} сут"
                  if item.status is galit.TreatmentStatus.ASSESSED else "факт не оценён")
        blocks.append(f"<code>{html.escape(item.id[:8])}</code> · {html.escape(item.well_name)} · "
                      f"{html.escape(item.reagent_name)} · {item.status.value} · rev={item.revision}\n{html.escape(effect)}")
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


def format_treatment_stats(records: list[galit.TreatmentRecord]) -> list[str]:
    summary = galit.treatment_summary(records, "reagent")
    if summary["status"] == "insufficient_data":
        return ["insufficient_data: оценённых наблюдений пока нет."]
    blocks = ["<b>Фактическая эффективность</b>"]
    for row in summary["groups"]:
        success = "—" if row["success_rate"] is None else f"{row['success_rate']:.0%}"
        blocks.append(f"{html.escape(row['group'])}: n={row['assessed_observations']}; "
                      f"успех={success}; confidence={row['confidence']}")
    blocks.append(html.escape(summary["observational_warning"]))
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
    "Последние расчёты: /plan; прогноз: /forecast [скважина]; сценарий: /scenario; экономика: /economics; очистка — /plan_clear."
)
HELP_TEXT = (
    "<b>Справка</b>\n\n"
    "ГАЛИТ выполняет предварительную оценку АСПО и помогает выбрать "
    "следующее действие. Результат требует инженерной проверки.\n\n"
    "<b>Пошагово:</b> нажмите «🔎 Новый расчёт» и введите 6 значений: "
    "глубина (м), НКТ (мм), нефть и вода (м³/сут), газовый фактор "
    "(м³/м³), WAT (°C). Доступна отмена.\n\n"
    "<b>Быстрая команда</b>\n"
    "<code>/aspo глубина НКТ нефть вода ГФ WAT</code>\n"
    "Пример: <code>/aspo 3200 62 8 72 65 34</code>\n\n"
    "Дополнительно: <code>скважина=</code>, <code>способ=</code>, "
    "<code>парафин=</code>, <code>температура=</code>, "
    "<code>градиент=</code>, <code>co2=</code>, <code>буферное=</code>. "
    "Десятичный разделитель — точка или запятая.\n\n"
    "<b>Команды:</b> <code>/plan</code>, <code>/forecast</code>, "
    "<code>/economics</code>, <code>/plan_clear</code>.\n"
    "Паспорт: <code>/passport</code>, <code>/passport_history</code>, "
    "<code>/passport_add</code>, <code>/passport_rate</code>.\n"
    "Сценарий: <code>/scenario oil_pct=-10 temperature=2 wash=yes</code>; "
    "выбор: <code>well=Имя</code>; явный эффект: "
    "<code>inhibitor_effect=0.8 source=паспорт</code>. История локальная и исчезает при перезапуске."
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
    try:
        records = await asyncio.to_thread(TREATMENTS.list)
        for chunk in format_treatment_stats(records):
            await message.answer(chunk)
    except galit.TreatmentStorageError as exc:
        await message.answer(html.escape(str(exc)))


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


@dp.message(Command("plan"))
async def cmd_plan(message: Message) -> None:
    chat_id = message.chat.id if message.chat else 0
    for chunk in format_plan_messages(RECENT_DIAGNOSES.get(chat_id)):
        await message.answer(chunk, disable_web_page_preview=True, reply_markup=MAIN_MENU)


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
