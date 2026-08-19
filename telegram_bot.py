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
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

import galit
from galit import DataProvenance, WellCase, diagnose
from galit.wax import recommend_wax_treatment
from galit.wellbore import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
)
from galit.scale import WaterAnalysis
from galit.wax import WaxProperties

BOT_TOKEN = os.environ.get("GALIT_BOT_TOKEN", "")

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


def format_report(result: galit.DiagnosisResult, case: WellCase,
                  treatment: str) -> str:
    """Строго структурированный ответ мастеру (HTML, без эмодзи)."""
    e = html.escape
    lines = [
        f"Скважина: <b>{e(result.well)}</b>",
        "<b>Режим: screening.</b> Типовая вода и значения по умолчанию "
        "не являются промышленным прогнозом.",
        f"Качество данных: {result.quality.grade} "
        f"({result.quality.completeness:.0%})",
        "",
    ]

    if result.wax_onset_m is None:
        lines += [
            "Глубина начала АСПО: <b>не прогнозируется</b> — "
            "поток по всему стволу теплее WAT",
        ]
    else:
        onset = result.wax_onset_m
        depth = case.geometry.depth_m
        zone_pct = 100.0 * onset / depth if depth else 0.0
        lines += [
            f"Глубина начала АСПО: <b>{onset:.0f} м</b> от устья",
            f"Зона отложений: устье — {onset:.0f} м ({zone_pct:.0f} % ствола)",
        ]

    sev = result.severity["wax"]
    lines += [
        f"Тяжесть АСПО: {sev:.2f}",
        f"Интегральный риск: {result.integrated_risk:.2f} "
        f"(доминирует: {e(_mech_ru(result.dominant))})",
        "",
        f"<b>Обработка АСПО:</b> {e(treatment)}",
    ]

    if result.recommendation:
        lines += ["", f"Рекомендация ядра ГАЛИТ: {e(result.recommendation)}"]
    if result.warnings:
        lines += ["", "Предупреждения расчёта:"]
        lines += [f"· {e(w)}" for w in result.warnings]
    lines += ["", "ГАЛИТ — расчётная оценка, не заменяет промысловые исследования."]
    return "\n".join(lines)


def _mech_ru(mech: str) -> str:
    return {"halite": "галит", "calcite": "кальцит", "wax": "АСПО",
            "corrosion": "коррозия"}.get(mech, mech)


HELP_TEXT = (
    "<b>ГАЛИТ · экспресс-диагностика АСПО</b>\n"
    "\n"
    "<b>Команда расчёта</b>\n"
    "<code>/aspo глубина НКТ нефть вода ГФ WAT</code>\n"
    "позиционно: глубина, м · НКТ, мм · дебит нефти, м3/сут · "
    "дебит воды, м3/сут · газовый фактор, м3/м3 · WAT, °C\n"
    "\n"
    "Пример:\n"
    "<code>/aspo 3200 62 8 72 65 34</code>\n"
    "\n"
    "<b>Необязательные параметры</b> (добавляются через пробел):\n"
    "<code>скважина=Речицкая-123</code> — название для отчёта\n"
    "<code>способ=ШГН</code> — ЭЦН | ШГН | фонтан\n"
    "<code>парафин=6.5</code> — содержание парафина, % масс.\n"
    "<code>температура=8</code> — температура пород у поверхности, °C\n"
    "<code>градиент=0.033</code> — геотермический градиент, К/м\n"
    "<code>co2=0.012</code> — доля CO2 в попутном газе\n"
    "<code>буферное=1.4</code> — буферное давление, МПа\n"
    "\n"
    "Запятая в числах допускается как десятичный разделитель.\n"
    "Расчёт выполняется моделью ГАЛИТ (Ramey T(z), Beggs &amp; Brill P(z), "
    "WAT(P)); солевой состав принимается типовым для Припятского прогиба."
)

# ==========================================================================
# Обработчики aiogram
# ==========================================================================

dp = Dispatcher()


@dp.message(CommandStart())
@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT, disable_web_page_preview=True)


@dp.message(Command("aspo"))
async def cmd_aspo(message: Message) -> None:
    args = (message.text or "").split()[1:]
    if not args:
        await message.answer("Формат: /aspo глубина НКТ нефть вода ГФ WAT\n"
                             "Пример: /aspo 3200 62 8 72 65 34\n"
                             "Полный список параметров — /help")
        return

    params, errors = parse_args(args)
    if errors:
        await message.answer(
            "Параметры не приняты:\n· " + "\n· ".join(errors) +
            "\n\nФормат: /aspo 3200 62 8 72 65 34 — подробнее /help"
        )
        return

    case = build_case(params)
    try:
        # ядро CPU-bound: уводим из event loop, чтобы бот отвечал параллельно
        result = await asyncio.to_thread(diagnose, case)
    except (ValueError, KeyError) as exc:
        await message.answer(f"Расчёт не выполнен: {html.escape(str(exc))}")
        return

    await message.answer(
        format_report(result, case, wax_treatment(result, case)),
        disable_web_page_preview=True,
    )


@dp.message()
async def fallback(message: Message) -> None:
    await message.answer(
        "Расчёт АСПО: /aspo глубина НКТ нефть вода ГФ WAT\n"
        "Пример: /aspo 3200 62 8 72 65 34\nПолный список параметров — /help"
    )


async def main() -> None:
    if not BOT_TOKEN:
        print("Не задан токен бота. Получите его у @BotFather и выполните:\n"
              "    set GALIT_BOT_TOKEN=123456:ABC...\n"
              "(Windows) или export GALIT_BOT_TOKEN=... (Linux/macOS)")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = Bot(token=BOT_TOKEN,
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
