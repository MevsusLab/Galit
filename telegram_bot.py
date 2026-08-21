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
    "Чтобы начать, нажмите «🔎 Новый расчёт»."
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
    "Десятичный разделитель — точка или запятая."
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
        "Выберите действие в меню. Для быстрого расчёта доступна команда /aspo, "
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
