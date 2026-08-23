"""Тесты Telegram-бота: разбор /aspo, сборка кейса, отчёт, подбор обработки.

Сеть не используется -- проверяются чистые функции и консистентность
с расчётным ядром galit.
"""
from __future__ import annotations

import galit
import pytest
import telegram_bot as tb
from galit import diagnose

POSITIONAL = ["3200", "62", "8", "72", "65", "34"]


# ------------------------------------------------------------------- разбор

def test_parse_positional_args():
    params, errors = tb.parse_args(POSITIONAL)
    assert errors == []
    assert params == {
        "depth_m": 3200.0, "tubing_mm": 62.0, "q_oil_m3d": 8.0,
        "q_water_m3d": 72.0, "gor_m3m3": 65.0, "wat_c": 34.0,
    }


def test_parse_named_args_with_comma_decimal():
    params, errors = tb.parse_args(
        POSITIONAL + ["способ=шгн", "скважина=Речицкая-123", "парафин=6,5",
                      "co2=0.012", "буферное=1,4"]
    )
    assert errors == []
    assert params["lift_type"] == "ШГН"
    assert params["name"] == "Речицкая-123"
    assert params["wax_pct"] == 6.5
    assert params["co2_mol_frac"] == 0.012
    assert params["p_wellhead_mpa"] == 1.4


def test_parse_missing_required_reports_all():
    _, errors = tb.parse_args(["3200"])
    assert any("не задано" in e for e in errors)
    assert any("НКТ" in e for e in errors)


def test_parse_rejects_out_of_range():
    _, errors = tb.parse_args(["9000", "62", "8", "72", "65", "34"])
    assert any("глубина" in e for e in errors)
    _, errors = tb.parse_args(["3200", "5", "8", "72", "65", "34"])
    assert any("НКТ" in e for e in errors)


def test_parse_rejects_zero_liquid_rate():
    _, errors = tb.parse_args(["3200", "62", "0", "0", "65", "34"])
    assert any("дебит" in e for e in errors)


def test_parse_unknown_key_and_bad_number():
    _, errors = tb.parse_args(POSITIONAL + ["foo=1", "парафин=много"])
    assert any("неизвестный параметр" in e for e in errors)
    assert any("не число" in e for e in errors)


def test_parse_bad_lift_type():
    _, errors = tb.parse_args(POSITIONAL + ["способ=гидробур"])
    assert any("ЭЦН | ШГН | фонтан" in e for e in errors)


# ------------------------------------------------------------- сборка кейса

def test_main_menu_is_small_and_professional():
    labels = [button.text for row in tb.MAIN_MENU.keyboard for button in row]
    assert labels == [tb.MENU_NEW, tb.MENU_HELP, tb.MENU_EXAMPLE]
    assert len(labels) == 3
    assert "/aspo" in tb.HELP_TEXT


def test_start_and_help_explain_six_steps_units_and_how_to_begin():
    assert "Шесть шагов" in tb.START_TEXT
    assert all(f"{number}." in tb.START_TEXT for number in range(1, 7))
    for unit in ("м", "мм", "м³/сут", "м³/м³", "°C"):
        assert unit in tb.START_TEXT
    assert tb.MENU_NEW in tb.START_TEXT
    assert "предварительную оценку" in tb.HELP_TEXT
    assert len(tb.START_TEXT) < 600
    assert len(tb.HELP_TEXT) < 1100


def test_fsm_field_order_and_value_validation():
    assert [key for key, _ in tb.FSM_FIELDS] == tb.REQUIRED_KEYS
    value, error = tb.validate_fsm_value("depth_m", "3200,5")
    assert value == 3200.5 and error is None
    value, error = tb.validate_fsm_value("tubing_mm", "не число")
    assert value is None and "число" in error
    value, error = tb.validate_fsm_value("wat_c", "100")
    assert value is None and "диапазона" in error


def test_step_values_produce_same_params_as_aspo():
    stepped = {}
    for (key, _), raw in zip(tb.FSM_FIELDS, POSITIONAL):
        value, error = tb.validate_fsm_value(key, raw)
        assert error is None
        stepped[key] = value
    command, errors = tb.parse_args(POSITIONAL)
    assert errors == []
    assert stepped == command


def test_import_does_not_load_bot_token():
    assert tb.BOT_TOKEN == ""


# ------------------------------------------------------------- сборка кейса

def test_build_case_units_and_defaults():
    params, _ = tb.parse_args(POSITIONAL)
    case = tb.build_case(params)
    assert case.geometry.tubing_id_m == 0.062        # мм -> м
    assert case.p_wellhead_pa == 1.2e6               # МПа -> Па
    assert case.lift_type == "ЭЦН"
    assert case.water.ions_mg_l == tb.TYPICAL_BRINE  # типовой рассол
    assert case.wax.wax_content_pct == 5.0


# -------------------------------------------------------------------- отчёт

def test_report_no_deposition_branch():
    params, _ = tb.parse_args(POSITIONAL + ["скважина=Речицкая 123"])
    case = tb.build_case(params)
    result = diagnose(case)
    assert result.wax_onset_m is None               # физика кейса: без отложений
    report = tb.format_report(result, case, tb.wax_treatment(result, case))
    assert "не прогнозируется" in report
    assert "<b>Рекомендуемое действие</b>" in report
    assert "Основное действие:" in report
    assert "<b>Речицкая 123</b>" in report


def test_report_deposition_branch():
    # WAT 48 °C -- выше типовой температуры потока на устье, отложения есть
    params, _ = tb.parse_args(["2900", "62", "6", "64", "45", "48"])
    case = tb.build_case(params)
    result = diagnose(case)
    assert result.wax_onset_m is not None
    report = tb.format_report(result, case, tb.wax_treatment(result, case))
    assert f"{result.wax_onset_m:.0f} м" in report
    assert "Зона:" in report
    assert "<b>Итог</b>" in report
    assert "Риск:" in report
    assert any(level in report for level in
               ("низкий", "умеренный", "высокий", "критический"))


def test_report_has_scan_friendly_blocks_and_summarized_warnings():
    params, _ = tb.parse_args(POSITIONAL)
    case = tb.build_case(params)
    result = diagnose(case)
    raw_warning = "Screening: " + "очень длинное техническое объяснение " * 30
    result.warnings = [raw_warning, "TDS 300 г/л -- LSI неприменим"]
    result.corrosion["rate_mm_yr"] = 0.876
    report = tb.format_report(result, case, "Проверить режим <сейчас>")

    for block in ("Предварительная оценка", "<b>Итог</b>", "<b>АСПО</b>",
                  "<b>Рекомендуемое действие</b>", "<b>Надёжность данных</b>",
                  "<b>Краткие ограничения</b>"):
        assert block in report
    assert "screening" not in report.lower()
    assert raw_warning not in report
    assert "Неполные/типовые данные" in report
    assert "Аномально высокая коррозия" in report
    assert report.count("• ") <= 4
    assert "&lt;сейчас&gt;" in report


def test_corrosion_small_nonzero_rate_is_not_rounded_to_zero():
    params, _ = tb.parse_args(POSITIONAL)
    case = tb.build_case(params)
    result = diagnose(case)
    result.corrosion["rate_mm_yr"] = 0.0004
    report = tb.format_report(result, case, "Наблюдение")
    assert "0.0004 мм/год" in report
    assert "Коррозия: <b>0 мм/год" not in report


def test_level_boundaries():
    assert [tb._level(value) for value in (0.0, 0.25, 0.5, 0.75)] == [
        "низкий", "умеренный", "высокий", "критический",
    ]


def test_report_escapes_well_name():
    params, _ = tb.parse_args(POSITIONAL + ["скважина=<script>x</script>"])
    case = tb.build_case(params)
    result = diagnose(case)
    report = tb.format_report(result, case, tb.wax_treatment(result, case))
    assert "<script>" not in report
    assert "&lt;script&gt;" in report


def test_recent_store_is_bounded_per_chat_and_clear_isolated():
    store = tb.RecentDiagnosisStore(per_chat_limit=2, chat_limit=2)
    params, _ = tb.parse_args(POSITIONAL)
    case = tb.build_case(params)
    item = galit.DiagnosedWell(case, diagnose(case))
    store.add(1, item)
    store.add(1, item)
    store.add(1, item)
    store.add(2, item)
    assert len(store.get(1)) == 2 and len(store.get(2)) == 1
    assert store.clear(1) == 2 and store.get(1) == [] and len(store.get(2)) == 1


def test_plan_format_is_safe_compact_and_documents_commands():
    params, _ = tb.parse_args(POSITIONAL + ["скважина=<A>"])
    case = tb.build_case(params)
    chunks = tb.format_plan_messages([galit.DiagnosedWell(case, diagnose(case))])
    assert all(len(chunk) <= tb.TELEGRAM_TEXT_LIMIT for chunk in chunks)
    text = "".join(chunks)
    assert "<A>" not in text and "&lt;A&gt;" in text
    assert "БЛОК: верифицировать данные" in text
    assert "/plan" in tb.START_TEXT and "/plan_clear" in tb.HELP_TEXT


# ------------------------------------------------------- подбор обработки

def _fake_result(onset_m: float, wax_severity: float) -> galit.DiagnosisResult:
    return galit.DiagnosisResult(
        well="x", depths=[], temps=[], pressures=[], wat_profile=[],
        wax_onset_m=onset_m, severity={"wax": wax_severity},
    )


def test_wax_treatment_depends_on_lift_type():
    """Глубокое начало АСПО: ШГН достаёт скребками-центраторами, ЭЦН -- нет."""
    result = _fake_result(onset_m=2000.0, wax_severity=0.6)

    base = tb.parse_args(["4000", "62", "5", "30", "45", "48"])[0]
    esp = tb.build_case({**base, "lift_type": "ЭЦН"})
    sgn = tb.build_case({**base, "lift_type": "ШГН"})
    assert esp.lift_type == "ЭЦН" and sgn.lift_type == "ШГН"

    esp_plan = tb.wax_treatment(result, esp)
    sgn_plan = tb.wax_treatment(result, sgn)
    assert "скребок" not in esp_plan          # 2000 м > 1500 м вылета скребка
    assert "растворитель" in esp_plan or "кабель" in esp_plan
    assert "2000 м" in sgn_plan               # ШГН: оборудование достаёт до забоя


# ----------------------------------------------------------- forecast command

def _diagnosed(name: str) -> galit.DiagnosedWell:
    params, errors = tb.parse_args(POSITIONAL + [f"скважина={name}"])
    assert errors == []
    case = tb.build_case(params)
    return galit.DiagnosedWell(case, diagnose(case))


def test_forecast_select_empty_latest_and_normalized_name():
    with pytest.raises(LookupError, match="Сначала рассчитайте скважину"):
        tb.select_forecast_diagnosis([])
    first = _diagnosed("Скважина Альфа")
    latest = _diagnosed("Скважина Бета")
    assert tb.select_forecast_diagnosis([first, latest]) is latest
    assert tb.select_forecast_diagnosis([first, latest], "  СКВАЖИНА   альфа ") is first


def test_forecast_select_unknown_and_ambiguous_names_are_clear():
    items = [_diagnosed("Well A"), _diagnosed("WELL A")]
    with pytest.raises(LookupError, match="не найдена"):
        tb.select_forecast_diagnosis(items, "missing")
    with pytest.raises(LookupError, match="неоднозначно"):
        tb.select_forecast_diagnosis(items, "well a")


def test_forecast_text_has_no_fake_date_or_probability_and_is_chunked():
    chunks = tb.format_forecast_messages(_diagnosed("Честная дата"))
    assert chunks and all(len(chunk) <= tb.TELEGRAM_TEXT_LIMIT for chunk in chunks)
    text = "\n".join(chunks)
    assert "дата недоступна" in text
    assert "status:" in text and "Основание:" in text and "Нужно:" in text
    assert "вероятность" not in text.lower()
    assert "не заменяет промысловые исследования" in text
    assert "/forecast" in tb.START_TEXT and "/forecast" in tb.HELP_TEXT
