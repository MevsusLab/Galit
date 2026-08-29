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

def test_build_case_supports_optional_coordinates():
    params, errors = tb.parse_args(POSITIONAL + ["lat=52,37", "lon=30.38"])
    assert errors == []
    case = tb.build_case(params)
    assert case.latitude == 52.37 and case.longitude == 30.38


def test_map_summary_has_statuses_coordinates_and_empty_state():
    assert "История пуста" in tb.format_map_messages([])[0]
    params, _ = tb.parse_args(POSITIONAL + ["скважина=<A>", "lat=52.37", "lon=30.38"])
    case = tb.build_case(params)
    text = "\n".join(tb.format_map_messages([galit.DiagnosedWell(case, diagnose(case))]))
    assert "Карта месторождения" in text and "52.37000, 30.38000" in text
    assert "<A>" not in text and "&lt;A&gt;" in text
    assert "/map" in tb.HELP_TEXT


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


# ------------------------------------------------------- treatment journal

def _treatment(**overrides):
    now = tb.datetime(2026, 8, 23, 12, tzinfo=tb.timezone.utc)
    values = dict(
        well_id="w-1", well_name="Скважина <12>", event_at=now,
        complication_type="АСПО", description="Промывка <НКТ>",
        reagent_name="R&1", reagent_id=None, dosage=2, dosage_unit="l/m3",
        cost=100, currency="BYN", treatment_type="промывка", well_group="куст 1",
    )
    values.update(overrides)
    return galit.new_treatment(now=now, **values)


def test_treatment_parser_supports_quotes_spaces_and_ru_en_aliases():
    values, errors = tb.parse_treatment_command(
        '/treatment_add скважина="Скважина 12" event=АСПО описание="Промывка НКТ" '
        'reagent="Реагент A" dose=2 unit=l/m3 cost=100 currency=BYN type=промывка'
    )
    assert errors == []
    assert values["well"] == "Скважина 12"
    assert values["description"] == "Промывка НКТ"
    assert values["reagent"] == "Реагент A"


def test_treatment_card_escapes_html_and_long_chunks_are_bounded():
    text = "\n".join(tb.format_treatment_card(_treatment()))
    assert "Скважина <12>" not in text and "Скважина &lt;12&gt;" in text
    assert "R&1" not in text and "R&amp;1" in text
    chunks = tb._forecast_chunks(["x " * 5000])
    assert len(chunks) > 1 and all(len(chunk) <= tb.TELEGRAM_TEXT_LIMIT for chunk in chunks)


def test_treatment_lifecycle_requires_each_explicit_transition_and_revision(tmp_path):
    repo = galit.TreatmentRepository(tmp_path / "journal.json")
    planned = repo.create(_treatment())
    with pytest.raises(ValueError, match="invalid treatment transition"):
        planned.transition(galit.TreatmentStatus.COMPLETED)
    started = repo.update(planned.transition(galit.TreatmentStatus.IN_PROGRESS), expected_revision=1)
    with pytest.raises(galit.TreatmentConflictError):
        repo.update(started.transition(galit.TreatmentStatus.COMPLETED), expected_revision=1)
    completed = repo.update(started.transition(galit.TreatmentStatus.COMPLETED), expected_revision=2)
    assessed = completed.transition(
        galit.TreatmentStatus.ASSESSED, actual_result="Эффект есть",
        result_metrics={"дебит": 2.5}, success=True, effect_duration_days=30,
        recurrence=False,
    )
    assert repo.update(assessed, expected_revision=3).status is galit.TreatmentStatus.ASSESSED


def test_treatment_comparison_formatter_shows_cohort_n_insufficient_and_warning():
    result = galit.compare_reagents(
        [], "A", "B", complication_type="АСПО", well_group="куст <1>", min_sample_size=2,
    )
    text = "\n".join(tb.format_treatment_comparison(result))
    assert "complication=АСПО" in text and "well_group=куст &lt;1&gt;" in text
    assert "A: n=0" in text and "B: n=0" in text
    assert "Недостаточно данных" in text and "confidence: low" in text
    assert "не доказывает причинность" in text


def test_treatment_card_and_list_explain_rate_change_and_nullable_values():
    measured = _treatment(rate_before_m3_day=10, rate_after_m3_day=12,
                          effect_duration_days=20, status=galit.TreatmentStatus.COMPLETED)
    card = "\n".join(tb.format_treatment_card(measured))
    listing = "\n".join(tb.format_treatments([measured]))
    assert "Изменение дебита: +2 м³/сут (+20.0%)" in card
    assert "эффективна" in card and "Δ дебита +2 м³/сут" in listing

    missing = "\n".join(tb.format_treatment_card(_treatment()))
    assert "Дебит до: —" in missing and "Дебит после: —" in missing
    assert "недостаточно данных" in missing


def test_treatment_stats_exposes_sample_gate_outliers_units_and_currency():
    rows = [
        _treatment(field_name="Поле <A>", rate_before_m3_day=10, rate_after_m3_day=9,
                   effect_duration_days=5, status=galit.TreatmentStatus.COMPLETED),
        _treatment(well_id="w-2", field_name="Поле <A>", rate_before_m3_day=10,
                   rate_after_m3_day=11, effect_duration_days=5,
                   status=galit.TreatmentStatus.COMPLETED, currency="USD"),
    ]
    chunks = tb.format_treatment_stats(rows, min_sample_size=5)
    text = "\n".join(chunks)
    assert all(len(chunk) <= tb.TELEGRAM_TEXT_LIMIT for chunk in chunks)
    assert "Неэффективные:" in text and "Потенциально избыточные:" in text
    assert "n=2" in text and "недостаточно данных" in text and "нужно 5" in text
    assert "Поле &lt;A&gt;" in text
    assert "валюты и единицы дозировки не объединяются" in text.casefold()


def test_treatment_stats_empty_and_parser_errors_are_explicit():
    text = "\n".join(tb.format_treatment_stats([], min_sample_size=5))
    assert "недостаточно данных" in text.lower()
    values, errors = tb.parse_treatment_command('/treatment_stats min_n=1 bad=2')
    assert values["min_sample_size"] == "1"
    assert any("неверный параметр" in error for error in errors)
    _, errors = tb.parse_treatment_command('/treatment_add description="broken')
    assert any("ошибка кавычек" in error for error in errors)


def test_treatment_help_documents_argument_commands_and_measurement_units():
    for command in ("/treatments", "/treatment_add", "/treatment_result",
                    "/treatment_stats", "/treatment_compare"):
        assert command in tb.HELP_TEXT
    assert "key=value" in tb.HELP_TEXT and "before=" in tb.HELP_TEXT
    assert "м³/сут" in tb.HELP_TEXT and "валюты и единицы" in tb.HELP_TEXT
