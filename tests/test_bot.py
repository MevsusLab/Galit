"""Тесты Telegram-бота: разбор /aspo, сборка кейса, отчёт, подбор обработки.

Сеть не используется -- проверяются чистые функции и консистентность
с расчётным ядром galit.
"""
from __future__ import annotations

import galit
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
    assert "Обработка АСПО" in report
    assert "<b>Речицкая 123</b>" in report


def test_report_deposition_branch():
    # WAT 48 °C -- выше типовой температуры потока на устье, отложения есть
    params, _ = tb.parse_args(["2900", "62", "6", "64", "45", "48"])
    case = tb.build_case(params)
    result = diagnose(case)
    assert result.wax_onset_m is not None
    report = tb.format_report(result, case, tb.wax_treatment(result, case))
    assert f"{result.wax_onset_m:.0f} м" in report
    assert "Зона отложений" in report
    assert "Интегральный риск" in report


def test_report_escapes_well_name():
    params, _ = tb.parse_args(POSITIONAL + ["скважина=<script>x</script>"])
    case = tb.build_case(params)
    result = diagnose(case)
    report = tb.format_report(result, case, tb.wax_treatment(result, case))
    assert "<script>" not in report
    assert "&lt;script&gt;" in report


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
