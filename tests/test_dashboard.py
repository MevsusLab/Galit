"""Тесты чистых функций дашборда: разбор файла, ранжирование, профили.

Дашборд импортируется целиком (вне Streamlit-рантайма его вызовы
безопасно игнорируются), поэтому тестируются только функции без UI.
"""
from __future__ import annotations

import io

import pandas as pd
from pandas.io.formats.style import Styler

import dashboard
from galit import diagnose


def base_row() -> dict:
    """Минимальная строка фонда: только обязательные колонки."""
    return {
        "name": "Речицкая 123",
        "depth_m": 3200.0,
        "tubing_id_m": 0.062,
        "q_oil_m3d": 8.0,
        "q_water_m3d": 72.0,
        "gor_m3m3": 65.0,
        "wat_stock_tank_c": 34.0,
    }


# ---------------------------------------------------------------- заголовки

def test_normalize_headers_aliases_and_ions():
    mapping = dashboard.normalize_headers(
        ["СКВАЖИНА", "Well", "Na", "hco3", "Способ", "МусорнаяКолонка"]
    )
    assert mapping["СКВАЖИНА"] == "name"
    assert mapping["Well"] == "name"
    assert mapping["Na"] == "Na"
    assert mapping["hco3"] == "HCO3"
    assert mapping["Способ"] == "lift_type"
    assert "МусорнаяКолонка" not in mapping


# ------------------------------------------------------------ разбор таблицы

def test_frame_to_cases_minimal_columns_use_defaults():
    cases, errors = dashboard.frame_to_cases(pd.DataFrame([base_row()]))
    assert errors == []
    assert len(cases) == 1
    case = cases[0]
    assert case.name == "Речицкая 123"
    # значения по умолчанию ядра подставлены для отсутствующих колонок
    assert case.geometry.inclination_deg == 0.0
    assert case.fluid.gamma_oil == 0.86
    assert case.wax.wax_content_pct == 5.0
    # ионный состав не задан -- подставлен типовой рассол
    assert case.water.ions_mg_l == dashboard.TYPICAL_BRINE


def test_frame_to_cases_missing_required_column():
    row = base_row()
    del row["q_oil_m3d"]
    cases, errors = dashboard.frame_to_cases(pd.DataFrame([row]))
    assert cases == []
    assert len(errors) == 1
    assert "q_oil_m3d" in errors[0]


def test_frame_to_cases_broken_row_does_not_block_others():
    broken = base_row() | {"name": "Плохая", "depth_m": None}
    good = base_row() | {"name": "Хорошая"}
    cases, errors = dashboard.frame_to_cases(pd.DataFrame([broken, good]))
    assert [c.name for c in cases] == ["Хорошая"]
    assert len(errors) == 1 and "Плохая" in errors[0]


def test_frame_to_cases_non_numeric_optional_falls_back_to_default():
    row = base_row() | {"gamma_oil": "много"}
    cases, errors = dashboard.frame_to_cases(pd.DataFrame([row]))
    assert len(cases) == 1
    assert cases[0].fluid.gamma_oil == 0.86   # ядро приняло замену
    assert any("gamma_oil" in e for e in errors)


def test_frame_to_cases_ion_columns_parsed():
    row = base_row() | {"Na": 90000.0, "Cl": 200000.0, "Ca": 0.0}
    cases, _ = dashboard.frame_to_cases(pd.DataFrame([row]))
    # Ca = 0 отбрасывается как отсутствующий, типовой состав не подставляется,
    # т.к. есть реальные ионы
    assert cases[0].water.ions_mg_l == {"Na": 90000.0, "Cl": 200000.0}


# ------------------------------------------------------- ранжирование/статусы

def test_risk_status_thresholds():
    assert dashboard.risk_status(0.10)[2] == "норма"
    assert dashboard.risk_status(dashboard.RISK_WARN)[2] == "повышенный"
    assert dashboard.risk_status(dashboard.RISK_CRIT)[2] == "критический"


def test_rank_frame_sorted_and_numbered():
    results = [diagnose(c) for c in dashboard.galit.synthetic.make_fund(6)]
    df = dashboard.rank_frame(results)
    risks = df["Риск"].tolist()
    assert risks == sorted(risks, reverse=True)
    assert df["№"].tolist() == list(range(1, len(df) + 1))
    assert {"Скважина", "Риск", "Статус", "Лидер", "Рекомендация"} <= set(df.columns)


def test_style_rank_returns_styler():
    results = [diagnose(c) for c in dashboard.galit.synthetic.make_fund(3)]
    styler = dashboard.style_rank(dashboard.rank_frame(results))
    assert isinstance(styler, Styler)


def test_result_labels_deduplicates_names():
    case = dashboard.galit.synthetic.make_fund(1, seed=1)[0]
    twin = diagnose(case)
    original = diagnose(case)
    pairs = dashboard.result_labels([original, twin])
    labels = [label for label, _ in pairs]
    assert labels[0] == case.name
    assert labels[1].startswith(case.name) and "(2)" in labels[1]
    assert dashboard.unique_labels([original, twin]) == labels


# ------------------------------------------------------------------- профили

def test_fig_profiles_traces_and_duplicate_labels():
    case = dashboard.galit.synthetic.make_fund(1, seed=7)[0]
    results = [diagnose(case), diagnose(case)]
    labels = dashboard.unique_labels(results)
    fig = dashboard.fig_profiles(results, labels, detail=results[0])
    # 2 скважины x (T + P) + WAT в детальном разборе
    assert len(fig.data) == 5


# -------------------------------------------------------------- шаблон XLSX

def test_template_bytes_roundtrip():
    book = dashboard.template_bytes()
    df = pd.read_excel(io.BytesIO(book), sheet_name="Данные")
    cases, errors = dashboard.frame_to_cases(df)
    assert errors == []
    assert len(cases) == 1
    assert cases[0].name == "Речицкая 123"
    # лист-инструкция на месте
    docs = pd.read_excel(io.BytesIO(book), sheet_name="Инструкция")
    assert list(docs.columns)[0] == "Колонка"


def test_read_table_csv_cp1251_semicolon():
    csv = "name;depth_m;tubing_id_m;q_oil_m3d;q_water_m3d;gor_m3m3;wat_stock_tank_c\n" \
          "Вишанская 7;2800;0,062;5;45;60;36\n".encode("cp1251")
    df = dashboard.read_table(csv, "fund.csv")
    cases, errors = dashboard.frame_to_cases(df)
    assert errors == []
    assert cases[0].name == "Вишанская 7"
    assert cases[0].geometry.depth_m == 2800.0
