"""Тесты чистых функций дашборда: разбор файла, ранжирование, профили.

Дашборд импортируется целиком (вне Streamlit-рантайма его вызовы
безопасно игнорируются), поэтому тестируются только функции без UI.
"""
from __future__ import annotations

import base64
import io
from dataclasses import replace
from pathlib import Path
import re

import pandas as pd
import pytest
from pandas.io.formats.style import Styler
from streamlit.testing.v1 import AppTest

import dashboard
from galit import diagnose


MINIMAL_CSV = (
    "name,depth_m,tubing_id_m,q_oil_m3d,q_water_m3d,gor_m3m3,wat_stock_tank_c\n"
    "139,3200,0.062,8,72,65,34\n"
).encode()


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


def test_frame_to_cases_preserves_location_metadata_for_map():
    row = base_row() | {
        "latitude": 52.371, "longitude": 30.387,
        "cluster": "╨Ъ╤Г╤Б╤В 3", "site": "╨а╨╡╤З╨╕╤Ж╨║╨╛╨╡",
    }
    cases, errors = dashboard.frame_to_cases(pd.DataFrame([row]))
    assert errors == []
    case = cases[0]
    assert (case.latitude, case.longitude, case.cluster, case.site) == (
        52.371, 30.387, "╨Ъ╤Г╤Б╤В 3", "╨а╨╡╤З╨╕╤Ж╨║╨╛╨╡",
    )

    diagnosed = dashboard.galit.DiagnosedWell(case, diagnose(case))
    map_data = dashboard.galit.prepare_field_map([diagnosed])
    assert map_data.summary.mapped_wells == 1
    assert (map_data.points[0].latitude, map_data.points[0].longitude) == (52.371, 30.387)
    assert (map_data.points[0].cluster, map_data.points[0].site) == ("╨Ъ╤Г╤Б╤В 3", "╨а╨╡╤З╨╕╤Ж╨║╨╛╨╡")


def test_frame_to_cases_empty_coordinates_are_safe_for_map():
    row = base_row() | {
        "latitude": "", "longitude": pd.NA,
        "cluster": "", "site": None,
    }
    cases, errors = dashboard.frame_to_cases(pd.DataFrame([row]))
    assert errors == []
    case = cases[0]
    assert case.latitude is None and case.longitude is None
    assert case.cluster is None and case.site is None

    diagnosed = dashboard.galit.DiagnosedWell(case, diagnose(case))
    map_data = dashboard.galit.prepare_field_map([diagnosed])
    assert map_data.points == ()
    assert map_data.summary.missing_coordinates == 1


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


def test_overview_figures_and_alerts_use_current_results_only():
    results = [diagnose(case) for case in dashboard.galit.synthetic.make_fund(8, seed=17)]
    risk_fig = dashboard.fig_fund_risk(results)
    mix_fig = dashboard.fig_mechanism_mix(results)
    alerts = dashboard.overview_alerts(results)

    assert len(risk_fig.data) == 1
    assert list(risk_fig.data[0].x) == [
        item.well for item in sorted(results, key=lambda row: row.integrated_risk, reverse=True)
    ]
    assert len(mix_fig.data) == 1
    assert sum(mix_fig.data[0].values) == len(results)
    assert all("timestamp" not in alert and "time" not in alert for alert in alerts)
    assert all(alert["well"] in {item.well for item in results} for alert in alerts)


def test_field_map_uses_scattermap_osm_and_stable_local_viewport():
    cases = dashboard.galit.synthetic.make_fund(3, seed=23)
    located = [
        replace(cases[0], latitude=52.4102, longitude=30.7541),
        replace(cases[1], latitude=52.4280, longitude=30.7810),
        replace(cases[2], latitude=52.4451, longitude=30.8124),
    ]
    items = []
    expected_sizes = {}
    expected_colors = {}
    for case, risk in zip(located, (.1, .45, .75)):
        result = diagnose(case)
        result.integrated_risk = risk
        items.append(dashboard.galit.DiagnosedWell(case, result))
    data = dashboard.galit.prepare_field_map(items)
    for point in data.points:
        expected_sizes[point.status] = point.marker_size
        expected_colors[point.status] = dashboard.galit.MAP_STATUS_COLORS[point.status]

    fig = dashboard.fig_field_map(data)
    assert fig.layout.map.style == "open-street-map"
    assert fig.layout.map.center.lat == pytest.approx((52.4102 + 52.4451) / 2)
    assert fig.layout.map.center.lon == pytest.approx((30.7541 + 30.8124) / 2)
    assert 8.0 <= fig.layout.map.zoom <= 10.5
    assert all(trace.type == "scattermap" for trace in fig.data)

    status_traces = {trace.name: trace for trace in fig.data if trace.name in dashboard.galit.MAP_STATUS_LABELS.values()}
    assert set(status_traces) == set(dashboard.galit.MAP_STATUS_LABELS.values())
    for status, label in dashboard.galit.MAP_STATUS_LABELS.items():
        trace = status_traces[label]
        assert list(trace.marker.size) == [expected_sizes[status]]
        assert trace.marker.color == expected_colors[status]


def test_field_map_has_explicit_approximate_pripyat_context():
    case = replace(dashboard.galit.synthetic.make_fund(1, seed=24)[0],
                   latitude=52.43, longitude=30.78)
    data = dashboard.galit.prepare_field_map([
        dashboard.galit.DiagnosedWell(case, diagnose(case))
    ])
    fig = dashboard.fig_field_map(data)
    contour = fig.data[0]
    label = fig.data[1]

    assert contour.type == "scattermap" and contour.fill == "toself"
    assert "обзорно" in contour.name.lower()
    assert contour.lon[0] == contour.lon[-1] and contour.lat[0] == contour.lat[-1]
    assert max(contour.lon) - min(contour.lon) >= 4.0
    assert max(contour.lat) - min(contour.lat) >= 1.3
    assert "ОБЗОРНО" in label.text[0]
    source = Path("dashboard.py").read_text(encoding="utf-8")
    assert "не является точной или лицензионной геологической границей" in source
    assert "Подложка OpenStreetMap требует интернет" in source


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


def test_exact_minimal_csv_screening_and_production_gate():
    df = dashboard.read_table(MINIMAL_CSV, "minimal.csv")
    cases, parse_errors = dashboard.frame_to_cases(df)
    assert parse_errors == []
    assert len(cases) == 1 and cases[0].name == "139"

    screening, screening_errors = dashboard.diagnose_frame(df, production_mode=False)
    assert len(screening) == 1 and screening_errors == []
    assert not screening[0].quality.production_ready
    assert dashboard.action_is_safe(screening[0])[0] is False

    production, quality_errors = dashboard.diagnose_frame(df, production_mode=True)
    assert production == []
    assert quality_errors
    assert "не ранжируется" in quality_errors[0]
    assert "нет фактических данных" in quality_errors[0]
    assert "water.ions_mg_l" in quality_errors[0]


def test_empty_results_status_distinguishes_parsing_from_quality_gate():
    kind, message = dashboard.empty_results_status(1, production_mode=True)
    assert kind == "quality_gate"
    assert "файл распознан" in message.lower()
    assert "промышленный контроль" in message

    kind, message = dashboard.empty_results_status(0, production_mode=False)
    assert kind == "parsing"
    assert "не распознана" in message


# ------------------------------------------ совместимость двух измеренных вод


def compatibility_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "Пластовая", "ph": 6.2, "t_c": 25, "p_pa": 5e6,
         "Na": 20000, "Cl": 32000, "Ca": 4000, "Ba": 1000,
         "HCO3": 50, "SO4": 0},
        {"name": "Закачиваемая", "ph": 7.0, "t_c": 25, "p_pa": 1e5,
         "Na": 1000, "Cl": 1500, "Ca": 100, "Ba": 0,
         "HCO3": 300, "SO4": 3000},
    ])


def test_compatibility_template_is_two_blank_water_rows_without_synthetic_chemistry():
    frame = dashboard.compatibility_template_frame()
    assert list(frame.columns) == dashboard.COMPATIBILITY_COLUMNS
    assert frame["name"].tolist() == ["Вода A", "Вода B"]
    assert frame[dashboard.ION_KEYS].isna().all().all()

    book = dashboard.compatibility_template_bytes()
    saved = pd.read_excel(io.BytesIO(book), sheet_name="Две воды")
    assert len(saved) == 2 and saved[dashboard.ION_KEYS].isna().all().all()


def test_compatibility_adapter_preserves_only_supplied_chemistry_and_validates_rows():
    water_a, water_b = dashboard.compatibility_waters_from_frame(compatibility_rows())
    assert water_a.name == "Пластовая" and water_b.name == "Закачиваемая"
    assert water_a.ions_mg_l["Ba"] == 1000
    assert "Mg" not in water_a.ions_mg_l
    with pytest.raises(ValueError, match="ровно две"):
        dashboard.compatibility_waters_from_frame(compatibility_rows().iloc[:1])
    broken = compatibility_rows().copy()
    broken.loc[0, "ph"] = pd.NA
    with pytest.raises(ValueError, match="ph"):
        dashboard.compatibility_waters_from_frame(broken)


def test_compatibility_plot_has_four_minerals_and_si_zero_reference():
    a, b = dashboard.compatibility_waters_from_frame(compatibility_rows())
    result = dashboard.galit.evaluate_compatibility(a, b, [0, .5, 1])
    frame = dashboard.compatibility_ratio_frame(result)
    assert {"Кальцит", "Барит", "Гипс", "Галит", "Небезопасно"} <= set(frame.columns)
    fig = dashboard.fig_compatibility_ratios(result)
    assert [trace.name for trace in fig.data] == ["Кальцит", "Барит", "Гипс", "Галит"]
    assert any(shape.y0 == 0 and shape.y1 == 0 for shape in fig.layout.shapes)


def test_compatibility_profile_and_validated_dose_curve_adapters():
    profile = dashboard.compatibility_profile_from_frame(pd.DataFrame([
        {"depth_m": 0, "t_c": 20, "p_pa": 1e5},
        {"depth_m": 1000, "t_c": 40, "p_pa": 8e6},
    ]))
    assert [point.depth_m for point in profile] == [0, 1000]
    curve_frame = pd.DataFrame([
        {"dose_mg_l": 10, "maximum_supported_si": 0.5},
        {"dose_mg_l": 20, "maximum_supported_si": 2.0},
    ])
    with pytest.raises(ValueError, match="подтвердите"):
        dashboard.compatibility_curve_from_frame("X", "barite", "lab-1", False, curve_frame)
    curve = dashboard.compatibility_curve_from_frame("X", "barite", "lab-1", True, curve_frame)
    assert curve.validated and curve.validation_reference == "lab-1"


# --------------------------------------------- trust / explainability helpers

def test_contributions_are_consistent_with_integrated_risk():
    result = diagnose(dashboard.galit.synthetic.make_fund(1, seed=11)[0])
    frame = dashboard.contribution_frame(result)
    assert len(frame) == 4
    assert abs(frame["Вклад в риск"].sum() - result.integrated_risk) < 1e-12
    if result.integrated_risk > 0:
        assert abs(frame["Доля integrated risk"].sum() - 1.0) < 1e-12


def test_provenance_grouping_and_warning_categories_preserve_originals():
    cases, _ = dashboard.frame_to_cases(pd.DataFrame([base_row()]))
    result = diagnose(cases[0])
    grouped = dashboard.provenance_groups(result)
    assert grouped["critical"]
    assert any(row["source"] == "synthetic" for row in grouped["rows"])
    warnings = ["Screening: качество данных D", "TDS 200 г/л -- LSI неприменим"]
    categories = dashboard.categorize_warnings(warnings)
    assert sorted(w for values in categories.values() for w in values) == sorted(warnings)


def test_decision_trace_has_required_fields():
    result = diagnose(dashboard.galit.synthetic.make_fund(1, seed=4)[0])
    trace = dashboard.decision_trace(result)
    assert trace["dominant"]
    assert trace["reason"] == result.recommendation
    assert trace["conflict"]
    assert trace["measure_next"]


def test_counterfactual_is_non_mutating_and_safe():
    case = dashboard.galit.synthetic.make_fund(1, seed=8)[0]
    original_efficiency = case.inhibitor_efficiency
    scenario = dashboard.corrosion_counterfactual(case, 0.9)
    assert scenario["supported"]
    assert case.inhibitor_efficiency == original_efficiency
    assert scenario["after"].corrosion["rate_mm_yr"] <= scenario["before"].corrosion["rate_mm_yr"]
    assert not dashboard.corrosion_counterfactual(case, 1.1)["supported"]


def test_screening_action_is_blocked():
    cases, _ = dashboard.frame_to_cases(pd.DataFrame([base_row()]))
    safe, note = dashboard.action_is_safe(diagnose(cases[0]))
    assert not safe
    assert "заблокировано" in note


def test_pilot_dashboard_helpers():
    book = dashboard.pilot_template_bytes()
    assert list(pd.read_excel(io.BytesIO(book), sheet_name="Pilot outcomes").columns)[:3] == [
        "well_id", "timestamp", "calendar_score"
    ]
    evaluation = dashboard.evaluate_uploaded_rows([{
        "well_id": "A", "timestamp": "2025-01-01T00:00:00+00:00",
        "event_outcome": 1, "calendar_score": .1,
        "independent_score": .2, "galit_score": .3,
    }], k=1)
    frame = dashboard.pilot_evaluation_frame(evaluation)
    assert len(frame) == 3 and "NDCG@K" in frame.columns


def test_master_plan_helpers_use_current_cases_and_export_russian_csv():
    cases = dashboard.galit.synthetic.make_fund(3, seed=19)
    results = [diagnose(case) for case in cases]
    plan = dashboard.build_master_plan({case.name: case for case in cases}, results)
    frame = dashboard.master_plan_frame(plan)
    assert list(frame.columns) == [
        "Скважина", "Осложнение", "Риск", "Возможная потеря, м³/сут",
        "Срок", "Действие", "Safe-to-act",
    ]
    csv = dashboard.master_plan_csv(plan).decode("utf-8-sig")
    assert "Скважина" in csv and "Safe-to-act" in csv


# --------------------------------------------- Streamlit upload regression

def test_twin_timeline_categories_and_figure():
    categories = dashboard.twin_timeline_categories()
    assert categories == [item.value for item in dashboard.galit.EventCategory]
    items = [
        {"occurred_at": "2026-08-22T12:00:00+00:00", "category": "repair",
         "title": "Ремонт", "summary": "Замена узла"},
        {"occurred_at": "2026-08-23T12:00:00+00:00", "category": "laboratory",
         "title": "Лаборатория", "summary": "Отобрана проба"},
    ]
    fig = dashboard.fig_twin_timeline(items)
    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ["laboratory", "repair"]
    assert fig.layout.yaxis.visible is False


def test_twin_components_use_configured_repository_paths(tmp_path, monkeypatch):
    paths = {name: tmp_path / file_name for name, file_name in {
        "GALIT_WATERCUT_STORAGE": "watercut.json",
        "GALIT_EQUIPMENT_STORAGE": "equipment.json",
        "GALIT_TREATMENT_STORAGE": "treatments.json",
        "GALIT_PASSPORT_STORAGE": "passports.json",
        "GALIT_TWIN_EVENT_STORAGE": "manual.json",
    }.items()}
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))
    dashboard.st.session_state.clear()
    service, manual = dashboard.get_twin_components()
    assert manual.path == paths["GALIT_TWIN_EVENT_STORAGE"]
    assert len(service.adapters) == 5
    assert service.list_wells() == []
    assert not paths["GALIT_TWIN_EVENT_STORAGE"].exists()

    case = dashboard.galit.synthetic.make_fund(1, seed=71)[0]
    loaded_service, _ = dashboard.get_twin_components([case])
    assert [item["display_name"] for item in loaded_service.list_wells()] == [case.name]
    assert loaded_service.snapshot(case.name).state == "insufficient_data"
    assert not paths["GALIT_TWIN_EVENT_STORAGE"].exists()


# --------------------------------------------------- Feature 9 / chemicals UI


def _chemical_product(**overrides):
    values = dict(
        id="p1", name="Ингибитор H", manufacturer="Завод",
        hazards=("halite",), price_per_kg="2.5", currency="BYN",
    )
    return dashboard.galit.ChemicalProduct(**(values | overrides))


def _chemical_envelope(**overrides):
    values = dict(
        id="e1", product_id="p1", hazard="halite",
        points=(dashboard.galit.ChemicalDoseResponsePoint("0.01", False),
                dashboard.galit.ChemicalDoseResponsePoint("0.025", True)),
        validated=True, validation_reference="LAB-42", conditions="80 °C",
    )
    return dashboard.galit.ChemicalDoseResponseEnvelope(**(values | overrides))


def test_chemical_adapters_empty_insufficient_eligible_and_zero_oil():
    empty = dashboard.chemical_catalog_frame([], [])
    assert empty.empty

    product = _chemical_product()
    rejected = dashboard.chemical_candidate_rows([product], [], ["halite"], 100, 10)[0]
    assert rejected["Статус"] == "отклонён"
    assert "нет валидированного" in rejected["Причина / недостающие данные"]

    eligible = dashboard.chemical_candidate_rows(
        [product], [_chemical_envelope()], ["halite"], 100, 10,
    )[0]
    assert eligible["Статус"] == "подходит"
    assert eligible["Минимальная испытанная доза, кг/м³"] == pytest.approx(.025)
    assert eligible["Расход, кг/сут"] == pytest.approx(2.5)
    assert eligible["Стоимость на м³ нефти"] == pytest.approx(.625)
    assert eligible["Ссылки"] == "LAB-42"

    zero_oil = dashboard.chemical_candidate_rows(
        [product], [_chemical_envelope()], ["halite"], 100, 0,
    )[0]
    assert zero_oil["Стоимость на м³ нефти"] is None
    assert "не определена" in zero_oil["Причина / недостающие данные"]


def test_chemical_import_is_explicit_and_strict():
    points = dashboard.chemical_points_from_frame(pd.DataFrame([
        {"dose_mg_l": 0.025, "effective": "да"},
        {"dose_mg_l": 0.05, "effective": "нет"},
    ]))
    assert points[0].dose_kg_m3 == dashboard.Decimal("0.025")
    assert points[0].effective is True
    with pytest.raises(ValueError, match="effective"):
        dashboard.chemical_points_from_frame(pd.DataFrame([
            {"dose_kg_m3": 1, "effective": "возможно"},
        ]))


def test_chemical_inventory_reservation_release_and_shortage(tmp_path):
    repo = dashboard.galit.ChemicalRepository(tmp_path / "chemicals.json")
    repo.put_product(_chemical_product())
    now = dashboard.datetime(2026, 8, 23, 12, tzinfo=dashboard.timezone.utc)
    repo.add_lot(
        dashboard.galit.StockLot("lot-1", "p1", now, dashboard.date(2027, 1, 1), 20),
        idempotency_key="receipt-1",
    )
    reserved = repo.reserve("p1", 5, dashboard.date(2026, 8, 25),
                            idempotency_key="reserve-1", now=now)
    frame = dashboard.chemical_inventory_frame(repo, as_of=dashboard.date(2026, 8, 25))
    assert frame.iloc[0]["На складе, кг"] == 20
    assert frame.iloc[0]["Зарезервировано, кг"] == 5
    assert frame.iloc[0]["Доступно, кг"] == 15
    repo.release_reservation(reserved.id, revision=reserved.revision)
    assert dashboard.chemical_inventory_frame(
        repo, as_of=dashboard.date(2026, 8, 25)
    ).iloc[0]["Доступно, кг"] == 20

    repo.consume("p1", 18, now, idempotency_key="use-1", reference="well-1")
    view = dashboard.chemical_forecast_view(
        repo, "p1", as_of=dashboard.date(2026, 8, 23), horizon_days=30,
        lead_time_days=5, safety_stock_days=3,
    )
    assert view["forecast"]["status"] == "available"
    assert view["shortage"]["risk"] is True
    assert dashboard.chemical_forecast_view(
        dashboard.galit.ChemicalRepository(tmp_path / "empty.json"), "p1",
        as_of=dashboard.date(2026, 8, 23), horizon_days=30,
        lead_time_days=5, safety_stock_days=3,
    )["forecast"]["daily_kg"] is None


def test_chemical_storage_env_and_confirmation_copy(monkeypatch, tmp_path):
    path = tmp_path / "dedicated-chemicals.json"
    monkeypatch.setenv("GALIT_CHEMICAL_STORAGE", str(path))
    assert dashboard.chemical_storage_path() == path
    source = Path("dashboard.py").read_text(encoding="utf-8")
    assert "Подтверждаю создание резерва" in source
    assert "Подтверждаю освобождение резерва" in source
    assert "фактические испытания, а не расчётные или демонстрационные" in source


def test_apptest_initial_state_is_welcome_without_tabs_or_auto_demo():
    app = AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=60).run()
    assert not app.exception
    assert list(app.tabs) == []
    assert any("note-box" in item.value for item in app.markdown)
    assert not any(metric.label == "\u0421\u043a\u0432\u0430\u0436\u0438\u043d \u0432 \u0440\u0430\u0441\u0447\u0451\u0442\u0435" for metric in app.metric)


def test_apptest_upload_renders_master_plan_as_first_tab():
    app = AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=120).run()
    app.get("file_uploader")[0].set_value(
        ("valid-fund.csv", MINIMAL_CSV, "text/csv")
    ).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "\u041f\u043b\u0430\u043d \u043c\u0430\u0441\u0442\u0435\u0440\u0430",
        "Карта месторождения",
        "Обводнение",
        "\u0420\u0430\u043d\u0436\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0444\u043e\u043d\u0434\u0430",
        "\u041f\u0440\u043e\u0444\u0438\u043b\u0438 T(z) \u00b7 P(z)",
        "\u0414\u0435\u0442\u0430\u043b\u044c\u043d\u043e \u043f\u043e \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u0435",
        "Что будет, если?",
        "Экономика риска",
        "Прогноз во времени",
        "Оборудование / Прогноз отказов",
        "\u0421\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u0441 baseline / \u041f\u0438\u043b\u043e\u0442",
        "Цифровой паспорт",
        "Цифровой двойник",
        "Журнал мероприятий",
    ]
    assert next(metric.value for metric in app.metric if metric.label == "\u0412\u0441\u0435\u0433\u043e \u0437\u0430\u0434\u0430\u0447") == "1"
    assert any(button.label == "\u0421\u043a\u0430\u0447\u0430\u0442\u044c \u043f\u043b\u0430\u043d CSV" for button in app.get("download_button"))


# ------------------------------------------------ accessibility / contrast

def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_project_theme_is_explicit_accessible_light_theme():
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    expected = {
        "base": "light",
        "primaryColor": dashboard.GREEN_700,
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": dashboard.SURFACE,
        "textColor": dashboard.INK,
    }
    for key, value in expected.items():
        assert re.search(rf'^{key}\s*=\s*"{re.escape(value)}"$', config, re.MULTILINE)
    assert _contrast_ratio(dashboard.INK, "#FFFFFF") >= 7
    assert _contrast_ratio(dashboard.INK_MUTED, "#FFFFFF") >= 4.5


def test_dashboard_css_covers_accessibility_critical_streamlit_selectors():
    source = Path("dashboard.py").read_text(encoding="utf-8")
    selectors = (
        'data-testid="stSidebar"', 'data-testid="stWidgetLabel"',
        'data-testid="stFileUploaderDropzone"',
        'data-testid="stFileUploaderDropzoneInstructions"',
        'data-testid="stCaptionContainer"', 'data-testid="stCheckbox"',
        'data-testid="stExpander"',
        'data-testid="stTabs"', 'data-testid="stDataFrame"',
        'data-baseweb="select"', 'data-baseweb="input"',
        'data-testid="stAlert"', 'data-testid="stMetric"',
    )
    for selector in selectors:
        assert selector in source


def test_status_pairs_and_styler_foregrounds_meet_wcag():
    for risk in (0.1, dashboard.RISK_WARN, dashboard.RISK_CRIT):
        background, accent, _ = dashboard.risk_status(risk)
        assert _contrast_ratio(dashboard.INK, background) >= 4.5
        assert _contrast_ratio("#FFFFFF", accent) >= 4.5

    results = [diagnose(c) for c in dashboard.galit.synthetic.make_fund(3)]
    html = dashboard.style_rank(dashboard.rank_frame(results)).to_html()
    assert f"color: {dashboard.INK}" in html
    assert "color: #FFFFFF" in html
    assert "#2E7D32" not in html and "#B26A00" not in html and "#C62828" not in html


# ------------------------------------------------- typography / icon library

def test_outfit_is_primary_font_with_cyrillic_fallback_and_local_faces():
    assert dashboard.FONT_FAMILY.startswith("Outfit, Manrope,")
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert re.search(r'^font\s*=\s*"Outfit, Manrope, sans serif"$', config, re.MULTILINE)

    # Outfit не содержит кириллицу, поэтому Manrope обязателен и должен покрывать U+04xx.
    families = {family for family, _, _ in dashboard.FONT_FACES}
    assert families == {"Outfit", "Manrope"}
    cyrillic = [
        unicode_range for family, _, unicode_range in dashboard.FONT_FACES
        if family == "Manrope"
    ]
    assert cyrillic and all("U+04" in item for item in cyrillic)

    css = dashboard.font_face_css()
    assert css.count("@font-face") == len(dashboard.FONT_FACES)
    for family, _, unicode_range in dashboard.FONT_FACES:
        assert f"font-family:'{family}'" in css
        assert f"unicode-range:{unicode_range}" in css
    # Переменный woff2 отдаёт весь диапазон весов одним файлом.
    assert "font-weight:200 700" in css
    assert "data:font/woff2;base64," in css


def test_font_faces_are_repository_local_woff2_files():
    for _, file_name, _ in dashboard.FONT_FACES:
        path = dashboard.FONT_DIR / file_name
        assert path.is_file(), f"нет локального шрифта {file_name}"
        assert path.suffix == ".woff2"
    assert dashboard.font_face_css(Path("no-such-font-dir")) == ""


def test_typography_uses_light_weights_only():
    assert dashboard.WEIGHT_REGULAR == 400
    assert dashboard.WEIGHT_MEDIUM == 500
    assert dashboard.WEIGHT_SEMIBOLD == 600

    source = Path("dashboard.py").read_text(encoding="utf-8")
    css_start = source.index("def inject_css")
    css = source[css_start:source.index("\ninject_css()")]
    # Прежний дизайн держал 650-900; после облегчения числовых литералов
    # в CSS не остаётся вовсе — только токены WEIGHT_*.
    assert not re.findall(r"font-weight:\s*[0-9]+", css)
    for token in ("{WEIGHT_REGULAR}", "{WEIGHT_MEDIUM}"):
        assert token in css
    assert max(
        int(value) for value in
        re.findall(r"WEIGHT_(?:REGULAR|MEDIUM|SEMIBOLD) = ([0-9]+)", source)
    ) <= 600


def test_borders_are_removed_or_kept_subtle():
    source = Path("dashboard.py").read_text(encoding="utf-8")
    css = source[source.index("def inject_css"):source.index("\ninject_css()")]

    # Жёсткая непрозрачная обводка карточек убрана полностью.
    assert not re.findall(r"border[^:}\n]*:\s*1px solid \{HAIRLINE\}", css)
    assert "{HAIRLINE_SOFT}" in css and "{HAIRLINE_FAINT}" in css

    for surface in (
        '.app-header', '.surface-card', '.note-box', '.panel-card',
        '.overview-table-wrap', '.status-chip',
    ):
        block = css[css.index(surface):]
        block = block[:block.index("}}")]
        assert re.search(r"border:\s*0|border:\s*none", block), surface

    # Полупрозрачные линии заметно светлее прежнего #E7EBE8.
    for token in ("HAIRLINE_SOFT", "HAIRLINE_FAINT"):
        declaration = re.search(rf'{token} = "(.+?)"', source).group(1)
        assert declaration.startswith("rgba(")
        assert float(declaration.rsplit(",", 1)[1].strip(" )")) <= 0.06


def test_icons_come_from_lucide_library_without_handwritten_paths():
    source = Path("dashboard.py").read_text(encoding="utf-8")
    # Ни одной вручную нарисованной геометрии: путей и viewBox в коде нет.
    assert "<path" not in source
    assert "viewBox" not in source
    # Единственное место, где вообще встречается тег svg, — постобработка
    # готового файла Lucide внутри lucide_icon().
    helper = source[source.index("def lucide_icon"):source.index("def icon_span")]
    assert source.count("<svg") == helper.count("<svg")

    icon = dashboard.lucide_icon("circle-check", 15)
    assert 'class="gx-icon lucide lucide-circle-check"' in icon
    assert 'width="15" height="15"' in icon
    assert 'aria-hidden="true"' in icon
    assert "<!--" not in icon  # лицензионный комментарий не попадает в разметку
    assert dashboard.lucide_icon("definitely-missing-icon") == ""
    assert dashboard.icon_span("definitely-missing-icon") == ""
    assert dashboard.icon_span("circle-check").startswith('<span class="gx-icon-wrap">')

    data_uri = dashboard.lucide_data_uri("triangle-alert", dashboard.STATUS_CRIT)
    assert data_uri.startswith("data:image/svg+xml;base64,")
    decoded = base64.b64decode(data_uri.split(",", 1)[1]).decode("utf-8")
    assert f'stroke="{dashboard.STATUS_CRIT}"' in decoded
    assert "currentColor" not in decoded


def test_icon_assets_are_vendored_with_license():
    license_file = dashboard.ICON_DIR / "LUCIDE-LICENSE.txt"
    assert license_file.is_file()
    assert "ISC" in license_file.read_text(encoding="utf-8")

    for name in ("layout-dashboard", "list-checks", "chart-column", "activity",
                 "circle-check", "triangle-alert", "trending-up", "circle-dot",
                 "layers", "map-pin", "chevron-down", "arrow-right", "gauge"):
        markup = (dashboard.ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
        assert "lucide-static" in markup  # апстрим-происхождение сохранено

    assert dashboard.FAVICON_ASSET.is_file()


def test_galit_brand_assets_are_flat_vector_and_transparent_rasters():
    import xml.etree.ElementTree as ET
    from PIL import Image

    svg = dashboard.BRAND_MARK_ASSET.read_text(encoding="utf-8")
    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 88 128"
    assert '#087A3D' in svg
    forbidden = ("<image", "data:image", "<rect", "linearGradient", "radialGradient",
                 "filter=", "mask=", "clipPath", "shadow")
    assert not any(token in svg for token in forbidden)
    assert len(root.findall("{http://www.w3.org/2000/svg}path")) == 1
    assert dashboard.FAVICON_ASSET == dashboard.BRAND_MARK_ASSET

    for size in (24, 32, 40, 48):
        image = Image.open(dashboard.ICON_DIR / f"galit-mark-{size}.png")
        assert image.mode == "RGBA" and image.size == (size, size)
        assert image.getchannel("A").getextrema() == (0, 255)
        opaque_colors = {
            pixel[:3] for pixel in image.getdata() if pixel[3] == 255
        }
        assert opaque_colors
        assert all(g > r * 5 and g > b * 1.8 for r, g, b in opaque_colors)

    favicon = Image.open(dashboard.ICON_DIR / "galit-favicon.ico").convert("RGBA")
    assert favicon.size == (48, 48)
    assert favicon.getchannel("A").getextrema() == (0, 255)

    source = Path("dashboard.py").read_text(encoding="utf-8")
    assert "page_icon=FAVICON_ASSET" in source
    assert 'class="dashboard-brand-mark"' in source


def test_ui_renders_lucide_icons_instead_of_emoji_placeholders():
    source = Path("dashboard.py").read_text(encoding="utf-8")
    # Символы-заглушки прежнего дизайна и эмодзи-favicon удалены.
    for placeholder in ("◌", "▦", "◆", "↗", "▣", "▽", "⌄", "◉", "🛢"):
        assert placeholder not in source
    assert 'page_icon="' not in source

    # Шапка рендерится через st.html и в AppTest.markdown не попадает,
    # поэтому её иконки проверяем в исходнике.
    header = source[source.index("def render_header"):source.index("def render_welcome")]
    assert 'icon_span("layers"' in header
    assert 'icon_span("map-pin"' in header
    assert header.count('icon_span("chevron-down"') == 2

    app = AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=120).run()
    app.get("file_uploader")[0].set_value(
        ("valid-fund.csv", MINIMAL_CSV, "text/csv")
    ).run()
    assert not app.exception

    markup = "\n".join(item.value for item in app.markdown)
    rendered = set(re.findall(r"lucide-([a-z-]+)", markup))
    assert {"chart-column", "list-checks", "layers", "arrow-right"} <= rendered
    assert rendered & {"triangle-alert", "trending-up", "circle-dot", "circle-check"}
    assert 'class="gx-icon-wrap"' in markup


# ----------------------------------------------- forecast parser / timeline

def test_forecast_history_parser_requires_timezone_and_preserves_undated():
    csv = (
        "well,timestamp,wax_severity,halite_severity,calcite_severity,"
        "corrosion_wall_loss_mm,oil_rate_m3_day,quality,source\n"
        "A,2026-08-01T00:00:00+00:00,0.2,,,,8,good,measured\n"
    ).encode()
    history = dashboard.forecast_history_from_csv(csv)
    assert len(history.snapshots) == 1
    bad = csv.replace(b"+00:00", b"")
    with pytest.raises(ValueError, match="timezone"):
        dashboard.forecast_history_from_csv(bad)

    case = dashboard.galit.synthetic.make_fund(1, seed=2)[0]
    forecast = dashboard.galit.forecast_well(diagnose(case), case)
    assert all(event.horizon_start_date is None for event in forecast.events)
    frame = dashboard.forecast_event_frame(forecast.events)
    assert frame["Ожидаемое окно"].eq("дата недоступна").all()
    assert len(dashboard.fig_forecast_timeline(forecast).data) == 0


def test_apptest_demo_has_economics_and_forecast_tabs():
    app = AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=120).run()
    button = next(item for item in app.button if item.label == "Демо-фонд (40 скважин)")
    button.click().run()
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    assert len(labels) == 14
    assert "Обводнение" in labels
    assert "Карта месторождения" in labels
    assert "Что будет, если?" in labels
    assert "Экономика риска" in labels and "Прогноз во времени" in labels
    assert "Оборудование / Прогноз отказов" in labels
    assert labels[-3:] == ["Цифровой паспорт", "Цифровой двойник", "Журнал мероприятий"]


# ------------------------------------------------ treatment journal UI

def test_treatment_helpers_autofill_and_currency_rows():
    case = dashboard.galit.synthetic.make_fund(1, seed=31)[0]
    result = diagnose(case)
    context = dashboard.treatment_well_context({case.name: case}, [result])[case.name]
    assert context["baseline_risk"] == result.integrated_risk
    assert context["complication_type"] == result.dominant
    assert context["well_group"] == case.lift_type

    now = dashboard.datetime(2026, 8, 23, 12, tzinfo=dashboard.timezone.utc)
    records = []
    for currency in ("BYN", "USD"):
        item = dashboard.galit.new_treatment(
            now=now, well_id="w-1", well_name="Well 1", event_at=now,
            complication_type="halite", description="Event", reagent_name="A",
            reagent_id=None, dosage=1, dosage_unit="mg/l", cost=10,
            currency=currency, treatment_type="inhibitor", well_group="ESP",
        )
        item = item.transition(dashboard.galit.TreatmentStatus.IN_PROGRESS, now=now)
        item = item.transition(dashboard.galit.TreatmentStatus.COMPLETED, now=now)
        records.append(item.transition(
            dashboard.galit.TreatmentStatus.ASSESSED, now=now,
            actual_result="Restored", result_metrics={"gain": 1}, success=True,
            effect_duration_days=10, recurrence=False,
        ))
    frame = dashboard.treatment_summary_frame(dashboard.galit.treatment_summary(records))
    assert set(frame["Валюта"]) == {"BYN", "USD"}
    assert frame["Оценено, n"].eq(2).all()


def test_apptest_journal_uses_temp_storage_and_reaches_assessed(tmp_path, monkeypatch):
    storage = tmp_path / "journal.json"
    monkeypatch.setenv("GALIT_TREATMENT_STORAGE", str(storage))
    app = AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=120).run()
    app.get("file_uploader")[0].set_value(
        ("valid-fund.csv", MINIMAL_CSV, "text/csv")
    ).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs].count("Журнал мероприятий") == 1

    next(item for item in app.text_area if item.label == "Описание события").set_value("Рост давления")
    next(item for item in app.text_input if item.label == "Реагент (название)").set_value("A")
    next(item for item in app.text_input if item.label == "Тип обработки").set_value("Ингибитор")
    next(item for item in app.button if item.label == "Сохранить план").click().run()
    assert not app.exception
    repository = dashboard.galit.TreatmentRepository(storage)
    record = repository.list()[0]
    assert record.well_name == "139" and record.baseline_risk is not None

    next(item for item in app.radio if item.label == "Раздел журнала").set_value(
        "Lifecycle и результат"
    ).run()
    next(item for item in app.button if item.label == "Начать мероприятие").click().run()
    assert repository.list()[0].status is dashboard.galit.TreatmentStatus.IN_PROGRESS
    next(item for item in app.button if item.label == "Завершить мероприятие").click().run()
    assert repository.list()[0].status is dashboard.galit.TreatmentStatus.COMPLETED

    next(item for item in app.text_area if item.label == "Фактический результат").set_value("Дебит восстановлен")
    next(item for item in app.number_input if item.label == "Значение показателя").set_value(2.0)
    next(item for item in app.number_input if item.label == "Длительность эффекта, сут").set_value(30.0)
    next(item for item in app.button if item.label == "Зафиксировать assessed").click().run()
    assessed = repository.list()[0]
    assert assessed.status is dashboard.galit.TreatmentStatus.ASSESSED
    assert assessed.revision == 4 and assessed.actual_result == "Дебит восстановлен"


def test_apptest_corrupt_treatment_storage_does_not_crash(tmp_path, monkeypatch):
    storage = tmp_path / "broken.json"
    storage.write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("GALIT_TREATMENT_STORAGE", str(storage))
    app = AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=120).run()
    app.get("file_uploader")[0].set_value(
        ("valid-fund.csv", MINIMAL_CSV, "text/csv")
    ).run()
    assert not app.exception
    assert any("повреждено" in item.value for item in app.error)
