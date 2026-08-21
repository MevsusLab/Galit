"""ГАЛИТ | Дашборд диагностики осложнений добычи (Streamlit).

Enterprise-интерфейс для инженеров и мастеров добычи:
  * загрузка фонда скважин из Excel/CSV;
  * ранжирование по интегральному риску с цветовой индикацией
    (норма / повышенный / критический);
  * интерактивные профили T(z) и P(z) (Plotly);
  * детальный разбор по каждой скважине.

Запуск:
    streamlit run dashboard.py

Формат входного файла -- см. шаблон «Скачать шаблон XLSX» в боковой панели.
Неизвестные столбцы игнорируются; необязательные столбцы при отсутствии
принимают значения по умолчанию ядра galit.

Дизайн -- корпоративный стиль «Белоруснефти»: белый фон, тёмно-зелёные
акценты, тёмно-серый текст, без декоративных элементов.
"""
from __future__ import annotations

import io
from dataclasses import replace
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pandas.io.formats.style import Styler

import galit
import galit.synthetic
from galit.calibration import ArtifactValidationError, ParameterSet
from galit.evaluation import evaluate_uploaded_rows, pilot_contract_frame
from galit.integrated import CRITICAL_FIELDS, QUALITY_FIELDS
from galit import (
    DataProvenance,
    DataQualityError,
    DiagnosisResult,
    FluidProperties,
    ProductionRate,
    ThermalParams,
    UncertaintyConfig,
    WaterAnalysis,
    WaxProperties,
    WellCase,
    WellGeometry,
    diagnose,
)

# ==========================================================================
# Корпоративная палитра «Белоруснефть»
# ==========================================================================

GREEN_900 = "#0B4A2F"   # тёмно-зелёный: заголовки, акцентные надписи
GREEN_700 = "#0F6B43"   # основной зелёный: кнопки, линии графиков
GREEN_500 = "#3D8B66"   # промежуточный
GREEN_100 = "#E4F0EA"   # светлый тинт: панели, разделители
INK = "#1F2422"         # основной текст (тёмно-серый, почти чёрный)
INK_MUTED = "#5A6560"   # вторичный текст
BORDER = "#D6DED9"      # границы панелей
SURFACE = "#F6F8F7"     # фон панелей

# Индикация статуса: норма / повышенный / критический
STATUS_OK_BG = "#E8F3EC"
STATUS_OK = "#2E7D32"
STATUS_WARN_BG = "#FBF3E4"
STATUS_WARN = "#B26A00"
STATUS_CRIT_BG = "#FBE9E7"
STATUS_CRIT = "#C62828"

RISK_WARN = 0.35   # ниже -- норма
RISK_CRIT = 0.60   # выше -- критический

# Последовательность цветов линий профилей (зелёная гамма + графитовый)
LINE_COLORS = ["#0F6B43", "#0B4A2F", "#3D8B66", "#6FA98C", "#275D57", "#5A6560"]

MECH_RU = {
    "halite": "Галит",
    "calcite": "Кальцит",
    "wax": "АСПО",
    "corrosion": "Коррозия",
}

FONT_FAMILY = "Segoe UI, Helvetica Neue, Arial, sans-serif"

# ==========================================================================
# Схема входного файла
# ==========================================================================

ION_KEYS = ["Na", "Cl", "Ca", "Mg", "K", "Ba", "Sr", "Fe", "HCO3", "SO4", "CO3"]

# Колонки, обязательные для расчёта
REQUIRED_COLUMNS = [
    "name", "depth_m", "tubing_id_m",
    "q_oil_m3d", "q_water_m3d", "gor_m3m3",
    "wat_stock_tank_c",
]

# Необязательные числовые колонки и их значения по умолчанию (из ядра galit)
OPTIONAL_DEFAULTS: dict[str, float | str] = {
    "inclination_deg": 0.0,
    "gamma_oil": 0.86,
    "gamma_gas": 0.75,
    "salinity_ppm": 300_000.0,
    "t_surface_c": 8.0,
    "geothermal_grad": 0.033,
    "u_to": 15.0,
    "production_days": 365.0,
    "ph": 6.0,
    "t_c": 40.0,
    "p_pa": 5.0e6,
    "wax_content_pct": 5.0,
    "co2_mol_frac": 0.02,
    "inhibitor_efficiency": 0.0,
    "p_wellhead_pa": 1.2e6,
    "lift_type": "ЭЦН",
}

# Синонимы заголовков (нормализуются: нижний регистр, без пробелов)
HEADER_ALIASES = {
    "name": "name", "well": "name", "well_name": "name",
    "скважина": "name", "название": "name",
    "lift": "lift_type", "способ": "lift_type", "тип насоса": "lift_type",
}

# Типовой солевой состав пластовой воды Припятского прогиба (мг/л).
# Используется, когда ионный состав в файле не задан: на профили T(z)/P(z)
# и глубину АСПО он не влияет, влияет только на блок солеотложений.
TYPICAL_BRINE: dict[str, float] = {
    "Na": 95_000.0, "Cl": 205_000.0, "Ca": 28_000.0, "Mg": 3_100.0,
    "K": 1_800.0, "HCO3": 130.0, "SO4": 250.0,
}

COLUMN_DOCS = [
    ("name", "—", "Название скважины", "да", "—"),
    ("depth_m", "м", "Глубина по стволу до забоя", "да", "—"),
    ("tubing_id_m", "м", "Внутренний диаметр НКТ", "да", "—"),
    ("inclination_deg", "град", "Средний угол от вертикали", "нет", "0"),
    ("q_oil_m3d", "м3/сут", "Дебит нефти", "да", "—"),
    ("q_water_m3d", "м3/сут", "Дебит воды", "да", "—"),
    ("gor_m3m3", "м3/м3", "Газовый фактор", "да", "—"),
    ("gamma_oil", "—", "Отн. плотность нефти по воде", "нет", "0.86"),
    ("gamma_gas", "—", "Отн. плотность газа по воздуху", "нет", "0.75"),
    ("salinity_ppm", "мг/л", "Минерализация воды", "нет", "300000"),
    ("t_surface_c", "°C", "Температура пород у поверхности", "нет", "8"),
    ("geothermal_grad", "К/м", "Геотермический градиент", "нет", "0.033"),
    ("u_to", "Вт/(м2·К)", "Коэф. теплопередачи", "нет", "15"),
    ("production_days", "сут", "Непрерывная наработка", "нет", "365"),
    ("Na … CO3", "мг/л", "Ионный состав воды (Na, Cl, Ca, Mg, K, HCO3, SO4, …)", "нет", "типовой состав"),
    ("ph", "—", "pH воды", "нет", "6.0"),
    ("t_c", "°C", "Температура отбора пробы воды", "нет", "40"),
    ("p_pa", "Па", "Давление отбора пробы воды", "нет", "5·10^6"),
    ("wat_stock_tank_c", "°C", "WAT дегазированной нефти", "да", "—"),
    ("wax_content_pct", "% масс.", "Содержание парафина", "нет", "5"),
    ("co2_mol_frac", "доли", "Доля CO2 в попутном газе", "нет", "0.02"),
    ("inhibitor_efficiency", "доли", "Эффективность ингибитора", "нет", "0"),
    ("lift_type", "—", "Способ эксплуатации: ЭЦН | ШГН | фонтан", "нет", "ЭЦН"),
    ("p_wellhead_pa", "Па", "Буферное давление", "нет", "1.2·10^6"),
]

# ==========================================================================
# Страница и корпоративный CSS
# ==========================================================================

st.set_page_config(
    page_title="ГАЛИТ — диагностика осложнений",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Корпоративный стиль: белый фон, зелёные акценты, тёмный текст."""
    st.markdown(
        f"""
        <style>
        /* ---------- базовая типографика ---------- */
        .stApp {{
            font-family: {FONT_FAMILY};
            color: {INK};
            background: #FFFFFF;
        }}
        p, li, span, label {{ color: {INK}; }}
        h1, h2, h3 {{ color: {INK}; }}

        /* строгий режим: без фирменной радужной полосы и колонтитула */
        div[data-testid="stDecoration"] {{ display: none; }}
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: #FFFFFF; }}

        /* ---------- шапка приложения ---------- */
        .app-header {{
            border-left: 5px solid {GREEN_700};
            padding: 2px 0 4px 18px;
            margin-bottom: 4px;
        }}
        .app-title {{
            font-size: 26px; font-weight: 700; color: {GREEN_900};
            letter-spacing: 0.5px; line-height: 1.2;
        }}
        .app-subtitle {{
            font-size: 14px; color: {INK_MUTED}; margin-top: 2px;
        }}
        .app-rule {{
            border: none; border-top: 1px solid {BORDER}; margin: 14px 0 20px 0;
        }}

        /* ---------- боковая панель ---------- */
        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid {BORDER};
        }}
        .sidebar-brand {{
            font-size: 15px; font-weight: 700; color: {GREEN_900};
            padding: 10px 0 2px 0; letter-spacing: 0.4px;
        }}
        .sidebar-brand::before {{
            content: ""; display: inline-block; width: 10px; height: 10px;
            background: {GREEN_700}; margin-right: 9px;
        }}
        .sidebar-hint {{ font-size: 12px; color: {INK_MUTED}; line-height: 1.45; }}

        /* ---------- кнопки ---------- */
        .stButton > button {{
            background: {GREEN_700}; color: #FFFFFF; border: none;
            border-radius: 4px; font-weight: 600; font-size: 14px;
            padding: 6px 16px; width: 100%;
            transition: background 0.15s ease;
        }}
        .stButton > button:hover {{ background: {GREEN_900}; color: #FFFFFF; }}
        .stButton > button:focus {{
            box-shadow: 0 0 0 2px {GREEN_100}; color: #FFFFFF;
        }}
        .stDownloadButton > button {{
            background: #FFFFFF; color: {GREEN_700};
            border: 1px solid {GREEN_700}; border-radius: 4px;
            font-weight: 600; font-size: 14px; padding: 4px 16px; width: 100%;
        }}
        .stDownloadButton > button:hover {{
            background: {GREEN_100}; color: {GREEN_900};
            border: 1px solid {GREEN_900};
        }}

        /* ---------- вкладки ---------- */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            background: #FFFFFF; color: {INK_MUTED};
            font-weight: 600; font-size: 14px;
            border-radius: 0; padding: 8px 18px;
        }}
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
            color: {GREEN_900};
            border-bottom: 3px solid {GREEN_700};
        }}

        /* ---------- карточки-метрики ---------- */
        div[data-testid="stMetric"] {{
            background: {SURFACE}; border: 1px solid {BORDER};
            border-radius: 6px; padding: 14px 16px 10px 16px;
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {INK_MUTED}; font-size: 12px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.6px;
            margin-bottom: 4px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {INK}; font-weight: 700;
        }}

        /* ---------- таблица и графики ---------- */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER}; border-radius: 6px;
        }}
        .js-plotly-plot .plotly .modebar {{ background: transparent; }}

        /* ---------- информационные блоки ---------- */
        .note-box {{
            background: {SURFACE}; border: 1px solid {BORDER};
            border-left: 4px solid {GREEN_700};
            border-radius: 4px; padding: 12px 16px; font-size: 14px;
        }}
        .legend-chip {{
            display: inline-block; margin-right: 22px;
            font-size: 13px; color: {INK};
        }}
        .legend-dot {{
            display: inline-block; width: 10px; height: 10px;
            border-radius: 2px; margin-right: 7px; vertical-align: -1px;
        }}
        .section-title {{
            font-size: 16px; font-weight: 700; color: {INK};
            margin: 6px 0 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ==========================================================================
# Чтение и разбор входного файла (чистые функции -- тестируются pytest)
# ==========================================================================


def normalize_headers(columns: Any) -> dict[str, str]:
    """Отображение «нормализованный заголовок -> каноническое имя»."""
    canonical = (
        {c.lower(): c for c in REQUIRED_COLUMNS}
        | {c.lower(): c for c in OPTIONAL_DEFAULTS}
        | {ion.lower(): ion for ion in ION_KEYS}
        | {alias: target for alias, target in HEADER_ALIASES.items()}
    )
    mapping: dict[str, str] = {}
    for col in columns:
        key = str(col).strip().lower()
        if key in canonical:
            mapping[str(col)] = canonical[key]
    return mapping


def frame_to_cases(df: pd.DataFrame) -> tuple[list[WellCase], list[str]]:
    """Преобразование строк таблицы в WellCase.

    Возвращает (успешные случаи, список сообщений об ошибках/заменах).
    Ошибка в одной строке не блокирует расчёт остальных.
    """
    errors: list[str] = []
    colmap = normalize_headers(df.columns)
    by_canon: dict[str, str] = {}
    for orig, canon in colmap.items():
        by_canon.setdefault(canon, orig)

    missing = [c for c in REQUIRED_COLUMNS if c not in by_canon]
    if missing:
        errors.append("В файле нет обязательных колонок: " + ", ".join(missing))
        return [], errors

    ion_cols = {orig: ion for orig, ion in colmap.items() if ion in ION_KEYS}
    cases: list[WellCase] = []

    for i, (_, row) in enumerate(df.iterrows(), start=2):
        label = _well_label(row[by_canon["name"]], i)
        vals: dict[str, float | str] = {}
        defaulted: set[str] = set()
        ok = True

        for canon, default in OPTIONAL_DEFAULTS.items():
            orig = by_canon.get(canon)
            if orig is None or pd.isna(row[orig]):
                vals[canon] = default
                defaulted.add(canon)
                continue
            value = pd.to_numeric(row[orig], errors="coerce") \
                if isinstance(default, float) else row[orig]
            if isinstance(default, float) and pd.isna(value):
                errors.append(
                    f"«{label}»: колонка {canon} — не число ({row[orig]!r}), "
                    f"принято значение по умолчанию {default}"
                )
                vals[canon] = default
                defaulted.add(canon)
            else:
                vals[canon] = float(value) if isinstance(default, float) else str(value).strip()

        for canon in ("depth_m", "tubing_id_m", "q_oil_m3d",
                      "q_water_m3d", "gor_m3m3", "wat_stock_tank_c"):
            orig = by_canon[canon]
            if pd.isna(row[orig]):
                errors.append(f"«{label}»: не задано поле {canon} — строка пропущена")
                ok = False
                continue
            value = pd.to_numeric(row[orig], errors="coerce")
            if pd.isna(value):
                errors.append(f"«{label}»: поле {canon} — не число — строка пропущена")
                ok = False
                continue
            vals[canon] = float(value)
        if not ok:
            continue

        ions: dict[str, float] = {}
        for orig, ion in ion_cols.items():
            value = pd.to_numeric(row[orig], errors="coerce")
            if pd.notna(value) and value > 0.0:
                ions[ion] = float(value)
        typical_brine = not ions
        if typical_brine:
            ions = dict(TYPICAL_BRINE)

        try:
            cases.append(_row_to_case(label, vals, ions, defaulted, typical_brine))
        except (ValueError, KeyError) as exc:
            errors.append(f"«{label}»: {exc} — строка пропущена")

    return cases, errors


def _row_to_case(label: str, vals: dict[str, float | str],
                 ions: dict[str, float], defaulted: set[str] | None = None,
                 typical_brine: bool = False) -> WellCase:
    defaulted = defaulted or set()
    field_paths = {
        "salinity_ppm": "fluid.salinity_ppm",
        "t_surface_c": "thermal.t_surface_c",
        "geothermal_grad": "thermal.geothermal_grad",
        "u_to": "thermal.u_to",
        "ph": "water.ph", "t_c": "water.t_c", "p_pa": "water.p_pa",
        "wax_content_pct": "wax.wax_content_pct",
        "co2_mol_frac": "co2_mol_frac",
        "inhibitor_efficiency": "inhibitor_efficiency",
        "p_wellhead_pa": "p_wellhead_pa",
    }
    sources = {field_paths[k]: "default" for k in defaulted if k in field_paths}
    if typical_brine:
        sources["water.ions_mg_l"] = "synthetic"
    provenance = DataProvenance(sources=sources)
    return WellCase(
        name=label,
        geometry=WellGeometry(
            depth_m=float(vals["depth_m"]),
            tubing_id_m=float(vals["tubing_id_m"]),
            inclination_deg=float(vals["inclination_deg"]),
        ),
        rate=ProductionRate(
            q_oil_m3d=float(vals["q_oil_m3d"]),
            q_water_m3d=float(vals["q_water_m3d"]),
            gor_m3m3=float(vals["gor_m3m3"]),
        ),
        fluid=FluidProperties(
            gamma_oil=float(vals["gamma_oil"]),
            gamma_gas=float(vals["gamma_gas"]),
            salinity_ppm=float(vals["salinity_ppm"]),
        ),
        thermal=ThermalParams(
            t_surface_c=float(vals["t_surface_c"]),
            geothermal_grad=float(vals["geothermal_grad"]),
            u_to=float(vals["u_to"]),
            production_days=float(vals["production_days"]),
        ),
        water=WaterAnalysis(
            ions_mg_l=ions,
            ph=float(vals["ph"]),
            t_c=float(vals["t_c"]),
            p_pa=float(vals["p_pa"]),
        ),
        wax=WaxProperties(
            wat_stock_tank_c=float(vals["wat_stock_tank_c"]),
            wax_content_pct=float(vals["wax_content_pct"]),
        ),
        co2_mol_frac=float(vals["co2_mol_frac"]),
        inhibitor_efficiency=float(vals["inhibitor_efficiency"]),
        lift_type=str(vals["lift_type"]),
        p_wellhead_pa=float(vals["p_wellhead_pa"]),
        provenance=provenance,
    )


def template_frame() -> pd.DataFrame:
    """Одна строка-пример с полным набором колонок (скв. Речицкая 123)."""
    example = {
        "name": "Речицкая 123",
        "depth_m": 3200.0, "tubing_id_m": 0.062, "inclination_deg": 15.0,
        "q_oil_m3d": 8.0, "q_water_m3d": 72.0, "gor_m3m3": 65.0,
        "gamma_oil": 0.86, "gamma_gas": 0.78, "salinity_ppm": 290_000.0,
        "t_surface_c": 8.0, "geothermal_grad": 0.033, "u_to": 15.0,
        "production_days": 400.0,
        "Na": 95_000.0, "Cl": 205_000.0, "Ca": 28_000.0, "Mg": 3_100.0,
        "K": 1_800.0, "HCO3": 130.0, "SO4": 250.0,
        "ph": 6.0, "t_c": 40.0, "p_pa": 5.0e6,
        "wat_stock_tank_c": 34.0, "wax_content_pct": 6.5,
        "co2_mol_frac": 0.012, "inhibitor_efficiency": 0.0,
        "lift_type": "ЭЦН", "p_wellhead_pa": 1.4e6,
    }
    return pd.DataFrame([example])


def template_bytes() -> bytes:
    """Шаблон XLSX: лист с данными + лист «Инструкция»."""
    docs = pd.DataFrame(
        COLUMN_DOCS, columns=["Колонка", "Ед. изм.", "Описание", "Обязательна", "По умолчанию"]
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_frame().to_excel(writer, sheet_name="Данные", index=False)
        docs.to_excel(writer, sheet_name="Инструкция", index=False)
    return buffer.getvalue()


def pilot_template_bytes() -> bytes:
    """XLSX contract for prospective outcomes and three frozen strategies."""
    columns = [row["field"] for row in pilot_contract_frame()]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(columns=columns).to_excel(writer, sheet_name="Pilot outcomes", index=False)
        pd.DataFrame(pilot_contract_frame()).to_excel(writer, sheet_name="Data contract", index=False)
    return buffer.getvalue()


def pilot_evaluation_frame(evaluation: dict[str, Any]) -> pd.DataFrame:
    """Compact dashboard table without relabeling illustrative output as accuracy."""
    rows = []
    for item in evaluation.get("strategies", []):
        rows.append({
            "Стратегия": item["strategy"], "K": item["k"],
            "Precision@K": item["precision_at_k"], "Recall@K": item["recall_at_k"],
            "NDCG@K": item["ndcg_at_k"], "Пропущено событий": item["missed_events"],
            "Лишних вмешательств": item["unnecessary_interventions"],
            "Prevented loss": item.get("prevented_loss"), "Net value": item.get("net_value"),
        })
    return pd.DataFrame(rows)


def _normalize_csv_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Запятая как десятичный разделитель в числовых колонках CSV.

    Файлы, сохранённые из русского Excel, содержат «0,062» вместо «0.062».
    Текстовые колонки (названия скважин) остаются нетронутыми.
    """
    name_columns = {
        original for original, canonical in normalize_headers(df.columns).items()
        if canonical == "name"
    }
    for col in df.columns:
        if str(col) in name_columns:
            continue
        # pandas 2 даёт строкам object, pandas 3 -- специализированный str
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            if converted.notna().any():
                df[col] = converted
    return df


def _well_label(value: Any, row_number: int) -> str:
    """Стабильная подпись: числовой идентификатор 139 не становится 139.0."""
    if pd.isna(value):
        return f"строка {row_number}"
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip() or f"строка {row_number}"


@st.cache_data(show_spinner=False)
def read_table(data: bytes, file_name: str) -> pd.DataFrame:
    """Чтение XLSX/CSV. CSV -- с автоопределением разделителя и кодировки."""
    if file_name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(data))
    try:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python",
                         encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python",
                         encoding="cp1251")
    return _normalize_csv_decimals(df)


def load_artifact_bytes(data: bytes) -> ParameterSet:
    """Strict dashboard helper; uploaded bytes are parsed without filesystem paths."""
    import json
    raw = json.loads(data.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
        ArtifactValidationError(f"Non-finite JSON value {value} is forbidden")
    ))
    if not isinstance(raw, dict):
        raise ArtifactValidationError("Calibration artifact root must be a JSON object")
    return ParameterSet.from_dict(raw)


def artifact_summary(parameters: ParameterSet | None) -> dict[str, Any]:
    if parameters is None:
        return {"id": "baseline", "version": "baseline", "status": "baseline",
                "holdout_metrics": {}, "limitations": []}
    return {"id": parameters.artifact_id, "version": parameters.schema_version,
            "status": parameters.validation_status,
            "holdout_metrics": dict(parameters.metrics).get("holdout", {}),
            "limitations": list(parameters.limitations)}


@st.cache_data(show_spinner="Расчёт фонда…")
def diagnose_frame(
    df: pd.DataFrame,
    production_mode: bool = False,
    include_uncertainty: bool = False,
    artifact_json: bytes | None = None,
) -> tuple[list[DiagnosisResult], list[str]]:
    cases, errors = frame_to_cases(df)
    results: list[DiagnosisResult] = []
    config = UncertaintyConfig() if include_uncertainty else None
    runtime = load_artifact_bytes(artifact_json).to_runtime() if artifact_json else None
    for case in cases:
        try:
            results.append(diagnose(
                case, production_mode=production_mode, uncertainty=config,
                runtime_calibration=runtime,
            ))
        except DataQualityError as exc:
            errors.append(f"«{case.name}»: не ранжируется — {'; '.join(exc.reasons)}")
    return results, errors


@st.cache_data(show_spinner=False)
def demo_fund() -> list[DiagnosisResult]:
    return [diagnose(c) for c in galit.synthetic.make_fund(40)]


# ==========================================================================
# Доверие, объяснимость и безопасное применение (чистые функции)
# ==========================================================================

FIELD_RU = {
    "geometry.depth_m": "глубина скважины",
    "geometry.tubing_id_m": "внутренний диаметр НКТ",
    "rate.q_oil_m3d": "дебит нефти",
    "rate.q_water_m3d": "дебит воды",
    "rate.gor_m3m3": "газовый фактор",
    "thermal.t_surface_c": "температура у поверхности",
    "thermal.geothermal_grad": "геотермический градиент",
    "thermal.u_to": "коэффициент теплопередачи",
    "p_wellhead_pa": "буферное давление",
    "water.ions_mg_l": "ионный состав воды",
    "water.ph": "pH воды",
    "water.t_c": "температура пробы воды",
    "water.p_pa": "давление отбора пробы",
    "fluid.salinity_ppm": "минерализация воды",
    "wax.wat_stock_tank_c": "WAT дегазированной нефти",
    "wax.wax_content_pct": "содержание парафина",
    "co2_mol_frac": "доля CO₂ в газе",
    "inhibitor_efficiency": "эффективность ингибитора",
}
SOURCE_RU = {
    "measured": "измерено",
    "default": "по умолчанию",
    "synthetic": "синтетическое",
    "derived": "оценено",
    "missing": "отсутствует",
}
GROUP_RU = {
    "wellbore": "Ствол и режим работы",
    "halite_calcite": "Вода и солеотложения",
    "wax": "АСПО",
    "corrosion": "Коррозия",
}


def contribution_frame(result: DiagnosisResult) -> pd.DataFrame:
    """Разложение интегрального риска без повторения физического расчёта."""
    rows = []
    total = result.integrated_risk
    for mechanism in ("halite", "calcite", "wax", "corrosion"):
        severity = result.severity[mechanism]
        weight = result.mechanism_weights[mechanism]
        contribution = severity * weight
        rows.append({
            "Механизм": MECH_RU[mechanism],
            "Тяжесть": severity,
            "Вес политики": weight,
            "Вклад в риск": contribution,
            "Доля integrated risk": contribution / total if total > 0.0 else 0.0,
        })
    return pd.DataFrame(rows)


def provenance_groups(result: DiagnosisResult) -> dict[str, Any]:
    """Подготовить происхождение ключевых входов и критичные пробелы."""
    quality = result.quality
    missing = set(quality.missing_fields)
    rows = []
    for field_name in sorted(QUALITY_FIELDS):
        source = "missing" if field_name in missing else quality.sources.get(
            field_name, "measured"
        )
        rows.append({
            "Поле": FIELD_RU.get(field_name, field_name),
            "Техническое поле": field_name,
            "Происхождение": SOURCE_RU[source],
            "source": source,
        })
    critical: dict[str, list[str]] = {}
    for group, fields in CRITICAL_FIELDS.items():
        bad = sorted(
            field_name for field_name in fields
            if field_name in missing
            or quality.sources.get(field_name, "measured") in {"default", "synthetic"}
        )
        if bad:
            critical[GROUP_RU.get(group, group)] = [
                FIELD_RU.get(field_name, field_name) for field_name in bad
            ]
    return {"rows": rows, "critical": critical}


def categorize_warnings(warnings: list[str]) -> dict[str, list[str]]:
    """Сгруппировать предупреждения, сохранив исходные строки без изменений."""
    categories: dict[str, list[str]] = {}
    for warning in warnings:
        text = warning.lower()
        if "screening" in text or "качество данных" in text or "default" in text:
            category = "Качество и назначение расчёта"
        elif any(token in text for token in ("неприменим", "stiff-davis", "tds", "интервал")):
            category = "Область применимости модели"
        elif any(token in text for token in ("сужение", "корроз", "плёнк", "кислот")):
            category = "Взаимодействия и технологические ограничения"
        else:
            category = "Расчётные допущения"
        categories.setdefault(category, []).append(warning)
    return categories


def decision_trace(result: DiagnosisResult) -> dict[str, Any]:
    """Краткая трассировка от доминирующего механизма к безопасному next step."""
    active = [MECH_RU[key] for key, value in result.severity.items() if value >= RISK_WARN]
    recommendation = result.recommendation
    conflict = "не выявлен"
    if len(active) > 1 or any(
        token in recommendation.lower() for token in ("внимание", "не применять", "сопутствующие")
    ):
        conflict = "требуется совместная проверка: " + ", ".join(active)

    bad_by_group = provenance_groups(result)["critical"]
    needed = [field for fields in bad_by_group.values() for field in fields]
    if not needed:
        mechanism_fields = {
            "halite": ["ионный состав воды", "минерализация и фактические T/P пробы"],
            "calcite": ["ионный состав, pH и фактические T/P пробы"],
            "wax": ["WAT и содержание парафина на актуальной пробе"],
            "corrosion": ["CO₂, pH, фактическая эффективность ингибитора и купон/датчик коррозии"],
        }
        needed = mechanism_fields.get(result.dominant, ["фактические промысловые входы"])
    return {
        "dominant": MECH_RU.get(result.dominant, result.dominant),
        "reason": recommendation,
        "conflict": conflict,
        "measure_next": needed,
    }


def corrosion_counterfactual(
    case: WellCase,
    efficiency: float,
    *,
    production_mode: bool = False,
) -> dict[str, Any]:
    """Повторный diagnose для сценария ингибирования; исходный case не изменяется."""
    if not 0.0 <= efficiency <= 1.0:
        return {"supported": False, "reason": "Эффективность должна быть в диапазоне 0–1."}
    sources = dict(case.provenance.sources)
    sources["inhibitor_efficiency"] = "derived"
    scenario_case = replace(
        case,
        inhibitor_efficiency=efficiency,
        provenance=DataProvenance(
            sources=sources,
            missing_fields=list(case.provenance.missing_fields),
            defaulted_fields=list(case.provenance.defaulted_fields),
            synthetic_fields=list(case.provenance.synthetic_fields),
        ),
    )
    try:
        before = diagnose(case, production_mode=production_mode)
        after = diagnose(scenario_case, production_mode=production_mode)
    except DataQualityError as exc:
        return {"supported": False, "reason": "; ".join(exc.reasons)}
    return {
        "supported": True,
        "label": "Сценарий чувствительности, не прогноз",
        "parameter": "эффективность ингибитора CO₂-коррозии",
        "before_value": case.inhibitor_efficiency,
        "after_value": efficiency,
        "before": before,
        "after": after,
    }


def action_is_safe(result: DiagnosisResult) -> tuple[bool, str]:
    if result.quality.production_ready:
        return True, "Данные пригодны для промышленного режима ядра."
    return False, "Действие заблокировано: есть критичные default/synthetic/missing inputs."


def empty_results_status(parsed_count: int, production_mode: bool) -> tuple[str, str]:
    """Классифицировать пустой результат без смешения parsing и quality gate."""
    if parsed_count > 0 and production_mode:
        return (
            "quality_gate",
            "Файл распознан, но строки не прошли промышленный контроль качества.",
        )
    return (
        "parsing",
        "Ни одна строка не распознана. Проверьте структуру файла по шаблону.",
    )


def render_file_remarks(errors: list[str]) -> None:
    """Показать замечания до любого раннего выхода из main."""
    if errors:
        with st.expander(f"Замечания при разборе и контроле качества ({len(errors)})"):
            for error in errors:
                st.markdown(f"- {error}")


# ==========================================================================
# Таблица ранжирования с условным форматированием
# ==========================================================================


def risk_status(risk: float) -> tuple[str, str, str]:
    """(фон строки, цвет текста, метка) по уровню интегрального риска."""
    if risk >= RISK_CRIT:
        return STATUS_CRIT_BG, STATUS_CRIT, "критический"
    if risk >= RISK_WARN:
        return STATUS_WARN_BG, STATUS_WARN, "повышенный"
    return STATUS_OK_BG, STATUS_OK, "норма"


def rank_frame(results: list[DiagnosisResult]) -> pd.DataFrame:
    """Рейтинговая таблица фонда (сортировка по убыванию риска)."""
    rows = []
    for r in sorted(results, key=lambda x: x.integrated_risk, reverse=True):
        risk_range = None
        deposition_probability = None
        if r.uncertainty is not None and r.uncertainty.integrated_risk is not None:
            interval = r.uncertainty.integrated_risk
            risk_range = f"{interval.p05:.2f}–{interval.p95:.2f}"
            deposition_probability = r.uncertainty.probability_of_deposition
        rows.append({
            "№": len(rows) + 1,
            "Скважина": r.well,
            "Риск": r.integrated_risk,
            "Сценарный диапазон риска": risk_range,
            "Вероятность АСПО": deposition_probability,
            "Статус": risk_status(r.integrated_risk)[2],
            "Качество": r.quality.grade,
            "Полнота": r.quality.completeness,
            "Production-ready": "да" if r.quality.production_ready else "нет",
            "Критичные defaults": "нет" if r.quality.production_ready else "есть",
            "Лидер": MECH_RU.get(r.dominant, r.dominant),
            "Галит": r.severity["halite"],
            "Кальцит": r.severity["calcite"],
            "АСПО": r.severity["wax"],
            "Коррозия": r.severity["corrosion"],
            "Начало АСПО, м": r.wax_onset_m,
            "V корр., мм/год": r.corrosion["rate_mm_yr"],
            "Рекомендация": r.recommendation,
        })
    return pd.DataFrame(rows)


def style_rank(df: pd.DataFrame) -> Styler:
    """Заливка строк по статусу; ячейка «Риск» — индикатор-плашка."""

    def row_paint(row: pd.Series) -> list[str]:
        bg, fg, _ = risk_status(float(row["Риск"]))
        styles = [f"background-color: {bg}" for _ in df.columns]
        risk_i = df.columns.get_loc("Риск")
        styles[risk_i] = f"background-color: {fg}; color: #FFFFFF"
        leader_i = df.columns.get_loc("Лидер")
        styles[leader_i] = f"{styles[leader_i]}; color: {GREEN_900}"
        return styles

    return (
        df.style.apply(row_paint, axis=1)
        .format({
            "Риск": "{:.2f}",
            "Полнота": "{:.0%}",
            "Галит": "{:.2f}", "Кальцит": "{:.2f}",
            "АСПО": "{:.2f}", "Коррозия": "{:.2f}",
            "Начало АСПО, м": "{:.0f}",
            "V корр., мм/год": "{:.2f}",
        }, na_rep="—")
        .hide(axis="index")
    )


# ==========================================================================
# Графики (Plotly)
# ==========================================================================


def _style_axes(fig: go.Figure) -> None:
    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=13, color=INK),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(font=dict(family=FONT_FAMILY)),
    )
    fig.update_xaxes(gridcolor=BORDER, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(
        gridcolor=BORDER, zeroline=False, linecolor=BORDER,
        autorange="reversed", title="Глубина, м",
    )


def fig_profiles(results: list[DiagnosisResult], labels: list[str],
                 detail: DiagnosisResult | None = None) -> go.Figure:
    """Профили T(z) и P(z) [МПа] по выбранным скважинам, глубина -- вниз."""
    by_label = dict(result_labels(results))
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=("Температура T(z)", "Давление P(z), МПа"),
    )
    palette = dict(zip(labels, LINE_COLORS))

    for label in labels:
        r = by_label.get(label)
        if r is None:
            continue
        color = palette[label]
        emphasis = detail is not None and r is detail
        width = 3.0 if emphasis else 1.6
        opacity = 1.0 if emphasis else 0.55
        fig.add_trace(go.Scatter(
            x=r.temps, y=r.depths, mode="lines", name=label,
            line=dict(color=color, width=width), opacity=opacity,
            hovertemplate="T = %{x:.1f} °C · z = %{y:.0f} м",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[p / 1e6 for p in r.pressures], y=r.depths, mode="lines",
            name=label, showlegend=False,
            line=dict(color=color, width=width), opacity=opacity,
            hovertemplate="P = %{x:.2f} МПа · z = %{y:.0f} м",
        ), row=1, col=2)

    # В детальном разборе: кривая WAT(P) и отметка глубины начала АСПО
    if detail is not None:
        fig.add_trace(go.Scatter(
            x=detail.wat_profile, y=detail.depths, mode="lines",
            name="WAT(P)", showlegend=True,
            line=dict(color=INK_MUTED, width=1.4, dash="dot"),
            hovertemplate="WAT = %{x:.1f} °C · z = %{y:.0f} м",
        ), row=1, col=1)
        if detail.wax_onset_m is not None:
            for col in (1, 2):
                fig.add_hline(
                    y=detail.wax_onset_m, line_dash="dash",
                    line_color=STATUS_CRIT, line_width=1.2,
                    annotation_text=f"начало АСПО · {detail.wax_onset_m:.0f} м",
                    annotation_font=dict(size=11, color=STATUS_CRIT),
                    row=1, col=col,
                )
    _style_axes(fig)
    fig.update_yaxes(title="Глубина, м", row=1, col=2, autorange="reversed")
    return fig


def fig_severity(detail: DiagnosisResult) -> go.Figure:
    """Вклад механизмов в риск по одной скважине."""
    mechanisms = ["halite", "calcite", "wax", "corrosion"]
    values = [detail.severity[m] for m in mechanisms]
    colors = [
        STATUS_OK if v < RISK_WARN else STATUS_WARN if v < RISK_CRIT else STATUS_CRIT
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=values,
        y=[MECH_RU[m] for m in mechanisms],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}",
    ))
    fig.update_xaxes(range=[0, 1.08], gridcolor=BORDER, title="Тяжесть, 0–1")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=13, color=INK),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=30, t=10, b=10), height=240,
        showlegend=False,
    )
    return fig


# ==========================================================================
# Компоновка интерфейса
# ==========================================================================


def render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">ГАЛИТ</div>
            <div class="app-subtitle">
                Интегрированная диагностика осложнений: галит · кальцит · АСПО ·
                коррозия. Ранжирование фонда и профили T(z), P(z).
            </div>
        </div>
        <hr class="app-rule">
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    required_rows = "".join(
        f"<tr><td><code>{c}</code></td><td>{u}</td><td>{d}</td></tr>"
        for c, u, d, req, _ in COLUMN_DOCS if req == "да"
    )
    st.markdown(
        f"""
        <div class="note-box">
            <b>Порядок работы</b>
            <ol style="margin: 8px 0 0 0; padding-left: 20px; line-height: 1.7;">
                <li>Скачайте шаблон XLSX в боковой панели и заполните фонд скважин
                    (или проверьте структуру своего файла).</li>
                <li>Загрузите файл — дашборд рассчитает интегральный риск по каждой
                    скважине и ранжирует фонд.</li>
                <li>Изучите рейтинг, профили T(z)/P(z) и детальный разбор
                    по интересующей скважине.</li>
            </ol>
        </div>
        <p style="margin-top: 18px; margin-bottom: 6px;" class="section-title">
            Обязательные колонки файла</p>
        <table style="border-collapse: collapse; font-size: 13px;">
            <tr style="color: {INK_MUTED}; text-align: left;">
                <th style="padding: 4px 18px 4px 0;">Колонка</th>
                <th style="padding: 4px 18px 4px 0;">Ед. изм.</th>
                <th style="padding: 4px 18px 4px 0;">Описание</th>
            </tr>{required_rows}
        </table>
        <p style="color: {INK_MUTED}; font-size: 13px; margin-top: 10px;">
            Остальные параметры необязательны — при отсутствии принимаются значения
            по умолчанию (лист «Инструкция» в шаблоне). Без файла можно открыть
            демо-фонд из 40 скважин кнопкой в боковой панели.</p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Боковая панель; возвращает файл и флаг промышленного режима."""
    upload = None
    production_mode = False
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">ПО «ГАЛИТ»</div>'
            '<div class="sidebar-hint">Диагностика осложнений добычи</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        upload = st.file_uploader(
            "Файл фонда (XLSX / CSV)",
            type=["xlsx", "xls", "csv"],
            help="Структура колонок — в шаблоне и на стартовой странице",
        )
        if upload is not None:
            st.session_state["demo"] = False

        artifact_upload = st.file_uploader(
            "Calibration artifact (JSON, optional)", type=["json"],
            help="По умолчанию baseline. Blocked/invalid artifacts не применяются.",
        )
        st.session_state["calibration_artifact_bytes"] = (
            artifact_upload.getvalue() if artifact_upload is not None else None
        )
        try:
            selected = load_artifact_bytes(artifact_upload.getvalue()) if artifact_upload else None
            summary = artifact_summary(selected)
            st.caption(f"Config: {summary['id']} · {summary['version']} · {summary['status']}")
            if summary["holdout_metrics"]:
                st.json(summary["holdout_metrics"], expanded=False)
            for limitation in summary["limitations"]:
                st.warning(limitation)
        except (ArtifactValidationError, ValueError, UnicodeDecodeError) as exc:
            st.error(f"Artifact отклонён: {exc}")
            st.session_state["calibration_artifact_bytes"] = None

        st.download_button(
            "Скачать шаблон XLSX",
            data=template_bytes(),
            file_name="galit_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Лист «Данные» — пример заполнения, лист «Инструкция» — все колонки",
        )

        if st.button("Демо-фонд (40 скважин)", key="btn_demo",
                     help="Синтетический фонд для знакомства с дашбордом"):
            st.session_state["demo"] = True

        st.divider()
        with st.expander("Методика расчёта"):
            st.markdown(
                """
                **Профили в стволе**
                · T(z) — аналитическое решение Ramey (1962)
                · P(z) — многофазный маршевой расчёт Beggs & Brill (1973)

                **Механизмы осложнений**
                · галит — индекс насыщения по Potter et al. (1977)
                · кальцит — индекс Stiff-Davis (ASTM D4582)
                · АСПО — пересечение T(z) с кривой WAT(P)
                · CO₂-коррозия — de Waard & Milliams (1995)

                **Интегральный риск** — взвешенная сумма тяжести механизмов
                на единой шкале 0–1 с учётом их взаимодействия.
                """
            )
        production_mode = st.toggle(
            "Промышленный режим",
            value=False,
            help="Включайте только для промысловых данных: строки с critical default/synthetic inputs будут заблокированы.",
        )
        if upload is not None and not production_mode:
            st.warning(
                "SCREENING · NOT FIELD VALIDATED: промышленный режим выключен. "
                "Результаты доступны для проверки, action recommendation заблокирована."
            )
        elif upload is not None:
            st.caption("Промышленный контроль включён: неполные строки не будут ранжироваться.")
        include_uncertainty = st.toggle(
            "Сценарные интервалы неопределённости",
            help="Воспроизводимый sensitivity ensemble; не калиброванный confidence interval",
        )
        st.caption(f"ГАЛИТ v{galit.__version__} · расчёт выполняется локально")
    return upload, production_mode, include_uncertainty


def result_labels(results: list[DiagnosisResult]) -> list[tuple[str, DiagnosisResult]]:
    """Пары (уникальная подпись, результат) в порядке убывания риска.

    При совпадении имён скважин подписи получают суффикс « (2)», « (3)»…
    """
    pairs: list[tuple[str, DiagnosisResult]] = []
    used: set[str] = set()
    for r in sorted(results, key=lambda x: x.integrated_risk, reverse=True):
        label = r.well
        i = 2
        while label in used:
            label = f"{r.well} ({i})"
            i += 1
        used.add(label)
        pairs.append((label, r))
    return pairs


def unique_labels(results: list[DiagnosisResult]) -> list[str]:
    return [label for label, _ in result_labels(results)]


def main() -> None:
    upload, production_mode, include_uncertainty = render_sidebar()
    render_header()

    # --- источник данных ---
    results: list[DiagnosisResult] | None = None
    errors: list[str] = []
    cases_by_name: dict[str, WellCase] = {}
    source_is_demo = False

    if upload is not None:
        df = read_table(upload.getvalue(), upload.name)
        parsed_cases, _ = frame_to_cases(df)
        cases_by_name = {case.name: case for case in parsed_cases}
        if df.empty:
            st.error("Файл не содержит данных.")
        else:
            results, errors = diagnose_frame(
                df,
                production_mode=production_mode,
                include_uncertainty=include_uncertainty,
                artifact_json=st.session_state.get("calibration_artifact_bytes"),
            )
            if not results:
                _, message = empty_results_status(len(parsed_cases), production_mode)
                st.error(message)
    elif st.session_state.get("demo"):
        source_is_demo = True
        demo_cases = galit.synthetic.make_fund(40)
        cases_by_name = {case.name: case for case in demo_cases}
        results = demo_fund()

    if results is None or not results:
        render_file_remarks(errors)
        render_welcome()
        return

    # --- постоянная маркировка назначения результата ---
    all_ready = all(r.quality.production_ready for r in results)
    if source_is_demo:
        st.error(
            "SCREENING · SYNTHETIC · ILLUSTRATIVE · NOT FIELD VALIDATED — "
            "демонстрационные результаты не являются промышленным прогнозом."
        )
    elif not all_ready:
        st.error(
            "SCREENING · NOT FIELD VALIDATED — в фонде есть критичные значения "
            "по умолчанию/синтетические входы; action recommendation не разрешена."
        )
    else:
        st.success("PRODUCTION-READY по критериям качества входов ядра GALIT.")

    # --- ключевые показатели фонда ---
    risks = [r.integrated_risk for r in results]
    crit = sum(1 for r in risks if r >= RISK_CRIT)
    elevated = sum(1 for r in risks if RISK_WARN <= r < RISK_CRIT)
    dominant_counts = pd.Series([MECH_RU[r.dominant] for r in results]).value_counts()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Скважин в расчёте", len(results))
    kpi2.metric("Критический риск", crit,
                help=f"Интегральный риск ≥ {RISK_CRIT:.2f}")
    kpi3.metric("Повышенный риск", elevated,
                help=f"Риск от {RISK_WARN:.2f} до {RISK_CRIT:.2f}")
    kpi4.metric("Средний риск фонда", f"{sum(risks) / len(risks):.2f}",
                help=f"Чаще всего лидирует: {dominant_counts.index[0]}")

    st.divider()
    tab_rank, tab_profiles, tab_well, tab_pilot = st.tabs(
        ["Ранжирование фонда", "Профили T(z) · P(z)", "Детально по скважине",
         "Сравнение с baseline / Пилот"]
    )

    # --- вкладка 1: рейтинг ---
    with tab_rank:
        st.markdown(
            f'<span class="legend-chip"><span class="legend-dot" '
            f'style="background:{STATUS_OK}"></span>норма &lt; {RISK_WARN:.2f}</span>'
            f'<span class="legend-chip"><span class="legend-dot" '
            f'style="background:{STATUS_WARN}"></span>повышенный {RISK_WARN:.2f}–{RISK_CRIT:.2f}</span>'
            f'<span class="legend-chip"><span class="legend-dot" '
            f'style="background:{STATUS_CRIT}"></span>критический ≥ {RISK_CRIT:.2f}</span>',
            unsafe_allow_html=True,
        )
        st.dataframe(style_rank(rank_frame(results)), width="stretch",
                     height=32 * (len(results) + 1))
        render_file_remarks(errors)

    # --- вкладка 2: профили ---
    with tab_profiles:
        labels = unique_labels(results)
        top = labels[:5]
        chosen = st.multiselect(
            "Скважины для сравнения (до 6)", labels, default=top,
            max_selections=6, key="ms_profiles",
        )
        if chosen:
            st.plotly_chart(fig_profiles(results, chosen), width="stretch",
                            config={"displaylogo": False,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        else:
            st.info("Выберите хотя бы одну скважину.")

    # --- вкладка 3: детальный разбор ---
    with tab_well:
        pairs = dict(result_labels(results))
        label = st.selectbox("Скважина", list(pairs), key="sb_well")
        detail = pairs[label]

        _, fg, status = risk_status(detail.integrated_risk)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Интегральный риск", f"{detail.integrated_risk:.2f} · {status}")
        m2.metric("Доминирующий механизм", MECH_RU.get(detail.dominant, detail.dominant))
        m3.metric("Начало АСПО, м",
                  "—" if detail.wax_onset_m is None else f"{detail.wax_onset_m:.0f}")
        m4.metric("Скорость коррозии", f"{detail.corrosion['rate_mm_yr']:.2f} мм/год")

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(fig_profiles(results, [label], detail=detail),
                            width="stretch",
                            config={"displaylogo": False,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        with right:
            st.markdown('<span class="section-title">Тяжесть механизмов</span>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_severity(detail), width="stretch",
                            config={"displaylogo": False, "displayModeBar": False})

        safe, safety_note = action_is_safe(detail)
        if safe:
            st.markdown(
                f'<div class="note-box"><b>Рекомендация:</b> {detail.recommendation}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error(f"SCREENING — {safety_note}")
            st.markdown(f"**Предварительная рекомендация, не к исполнению:** {detail.recommendation}")

        st.markdown("### Почему такое решение")
        st.caption(f"Политика риска: {detail.policy_id}, версия {detail.policy_version}")
        st.dataframe(
            contribution_frame(detail).style.format({
                "Тяжесть": "{:.3f}", "Вес политики": "{:.3f}",
                "Вклад в риск": "{:.3f}", "Доля integrated risk": "{:.1%}",
            }),
            width="stretch", hide_index=True,
        )

        trace = decision_trace(detail)
        st.markdown("### Decision trace")
        st.markdown(
            f"- **Доминирующий механизм:** {trace['dominant']}\n"
            f"- **Правило/причина рекомендации:** {trace['reason']}\n"
            f"- **Конфликт технологий:** {trace['conflict']}\n"
            f"- **Измерить следующим:** {', '.join(trace['measure_next'])}"
        )

        provenance = provenance_groups(detail)
        st.markdown("### Provenance входных данных")
        st.dataframe(pd.DataFrame(provenance["rows"])[
            ["Поле", "Происхождение", "Техническое поле"]
        ], width="stretch", hide_index=True)
        if provenance["critical"]:
            st.warning("Критичные missing/default/synthetic поля:")
            for group, fields in provenance["critical"].items():
                st.markdown(f"- **{group}:** {', '.join(fields)}")

        if detail.dominant == "corrosion" and detail.well in cases_by_name:
            st.markdown("### Counterfactual профилактики")
            efficiency = st.slider(
                "Сценарная эффективность ингибитора, %", 0, 100, 90, 5,
                key=f"cf_{label}",
            ) / 100.0
            scenario = corrosion_counterfactual(cases_by_name[detail.well], efficiency)
            if scenario["supported"]:
                before, after = scenario["before"], scenario["after"]
                st.warning("СЦЕНАРИЙ ЧУВСТВИТЕЛЬНОСТИ, НЕ ПРОГНОЗ")
                st.dataframe(pd.DataFrame([
                    {"Показатель": "Интегральный риск", "До": before.integrated_risk, "После": after.integrated_risk},
                    {"Показатель": "Тяжесть коррозии", "До": before.severity["corrosion"], "После": after.severity["corrosion"]},
                    {"Показатель": "Скорость коррозии, мм/год", "До": before.corrosion["rate_mm_yr"], "После": after.corrosion["rate_mm_yr"]},
                ]).style.format({"До": "{:.3f}", "После": "{:.3f}"}),
                width="stretch", hide_index=True)
            else:
                st.info("Сценарий не рассчитан: " + scenario["reason"])
        else:
            st.caption("Counterfactual доступен только для поддерживаемой физики ингибирования CO₂-коррозии.")

        if detail.warnings:
            st.markdown("### Applicability и limitations")
            for category, warnings in categorize_warnings(detail.warnings).items():
                st.markdown(f"**{category}**")
                for warning in warnings:
                    st.markdown(f"- {warning}")
            with st.expander("Оригинальные технические warnings"):
                for warning in detail.warnings:
                    st.code(warning, language=None)

    # --- вкладка 4: outcomes-based pilot, отдельно от recommendations ---
    with tab_pilot:
        st.warning("SHADOW MODE · эта вкладка не создаёт production recommendation.")
        st.markdown(
            "Сравниваются calendar/fixed schedule, independent-mechanism threshold "
            "и GALIT integrated ranking. Без фактического `event_outcome` оценка блокируется."
        )
        st.download_button(
            "Скачать шаблон пилота XLSX", pilot_template_bytes(),
            file_name="galit_pilot_contract.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        outcomes_upload = st.file_uploader(
            "Outcomes пилота (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="pilot_outcomes"
        )
        if outcomes_upload is None:
            evaluation = evaluate_uploaded_rows([])
        else:
            outcomes = read_table(outcomes_upload.getvalue(), outcomes_upload.name)
            evaluation = evaluate_uploaded_rows(outcomes.to_dict(orient="records"),
                                                k=min(5, max(1, len(outcomes))))
        if evaluation["status"] == "blocked":
            st.error("BLOCKED — " + evaluation["reason"])
        else:
            st.success("Outcomes supplied · holdout comparison")
            st.dataframe(pilot_evaluation_frame(evaluation), width="stretch", hide_index=True)
        st.markdown("**Labels:** " + ", ".join(evaluation["labels"]))
        st.markdown("**Assumptions:**")
        st.json(evaluation["assumptions"], expanded=False)
        st.markdown("**Split summary:**")
        st.json(evaluation["split_summary"], expanded=False)
        with st.expander("Data contract"):
            st.dataframe(pd.DataFrame(pilot_contract_frame()), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
