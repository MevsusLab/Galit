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
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pandas.io.formats.style import Styler

import galit
import galit.synthetic
from galit import (
    DiagnosisResult,
    FluidProperties,
    ProductionRate,
    ThermalParams,
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
        name_raw = row[by_canon["name"]]
        label = str(name_raw).strip() if pd.notna(name_raw) else ""
        label = label or f"строка {i}"
        vals: dict[str, float | str] = {}
        ok = True

        for canon, default in OPTIONAL_DEFAULTS.items():
            orig = by_canon.get(canon)
            if orig is None or pd.isna(row[orig]):
                vals[canon] = default
                continue
            value = pd.to_numeric(row[orig], errors="coerce") \
                if isinstance(default, float) else row[orig]
            if isinstance(default, float) and pd.isna(value):
                errors.append(
                    f"«{label}»: колонка {canon} — не число ({row[orig]!r}), "
                    f"принято значение по умолчанию {default}"
                )
                vals[canon] = default
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
        if not ions:
            ions = dict(TYPICAL_BRINE)

        try:
            cases.append(_row_to_case(label, vals, ions))
        except (ValueError, KeyError) as exc:
            errors.append(f"«{label}»: {exc} — строка пропущена")

    return cases, errors


def _row_to_case(label: str, vals: dict[str, float | str],
                 ions: dict[str, float]) -> WellCase:
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


def _normalize_csv_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Запятая как десятичный разделитель в числовых колонках CSV.

    Файлы, сохранённые из русского Excel, содержат «0,062» вместо «0.062».
    Текстовые колонки (названия скважин) остаются нетронутыми.
    """
    for col in df.columns:
        # pandas 2 даёт строкам object, pandas 3 -- специализированный str
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            if converted.notna().any():
                df[col] = converted
    return df


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


@st.cache_data(show_spinner="Расчёт фонда…")
def diagnose_frame(df: pd.DataFrame) -> tuple[list[DiagnosisResult], list[str]]:
    cases, errors = frame_to_cases(df)
    return [diagnose(c) for c in cases], errors


@st.cache_data(show_spinner=False)
def demo_fund() -> list[DiagnosisResult]:
    return [diagnose(c) for c in galit.synthetic.make_fund(40)]


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
        rows.append({
            "№": len(rows) + 1,
            "Скважина": r.well,
            "Риск": r.integrated_risk,
            "Статус": risk_status(r.integrated_risk)[2],
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
    """Боковая панель; возвращает загруженный файл (или None)."""
    upload = None
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
        st.caption(f"ГАЛИТ v{galit.__version__} · расчёт выполняется локально")
    return upload


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
    upload = render_sidebar()
    render_header()

    # --- источник данных ---
    results: list[DiagnosisResult] | None = None
    errors: list[str] = []

    if upload is not None:
        df = read_table(upload.getvalue(), upload.name)
        if df.empty:
            st.error("Файл не содержит данных.")
        else:
            results, errors = diagnose_frame(df)
            if not results:
                st.error("Ни одна строка не распознана. Проверьте структуру файла "
                         "по шаблону из боковой панели.")
                results = None
    elif st.session_state.get("demo"):
        results = demo_fund()

    if results is None or not results:
        render_welcome()
        return

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
    tab_rank, tab_profiles, tab_well = st.tabs(
        ["Ранжирование фонда", "Профили T(z) · P(z)", "Детально по скважине"]
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
        if errors:
            with st.expander(f"Замечания при разборе файла ({len(errors)})"):
                for e in errors:
                    st.markdown(f"- {e}")

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

        st.markdown(
            f'<div class="note-box"><b>Рекомендация:</b> {detail.recommendation}</div>',
            unsafe_allow_html=True,
        )
        if detail.warnings:
            st.markdown('<span class="section-title">Предупреждения расчёта</span>',
                        unsafe_allow_html=True)
            for w in detail.warnings:
                st.markdown(f"- {w}")


if __name__ == "__main__":
    main()
