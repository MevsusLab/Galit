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

import base64
import io
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
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
    DiagnosedWell,
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
    generate_master_plan,
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

# Индикация статуса: норма / повышенный / критический.
# Тёплая шкала (янтарь -> красный) взята с референсного макета; каждая пара
# «подложка + акцент» проверена на WCAG в tests/test_dashboard.py.
STATUS_OK_BG = "#E9F4EC"
STATUS_OK = "#246B2A"
STATUS_WARN_BG = "#FDF3E0"
STATUS_WARN = "#9A4A00"
STATUS_CRIT_BG = "#FCE9E6"
STATUS_CRIT = "#B02020"

# Промежуточный «высокий» тон шкалы: чипы и левые акценты карточек.
STATUS_HIGH_BG = "#FDEEE6"
STATUS_HIGH = "#C2410C"

RISK_WARN = 0.35   # ниже -- норма
RISK_CRIT = 0.60   # выше -- критический

# Нейтральная основа поверхностей: почти белый лист, как на референсе.
CANVAS = "#FFFFFF"          # плоскость карточек
HAIRLINE = "#E7EBE8"        # тонкая линия вместо рамок и сеток
SHADOW_SOFT = "0 1px 2px rgba(16, 40, 30, 0.04), 0 8px 24px rgba(16, 40, 30, 0.07)"
SHADOW_RAISED = "0 2px 4px rgba(16, 40, 30, 0.05), 0 14px 34px rgba(16, 40, 30, 0.10)"
RADIUS = "14px"             # крупный радиус скругления с макета

# Последовательность цветов линий профилей (зелёная гамма + графитовый)
LINE_COLORS = ["#0F6B43", "#0B4A2F", "#3D8B66", "#6FA98C", "#275D57", "#5A6560"]

MECH_RU = {
    "halite": "Галит",
    "calcite": "Кальцит",
    "wax": "АСПО",
    "corrosion": "Коррозия",
}

FONT_FAMILY = "Segoe UI, Helvetica Neue, Arial, sans-serif"

PROJECT_ROOT = Path(__file__).resolve().parent
BACKGROUND_ASSET = PROJECT_ROOT / "assets" / "dashboard-background.png"
HEADER_LOGO_ASSET = PROJECT_ROOT / "assets" / "header-logo.png"


def local_image_data_uri(path: Path) -> str:
    """Return a data URI for a local image, independent of the process CWD."""
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    mime_type = mime_types.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime_type};base64,{encoded}"


def background_data_uri(path: Path = BACKGROUND_ASSET) -> str:
    """Return the repository-local dashboard background as a data URI."""
    return local_image_data_uri(path)


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
    """Apply the corporate theme over a repository-local image background."""
    background_uri = background_data_uri()
    background_layer = (
        f"url('{background_uri}') center / cover fixed no-repeat"
        if background_uri else "linear-gradient(135deg, #DCE9E2, #F6F8F7)"
    )
    st.markdown(
        f"""
        <style>
        /* ---------- базовая типографика ---------- */
        .stApp {{
            font-family: {FONT_FAMILY};
            color: {INK};
            /* Референс — почти белый лист: фото остаётся текстурой подложки,
               поверх него плотная белая вуаль, чтобы карточки читались. */
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.90) 0%, rgba(252, 253, 252, 0.94) 100%),
                {background_layer};
            background-attachment: fixed;
            background-position: center;
            background-size: cover;
        }}
        .stApp [data-testid="stMain"] {{ background: transparent; }}
        .stApp [data-testid="stMainBlockContainer"] {{
            background: transparent;
            border: 0;
            border-radius: 0;
            box-shadow: none;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
            margin-top: 1rem;
            margin-bottom: 2rem;
            padding-left: clamp(1rem, 3vw, 3rem);
            padding-right: clamp(1rem, 3vw, 3rem);
        }}
        p, li, span, label {{ color: {INK}; }}
        h1, h2, h3 {{
            color: {INK};
            letter-spacing: -0.2px;
        }}

        /* строгий режим: без фирменной радужной полосы и колонтитула */
        div[data-testid="stDecoration"] {{ display: none; }}
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.92);
            border-bottom: 1px solid {HAIRLINE};
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }}

        /* ---------- шапка приложения ---------- */
        .app-header {{
            position: relative;
            display: flex;
            align-items: center;
            min-height: 88px;
            background: {CANVAS};
            border: 1px solid {HAIRLINE};
            border-radius: {RADIUS};
            box-shadow: {SHADOW_SOFT};
            padding: 18px 24px;
            margin-bottom: 18px;
        }}
        .app-header-text {{
            position: relative;
            z-index: 1;
            flex: 1 1 auto;
            min-width: 0;
        }}
        .app-title {{
            font-size: 30px; font-weight: 750; color: {GREEN_900};
            letter-spacing: -0.4px; line-height: 1.15;
        }}
        .app-subtitle {{
            font-size: 14px; color: {INK_MUTED}; margin-top: 4px;
        }}
        @media (max-width: 900px) {{
            .app-header {{ min-height: 0; }}
        }}
        .app-rule {{
            border: none; border-top: 1px solid {BORDER}; margin: 14px 0 20px 0;
        }}

        /* ---------- боковая панель ---------- */
        section[data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.94);
            border-right: 1px solid {HAIRLINE};
            box-shadow: none;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}
        section[data-testid="stSidebar"] > div {{ background: transparent; }}
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            padding-top: 0;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            transform: translateY(-68px);
        }}
        .sidebar-logo {{
            display: block;
            width: min(100%, 220px);
            height: auto;
            margin: 0 auto 0 0;
            object-fit: contain;
        }}
        .sidebar-logo + div[data-testid="stDivider"] {{
            margin-top: 10px;
            margin-bottom: 12px;
        }}

        /* Streamlit/BaseWeb controls: explicit light surfaces survive host dark mode. */
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: {INK_MUTED} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p {{
            color: {INK} !important; font-weight: 650;
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: {SURFACE}; border: 1px dashed {GREEN_700};
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
            color: {INK} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {{
            color: {INK_MUTED} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
            background: #FFFFFF; color: {GREEN_900}; border: 1px solid {GREEN_700};
            font-weight: 650;
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] {{
            background: #FFFFFF !important; color: {INK} !important;
            border-color: {BORDER} !important;
        }}
        div[data-baseweb="select"] *, input, textarea {{ color: {INK} !important; }}
        div[role="listbox"], div[role="option"] {{
            background: #FFFFFF !important; color: {INK} !important;
        }}
        div[data-testid="stExpander"] {{
            background: rgba(255, 255, 255, 0.80); color: {INK};
            border-color: rgba(214, 222, 217, 0.82);
            box-shadow: 0 6px 18px rgba(9, 47, 32, 0.08);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }}
        div[data-testid="stExpander"] summary *,
        div[data-testid="stCheckbox"] label *,
        [data-testid="stTooltipIcon"] {{ color: {INK} !important; }}

        /* ---------- кнопки ---------- */
        .stButton > button {{
            background: {GREEN_700}; color: #FFFFFF; border: none;
            border-radius: 10px; font-weight: 650; font-size: 14px;
            padding: 9px 18px; width: 100%;
            box-shadow: 0 1px 2px rgba(16, 40, 30, 0.10);
            transition: background 0.15s ease, transform 0.15s ease;
        }}
        .stButton > button:hover {{
            background: {GREEN_900}; color: #FFFFFF; transform: translateY(-1px);
        }}
        .stButton > button:focus {{
            box-shadow: 0 0 0 3px {GREEN_100}; color: #FFFFFF;
        }}
        .stDownloadButton > button {{
            background: #FFFFFF; color: {GREEN_700};
            border: 1px solid {GREEN_700}; border-radius: 10px;
            font-weight: 650; font-size: 14px; padding: 8px 18px; width: 100%;
        }}
        .stDownloadButton > button:hover {{
            background: {GREEN_100}; color: {GREEN_900};
            border: 1px solid {GREEN_900};
        }}

        /* ---------- вкладки ---------- */
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 6px;
            background: {SURFACE};
            border: 1px solid {HAIRLINE};
            border-radius: 12px;
            padding: 6px;
        }}
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            background: transparent; color: {INK_MUTED};
            font-weight: 650; font-size: 14px;
            border-radius: 9px; padding: 8px 18px; border-bottom: none;
        }}
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
            background: {CANVAS}; color: {GREEN_900};
            border-bottom: none;
            box-shadow: 0 1px 2px rgba(16, 40, 30, 0.10);
        }}
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

        /* ---------- карточки-метрики ---------- */
        div[data-testid="stMetric"] {{
            background: {CANVAS};
            border: 1px solid {HAIRLINE};
            border-radius: {RADIUS}; padding: 18px 20px 14px 20px;
            box-shadow: {SHADOW_SOFT};
            transition: box-shadow 0.18s ease, transform 0.18s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            box-shadow: {SHADOW_RAISED};
            transform: translateY(-1px);
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {INK_MUTED}; font-size: 11px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.8px;
            margin-bottom: 6px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {INK}; font-weight: 750; letter-spacing: -0.6px;
        }}

        /* ---------- таблицы, canvas-grid и графики ---------- */
        div[data-testid="stDataFrame"] {{
            background: {CANVAS}; color: {INK};
            border: 1px solid {HAIRLINE}; border-radius: {RADIUS};
            box-shadow: {SHADOW_SOFT};
            overflow: hidden;
        }}
        div[data-testid="stDataFrame"] *,
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {{
            color: {INK} !important;
        }}
        div[data-testid="stDataFrame"] button {{
            background: #FFFFFF; color: {INK} !important;
        }}
        div[data-testid="stDataFrame"] canvas {{ background: #FFFFFF; }}
        /* Референс без внутренней сетки: только горизонтальные волосяные линии. */
        table {{ border-collapse: collapse; }}
        table, th, td {{ color: {INK}; border-color: {HAIRLINE}; }}
        td {{ border-left: none; border-right: none; }}
        th {{
            background: {SURFACE}; font-weight: 700;
            border-left: none; border-right: none;
            font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px;
            color: {INK_MUTED};
        }}
        .js-plotly-plot .plotly .modebar {{ background: transparent; }}

        /* Alerts own their pale surfaces; force readable foreground in every variant. */
        div[data-testid="stAlert"] {{ color: {INK}; }}
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] li,
        div[data-testid="stAlert"] span {{ color: {INK} !important; }}

        /* ---------- информационные блоки ---------- */
        .note-box {{
            background: {CANVAS};
            border: 1px solid {HAIRLINE};
            border-left: 4px solid {GREEN_700};
            border-radius: {RADIUS}; padding: 16px 20px; font-size: 14px;
            box-shadow: {SHADOW_SOFT};
        }}
        .legend-chip {{
            display: inline-block; margin-right: 22px;
            font-size: 13px; color: {INK};
        }}
        .legend-dot {{
            display: inline-block; width: 10px; height: 10px;
            border-radius: 3px; margin-right: 7px; vertical-align: -1px;
        }}
        .section-title {{
            font-size: 17px; font-weight: 750; color: {INK};
            letter-spacing: -0.2px; margin: 6px 0 12px 0;
        }}

        /* ---------- статус-чипы тёплой шкалы (как на референсе) ---------- */
        .status-chip {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 999px;
            font-size: 12px; font-weight: 700; letter-spacing: 0.2px;
            border: 1px solid transparent;
        }}
        .status-chip.is-ok {{ background: {STATUS_OK_BG}; color: {STATUS_OK}; }}
        .status-chip.is-warn {{ background: {STATUS_WARN_BG}; color: {STATUS_WARN}; }}
        .status-chip.is-high {{ background: {STATUS_HIGH_BG}; color: {STATUS_HIGH}; }}
        .status-chip.is-crit {{ background: {STATUS_CRIT_BG}; color: {STATUS_CRIT}; }}

        /* Мягкая карточка-контейнер для блоков плана и прогноза. */
        .surface-card {{
            background: {CANVAS};
            border: 1px solid {HAIRLINE};
            border-radius: {RADIUS};
            box-shadow: {SHADOW_SOFT};
            padding: 18px 20px;
        }}

        /* ---------- shell: navigation / overview / alerts ---------- */
        .shell-eyebrow {{
            color: {GREEN_700}; font-size: 11px; font-weight: 800;
            letter-spacing: 1.2px; text-transform: uppercase;
            margin: 0 0 5px 0;
        }}
        .shell-heading {{
            color: {INK}; font-size: 22px; font-weight: 760;
            letter-spacing: -0.35px; margin: 0 0 4px 0;
        }}
        .shell-copy {{ color: {INK_MUTED}; font-size: 13px; margin-bottom: 14px; }}
        .sidebar-nav-title {{
            color: {INK_MUTED}; font-size: 10px; font-weight: 800;
            letter-spacing: 1.1px; text-transform: uppercase; margin: 4px 0 8px;
        }}
        .sidebar-nav {{ display: grid; gap: 3px; margin: 0 0 14px; }}
        .sidebar-nav-item {{
            display: flex; align-items: center; gap: 9px;
            min-height: 34px; padding: 7px 9px; border-radius: 9px;
            color: {INK_MUTED}; font-size: 12px; font-weight: 650;
        }}
        .sidebar-nav-item:first-child {{ background: {GREEN_100}; color: {GREEN_900}; }}
        .sidebar-nav-index {{
            display: inline-grid; place-items: center; width: 20px; height: 20px;
            border-radius: 6px; background: #FFFFFF; color: {GREEN_700};
            font-size: 10px; font-weight: 800;
        }}
        .alerts-rail {{
            border-left: 1px solid {HAIRLINE}; padding-left: 14px;
            min-height: 100%;
        }}
        .alert-card {{
            background: {CANVAS}; border: 1px solid {HAIRLINE};
            border-left: 3px solid var(--alert-accent, {STATUS_WARN});
            border-radius: 11px; padding: 11px 12px; margin-bottom: 9px;
            box-shadow: {SHADOW_SOFT}; overflow-wrap: anywhere;
        }}
        .alert-card.is-critical {{ --alert-accent: {STATUS_CRIT}; }}
        .alert-card.is-warning {{ --alert-accent: {STATUS_WARN}; }}
        .alert-card.is-ok {{ --alert-accent: {STATUS_OK}; }}
        .alert-card-title {{ color: {INK}; font-size: 13px; font-weight: 750; }}
        .alert-card-meta {{ color: {INK_MUTED}; font-size: 11px; margin-top: 3px; }}
        .overview-table {{ width: 100%; font-size: 12px; }}
        .overview-table td {{ padding: 8px 6px; border-bottom: 1px solid {HAIRLINE}; }}
        .overview-table td:last-child {{ text-align: right; font-weight: 750; }}

        @media (max-width: 1100px) {{
            .alerts-rail {{ border-left: 0; border-top: 1px solid {HAIRLINE};
                            padding: 14px 0 0; margin-top: 8px; }}
            .app-title {{ font-size: 26px; }}
        }}
        @media (max-width: 720px) {{
            .stApp [data-testid="stMainBlockContainer"] {{ padding-left: 12px; padding-right: 12px; }}
            .app-header {{ padding: 14px 16px; border-radius: 12px; }}
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
                overflow-x: auto; flex-wrap: nowrap; justify-content: flex-start;
                scrollbar-width: thin;
            }}
            div[data-testid="stTabs"] button[data-baseweb="tab"] {{
                flex: 0 0 auto; padding: 7px 12px; font-size: 12px;
            }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ scroll-behavior: auto !important;
                transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }}
        }}
        div[data-testid="stExpander"] {{ border-radius: {RADIUS}; }}
        div[data-testid="stAlert"] {{ border-radius: 12px; }}
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


def build_master_plan(cases_by_name: dict[str, WellCase], results: list[DiagnosisResult]):
    """Связать result→case по нормализованному имени без молчаливой потери строк."""
    normalised_cases: dict[str, WellCase] = {}
    for case in cases_by_name.values():
        key = " ".join(case.name.strip().casefold().split())
        if key in normalised_cases:
            raise ValueError(f"Дублирующееся имя скважины в исходных данных: {case.name}")
        normalised_cases[key] = case

    diagnosed: list[DiagnosedWell] = []
    for result in results:
        key = " ".join(result.well.strip().casefold().split())
        case = normalised_cases.get(key)
        if case is None:
            raise ValueError(f"Для результата не найден исходный кейс: {result.well}")
        diagnosed.append(DiagnosedWell(case, result))
    return generate_master_plan(diagnosed)

def master_plan_frame(plan: Any) -> pd.DataFrame:
    """Русская плоская таблица плана для UI и CSV без дублирования правил."""
    return pd.DataFrame([{
        "Скважина": task.well,
        "Осложнение": task.dominant_label,
        "Риск": task.risk,
        "Возможная потеря, м³/сут": task.possible_oil_loss.central_m3d,
        "Срок": task.response_deadline,
        "Действие": task.recommended_action,
        "Safe-to-act": "да" if task.safe_to_act else "нет — нужна верификация",
    } for task in plan.tasks])


def master_plan_csv(plan: Any) -> bytes:
    return master_plan_frame(plan).to_csv(index=False).encode("utf-8-sig")


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
        styles = [f"background-color: {bg}; color: {INK}" for _ in df.columns]
        risk_i = df.columns.get_loc("Риск")
        styles[risk_i] = f"background-color: {fg}; color: #FFFFFF; font-weight: 700"
        leader_i = df.columns.get_loc("Лидер")
        styles[leader_i] = f"{styles[leader_i]}; color: {GREEN_900}; font-weight: 600"
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
    fig.update_xaxes(gridcolor=HAIRLINE, zeroline=False, linecolor=HAIRLINE)
    fig.update_yaxes(
        gridcolor=HAIRLINE, zeroline=False, linecolor=HAIRLINE,
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


FORECAST_HISTORY_COLUMNS = (
    "well", "timestamp", "wax_severity", "halite_severity", "calcite_severity",
    "corrosion_wall_loss_mm", "oil_rate_m3_day", "quality", "source",
)


def forecast_history_from_csv(data: bytes) -> galit.ForecastHistory:
    """Parse documented temporal CSV; timestamps must carry an explicit timezone."""
    frame = pd.read_csv(io.BytesIO(data))
    missing = {"well", "timestamp"} - set(frame.columns)
    if missing:
        raise ValueError("Forecast history CSV: missing columns " + ", ".join(sorted(missing)))
    snapshots = []
    numeric = set(FORECAST_HISTORY_COLUMNS[2:7])
    for index, row in frame.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None:
            raise ValueError(f"Forecast history row {index + 2}: timestamp must include timezone")
        values: dict[str, Any] = {"well": str(row["well"]), "timestamp": timestamp.to_pydatetime()}
        for column in numeric:
            if column in frame.columns and pd.notna(row.get(column)):
                values[column] = float(row[column])
        for column, default in (("quality", "good"), ("source", "measured")):
            values[column] = str(row[column]) if column in frame.columns and pd.notna(row[column]) else default
        snapshots.append(galit.ForecastSnapshot(**values))
    return galit.ForecastHistory(tuple(snapshots))


def fig_forecast_timeline(forecast: galit.WellForecast) -> go.Figure:
    """Render only events whose contract actually supplies a dated window."""
    dated = [event for event in forecast.events if event.horizon_start_date is not None]
    fig = go.Figure()
    for event in dated:
        start = event.horizon_start_date
        end = event.horizon_end_date or start
        fig.add_trace(go.Bar(
            x=[max((end - start).days, 0) + 1], y=[event.title], base=[start],
            orientation="h", name=event.status.value, showlegend=False,
            marker_color=GREEN_700 if event.status.value == "calibrated" else STATUS_WARN,
            customdata=[[start.isoformat(), end.isoformat(), event.status.value]],
            hovertemplate="%{y}<br>%{customdata[0]} — %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        ))
    fig.update_layout(height=max(220, 65 * len(dated)), barmode="overlay",
                      xaxis_title="Календарное окно", yaxis_title="",
                      font=dict(family=FONT_FAMILY, color=INK),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig


def forecast_event_frame(events: list[galit.ForecastEvent] | tuple[galit.ForecastEvent, ...]) -> pd.DataFrame:
    rows = []
    for event in events:
        window = "дата недоступна"
        if event.horizon_start_date is not None:
            window = (event.horizon_start_date.isoformat() if event.horizon_start_date == event.horizon_end_date
                      else f"{event.horizon_start_date.isoformat()} — {event.horizon_end_date.isoformat()}")
        band = "—" if event.risk_band is None else f"{event.risk_band[0]:.2f}–{event.risk_band[1]:.2f}"
        rows.append({"Событие": event.title, "Статус": event.status.value,
                     "Ожидаемое окно": window, "Likelihood": event.likelihood.value,
                     "Risk band": band, "Actionable": "да" if event.actionable else "нет",
                     "Basis": event.basis, "Assumptions": "; ".join(event.assumptions) or "—",
                     "Limitations": "; ".join(event.limitations) or "—",
                     "Required inputs": "; ".join(event.required_inputs) or "—"})
    return pd.DataFrame(rows)


def fig_fund_risk(results: list[DiagnosisResult]) -> go.Figure:
    """Compact overview chart built only from the current diagnosis results."""
    ordered = sorted(results, key=lambda item: item.integrated_risk, reverse=True)[:12]
    colors = [risk_status(item.integrated_risk)[1] for item in ordered]
    fig = go.Figure(go.Bar(
        x=[item.well for item in ordered],
        y=[item.integrated_risk for item in ordered],
        marker_color=colors,
        text=[f"{item.integrated_risk:.2f}" for item in ordered],
        textposition="outside",
        hovertemplate="%{x}<br>Интегральный риск: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=RISK_WARN, line_color=STATUS_WARN, line_dash="dot", line_width=1)
    fig.add_hline(y=RISK_CRIT, line_color=STATUS_CRIT, line_dash="dot", line_width=1)
    fig.update_layout(
        height=290, showlegend=False, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, color=INK), margin=dict(l=8, r=8, t=20, b=8),
        xaxis=dict(title="", tickangle=-28, gridcolor=HAIRLINE),
        yaxis=dict(title="Риск, 0–1", range=[0, 1.08], gridcolor=HAIRLINE, zeroline=False),
    )
    return fig


def fig_mechanism_mix(results: list[DiagnosisResult]) -> go.Figure:
    """Dominant-mechanism structure for the current fund, without inferred history."""
    counts = {mechanism: 0 for mechanism in MECH_RU}
    for result in results:
        counts[result.dominant] = counts.get(result.dominant, 0) + 1
    nonzero = [(mechanism, count) for mechanism, count in counts.items() if count]
    fig = go.Figure(go.Pie(
        labels=[MECH_RU.get(mechanism, mechanism) for mechanism, _ in nonzero],
        values=[count for _, count in nonzero], hole=0.68,
        marker_colors=[GREEN_700, STATUS_WARN, STATUS_HIGH, STATUS_CRIT][:len(nonzero)],
        textinfo="percent", hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        height=290, showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        annotations=[dict(text=str(len(results)), x=0.5, y=0.5, showarrow=False,
                          font=dict(size=22, color=INK, family=FONT_FAMILY))],
        paper_bgcolor="#FFFFFF", font=dict(family=FONT_FAMILY, color=INK),
        margin=dict(l=8, r=8, t=20, b=35),
    )
    return fig


def overview_alerts(results: list[DiagnosisResult]) -> list[dict[str, str]]:
    """Return current-state alerts; deliberately contains no invented timestamps."""
    alerts: list[dict[str, str]] = []
    for result in sorted(results, key=lambda item: item.integrated_risk, reverse=True):
        if result.integrated_risk >= RISK_CRIT:
            level = "critical"
        elif result.integrated_risk >= RISK_WARN:
            level = "warning"
        else:
            continue
        alerts.append({
            "level": level,
            "well": result.well,
            "title": f"{MECH_RU.get(result.dominant, result.dominant)} · риск {result.integrated_risk:.2f}",
            "quality": f"Качество {result.quality.grade} · "
                       f"{'действие разрешено' if action_is_safe(result)[0] else 'trust-gate блокирует действие'}",
        })
    return alerts


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
        """
        <div class="app-header">
            <div class="app-header-text">
                <div class="app-title">ГАЛИТ</div>
                <div class="app-subtitle">
                    Интегрированная диагностика осложнений: галит · кальцит · АСПО ·
                    коррозия. Ранжирование фонда и профили T(z), P(z).
                </div>
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
        logo_uri = local_image_data_uri(HEADER_LOGO_ASSET)
        if logo_uri:
            st.markdown(
                f'<img class="sidebar-logo" src="{logo_uri}" alt="Логотип ГАЛИТ">',
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown(
            """
            <div class="sidebar-nav-title">Рабочие разделы</div>
            <nav class="sidebar-nav" aria-label="Навигация по разделам">
                <div class="sidebar-nav-item"><span class="sidebar-nav-index">01</span>Обзор фонда</div>
                <div class="sidebar-nav-item"><span class="sidebar-nav-index">02</span>План мастера</div>
                <div class="sidebar-nav-item"><span class="sidebar-nav-index">03</span>Ранжирование фонда</div>
                <div class="sidebar-nav-item"><span class="sidebar-nav-index">04</span>Профили T(z) · P(z)</div>
                <div class="sidebar-nav-item"><span class="sidebar-nav-index">05</span>Детально по скважине</div>
                <div class="sidebar-nav-item"><span class="sidebar-nav-index">06</span>Прогноз и пилот</div>
            </nav>
            """,
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

        history_upload = st.file_uploader(
            "История наблюдений для прогноза (CSV, optional)", type=["csv"],
            key="forecast_history",
            help=("Schema: well,timestamp,wax_severity,halite_severity,calcite_severity,"
                  "corrosion_wall_loss_mm,oil_rate_m3_day,quality,source. "
                  "timestamp — ISO 8601 с timezone; пустые metrics разрешены."),
        )
        st.session_state["forecast_history_bytes"] = (
            history_upload.getvalue() if history_upload is not None else None
        )
        with st.expander("Коррозия: измеренный запас стенки (optional)"):
            use_wall = st.checkbox("Есть инструментальные замеры", value=False)
            if use_wall:
                st.number_input("Текущая толщина стенки, мм", min_value=0.0, value=0.0,
                                key="forecast_wall_current")
                st.number_input("Минимально допустимая толщина, мм", min_value=0.0, value=0.0,
                                key="forecast_wall_minimum")
            else:
                st.session_state["forecast_wall_current"] = None
                st.session_state["forecast_wall_minimum"] = None

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

    # --- доменный план мастера для отдельной вкладки ---
    plan = None
    plan_error: str | None = None
    try:
        plan = build_master_plan(cases_by_name, results)
    except Exception as exc:  # UI boundary: the other diagnostic tabs must remain usable.
        plan_error = str(exc)

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

    # --- overview shell: current snapshot only, no invented history/timestamps ---
    st.markdown(
        '<div class="shell-eyebrow">Оперативная картина</div>'
        '<div class="shell-heading">Обзор фонда</div>'
        '<div class="shell-copy">Риски, структура осложнений и текущие сигналы '
        'по загруженному расчёту.</div>',
        unsafe_allow_html=True,
    )
    overview_main, overview_alerts_col = st.columns([3, 1], gap="large")
    with overview_main:
        risk_col, mix_col = st.columns([3, 2])
        with risk_col:
            st.markdown('<span class="section-title">Риск по скважинам</span>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_fund_risk(results), width="stretch",
                            config={"displaylogo": False, "displayModeBar": False})
        with mix_col:
            st.markdown('<span class="section-title">Структура осложнений</span>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_mechanism_mix(results), width="stretch",
                            config={"displaylogo": False, "displayModeBar": False})
        st.markdown('<span class="section-title">Скважины с наибольшим риском</span>',
                    unsafe_allow_html=True)
        top_rows = "".join(
            f'<tr><td>{item.well}</td><td>{MECH_RU.get(item.dominant, item.dominant)}</td>'
            f'<td>{item.integrated_risk:.2f}</td></tr>'
            for item in sorted(results, key=lambda row: row.integrated_risk, reverse=True)[:5]
        )
        st.markdown(
            '<table class="overview-table"><thead><tr><th>Скважина</th>'
            '<th>Лидер</th><th>Риск</th></tr></thead><tbody>' + top_rows + '</tbody></table>',
            unsafe_allow_html=True,
        )
    with overview_alerts_col:
        st.markdown('<div class="alerts-rail"><div class="shell-eyebrow">Сигналы</div>'
                    '<div class="shell-heading">Alerts</div></div>', unsafe_allow_html=True)
        alerts = overview_alerts(results)
        if not alerts:
            st.markdown(
                '<div class="alert-card is-ok"><div class="alert-card-title">'
                'Активных сигналов нет</div><div class="alert-card-meta">'
                'В текущем расчёте нет скважин выше порога повышенного риска.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            for alert in alerts[:6]:
                st.markdown(
                    f'<div class="alert-card is-{alert["level"]}">'
                    f'<div class="alert-card-title">{alert["well"]}</div>'
                    f'<div class="alert-card-meta">{alert["title"]}<br>{alert["quality"]}</div>'
                    '</div>', unsafe_allow_html=True,
                )
        if len(alerts) > 6:
            st.caption(f"Ещё сигналов: {len(alerts) - 6}. Полный список — в ранжировании.")

    st.divider()
    tab_plan, tab_rank, tab_profiles, tab_well, tab_forecast, tab_pilot = st.tabs(
        ["План мастера", "Ранжирование фонда", "Профили T(z) · P(z)",
         "Детально по скважине", "Прогноз во времени", "Сравнение с baseline / Пилот"]
    )

    # --- первая вкладка: тот же уже сформированный доменный план ---
    with tab_plan:
        if plan is None:
            st.error(
                "План мастера не сформирован. Остальные результаты доступны; "
                "проверьте сопоставление исходных строк и повторите расчёт."
            )
            if plan_error:
                st.caption(f"Техническая причина: {plan_error}")
        else:
            loss = plan.summary.possible_oil_loss_central_m3d
            p1, p2, p3 = st.columns(3)
            p1.metric("Всего задач", plan.summary.task_count)
            p2.metric("Срочных", sum(task.response_deadline in {"немедленно", "24ч"} for task in plan.tasks))
            p3.metric("Потенциальная потеря", "—" if loss is None else f"{loss:.1f} м³/сут")
            st.caption(plan.advisory_notice)
            frame = master_plan_frame(plan)
            st.dataframe(frame, width="stretch", hide_index=True)
            st.download_button(
                "Скачать план CSV", master_plan_csv(plan), file_name="galit_master_plan.csv",
                mime="text/csv",
            )
        if plan is not None and plan.tasks:
            selected_well = st.selectbox(
                "Задание для подробного просмотра", [task.well for task in plan.tasks],
                key="master_plan_task",
            )
            task = next(item for item in plan.tasks if item.well == selected_well)
            if not task.safe_to_act:
                st.error("Действие заблокировано: сначала верифицируйте качество данных и повторите диагностику.")
            st.markdown(f"**Причины приоритета:** {'; '.join(task.priority_reasons)}")
            st.markdown(f"**Checklist:**  " + "  \n".join(f"- {item}" for item in task.pre_trip_checklist))
            st.markdown(f"**Материалы:** {', '.join(task.materials)}")
            st.markdown(f"**Оборудование:** {', '.join(task.equipment)}")
            if task.quality_warnings:
                st.warning("Предупреждения качества: " + "; ".join(task.quality_warnings))

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

    # --- прогноз: временной контракт отделён от snapshot-диагностики ---
    with tab_forecast:
        st.warning("SCREENING: временные окна — сценарии, не гарантированные даты или вероятности отказа.")
        forecast_well_name = st.selectbox("Скважина для прогноза", list(cases_by_name),
                                          key="forecast_well")
        case = cases_by_name[forecast_well_name]
        result = next(item for item in results if item.well == forecast_well_name)
        as_of = datetime.now(timezone.utc)
        history = None
        history_bytes = st.session_state.get("forecast_history_bytes")
        try:
            if history_bytes:
                parsed = forecast_history_from_csv(history_bytes)
                history = galit.ForecastHistory(tuple(
                    item for item in parsed.snapshots if item.well == forecast_well_name
                ))
            current = st.session_state.get("forecast_wall_current")
            minimum = st.session_state.get("forecast_wall_minimum")
            integrity = None
            if current is not None and minimum is not None:
                integrity = galit.CorrosionIntegrityInput(float(current), float(minimum), as_of)
            forecast = galit.forecast_well(result, case, history=history, as_of=as_of,
                                            corrosion_integrity=integrity)
            if source_is_demo:
                st.error("ДЕМО · СЦЕНАРНЫЙ ПРОГНОЗ · SYNTHETIC · NOT FIELD VALIDATED")
            if not history_bytes:
                st.info("Фонд использован только как snapshot. Временная история не загружена; недоступные окна не дорисовываются.")
            dated = [event for event in forecast.events if event.horizon_start_date is not None]
            undated = [event for event in forecast.events if event.horizon_start_date is None]
            if dated:
                st.plotly_chart(fig_forecast_timeline(forecast), width="stretch",
                                config={"displaylogo": False})
            else:
                st.info("Датированных событий нет: контракт вернул только screening/unavailable без временных окон.")
            st.markdown("### Все события и основания")
            st.dataframe(forecast_event_frame(forecast.events), width="stretch", hide_index=True)
            if undated:
                st.markdown("### Без даты — отдельно от timeline")
                st.dataframe(forecast_event_frame(undated), width="stretch", hide_index=True)
        except (ValueError, KeyError) as exc:
            st.error(f"Прогноз не рассчитан: {exc}")
        with st.expander("CSV schema истории"):
            st.code(",".join(FORECAST_HISTORY_COLUMNS), language=None)
            st.caption("timestamp: ISO 8601 с timezone; severity 0..1; wall loss/rate ≥ 0; quality good/questionable/bad; source measured/derived/laboratory.")

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
