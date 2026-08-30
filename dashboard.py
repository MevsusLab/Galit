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
import html
import io
import json
import math
import os
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

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
HAIRLINE = "#E7EBE8"        # legacy palette value for non-CSS compatibility
HAIRLINE_SOFT = "rgba(31, 36, 34, 0.06)"
HAIRLINE_FAINT = "rgba(31, 36, 34, 0.04)"
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

FONT_FAMILY = "Outfit, Manrope, Segoe UI, Arial, sans-serif"
WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
BACKGROUND_ASSET = ASSETS_DIR / "dashboard-background.png"
HEADER_LOGO_ASSET = ASSETS_DIR / "header-logo.png"
FONT_DIR = ASSETS_DIR / "fonts"
ICON_DIR = ASSETS_DIR / "icons"
FAVICON_ASSET = ICON_DIR / "brand-favicon.svg"
FONT_FACES = (
    ("Outfit", "outfit-latin-ext-wght-normal.woff2", "U+0100-02FF, U+1E00-1EFF, U+2000-206F"),
    ("Outfit", "outfit-latin-wght-normal.woff2", "U+0000-00FF"),
    ("Manrope", "manrope-cyrillic-ext-wght-normal.woff2", "U+0400-052F, U+2DE0-2DFF, U+A640-A69F"),
    ("Manrope", "manrope-cyrillic-wght-normal.woff2", "U+0400-04FF"),
)


def local_asset_data_uri(path: Path) -> str:
    """Return a data URI for a repository-local UI asset."""
    mime_types = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml", ".woff2": "font/woff2",
    }
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    mime_type = mime_types.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime_type};base64,{encoded}"


def local_image_data_uri(path: Path) -> str:
    """Backward-compatible image asset helper."""
    return local_asset_data_uri(path)


def font_face_css(font_dir: Path = FONT_DIR) -> str:
    """Build local variable-font declarations for the supported unicode ranges."""
    declarations: list[str] = []
    for family, file_name, unicode_range in FONT_FACES:
        uri = local_asset_data_uri(font_dir / file_name)
        if not uri:
            return ""
        declarations.append(
            "@font-face{"
            f"font-family:'{family}';src:url('{uri}') format('woff2');"
            "font-style:normal;font-weight:200 700;font-display:swap;"
            f"unicode-range:{unicode_range}"
            "}"
        )
    return "\n".join(declarations)


def lucide_icon(name: str, size: int = 17) -> str:
    """Return sanitized inline markup from a vendored Lucide icon."""
    try:
        markup = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    except OSError:
        return ""
    markup = markup[markup.find("<svg"):].strip()
    if not markup:
        return ""
    markup = markup.replace(
        f'class="lucide lucide-{name}"',
        f'class="gx-icon lucide lucide-{name}"',
        1,
    )
    markup = markup.replace('width="24"', f'width="{size}"', 1)
    markup = markup.replace('height="24"', f'height="{size}"', 1)
    markup = markup.replace(
        f'width="{size}"\n  height="{size}"',
        f'width="{size}" height="{size}"',
        1,
    )
    return markup.replace("<svg", '<svg aria-hidden="true"', 1)


def icon_span(name: str, size: int = 17) -> str:
    """Wrap a Lucide icon for consistent inline alignment."""
    icon = lucide_icon(name, size)
    return f'<span class="gx-icon-wrap">{icon}</span>' if icon else ""


def lucide_data_uri(name: str, stroke: str = "currentColor") -> str:
    """Return a recolored vendored Lucide icon as an SVG data URI."""
    markup = lucide_icon(name)
    if not markup:
        return ""
    markup = markup.replace('stroke="currentColor"', f'stroke="{stroke}"')
    encoded = base64.b64encode(markup.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def background_data_uri(path: Path = BACKGROUND_ASSET) -> str:
    """Return the repository-local dashboard background as a data URI."""
    return local_image_data_uri(path)


# ==========================================================================
# Схема входного файла
# ==========================================================================

ION_KEYS = ["Na", "Cl", "Ca", "Mg", "K", "Ba", "Sr", "Fe", "HCO3", "SO4", "CO3"]

COMPATIBILITY_MINERAL_RU = {
    "calcite": "Кальцит",
    "barite": "Барит",
    "gypsum": "Гипс",
    "halite": "Галит",
}
COMPATIBILITY_META_COLUMNS = ["name", "ph", "t_c", "p_pa"]
COMPATIBILITY_COLUMNS = COMPATIBILITY_META_COLUMNS + ION_KEYS

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

LOCATION_COLUMNS = ("latitude", "longitude", "cluster", "site")

# Синонимы заголовков (нормализуются: нижний регистр, без пробелов)
HEADER_ALIASES = {
    "name": "name", "well": "name", "well_name": "name",
    "скважина": "name", "название": "name",
    "lift": "lift_type", "способ": "lift_type", "тип насоса": "lift_type",
    "широта": "latitude", "lat": "latitude",
    "долгота": "longitude", "lon": "longitude", "lng": "longitude",
    "куст": "cluster", "участок": "site", "площадка": "site",
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
    ("latitude", "град", "Широта WGS84 для карты", "нет", "—"),
    ("longitude", "град", "Долгота WGS84 для карты", "нет", "—"),
    ("cluster", "—", "Куст скважины", "нет", "—"),
    ("site", "—", "Участок / месторождение", "нет", "—"),
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
    page_icon=FAVICON_ASSET,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Apply the corporate theme with local fonts and repository assets."""
    background_uri = background_data_uri()
    background_layer = (
        f"url('{background_uri}') center / cover fixed no-repeat"
        if background_uri else "linear-gradient(135deg, #DCE9E2, #F6F8F7)"
    )
    fonts_css = font_face_css()
    kpi_default_icon = lucide_data_uri("gauge", GREEN_700)
    kpi_critical_icon = lucide_data_uri("triangle-alert", STATUS_CRIT)
    kpi_trend_icon = lucide_data_uri("trending-up", STATUS_WARN)
    st.html(
        f"""
        <style>
        {fonts_css}
        /* ---------- базовая типографика ---------- */
        .stApp, .stApp button, .stApp input, .stApp textarea, .stApp select {{
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
        p, li, span, label {{ color: {INK}; font-weight: {WEIGHT_REGULAR}; }}
        h1, h2, h3 {{
            color: {INK};
            letter-spacing: -0.2px;
        }}

        /* строгий режим: без фирменной радужной полосы и колонтитула */
        div[data-testid="stDecoration"] {{ display: none; }}
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"],
        [data-testid="stHeader"] {{
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
        }}

        /* ---------- шапка приложения ---------- */
        .app-header {{
            position: relative;
            display: flex;
            align-items: center;
            min-height: 88px;
            background: {CANVAS};
            border: 0;
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
            font-size: 30px; font-weight: {WEIGHT_SEMIBOLD}; color: {GREEN_900};
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
            border-right: 1px solid {HAIRLINE_SOFT};
            box-shadow: none;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}
        section[data-testid="stSidebar"] > div {{ background: transparent; }}
        /* Панель — постоянная рабочая область. Прячем только кнопку закрытия.
           Кнопку восстановления намеренно оставляем доступной: она нужна,
           если браузер сохранил свёрнутое состояние от прошлого запуска. */
        [data-testid="stSidebarCollapseButton"],
        button[aria-label="Close sidebar"],
        button[aria-label="Скрыть боковую панель"] {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }}
        /* В свёрнутом состоянии expand-кнопка находится внутри stHeader.
           Возвращаем только этот узкий слой шапки, иначе кнопка не кликается. */
        header[data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"]),
        [data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"]) {{
            display: flex !important;
            visibility: visible !important;
            height: 3.75rem !important;
            min-height: 3.75rem !important;
            background: transparent !important;
            pointer-events: none !important;
        }}
        [data-testid="stExpandSidebarButton"] {{
            display: flex !important;
            visibility: visible !important;
            pointer-events: auto !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            padding-top: 0;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            --sidebar-logo-lift: clamp(44px, 6vh, 56px);
            --sidebar-content-lift: clamp(92px, 12vh, 112px);
            padding-top: clamp(12px, 2vh, 18px);
        }}
        .sidebar-logo {{
            display: block;
            width: clamp(150px, 66%, 166px);
            height: auto;
            margin: 0 auto calc(-1 * var(--sidebar-content-lift)) 0;
            object-fit: contain;
            transform: translateY(calc(-1 * var(--sidebar-logo-lift)));
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
            color: {INK} !important; font-weight: {WEIGHT_MEDIUM};
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
            font-weight: {WEIGHT_MEDIUM};
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
            border-radius: 10px; font-weight: {WEIGHT_MEDIUM}; font-size: 14px;
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
            font-weight: {WEIGHT_MEDIUM}; font-size: 14px; padding: 8px 18px; width: 100%;
        }}
        .stDownloadButton > button:hover {{
            background: {GREEN_100}; color: {GREEN_900};
            border: 1px solid {GREEN_900};
        }}

        /* ---------- вкладки ---------- */
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 6px;
            background: {SURFACE};
            border: 0;
            border-radius: 12px;
            padding: 6px;
        }}
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            background: transparent; color: {INK_MUTED};
            font-weight: {WEIGHT_MEDIUM}; font-size: 14px;
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
            border: 0;
            border-radius: {RADIUS}; padding: 18px 20px 14px 20px;
            box-shadow: {SHADOW_SOFT};
            transition: box-shadow 0.18s ease, transform 0.18s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            box-shadow: {SHADOW_RAISED};
            transform: translateY(-1px);
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {INK_MUTED}; font-size: 11px; font-weight: {WEIGHT_SEMIBOLD};
            text-transform: uppercase; letter-spacing: 0.8px;
            margin-bottom: 6px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {INK}; font-weight: {WEIGHT_SEMIBOLD}; letter-spacing: -0.6px;
        }}

        /* ---------- таблицы, canvas-grid и графики ---------- */
        div[data-testid="stDataFrame"] {{
            background: {CANVAS}; color: {INK};
            border: 0; border-radius: {RADIUS};
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
        table, th, td {{ color: {INK}; border-color: {HAIRLINE_FAINT}; }}
        td {{ border-left: none; border-right: none; }}
        th {{
            background: {SURFACE}; font-weight: {WEIGHT_SEMIBOLD};
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
            border: 0;
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
            font-size: 17px; font-weight: {WEIGHT_SEMIBOLD}; color: {INK};
            letter-spacing: -0.2px; margin: 6px 0 12px 0;
        }}

        /* ---------- статус-чипы тёплой шкалы (как на референсе) ---------- */
        .status-chip {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 999px;
            font-size: 12px; font-weight: {WEIGHT_SEMIBOLD}; letter-spacing: 0.2px;
            border: 0;
        }}
        .status-chip.is-ok {{ background: {STATUS_OK_BG}; color: {STATUS_OK}; }}
        .status-chip.is-warn {{ background: {STATUS_WARN_BG}; color: {STATUS_WARN}; }}
        .status-chip.is-high {{ background: {STATUS_HIGH_BG}; color: {STATUS_HIGH}; }}
        .status-chip.is-crit {{ background: {STATUS_CRIT_BG}; color: {STATUS_CRIT}; }}

        /* Мягкая карточка-контейнер для блоков плана и прогноза. */
        .surface-card {{
            background: {CANVAS};
            border: 0;
            border-radius: {RADIUS};
            box-shadow: {SHADOW_SOFT};
            padding: 18px 20px;
        }}

        /* ---------- shell: navigation / overview / alerts ---------- */
        .shell-eyebrow {{
            color: {GREEN_700}; font-size: 11px; font-weight: {WEIGHT_SEMIBOLD};
            letter-spacing: 1.2px; text-transform: uppercase;
            margin: 0 0 5px 0;
        }}
        .shell-heading {{
            color: {INK}; font-size: 22px; font-weight: {WEIGHT_SEMIBOLD};
            letter-spacing: -0.35px; margin: 0 0 4px 0;
        }}
        .shell-copy {{ color: {INK_MUTED}; font-size: 13px; margin-bottom: 14px; }}
        .sidebar-nav-title {{
            color: {INK_MUTED}; font-size: 10px; font-weight: {WEIGHT_SEMIBOLD};
            letter-spacing: 1.1px; text-transform: uppercase; margin: 4px 0 8px;
        }}
        .sidebar-nav {{ display: grid; gap: 3px; margin: 0 0 14px; }}
        .sidebar-nav-item {{
            display: flex; align-items: center; gap: 9px;
            min-height: 34px; padding: 7px 9px; border-radius: 9px;
            color: {INK_MUTED}; font-size: 12px; font-weight: {WEIGHT_MEDIUM};
        }}
        .sidebar-nav-item:first-child {{ background: {GREEN_100}; color: {GREEN_900}; }}
        .sidebar-nav-index {{
            display: inline-grid; place-items: center; width: 20px; height: 20px;
            border-radius: 6px; background: #FFFFFF; color: {GREEN_700};
            font-size: 10px; font-weight: {WEIGHT_SEMIBOLD};
        }}
        .alerts-rail {{
            border-left: 1px solid {HAIRLINE_SOFT}; padding-left: 14px;
            min-height: 100%;
        }}
        .alert-card {{
            background: {CANVAS}; border: 0;
            border-left: 3px solid var(--alert-accent, {STATUS_WARN});
            border-radius: 11px; padding: 11px 12px; margin-bottom: 9px;
            box-shadow: {SHADOW_SOFT}; overflow-wrap: anywhere;
        }}
        .alert-card.is-critical {{ --alert-accent: {STATUS_CRIT}; }}
        .alert-card.is-warning {{ --alert-accent: {STATUS_WARN}; }}
        .alert-card.is-ok {{ --alert-accent: {STATUS_OK}; }}
        .alert-card-title {{ color: {INK}; font-size: 13px; font-weight: {WEIGHT_SEMIBOLD}; }}
        .alert-card-meta {{ color: {INK_MUTED}; font-size: 11px; margin-top: 3px; }}
        .overview-table {{ width: 100%; font-size: 12px; }}
        .overview-table td {{ padding: 8px 6px; border-bottom: 1px solid {HAIRLINE_FAINT}; }}
        .overview-table td:last-child {{ text-align: right; font-weight: {WEIGHT_SEMIBOLD}; }}

        @media (max-width: 1100px) {{
            .alerts-rail {{ border-left: 0; border-top: 1px solid {HAIRLINE_SOFT};
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

        /* ---------- redesign 2026: light three-column operations dashboard ---------- */
        .stApp {{ background: #F7F9F8; }}
        .stApp [data-testid="stMainBlockContainer"] {{
            max-width: 1680px; padding-top: 24px; padding-left: 2rem;
            padding-right: 2rem; margin-top: 0;
        }}
        section[data-testid="stSidebar"] {{ width: 248px !important; background: #FFFFFF; }}
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            --sidebar-logo-lift: clamp(52px, 7vh, 64px);
            --sidebar-content-lift: clamp(48px, 6vh, 60px);
            transform: none;
            padding-top: 0;
        }}
        .sidebar-logo {{
            width: min(180px, calc(100% - 16px));
            margin: 0 0 calc(-1 * var(--sidebar-content-lift)) 8px;
            transform: translateY(calc(-1 * var(--sidebar-logo-lift)));
            transform-origin: left top;
        }}
        .sidebar-nav {{ gap: 5px; }}
        .sidebar-nav-item {{ min-height: 42px; padding: 9px 11px; border-radius: 10px;
            font-size: 13px; font-weight: {WEIGHT_SEMIBOLD}; }}
        .sidebar-nav-item:first-child {{ background: #EAF6EF; color: {GREEN_700}; }}
        .sidebar-nav-index {{ width: 23px; height: 23px; border-radius: 7px;
            background: transparent; color: inherit; }}
        .ui-icon {{ display:inline-block; width:17px; height:17px; flex:0 0 auto;
            object-fit:contain; vertical-align:-3px; }}
        .sidebar-nav-index .ui-icon {{ width:18px; height:18px; }}
        .filter-pill .ui-icon:last-child {{ width:14px; height:14px; margin-left:2px; opacity:.72; }}
        .alert-card-title .ui-icon {{ width:15px; height:15px; margin-right:6px; }}
        .dashboard-header {{ display:flex; align-items:flex-end; justify-content:space-between;
            gap:18px; margin: 2px 0 18px; }}
        .dashboard-title {{ font-size: 28px; line-height:1.1; font-weight: {WEIGHT_SEMIBOLD};
            letter-spacing:-.55px; color:{GREEN_700}; }}
        .dashboard-subtitle {{ margin-top:6px; color:{INK_MUTED}; font-size:13px; }}
        .dashboard-filters {{ display:flex; align-items:center; gap:9px; flex-wrap:wrap; }}
        .filter-pill {{ display:inline-flex; align-items:center; gap:8px; min-height:40px;
            padding:8px 13px; border:0; border-radius:10px;
            background:#FFFFFF; color:{INK}; font-size:12px; font-weight:{WEIGHT_MEDIUM};
            box-shadow:{SHADOW_SOFT}; }}
        .filter-pill .filter-icon {{ color:{GREEN_700}; font-size:14px; }}
        div[data-testid="stMetric"] {{ min-height: 118px; padding:18px 18px 14px 62px;
            position:relative; border:0; }}
        div[data-testid="stMetric"]::before {{ position:absolute; left:18px; top:20px;
            width:32px; height:32px; border-radius:9px; content:'';
            background: {STATUS_OK_BG} url('{kpi_default_icon}') center / 17px 17px no-repeat; }}
        div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] {{ background:{STATUS_CRIT_BG}; }}
        div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"]::before {{
            background: #FFFFFF url('{kpi_critical_icon}') center / 17px 17px no-repeat; }}
        div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"]::before {{
            background: {STATUS_WARN_BG} url('{kpi_trend_icon}') center / 17px 17px no-repeat; }}
        div[data-testid="stMetricValue"] {{ font-size:30px; }}
        .panel-title {{ display:flex; align-items:center; justify-content:space-between;
            font-size:15px; font-weight:{WEIGHT_SEMIBOLD}; color:{INK}; margin:0 0 4px; }}
        .panel-card {{ background:#FFFFFF; border:0; border-radius:{RADIUS};
            box-shadow:{SHADOW_SOFT}; padding:16px 18px; }}
        div[data-testid="stPlotlyChart"] {{ background:#FFFFFF; border:0;
            border-radius:{RADIUS}; box-shadow:{SHADOW_SOFT}; overflow:hidden; }}
        .overview-table-wrap {{ background:#FFFFFF; border:0;
            border-radius:{RADIUS}; box-shadow:{SHADOW_SOFT}; padding:15px 17px; }}
        .overview-table th {{ text-align:left; padding:9px 6px; color:{INK_MUTED};
            font-size:10px; letter-spacing:.55px; text-transform:uppercase; }}
        .risk-badge {{ display:inline-block; min-width:46px; padding:4px 8px; border-radius:7px;
            color:#FFFFFF; text-align:center; font-weight:{WEIGHT_SEMIBOLD}; }}
        .alerts-rail {{ border-left:0; padding-left:0; }}
        .alert-card {{ border:0; border-left:0; padding:13px 14px; margin-bottom:10px; }}
        .alert-card.is-critical {{ background:{STATUS_CRIT_BG}; }}
        .alert-card.is-warning {{ background:{STATUS_WARN_BG}; }}
        .alerts-count {{ display:inline-grid; place-items:center; min-width:22px; height:22px;
            margin-left:6px; padding:0 6px; border-radius:999px; color:#FFFFFF;
            background:{STATUS_CRIT}; font-size:11px; }}
        @media (max-width: 1180px) {{
            section[data-testid="stSidebar"] {{ width: 220px !important; }}
            .stApp [data-testid="stMainBlockContainer"] {{ padding-left:1rem; padding-right:1rem; }}
        }}
        @media (max-width: 760px) {{
            .dashboard-header {{ align-items:flex-start; flex-direction:column; }}
            .dashboard-filters {{ width:100%; }}
            .filter-pill {{ flex:1 1 170px; }}
            div[data-testid="stMetric"] {{ min-height:100px; }}
        }}
        </style>
        """,
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
        | {c.lower(): c for c in LOCATION_COLUMNS}
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

        for canon in LOCATION_COLUMNS:
            orig = by_canon.get(canon)
            raw = None if orig is None or pd.isna(row[orig]) else row[orig]
            if isinstance(raw, str) and not raw.strip():
                raw = None
            if canon in {"latitude", "longitude"}:
                value = pd.to_numeric(raw, errors="coerce") if raw is not None else None
                vals[canon] = None if value is None or pd.isna(value) else float(value)
                if raw is not None and vals[canon] is None:
                    errors.append(f"«{label}»: координата {canon} некорректна и не будет показана на карте")
            else:
                vals[canon] = None if raw is None else str(raw).strip() or None

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
        latitude=vals.get("latitude"), longitude=vals.get("longitude"),
        cluster=vals.get("cluster"), site=vals.get("site"),
        provenance=provenance,
    )


def template_frame() -> pd.DataFrame:
    """Одна строка-пример с полным набором колонок (скв. Речицкая 123)."""
    example = {
        "name": "Речицкая 123",
        "latitude": 52.371, "longitude": 30.387,
        "cluster": "Куст 3", "site": "Речицкое",
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


def compatibility_template_frame() -> pd.DataFrame:
    """Пустой двухстрочный контракт; химия намеренно не подставляется."""
    return pd.DataFrame([
        {"name": "Вода A", "ph": pd.NA, "t_c": pd.NA, "p_pa": pd.NA,
         **{ion: pd.NA for ion in ION_KEYS}},
        {"name": "Вода B", "ph": pd.NA, "t_c": pd.NA, "p_pa": pd.NA,
         **{ion: pd.NA for ion in ION_KEYS}},
    ], columns=COMPATIBILITY_COLUMNS)


def compatibility_template_bytes() -> bytes:
    """XLSX-шаблон именно для двух измеренных анализов воды."""
    instructions = pd.DataFrame([
        {"Поле": "name", "Единица": "—", "Описание": "Название воды; обязательно"},
        {"Поле": "ph", "Единица": "—", "Описание": "Измеренный pH, 0…14; обязательно"},
        {"Поле": "t_c", "Единица": "°C", "Описание": "Температура анализа; обязательно"},
        {"Поле": "p_pa", "Единица": "Па", "Описание": "Давление анализа, неотрицательное; обязательно"},
        {"Поле": "Na…CO3", "Единица": "мг/л", "Описание": "Только измеренные ионы; пусто означает «нет данных», не ноль"},
    ])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        compatibility_template_frame().to_excel(writer, sheet_name="Две воды", index=False)
        instructions.to_excel(writer, sheet_name="Инструкция", index=False)
    return buffer.getvalue()


def compatibility_waters_from_frame(df: pd.DataFrame) -> tuple[galit.CompatibilityWater, galit.CompatibilityWater]:
    """Строго преобразовать ровно две строки без типовой/синтетической химии."""
    if len(df) != 2:
        raise ValueError("нужны ровно две строки: вода A и вода B")
    missing = [column for column in COMPATIBILITY_META_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("нет обязательных колонок: " + ", ".join(missing))
    waters = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        name = str(row["name"]).strip() if not pd.isna(row["name"]) else ""
        if not name:
            raise ValueError(f"строка {index}: укажите название воды")
        required = {}
        for column in ("ph", "t_c", "p_pa"):
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(value) or not math.isfinite(float(value)):
                raise ValueError(f"«{name}»: укажите числовое значение {column}")
            required[column] = float(value)
        ions: dict[str, float] = {}
        for ion in ION_KEYS:
            if ion not in df.columns or pd.isna(row[ion]) or str(row[ion]).strip() == "":
                continue
            value = pd.to_numeric(pd.Series([row[ion]]), errors="coerce").iloc[0]
            if pd.isna(value) or not math.isfinite(float(value)):
                raise ValueError(f"«{name}»: {ion} должно быть числом")
            ions[ion] = float(value)
        if not ions:
            raise ValueError(f"«{name}»: внесите хотя бы один измеренный ион")
        waters.append(galit.CompatibilityWater(
            ions, required["ph"], required["t_c"], required["p_pa"], name,
        ))
    return waters[0], waters[1]


def compatibility_profile_from_frame(df: pd.DataFrame) -> tuple[galit.ProfilePoint, ...]:
    """Преобразовать измеренный профиль; пустые строки игнорируются."""
    points = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        values = [row.get("depth_m"), row.get("t_c"), row.get("p_pa")]
        if all(pd.isna(value) or str(value).strip() == "" for value in values):
            continue
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"профиль, строка {index}: заполните глубину, температуру и давление")
        points.append(galit.ProfilePoint(*map(float, numeric)))
    return tuple(points)


def compatibility_curve_from_frame(product: str, mineral: str, reference: str,
                                   validated: bool, df: pd.DataFrame) -> galit.DoseResponseCurve:
    """Создать кривую только после явного подтверждения лабораторной валидации."""
    if not validated:
        raise ValueError("подтвердите, что кривая валидирована лабораторией для этих вод и условий")
    points = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        values = [row.get("dose_mg_l"), row.get("maximum_supported_si")]
        if all(pd.isna(value) or str(value).strip() == "" for value in values):
            continue
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"кривая дозы, строка {index}: заполните оба значения")
        points.append(galit.DoseResponsePoint(*map(float, numeric)))
    return galit.DoseResponseCurve(product, mineral, tuple(points), True, reference)


def compatibility_ratio_frame(result: galit.CompatibilityResult) -> pd.DataFrame:
    """Плоские данные четырёх SI для таблицы и Plotly."""
    return pd.DataFrame([{
        "Доля воды B, %": row.fraction_b * 100,
        "A:B": row.ratio_a_to_b,
        **{COMPATIBILITY_MINERAL_RU[mineral]: row.minerals[mineral].saturation_index
           for mineral in galit.compatibility.SUPPORTED_MINERALS},
        "Небезопасно": row.unsafe,
    } for row in result.ratios])


def fig_compatibility_ratios(result: galit.CompatibilityResult) -> go.Figure:
    """SI по доле B: четыре минерала и явная линия равновесия SI=0."""
    frame = compatibility_ratio_frame(result)
    colors = [GREEN_700, "#6D4C41", "#C47F00", "#546E7A"]
    fig = go.Figure()
    for (mineral, label), color in zip(COMPATIBILITY_MINERAL_RU.items(), colors):
        fig.add_trace(go.Scatter(
            x=frame["Доля воды B, %"], y=frame[label], mode="lines", name=label,
            line=dict(width=2.5, color=color), connectgaps=False,
            hovertemplate=f"{label}: %{{y:.3f}}<br>Вода B: %{{x:.0f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=STATUS_CRIT,
                  annotation_text="SI = 0 — граница пересыщения")
    fig.update_layout(
        xaxis_title="Доля воды B в смеси, %", yaxis_title="Индекс насыщения SI",
        font=dict(family=FONT_FAMILY, color=INK), paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF", hovermode="x unified", height=430,
        margin=dict(l=20, r=20, t=35, b=20), legend=dict(orientation="h", y=1.12),
    )
    fig.update_xaxes(range=[0, 100], gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


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
        styles[risk_i] = f"background-color: {fg}; color: #FFFFFF; font-weight: {WEIGHT_SEMIBOLD}"
        leader_i = df.columns.get_loc("Лидер")
        styles[leader_i] = f"{styles[leader_i]}; color: {GREEN_900}; font-weight: {WEIGHT_SEMIBOLD}"
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
    "corrosion_wall_loss_mm", "oil_rate_m3_day", "quality", "source", "regime_id",
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
        if "regime_id" in frame.columns and pd.notna(row.get("regime_id")):
            values["regime_id"] = str(row["regime_id"])
        snapshots.append(galit.ForecastSnapshot(**values))
    return galit.ForecastHistory(tuple(snapshots))


def fig_forecast_timeline(forecast: galit.WellForecast, *, calendar: bool = True) -> go.Figure:
    """Render honest range bars only for events with a computed temporal window."""
    dated = [event for event in forecast.events if event.horizon_start_days is not None]
    fig = go.Figure()
    for event in dated:
        if calendar:
            start = event.horizon_start_date
            end = event.horizon_end_date or start
            base, width = start, max((end - start).days, 0) + 1
            window = f"{start.isoformat()} — {end.isoformat()}"
        else:
            base = event.horizon_start_days
            width = event.horizon_end_days - event.horizon_start_days
            window = f"{event.horizon_start_days:.1f}–{event.horizon_end_days:.1f} дней"
        fig.add_trace(go.Bar(
            x=[width], y=[event.title], base=[base], orientation="h",
            name=event.status.value, showlegend=False,
            marker_color=GREEN_700 if event.status.value == "calibrated" else STATUS_WARN,
            customdata=[[window, event.status.value]],
            hovertemplate="%{y}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        ))
    if dated:
        marker = forecast.as_of.date() if calendar and forecast.as_of else 0
        fig.add_vline(x=marker, line_dash="dash", line_color=STATUS_CRIT,
                      annotation_text="сегодня")
    fig.update_layout(height=max(220, 65 * len(dated)), barmode="overlay",
                      xaxis_title="Календарное окно" if calendar else "Дни от расчёта",
                      yaxis_title="", font=dict(family=FONT_FAMILY, color=INK),
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


def parse_optional_nonnegative(raw: str, label: str) -> float | None:
    """Parse an optional dashboard economic input without inventing a default."""
    text = raw.strip().replace(",", ".")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{label}: требуется число") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: требуется конечное неотрицательное число")
    return value


def scenario_for_dashboard(case: WellCase, changes: galit.ScenarioChanges,
                           economics: galit.ScenarioEconomics | None = None) -> galit.ScenarioComparison:
    """Pure adapter used by the scenario tab and unit tests."""
    return galit.compare_scenario(case, changes, economics)


def risk_economics_for_dashboard(case: WellCase, *, probability: float | None,
                                 horizon_days: float, efficiency: float,
                                 event_downtime_days: float, treatment_downtime_days: float,
                                 price: float | None, operating_loss: float | None,
                                 treatment_cost: float | None, currency: str | None,
                                 loss_fraction: float = 1.0) -> galit.RiskEconomicsResult:
    """Pure adapter shared by Streamlit rendering and tests."""
    return galit.calculate_risk_economics(galit.RiskEconomicsInput(
        event_probability=probability, horizon_days=horizon_days,
        treatment_efficiency=efficiency, event_downtime_days=event_downtime_days,
        treatment_downtime_days=treatment_downtime_days,
        oil_rate_m3_day=case.rate.q_oil_m3d, product_price_per_m3=price,
        operating_loss_per_day=operating_loss, treatment_cost=treatment_cost,
        currency=currency.strip().upper() if currency and currency.strip() else None,
        production_loss_fraction=loss_fraction,
        probability_source="dashboard_explicit_input",
    ))


def field_map_data(cases_by_name: dict[str, WellCase],
                   results: list[DiagnosisResult]) -> galit.FieldMapData:
    """Pair current cases/results and delegate all map semantics to the core."""
    items = [DiagnosedWell(cases_by_name[result.well], result)
             for result in results if result.well in cases_by_name]
    return galit.prepare_field_map(items)


# Ориентировочная зона Припятского прогиба: около 280 x 150 км, NW-SE.
# Это визуальный региональный контекст, а не лицензионная геологическая граница.
PRIPYAT_OVERVIEW_LON = (27.05, 27.45, 28.65, 29.95, 31.25, 31.30, 30.25, 28.90, 27.70, 27.05)
PRIPYAT_OVERVIEW_LAT = (52.18, 52.73, 52.78, 52.57, 52.35, 51.65, 51.32, 51.43, 51.76, 52.18)


def field_map_viewport(data: galit.FieldMapData) -> tuple[dict[str, float], float]:
    """Return a stable wells-only viewport; never let a tiny spread or outlier show the world."""
    latitudes = [point.latitude for point in data.points]
    longitudes = [point.longitude for point in data.points]
    if not latitudes:
        return {"lat": 52.10, "lon": 29.10}, 6.4

    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_min, lon_max = min(longitudes), max(longitudes)
    center = {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2}
    lat_span = max(lat_max - lat_min, 0.025)
    lon_span = max(lon_max - lon_min, 0.04)
    latitude_scale = max(math.cos(math.radians(center["lat"])), 0.2)
    effective_span = max(lat_span, lon_span * latitude_scale)
    # 1.9 supplies padding; clamps retain local detail and prevent a world view.
    zoom = math.log2(180.0 / (effective_span * 1.9)) - 1.0
    return center, max(6.0, min(10.5, zoom))


def smart_map_service(items: list[DiagnosedWell] | None = None,
                      occurred_at: datetime | None = None) -> galit.SmartMapService:
    """Combine the current diagnosis slice with persisted history, without copying sources."""
    configured = Path(os.environ.get("GALIT_SMART_MAP_STORAGE", "data/smart_map.json"))
    path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    observations = []
    for item in items or []:
        if galit.valid_coordinates(item.case.latitude, item.case.longitude):
            observations.append(galit.observation_from_diagnosed(
                item, occurred_at or datetime.now(timezone.utc), source="dashboard_current_slice"
            ))
    return galit.SmartMapService(galit.SmartMapRepository(path), observations=observations)


def fig_smart_map(snapshot: dict[str, Any], *, infrastructure: dict[str, Any] | None = None,
                  hotspots: list[dict[str, Any]] | None = None, frames: list[dict[str, Any]] | None = None,
                  show_heatmap: bool = True, show_markers: bool = True) -> go.Figure:
    """Plotly 6.9 token-free Densitymap + Scattermap with persistent base layers."""
    points = snapshot.get("points", [])
    dummy = galit.FieldMapData((), galit.FieldMapSummary(0, 0, 0, 0, 0, {}, None))
    if points:
        center = {"lat": sum(x["latitude"] for x in points)/len(points),
                  "lon": sum(x["longitude"] for x in points)/len(points)}
        zoom = 8.2
    else:
        center, zoom = field_map_viewport(dummy)
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=PRIPYAT_OVERVIEW_LAT, lon=PRIPYAT_OVERVIEW_LON,
        mode="lines", fill="toself", name="Припятский прогиб · обзорно",
        line={"color":"rgba(15,107,67,.62)","width":2}, fillcolor="rgba(61,139,102,.10)", hoverinfo="skip"))
    if show_heatmap and points:
        fig.add_trace(go.Densitymap(lat=[x["latitude"] for x in points],lon=[x["longitude"] for x in points],
            z=[x["heat_weight"] for x in points],radius=28,name="Концентрация риска",
            colorscale=[[0,"rgba(255,255,178,.15)"],[.45,"#FDBB2D"],[1,"#B02020"]],showscale=True,
            colorbar={"title":"нормир. вклад"},hoverinfo="skip"))
    if infrastructure:
        for feature in infrastructure.get("features", []):
            geometry=feature["geometry"]; props=feature.get("properties",{}); coordinates=geometry["coordinates"]
            if geometry["type"]=="Point":
                fig.add_trace(go.Scattermap(lat=[coordinates[1]],lon=[coordinates[0]],mode="markers",name="Объекты GIS",
                    marker={"size":13,"symbol":"square","color":"#0E7490"},text=[props.get("name","Объект")],
                    hovertemplate="<b>%{text}</b><br>Тип: "+str(props.get("asset_type","—"))+"<br>Статус: "+str(props.get("status","—"))+"<extra></extra>"))
            else:
                segments=coordinates if geometry["type"]=="MultiLineString" else [coordinates]
                for segment in segments:
                    fig.add_trace(go.Scattermap(lat=[x[1] for x in segment],lon=[x[0] for x in segment],mode="lines",name="Трубопроводы GIS",
                        line={"width":4,"color":"#475569"},text=[props.get("name","Линия")]*len(segment),hovertemplate="%{text}<extra></extra>"))
    for zone in hotspots or []:
        c=zone["centroid"]; fig.add_trace(go.Scattermap(lat=[c["latitude"]],lon=[c["longitude"]],mode="markers+text",
            text=[f"Зона · {len(zone['member_wells'])}"],textposition="top center",name="Системная зона",
            marker={"size":max(24,zone["radius_km"]*15),"color":"rgba(176,32,32,.28)"},
            hovertemplate="%{text}<br>confidence: "+zone["confidence"]+"<extra></extra>"))
    if show_markers:
        symbols={"producer":"circle","injector":"triangle-up","unknown":"diamond"}
        colors={"normal":"#246B2A","growing":"#B26A00","critical":"#B02020"}
        for role in ("producer","injector","unknown"):
            rows=[x for x in points if x["well_role"]==role]
            fig.add_trace(go.Scattermap(lat=[x["latitude"] for x in rows],lon=[x["longitude"] for x in rows],mode="markers",name=role,
                text=[x["well"]["display_name"] for x in rows],customdata=[[x["selected_severity"],x["status"],x["occurred_at"],x["source_quality"]] for x in rows],
                marker={"size":14,"symbol":symbols[role],"color":[colors[x["status"]] for x in rows],"opacity":.9},
                hovertemplate="<b>%{text}</b><br>Тяжесть: %{customdata[0]:.2f}<br>Статус: %{customdata[1]}<br>Дата: %{customdata[2]}<br>Качество: %{customdata[3]}<extra></extra>"))
    plot_frames=[]
    for frame in frames or []:
        rows=frame.get("points",[])
        plot_frames.append(go.Frame(name=frame["date"],data=[go.Scattermap(lat=[x["latitude"] for x in rows],lon=[x["longitude"] for x in rows],
            mode="markers",marker={"size":14,"color":[x["selected_severity"] for x in rows],"cmin":0,"cmax":1,"colorscale":"YlOrRd"},text=[x["well"]["display_name"] for x in rows])]))
    fig.frames=plot_frames
    if plot_frames:
        fig.update_layout(updatemenus=[{"type":"buttons","buttons":[{"label":"▶","method":"animate","args":[None,{"frame":{"duration":500,"redraw":True},"fromcurrent":True}]},{"label":"⏸","method":"animate","args":[[None],{"mode":"immediate"}]}]}],
            sliders=[{"steps":[{"label":f.name,"method":"animate","args":[[f.name],{"mode":"immediate","frame":{"duration":0,"redraw":True}}]} for f in plot_frames]}])
    fig.update_layout(map={"style":"open-street-map","center":center,"zoom":zoom},height=620,margin={"l":4,"r":4,"t":45,"b":4},
        legend={"orientation":"h","y":1.02},uirevision="galit-smart-map-2")
    return fig


def fig_field_map(data: galit.FieldMapData) -> go.Figure:
    """Interactive token-free OSM tile map with risk semantics and regional context."""
    center, zoom = field_map_viewport(data)
    fig = go.Figure()

    fig.add_trace(go.Scattermap(
        lat=PRIPYAT_OVERVIEW_LAT, lon=PRIPYAT_OVERVIEW_LON,
        mode="lines", fill="toself", name="Припятский прогиб · обзорно",
        line=dict(color="rgba(15,107,67,.62)", width=2),
        fillcolor="rgba(61,139,102,.13)", hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scattermap(
        lat=[52.12], lon=[30.22], mode="text", showlegend=False,
        text=["ПРИПЯТСКИЙ ПРОГИБ · ОБЗОРНО"],
        textfont=dict(size=11, color=GREEN_900), hoverinfo="skip",
    ))

    for status, label in galit.MAP_STATUS_LABELS.items():
        points = [point for point in data.points if point.status == status]
        fig.add_trace(go.Scattermap(
            lat=[point.latitude for point in points],
            lon=[point.longitude for point in points],
            text=[point.well for point in points],
            customdata=[[
                point.risk, MECH_RU.get(point.dominant, point.dominant),
                point.possible_oil_loss_m3d if point.possible_oil_loss_m3d is not None else "—",
                point.cluster or "—", point.site or "—",
            ] for point in points],
            mode="markers", name=label,
            marker=dict(size=[point.marker_size for point in points],
                        color=galit.MAP_STATUS_COLORS[status], opacity=.88),
            hovertemplate=("<b>%{text}</b><br>Статус: " + label +
                           "<br>Риск: %{customdata[0]:.2f}<br>Механизм: %{customdata[1]}"
                           "<br>Потеря под риском: %{customdata[2]} м³/сут"
                           "<br>Куст: %{customdata[3]}<br>Участок: %{customdata[4]}<extra></extra>"),
        ))
    fig.update_layout(
        map=dict(style="open-street-map", center=center, zoom=zoom),
        height=560, autosize=True, paper_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, color=INK),
        margin=dict(l=4, r=4, t=48, b=4),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font=dict(color=INK)),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(255,255,255,.88)"),
        uirevision="galit-field-map",
    )
    return fig


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


def fig_risk_overview(results: list[DiagnosisResult]) -> go.Figure:
    """Two-line current-fund profile; it deliberately does not pretend to be history."""
    ordered = sorted(results, key=lambda item: item.integrated_risk, reverse=True)
    risks = [item.integrated_risk for item in ordered]
    running_average = [sum(risks[:index]) / index for index in range(1, len(risks) + 1)]
    labels = [item.well for item in ordered]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=running_average, mode="lines+markers",
        name="Средний риск выбранного фонда", line=dict(color=GREEN_700, width=2.5),
        marker=dict(size=5), hovertemplate="%{x}<br>Средний риск: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=risks, mode="lines+markers", name="Риск скважины",
        line=dict(color=STATUS_HIGH, width=2.2), marker=dict(size=5),
        hovertemplate="%{x}<br>Риск: %{y:.2f}<extra></extra>",
    ))
    fig.add_hrect(y0=RISK_CRIT, y1=1.0, fillcolor=STATUS_CRIT_BG, opacity=.55,
                  line_width=0, layer="below")
    fig.update_layout(
        height=300, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, color=INK), hovermode="x unified",
        legend=dict(orientation="h", y=-.24, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=12, r=12, t=18, b=60),
        xaxis=dict(title="", showgrid=False, tickangle=-25),
        yaxis=dict(title="Риск, 0–1", range=[0, 1], dtick=.2,
                   gridcolor=HAIRLINE, zeroline=False),
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
    """Reference-inspired page header without inventing dates or field metadata."""
    context_icon = icon_span("layers")
    location_icon = icon_span("map-pin")
    date_chevron = icon_span("chevron-down")
    location_chevron = icon_span("chevron-down")
    st.html(
        f"""
        <div class="dashboard-header">
            <div>
                <div class="dashboard-title">ГАЛИТ</div>
                <div class="dashboard-subtitle">Диагностика осложнений и приоритеты обслуживания</div>
            </div>
            <div class="dashboard-filters" aria-label="Контекст текущего расчёта">
                <div class="filter-pill">{context_icon}Текущий расчёт {date_chevron}</div>
                <div class="filter-pill">{location_icon}Все месторождения {location_chevron}</div>
            </div>
        </div>
        """
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
        with st.expander("Совместимость двух вод: файл"):
            compatibility_upload = st.file_uploader(
                "Двухстрочный XLSX / CSV", type=["xlsx", "xls", "csv"],
                key="compatibility-upload",
                help="Скачайте специальный шаблон в разделе совместимости и заполните две строки.",
            )
            st.session_state["compatibility-upload-bytes"] = (
                compatibility_upload.getvalue() if compatibility_upload is not None else None
            )
            st.session_state["compatibility-upload-name"] = (
                compatibility_upload.name if compatibility_upload is not None else None
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


def watercut_storage_path() -> Path:
    configured = Path(os.environ.get("GALIT_WATERCUT_STORAGE", "data/watercut.json"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def get_watercut_repository() -> galit.WatercutRepository:
    path = str(watercut_storage_path().resolve())
    cached = st.session_state.get("watercut_repository")
    if cached is None or str(cached.path.resolve()) != path:
        cached = galit.WatercutRepository(path)
        st.session_state["watercut_repository"] = cached
    return cached


def watercut_portfolio(repository: galit.WatercutRepository) -> list[galit.WatercutDiagnosis]:
    metadata = repository.list_metadata(); production = repository.list_production(); injection = repository.list_injection()
    injectors = [item for item in metadata if item.role == "injector"]
    results = []
    for producer in (item for item in metadata if item.role == "producer"):
        try: results.append(galit.diagnose_watercut(producer, production, injectors, injection))
        except ValueError: continue
    return sorted(results, key=lambda item: ({"critical": 2, "growing": 1, "low": 0}.get(item.severity, -1), item.current_water_cut or 0), reverse=True)


def fig_watercut_links(metadata: list[galit.WellMetadata], links: list[galit.WatercutLink]) -> go.Figure:
    fig = go.Figure()
    for role, symbol, color in (("producer", "circle", GREEN_700), ("injector", "diamond", "#1665A5")):
        rows = [x for x in metadata if x.role == role and x.latitude is not None and x.longitude is not None]
        fig.add_trace(go.Scattermap(lat=[x.latitude for x in rows], lon=[x.longitude for x in rows], text=[x.well for x in rows],
                                    mode="markers", name="Добывающие" if role == "producer" else "Нагнетательные",
                                    marker={"size": 12, "color": color, "symbol": symbol}, hovertemplate="%{text}<extra></extra>"))
    for link in links:
        fig.add_trace(go.Scattermap(lat=[link.injector_latitude, link.producer_latitude], lon=[link.injector_longitude, link.producer_longitude],
                                    mode="lines+markers", name=link.label, showlegend=False,
                                    line={"color": STATUS_CRIT if link.status == "critical" else STATUS_WARN, "width": 2 + 3 * link.score},
                                    marker={"size": [2, 8], "symbol": ["circle", "triangle-up"]},
                                    text=[link.label, f"score={link.score:.2f}; {link.distance_km:.1f} км; lag={link.lag_days}; confidence={link.confidence}"],
                                    hovertemplate="%{text}<extra></extra>"))
    located = [x for x in metadata if x.latitude is not None and x.longitude is not None]
    center = {"lat": sum(x.latitude for x in located)/len(located), "lon": sum(x.longitude for x in located)/len(located)} if located else {"lat": 52.3, "lon": 29.5}
    fig.update_layout(map={"style": "open-street-map", "center": center, "zoom": 7}, margin={"l":0,"r":0,"t":0,"b":0}, height=520)
    return fig


def render_watercut(repository: galit.WatercutRepository) -> None:
    st.subheader("Диагностика обводнения")
    st.warning(galit.WATERCUT_DISCLAIMER)
    a,b,c = st.columns(3)
    a.download_button("Шаблон metadata", galit.metadata_csv_template(), "watercut-metadata.csv", "text/csv", key="wc-meta-template")
    b.download_button("Шаблон добычи", galit.production_csv_template(), "watercut-production.csv", "text/csv", key="wc-prod-template")
    c.download_button("Шаблон закачки", galit.injection_csv_template(), "watercut-injection.csv", "text/csv", key="wc-inj-template")
    with st.form("watercut-import-form"):
        meta_file=st.file_uploader("Metadata CSV",type=["csv"],key="wc-meta-file")
        prod_file=st.file_uploader("Production CSV",type=["csv"],key="wc-prod-file")
        inj_file=st.file_uploader("Injection CSV",type=["csv"],key="wc-inj-file")
        submit=st.form_submit_button("Проверить и импортировать")
    if submit:
        messages=[]; errors=[]
        try:
            if meta_file:
                rows,bad=galit.metadata_from_csv(meta_file.getvalue().decode("utf-8-sig")); errors+=bad
                for row in rows: repository.upsert_metadata(row)
                messages.append(f"metadata: {len(rows)}")
            if prod_file:
                rows,bad=galit.production_from_csv(prod_file.getvalue().decode("utf-8-sig")); errors+=bad; messages.append(str(repository.ingest_production(rows)))
            if inj_file:
                rows,bad=galit.injection_from_csv(inj_file.getvalue().decode("utf-8-sig")); errors+=bad; messages.append(str(repository.ingest_injection(rows)))
            if messages: st.success("; ".join(messages))
            if errors: st.warning("Часть строк отклонена: " + "; ".join(errors[:20]))
        except (UnicodeDecodeError, galit.WatercutStorageError, galit.WatercutConflictError, ValueError) as exc: st.error(f"Импорт не выполнен: {exc}")
    try: results=watercut_portfolio(repository); metadata=repository.list_metadata()
    except galit.WatercutStorageError as exc: st.error(str(exc)); return
    if not results: st.info("Нет добывающих скважин с историей. Импортируйте metadata и production CSV."); return
    fields=sorted({x.field_name for x in metadata if x.field_name}); clusters=sorted({x.cluster for x in metadata if x.cluster}); reservoirs=sorted({x.reservoir for x in metadata if x.reservoir})
    f1,f2,f3,f4=st.columns(4); field_filter=f1.multiselect("Field",fields,default=fields,key="wc-field"); cluster_filter=f2.multiselect("Cluster",clusters,default=clusters,key="wc-cluster"); reservoir_filter=f3.multiselect("Reservoir",reservoirs,default=reservoirs,key="wc-reservoir"); status_filter=f4.multiselect("Status",["low","growing","critical"],default=["low","growing","critical"],key="wc-status")
    by_well={x.well:x for x in metadata}; shown=[x for x in results if x.severity in status_filter and (not fields or by_well[x.well].field_name in field_filter) and (not clusters or by_well[x.well].cluster in cluster_filter) and (not reservoirs or by_well[x.well].reservoir in reservoir_filter)]
    k=st.columns(4); k[0].metric("Добывающий фонд",len(shown)); k[1].metric("Растущие",sum(x.severity=="growing" for x in shown)); k[2].metric("Критические",sum(x.severity=="critical" for x in shown)); k[3].metric("Возможные потери нефти",f"{sum(x.possible_oil_loss_m3d or 0 for x in shown):.1f} м³/сут")
    st.dataframe(pd.DataFrame([{"Скважина":x.well,"Статус":x.severity,"Обводнённость":x.current_water_cut,"Δ п.п.":x.absolute_change_pp,"Confidence":x.confidence,"Потеря нефти":x.possible_oil_loss_m3d} for x in shown]),width="stretch",hide_index=True)
    links=list(galit.build_watercut_links(metadata,repository.list_production(),repository.list_injection(),top_n=30))
    if links: st.plotly_chart(fig_watercut_links(metadata,links),width="stretch"); st.caption("Слой показывает только top-N связей выше policy-порога; стрелка направлена injector → producer.")
    if not shown: return
    selected=st.selectbox("Карточка producer",[x.well for x in shown],key="wc-well"); item=next(x for x in shown if x.well==selected); history=repository.list_production(selected)
    graph=pd.DataFrame([{"date":x.timestamp,"water_cut":x.water_cut,"q_oil":x.q_oil_m3d,"q_water":x.q_water_m3d} for x in history]).set_index("date"); st.line_chart(graph)
    st.markdown(f"**Onset:** {item.onset_window or 'unavailable'} · **confidence:** {item.confidence} · **quality:** {item.data_quality:.0%} · **policy:** {item.policy_version}")
    if item.oil_forecast: st.dataframe(pd.DataFrame([asdict(x) for x in item.oil_forecast]),width="stretch",hide_index=True)
    st.markdown("**Evidence:** " + ", ".join(f"{k}={v:.2f}" for k,v in item.evidence.items()))
    st.dataframe(pd.DataFrame([asdict(x) for x in item.candidate_injectors]),width="stretch",hide_index=True)
    st.info("Альтернативы: " + "; ".join(item.alternative_explanations))


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


def treatment_storage_path() -> Path:
    """Resolve the configured journal path relative to the project, not process CWD."""
    configured = Path(os.environ.get("GALIT_TREATMENT_STORAGE", "data/treatments.json"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def get_treatment_repository() -> galit.TreatmentRepository:
    """Return one repository per configured path for the current Streamlit session."""
    path = str(treatment_storage_path().resolve())
    cached = st.session_state.get("treatment_repository")
    if cached is None or str(cached.path.resolve()) != path:
        cached = galit.TreatmentRepository(path)
        st.session_state["treatment_repository"] = cached
    return cached


def passport_storage_path() -> Path:
    configured = Path(os.environ.get("GALIT_PASSPORT_STORE", "data/well_passports.json"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def get_passport_repository() -> galit.PassportRepository:
    path = str(passport_storage_path().resolve())
    cached = st.session_state.get("passport_repository")
    if cached is None or str(cached.path.resolve()) != path:
        cached = galit.PassportRepository(path)
        st.session_state["passport_repository"] = cached
    return cached


def passport_event_frame(events: list[galit.PassportEvent]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Дата": item.event_at, "Тип": item.event_type.value, "Заголовок": item.title,
        "Данные": json.dumps(item.data, ensure_ascii=False), "Заметки": item.notes,
        "Вложение": item.attachment.filename if item.attachment else None,
        "Источник": item.source, "Ревизия": item.revision,
    } for item in events])


def passport_series(events: list[galit.PassportEvent], kind: galit.PassportEventType,
                    keys: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for item in sorted(events, key=lambda value: value.event_at):
        if item.event_type is kind:
            row = {"Дата": item.event_at}
            row.update({key: item.data.get(key) for key in keys})
            rows.append(row)
    return pd.DataFrame(rows)


def render_well_passport(repository: galit.PassportRepository,
                         treatments: galit.TreatmentRepository,
                         wells: list[str]) -> None:
    st.subheader("Цифровой паспорт скважины")
    st.caption("Единая UTC-хронология. Журнал мероприятий агрегируется при чтении и не дублируется.")
    manual = st.text_input("Или укажите скважину вручную", key="passport_manual_well")
    well = manual.strip() or st.selectbox("Скважина паспорта", wells or ["Скважина"], key="passport_well")
    try:
        events = repository.list(well=well)
        treatment_rows = treatments.list(well=well)
        summary = galit.passport_summary(events, treatment_rows)
    except (galit.PassportStorageError, galit.TreatmentStorageError) as exc:
        st.error(f"Паспорт недоступен: {exc}")
        return
    a, b, c, d = st.columns(4)
    a.metric("Событий паспорта", summary["event_count"])
    b.metric("Обработок", summary["treatment_count"])
    c.metric("Оценено обработок", summary["assessed_treatments"])
    d.metric("Последнее событие", summary["last_event_at"].date().isoformat() if summary["last_event_at"] else "—")

    risk = passport_series(events, galit.PassportEventType.RISK_SNAPSHOT,
                           ("integrated_risk", "wax_risk", "halite_risk", "calcite_risk", "corrosion_risk"))
    rate = passport_series(events, galit.PassportEventType.RATE_CHANGE,
                           ("oil_rate_m3d", "water_rate_m3d", "gas_rate_m3d"))
    left, right = st.columns(2)
    with left:
        st.markdown("#### Риск")
        st.info("Нет снимков риска.") if risk.empty else st.line_chart(risk.set_index("Дата"))
    with right:
        st.markdown("#### Дебит")
        st.info("Нет снимков дебита.") if rate.empty else st.line_chart(rate.set_index("Дата"))

    st.markdown("#### Единая временная шкала")
    timeline = galit.passport_timeline(events, treatment_rows)
    st.dataframe(pd.DataFrame([{
        "Дата": row["event_at"], "Источник": row["origin"], "Тип": row["event_type"],
        "Событие": row["title"], "Описание": row.get("notes"),
    } for row in timeline]), width="stretch", hide_index=True)
    categories = [galit.PassportEventType.REPAIR, galit.PassportEventType.COMPLICATION,
                  galit.PassportEventType.REAGENT_EFFECTIVENESS]
    labels = ["Ремонты", "Осложнения", "Эффективность реагентов"]
    cols = st.columns(3)
    for column, category, label in zip(cols, categories, labels):
        with column:
            st.markdown(f"#### {label}")
            frame = passport_event_frame([item for item in events if item.event_type is category])
            st.dataframe(frame, width="stretch", hide_index=True)

    st.markdown("#### Добавить событие")
    kind = st.selectbox("Тип события", [item.value for item in galit.PassportEventType], key="passport_kind")
    event_type = galit.PassportEventType(kind)
    with st.form("passport-add", clear_on_submit=True):
        title = st.text_input("Заголовок")
        notes = st.text_area("Описание / заключение")
        data_text = st.text_area("Данные JSON", value="{}", help='Например: {"oil_rate_m3d": 12.5}')
        uploaded = st.file_uploader("Фотография или лабораторный файл",
                                    type=["jpg", "jpeg", "png", "pdf", "txt", "csv"])
        submitted = st.form_submit_button("Добавить в паспорт")
    if submitted:
        attachment = None
        try:
            data = json.loads(data_text)
            if event_type in {galit.PassportEventType.DEPOSIT_PHOTO, galit.PassportEventType.LAB_REPORT}:
                if uploaded is None:
                    raise ValueError("для этого типа требуется вложение")
                attachment = repository.save_attachment(uploaded.name, uploaded.type, uploaded.getvalue())
            elif uploaded is not None:
                raise ValueError("вложения разрешены только для deposit_photo и lab_report")
            event = galit.new_passport_event(
                well_id=well, well_name=well, event_type=event_type,
                event_at=datetime.now(timezone.utc), title=title, data=data,
                notes=notes or None, source="dashboard", attachment=attachment,
            )
            repository.create(event)
            st.success("Событие добавлено.")
            st.rerun()
        except (ValueError, json.JSONDecodeError, galit.PassportStorageError) as exc:
            st.error(f"Событие не сохранено: {exc}")


def ensure_utc(value: datetime) -> datetime:
    """Normalize Streamlit datetime values to the repository's aware UTC contract."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def equipment_portfolio_frame(rows: list[galit.EquipmentForecast]) -> pd.DataFrame:
    """Stable table for equipment priorities, safe with partial/unavailable forecasts."""
    return pd.DataFrame([{
        "Скважина": row.well, "Тип": row.lift_type,
        "Baseline-риск": row.baseline_failure_risk, "Уровень": row.risk_level,
        "RUL, сут": "—" if row.rul_days is None else f"{row.rul_days[0]}–{row.rul_days[1]}",
        "Причина": row.causes[0].label if row.causes else "—",
        "Обслуживание": "—" if row.maintenance_window_start is None else
            f"{row.maintenance_window_start.date()} — {row.maintenance_window_end.date()}",
        "Confidence": row.confidence, "Полнота": row.data_completeness,
    } for row in rows])


def equipment_trend_frame(rows: list[galit.TelemetrySnapshot]) -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp": row.timestamp, "Ток, A": row.current_a,
        "Давление нагнетания": row.discharge_pressure,
        "Температура двигателя, °C": row.motor_temperature_c,
        "Вибрация, мм/с": row.vibration_mm_s,
        "Нагрузка ШГН, кН": row.rod_load_kn,
    } for row in rows])


def render_equipment_forecasts(repository: galit.EquipmentRepository) -> None:
    st.subheader("Прогноз отказов ЭЦН и ШГН")
    st.warning(galit.EQUIPMENT_DISCLAIMER)
    st.download_button("Скачать CSV-шаблон телеметрии", galit.telemetry_csv_template(),
                       "equipment-telemetry-template.csv", "text/csv", key="equipment-template")
    uploaded = st.file_uploader("Импорт телеметрии CSV", type=["csv"], key="equipment-csv")
    if uploaded is not None and st.button("Проверить и импортировать", key="equipment-import"):
        try:
            imported = 0
            for item in galit.telemetry_from_csv(uploaded.getvalue()):
                repository.ingest(item, idempotent=True); imported += 1
            st.success(f"Принято snapshots: {imported}. Повторы обработаны идемпотентно.")
        except (ValueError, galit.EquipmentConflictError, galit.EquipmentStorageError) as exc:
            st.error(f"Импорт не выполнен: {exc}")
    try:
        portfolio = repository.portfolio()
    except galit.EquipmentStorageError as exc:
        st.error(f"Хранилище оборудования недоступно: {exc}"); return
    if not portfolio:
        st.info("Нет metadata оборудования. Добавьте его через API, затем загрузите телеметрию.")
        return
    lift = st.multiselect("Тип оборудования", ["ESP", "ROD_PUMP", "UNSUPPORTED"],
                          default=["ESP", "ROD_PUMP"], key="equipment-lift-filter")
    levels = st.multiselect("Уровень риска", ["normal", "warning", "critical", "unavailable"],
                            default=["normal", "warning", "critical", "unavailable"], key="equipment-risk-filter")
    shown = [row for row in portfolio if row.lift_type in lift and row.risk_level in levels]
    cols = st.columns(4)
    cols[0].metric("Оборудование", len(shown)); cols[1].metric("Критические", sum(x.risk_level == "critical" for x in shown))
    cols[2].metric("Требуют внимания", sum(x.risk_level == "warning" for x in shown))
    cols[3].metric("Средняя полнота", f"{sum(x.data_completeness for x in shown)/len(shown):.0%}" if shown else "—")
    st.dataframe(equipment_portfolio_frame(shown), width="stretch", hide_index=True)
    if not shown: return
    selected = st.selectbox("Карточка скважины", [x.well for x in shown], key="equipment-well")
    forecast = next(x for x in shown if x.well == selected)
    left, right = st.columns([1, 2])
    with left:
        st.metric("Baseline failure risk", "—" if forecast.baseline_failure_risk is None else f"{forecast.baseline_failure_risk:.0%}")
        st.metric("RUL range", "unavailable" if forecast.rul_days is None else f"{forecast.rul_days[0]}–{forecast.rul_days[1]} сут")
        st.caption(f"Качество {forecast.data_completeness:.0%} · confidence {forecast.confidence} · {forecast.model_version}")
    with right:
        st.markdown("#### Ранжированные причины")
        st.dataframe(pd.DataFrame([{"Причина": x.label, "Группа": x.group,
                                   "Индикатор": x.indicator, "Вклад": x.contribution,
                                   "Объяснение": x.explanation} for x in forecast.causes]),
                     width="stretch", hide_index=True)
        st.markdown(f"**Рекомендуемое действие:** {forecast.recommended_action}")
    trend = equipment_trend_frame(repository.list_telemetry(selected))
    if len(trend) > 1:
        st.line_chart(trend.set_index("timestamp"))
    elif len(trend) == 1:
        st.info("Доступен один snapshot: тренды не рассчитываются, confidence снижен.")


def treatment_well_context(cases_by_name: dict[str, WellCase],
                           results: list[DiagnosisResult]) -> dict[str, dict[str, Any]]:
    """Build safe autofill context from the current calculated fund."""
    diagnosed = {item.well: item for item in results}
    context: dict[str, dict[str, Any]] = {}
    for name, case in cases_by_name.items():
        result = diagnosed.get(name)
        context[name] = {
            "well_id": name,
            "well_name": name,
            "baseline_risk": result.integrated_risk if result else None,
            "baseline_state": result.dominant if result else None,
            "complication_type": result.dominant if result else "other",
            "well_group": case.lift_type or None,
        }
    return context


def treatment_history_frame(records: list[galit.TreatmentRecord],
                            currency: str | None = None) -> pd.DataFrame:
    """Build an audit-friendly history table without mixing currencies."""
    rows = []
    for item in records:
        if currency and item.currency != currency:
            continue
        rows.append({
            "ID": item.id, "Скважина": item.well_name, "Дата": item.event_at,
            "Осложнение": item.complication_type, "Группа": item.well_group,
            "Месторождение": item.field_name, "Куст": item.cluster, "Участок": item.site,
            "Реагент": item.reagent_name, "Дозировка": f"{item.dosage:g} {item.dosage_unit}",
            "Дебит до, м³/сут": item.rate_before_m3_day,
            "Дебит после, м³/сут": item.rate_after_m3_day,
            "Стоимость": item.cost, "Валюта": item.currency,
            "Мероприятие": item.treatment_type, "Статус": item.status.value,
            "Успех": item.success, "Эффект, сут": item.effect_duration_days,
            "Повтор": item.recurrence, "Дата повтора": item.recurrence_date,
            "Архив": item.archived, "Ревизия": item.revision,
            "Источник": item.source, "Обновлено": item.updated_at,
        })
    return pd.DataFrame(rows)


def treatment_summary_frame(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten assessed-only summary and keep every currency in a separate row."""
    rows: list[dict[str, Any]] = []
    for group in summary.get("groups", []):
        costs = group.get("costs_by_currency", {})
        currencies = costs or {"—": {"total": None, "per_success": None,
                                     "per_effect_day": None}}
        for currency, values in currencies.items():
            rows.append({
                "Реагент": group["group"], "Всего записей": group["treatments"],
                "Оценено, n": group["assessed_observations"],
                "Успех": group["success_rate"], "Средний эффект, сут": group["effect_days_mean"],
                "Повтор": group["recurrence_rate"], "Confidence": group["confidence"],
                "Валюта": currency, "Затраты": values["total"],
                "Затраты / успех": values["per_success"],
                "Затраты / день эффекта": values["per_effect_day"],
            })
    return pd.DataFrame(rows)


def _journal_error(exc: Exception) -> None:
    if isinstance(exc, galit.TreatmentConflictError):
        st.error("Конфликт ревизии: запись уже изменилась. Обновите экран и повторите действие.")
    elif isinstance(exc, galit.TreatmentNotFoundError):
        st.error("Запись больше не найдена. Обновите экран.")
    elif isinstance(exc, galit.TreatmentStorageError):
        st.error(f"Хранилище журнала повреждено или недоступно: {exc}")
    else:
        st.error(f"Не удалось сохранить запись: {exc}")


def render_treatment_journal(repository: galit.TreatmentRepository,
                             well_context: dict[str, dict[str, Any]]) -> None:
    """Create, progress, assess, archive, filter and aggregate journal records."""
    st.subheader("Журнал мероприятий и фактического эффекта")
    st.caption("План и факт разделены. В KPI и A/B входят только assessed-записи.")
    try:
        records = repository.list(include_archived=True)
    except galit.TreatmentStorageError as exc:
        _journal_error(exc)
        st.info("Исправьте или удалите повреждённый JSON-файл хранилища, затем обновите экран.")
        return

    section = st.radio(
        "Раздел журнала", ["Новое мероприятие", "Lifecycle и результат", "История", "Сводка и A/B"],
        horizontal=True, key="treatment_section",
    )
    active_records = [item for item in records if not item.archived]

    if section == "Новое мероприятие":
        known = sorted(well_context)
        mode = st.radio("Скважина", ["Из рассчитанного фонда", "Указать вручную"],
                        horizontal=True, disabled=not known, key="treatment_well_mode")
        selected = st.selectbox("Существующая скважина", known, key="treatment_known_well") if known and mode == "Из рассчитанного фонда" else None
        defaults = well_context.get(selected or "", {})
        if defaults:
            st.caption("Контекст подставлен из текущего расчёта: риск, механизм и группа эксплуатации.")
        with st.form("treatment-create", clear_on_submit=True):
            well_name = selected or st.text_input("Название скважины")
            well_id = st.text_input("ID скважины", value=str(defaults.get("well_id", well_name)))
            event_at = st.datetime_input("Дата и время события", value=datetime.now(timezone.utc))
            complication_values = ["wax", "halite", "calcite", "corrosion", "other"]
            default_complication = defaults.get("complication_type", "other")
            complication = st.selectbox("Осложнение", complication_values,
                                        index=complication_values.index(default_complication))
            description = st.text_area("Описание события")
            reagent_col, id_col = st.columns(2)
            reagent = reagent_col.text_input("Реагент (название)")
            reagent_id = id_col.text_input("ID реагента (необязательно)")
            dosage_col, cost_col = st.columns(2)
            dosage = dosage_col.number_input("Дозировка", min_value=0.0)
            dosage_unit = dosage_col.selectbox("Единица", sorted(galit.treatments.VALID_DOSAGE_UNITS))
            cost = cost_col.number_input("Стоимость", min_value=0.0)
            currency = cost_col.selectbox("Валюта", sorted(galit.treatments.VALID_CURRENCIES))
            treatment_type = st.text_input("Тип обработки")
            loc1, loc2, loc3 = st.columns(3)
            field_name = loc1.text_input("Месторождение")
            cluster = loc2.text_input("Куст")
            site = loc3.text_input("Участок")
            rate_before = st.number_input("Дебит до, м³/сут", min_value=0.0)
            expected = st.text_area("Ожидаемый результат (не факт)")
            baseline_risk = st.number_input("Исходный риск", 0.0, 1.0,
                                            value=float(defaults.get("baseline_risk") or 0.0))
            baseline_state = st.text_input("Исходное состояние",
                                           value=str(defaults.get("baseline_state") or ""))
            well_group = st.text_input("Группа скважин",
                                       value=str(defaults.get("well_group") or ""))
            source = st.text_input("Источник", value="dashboard")
            comment = st.text_area("Комментарий")
            submitted = st.form_submit_button("Сохранить план")
        if submitted:
            try:
                created = galit.new_treatment(
                    well_id=well_id, well_name=well_name, event_at=ensure_utc(event_at),
                    complication_type=complication, description=description,
                    reagent_name=reagent, reagent_id=reagent_id or None, dosage=dosage,
                    dosage_unit=dosage_unit, cost=cost, currency=currency,
                    treatment_type=treatment_type, field_name=field_name or None,
                    cluster=cluster or None, site=site or None,
                    rate_before_m3_day=rate_before,
                    baseline_risk=baseline_risk,
                    baseline_state=baseline_state or None, expected_result=expected or None,
                    source=source, well_group=well_group or None, comment=comment or None,
                )
                repository.create(created)
                st.success(f"Запись создана · revision {created.revision}")
            except (ValueError, galit.TreatmentStorageError, galit.TreatmentConflictError) as exc:
                _journal_error(exc)

    elif section == "Lifecycle и результат":
        candidates = [item for item in active_records if item.status is not galit.TreatmentStatus.ASSESSED]
        if not candidates:
            st.info("Нет незавершённых записей.")
        else:
            selected_id = st.selectbox(
                "Запись", [item.id for item in candidates],
                format_func=lambda value: next(
                    f"{item.well_name} · {item.reagent_name} · {item.status.value} · r{item.revision}"
                    for item in candidates if item.id == value), key="treatment_lifecycle_record",
            )
            item = next(row for row in candidates if row.id == selected_id)
            st.json({"revision": item.revision, "expected_result": item.expected_result,
                     "baseline_risk": item.baseline_risk, "baseline_state": item.baseline_state,
                     "updated_at": item.updated_at.isoformat()}, expanded=False)
            next_status = next(iter(galit.treatments.ALLOWED_TRANSITIONS[item.status]), None)
            if next_status in {galit.TreatmentStatus.IN_PROGRESS, galit.TreatmentStatus.COMPLETED}:
                label = "Начать" if next_status is galit.TreatmentStatus.IN_PROGRESS else "Завершить"
                if st.button(f"{label} мероприятие", key="treatment_progress"):
                    try:
                        repository.update(item.transition(next_status), expected_revision=item.revision)
                        st.success(f"Статус изменён на {next_status.value}.")
                        st.rerun()
                    except (ValueError, galit.TreatmentStorageError, galit.TreatmentConflictError,
                            galit.TreatmentNotFoundError) as exc:
                        _journal_error(exc)
            elif next_status is galit.TreatmentStatus.ASSESSED:
                with st.form("treatment-assess"):
                    actual = st.text_area("Фактический результат")
                    rate_after = st.number_input("Дебит после, м³/сут", min_value=0.0)
                    metric_name = st.text_input("Измеримый показатель", value="oil_rate_delta_m3_day")
                    metric_value = st.number_input("Значение показателя", min_value=0.0)
                    success_choice = st.selectbox("Успех", ["да", "нет"])
                    days = st.number_input("Длительность эффекта, сут", min_value=0.0)
                    recurrence_choice = st.selectbox("Повтор осложнения", ["нет", "да"])
                    recurrence_date = st.datetime_input("Дата повтора", value=datetime.now(timezone.utc),
                                                        disabled=recurrence_choice == "нет")
                    comment = st.text_area("Комментарий", value=item.comment or "")
                    assess = st.form_submit_button("Зафиксировать assessed")
                if assess:
                    try:
                        recurrence = recurrence_choice == "да"
                        updated = item.transition(
                            galit.TreatmentStatus.ASSESSED, actual_result=actual,
                            result_metrics={metric_name: metric_value}, success=success_choice == "да",
                            rate_after_m3_day=rate_after,
                            effect_duration_days=days, recurrence=recurrence,
                            recurrence_date=ensure_utc(recurrence_date) if recurrence else None,
                            comment=comment or None,
                        )
                        saved = repository.update(updated, expected_revision=item.revision)
                        st.success(f"Результат сохранён · revision {saved.revision}.")
                    except (ValueError, galit.TreatmentStorageError, galit.TreatmentConflictError,
                            galit.TreatmentNotFoundError) as exc:
                        _journal_error(exc)

    elif section == "История":
        include_archived = st.checkbox("Показывать архив", key="treatment_include_archived")
        visible = records if include_archived else active_records
        wells = sorted({item.well_name for item in visible})
        statuses = sorted({item.status.value for item in visible})
        complications = sorted({item.complication_type for item in visible})
        f1, f2, f3 = st.columns(3)
        selected_well = f1.selectbox("Скважина", ["Все", *wells], key="treatment_history_well")
        selected_status = f2.selectbox("Статус", ["Все", *statuses], key="treatment_history_status")
        selected_complication = f3.selectbox("Осложнение", ["Все", *complications], key="treatment_history_complication")
        currencies = sorted({item.currency for item in visible})
        selected_currency = st.selectbox("Валюта", ["Раздельно", *currencies], key="treatment_history_currency")
        visible = [item for item in visible
                   if (selected_well == "Все" or item.well_name == selected_well)
                   and (selected_status == "Все" or item.status.value == selected_status)
                   and (selected_complication == "Все" or item.complication_type == selected_complication)]
        st.dataframe(treatment_history_frame(
            visible, None if selected_currency == "Раздельно" else selected_currency),
            width="stretch", hide_index=True,
        )
        archivable = [item for item in visible if not item.archived]
        if archivable:
            archive_id = st.selectbox("Запись для архива", [item.id for item in archivable],
                                      format_func=lambda value: next(
                                          f"{item.well_name} · {item.reagent_name} · r{item.revision}"
                                          for item in archivable if item.id == value))
            archive_item = next(item for item in archivable if item.id == archive_id)
            if st.button("Архивировать запись", key="treatment_archive"):
                try:
                    repository.archive(archive_id, expected_revision=archive_item.revision)
                    st.success("Запись перенесена в архив.")
                    st.rerun()
                except (galit.TreatmentStorageError, galit.TreatmentConflictError,
                        galit.TreatmentNotFoundError) as exc:
                    _journal_error(exc)
        st.caption("Валюты не конвертируются; ID, revision, источник и timestamps сохранены для аудита.")

    else:
        assessed = [item for item in active_records if item.status is galit.TreatmentStatus.ASSESSED]
        summary = galit.treatment_summary(assessed, "reagent")
        if not assessed:
            st.info("insufficient_data: нет assessed-наблюдений для KPI.")
        else:
            summary_frame = treatment_summary_frame(summary)
            st.dataframe(summary_frame.style.format({"Успех": "{:.1%}", "Повтор": "{:.1%}"},
                                                     na_rep="—"), width="stretch", hide_index=True)
            for currency in sorted({item.currency for item in assessed}):
                same = [item for item in assessed if item.currency == currency]
                st.metric(f"Затраты assessed · {currency}", f"{sum(item.cost for item in same):,.2f}",
                          help=f"n={len(same)}; валюты не суммируются между собой")
        complications = sorted({item.complication_type for item in assessed})
        groups = sorted({item.well_group for item in assessed if item.well_group})
        reagents = sorted({item.reagent_name for item in assessed})
        if len(reagents) < 2 or not complications or not groups:
            st.info("Для A/B нужны assessed-записи двух реагентов с одинаковыми complication и well_group.")
        else:
            c1, c2 = st.columns(2)
            reagent_a = c1.selectbox("Реагент A", reagents, index=0)
            reagent_b = c2.selectbox("Реагент B", reagents, index=1)
            complication = c1.selectbox("Сопоставимое осложнение", complications)
            well_group = c2.selectbox("Сопоставимая группа скважин", groups)
            metric = st.selectbox("Метрика", ["success_rate", "mean_effect_days"])
            min_n = st.number_input("Минимум наблюдений на реагент", min_value=2,
                                    value=galit.DEFAULT_MIN_SAMPLE_SIZE)
            try:
                comparison = galit.compare_reagents(
                    assessed, reagent_a, reagent_b, metric=metric, min_sample_size=int(min_n),
                    complication_type=complication, well_group=well_group,
                )
                a, b = comparison["reagent_a"], comparison["reagent_b"]
                st.write(f"A: n={a['n']}, value={a['value'] if a['value'] is not None else '—'} · "
                         f"B: n={b['n']}, value={b['value'] if b['value'] is not None else '—'}")
                if comparison["status"] == "available":
                    st.metric("Relative uplift B vs A", f"{comparison['relative_uplift']:.1%}")
                    st.success(f"Confidence: {comparison['confidence']}")
                else:
                    st.warning(f"insufficient_data: {comparison['reason']} · confidence={comparison['confidence']}")
                st.warning(comparison["warning"])
            except ValueError as exc:
                st.error(f"A/B сравнение недоступно: {exc}")


def twin_storage_path() -> Path:
    configured = Path(os.environ.get("GALIT_TWIN_EVENT_STORAGE", "data/digital_twin_events.json"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def equipment_storage_path() -> Path:
    configured = Path(os.environ.get("GALIT_EQUIPMENT_STORAGE", "data/equipment.json"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


class DashboardFundTwinAdapter:
    """Expose loaded fund identities without persisting or inventing timeline events."""
    def __init__(self, cases: list[WellCase]):
        self.cases = tuple(cases)

    def identities(self):
        for case in self.cases:
            yield galit.WellIdentity(case.name, cluster=case.cluster, site=case.site)

    def events(self):
        return ()


def get_twin_components(cases: list[WellCase] | None = None) -> tuple[galit.DigitalTwinService, galit.ManualEventRepository]:
    """Build the aggregate from configured repositories and the loaded fund."""
    manual = galit.ManualEventRepository(twin_storage_path())
    service = galit.build_default_service(
        watercut=get_watercut_repository(),
        equipment=galit.EquipmentRepository(equipment_storage_path()),
        treatments=get_treatment_repository(),
        passports=get_passport_repository(),
        manual=manual,
    )
    if cases:
        service = galit.DigitalTwinService((*service.adapters, DashboardFundTwinAdapter(cases)), service.policy)
    return service, manual


def twin_timeline_categories() -> list[str]:
    """Stable category order shared by the filter and regression tests."""
    return [category.value for category in galit.EventCategory]


def fig_twin_timeline(items: list[dict[str, Any]]) -> go.Figure:
    """Compact, deterministic timeline chart; annotations never contain raw HTML."""
    colors = {"production": GREEN_700, "watercut": "#1665A5", "repair": STATUS_WARN,
              "equipment_failure": STATUS_CRIT, "equipment_telemetry": GREEN_500,
              "treatment": "#7C3AED", "laboratory": "#0E7490", "complication": STATUS_HIGH,
              "economic_loss": "#9A3412", "pressure_temperature": "#475569"}
    ordered = list(reversed(items)); fig = go.Figure()
    for index, item in enumerate(ordered):
        fig.add_trace(go.Scatter(x=[item["occurred_at"]], y=[index], mode="markers+text",
            marker={"size": 12, "color": colors.get(item["category"], INK_MUTED)},
            text=[item["title"]], textposition="middle right", name=item["category"],
            hovertext=[item["summary"]], hovertemplate="%{x}<br>%{hovertext}<extra></extra>"))
    fig.update_layout(height=max(320, 52 * len(ordered)), showlegend=False,
                      margin={"l": 20, "r": 20, "t": 10, "b": 20}, yaxis={"visible": False})
    return fig


def render_digital_twin(service: galit.DigitalTwinService,
                        repository: galit.ManualEventRepository) -> None:
    st.subheader("Цифровой двойник фонда скважин")
    st.caption("Единая история агрегирует существующие источники; ручное хранилище содержит только новые события.")
    wells = service.list_wells()
    if not wells:
        st.info("Источники пока пусты. Добавьте ручное событие ниже или загрузите данные в профильные разделы.")
        selected_name = st.text_input("Скважина", key="twin-empty-well")
    else:
        labels = [f"{x['display_name']} · {x.get('field') or x.get('site') or 'контекст не задан'}" for x in wells]
        selected_label = st.selectbox("Скважина", labels, key="twin-well")
        selected_name = wells[labels.index(selected_label)]["canonical_id"]
    available_categories = twin_timeline_categories()
    categories = st.multiselect("Категории", available_categories,
                                default=available_categories, key="twin-categories")
    days = st.slider("Период, суток", 7, 1095, 180, key="twin-days")
    if selected_name:
        try:
            end = datetime.now(timezone.utc); start = end - timedelta(days=days)
            snapshot = service.snapshot(selected_name, as_of=end)
            timeline = service.timeline(selected_name, date_from=start, date_to=end,
                                        categories=categories, limit=500)
            a, b, c, d = st.columns(4)
            a.metric("Состояние", snapshot.state)
            b.metric("Health screening", "—" if snapshot.health_score is None else f"{snapshot.health_score:.0%}")
            c.metric("Событий", timeline["total"])
            d.metric("Stale sources", len(snapshot.stale_sources))
            st.caption("Freshness: " + (", ".join(snapshot.stale_sources) or "актуальные датированные источники") +
                       " · Missing: " + (", ".join(snapshot.missing_sources) or "нет"))
            if timeline["items"]: st.plotly_chart(fig_twin_timeline(timeline["items"]), width="stretch")
            else: st.info("В выбранном периоде событий нет.")
            st.markdown("### Почему изменилось состояние")
            changes = service.changes(selected_name, as_of=end, limit=20)
            if not changes: st.info("Нет значимых датированных событий для before/after сопоставления.")
            for change in changes:
                with st.container(border=True):
                    st.markdown(f"**{html.escape(change.title)}** · confidence: `{change.confidence}`")
                    st.write(change.statement)
                    st.json({"before": change.before, "after": change.after}, expanded=False)
                    st.caption("Альтернативы: " + "; ".join(change.alternative_explanations))
            st.warning(galit.ASSOCIATION_DISCLAIMER)
        except (galit.TwinNotFoundError, galit.TwinAmbiguousError, galit.TwinStorageError, ValueError) as exc:
            st.error(html.escape(str(exc)))
    st.download_button("CSV-шаблон ручных событий", galit.manual_csv_template(),
                       "digital-twin-events.csv", "text/csv", key="twin-template")
    with st.form("twin-manual-event", clear_on_submit=True):
        well = st.text_input("Скважина события", value="" if not wells else wells[0]["display_name"])
        field_name = st.text_input("Месторождение / field")
        occurred = st.text_input("Время ISO 8601 с timezone", value=datetime.now(timezone.utc).isoformat())
        category = st.selectbox("Тип", [galit.EventCategory.REPAIR.value,
            galit.EventCategory.EQUIPMENT_FAILURE.value, galit.EventCategory.LABORATORY.value])
        title = st.text_input("Заголовок")
        summary = st.text_area("Описание")
        source_id = st.text_input("Внешний ID (для идемпотентности)")
        submitted = st.form_submit_button("Сохранить событие")
    if submitted:
        try:
            event = galit.manual_event(well=galit.WellIdentity(well, field_name or None),
                occurred_at=datetime.fromisoformat(occurred.replace("Z", "+00:00")), category=category,
                event_type=category, title=title, summary=summary,
                source_record_id=source_id or None, metadata={"entered_via": "streamlit"})
            repository.add(event); st.success("Событие сохранено идемпотентно.")
        except (ValueError, galit.TwinConflictError, galit.TwinStorageError) as exc: st.error(html.escape(str(exc)))


def _compatibility_note(text: str) -> str:
    """Перевести фиксированные пояснения ядра, не меняя их смысл."""
    translations = {
        "conservative-volume linear ion mixing": "линейное смешение ионов при сохранении объёма",
        "pH mixed from hydrogen-ion activities without buffering/speciation": "pH смешивается по активности H⁺ без учёта буферности и полного химического равновесия",
        "no reaction or precipitation depletion during mixing": "реакции и выпадение осадка в процессе смешения не уменьшают концентрации",
        "unsafe means at least one available SI > 0": "небезопасно означает: хотя бы для одного рассчитанного минерала SI > 0",
        "insufficient supplied chemistry": "недостаточно переданных результатов химического анализа",
        "Supersaturation indicates thermodynamic tendency, not deposition rate or deposited mass.": "Пересыщение показывает термодинамическую возможность, а не скорость или массу отложений.",
        "simplified NaCl activity model; rigorous brine decisions require validated Pitzer modelling": "Упрощённая модель активности NaCl; для решения по рассолам нужна валидированная модель Питцера.",
        "Davies activity model is outside its usual I<=0.5 mol/L range; use a validated Pitzer/speciation model": "Модель Дэвиса применена вне обычного диапазона I ≤ 0,5 моль/л; нужна валидированная модель Питцера/специации.",
        "25 C Ksp used outside the 5..50 C screening range; temperature correction is not available": "Ksp при 25 °C применён вне диапазона screening 5…50 °C; температурная поправка недоступна.",
    }
    return translations.get(text, text)


def render_compatibility_result(result: galit.CompatibilityResult) -> None:
    """Показать инженерный screening совместимости понятными блоками."""
    dangerous = "—" if result.dangerous_fraction_b is None else f"{result.dangerous_fraction_b:.0%} воды B"
    c1, c2, c3 = st.columns(3)
    c1.metric("Наиболее опасная смесь", dangerous)
    c2.metric("Соотношение A:B", result.dangerous_ratio_a_to_b or "—")
    c3.metric("Максимальный SI", "—" if result.dangerous_risk_score is None else f"{result.dangerous_risk_score:.3f}")
    if result.dangerous_risk_score is not None and result.dangerous_risk_score > 0:
        st.error("В наиболее опасной смеси есть термодинамическое пересыщение (SI > 0). Не смешивайте воды без инженерной и лабораторной проверки.")
    else:
        st.success("На рассчитанной сетке положительный SI не выявлен. Это screening, а не гарантия отсутствия отложений.")

    st.plotly_chart(fig_compatibility_ratios(result), width="stretch",
                    config={"displaylogo": False, "responsive": True})
    st.caption("SI = log₁₀(IAP/Ksp): выше нуля — пересыщение; пустая линия означает, что нужные ионы не были предоставлены.")

    if result.unsafe_intervals:
        intervals = ", ".join(
            f"{item.start_fraction_b:.0%}…{item.end_fraction_b:.0%} воды B"
            for item in result.unsafe_intervals
        )
        st.warning("Небезопасные интервалы на выбранном шаге: " + intervals)
    else:
        st.info("Небезопасные интервалы на выбранном шаге не найдены.")

    if result.deposition_locations:
        location_rows = [{
            "Минерал": COMPATIBILITY_MINERAL_RU[item.mineral],
            "Первое пересыщение по ходу потока, м": item.first_supersaturation_depth_m,
            "Глубина максимального SI, м": item.maximum_risk_depth_m,
            "Максимальный SI": item.maximum_saturation_index,
        } for item in result.deposition_locations]
        st.markdown("#### Вероятная зона проявления по заданному профилю")
        st.dataframe(pd.DataFrame(location_rows), width="stretch", hide_index=True)
        st.caption("Это вероятная зона первого пересыщения по направлению потока, а не прогноз массы осадка.")
    else:
        st.info("Место возможного выпадения не рассчитано: добавьте измеренный профиль глубина–температура–давление.")

    inhibitor = result.inhibitor
    st.markdown("#### Ингибитор")
    if inhibitor.status == "dose_from_validated_curve" and inhibitor.dose_mg_l is not None:
        st.success(
            f"По введённой валидированной кривой: не менее {inhibitor.dose_mg_l:g} мг/л продукта "
            f"«{inhibitor.product}» для минерала {COMPATIBILITY_MINERAL_RU.get(inhibitor.mineral or '', inhibitor.mineral)}."
        )
        st.caption(f"Основание валидации: {inhibitor.validation_reference}. Интерполяция и экстраполяция не выполняются.")
    else:
        st.warning("Требуется лабораторный тест: без валидированной кривой «доза–поддерживаемый SI» ядро не назначает дозу.")
        st.caption(inhibitor.basis)

    with st.expander("Допущения, предупреждения и качество данных"):
        st.markdown("**Допущения модели**")
        for item in result.assumptions:
            st.write("• " + _compatibility_note(item))
        st.markdown("**Предупреждения**")
        for item in result.warnings:
            st.warning(_compatibility_note(item))
        missing = []
        for mineral in COMPATIBILITY_MINERAL_RU:
            if all(row.minerals[mineral].saturation_index is None for row in result.ratios):
                required = ", ".join(result.ratios[0].minerals[mineral].required_inputs)
                missing.append(f"{COMPATIBILITY_MINERAL_RU[mineral]}: нужны {required}")
        if missing:
            st.error("Не рассчитано из-за отсутствующих измерений: " + "; ".join(missing))
        st.caption(f"Версия модели: {result.model_version}. Концентрации — мг/л, давление — Па.")


def render_compatibility_section() -> None:
    """Независимый additive-раздел: не использует и не меняет файл фонда."""
    with st.expander("Совместимость двух вод", expanded=False):
        st.markdown("### Проверка совместимости двух измеренных вод")
        st.info("Введите только лабораторные данные. Типовой или синтетический состав здесь никогда не подставляется; пустой ион означает «нет данных», а не ноль.")
        st.download_button(
            "Скачать двухстрочный шаблон XLSX", compatibility_template_bytes(),
            "galit_compatibility_two_waters.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="compatibility-template",
        )
        upload_bytes = st.session_state.get("compatibility-upload-bytes")
        upload_name = st.session_state.get("compatibility-upload-name")
        if upload_bytes and upload_name:
            try:
                water_frame = read_table(upload_bytes, upload_name)
                st.success(f"Используется файл «{upload_name}» из боковой панели.")
                st.dataframe(water_frame, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Не удалось прочитать файл двух вод: {exc}")
                water_frame = compatibility_template_frame()
        else:
            st.caption("Или заполните две строки вручную. Обязательны name, pH, t_c и p_pa; ионы — в мг/л.")
            water_frame = st.data_editor(
                compatibility_template_frame(), width="stretch", hide_index=True,
                num_rows="fixed", key="compatibility-waters-editor",
            )

        step_percent = st.select_slider(
            "Шаг перебора доли воды B", options=[1, 2, 5, 10], value=1,
            format_func=lambda value: f"{value}%",
            help="Меньший шаг точнее показывает узкие опасные интервалы.",
        )
        use_profile = st.checkbox("Рассчитать вероятную глубину проявления по измеренному профилю", key="compatibility-use-profile")
        profile = None
        flow_direction = "bottom_to_surface"
        if use_profile:
            flow_label = st.radio(
                "Направление потока", ["От забоя к устью", "От устья к забою"],
                horizontal=True, key="compatibility-flow",
            )
            flow_direction = "bottom_to_surface" if flow_label == "От забоя к устью" else "surface_to_bottom"
            profile = st.data_editor(
                pd.DataFrame([
                    {"depth_m": 0.0, "t_c": pd.NA, "p_pa": pd.NA},
                    {"depth_m": pd.NA, "t_c": pd.NA, "p_pa": pd.NA},
                ]), num_rows="dynamic", hide_index=True, width="stretch",
                key="compatibility-profile-editor",
            )

        use_curve = st.checkbox("У меня есть валидированная лабораторная кривая ингибитора", key="compatibility-use-curve")
        curve_fields = None
        if use_curve:
            a, b, c = st.columns(3)
            product = a.text_input("Продукт", key="compatibility-product")
            mineral_label = b.selectbox("Минерал", list(COMPATIBILITY_MINERAL_RU.values()), key="compatibility-mineral")
            reference = c.text_input("Номер отчёта / ссылка", key="compatibility-reference")
            validated = st.checkbox(
                "Подтверждаю: кривая валидирована для этих вод и условий",
                key="compatibility-validated",
            )
            curve_frame = st.data_editor(
                pd.DataFrame([
                    {"dose_mg_l": pd.NA, "maximum_supported_si": pd.NA},
                    {"dose_mg_l": pd.NA, "maximum_supported_si": pd.NA},
                ]), num_rows="dynamic", hide_index=True, width="stretch",
                key="compatibility-curve-editor",
            )
            mineral = next(key for key, value in COMPATIBILITY_MINERAL_RU.items() if value == mineral_label)
            curve_fields = (product, mineral, reference, validated, curve_frame)

        if st.button("Рассчитать совместимость", type="primary", key="compatibility-calculate"):
            try:
                water_a, water_b = compatibility_waters_from_frame(water_frame)
                profile_points = compatibility_profile_from_frame(profile) if profile is not None else None
                curve = compatibility_curve_from_frame(*curve_fields) if curve_fields is not None else None
                result = galit.evaluate_compatibility(
                    water_a, water_b,
                    fractions_b=galit.default_mix_fractions(step_percent / 100),
                    profile=profile_points, flow_direction=flow_direction,
                    dose_response=curve,
                )
                st.session_state["compatibility-result"] = result
            except (ValueError, TypeError) as exc:
                st.session_state.pop("compatibility-result", None)
                st.error(f"Проверьте входные данные: {exc}")
        result = st.session_state.get("compatibility-result")
        if result is not None:
            render_compatibility_result(result)


# ==========================================================================
# Feature 9: реагенты и склад (независимый evidence-gated контур)
# ==========================================================================

CHEMICAL_HAZARD_RU = {
    "halite": "Галит", "calcite": "Кальцит", "wax": "АСПО",
    "corrosion": "Коррозия", "barite": "Барит", "gypsum": "Гипс",
}


def chemical_storage_path() -> Path:
    """Storage is explicit and independent from diagnosis/treatment repositories."""
    return Path(os.environ.get("GALIT_CHEMICAL_STORAGE", PROJECT_ROOT / "data" / "chemicals.json"))


def get_chemical_repository() -> galit.ChemicalRepository:
    return galit.ChemicalRepository(chemical_storage_path())


def _chemical_number(value: Any, digits: int = 3) -> str:
    """Format a domain decimal without turning missing values into zero."""
    if value is None:
        return "не определено"
    number = Decimal(str(value))
    rendered = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def chemical_catalog_frame(products: list[galit.ChemicalProduct],
                           envelopes: list[galit.ChemicalDoseResponseEnvelope]) -> pd.DataFrame:
    """Pure catalog adapter; validation status comes only from stored evidence."""
    evidence = {product.id: [item for item in envelopes if item.product_id == product.id]
                for product in products}
    return pd.DataFrame([{
        "ID": product.id,
        "Продукт": product.name,
        "Производитель": product.manufacturer,
        "Назначение": ", ".join(CHEMICAL_HAZARD_RU.get(x, x) for x in product.hazards),
        "Цена за кг": None if product.price_per_kg is None else float(product.price_per_kg),
        "Валюта": product.currency,
        "Активен": product.active,
        "Конвертов эффективности": len(evidence[product.id]),
        "Валидированных": sum(item.validated for item in evidence[product.id]),
    } for product in products])


def chemical_points_from_frame(frame: pd.DataFrame) -> tuple[galit.ChemicalDoseResponsePoint, ...]:
    """Strict import adapter: every row is an explicit tested dose and outcome."""
    aliases = {str(column).strip().lower(): column for column in frame.columns}
    dose_column = aliases.get("dose_kg_m3") or aliases.get("dose_mg_l")
    effective_column = aliases.get("effective")
    if dose_column is None or effective_column is None:
        raise ValueError("нужны колонки dose_kg_m3 (или dose_mg_l) и effective")
    points = []
    for index, row in frame.iterrows():
        raw_effective = row[effective_column]
        if isinstance(raw_effective, bool):
            effective = raw_effective
        else:
            token = str(raw_effective).strip().lower()
            if token not in {"true", "false", "1", "0", "да", "нет"}:
                raise ValueError(f"строка {index + 2}: effective должен быть true/false")
            effective = token in {"true", "1", "да"}
        points.append(galit.ChemicalDoseResponsePoint(
            galit.dose_to_kg_m3(row[dose_column], "mg/L" if str(dose_column).lower() == "dose_mg_l" else "kg/m3"),
            effective,
        ))
    if not points:
        raise ValueError("таблица испытаний пуста")
    return tuple(points)


def chemical_candidate_rows(products: list[galit.ChemicalProduct],
                            envelopes: list[galit.ChemicalDoseResponseEnvelope],
                            hazards: list[str], treated_m3_day: Any,
                            oil_m3_day: Any) -> list[dict[str, Any]]:
    """Explain every active candidate, including rejected products and missing data."""
    wanted = tuple(sorted({str(item).strip().lower() for item in hazards if str(item).strip()}))
    recommendations = {
        item.product_id: item for item in galit.recommend_products(
            products, envelopes, wanted, treated_m3_day, oil_m3_day
        ) if item.status == "available" and item.product_id
    }
    rows: list[dict[str, Any]] = []
    for product in products:
        if not product.active:
            continue
        reasons: list[str] = []
        missing_hazards = sorted(set(wanted) - set(product.hazards))
        if missing_hazards:
            reasons.append("не предназначен для: " + ", ".join(missing_hazards))
        selected = []
        for hazard in wanted:
            matches = [item for item in envelopes if item.product_id == product.id
                       and item.hazard == hazard and item.validated
                       and item.validation_reference and item.minimum_effective_dose is not None]
            if not matches:
                reasons.append(f"нет валидированного эффективного испытания: {hazard}")
            else:
                selected.append(min(matches, key=lambda item: (item.minimum_effective_dose, item.id)))
        if len(wanted) > 1 and not set(wanted).issubset(product.compatible_with):
            reasons.append("совместное применение для всех опасностей не подтверждено")
        recommendation = recommendations.get(product.id)
        evidence = recommendation.evidence_ids if recommendation else tuple(item.id for item in selected)
        references = [item.validation_reference for item in envelopes if item.id in evidence]
        rows.append({
            "Статус": "подходит" if recommendation else "отклонён",
            "Продукт": product.name,
            "Опасности": ", ".join(CHEMICAL_HAZARD_RU.get(x, x) for x in wanted) or "не заданы",
            "Минимальная испытанная доза, кг/м³": None if recommendation is None else float(recommendation.dose_kg_m3),
            "Основа дозы": "масса реагента / объём обрабатываемой жидкости",
            "Расход, кг/сут": None if recommendation is None else float(recommendation.daily_consumption_kg),
            "Стоимость/сут": None if recommendation is None or recommendation.daily_cost is None else float(recommendation.daily_cost),
            "Стоимость на м³ нефти": None if recommendation is None or recommendation.cost_per_m3_oil is None else float(recommendation.cost_per_m3_oil),
            "Валюта": product.currency,
            "Доказательства": ", ".join(evidence) or "нет",
            "Ссылки": "; ".join(str(x) for x in references if x) or "нет",
            "Причина / недостающие данные": "; ".join(reasons) if reasons else (
                "стоимость не определена: нет цены" if product.price_per_kg is None else
                "стоимость на м³ нефти не определена: дебит нефти равен нулю" if Decimal(str(oil_m3_day)) == 0 else "—"
            ),
        })
    if not rows:
        rows.append({"Статус": "нет кандидатов", "Продукт": None,
                     "Причина / недостающие данные": "каталог активных продуктов пуст"})
    return rows


def chemical_inventory_frame(repository: galit.ChemicalRepository, *, as_of: date) -> pd.DataFrame:
    """Pure projection over append-only ledger: physical, reserved, available, expiry."""
    transactions = repository.list_transactions()
    reservations = repository.list_reservations()
    rows = []
    for lot in repository.list_lots():
        on_hand = sum((item.signed_quantity for item in transactions if item.lot_id == lot.id), Decimal(0))
        reserved = sum((quantity for item in reservations if item.status == "active"
                        for lot_id, quantity in item.allocations if lot_id == lot.id), Decimal(0))
        expired = lot.expires_on < as_of
        available = Decimal(0) if expired else max(Decimal(0), on_hand - reserved)
        rows.append({
            "Партия": lot.id, "Продукт ID": lot.product_id,
            "Годен до": lot.expires_on.isoformat(), "Просрочена": expired,
            "На складе, кг": float(on_hand), "Зарезервировано, кг": float(reserved),
            "Доступно, кг": float(available),
        })
    return pd.DataFrame(rows)


def chemical_consumption_history(transactions: list[galit.StockTransaction],
                                 product_id: str) -> list[tuple[date, Decimal]]:
    """Aggregate immutable consumption entries by UTC calendar day."""
    totals: dict[date, Decimal] = {}
    for item in transactions:
        if item.product_id == product_id and item.kind == "consumption":
            day = item.occurred_at.date()
            totals[day] = totals.get(day, Decimal(0)) + item.quantity_kg
    return sorted(totals.items())


def chemical_forecast_view(repository: galit.ChemicalRepository, product_id: str, *,
                           as_of: date, horizon_days: int, lead_time_days: int,
                           safety_stock_days: int) -> dict[str, Any]:
    history = chemical_consumption_history(repository.list_transactions(), product_id)
    forecast = galit.deterministic_consumption_forecast(
        history, horizon_days=horizon_days, as_of=as_of,
    )
    if forecast["status"] != "available":
        return {"forecast": forecast, "shortage": None,
                "assumptions": "Истории списаний нет; нулевой расход не предполагается."}
    stock = repository.stock(product_id, as_of=as_of)
    shortage = galit.shortage_report(
        stock["available_kg"], forecast["daily_kg"], lead_time_days=lead_time_days,
        safety_stock_days=safety_stock_days, as_of=as_of,
    )
    return {
        "forecast": forecast, "shortage": shortage,
        "assumptions": ("Детерминированное среднее по календарным дням от первого списания; "
                        "тренд, сезонность и будущие поступления не моделируются. "
                        "Просроченные и зарезервированные остатки исключены."),
    }


def fig_chemical_stock(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_bar(x=frame["Партия"], y=frame["Доступно, кг"], name="Доступно", marker_color=GREEN_700)
        fig.add_bar(x=frame["Партия"], y=frame["Зарезервировано, кг"], name="Резерв", marker_color=STATUS_WARN)
    fig.update_layout(barmode="stack", height=280, margin=dict(l=8, r=8, t=20, b=8),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      yaxis_title="кг", font=dict(family=FONT_FAMILY, color=INK))
    return fig


def _chemical_ui_error(exc: Exception) -> None:
    st.error(f"Операция не выполнена: {exc}. Проверьте поля и повторите.")


def render_chemicals_section() -> None:
    """Independent novice-friendly UI; it never creates efficacy evidence implicitly."""
    with st.expander("Реагенты и склад", expanded=False):
        st.caption(f"Хранилище: {chemical_storage_path()} · данные эффективности добавляются только явно.")
        try:
            repository = get_chemical_repository()
            products = repository.list_products()
            envelopes = repository.list_envelopes()
        except (galit.ChemicalStorageError, ValueError) as exc:
            _chemical_ui_error(exc)
            return

        catalog_tab = st.expander("1. Каталог и доказательства", expanded=True)
        recommendation_tab = st.expander("2. Подбор для скважины")
        stock_tab = st.expander("3. Склад и резервы")
        forecast_tab = st.expander("4. Прогноз потребления")
        product_labels = {f"{item.name} · {item.id}": item for item in products}

        with catalog_tab:
            st.markdown("#### Каталог продуктов")
            if products:
                st.dataframe(chemical_catalog_frame(products, envelopes), width="stretch", hide_index=True)
            else:
                st.info("Каталог пуст. Добавьте реальный продукт; доказательства эффективности не создаются автоматически.")
            with st.form("chemical-product-form", clear_on_submit=True):
                st.markdown("##### Добавить или обновить продукт")
                c1, c2, c3 = st.columns(3)
                product_id = c1.text_input("ID продукта *")
                product_name = c2.text_input("Название *")
                manufacturer = c3.text_input("Производитель *")
                hazards_text = st.text_input("Опасности *", help="Коды через запятую: halite, calcite, wax, corrosion")
                compatible_text = st.text_input("Явно подтверждённая совместимость", help="Для многокомпонентного назначения; коды через запятую")
                p1, p2, p3 = st.columns(3)
                density_text = p1.text_input("Плотность, кг/л (необязательно)")
                price_text = p2.text_input("Цена за кг (необязательно)")
                currency = p3.selectbox("Валюта", ["BYN", "RUB", "USD", "EUR"])
                notes = st.text_area("Примечание")
                if st.form_submit_button("Сохранить продукт"):
                    try:
                        repository.put_product(galit.ChemicalProduct(
                            product_id, product_name, manufacturer,
                            tuple(x.strip() for x in hazards_text.split(",") if x.strip()),
                            tuple(x.strip() for x in compatible_text.split(",") if x.strip()),
                            Decimal(density_text) if density_text.strip() else None,
                            Decimal(price_text) if price_text.strip() else None,
                            currency if price_text.strip() else None, True, notes.strip() or None,
                        ), expected_revision=repository.revision)
                        st.success("Продукт сохранён. Эффективность по-прежнему не считается подтверждённой.")
                        st.rerun()
                    except (ValueError, galit.ChemicalStorageError, galit.ChemicalConflictError) as exc:
                        _chemical_ui_error(exc)

            if products:
                st.markdown("##### Ввести или импортировать конверт эффективности")
                st.warning("Отметка «валидировано» допустима только при наличии проверяемой ссылки и фактических испытаний.")
                selected_label = st.selectbox("Продукт", list(product_labels), key="chemical-envelope-product")
                e1, e2 = st.columns(2)
                envelope_id = e1.text_input("ID конверта *", key="chemical-envelope-id")
                envelope_hazard = e2.selectbox("Опасность", list(CHEMICAL_HAZARD_RU), key="chemical-envelope-hazard")
                reference = st.text_input("Ссылка / номер отчёта *", key="chemical-envelope-reference")
                conditions = st.text_area("Условия испытаний *", key="chemical-envelope-conditions")
                points_upload = st.file_uploader("CSV испытаний", type=["csv"], key="chemical-envelope-file",
                    help="dose_kg_m3,effective или dose_mg_l,effective")
                points_text = st.text_area("Или CSV вручную", key="chemical-envelope-text",
                    placeholder="dose_kg_m3,effective\n0.025,true")
                validated = st.checkbox("Данные валидированы ответственным специалистом", key="chemical-envelope-validated")
                confirm_evidence = st.checkbox("Подтверждаю: это фактические испытания, а не расчётные или демонстрационные данные",
                                               key="chemical-envelope-confirm")
                if st.button("Сохранить доказательство", key="chemical-envelope-save"):
                    try:
                        if not validated or not confirm_evidence:
                            raise ValueError("для сохранения требуется явное подтверждение валидации")
                        raw = points_upload.getvalue() if points_upload is not None else points_text.encode("utf-8")
                        points = chemical_points_from_frame(pd.read_csv(io.BytesIO(raw)))
                        repository.put_envelope(galit.ChemicalDoseResponseEnvelope(
                            envelope_id, product_labels[selected_label].id, envelope_hazard,
                            points, True, reference, conditions,
                        ), expected_revision=repository.revision)
                        st.success("Валидированный конверт сохранён с исходной ссылкой.")
                        st.rerun()
                    except (ValueError, pd.errors.ParserError, UnicodeDecodeError,
                            galit.ChemicalStorageError, galit.ChemicalConflictError,
                            galit.ChemicalNotFoundError) as exc:
                        _chemical_ui_error(exc)
            if envelopes:
                st.markdown("##### Реестр доказательств")
                st.dataframe(pd.DataFrame([{
                    "ID": item.id, "Продукт ID": item.product_id,
                    "Опасность": CHEMICAL_HAZARD_RU.get(item.hazard, item.hazard),
                    "Валидировано": item.validated, "Ссылка": item.validation_reference,
                    "Минимальная эффективная доза, кг/м³": (
                        None if item.minimum_effective_dose is None else float(item.minimum_effective_dose)
                    ), "Условия": item.conditions, "Ревизия": item.revision,
                } for item in envelopes]), width="stretch", hide_index=True)

        with recommendation_tab:
            st.markdown("#### Evidence-gated подбор для контекста скважины")
            hazard_options = list(CHEMICAL_HAZARD_RU)
            hazards = st.multiselect("Действующие опасности *", hazard_options,
                                     format_func=lambda x: CHEMICAL_HAZARD_RU[x], key="chemical-rec-hazards")
            r1, r2 = st.columns(2)
            treated = r1.number_input("Обрабатываемая жидкость, м³/сут", min_value=0.0, value=0.0, key="chemical-rec-fluid")
            oil = r2.number_input("Дебит нефти, м³/сут", min_value=0.0, value=0.0, key="chemical-rec-oil")
            st.caption("Доза: кг реагента на м³ обрабатываемой жидкости. Минимум выбирается только из эффективных испытанных точек.")
            if st.button("Проверить кандидатов", key="chemical-rec-run"):
                if not hazards:
                    st.warning("Выберите хотя бы одну действующую опасность.")
                else:
                    rows = chemical_candidate_rows(products, envelopes, hazards, treated, oil)
                    st.session_state["chemical-recommendations"] = rows
            rows = st.session_state.get("chemical-recommendations")
            if rows:
                frame = pd.DataFrame(rows)
                eligible = frame[frame["Статус"] == "подходит"] if "Статус" in frame else pd.DataFrame()
                if eligible.empty:
                    st.warning("Назначение невозможно: нет кандидата с достаточными валидированными доказательствами.")
                else:
                    st.success(f"Подходящих кандидатов: {len(eligible)}. Перед назначением проверьте условия испытаний и ссылку.")
                    if oil == 0:
                        st.warning("Дебит нефти равен нулю: стоимость на м³ нефти не определена, а не равна нулю.")
                st.dataframe(frame, width="stretch", hide_index=True)

        with stock_tab:
            st.markdown("#### Остатки по партиям и append-only журнал")
            as_of = st.date_input("Состояние на дату", value=date.today(), key="chemical-stock-asof")
            inventory = chemical_inventory_frame(repository, as_of=as_of)
            if inventory.empty:
                st.info("Партий пока нет. Поступление создаёт партию и неизменяемую запись журнала.")
            else:
                totals = inventory[["На складе, кг", "Зарезервировано, кг", "Доступно, кг"]].sum()
                m1, m2, m3 = st.columns(3)
                m1.metric("На складе", f"{totals['На складе, кг']:.3f} кг")
                m2.metric("Зарезервировано", f"{totals['Зарезервировано, кг']:.3f} кг")
                m3.metric("Доступно", f"{totals['Доступно, кг']:.3f} кг")
                st.dataframe(inventory, width="stretch", hide_index=True)
                st.plotly_chart(fig_chemical_stock(inventory), width="stretch",
                                config={"displaylogo": False, "displayModeBar": False})
            if products:
                with st.form("chemical-lot-form", clear_on_submit=True):
                    st.markdown("##### Зарегистрировать поступление")
                    lot_product = st.selectbox("Продукт", list(product_labels), key="chemical-lot-product")
                    l1, l2, l3 = st.columns(3)
                    lot_id = l1.text_input("ID партии *")
                    quantity = l2.number_input("Количество, кг *", min_value=0.001, value=1.0)
                    expires = l3.date_input("Годен до *", value=date.today() + timedelta(days=365))
                    received = st.date_input("Дата поступления", value=date.today(), key="chemical-lot-received")
                    if st.form_submit_button("Записать поступление"):
                        try:
                            repository.add_lot(galit.StockLot(
                                lot_id, product_labels[lot_product].id,
                                datetime(received.year, received.month, received.day, tzinfo=timezone.utc),
                                expires, Decimal(str(quantity)),
                            ), idempotency_key=f"ui-receipt:{lot_id}", expected_revision=repository.revision)
                            st.success("Поступление записано в append-only журнал.")
                            st.rerun()
                        except (ValueError, galit.ChemicalStorageError, galit.ChemicalConflictError,
                                galit.ChemicalNotFoundError) as exc:
                            _chemical_ui_error(exc)

                st.markdown("##### Резервирование")
                reserve_product = st.selectbox("Продукт для резерва", list(product_labels), key="chemical-reserve-product")
                q1, q2 = st.columns(2)
                reserve_quantity = q1.number_input("Количество резерва, кг", min_value=0.001, value=1.0, key="chemical-reserve-qty")
                required_on = q2.date_input("Требуется на дату", value=date.today(), key="chemical-reserve-date")
                reserve_confirm = st.checkbox("Подтверждаю создание резерва и уменьшение доступного остатка",
                                              key="chemical-reserve-confirm")
                if st.button("Создать резерв", key="chemical-reserve-save"):
                    if not reserve_confirm:
                        st.warning("Сначала явно подтвердите операцию.")
                    else:
                        try:
                            item = repository.reserve(
                                product_labels[reserve_product].id, Decimal(str(reserve_quantity)), required_on,
                                idempotency_key=f"ui-reserve:{uuid4()}", expected_revision=repository.revision,
                            )
                            st.success(f"Резерв создан: {item.quantity_kg} кг; распределение FEFO: {item.allocations}.")
                            st.rerun()
                        except (ValueError, galit.ChemicalStorageError, galit.ChemicalConflictError) as exc:
                            _chemical_ui_error(exc)

            reservations = repository.list_reservations()
            if reservations:
                st.markdown("##### Резервы")
                st.dataframe(pd.DataFrame([{
                    "ID": item.id, "Продукт ID": item.product_id, "Количество, кг": float(item.quantity_kg),
                    "Требуется": item.required_on.isoformat(), "Статус": item.status, "Ревизия": item.revision,
                } for item in reservations]), width="stretch", hide_index=True)
                active = {f"{item.product_id} · {item.quantity_kg} кг · {item.required_on} · {item.id}": item
                          for item in reservations if item.status == "active"}
                if active:
                    release_label = st.selectbox("Активный резерв", list(active), key="chemical-release-id")
                    release_confirm = st.checkbox("Подтверждаю освобождение резерва",
                                                  key="chemical-release-confirm")
                    if st.button("Освободить резерв", key="chemical-release-save"):
                        if not release_confirm:
                            st.warning("Сначала явно подтвердите освобождение.")
                        else:
                            try:
                                item = active[release_label]
                                repository.release_reservation(item.id, revision=item.revision,
                                                               expected_revision=repository.revision)
                                st.success("Резерв освобождён; история операции сохранена.")
                                st.rerun()
                            except (galit.ChemicalStorageError, galit.ChemicalConflictError,
                                    galit.ChemicalNotFoundError) as exc:
                                _chemical_ui_error(exc)
            transactions = repository.list_transactions()
            if transactions:
                with st.expander("Журнал складских транзакций"):
                    st.dataframe(pd.DataFrame([{
                        "Дата": item.occurred_at.isoformat(), "Тип": item.kind,
                        "Продукт ID": item.product_id, "Партия": item.lot_id,
                        "Количество, кг": float(item.quantity_kg), "Ссылка": item.reference,
                    } for item in transactions]), width="stretch", hide_index=True)

        with forecast_tab:
            st.markdown("#### Детерминированный прогноз и риск дефицита")
            if not products:
                st.info("Сначала добавьте продукт и реальные складские операции.")
            else:
                forecast_product = st.selectbox("Продукт", list(product_labels), key="chemical-forecast-product")
                f1, f2, f3 = st.columns(3)
                horizon = int(f1.number_input("Горизонт, суток", min_value=1, value=30, step=1))
                lead = int(f2.number_input("Срок поставки, суток", min_value=0, value=14, step=1))
                safety = int(f3.number_input("Страховой запас, суток", min_value=0, value=7, step=1))
                forecast_as_of = st.date_input("Расчёт на дату", value=date.today(), key="chemical-forecast-asof")
                view = chemical_forecast_view(
                    repository, product_labels[forecast_product].id, as_of=forecast_as_of,
                    horizon_days=horizon, lead_time_days=lead, safety_stock_days=safety,
                )
                forecast = view["forecast"]
                if forecast["status"] != "available":
                    st.warning("Прогноз недоступен: нет истории списаний. Отсутствие данных не трактуется как нулевой расход.")
                else:
                    shortage = view["shortage"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Средний расход", f"{_chemical_number(forecast['daily_kg'])} кг/сут")
                    c2.metric(f"Потребность за {horizon} сут", f"{_chemical_number(forecast['required_kg'])} кг")
                    c3.metric("Доступно", f"{_chemical_number(shortage['available_kg'])} кг")
                    c4.metric("Дефицит", f"{_chemical_number(shortage['shortage_kg'])} кг")
                    if shortage["risk"]:
                        st.error(f"Риск дефицита: к сроку поставки + страховому окну не хватает {_chemical_number(shortage['shortage_kg'])} кг.")
                    else:
                        st.success("Доступный непросроченный остаток покрывает срок поставки и страховой запас.")
                    st.caption(f"Покрытие: {_chemical_number(shortage['days_cover'])} суток; горизонт: {horizon}; срок поставки: {lead}; страховой запас: {safety}.")
                st.info("Допущения: " + view["assumptions"])


def main() -> None:
    upload, production_mode, include_uncertainty = render_sidebar()
    render_header()
    # Независимые additive-разделы доступны даже без файла фонда и не меняют legacy upload.
    render_compatibility_section()
    render_chemicals_section()

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

    treatment_repository = get_treatment_repository()
    passport_repository = get_passport_repository()

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

    # --- reference dashboard: center analytics plus a dedicated right alerts rail ---
    overview_main, overview_alerts_col = st.columns([3.35, 1], gap="large")
    with overview_main:
        st.markdown(f'<div class="panel-title">{icon_span("chart-column")}Динамика риска по текущему фонду '
                    f'<span class="status-chip is-ok">{icon_span("circle-check")}текущий срез</span></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_risk_overview(results), width="stretch",
                        config={"displaylogo": False, "displayModeBar": False})
        table_col, mix_col = st.columns([3, 2], gap="large")
        with table_col:
            top_rows = "".join(
                '<tr>'
                f'<td>{html.escape(item.well)}</td>'
                f'<td>{MECH_RU.get(item.dominant, item.dominant)}</td>'
                f'<td><span class="risk-badge" style="background:{risk_status(item.integrated_risk)[1]}">'
                f'{item.integrated_risk:.2f}</span></td>'
                f'<td>{item.quality.grade}</td>'
                '</tr>'
                for item in sorted(results, key=lambda row: row.integrated_risk, reverse=True)[:5]
            )
            st.markdown(
                f'<div class="overview-table-wrap"><div class="panel-title">{icon_span("list-checks")}Топ скважин по риску</div>'
                '<table class="overview-table"><thead><tr><th>Скважина</th>'
                '<th>Ключевое осложнение</th><th>Риск</th><th>Качество</th></tr></thead>'
                f'<tbody>{top_rows}</tbody></table>'
                '<div class="shell-copy" style="color:#0F6B43;margin:12px 0 0">'
                f'Полный список — во вкладке «Ранжирование фонда» {icon_span("arrow-right")}</div></div>',
                unsafe_allow_html=True,
            )
        with mix_col:
            st.markdown(f'<div class="panel-title">{icon_span("layers")}Структура осложнений</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_mechanism_mix(results), width="stretch",
                            config={"displaylogo": False, "displayModeBar": False})
    with overview_alerts_col:
        alerts = overview_alerts(results)
        st.markdown(
            '<div class="alerts-rail"><div class="panel-title">Критические оповещения'
            f'<span class="alerts-count">{sum(a["level"] == "critical" for a in alerts)}</span>'
            '</div><div class="shell-copy">Текущие сигналы без вымышленных дат</div></div>',
            unsafe_allow_html=True,
        )
        if not alerts:
            st.markdown(
                '<div class="alert-card is-ok"><div class="alert-card-title">Активных сигналов нет</div>'
                '<div class="alert-card-meta">Все скважины ниже порога повышенного риска.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            for alert in alerts[:6]:
                alert_icon = icon_span(
                    "triangle-alert" if alert["level"] == "critical" else "circle-dot"
                )
                st.markdown(
                    f'<div class="alert-card is-{alert["level"]}">'
                    f'<div class="alert-card-title">{alert_icon}{html.escape(alert["well"])}</div>'
                    f'<div class="alert-card-meta">{html.escape(alert["title"])}<br>'
                    f'{html.escape(alert["quality"])}</div></div>', unsafe_allow_html=True,
                )
        if len(alerts) > 6:
            st.caption(f"Ещё сигналов: {len(alerts) - 6}. Полный список — в ранжировании.")

    st.divider()
    tab_plan, tab_map, tab_watercut, tab_rank, tab_profiles, tab_well, tab_scenario, tab_economics, tab_forecast, tab_equipment, tab_pilot, tab_passport, tab_twin, tab_journal = st.tabs(
        ["План мастера", "Карта месторождения", "Обводнение", "Ранжирование фонда", "Профили T(z) · P(z)",
         "Детально по скважине", "Что будет, если?", "Экономика риска",
         "Прогноз во времени", "Оборудование / Прогноз отказов", "Сравнение с baseline / Пилот",
         "Цифровой паспорт", "Цифровой двойник", "Журнал мероприятий"]
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

    # --- Умная карта 2.0: текущий срез + явная история/GIS без подмены источников ---
    with tab_map:
        diagnosed = [DiagnosedWell(cases_by_name[x.well], x) for x in results if x.well in cases_by_name]
        service = smart_map_service(diagnosed)
        st.markdown("### Умная карта месторождения 2.0")
        c1,c2,c3,c4 = st.columns(4)
        mechanism = c1.selectbox("Слой риска", ["integrated","wax","halite","calcite","corrosion","watercut","equipment"], key="smart-map-mechanism")
        show_heat = c2.checkbox("Heatmap", True, key="smart-map-heat")
        show_markers = c2.checkbox("Скважины", True, key="smart-map-markers")
        show_hotspots = c3.checkbox("Системные зоны", True, key="smart-map-hotspots")
        show_infra = c3.checkbox("Инфраструктура GIS", True, key="smart-map-infra")
        animate = c4.checkbox("Анимация истории", False, key="smart-map-animation")
        as_of_date = c4.date_input("Срез as-of", value=datetime.now(timezone.utc).date(), key="smart-map-asof")
        as_of = datetime(as_of_date.year,as_of_date.month,as_of_date.day,23,59,59,tzinfo=timezone.utc)
        snapshot = service.snapshot(as_of=as_of, mechanism=mechanism)
        groups = service.groups(level="cluster", as_of=as_of, mechanism=mechanism)
        hotspots = service.hotspots(as_of=as_of, mechanism=mechanism) if show_hotspots else []
        frames = service.frames(mechanism=mechanism) if animate else []
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Точек as-of", snapshot["sample_size"])
        k2.metric("Coverage", f"{snapshot['coverage']:.0%}")
        k3.metric("Системных зон", len(hotspots))
        k4.metric("Кадров истории", len(frames))
        if snapshot["points"]:
            st.plotly_chart(fig_smart_map(snapshot,infrastructure=service.infrastructure_geojson() if show_infra else None,
                hotspots=hotspots,frames=frames,show_heatmap=show_heat,show_markers=show_markers),width="stretch",
                config={"displaylogo":False,"scrollZoom":True,"responsive":True})
        else:
            st.info("Для выбранного as-of/механизма нет свежих точек. Missing не считается нулевым риском.")
        st.warning(galit.SMART_MAP_DISCLAIMER)
        st.caption(f"Окно актуальности: {snapshot['stale_after_days']} суток · sample n={snapshot['sample_size']}. Heat-вклад нормирован на размер фонда. Галит и кальцит доступны раздельно.")
        st.caption("Зона «Припятский прогиб» показана обзорно для ориентации и не является точной или лицензионной геологической границей. Подложка OpenStreetMap требует интернет.")
        if animate and len(frames)<2: st.info("Анимация недоступна: требуется минимум два датированных среза.")
        if groups:
            st.markdown("#### Кусты (административные группы, не hotspot-зоны)")
            st.dataframe(pd.DataFrame(groups),width="stretch",hide_index=True)
        if hotspots:
            st.markdown("#### Ранжированные системные зоны")
            st.dataframe(pd.DataFrame([{"Зона":z["zone_id"][-8:],"Скважин":len(z["member_wells"]),"Механизмы":", ".join(z["common_mechanisms"]),"Coverage":z["coverage"],"Confidence":z["confidence"]} for z in hotspots]),width="stretch",hide_index=True)
        with st.expander("Импорт пользовательской инфраструктуры GeoJSON"):
            upload_gis=st.file_uploader("FeatureCollection: Point facilities / LineString pipelines",type=["geojson","json"],key="smart-map-gis-upload")
            if upload_gis and st.button("Проверить и импортировать GIS",key="smart-map-gis-submit"):
                try:
                    rows=galit.assets_from_geojson(json.loads(upload_gis.getvalue().decode("utf-8")))
                    for row in rows: service.repository.upsert_asset(row)
                    st.success(f"Импортировано: {len(rows)}. Реальные объекты не генерируются автоматически.")
                except (ValueError,json.JSONDecodeError,galit.SmartMapStorageError,galit.SmartMapConflictError) as exc: st.error(str(exc))
            st.download_button("CSV-шаблон истории рисков",galit.risk_csv_template(),"smart-map-risk-history.csv","text/csv",key="smart-map-risk-template")
            st.download_button("Экспорт текущего GeoJSON",json.dumps(service.infrastructure_geojson(),ensure_ascii=False,indent=2),"smart-map-infrastructure.geojson","application/geo+json",key="smart-map-export")

    with tab_watercut:
        render_watercut(get_watercut_repository())

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

    # --- полноценный сценарный конструктор ---
    with tab_scenario:
        st.markdown("### Конструктор «Что будет, если?»")
        st.caption("Изменения дебита, давления и температуры идут в существующее ядро. Для промывки, режима и дозировки нужны явные эффекты; скрытых коэффициентов нет.")
        scenario_name = st.selectbox("Скважина сценария", list(cases_by_name), key="scenario_well")
        scenario_case = cases_by_name[scenario_name]
        a, b, c = st.columns(3)
        with a:
            oil_pct = st.number_input("Изменение дебита нефти, %", value=0.0, key="scenario_oil_pct")
            water_pct = st.number_input("Изменение дебита воды, %", value=0.0, key="scenario_water_pct")
            pressure_mpa = st.number_input("Изменение буферного давления, МПа", value=0.0, key="scenario_pressure")
        with b:
            temperature_c = st.number_input("Изменение температуры, °C", value=0.0, key="scenario_temperature")
            inhibitor_dose = st.number_input("Изменение дозировки ингибитора, мг/л", min_value=0.0, value=0.0, key="scenario_dose")
            wash = st.checkbox("Промывка", key="scenario_wash")
            mode = st.text_input("Новый режим эксплуатации", value="", key="scenario_mode")
        with c:
            override_enabled = st.checkbox("Есть документированный effect override", key="scenario_override")
            override_eff = st.number_input("Итоговая эффективность ингибитора, 0–1", 0.0, 1.0, 0.0, key="scenario_override_eff")
            override_source = st.text_input("Источник эффекта", value="", key="scenario_override_source")
            st.markdown("**Экономика (необязательно)**")
            probability_raw = st.text_input("Вероятность события, 0–1", value="", key="scenario_probability")
            efficiency_raw = st.text_input("Эффективность мероприятия, 0–1", value="", key="scenario_econ_eff")
            price_raw = st.text_input("Цена продукции / м³", value="", key="scenario_price")
            cost_raw = st.text_input("Стоимость мероприятия", value="", key="scenario_cost")
            currency = st.text_input("Валюта", value="", key="scenario_currency")
        try:
            override = (galit.EffectOverride(inhibitor_efficiency=float(override_eff),
                        source=override_source or None) if override_enabled else None)
            changes = galit.ScenarioChanges(
                oil_rate_relative_change=float(oil_pct) / 100,
                water_rate_relative_change=float(water_pct) / 100,
                wellhead_pressure_delta_pa=float(pressure_mpa) * 1e6,
                surface_temperature_delta_c=float(temperature_c),
                inhibitor_dosage_delta_mg_l=float(inhibitor_dose) if inhibitor_dose else None,
                wash_treatment=wash, operating_mode=mode or None, effect_override=override,
            )
            probability = parse_optional_nonnegative(probability_raw, "Вероятность")
            econ_eff = parse_optional_nonnegative(efficiency_raw, "Эффективность")
            price = parse_optional_nonnegative(price_raw, "Цена")
            cost = parse_optional_nonnegative(cost_raw, "Стоимость")
            economics = None
            if any(value is not None for value in (probability, econ_eff, price, cost)) or currency:
                economics = galit.ScenarioEconomics(
                    horizon_days=30, event_probability=probability,
                    treatment_efficiency=econ_eff, product_price_per_m3=price,
                    operating_loss_per_day=0 if price is not None else None,
                    treatment_cost=cost, currency=currency or None,
                    probability_source="dashboard_explicit_input",
                )
            comparison = scenario_for_dashboard(scenario_case, changes, economics)
            st.warning(f"Статус: {comparison.status.value}. Screening score не является вероятностью или причинной оценкой.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Риск до → после", f"{comparison.before.integrated_risk:.3f} → {comparison.after.integrated_risk:.3f}", f"{comparison.delta['integrated_risk']:+.3f}")
            m2.metric("Дебит до → после", f"{comparison.before.forecast_oil_rate_m3_day:.2f} → {comparison.after.forecast_oil_rate_m3_day:.2f} м³/сут")
            breakdown = comparison.economics.breakdown if comparison.economics else None
            unit = comparison.economics.currency if comparison.economics else None
            m3.metric("Стоимость", "unavailable" if not breakdown or breakdown.total_treatment_cost is None else f"{breakdown.total_treatment_cost:.2f} {unit}")
            m4.metric("Net effect / ROI", "unavailable" if not breakdown or breakdown.net_expected_effect is None else f"{breakdown.net_expected_effect:.2f} {unit} / {breakdown.roi_ratio if breakdown.roi_ratio is not None else '—'}")
            st.dataframe(pd.DataFrame([
                {"Механизм": MECH_RU.get(key, key), "До": comparison.before.severity[key],
                 "После": comparison.after.severity[key], "Δ": comparison.delta["severity"][key]}
                for key in comparison.before.severity
            ]), width="stretch", hide_index=True)
            if comparison.missing_inputs:
                st.info("Missing inputs: " + ", ".join(comparison.missing_inputs))
            for warning in comparison.warnings:
                st.warning(warning)
            with st.expander("Audit trail, формулы и допущения"):
                st.json(comparison.audit_trail)
                st.markdown("\n".join(f"- `{key}`: {value}" for key, value in comparison.formulas.items()))
                st.markdown("\n".join(f"- {value}" for value in comparison.assumptions))
        except ValueError as exc:
            st.error(f"Сценарий не рассчитан: {exc}")

    # --- экономика риска: только явные денежные входы ---
    with tab_economics:
        st.markdown("### Экономика риска по скважине")
        st.caption("Пустые денежные поля не заменяются примерными ставками: результат будет partial/unavailable.")
        econ_well = st.selectbox("Скважина для экономики", list(cases_by_name), key="economics_well")
        econ_case = cases_by_name[econ_well]
        c1, c2, c3 = st.columns(3)
        with c1:
            probability = st.number_input("Вероятность события", 0.0, 1.0, 0.0, 0.01,
                                          key="economics_probability")
            horizon = st.number_input("Горизонт, сут.", 1.0, 3650.0, 30.0,
                                      key="economics_horizon")
            efficiency = st.number_input("Эффективность обработки", 0.0, 1.0, 0.7, 0.05,
                                         key="economics_efficiency")
        with c2:
            event_days = st.number_input("Простой при событии, сут.", 0.0, value=0.0,
                                         key="economics_event_days")
            treatment_days = st.number_input("Простой на обработку, сут.", 0.0, value=0.0,
                                             key="economics_treatment_days")
            loss_fraction = st.number_input("Доля потери дебита", 0.0, 1.0, 1.0, 0.05,
                                            key="economics_loss_fraction")
        with c3:
            currency = st.text_input("Валюта", value="", placeholder="например, BYN",
                                     key="economics_currency")
            price_raw = st.text_input("Цена продукции за м³", value="", key="economics_price")
            operating_raw = st.text_input("Операционные потери в сутки", value="",
                                          key="economics_operating")
            treatment_raw = st.text_input("Стоимость обработки", value="",
                                          key="economics_treatment_cost")
        try:
            economics = risk_economics_for_dashboard(
                econ_case, probability=float(probability), horizon_days=float(horizon),
                efficiency=float(efficiency), event_downtime_days=float(event_days),
                treatment_downtime_days=float(treatment_days),
                price=parse_optional_nonnegative(price_raw, "Цена продукции"),
                operating_loss=parse_optional_nonnegative(operating_raw, "Операционные потери"),
                treatment_cost=parse_optional_nonnegative(treatment_raw, "Стоимость обработки"),
                currency=currency, loss_fraction=float(loss_fraction),
            )
            b = economics.breakdown
            unit = economics.currency or "валюта не задана"
            if economics.status is galit.RiskEconomicsStatus.AVAILABLE:
                st.success(f"Статус: available · единая валюта {unit}")
            else:
                st.warning("Статус: " + economics.status.value + ". Не хватает: " +
                           ", ".join(economics.missing_inputs))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ожидаемая потеря", "—" if b.expected_production_loss_m3 is None
                      else f"{b.expected_production_loss_m3:.2f} м³")
            m2.metric("Ущерб без обработки", "—" if b.expected_damage_without_treatment is None
                      else f"{b.expected_damage_without_treatment:.2f} {unit}")
            m3.metric("Предотвращаемый ущерб", "—" if b.potential_avoided_damage is None
                      else f"{b.potential_avoided_damage:.2f} {unit}")
            m4.metric("Чистый эффект", "—" if b.net_expected_effect is None
                      else f"{b.net_expected_effect:.2f} {unit}")
            st.dataframe(pd.DataFrame([
                {"Показатель": key, "Значение": value}
                for key, value in vars(b).items()
            ]), width="stretch", hide_index=True)
            with st.expander("Формулы, допущения и ограничения"):
                for name, formula in economics.formulas.items():
                    st.markdown(f"- `{name}` = {formula}")
                st.markdown("**Допущения**\n" + "\n".join(f"- {x}" for x in economics.assumptions))
                st.markdown("**Ограничения**\n" + "\n".join(f"- {x}" for x in economics.limitations))
        except ValueError as exc:
            st.error(f"Экономика риска не рассчитана: {exc}")

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
            mode = st.radio("Шкала timeline", ("Календарь", "Дни от расчёта"),
                            horizontal=True, key="forecast_timeline_mode")
            if dated:
                st.plotly_chart(fig_forecast_timeline(forecast, calendar=mode == "Календарь"),
                                width="stretch", config={"displaylogo": False})
            else:
                st.info("Датированных событий нет: контракт вернул только screening/unavailable без временных окон.")
            st.markdown("### Механизмы: trust, evidence и calibration")
            for event in forecast.events:
                evidence = event.evidence
                evidence_chip = (f"evidence · {evidence.points} точек · {evidence.span_days:.0f} сут"
                                 if evidence.span_days is not None else "evidence · недостаточно")
                calibration_chip = ("calibration · matched" if event.calibration.matched
                                    else f"calibration · {event.calibration.validation_status}")
                with st.expander(f"{event.title} · {event.status.value}"):
                    st.caption(f"trust · {'production-ready' if event.production_ready else 'screening'}  |  "
                               f"{evidence_chip}  |  {calibration_chip}")
                    st.markdown(f"**Основание:** {event.basis}\n\n**Метод:** {event.method}")
                    st.markdown("**Assumptions**\n" + "\n".join(f"- {x}" for x in event.assumptions) if event.assumptions else "**Assumptions:** —")
                    st.markdown("**Limitations**\n" + "\n".join(f"- {x}" for x in event.limitations) if event.limitations else "**Limitations:** —")
                    st.markdown("**Required inputs**\n" + "\n".join(f"- {x}" for x in event.required_inputs) if event.required_inputs else "**Required inputs:** —")
            if undated:
                st.markdown("### Unavailable / без даты")
                for event in undated:
                    st.warning(f"{event.title}: дата недоступна. " +
                               ("Требуется: " + "; ".join(event.required_inputs)
                                if event.required_inputs else event.basis))
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

    with tab_passport:
        render_well_passport(passport_repository, treatment_repository,
                             sorted(cases_by_name))

    with tab_equipment:
        render_equipment_forecasts(galit.EquipmentRepository(equipment_storage_path()))

    with tab_twin:
        try:
            twin_service, manual_events = get_twin_components(list(cases_by_name.values()))
            render_digital_twin(twin_service, manual_events)
        except Exception as exc:  # UI boundary: a broken optional source must not hide other tabs.
            st.error(f"Цифровой двойник временно недоступен: {html.escape(str(exc))}")

    with tab_journal:
        render_treatment_journal(
            treatment_repository, treatment_well_context(cases_by_name, results)
        )


if __name__ == "__main__":
    main()
