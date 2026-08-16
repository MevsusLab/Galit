"""PVT-свойства пластовых флюидов.

Стандартные промысловые корреляции. Каждая функция помечена источником.
Единицы внутри модуля — СИ, если явно не указано иное в docstring.
"""
from __future__ import annotations

import math

G = 9.80665  # м/с^2


# --------------------------------------------------------------------------
# Вода / рассол
# --------------------------------------------------------------------------

def brine_density(t_c: float, salinity_ppm: float) -> float:
    """Плотность рассола, кг/м3.

    McCain (1991), упрощённая форма: плотность при н.у. по массовой доле
    NaCl-эквивалента, затем термическое расширение.
    Для условий Припятского прогиба (до 300 000 ppm) даёт ~1,20 г/см3,
    что согласуется с промысловыми данными Белоруснефти.
    """
    ws = salinity_ppm / 1e6  # массовая доля
    # плотность при 20 C, г/см3 (McCain)
    rho20 = 0.99823 + 0.7650 * ws - 0.0331 * ws ** 2
    # температурная поправка
    rho = rho20 * (1.0 - 3.1e-4 * (t_c - 20.0) - 2.7e-6 * (t_c - 20.0) ** 2)
    return rho * 1000.0


def brine_viscosity(t_c: float, salinity_ppm: float) -> float:
    """Динамическая вязкость рассола, Па*с.

    McCain (1991) корреляция для пластовых вод.
    """
    ws_pct = salinity_ppm / 1e4  # % масс.
    a = 109.574 - 8.40564 * ws_pct + 0.313314 * ws_pct ** 2 + 8.72213e-3 * ws_pct ** 3
    b = (
        1.12166
        - 2.63951e-2 * ws_pct
        + 6.79461e-4 * ws_pct ** 2
        + 5.47119e-5 * ws_pct ** 3
        - 1.55586e-6 * ws_pct ** 4
    )
    t_f = t_c * 9.0 / 5.0 + 32.0
    mu_cp = a * max(t_f, 1.0) ** (-b)
    return mu_cp * 1e-3


def water_ionic_strength(ions_mol_l: dict[str, float]) -> float:
    """Ионная сила, моль/л.  I = 0.5 * sum(c_i * z_i^2)."""
    charges = {
        "Na": 1, "K": 1, "Cl": -1, "HCO3": -1,
        "Ca": 2, "Mg": 2, "Ba": 2, "Sr": 2, "SO4": -2, "CO3": -2, "Fe": 2,
    }
    total = 0.0
    for ion, conc in ions_mol_l.items():
        z = charges.get(ion)
        if z is None:
            raise KeyError(f"неизвестный ион: {ion}")
        total += conc * z * z
    return 0.5 * total


# --------------------------------------------------------------------------
# Нефть
# --------------------------------------------------------------------------

def bubble_point_standing(rs_m3m3: float, gamma_g: float, t_c: float,
                          gamma_o: float) -> float:
    """Давление насыщения, Па.  Standing (1947).

    rs_m3m3 -- газосодержание, м3/м3
    gamma_g -- относительная плотность газа по воздуху
    t_c     -- температура, C
    gamma_o -- относительная плотность нефти по воде
    """
    api = 141.5 / gamma_o - 131.5
    rs_scf_stb = rs_m3m3 * 5.6146
    t_f = t_c * 9.0 / 5.0 + 32.0
    a = 0.00091 * t_f - 0.0125 * api
    pb_psia = 18.2 * ((rs_scf_stb / gamma_g) ** 0.83 * 10.0 ** a - 1.4)
    return max(pb_psia, 14.7) * 6894.757


def solution_gor_standing(p_pa: float, pb_pa: float, rs_max: float,
                          gamma_g: float, t_c: float, gamma_o: float) -> float:
    """Газосодержание Rs при давлении p, м3/м3.  Standing (1947).

    Выше давления насыщения Rs постоянно и равно rs_max.
    """
    if p_pa >= pb_pa:
        return rs_max
    api = 141.5 / gamma_o - 131.5
    t_f = t_c * 9.0 / 5.0 + 32.0
    p_psia = p_pa / 6894.757
    a = 0.00091 * t_f - 0.0125 * api
    rs_scf_stb = gamma_g * ((p_psia / 18.2 + 1.4) * 10.0 ** (-a)) ** (1.0 / 0.83)
    return max(0.0, min(rs_scf_stb / 5.6146, rs_max))


def oil_fvf_standing(rs_m3m3: float, gamma_g: float, t_c: float,
                     gamma_o: float) -> float:
    """Объёмный коэффициент нефти Bo, м3/м3.  Standing (1947)."""
    rs_scf_stb = rs_m3m3 * 5.6146
    t_f = t_c * 9.0 / 5.0 + 32.0
    f = rs_scf_stb * math.sqrt(gamma_g / gamma_o) + 1.25 * t_f
    return 0.9759 + 12e-5 * f ** 1.2


def oil_viscosity_beggs_robinson(t_c: float, gamma_o: float,
                                 rs_m3m3: float) -> float:
    """Вязкость насыщенной газом нефти, Па*с.  Beggs & Robinson (1975)."""
    api = 141.5 / gamma_o - 131.5
    t_f = t_c * 9.0 / 5.0 + 32.0
    # дегазированная нефть
    z = 3.0324 - 0.02023 * api
    y = 10.0 ** z
    x = y * t_f ** (-1.163)
    mu_od = 10.0 ** x - 1.0
    # При высокой T и лёгкой нефти корреляция даёт mu_od <= 0 (она
    # калибрована на 70-295 F). Отрицательное основание в дробной
    # степени ниже дало бы комплексное число -- подрезаем по нижней
    # физической границе вязкости дегазированной нефти.
    mu_od = max(mu_od, 0.1)
    # поправка на растворённый газ
    rs_scf_stb = rs_m3m3 * 5.6146
    a = 10.715 * (rs_scf_stb + 100.0) ** (-0.515)
    b = 5.44 * (rs_scf_stb + 150.0) ** (-0.338)
    mu_ob = a * mu_od ** b
    return max(mu_ob, 0.1) * 1e-3


# --------------------------------------------------------------------------
# Газ
# --------------------------------------------------------------------------

def z_factor_dak(p_pa: float, t_c: float, gamma_g: float) -> float:
    """Коэффициент сверхсжимаемости газа.  Dranchuk & Abou-Kassem (1975).

    Решение нелинейного уравнения методом Ньютона по приведённой плотности.
    """
    # псевдокритические параметры, Standing
    tpc_r = 168.0 + 325.0 * gamma_g - 12.5 * gamma_g ** 2  # R
    ppc_psia = 677.0 + 15.0 * gamma_g - 37.5 * gamma_g ** 2

    t_r = (t_c * 9.0 / 5.0 + 491.67) / tpc_r
    p_r = (p_pa / 6894.757) / ppc_psia
    if p_r <= 0:
        return 1.0

    a = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475,
         -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)

    c1 = a[0] + a[1] / t_r + a[2] / t_r ** 3 + a[3] / t_r ** 4 + a[4] / t_r ** 5
    c2 = a[5] + a[6] / t_r + a[7] / t_r ** 2
    c3 = a[8] * (a[6] / t_r + a[7] / t_r ** 2)

    rho_r = 0.27 * p_r / t_r  # начальное приближение
    for _ in range(64):
        e = math.exp(-a[10] * rho_r ** 2)
        c4 = a[9] * (1.0 + a[10] * rho_r ** 2) * (rho_r ** 2 / t_r ** 3) * e
        f = (
            -0.27 * p_r / t_r
            + rho_r
            + c1 * rho_r ** 2
            + c2 * rho_r ** 3
            - c3 * rho_r ** 6
            + c4
        )
        df = (
            1.0
            + 2.0 * c1 * rho_r
            + 3.0 * c2 * rho_r ** 2
            - 6.0 * c3 * rho_r ** 5
            + (a[9] * rho_r / t_r ** 3)
            * (2.0 + 2.0 * a[10] * rho_r ** 2 - 2.0 * (a[10] * rho_r ** 2) ** 2)
            * e
        )
        if abs(df) < 1e-14:
            break
        step = f / df
        rho_r -= step
        rho_r = max(rho_r, 1e-8)
        if abs(step) < 1e-10:
            break

    return 0.27 * p_r / (rho_r * t_r)


def gas_density(p_pa: float, t_c: float, gamma_g: float) -> float:
    """Плотность газа, кг/м3."""
    z = z_factor_dak(p_pa, t_c, gamma_g)
    m = gamma_g * 0.028964  # кг/моль
    r = 8.31446
    return p_pa * m / (z * r * (t_c + 273.15))


def gas_viscosity_lge(p_pa: float, t_c: float, gamma_g: float) -> float:
    """Вязкость газа, Па*с.  Lee, Gonzalez & Eakin (1966)."""
    m = gamma_g * 28.964  # г/моль
    t_r = t_c * 9.0 / 5.0 + 491.67  # Ранкин
    rho_gcm3 = gas_density(p_pa, t_c, gamma_g) / 1000.0

    k = (9.4 + 0.02 * m) * t_r ** 1.5 / (209.0 + 19.0 * m + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * m
    y = 2.4 - 0.2 * x
    mu_cp = 1e-4 * k * math.exp(x * rho_gcm3 ** y)
    return mu_cp * 1e-3


def gas_solubility_co2_henry(p_co2_pa: float, t_c: float,
                             salinity_ppm: float) -> float:
    """Растворимость CO2 в рассоле, моль/кг H2O.

    Закон Генри с температурной зависимостью и высаливанием (Setschenow).
    Константа Сеченова для NaCl-CO2 принята 0.10 л/моль (типовое значение).
    """
    t_k = t_c + 273.15
    # константа Генри для CO2 в чистой воде, МПа/(моль/кг) -- аппроксимация
    # по данным растворимости 0-150 C
    kh = 10.0 ** (
        -6.8346 + 1.2817e4 / t_k - 3.7668e6 / t_k ** 2 + 2.997e8 / t_k ** 3
    )
    m_pure = (p_co2_pa / 1e6) / max(kh, 1e-9)
    # высаливание
    m_nacl = salinity_ppm / 1e6 * 1000.0 / 58.443 / (1.0 - salinity_ppm / 1e6)
    return m_pure * 10.0 ** (-0.10 * m_nacl)
