"""Профили давления и температуры в стволе добывающей скважины.

Температура -- Ramey (1962), аналитическое решение.
Давление -- Beggs & Brill (1973), маршевый расчёт снизу вверх.

Обе корреляции выбраны сознательно:
  * Ramey имеет замкнутое решение и открытый бенчмарк для валидации
    (OpenGeoSys EUBHE, расхождение < 0.15 %);
  * Beggs-Brill -- единственная из классической тройки, работающая
    при любых углах наклона, что нужно для наклонно-направленных
    и горизонтальных скважин Речицкого месторождения.

Ограничение Ramey: метод завышает температуру в раннем переходном
периоде (Hagoort 2004, SPE J 9(4) 465-474). Применять при t > ~1 недели
непрерывной работы. Для скважин в режиме КВЧ это ограничение
существенно -- см. предупреждение в temperature_profile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fluids import (
    G,
    brine_density,
    brine_viscosity,
    bubble_point_standing,
    gas_density,
    gas_viscosity_lge,
    oil_fvf_standing,
    oil_viscosity_beggs_robinson,
    solution_gor_standing,
    z_factor_dak,
)


@dataclass
class WellGeometry:
    """Геометрия и режим работы скважины."""
    depth_m: float                 # глубина по стволу до забоя
    tubing_id_m: float             # внутренний диаметр НКТ
    inclination_deg: float = 0.0   # средний угол от вертикали
    roughness_m: float = 4.6e-5    # шероховатость стали

    @property
    def theta_rad(self) -> float:
        """Угол от горизонтали (для Beggs-Brill)."""
        return math.radians(90.0 - self.inclination_deg)


@dataclass
class ProductionRate:
    """Дебиты в поверхностных условиях."""
    q_oil_m3d: float
    q_water_m3d: float
    gor_m3m3: float                # газовый фактор

    @property
    def q_liquid_m3d(self) -> float:
        return self.q_oil_m3d + self.q_water_m3d

    @property
    def watercut(self) -> float:
        if self.q_liquid_m3d <= 0:
            return 0.0
        return self.q_water_m3d / self.q_liquid_m3d


@dataclass
class FluidProperties:
    """Свойства флюидов."""
    gamma_oil: float = 0.86
    gamma_gas: float = 0.75
    salinity_ppm: float = 300_000.0
    surface_tension: float = 0.03   # Н/м


@dataclass
class ThermalParams:
    """Теплофизика для модели Ramey."""
    t_surface_c: float = 8.0            # температура пород у поверхности
    geothermal_grad: float = 0.033      # K/м
    k_earth: float = 2.5                # Вт/(м*К)
    alpha_earth: float = 1.0e-6         # м2/с
    u_to: float = 15.0                  # Вт/(м2*К), к наружному радиусу НКТ
    r_to: float = 0.038                 # м
    r_wb: float = 0.108                 # м
    cp_fluid: float = 2100.0            # Дж/(кг*К)
    production_days: float = 365.0


def ramey_time_function(tp: ThermalParams) -> float:
    """Безразмерная функция времени f(t) в модели Ramey.

        f(t) = -ln[ r_wb / (2*sqrt(alpha*t)) ] - 0.290,   t_D > 100

    Константа 0.290 подтверждена независимо (Hasan & Kabir 1991).
    """
    t_s = tp.production_days * 86400.0
    t_d = tp.alpha_earth * t_s / tp.r_wb ** 2
    if t_d <= 1.5:
        # ранний период: логарифмическое приближение неустойчиво
        return max(2.0 * math.sqrt(t_d / math.pi) * (1.0 - 0.3 * math.sqrt(t_d)),
                   1e-3)
    return -math.log(tp.r_wb / (2.0 * math.sqrt(tp.alpha_earth * t_s))) - 0.290


def ramey_relaxation_length(w_kg_s: float, tp: ThermalParams) -> float:
    """Релаксационная длина A, м.

        A = w*cp*[k_e + r_to*U_to*f(t)] / (2*pi*r_to*U_to*k_e)
    """
    if w_kg_s <= 0:
        return 1e-6
    f_t = ramey_time_function(tp)
    num = w_kg_s * tp.cp_fluid * (tp.k_earth + tp.r_to * tp.u_to * f_t)
    den = 2.0 * math.pi * tp.r_to * tp.u_to * tp.k_earth
    return num / max(den, 1e-12)


def temperature_profile(
    geom: WellGeometry,
    rate: ProductionRate,
    fluid: FluidProperties,
    tp: ThermalParams,
    n_nodes: int = 60,
) -> tuple[list[float], list[float], str | None]:
    """Профиль температуры потока T(z), C.  Ramey (1962).

    Вывод (y -- расстояние вверх от забоя, T_e(y) = T_fbh - g*y):

        A * dT_f/dy + T_f = T_e(y)
        T_f(y) = T_e(y) + g*A*(1 - exp(-y/A))

    То есть поток ВСЕГДА горячее вмещающих пород на g*A*(1-exp(-y/A)):
    он не успевает остыть до породы, поднимаясь. На забое (y=0)
    превышение нулевое, у устья стремится к пределу g*A.

    Возвращает (глубины сверху вниз, температуры, предупреждение).
    """
    l_total = geom.depth_m
    # массовый расход
    rho_o = fluid.gamma_oil * 1000.0
    rho_w = brine_density(60.0, fluid.salinity_ppm)
    w = (rate.q_oil_m3d * rho_o + rate.q_water_m3d * rho_w) / 86400.0

    a = ramey_relaxation_length(w, tp)
    g_grad = tp.geothermal_grad
    t_es = tp.t_surface_c

    depths, temps = [], []
    for i in range(n_nodes + 1):
        z = l_total * i / n_nodes
        t_ei = t_es + g_grad * z           # температура пород на глубине z
        y = l_total - z                    # путь, пройденный потоком вверх
        expo = max(-y / max(a, 1e-6), -700.0)   # защита от underflow
        t_f = t_ei + g_grad * a * (1.0 - math.exp(expo))
        depths.append(z)
        temps.append(t_f)

    warn = None
    t_d = tp.alpha_earth * tp.production_days * 86400.0 / tp.r_wb ** 2
    if t_d < 100.0:
        warn = (
            f"Ramey завышает T в раннем переходном периоде (t_D={t_d:.0f}<100, "
            f"Hagoort 2004). Для скважин в режиме КВЧ нужна модель "
            f"Hasan-Kabir с учётом циклов останова."
        )
    return depths, temps, warn


# --------------------------------------------------------------------------
# Beggs & Brill (1973)
# --------------------------------------------------------------------------

# Коэффициенты горизонтального holdup: режим -> (a, b, c)
_HOLDUP_COEF = {
    "segregated": (0.980, 0.4846, 0.0868),
    "intermittent": (0.845, 0.5351, 0.0173),
    "distributed": (1.065, 0.5824, 0.0609),
}

# Коэффициенты поправки на наклон: режим -> (d, e, f, g)
# Внимание: в downhill f = 0.1244, а не -0.3692 -- частая ошибка реализаций.
_INCL_COEF_UP = {
    "segregated": (0.011, -3.768, 3.539, -1.614),
    "intermittent": (2.96, 0.305, -0.4473, 0.0978),
    "distributed": None,  # C = 0
}
_INCL_COEF_DOWN = (4.70, -0.3692, 0.1244, -0.5056)


def _flow_regime(lam_l: float, n_fr: float) -> str:
    """Определение режима течения по карте Beggs-Brill."""
    lam = max(lam_l, 1e-9)
    l1 = 316.0 * lam ** 0.302
    l2 = 0.0009252 * lam ** (-2.4684)
    l3 = 0.10 * lam ** (-1.4516)
    l4 = 0.5 * lam ** (-6.738)

    if (lam < 0.01 and n_fr < l1) or (lam >= 0.01 and n_fr < l2):
        return "segregated"
    if lam >= 0.01 and l2 <= n_fr <= l3:
        return "transition"
    if (0.01 <= lam < 0.4 and l3 < n_fr <= l1) or (lam >= 0.4 and l3 < n_fr <= l4):
        return "intermittent"
    return "distributed"


def _holdup_horizontal(regime: str, lam_l: float, n_fr: float) -> float:
    a, b, c = _HOLDUP_COEF[regime]
    hl0 = a * max(lam_l, 1e-9) ** b / max(n_fr, 1e-9) ** c
    return min(max(hl0, lam_l), 1.0)


def _incl_correction(regime: str, lam_l: float, n_lv: float, n_fr: float,
                     theta: float) -> float:
    """Множитель psi для поправки holdup на угол наклона."""
    if theta >= 0:
        coef = _INCL_COEF_UP.get(regime)
        if coef is None:
            return 1.0
    else:
        coef = _INCL_COEF_DOWN
    d, e, f, g = coef
    lam = max(lam_l, 1e-9)
    arg = d * lam ** e * max(n_lv, 1e-9) ** f * max(n_fr, 1e-9) ** g
    if arg <= 0:
        return 1.0
    c_val = (1.0 - lam) * math.log(arg)
    c_val = max(c_val, 0.0)
    s18 = math.sin(1.8 * theta)
    return 1.0 + c_val * (s18 - 0.333 * s18 ** 3)


def _friction_factor_moody(re: float, rel_rough: float) -> float:
    """Коэффициент трения Муди. Ламинарный / Colebrook через Haaland."""
    if re < 2000.0:
        return 64.0 / max(re, 1e-3)
    inv = -1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / max(re, 1e-3))
    return (1.0 / inv) ** 2


def _two_phase_friction(f_n: float, lam_l: float, h_l: float) -> float:
    """Двухфазный коэффициент трения через множитель exp(S)."""
    y = max(lam_l, 1e-9) / max(h_l, 1e-9) ** 2
    if abs(y - 1.0) < 1e-9:
        return f_n
    if 1.0 < y < 1.2:
        s = math.log(2.2 * y - 1.2)  # обход сингулярности
    else:
        ln_y = math.log(max(y, 1e-9))
        den = -0.0523 + 3.182 * ln_y - 0.8725 * ln_y ** 2 + 0.01853 * ln_y ** 4
        if abs(den) < 1e-9:
            return f_n
        s = ln_y / den
    s = max(min(s, 50.0), -50.0)
    return f_n * math.exp(s)


def pressure_profile(
    geom: WellGeometry,
    rate: ProductionRate,
    fluid: FluidProperties,
    depths: list[float],
    temps: list[float],
    p_wellhead_pa: float = 1.2e6,
) -> list[float]:
    """Профиль давления от устья к забою, Па.

    Маршевый расчёт сверху вниз по узлам, заданным профилем температуры.
    Возвращает список давлений, соответствующий depths.
    """
    n = len(depths)
    pressures = [0.0] * n
    pressures[0] = p_wellhead_pa

    area = math.pi * (geom.tubing_id_m / 2.0) ** 2
    theta = geom.theta_rad
    rel_rough = geom.roughness_m / geom.tubing_id_m

    # газосодержание при насыщении -- полный ГФ
    rs_max = rate.gor_m3m3

    for i in range(1, n):
        dz = depths[i] - depths[i - 1]
        p = pressures[i - 1]
        t = 0.5 * (temps[i] + temps[i - 1])

        # --- PVT в узле ---
        pb = bubble_point_standing(rs_max, fluid.gamma_gas, t, fluid.gamma_oil)
        rs = solution_gor_standing(p, pb, rs_max, fluid.gamma_gas, t,
                                   fluid.gamma_oil)
        bo = oil_fvf_standing(rs, fluid.gamma_gas, t, fluid.gamma_oil)

        rho_o_sc = fluid.gamma_oil * 1000.0
        rho_g_sc = fluid.gamma_gas * 1.2254
        rho_oil = (rho_o_sc + rs * rho_g_sc) / max(bo, 1e-6)
        rho_w = brine_density(t, fluid.salinity_ppm)
        rho_g = gas_density(max(p, 1e5), t, fluid.gamma_gas)

        # --- расходы в условиях узла, м3/с ---
        q_o = rate.q_oil_m3d * bo / 86400.0
        q_w = rate.q_water_m3d / 86400.0
        free_gor = max(rs_max - rs, 0.0)
        z = z_factor_dak(max(p, 1e5), t, fluid.gamma_gas)
        bg = (z * (t + 273.15) * 101325.0) / (max(p, 1e5) * 288.15)
        q_g = rate.q_oil_m3d * free_gor * bg / 86400.0

        q_l = q_o + q_w
        if q_l <= 0 and q_g <= 0:
            pressures[i] = p
            continue

        v_sl = q_l / area
        v_sg = q_g / area
        v_m = v_sl + v_sg
        lam_l = v_sl / max(v_m, 1e-9)

        # --- свойства жидкой смеси ---
        f_o = q_o / max(q_l, 1e-12)
        rho_l = rho_oil * f_o + rho_w * (1.0 - f_o)
        mu_o = oil_viscosity_beggs_robinson(t, fluid.gamma_oil, rs)
        mu_w = brine_viscosity(t, fluid.salinity_ppm)
        mu_l = mu_o * f_o + mu_w * (1.0 - f_o)
        mu_g = gas_viscosity_lge(max(p, 1e5), t, fluid.gamma_gas)

        # --- безразмерные группы ---
        n_fr = v_m ** 2 / (G * geom.tubing_id_m)
        n_lv = 1.938 * v_sl * (rho_l / (G * fluid.surface_tension)) ** 0.25

        regime = _flow_regime(lam_l, n_fr)
        if regime == "transition":
            lam = max(lam_l, 1e-9)
            l2 = 0.0009252 * lam ** (-2.4684)
            l3 = 0.10 * lam ** (-1.4516)
            hl_seg = _holdup_horizontal("segregated", lam_l, n_fr)
            hl_int = _holdup_horizontal("intermittent", lam_l, n_fr)
            psi_seg = _incl_correction("segregated", lam_l, n_lv, n_fr, theta)
            psi_int = _incl_correction("intermittent", lam_l, n_lv, n_fr, theta)
            span = max(l3 - l2, 1e-12)
            h_l = ((l3 - n_fr) * hl_seg * psi_seg
                   + (n_fr - l2) * hl_int * psi_int) / span
        else:
            hl0 = _holdup_horizontal(regime, lam_l, n_fr)
            h_l = hl0 * _incl_correction(regime, lam_l, n_lv, n_fr, theta)
        h_l = min(max(h_l, lam_l, 1e-6), 1.0)

        # --- градиент давления ---
        rho_s = rho_l * h_l + rho_g * (1.0 - h_l)
        rho_n = rho_l * lam_l + rho_g * (1.0 - lam_l)
        mu_n = mu_l * lam_l + mu_g * (1.0 - lam_l)

        re_n = rho_n * v_m * geom.tubing_id_m / max(mu_n, 1e-9)
        f_n = _friction_factor_moody(re_n, rel_rough)
        f_tp = _two_phase_friction(f_n, lam_l, h_l)

        dp_grav = rho_s * G * math.sin(theta)
        dp_fric = f_tp * rho_n * v_m ** 2 / (2.0 * geom.tubing_id_m)

        # кинетическая составляющая (Ek)
        e_k = rho_s * v_m * v_sg / max(p, 1e5)
        e_k = min(e_k, 0.95)

        dpdz = (dp_grav + dp_fric) / (1.0 - e_k)
        pressures[i] = max(p + dpdz * dz, 1e5)

    return pressures
