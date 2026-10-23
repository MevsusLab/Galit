from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CorrosionConditions:
    t_c: float
    p_total_pa: float
    co2_mol_frac: float
    velocity_m_s: float
    diameter_m: float
    ph_actual: float | None = None
    watercut: float = 1.0
    oil_velocity_m_s: float = 0.0
    inhibitor_efficiency: float = 0.0
    glycol_wt_pct: float = 0.0
    t_scale_c: float | None = None


def co2_fugacity(cond: CorrosionConditions) -> float:
    t_k = cond.t_c + 273.15
    p_bar = cond.p_total_pa / 1e5
    p_eff = min(p_bar, 250.0)
    p_co2 = p_bar * cond.co2_mol_frac
    log_a = (0.0031 - 1.4 / t_k) * p_eff
    return p_co2 * 10.0 ** log_a


def ph_saturated(f_co2_bar: float, t_c: float) -> float:
    f = max(f_co2_bar, 1e-6)
    ph1 = 1.36 + 1307.0 / (t_c + 273.0) - 0.17 * math.log10(f)
    ph2 = 5.4 - 0.66 * math.log10(f)
    return min(ph1, ph2)


def ph_co2_water(f_co2_bar: float, t_c: float) -> float:
    f = max(f_co2_bar, 1e-6)
    t_k = t_c + 273.15

    log_kh = 108.3865 + 0.01985076 * t_k - 6919.53 / t_k \
        - 40.45154 * math.log10(t_k) + 669365.0 / t_k ** 2
    k1 = 10.0 ** (-(3404.71 / t_k - 14.8435 + 0.032786 * t_k))
    c_h2co3 = f * 10.0 ** (log_kh - 1.468)
    c_h2co3 = max(c_h2co3, 1e-12)
    h_plus = math.sqrt(k1 * c_h2co3)
    return -math.log10(max(h_plus, 1e-14))


def scale_temperature(f_co2_bar: float) -> float:
    f = max(f_co2_bar, 1e-6)
    den = 6.7 + 0.6 * math.log10(f)
    if den <= 0.1:
        return 1e6
    return 2400.0 / den - 273.0


def corrosion_rate_1975(cond: CorrosionConditions) -> float:
    t_k = cond.t_c + 273.15
    p_co2 = max(cond.p_total_pa / 1e5 * cond.co2_mol_frac, 1e-6)
    log_v = 5.8 - 1710.0 / t_k + 0.67 * math.log10(p_co2)
    return 10.0 ** log_v


def corrosion_rate_1995(cond: CorrosionConditions) -> dict[str, float]:
    f_co2 = co2_fugacity(cond)
    t_c = cond.t_c

    ph_c = ph_co2_water(f_co2, t_c)
    ph_a = cond.ph_actual if cond.ph_actual is not None else ph_c

    log_vr = (
        4.84
        - 1119.0 / (t_c + 273.0)
        + 0.58 * math.log10(max(f_co2, 1e-6))
        - 0.34 * (ph_a - ph_c)
    )
    v_r = 10.0 ** log_vr

    u = max(cond.velocity_m_s, 1e-3)
    d = max(cond.diameter_m, 1e-3)
    v_m = 2.45 * (u ** 0.8 / d ** 0.2) * max(f_co2, 1e-6)

    if v_r <= 0 or v_m <= 0:
        v_base = 0.0
    else:
        v_base = 1.0 / (1.0 / v_r + 1.0 / v_m)

    t_scale = (cond.t_scale_c if cond.t_scale_c is not None
               else scale_temperature(f_co2))
    f_scale = 1.0
    if t_c > t_scale:
        log_fs = 2400.0 * (1.0 / (t_c + 273.15) - 1.0 / (t_scale + 273.15))
        f_scale = min(10.0 ** log_fs, 1.0)

    f_oil = 1.0
    if cond.watercut < 0.30 and cond.oil_velocity_m_s > 1.0:
        f_oil = 0.1

    f_glyc = 1.0
    if cond.glycol_wt_pct > 0.0:
        w_water = max(100.0 - cond.glycol_wt_pct, 1.0)
        f_glyc = min(10.0 ** (1.6 * math.log10(w_water) - 3.2), 1.0)

    v_uninhibited = v_base * f_scale * f_oil * f_glyc
    v_final = v_uninhibited * (1.0 - min(max(cond.inhibitor_efficiency, 0.0), 0.99))

    return {
        "rate_mm_yr": v_final,
        "rate_uninhibited": v_uninhibited,
        "v_reaction": v_r,
        "v_masstransfer": v_m,
        "limiting": "массоперенос" if v_m < v_r else "реакция",
        "f_co2_bar": f_co2,
        "ph_actual": ph_a,
        "ph_co2": ph_c,
        "f_scale": f_scale,
        "t_scale_c": t_scale,
        "f_oil": f_oil,
    }


def corrosion_severity(rate_mm_yr: float) -> tuple[str, float]:
    if rate_mm_yr < 0.025:
        return "низкая", min(max(rate_mm_yr, 0.0) / 0.025 * 0.25, 0.25)
    if rate_mm_yr < 0.12:
        return "умеренная", 0.25 + (rate_mm_yr - 0.025) / 0.095 * 0.25
    if rate_mm_yr < 0.25:
        return "высокая", 0.5 + (rate_mm_yr - 0.12) / 0.13 * 0.25
    span = math.log10(10.0 / 0.25)
    frac = math.log10(rate_mm_yr / 0.25) / span
    return "очень высокая", min(0.75 + 0.25 * frac, 1.0)
