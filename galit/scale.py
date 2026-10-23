from __future__ import annotations

import math
from dataclasses import dataclass

from .fluids import water_ionic_strength

MW = {
    "Na": 22.990, "K": 39.098, "Ca": 40.078, "Mg": 24.305,
    "Ba": 137.327, "Sr": 87.62, "Fe": 55.845,
    "Cl": 35.453, "SO4": 96.06, "HCO3": 61.016, "CO3": 60.008,
}

LSI_TDS_LIMIT = 10_000.0


@dataclass(frozen=True)
class WaterAnalysis:
    ions_mg_l: dict[str, float]
    ph: float
    t_c: float
    p_pa: float

    @property
    def tds_mg_l(self) -> float:
        return sum(self.ions_mg_l.values())

    def molarity(self) -> dict[str, float]:
        return {
            ion: mg / MW[ion] / 1000.0
            for ion, mg in self.ions_mg_l.items()
            if ion in MW
        }

    @property
    def ionic_strength(self) -> float:
        return water_ionic_strength(self.molarity())

    def water_kg_per_l(self) -> float:
        rho_g_l = 1000.0 + 0.7 * self.tds_mg_l / 1000.0
        return max((rho_g_l - self.tds_mg_l / 1000.0) / 1000.0, 0.1)

    def molality(self) -> dict[str, float]:
        w = self.water_kg_per_l()
        return {ion: c / w for ion, c in self.molarity().items()}

    def molality_na_cl(self) -> tuple[float, float]:
        mol = self.molality()
        return mol.get("Na", 0.0), mol.get("Cl", 0.0)

    def molality_nacl(self) -> float:
        na, cl = self.molality_na_cl()
        return math.sqrt(max(na, 0.0) * max(cl, 0.0))


def halite_saturation_molality(t_c: float) -> float:
    w = 26.218 + 0.0072 * t_c + 0.000106 * t_c ** 2
    return 1000.0 * w / (58.443 * (100.0 - w))


def halite_activity_coefficient(m_nacl: float, t_c: float,
                                ionic_strength: float | None = None) -> float:
    if m_nacl <= 0:
        return 1.0

    b0, b1, c_phi = 0.0765, 0.2664, 0.00127
    a_phi = 0.392
    b, alpha = 1.2, 2.0

    i = m_nacl if ionic_strength is None else max(ionic_strength, 1e-9)
    sqrt_i = math.sqrt(i)

    f_gamma = -a_phi * (
        sqrt_i / (1.0 + b * sqrt_i) + (2.0 / b) * math.log(1.0 + b * sqrt_i)
    )
    x = alpha * sqrt_i
    b_gamma = 2.0 * b0 + (2.0 * b1 / (alpha ** 2 * i)) * (
        1.0 - (1.0 + x + 0.5 * x ** 2) * math.exp(-x)
    )
    c_gamma = 1.5 * c_phi

    ln_gamma = f_gamma + m_nacl * b_gamma + m_nacl ** 2 * c_gamma
    return math.exp(ln_gamma)


def halite_saturation_index(water: WaterAnalysis) -> float:
    m_na, m_cl = water.molality_na_cl()
    if m_na <= 0 or m_cl <= 0:
        return -99.0

    m_sat = halite_saturation_molality(water.t_c)
    i_solution = water.ionic_strength / water.water_kg_per_l()

    g = halite_activity_coefficient(math.sqrt(m_na * m_cl), water.t_c,
                                    ionic_strength=i_solution)
    g_sat = halite_activity_coefficient(m_sat, water.t_c)

    iap = (g * m_na) * (g * m_cl)
    ksp = (g_sat * m_sat) ** 2
    return math.log10(iap / ksp)


def halite_precipitation_potential(water: WaterAnalysis) -> float:
    m_na, m_cl = water.molality_na_cl()
    if m_na <= 0 or m_cl <= 0:
        return 0.0

    m_sat = halite_saturation_molality(water.t_c)
    i_solution = water.ionic_strength / water.water_kg_per_l()
    g = halite_activity_coefficient(math.sqrt(m_na * m_cl), water.t_c,
                                    ionic_strength=i_solution)
    g_sat = halite_activity_coefficient(m_sat, water.t_c)

    ksp_m = (g_sat * m_sat / max(g, 1e-9)) ** 2
    if m_na * m_cl <= ksp_m:
        return 0.0

    b = m_na + m_cl
    c = m_na * m_cl - ksp_m
    disc = max(b * b - 4.0 * c, 0.0)
    x = (b - math.sqrt(disc)) / 2.0
    x = min(max(x, 0.0), min(m_na, m_cl))

    water_kg_m3 = water.water_kg_per_l() * 1000.0
    return x * 58.443e-3 * water_kg_m3

SDSI_IONIC_STRENGTH_LIMIT = 4.0

SDSI_FIT_BREAK = 1.2


def stiff_davis_k(ionic_strength: float, t_c: float) -> float:
    i = max(ionic_strength, 1e-6)
    if i < SDSI_FIT_BREAK:
        return (
            2.022 * math.exp((math.log(i) + 7.544) ** 2 / 102.60)
            - 0.0002 * t_c ** 2
            + 0.00097 * t_c
            + 0.262
        )
    return -0.1 * i - 0.0002 * t_c ** 2 - 0.00097 * t_c + 3.887


def stiff_davis_index(water: WaterAnalysis) -> float:
    mol = water.molarity()
    ca = mol.get("Ca", 0.0)

    alk = mol.get("HCO3", 0.0) + 2.0 * mol.get("CO3", 0.0)
    if ca <= 0 or alk <= 0:
        return -99.0

    p_ca = -math.log10(ca)
    p_alk = -math.log10(alk)
    k = stiff_davis_k(water.ionic_strength, water.t_c)
    return water.ph - p_ca - p_alk - k


def stiff_davis_index_checked(water: WaterAnalysis) -> tuple[float, list[str]]:
    warnings: list[str] = []
    i = water.ionic_strength

    if i > SDSI_IONIC_STRENGTH_LIMIT:
        warnings.append(
            f"ионная сила {i:.1f} моль/л выше предела калибровки "
            f"Stiff-Davis (I<={SDSI_IONIC_STRENGTH_LIMIT:.0f}): результат "
            f"экстраполяция, нужна модель Питцера (ScaleSoftPitzer/OLI MSE)"
        )
    if abs(i - SDSI_FIT_BREAK) < 0.15:
        warnings.append(
            f"ионная сила {i:.2f} близка к точке излома фита USBR "
            f"(I={SDSI_FIT_BREAK}): в ней разрыв ~0.16 лог-единиц"
        )
    if 40.0 <= water.t_c <= 70.0:
        warnings.append(
            "T в диапазоне 40-70 C, где фит USBR завышен на +0.2...+0.6 "
            "относительно номограммы ASTM D4582 -- прогноз консервативен"
        )
    if water.t_c > 90.0:
        warnings.append(
            f"T={water.t_c:.0f} C выше диапазона Stiff-Davis (0-90 C)"
        )

    return stiff_davis_index(water), warnings


def lsi_langelier(water: WaterAnalysis) -> tuple[float, str | None]:
    mol = water.molarity()
    ca_mg_caco3 = mol.get("Ca", 0.0) * 100_090.0
    alk_mg_caco3 = (
        mol.get("HCO3", 0.0) + 2.0 * mol.get("CO3", 0.0)
    ) * 50_045.0
    if ca_mg_caco3 <= 0 or alk_mg_caco3 <= 0:
        return -99.0, "нет данных по Ca или щёлочности"

    a = (math.log10(max(water.tds_mg_l, 1.0)) - 1.0) / 10.0
    b = -13.12 * math.log10(water.t_c + 273.15) + 34.55
    c = math.log10(ca_mg_caco3) - 0.4
    d = math.log10(alk_mg_caco3)
    ph_s = (9.3 + a + b) - (c + d)

    warn = None
    if water.tds_mg_l > LSI_TDS_LIMIT:
        warn = (
            f"LSI неприменим: TDS {water.tds_mg_l / 1000:.0f} г/л превышает "
            f"границу ASTM D3739/D4582 ({LSI_TDS_LIMIT / 1000:.0f} г/л) "
            f"в {water.tds_mg_l / LSI_TDS_LIMIT:.0f} раз. "
            f"Используйте Stiff-Davis."
        )
    return water.ph - ph_s, warn


def calcite_degas_ph_screening(
    ph_initial: float,
    t_c: float,
    p_initial_pa: float,
    p_local_pa: float,
    co2_mol_frac: float,
    max_ph_rise: float = 2.0,
) -> tuple[float, str | None]:
    if co2_mol_frac <= 0.0 or p_initial_pa <= p_local_pa:
        return ph_initial, None

    p_co2_initial = max(p_initial_pa * co2_mol_frac, 1.0)
    p_co2_local = max(p_local_pa * co2_mol_frac, 1.0)
    delta_ph = min(max(0.5 * math.log10(p_co2_initial / p_co2_local), 0.0),
                   max(max_ph_rise, 0.0))
    ph_local = min(max(ph_initial + delta_ph, 0.0), 14.0)
    warning = (
        f"pH={ph_local:.2f} при T={t_c:.1f} C оценён screening-моделью "
        "дегазации CO2 (идеальный газ, без карбонатного speciation и "
        "буферной ёмкости); не использовать как точный прогноз"
    )
    return ph_local, warning


def scale_risk_profile(water: WaterAnalysis) -> dict[str, float]:
    si_halite = halite_saturation_index(water)
    si_calcite = stiff_davis_index(water)
    return {
        "si_halite": si_halite,
        "si_calcite": si_calcite,
        "halite_kg_m3": halite_precipitation_potential(water),
        "m_nacl": water.molality_nacl(),
        "m_sat": halite_saturation_molality(water.t_c),
        "ionic_strength": water.ionic_strength,
        "tds_g_l": water.tds_mg_l / 1000.0,
    }
