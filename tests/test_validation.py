"""Валидация против опубликованных бенчмарков и физических инвариантов.

Каждый тест ссылается на источник эталона. Тесты, для которых
опубликованного эталона не существует, проверяют физические
инварианты (монотонность, знаки, предельные переходы) -- и это
явно помечено в docstring.
"""
from __future__ import annotations

import math

import pytest

from galit.corrosion import (
    CorrosionConditions,
    co2_fugacity,
    corrosion_rate_1975,
    corrosion_rate_1995,
    corrosion_severity,
    scale_temperature,
)
from galit.fluids import (
    brine_density,
    brine_viscosity,
    gas_density,
    oil_viscosity_beggs_robinson,
    z_factor_dak,
)
from galit.scale import (
    WaterAnalysis,
    halite_activity_coefficient,
    halite_saturation_molality,
    halite_saturation_index,
    lsi_langelier,
    stiff_davis_index,
    stiff_davis_index_checked,
    stiff_davis_k,
)
from galit.wax import WaxProperties, wat_at_pressure, wax_onset_depth
from galit.wellbore import (
    _flow_regime,
    _holdup_horizontal,
    _two_phase_friction,
    ramey_time_function,
    temperature_profile,
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
)


# ==========================================================================
# Растворимость галита -- эталон Potter, Babcock & Brown (1977)
# ==========================================================================

class TestHaliteSolubility:
    """Potter R.W. II et al., J. Research USGS v.5 no.3 p.389-395 (1977)."""

    def test_solubility_at_25c(self):
        """Справочное значение: 6.146 моль/кг при 25 C (+-0.05 wt%)."""
        m = halite_saturation_molality(25.0)
        assert m == pytest.approx(6.146, abs=0.03)

    def test_solubility_at_0c(self):
        """Справочное значение: ~6.08 моль/кг при 0 C."""
        m = halite_saturation_molality(0.0)
        assert m == pytest.approx(6.080, abs=0.03)

    def test_solubility_at_100c(self):
        """Справочное значение: ~6.654 моль/кг при 100 C.

        Именно здесь неверифицированная корреляция
        m = 6.044 + 2.8e-3*T + 6.0e-5*T^2 даёт 6.924, то есть
        завышает на 4.1 %. Тест фиксирует, что мы её не используем.
        """
        m = halite_saturation_molality(100.0)
        assert m == pytest.approx(6.654, abs=0.05)
        # контрольная проверка: отвергнутая формула действительно хуже
        rejected = 6.044 + 2.8e-3 * 100.0 + 6.0e-5 * 100.0 ** 2
        assert abs(rejected - 6.654) > abs(m - 6.654)

    def test_monotonic_increase(self):
        """Растворимость NaCl монотонно растёт с температурой."""
        temps = [0, 20, 40, 60, 80, 100, 120]
        vals = [halite_saturation_molality(t) for t in temps]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_activity_coefficient_dilute_limit(self):
        """При m -> 0 коэффициент активности -> 1 (предельный переход)."""
        assert halite_activity_coefficient(1e-6, 25.0) == pytest.approx(1.0, abs=0.01)

    def test_activity_coefficient_at_saturation(self):
        """gamma± для NaCl при насыщении ~1.0 (известный факт: проходит
        через минимум ~0.68 при m~0.8 и возвращается к ~1 при m~6)."""
        g = halite_activity_coefficient(6.15, 25.0)
        assert 0.8 < g < 1.4

    def test_activity_coefficient_minimum(self):
        """gamma± имеет минимум в районе m = 0.5...1.5 моль/кг."""
        ms = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
        gs = [halite_activity_coefficient(m, 25.0) for m in ms]
        assert min(gs) == min(gs[1:4])  # минимум не на краях


# ==========================================================================
# Индекс насыщения галита
# ==========================================================================

class TestHaliteSaturationIndex:

    @staticmethod
    def _water(na_mg: float, cl_mg: float, t_c: float = 60.0) -> WaterAnalysis:
        return WaterAnalysis(
            ions_mg_l={"Na": na_mg, "Cl": cl_mg, "Ca": 5000.0, "HCO3": 150.0},
            ph=6.2, t_c=t_c, p_pa=5e6,
        )

    def test_undersaturated_gives_negative(self):
        """Разбавленный рассол -- SI < 0."""
        w = self._water(20_000.0, 30_000.0)
        assert halite_saturation_index(w) < 0.0

    def test_supersaturated_gives_positive(self):
        """Рассол выше растворимости -- SI > 0."""
        # ~6.5 моль/кг NaCl -> Na ~150 г/л, Cl ~230 г/л
        w = self._water(150_000.0, 231_000.0, t_c=20.0)
        assert halite_saturation_index(w) > 0.0

    def test_cooling_increases_si(self):
        """Охлаждение снижает растворимость -> повышает SI.

        Это ключевая физика галита в стволе: рассол поднимается,
        остывает и начинает кристаллизоваться у устья.
        """
        hot = self._water(120_000.0, 185_000.0, t_c=90.0)
        cold = self._water(120_000.0, 185_000.0, t_c=20.0)
        assert halite_saturation_index(cold) > halite_saturation_index(hot)

    def test_limiting_ion(self):
        """Молальность определяется лимитирующим ионом."""
        w = self._water(150_000.0, 10_000.0)  # Cl в дефиците
        m = w.molality_nacl()
        mol = w.molarity()
        assert m < mol["Na"] * 2  # Cl ограничивает


# ==========================================================================
# LSI vs Stiff-Davis -- главный тезис заявки
# ==========================================================================

class TestScaleIndexApplicability:
    """ASTM D4582 (Stiff-Davis, >10 000 мг/л) vs D3739 (Langelier)."""

    @staticmethod
    def _brine(tds_scale: float) -> WaterAnalysis:
        base = {
            "Na": 92_000.0, "Cl": 175_000.0, "Ca": 22_000.0,
            "Mg": 2_400.0, "HCO3": 120.0, "SO4": 300.0,
        }
        return WaterAnalysis(
            ions_mg_l={k: v * tds_scale for k, v in base.items()},
            ph=6.1, t_c=45.0, p_pa=4e6,
        )

    def test_lsi_warns_above_limit(self):
        """LSI обязан предупреждать о выходе за границу применимости."""
        w = self._brine(1.0)
        assert w.tds_mg_l > 100_000
        _, warn = lsi_langelier(w)
        assert warn is not None
        assert "неприменим" in warn

    def test_lsi_no_warning_for_fresh_water(self):
        """Для пресной воды предупреждения быть не должно."""
        w = WaterAnalysis(
            ions_mg_l={"Na": 50.0, "Cl": 60.0, "Ca": 80.0, "HCO3": 150.0},
            ph=7.5, t_c=20.0, p_pa=1e5,
        )
        assert w.tds_mg_l < 10_000
        _, warn = lsi_langelier(w)
        assert warn is None

    def test_lsi_overestimates_vs_stiff_davis(self):
        """LSI систематически ЗАВЫШАЕТ склонность к отложению
        на высокоминерализованных водах.

        Причина: высокая ионная сила повышает растворимость
        малорастворимых солей, что LSI не учитывает.
        Stiff-Davis всегда даёт меньшее значение при той же химии,
        и расхождение растёт с ионной силой.
        """
        w = self._brine(1.0)
        lsi, _ = lsi_langelier(w)
        sdsi = stiff_davis_index(w)
        assert lsi > sdsi, "LSI должен быть оптимистичнее (выше) Stiff-Davis"
        assert lsi - sdsi > 0.5, "расхождение должно быть значимым"

    def test_divergence_grows_with_ionic_strength(self):
        """Расхождение LSI и S&DSI растёт с ионной силой.

        Ferguson (French Creek): "The deviation between the indices
        increases with ionic strength". Проверяем на восходящей ветви
        K, то есть ниже точки излома I = 1.2.
        """
        gaps = []
        for scale in (0.02, 0.05, 0.10):
            w = self._brine(scale)
            assert w.ionic_strength < 1.2
            lsi, _ = lsi_langelier(w)
            gaps.append(lsi - stiff_davis_index(w))
        assert all(b > a for a, b in zip(gaps, gaps[1:])), gaps

    def test_stiff_davis_k_decreases_with_temperature(self):
        """K убывает с ростом T в обеих ветвях фита.

        Эталон -- оцифровка номограммы: Tian, Yan & Chen, IOP Conf.
        Ser. Earth Environ. Sci. 781 (2021) 022033, Table 1:
        dK/dT ~ -0.020 1/C, ΔK(20->80 C) ~ -1.20.
        """
        for i in (0.5, 1.0, 2.0, 3.0):
            ks = [stiff_davis_k(i, t) for t in (10.0, 30.0, 50.0, 70.0, 90.0)]
            assert all(b < a for a, b in zip(ks, ks[1:])), f"I={i}"

    def test_stiff_davis_k_temperature_slope(self):
        """ΔK при 20->80 C сопоставима с номограммой (-1.20)."""
        for i in (0.5, 1.0):
            d = stiff_davis_k(i, 80.0) - stiff_davis_k(i, 20.0)
            assert d == pytest.approx(-1.142, abs=0.02), f"I={i}"
        for i in (2.0, 3.0):
            d = stiff_davis_k(i, 80.0) - stiff_davis_k(i, 20.0)
            assert d == pytest.approx(-1.258, abs=0.02), f"I={i}"

    def test_stiff_davis_k_is_nonmonotonic_in_ionic_strength(self):
        """K НЕМОНОТОННА: растёт до I = 1.2, затем убывает.

        Это не дефект реализации, а реальная форма кривой.
        Подтверждение: независимый термодинамический расчёт
        K ~ pK2 - pKsp - log g_Ca - log g_HCO3 (константы
        Plummer & Busenberg 1982, активности по Дэвису) даёт
        максимум при I ~ 0.5; фит USBR -- при I = 1.2.
        Порядок величины и форма совпадают.
        """
        for t in (20.0, 50.0, 80.0):
            rising = [stiff_davis_k(i, t) for i in (0.05, 0.1, 0.3, 0.5, 1.0)]
            assert all(b > a for a, b in zip(rising, rising[1:])), f"T={t}"
            falling = [stiff_davis_k(i, t) for i in (1.5, 2.0, 3.0, 4.0)]
            assert all(b < a for a, b in zip(falling, falling[1:])), f"T={t}"

    def test_stiff_davis_k_second_branch_slope(self):
        """Наклон второй ветви ровно -0.1 на единицу I.

        Фиксирует, что не используется искажённая перепечатка
        с +0.1*I (Hamad & Kuwairi, doi:10.18280/mmep.110921).
        """
        k2 = stiff_davis_k(2.0, 50.0)
        k3 = stiff_davis_k(3.0, 50.0)
        assert k3 - k2 == pytest.approx(-0.1, abs=1e-9)

    def test_stiff_davis_k_reference_values(self):
        """Табличные значения фита USBR при T = 25 C."""
        expected = {0.1: 2.804, 0.5: 3.356, 1.0: 3.682,
                    1.5: 3.588, 2.0: 3.538, 3.0: 3.438, 4.0: 3.338}
        for i, k_ref in expected.items():
            assert stiff_davis_k(i, 25.0) == pytest.approx(k_ref, abs=0.01), f"I={i}"

    def test_stiff_davis_branch_discontinuity_documented(self):
        """Разрыв при I = 1.2 существует и растёт с температурой.

        Это документированный дефект фита USBR, а не наша ошибка.
        Ожидаемые значения скачка: -0.15 (20 C), -0.21 (50 C), -0.27 (80 C).
        """
        for t, expected in ((20.0, -0.152), (50.0, -0.210), (80.0, -0.268)):
            jump = stiff_davis_k(1.2001, t) - stiff_davis_k(1.1999, t)
            assert jump == pytest.approx(expected, abs=0.01), f"T={t}"

    def test_high_ionic_strength_triggers_warning(self):
        """Для рассолов Припятского прогиба S&DSI тоже вне калибровки.

        Главный тезис: у вод с плотностью 1.18-1.25 г/см3 ионная сила
        5-7 моль/л, тогда как Stiff-Davis калиброван до I = 4.
        Продукт обязан об этом предупреждать."""
        w = self._brine(1.0)
        assert w.ionic_strength > 4.0
        _, warns = stiff_davis_index_checked(w)
        assert any("Питцер" in x for x in warns)


# ==========================================================================
# de Waard - Milliams
# ==========================================================================

class TestCorrosion:
    """de Waard & Milliams (1975), de Waard/Lotz/Dugstad CORROSION/95."""

    def test_1975_reference_point(self):
        """Прямая подстановка в log V = 5.8 - 1710/T + 0.67*log(pCO2).

        При 20 C (293.15 K) и pCO2 = 1 бар:
        log V = 5.8 - 1710/293.15 + 0 = -0.0326 -> V = 0.928 мм/год
        """
        cond = CorrosionConditions(
            t_c=20.0, p_total_pa=1e5, co2_mol_frac=1.0,
            velocity_m_s=1.0, diameter_m=0.1,
        )
        expected = 10.0 ** (5.8 - 1710.0 / 293.15)
        assert corrosion_rate_1975(cond) == pytest.approx(expected, rel=1e-9)
        assert expected == pytest.approx(0.928, abs=0.01)

    def test_1975_arrhenius_activation_energy(self):
        """Коэффициент 1710 соответствует Ea ~32.7 кДж/моль."""
        ea = 1710.0 * 2.303 * 8.314 / 1000.0
        assert ea == pytest.approx(32.7, abs=0.5)

    def test_rate_increases_with_co2(self):
        """Скорость коррозии растёт с парциальным давлением CO2."""
        rates = []
        for frac in (0.005, 0.02, 0.05, 0.10):
            cond = CorrosionConditions(
                t_c=60.0, p_total_pa=5e6, co2_mol_frac=frac,
                velocity_m_s=1.5, diameter_m=0.062,
            )
            rates.append(corrosion_rate_1995(cond)["rate_mm_yr"])
        assert all(b > a for a, b in zip(rates, rates[1:]))

    def test_rate_increases_with_velocity(self):
        """Рост скорости потока усиливает массоперенос -> коррозию."""
        rates = []
        for v in (0.3, 1.0, 3.0, 6.0):
            cond = CorrosionConditions(
                t_c=60.0, p_total_pa=5e6, co2_mol_frac=0.02,
                velocity_m_s=v, diameter_m=0.062,
            )
            rates.append(corrosion_rate_1995(cond)["rate_mm_yr"])
        assert all(b > a for a, b in zip(rates, rates[1:]))

    def test_resistance_model_below_both_components(self):
        """1/V = 1/Vr + 1/Vm -> V меньше каждой из составляющих."""
        cond = CorrosionConditions(
            t_c=70.0, p_total_pa=8e6, co2_mol_frac=0.03,
            velocity_m_s=2.0, diameter_m=0.062,
        )
        r = corrosion_rate_1995(cond)
        assert r["rate_mm_yr"] < r["v_reaction"]
        assert r["rate_mm_yr"] < r["v_masstransfer"]

    def test_inhibitor_reduces_rate(self):
        """Ингибитор с эффективностью 90 % снижает скорость в 10 раз."""
        base = CorrosionConditions(
            t_c=60.0, p_total_pa=5e6, co2_mol_frac=0.02,
            velocity_m_s=1.5, diameter_m=0.062,
        )
        inh = CorrosionConditions(
            t_c=60.0, p_total_pa=5e6, co2_mol_frac=0.02,
            velocity_m_s=1.5, diameter_m=0.062, inhibitor_efficiency=0.9,
        )
        r0 = corrosion_rate_1995(base)["rate_mm_yr"]
        r1 = corrosion_rate_1995(inh)["rate_mm_yr"]
        assert r1 == pytest.approx(r0 * 0.1, rel=1e-6)

    def test_oil_wetting_protects_at_low_watercut(self):
        """При обводнённости < 30 % и скорости нефти > 1 м/с F_oil = 0.1."""
        dry = CorrosionConditions(
            t_c=60.0, p_total_pa=5e6, co2_mol_frac=0.02,
            velocity_m_s=1.5, diameter_m=0.062,
            watercut=0.15, oil_velocity_m_s=1.5,
        )
        wet = CorrosionConditions(
            t_c=60.0, p_total_pa=5e6, co2_mol_frac=0.02,
            velocity_m_s=1.5, diameter_m=0.062,
            watercut=0.85, oil_velocity_m_s=1.5,
        )
        assert corrosion_rate_1995(dry)["f_oil"] == pytest.approx(0.1)
        assert corrosion_rate_1995(wet)["f_oil"] == pytest.approx(1.0)
        assert (corrosion_rate_1995(dry)["rate_mm_yr"]
                < corrosion_rate_1995(wet)["rate_mm_yr"])

    def test_fugacity_below_partial_pressure(self):
        """При высоком давлении fCO2 < pCO2 (коэффициент a < 1)."""
        cond = CorrosionConditions(
            t_c=60.0, p_total_pa=2e7, co2_mol_frac=0.02,
            velocity_m_s=1.0, diameter_m=0.062,
        )
        f = co2_fugacity(cond)
        p_co2 = 2e7 / 1e5 * 0.02
        assert f < p_co2

    def test_severity_categories(self):
        """Пороги NACE: 0.025 / 0.12 / 0.25 мм/год."""
        assert corrosion_severity(0.01)[0] == "низкая"
        assert corrosion_severity(0.05)[0] == "умеренная"
        assert corrosion_severity(0.18)[0] == "высокая"
        assert corrosion_severity(0.5)[0] == "очень высокая"

    def test_severity_monotonic(self):
        """Нормированная тяжесть монотонна и лежит в [0, 1]."""
        vals = [corrosion_severity(r)[1] for r in (0.001, 0.03, 0.1, 0.2, 0.4, 2.0)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_severity_discriminates_above_1_mm_yr(self):
        """Верхняя категория должна РАЗЛИЧАТЬ скорости, а не насыщаться.

        Неингибированная CO2-коррозия штатно даёт единицы мм/год.
        При насыщении на 1.0 почти весь фонд получал одинаковый балл
        и ранжирование вырождалось.
        """
        vals = [corrosion_severity(r)[1] for r in (0.5, 1.0, 3.0, 7.0)]
        assert all(b > a for a, b in zip(vals, vals[1:]))
        assert vals[-1] <= 1.0

    def test_scale_temperature_falls_with_fugacity(self):
        """T_scale = 2400/(6.7+0.6*log10(fCO2)) - 273, убывает по fCO2."""
        temps = [scale_temperature(f) for f in (0.1, 1.0, 3.0, 10.0)]
        assert all(b < a for a, b in zip(temps, temps[1:]))
        assert scale_temperature(1.0) == pytest.approx(2400.0 / 6.7 - 273.0,
                                                       rel=1e-9)

    def test_rate_peaks_then_falls_with_temperature(self):
        """Максимум скорости коррозии по T, затем спад из-за плёнки FeCO3.

        Классическая подпись модели de Waard: выше T_scale карбонат
        железа образует защитный слой. Без этого члена расчёт при
        забойных 90-110 C завышал скорость на порядок.
        """
        temps = [30.0, 50.0, 70.0, 90.0, 110.0, 130.0]
        rates = []
        for t in temps:
            cond = CorrosionConditions(
                t_c=t, p_total_pa=5e6, co2_mol_frac=0.02,
                velocity_m_s=1.5, diameter_m=0.062,
            )
            rates.append(corrosion_rate_1995(cond)["rate_mm_yr"])
        i_max = rates.index(max(rates))
        assert 0 < i_max < len(rates) - 1
        assert rates[-1] < rates[i_max]

    def test_scale_factor_inactive_below_scale_temperature(self):
        """Ниже T_scale защитной плёнки нет: F_scale = 1."""
        cond = CorrosionConditions(
            t_c=40.0, p_total_pa=2e6, co2_mol_frac=0.01,
            velocity_m_s=1.0, diameter_m=0.062,
        )
        r = corrosion_rate_1995(cond)
        assert r["t_scale_c"] > 40.0
        assert r["f_scale"] == pytest.approx(1.0)


# ==========================================================================
# Beggs & Brill
# ==========================================================================

class TestBeggsBrill:
    """Beggs & Brill, JPT 255, 607-617 (1973), SPE-4007-PA."""

    def test_worked_example_holdup(self):
        """Канонический пример: lam_L = 0.167, N_Fr = 8.95 -> intermittent,
        H_L(0) = 0.845 * 0.167^0.5351 / 8.95^0.0173 = 0.31.
        """
        lam_l, n_fr = 0.167, 8.95
        assert _flow_regime(lam_l, n_fr) == "intermittent"
        hl = _holdup_horizontal("intermittent", lam_l, n_fr)
        assert hl == pytest.approx(0.31, abs=0.01)

    def test_holdup_never_below_no_slip(self):
        """H_L(0) >= lam_L -- физический инвариант."""
        for lam in (0.05, 0.2, 0.5, 0.9):
            for n_fr in (0.5, 5.0, 50.0):
                for regime in ("segregated", "intermittent", "distributed"):
                    assert _holdup_horizontal(regime, lam, n_fr) >= lam

    def test_holdup_bounded(self):
        """H_L <= 1."""
        for lam in (0.01, 0.5, 0.99):
            for n_fr in (0.01, 1.0, 100.0):
                for regime in ("segregated", "intermittent", "distributed"):
                    assert _holdup_horizontal(regime, lam, n_fr) <= 1.0

    def test_friction_no_slip_gives_unity(self):
        """При y = 1 (H_L = lam_L... точнее lam_L/H_L^2 = 1) S = 0."""
        lam = 0.25
        h_l = math.sqrt(lam)  # даёт y = 1
        assert _two_phase_friction(1.0, lam, h_l) == pytest.approx(1.0, abs=1e-6)

    def test_friction_singularity_guarded(self):
        """Ветвь 1 < y < 1.2 обязательна -- иначе уход в бесконечность."""
        lam = 0.25
        for y_target in (1.001, 1.05, 1.1, 1.19):
            h_l = math.sqrt(lam / y_target)
            f = _two_phase_friction(1.0, lam, h_l)
            assert math.isfinite(f)
            assert 0.0 < f < 100.0

    def test_regime_map_covers_all(self):
        """Карта режимов возвращает валидный режим на всей сетке."""
        valid = {"segregated", "intermittent", "distributed", "transition"}
        for lam in (0.001, 0.01, 0.1, 0.3, 0.5, 0.9, 0.999):
            for n_fr in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
                assert _flow_regime(lam, n_fr) in valid


# ==========================================================================
# Ramey
# ==========================================================================

class TestRamey:
    """Ramey H.J. Jr., JPT 14(4) 427-435 (1962), SPE-96-PA."""

    def test_time_function_positive_and_growing(self):
        """f(t) растёт со временем добычи."""
        vals = []
        for days in (7.0, 30.0, 180.0, 365.0, 1825.0):
            tp = ThermalParams(production_days=days)
            vals.append(ramey_time_function(tp))
        assert all(v > 0 for v in vals)
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_time_function_formula(self):
        """Прямая подстановка при t_D > 100."""
        tp = ThermalParams(production_days=365.0, alpha_earth=1e-6, r_wb=0.108)
        t_s = 365.0 * 86400.0
        expected = -math.log(0.108 / (2.0 * math.sqrt(1e-6 * t_s))) - 0.290
        assert ramey_time_function(tp) == pytest.approx(expected, rel=1e-9)

    def _case(self, depth=3000.0, q_oil=20.0, q_water=80.0):
        return (
            WellGeometry(depth_m=depth, tubing_id_m=0.062),
            ProductionRate(q_oil_m3d=q_oil, q_water_m3d=q_water, gor_m3m3=80.0),
            FluidProperties(),
            ThermalParams(t_surface_c=8.0, geothermal_grad=0.033),
        )

    def test_bottomhole_equals_reservoir(self):
        """На забое поток ещё не остыл: T_f(L) = T_пласта."""
        geom, rate, fluid, tp = self._case()
        _, temps, _ = temperature_profile(geom, rate, fluid, tp)
        t_res = tp.t_surface_c + tp.geothermal_grad * geom.depth_m
        assert temps[-1] == pytest.approx(t_res, rel=1e-9)

    def test_profile_physical_and_monotonic(self):
        """T растёт с глубиной и лежит между устьевой и пластовой."""
        geom, rate, fluid, tp = self._case()
        _, temps, _ = temperature_profile(geom, rate, fluid, tp)
        t_res = tp.t_surface_c + tp.geothermal_grad * geom.depth_m
        assert all(b > a for a, b in zip(temps, temps[1:]))
        assert all(tp.t_surface_c < t < t_res + 1e-9 for t in temps)

    def test_flow_hotter_than_formation(self):
        """Поток везде теплее вмещающих пород -- он не успевает остыть.

        Это ровно тот знак, который был перепутан: прежняя формула
        давала T ниже пород и уводила устье в минус на глубоком фонде.
        """
        geom, rate, fluid, tp = self._case()
        depths, temps, _ = temperature_profile(geom, rate, fluid, tp)
        for z, t in zip(depths, temps):
            t_earth = tp.t_surface_c + tp.geothermal_grad * z
            assert t >= t_earth - 1e-9

    def test_higher_rate_delivers_hotter_wellhead(self):
        """Больше расход -> длиннее релаксация -> горячее устье."""
        heads = []
        for q in (20.0, 60.0, 150.0):
            geom, rate, fluid, tp = self._case(q_oil=q * 0.2, q_water=q * 0.8)
            _, temps, _ = temperature_profile(geom, rate, fluid, tp)
            heads.append(temps[0])
        assert all(b > a for a, b in zip(heads, heads[1:]))


# ==========================================================================
# WAT
# ==========================================================================

class TestWax:

    def test_wat_minimum_at_bubble_point(self):
        """WAT минимальна вблизи давления насыщения.

        Ниже Pb: рост P растворяет лёгкие -> WAT падает.
        Выше Pb: только сжатие -> WAT растёт.
        Это исправление распространённого заблуждения о максимуме.
        """
        wax = WaxProperties(wat_stock_tank_c=35.0)
        pb = 8e6
        pressures = [0.5e6, 2e6, 4e6, 6e6, pb, 1.2e7, 2e7, 3e7]
        wats = [wat_at_pressure(wax, p, pb) for p in pressures]
        i_min = wats.index(min(wats))
        assert pressures[i_min] == pytest.approx(pb, rel=0.35)

    def test_wat_decreases_below_bubble_point(self):
        """Ниже Pb: dWAT/dP < 0."""
        wax = WaxProperties(wat_stock_tank_c=35.0)
        pb = 1e7
        a = wat_at_pressure(wax, 1e6, pb)
        b = wat_at_pressure(wax, 8e6, pb)
        assert b < a

    def test_wat_increases_above_bubble_point(self):
        """Выше Pb: dWAT/dP > 0."""
        wax = WaxProperties(wat_stock_tank_c=35.0)
        pb = 1e7
        a = wat_at_pressure(wax, 1.2e7, pb)
        b = wat_at_pressure(wax, 3e7, pb)
        assert b > a

    def test_onset_found_at_crossing(self):
        """Глубина начала АСПО = точка пересечения T(z) и WAT(P(z))."""
        depths = [float(z) for z in range(0, 2100, 100)]
        # линейный профиль: 20 C на устье, 75 C на забое
        temps = [20.0 + 55.0 * z / 2000.0 for z in depths]
        pressures = [1.2e6 + 9000.0 * z for z in depths]
        wax = WaxProperties(wat_stock_tank_c=40.0)
        onset, wat_prof = wax_onset_depth(depths, temps, pressures, wax, 8e6)
        assert onset is not None
        assert 0.0 < onset < 2000.0
        # в точке пересечения температуры совпадают
        idx = min(range(len(depths)), key=lambda i: abs(depths[i] - onset))
        assert abs(temps[idx] - wat_prof[idx]) < 3.0

    def test_no_onset_when_hot(self):
        """Если весь ствол горячее WAT -- отложений нет."""
        depths = [float(z) for z in range(0, 2100, 100)]
        temps = [80.0 + 0.01 * z for z in depths]
        pressures = [1.2e6 + 9000.0 * z for z in depths]
        wax = WaxProperties(wat_stock_tank_c=25.0)
        onset, _ = wax_onset_depth(depths, temps, pressures, wax, 8e6)
        assert onset is None


# ==========================================================================
# PVT
# ==========================================================================

class TestFluids:

    def test_brine_density_matches_field_data(self):
        """Пластовые воды Белоруснефти: 1.18-1.25 г/см3.

        Промысловый факт: осаждение галита связано с попутной добычей
        минерализованной воды плотностью 1.2 г/см3 и выше.
        """
        rho = brine_density(20.0, 300_000.0)
        assert 1150.0 < rho < 1280.0

    def test_brine_density_decreases_with_temperature(self):
        vals = [brine_density(t, 250_000.0) for t in (20.0, 50.0, 80.0)]
        assert all(b < a for a, b in zip(vals, vals[1:]))

    def test_brine_viscosity_decreases_with_temperature(self):
        vals = [brine_viscosity(t, 250_000.0) for t in (20.0, 50.0, 90.0)]
        assert all(b < a for a, b in zip(vals, vals[1:]))
        assert all(v > 0 for v in vals)

    def test_z_factor_ideal_gas_limit(self):
        """При низком давлении z -> 1."""
        assert z_factor_dak(1e5, 40.0, 0.75) == pytest.approx(1.0, abs=0.02)

    def test_z_factor_reasonable_range(self):
        """z в физичных пределах на промысловой сетке."""
        for p in (1e6, 5e6, 1e7, 2e7, 4e7):
            for t in (30.0, 60.0, 100.0):
                z = z_factor_dak(p, t, 0.75)
                assert 0.5 < z < 1.6, f"z={z} при p={p}, t={t}"

    def test_gas_density_increases_with_pressure(self):
        vals = [gas_density(p, 60.0, 0.75) for p in (1e6, 5e6, 1e7, 2e7)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_oil_viscosity_real_and_positive_across_grid(self):
        """Beggs-Robinson не должна давать комплексных чисел.

        Корреляция калибрована на 70-295 F. За её пределами (лёгкая
        нефть, T > 120 C) mu_od уходит в минус, и дробная степень
        от отрицательного основания давала complex -- расчёт падал
        с TypeError уже на этапе профиля давления.
        """
        for t_c in (20.0, 60.0, 100.0, 140.0, 180.0):
            for gamma_o in (0.78, 0.83, 0.86, 0.92):
                for rs in (0.0, 50.0, 150.0):
                    mu = oil_viscosity_beggs_robinson(t_c, gamma_o, rs)
                    assert isinstance(mu, float)
                    assert mu > 0.0

    def test_oil_viscosity_decreases_with_temperature(self):
        vals = [oil_viscosity_beggs_robinson(t, 0.86, 50.0)
                for t in (20.0, 40.0, 70.0)]
        assert all(b <= a for a, b in zip(vals, vals[1:]))
