"""Unit tests for Hybrid Physics-Informed Wellbore Digital Twin (Residual Learning)."""
import pytest
from galit.wellbore import (
    WellGeometry,
    ProductionRate,
    FluidProperties,
    ThermalParams,
)
from galit.wax import WaxProperties
from galit.hybrid_wellbore import (
    WellboreMeasurement,
    HybridProfileResult,
    calculate_hybrid_profiles,
    generate_realistic_measurements,
    MODEL_VERSION,
)


def sample_well_components():
    geom = WellGeometry(depth_m=3200.0, tubing_id_m=0.062, inclination_deg=15.0)
    rate = ProductionRate(q_oil_m3d=8.0, q_water_m3d=72.0, gor_m3m3=65.0)
    fluid = FluidProperties(gamma_oil=0.86, gamma_gas=0.78, salinity_ppm=290000.0)
    tp = ThermalParams(t_surface_c=8.0, geothermal_grad=0.033, u_to=15.0, production_days=400.0)
    wax = WaxProperties(wat_stock_tank_c=34.0, wax_content_pct=6.5)
    return geom, rate, fluid, tp, wax


def test_hybrid_profiles_no_measurements():
    """Без замеров гибридный профиль точно равен физическому baseline."""
    geom, rate, fluid, tp, wax = sample_well_components()
    res = calculate_hybrid_profiles("Well-1", geom, rate, fluid, tp, wax, p_wellhead_pa=1.4e6, measurements=None)

    assert isinstance(res, HybridProfileResult)
    assert res.well == "Well-1"
    assert res.model_version == MODEL_VERSION
    assert len(res.depths) == 61
    # Невязка везде нулевая
    assert all(abs(dt) < 1e-6 for dt in res.delta_temps)
    assert all(abs(dp) < 1e-6 for dp in res.delta_pressures)
    assert res.temps_hybrid == res.temps_physics
    assert res.wax_onset_physics_m == res.wax_onset_hybrid_m
    assert res.wax_onset_shift_m == 0.0 or res.wax_onset_shift_m is None


def test_hybrid_profiles_with_cooling_shifts_wax_onset():
    """При остывании устья (холодный грунт, парафиновая корка) зона АСПО смещается глубже в ствол."""
    geom, rate, fluid, tp, wax = sample_well_components()

    # Сначала считаем физический baseline
    base = calculate_hybrid_profiles("Well-1", geom, rate, fluid, tp, wax, p_wellhead_pa=1.4e6)
    t_wh_phys = base.temps_physics[0]

    # Реальный замер на устье на 4.5 °C холоднее теории Рэмея
    t_wh_real = round(t_wh_phys - 4.5, 2)
    measurements = [
        WellboreMeasurement(depth_m=0.0, temperature_c=t_wh_real, pressure_pa=1.4e6, sensor_type="surface"),
        WellboreMeasurement(depth_m=3200.0, temperature_c=base.temps_physics[-1], pressure_pa=base.pressures_physics[-1] + 1.2e6, sensor_type="tms_bottomhole"),
    ]

    res = calculate_hybrid_profiles("Well-1", geom, rate, fluid, tp, wax, p_wellhead_pa=1.4e6, measurements=measurements)

    # В точке устья гибридная температура равна реальному замеру
    assert abs(res.temps_hybrid[0] - t_wh_real) < 0.1
    # Невязка на устье отрицательная (поток холоднее теории)
    assert res.delta_temps[0] < -4.0

    # Если АСПО есть, глубина начала отложений АСПО сместилась вглубь ствола
    if base.wax_onset_physics_m is not None:
        assert res.wax_onset_hybrid_m is not None
        assert res.wax_onset_hybrid_m > base.wax_onset_physics_m
        assert res.wax_onset_shift_m > 0
        print(f"Shift detected: {base.wax_onset_physics_m:.1f} m -> {res.wax_onset_hybrid_m:.1f} m (delta = +{res.wax_onset_shift_m:.1f} m)")


def test_realistic_measurements_generator():
    """Генератор синтетических замеров выдает корректные устьевой и забойный сенсоры."""
    meas = generate_realistic_measurements("Well-Test", 3000.0, 8.0, 0.033)
    assert len(meas) == 2
    assert meas[0].depth_m == 0.0
    assert meas[0].sensor_type == "surface"
    assert meas[1].depth_m > 2000.0
    assert meas[1].sensor_type == "tms_bottomhole"
    assert meas[1].pressure_pa > 30e6


def test_multi_sensor_survey_interpolation():
    """Три замера с промежуточным глубинным манометром/термометром отрабатывают без ошибок."""
    geom, rate, fluid, tp, wax = sample_well_components()
    measurements = [
        WellboreMeasurement(depth_m=0.0, temperature_c=28.0, pressure_pa=1.4e6),
        WellboreMeasurement(depth_m=1200.0, temperature_c=61.0, pressure_pa=14.5e6, sensor_type="survey"),
        WellboreMeasurement(depth_m=2800.0, temperature_c=98.0, pressure_pa=32.0e6, sensor_type="tms_bottomhole"),
    ]
    res = calculate_hybrid_profiles("Well-Survey", geom, rate, fluid, tp, wax, measurements=measurements)
    assert len(res.temps_hybrid) == len(res.depths)
    # Проверяем монотонность давления с глубиной
    for i in range(1, len(res.pressures_hybrid)):
        assert res.pressures_hybrid[i] >= res.pressures_hybrid[i - 1]
