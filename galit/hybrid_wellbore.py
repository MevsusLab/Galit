from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import math
from typing import Any, Sequence

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

from .wellbore import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
    pressure_profile,
    temperature_profile,
)
from .wax import WaxProperties, wax_onset_depth
from .fluids import bubble_point_standing

MODEL_VERSION = "galit-hybrid-digital-twin-1.0"
DISCLAIMER = (
    "Гибридный профиль ствола (Physics-Informed Residual Learning) объединяет "
    "аналитическую физику Рэмея / Беггса-Брилла и машинное обучение невязок "
    "по замерам датчиков. Требует наличия калиброванных устьевых и забойных "
    "датчиков ТМС/манометрии. Не является заменой гидродинамических исследований (ГДИС)."
)


@dataclass(frozen=True)
class WellboreMeasurement:
    depth_m: float
    temperature_c: float | None = None
    pressure_pa: float | None = None
    sensor_type: str = "surface"
    timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        if self.timestamp:
            res["timestamp"] = self.timestamp.isoformat()
        return res


@dataclass(frozen=True)
class HybridProfileResult:
    well: str
    depths: list[float]

    temps_physics: list[float]
    pressures_physics: list[float]

    temps_hybrid: list[float]
    pressures_hybrid: list[float]

    delta_temps: list[float]
    delta_pressures: list[float]

    measurements: tuple[WellboreMeasurement, ...]

    wax_onset_physics_m: float | None
    wax_onset_hybrid_m: float | None
    wax_onset_shift_m: float | None
    wat_profile_physics: list[float]
    wat_profile_hybrid: list[float]

    temp_residual_max_c: float
    pressure_residual_max_mpa: float
    model_version: str = MODEL_VERSION
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "well": self.well,
            "depths": self.depths,
            "temps_physics": self.temps_physics,
            "pressures_physics": self.pressures_physics,
            "temps_hybrid": self.temps_hybrid,
            "pressures_hybrid": self.pressures_hybrid,
            "delta_temps": self.delta_temps,
            "delta_pressures": self.delta_pressures,
            "measurements": [m.to_dict() for m in self.measurements],
            "wax_onset_physics_m": self.wax_onset_physics_m,
            "wax_onset_hybrid_m": self.wax_onset_hybrid_m,
            "wax_onset_shift_m": self.wax_onset_shift_m,
            "wat_profile_physics": self.wat_profile_physics,
            "wat_profile_hybrid": self.wat_profile_hybrid,
            "temp_residual_max_c": self.temp_residual_max_c,
            "pressure_residual_max_mpa": self.pressure_residual_max_mpa,
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
        }


def _fit_predict_residual_curve(
    depths: list[float],
    physics_values: list[float],
    measured_points: list[tuple[float, float]],
    is_pressure: bool = False,
) -> list[float]:
    n = len(depths)
    if not measured_points:
        return [0.0] * n

    total_depth = max(depths[-1], 1.0)
    measured_points.sort(key=lambda x: x[0])

    anchors: list[tuple[float, float]] = []
    for d_meas, val_meas in measured_points:
        idx = min(range(n), key=lambda i: abs(depths[i] - d_meas))
        val_phys = physics_values[idx]
        anchors.append((d_meas, val_meas - val_phys))

    if len(anchors) == 1:
        z0, d0 = anchors[0]
        residuals = []
        for z in depths:
            factor = math.exp(-abs(z - z0) / (0.35 * total_depth))
            residuals.append(d0 * factor)
        return residuals

    if _SKLEARN_AVAILABLE and len(anchors) >= 3:
        X_train = []
        y_train = []
        for z_a, res_a in anchors:
            idx = min(range(n), key=lambda i: abs(depths[i] - z_a))
            z_norm = z_a / total_depth
            phys_v = physics_values[idx]
            grad = (physics_values[min(idx + 1, n - 1)] - physics_values[max(idx - 1, 0)]) / max(1.0, depths[min(idx + 1, n - 1)] - depths[max(idx - 1, 0)])
            X_train.append([z_norm, phys_v, grad, z_norm ** 2])
            y_train.append(res_a)

        X_all = []
        for i, z in enumerate(depths):
            z_norm = z / total_depth
            phys_v = physics_values[i]
            grad = (physics_values[min(i + 1, n - 1)] - physics_values[max(i - 1, 0)]) / max(1.0, depths[min(i + 1, n - 1)] - depths[max(i - 1, 0)])
            X_all.append([z_norm, phys_v, grad, z_norm ** 2])

        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        pred = model.predict(X_all).tolist()

        residuals = list(pred)
        for z_a, target_res in anchors:
            idx = min(range(n), key=lambda i: abs(depths[i] - z_a))
            err = target_res - residuals[idx]

            for i, z in enumerate(depths):
                w = math.exp(-((z - z_a) / (0.15 * total_depth)) ** 2)
                residuals[i] += err * w
        return residuals

    z1, d1 = anchors[0]
    z2, d2 = anchors[-1]
    span = max(z2 - z1, 1e-6)

    residuals = []
    for z in depths:
        rel = min(max((z - z1) / span, 0.0), 1.0)
        if is_pressure:
            shape = rel ** 1.3
        else:
            shape = 1.0 - (1.0 - rel) ** 1.8
        delta = d1 + (d2 - d1) * shape
        residuals.append(delta)

    return residuals


def calculate_hybrid_profiles(
    well_name: str,
    geom: WellGeometry,
    rate: ProductionRate,
    fluid: FluidProperties,
    tp: ThermalParams,
    wax: WaxProperties,
    p_wellhead_pa: float = 1.2e6,
    measurements: Sequence[WellboreMeasurement] | None = None,
    n_nodes: int = 60,
) -> HybridProfileResult:

    depths, temps_phys, _ = temperature_profile(geom, rate, fluid, tp, n_nodes=n_nodes)
    pressures_phys = pressure_profile(geom, rate, fluid, depths, temps_phys, p_wellhead_pa=p_wellhead_pa)

    meas_list = list(measurements or [])

    temp_pts = [(m.depth_m, m.temperature_c) for m in meas_list if m.temperature_c is not None]

    press_pts = [(m.depth_m, m.pressure_pa) for m in meas_list if m.pressure_pa is not None]

    delta_t = _fit_predict_residual_curve(depths, temps_phys, temp_pts, is_pressure=False)
    delta_p = _fit_predict_residual_curve(depths, pressures_phys, press_pts, is_pressure=True)

    if not temp_pts:
        temps_hybrid = list(temps_phys)
    else:
        temps_hybrid = [round(t + dt, 3) for t, dt in zip(temps_phys, delta_t)]

    if not press_pts:
        pressures_hybrid = list(pressures_phys)
    else:
        pressures_hybrid = [round(max(p + dp, 1e5), 1) for p, dp in zip(pressures_phys, delta_p)]

    t_res_c = tp.t_surface_c + tp.geothermal_grad * geom.depth_m
    pb = bubble_point_standing(rate.gor_m3m3, fluid.gamma_gas, t_res_c, fluid.gamma_oil)

    onset_phys, wat_prof_phys = wax_onset_depth(depths, temps_phys, pressures_phys, wax, pb)
    onset_hyb, wat_prof_hyb = wax_onset_depth(depths, temps_hybrid, pressures_hybrid, wax, pb)

    shift = (onset_hyb - onset_phys) if (onset_phys is not None and onset_hyb is not None) else None

    max_dt = max(abs(dt) for dt in delta_t) if delta_t else 0.0
    max_dp_mpa = (max(abs(dp) for dp in delta_p) / 1e6) if delta_p else 0.0

    return HybridProfileResult(
        well=well_name,
        depths=depths,
        temps_physics=temps_phys,
        pressures_physics=pressures_phys,
        temps_hybrid=temps_hybrid,
        pressures_hybrid=pressures_hybrid,
        delta_temps=delta_t,
        delta_pressures=delta_p,
        measurements=tuple(meas_list),
        wax_onset_physics_m=onset_phys,
        wax_onset_hybrid_m=onset_hyb,
        wax_onset_shift_m=shift,
        wat_profile_physics=wat_prof_phys,
        wat_profile_hybrid=wat_prof_hyb,
        temp_residual_max_c=round(max_dt, 3),
        pressure_residual_max_mpa=round(max_dp_mpa, 4),
    )


def generate_realistic_measurements(
    well_name: str,
    depth_m: float,
    t_surface_c: float,
    geothermal_grad: float,
    delta_t_wellhead: float = -3.2,
    delta_p_bottomhole_mpa: float = 0.85,
    p_wellhead_pa: float = 1.4e6,
) -> list[WellboreMeasurement]:
    t_reservoir = t_surface_c + geothermal_grad * depth_m

    t_wh_meas = (t_surface_c + 24.0) + delta_t_wellhead

    p_tms_meas = 32.5e6 + delta_p_bottomhole_mpa * 1e6
    t_tms_meas = t_reservoir - 1.5

    return [
        WellboreMeasurement(
            depth_m=0.0,
            temperature_c=round(t_wh_meas, 2),
            pressure_pa=p_wellhead_pa,
            sensor_type="surface",
        ),
        WellboreMeasurement(
            depth_m=round(depth_m * 0.85, 1),
            temperature_c=round(t_tms_meas, 2),
            pressure_pa=round(p_tms_meas, 1),
            sensor_type="tms_bottomhole",
        ),
    ]
