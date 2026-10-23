from __future__ import annotations

from dataclasses import dataclass

DWAT_DP_BELOW_PB = -0.6e-6
DWAT_DP_ABOVE_PB = 0.2e-6


@dataclass
class WaxProperties:
    wat_stock_tank_c: float
    wax_content_pct: float = 5.0
    pour_point_c: float | None = None


def wat_at_pressure(wax: WaxProperties, p_pa: float, pb_pa: float) -> float:
    wat_at_pb = wax.wat_stock_tank_c + DWAT_DP_BELOW_PB * pb_pa

    if p_pa <= pb_pa:
        return wax.wat_stock_tank_c + DWAT_DP_BELOW_PB * p_pa

    return wat_at_pb + DWAT_DP_ABOVE_PB * (p_pa - pb_pa)


def wax_onset_depth(
    depths: list[float],
    temps: list[float],
    pressures: list[float],
    wax: WaxProperties,
    pb_pa: float,
) -> tuple[float | None, list[float]]:
    wat_profile = [wat_at_pressure(wax, p, pb_pa) for p in pressures]

    delta = [t - w for t, w in zip(temps, wat_profile)]
    if not delta or not depths:
        return None, wat_profile

    if all(d > 0.0 for d in delta):
        return None, wat_profile
    if all(d <= 0.0 for d in delta):
        return depths[-1], wat_profile

    for i in range(1, len(delta)):
        if delta[i - 1] <= 0.0 < delta[i]:
            d0, d1 = delta[i - 1], delta[i]
            z0, z1 = depths[i - 1], depths[i]
            frac = -d0 / (d1 - d0) if abs(d1 - d0) > 1e-12 else 0.0
            return z0 + frac * (z1 - z0), wat_profile

    cold_depths = [z for z, d in zip(depths, delta) if d <= 0.0]
    return max(cold_depths), wat_profile


def wax_deposition_severity(
    depths: list[float],
    temps: list[float],
    wat_profile: list[float],
    onset_depth: float | None,
    wax: WaxProperties,
) -> float:
    if onset_depth is None:
        return 0.0
    total = depths[-1] if depths else 1.0
    zone_frac = max(0.0, min(onset_depth / max(total, 1e-6), 1.0))

    driving = [
        max(w - t, 0.0)
        for z, t, w in zip(depths, temps, wat_profile)
        if z <= onset_depth
    ]
    mean_dt = sum(driving) / len(driving) if driving else 0.0

    dt_term = min(mean_dt / 20.0, 1.0)
    wax_term = min(wax.wax_content_pct / 10.0, 1.0)

    return min(zone_frac * (0.6 * dt_term + 0.4 * wax_term), 1.0)


def recommend_wax_treatment(onset_depth: float | None, severity: float,
                            max_scraper_depth_m: float = 1500.0) -> str:
    if onset_depth is None or severity < 0.05:
        return "не требуется"
    if severity < 0.2:
        return "профилактика: скребкование по графику"
    if onset_depth <= max_scraper_depth_m:
        if severity < 0.5:
            return f"скребок (зона с {onset_depth:.0f} м) + КР-01 периодически"
        return f"греющий кабель до {onset_depth:.0f} м + КР-01"
    if severity < 0.5:
        return f"растворитель КР-01/СГБ, закачка на {onset_depth:.0f} м"
    return (
        f"глубокая зона ({onset_depth:.0f} м, вне досягаемости скребка): "
        f"растворитель СГБ + пересмотр режима насоса"
    )
