"""Versioned engineering screening for compatibility of two measured waters.

The module deliberately does not invent missing chemistry.  It linearly mixes
reported ion concentrations and screens mineral saturation; it is not an
aqueous speciation, precipitation-path, kinetic, or transport simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from .scale import MW, WaterAnalysis, halite_saturation_index, stiff_davis_index_checked

MODEL_VERSION = "water-compatibility-screening/1.0"
MAX_RATIOS = 1001
MAX_PROFILE_POINTS = 2000
SUPPORTED_MINERALS = ("calcite", "barite", "gypsum", "halite")
_CHARGES = {"Na": 1, "K": 1, "Ca": 2, "Mg": 2, "Ba": 2, "Sr": 2,
            "Fe": 2, "Cl": -1, "SO4": -2, "HCO3": -1, "CO3": -2}
# Thermodynamic Ksp at 25 C. Used only for explicitly labelled screening.
_KSP_25C = {"barite": 1.08e-10, "gypsum": 2.4e-5}
_REQUIRED = {
    "calcite": ("Ca", "HCO3_or_CO3"),
    "barite": ("Ba", "SO4"),
    "gypsum": ("Ca", "SO4"),
    "halite": ("Na", "Cl"),
}


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class CompatibilityWater:
    """A supplied water analysis; ions are mg/L, temperature C, pressure Pa."""
    ions_mg_l: Mapping[str, float]
    ph: float
    t_c: float
    p_pa: float
    name: str = "water"

    def __post_init__(self) -> None:
        ions = dict(self.ions_mg_l)
        if len(ions) > len(MW):
            raise ValueError("too many ion entries")
        unknown = sorted(set(ions) - set(MW))
        if unknown:
            raise ValueError("unsupported ions: " + ", ".join(unknown))
        for ion, value in ions.items():
            value = _finite(value, f"ions_mg_l.{ion}")
            if value < 0:
                raise ValueError(f"ions_mg_l.{ion} must be non-negative")
            ions[ion] = value
        ph = _finite(self.ph, "ph")
        if not 0 <= ph <= 14:
            raise ValueError("ph must be within [0, 14]")
        if _finite(self.p_pa, "p_pa") < 0:
            raise ValueError("p_pa must be non-negative")
        _finite(self.t_c, "t_c")
        object.__setattr__(self, "ions_mg_l", MappingProxyType(ions))

    def analysis(self) -> WaterAnalysis:
        return WaterAnalysis(dict(self.ions_mg_l), self.ph, self.t_c, self.p_pa)


@dataclass(frozen=True)
class ProfilePoint:
    depth_m: float
    t_c: float
    p_pa: float

    def __post_init__(self) -> None:
        if _finite(self.depth_m, "depth_m") < 0:
            raise ValueError("depth_m must be non-negative")
        _finite(self.t_c, "t_c")
        if _finite(self.p_pa, "p_pa") < 0:
            raise ValueError("p_pa must be non-negative")


@dataclass(frozen=True)
class DoseResponsePoint:
    dose_mg_l: float
    maximum_supported_si: float

    def __post_init__(self) -> None:
        if _finite(self.dose_mg_l, "dose_mg_l") < 0:
            raise ValueError("dose_mg_l must be non-negative")
        _finite(self.maximum_supported_si, "maximum_supported_si")


@dataclass(frozen=True)
class DoseResponseCurve:
    product: str
    mineral: str
    points: Sequence[DoseResponsePoint]
    validated: bool
    validation_reference: str

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not self.product.strip() or not self.validation_reference.strip():
            raise ValueError("product and validation_reference are required")
        if self.mineral not in SUPPORTED_MINERALS:
            raise ValueError("unsupported dose-response mineral")
        if not self.validated:
            raise ValueError("dose-response curve must be validated")
        if not 2 <= len(points) <= 100:
            raise ValueError("dose-response curve requires 2..100 points")
        for left, right in zip(points, points[1:]):
            if right.dose_mg_l <= left.dose_mg_l:
                raise ValueError("dose must be strictly increasing")
            if right.maximum_supported_si < left.maximum_supported_si:
                raise ValueError("dose-response must be monotonic")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class MineralScreen:
    mineral: str
    saturation_index: float | None
    supersaturated: bool | None
    method: str
    required_inputs: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RatioScreen:
    fraction_b: float
    ratio_a_to_b: str
    minerals: Mapping[str, MineralScreen]
    risk_score: float | None
    unsafe: bool


@dataclass(frozen=True)
class UnsafeInterval:
    start_fraction_b: float
    end_fraction_b: float


@dataclass(frozen=True)
class DepositionLocation:
    mineral: str
    first_supersaturation_depth_m: float | None
    maximum_risk_depth_m: float | None
    maximum_saturation_index: float | None
    flow_direction: str


@dataclass(frozen=True)
class InhibitorRecommendation:
    status: str
    dose_mg_l: float | None
    product: str | None
    mineral: str | None
    validation_reference: str | None
    basis: str


@dataclass(frozen=True)
class CompatibilityResult:
    model_version: str
    units: Mapping[str, str]
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    ratios: tuple[RatioScreen, ...]
    dangerous_fraction_b: float | None
    dangerous_ratio_a_to_b: str | None
    dangerous_risk_score: float | None
    unsafe_intervals: tuple[UnsafeInterval, ...]
    deposition_locations: tuple[DepositionLocation, ...]
    inhibitor: InhibitorRecommendation


def default_mix_fractions(step: float = 0.01) -> tuple[float, ...]:
    step = _finite(step, "step")
    if step <= 0 or step > 1:
        raise ValueError("step must be within (0, 1]")
    count = int(math.floor(1.0 / step + 1e-12))
    values = [round(i * step, 12) for i in range(count + 1)]
    if values[-1] != 1.0:
        values.append(1.0)
    if len(values) > MAX_RATIOS:
        raise ValueError(f"at most {MAX_RATIOS} mix fractions are allowed")
    return tuple(values)


def mix_waters(water_a: CompatibilityWater, water_b: CompatibilityWater,
               fraction_b: float, *, t_c: float | None = None,
               p_pa: float | None = None) -> CompatibilityWater:
    """Conservative-volume linear mixing with hydrogen-activity pH mixing."""
    f = _finite(fraction_b, "fraction_b")
    if not 0 <= f <= 1:
        raise ValueError("fraction_b must be within [0, 1]")
    # A missing analysis is unknown, not a measured zero. At an endpoint only
    # the selected water contributes; an interior mixture needs both values.
    if f == 0:
        ions = dict(water_a.ions_mg_l)
    elif f == 1:
        ions = dict(water_b.ions_mg_l)
    else:
        ions = {
            ion: (1 - f) * water_a.ions_mg_l[ion] + f * water_b.ions_mg_l[ion]
            for ion in sorted(set(water_a.ions_mg_l) & set(water_b.ions_mg_l))
        }
    hydrogen = (1 - f) * 10 ** (-water_a.ph) + f * 10 ** (-water_b.ph)
    return CompatibilityWater(
        ions, -math.log10(hydrogen),
        (1 - f) * water_a.t_c + f * water_b.t_c if t_c is None else t_c,
        (1 - f) * water_a.p_pa + f * water_b.p_pa if p_pa is None else p_pa,
        f"{water_a.name}/{water_b.name} mixture",
    )


def _davies_gamma(charge: int, ionic_strength: float) -> float:
    root = math.sqrt(max(ionic_strength, 0.0))
    log_gamma = -0.509 * charge * charge * (root / (1 + root) - 0.3 * ionic_strength)
    return 10 ** log_gamma


def screen_minerals(water: CompatibilityWater) -> dict[str, MineralScreen]:
    analysis = water.analysis()
    present = set(water.ions_mg_l)
    results: dict[str, MineralScreen] = {}

    calcite_missing = tuple(x for x in _REQUIRED["calcite"] if
                            (x == "Ca" and "Ca" not in present) or
                            (x == "HCO3_or_CO3" and not ({"HCO3", "CO3"} & present)))
    if calcite_missing:
        results["calcite"] = MineralScreen("calcite", None, None, "Stiff-Davis screening",
                                            calcite_missing, ("insufficient supplied chemistry",))
    else:
        si, warnings = stiff_davis_index_checked(analysis)
        results["calcite"] = MineralScreen("calcite", si, si > 0, "Stiff-Davis screening",
                                            _REQUIRED["calcite"], tuple(warnings))

    mol = analysis.molarity()
    ionic_strength = analysis.ionic_strength
    general = []
    if ionic_strength > 0.5:
        general.append("Davies activity model is outside its usual I<=0.5 mol/L range; use a validated Pitzer/speciation model")
    if not 5 <= water.t_c <= 50:
        general.append("25 C Ksp used outside the 5..50 C screening range; temperature correction is not available")
    for mineral, ions in (("barite", ("Ba", "SO4")), ("gypsum", ("Ca", "SO4"))):
        missing = tuple(ion for ion in ions if ion not in present)
        if missing:
            results[mineral] = MineralScreen(mineral, None, None, "25 C Ksp + Davies activity screening",
                                              _REQUIRED[mineral], ("insufficient supplied chemistry",))
            continue
        activities = [mol.get(ion, 0.0) * _davies_gamma(_CHARGES[ion], ionic_strength) for ion in ions]
        si = -99.0 if min(activities) <= 0 else math.log10(activities[0] * activities[1] / _KSP_25C[mineral])
        results[mineral] = MineralScreen(mineral, si, si > 0, "25 C Ksp + Davies activity screening",
                                          _REQUIRED[mineral], tuple(general))

    missing = tuple(ion for ion in _REQUIRED["halite"] if ion not in present)
    if missing:
        results["halite"] = MineralScreen("halite", None, None, "simplified Pitzer halite screening",
                                           _REQUIRED["halite"], ("insufficient supplied chemistry",))
    else:
        si = halite_saturation_index(analysis)
        warning = ("simplified NaCl activity model; rigorous brine decisions require validated Pitzer modelling",)
        results["halite"] = MineralScreen("halite", si, si > 0, "simplified Pitzer halite screening",
                                           _REQUIRED["halite"], warning)
    return results


def _risk(minerals: Mapping[str, MineralScreen]) -> float | None:
    values = [item.saturation_index for item in minerals.values() if item.saturation_index is not None]
    return max(values) if values else None


def _intervals(rows: Sequence[RatioScreen]) -> tuple[UnsafeInterval, ...]:
    result: list[UnsafeInterval] = []
    start: float | None = None
    previous = 0.0
    for row in rows:
        if row.unsafe and start is None:
            start = row.fraction_b
        if not row.unsafe and start is not None:
            result.append(UnsafeInterval(start, previous))
            start = None
        previous = row.fraction_b
    if start is not None:
        result.append(UnsafeInterval(start, previous))
    return tuple(result)


def _ratio_label(fraction_b: float) -> str:
    if fraction_b == 0:
        return "1:0"
    if fraction_b == 1:
        return "0:1"
    return f"{1-fraction_b:.6g}:{fraction_b:.6g}"


def _dose(curve: DoseResponseCurve | None, mineral: str | None,
          target_si: float | None) -> InhibitorRecommendation:
    if curve is None:
        return InhibitorRecommendation("laboratory_test_required", None, None, mineral, None,
            "No validated product dose-response curve supplied; no dose inferred.")
    if mineral != curve.mineral or target_si is None:
        return InhibitorRecommendation("laboratory_test_required", None, curve.product, curve.mineral,
            curve.validation_reference, "Curve does not cover the controlling mineral or risk is unavailable.")
    for point in curve.points:
        if point.maximum_supported_si >= target_si:
            return InhibitorRecommendation("dose_from_validated_curve", point.dose_mg_l, curve.product,
                curve.mineral, curve.validation_reference, "Lowest tested dose whose validated SI capacity covers the screening SI; no interpolation.")
    return InhibitorRecommendation("laboratory_test_required", None, curve.product, curve.mineral,
        curve.validation_reference, "Screening SI exceeds the validated curve range; extrapolation forbidden.")


def evaluate_compatibility(water_a: CompatibilityWater, water_b: CompatibilityWater,
                           fractions_b: Sequence[float] | None = None,
                           profile: Sequence[ProfilePoint] | None = None,
                           flow_direction: str = "bottom_to_surface",
                           dose_response: DoseResponseCurve | None = None) -> CompatibilityResult:
    fractions = tuple(default_mix_fractions() if fractions_b is None else fractions_b)
    if not 1 <= len(fractions) <= MAX_RATIOS:
        raise ValueError(f"fractions_b requires 1..{MAX_RATIOS} values")
    checked = tuple(_finite(value, "fraction_b") for value in fractions)
    if any(not 0 <= value <= 1 for value in checked):
        raise ValueError("fractions_b values must be within [0, 1]")
    if any(right <= left for left, right in zip(checked, checked[1:])):
        raise ValueError("fractions_b must be strictly increasing and unique")
    rows: list[RatioScreen] = []
    for f in checked:
        minerals = screen_minerals(mix_waters(water_a, water_b, f))
        risk = _risk(minerals)
        rows.append(RatioScreen(f, _ratio_label(f), MappingProxyType(minerals), risk,
                                any(item.supersaturated is True for item in minerals.values())))
    available = [row for row in rows if row.risk_score is not None]
    dangerous = max(available, key=lambda row: (row.risk_score, -row.fraction_b)) if available else None

    locations: list[DepositionLocation] = []
    if profile is not None:
        points = tuple(profile)
        if not 1 <= len(points) <= MAX_PROFILE_POINTS:
            raise ValueError(f"profile requires 1..{MAX_PROFILE_POINTS} points")
        if flow_direction not in {"bottom_to_surface", "surface_to_bottom"}:
            raise ValueError("flow_direction must be bottom_to_surface or surface_to_bottom")
        if any(right.depth_m <= left.depth_m for left, right in zip(points, points[1:])):
            raise ValueError("profile depths must be strictly increasing")
        ordered = tuple(reversed(points)) if flow_direction == "bottom_to_surface" else points
        f = dangerous.fraction_b if dangerous else checked[0]
        node_screens = [(point, screen_minerals(mix_waters(water_a, water_b, f, t_c=point.t_c, p_pa=point.p_pa)))
                        for point in ordered]
        for mineral in SUPPORTED_MINERALS:
            known = [(point, screens[mineral].saturation_index) for point, screens in node_screens
                     if screens[mineral].saturation_index is not None]
            first = next((point.depth_m for point, si in known if si > 0), None)
            maximum = max(known, key=lambda item: item[1]) if known else None
            locations.append(DepositionLocation(mineral, first,
                maximum[0].depth_m if maximum else None, maximum[1] if maximum else None, flow_direction))

    controlling = None
    if dangerous:
        controlling = max((item for item in dangerous.minerals.values() if item.saturation_index is not None),
                          key=lambda item: item.saturation_index).mineral
    warnings = sorted({warning for row in rows for item in row.minerals.values() for warning in item.warnings})
    warnings.append("Supersaturation indicates thermodynamic tendency, not deposition rate or deposited mass.")
    return CompatibilityResult(
        MODEL_VERSION,
        MappingProxyType({"ion_concentration": "mg/L", "temperature": "degC", "pressure": "Pa",
                          "depth": "m", "saturation_index": "log10(IAP/Ksp)", "dose": "mg/L product"}),
        ("conservative-volume linear ion mixing", "pH mixed from hydrogen-ion activities without buffering/speciation",
         "no reaction or precipitation depletion during mixing", "unsafe means at least one available SI > 0"),
        tuple(warnings), tuple(rows), dangerous.fraction_b if dangerous else None,
        dangerous.ratio_a_to_b if dangerous else None, dangerous.risk_score if dangerous else None,
        _intervals(rows), tuple(locations), _dose(dose_response, controlling,
                                                  dangerous.risk_score if dangerous else None),
    )
