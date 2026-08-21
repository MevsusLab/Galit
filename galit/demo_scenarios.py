"""Deterministic, explicitly illustrative competition-demo scenarios.

The cases use the public :class:`WellCase` structure and every displayed
result is calculated by :func:`galit.integrated.diagnose`.  They are teaching
inputs, not fitted wells and not evidence of field accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .integrated import DataProvenance, DiagnosisResult, WellCase, diagnose
from .scale import WaterAnalysis
from .wax import WaxProperties
from .wellbore import FluidProperties, ProductionRate, ThermalParams, WellGeometry

DEMO_LABELS = ("synthetic", "illustrative", "not field validated")


@dataclass(frozen=True)
class DemoScenario:
    key: str
    title: str
    educational_focus: str
    case: WellCase
    interpretation_note: str


@dataclass(frozen=True)
class SensitivityPoint:
    co2_mol_frac: float
    corrosion_rate_mm_yr: float
    corrosion_severity: float
    dominant: str


@dataclass(frozen=True)
class Counterfactual:
    action: str
    before: DiagnosisResult
    after: DiagnosisResult


@dataclass(frozen=True)
class DemoScenarioResult:
    scenario: DemoScenario
    diagnosis: DiagnosisResult
    actual_dominant: str
    labels: tuple[str, ...] = DEMO_LABELS
    co2_sensitivity: tuple[SensitivityPoint, ...] = ()
    counterfactual: Counterfactual | None = None


def _provenance() -> DataProvenance:
    fields = {
        "geometry.depth_m", "geometry.tubing_id_m",
        "rate.q_oil_m3d", "rate.q_water_m3d", "rate.gor_m3m3",
        "thermal.t_surface_c", "thermal.geothermal_grad", "thermal.u_to",
        "p_wellhead_pa", "water.ions_mg_l", "water.ph", "water.t_c",
        "water.p_pa", "fluid.salinity_ppm", "wax.wat_stock_tank_c",
        "wax.wax_content_pct", "co2_mol_frac", "inhibitor_efficiency",
    }
    return DataProvenance(sources={name: "synthetic" for name in fields})


def _case(
    name: str,
    ions: dict[str, float],
    *,
    ph: float = 6.0,
    salinity_ppm: float = 120_000.0,
    wat_c: float = 18.0,
    wax_pct: float = 2.0,
    co2: float = 0.002,
    inhibitor: float = 0.90,
    q_oil: float = 12.0,
    q_water: float = 68.0,
    gor: float = 70.0,
    depth: float = 2800.0,
    tubing: float = 0.062,
    surface_t: float = 10.0,
    gradient: float = 0.032,
    u_to: float = 14.0,
    p_wh: float = 1.5e6,
) -> WellCase:
    return WellCase(
        name=name,
        geometry=WellGeometry(depth_m=depth, tubing_id_m=tubing),
        rate=ProductionRate(q_oil_m3d=q_oil, q_water_m3d=q_water, gor_m3m3=gor),
        fluid=FluidProperties(gamma_oil=0.86, gamma_gas=0.75, salinity_ppm=salinity_ppm),
        thermal=ThermalParams(t_surface_c=surface_t, geothermal_grad=gradient,
                              u_to=u_to, production_days=730.0),
        water=WaterAnalysis(ions_mg_l=ions, ph=ph, t_c=45.0, p_pa=5e6),
        wax=WaxProperties(wat_stock_tank_c=wat_c, wax_content_pct=wax_pct),
        co2_mol_frac=co2,
        inhibitor_efficiency=inhibitor,
        p_wellhead_pa=p_wh,
        provenance=_provenance(),
    )


def competition_scenarios() -> tuple[DemoScenario, ...]:
    """Return fresh deterministic inputs for the five teaching archetypes."""
    halite = _case(
        "DEMO-HALITE-01",
        {"Na": 155_000.0, "Cl": 285_000.0, "Ca": 34_000.0,
         "Mg": 4_000.0, "K": 2_000.0, "HCO3": 50.0, "SO4": 80.0},
        ph=5.4, salinity_ppm=480_000.0, wat_c=12.0, wax_pct=1.0,
    )
    calcite = _case(
        "DEMO-CALCITE-01",
        {"Na": 1_200.0, "Cl": 1_800.0, "Ca": 1_100.0,
         "Mg": 120.0, "HCO3": 2_200.0, "SO4": 150.0},
        ph=7.7, salinity_ppm=7_000.0, wat_c=12.0, wax_pct=1.0,
        co2=0.0005,
    )
    wax = _case(
        "DEMO-WAX-01",
        {"Na": 2_000.0, "Cl": 3_000.0, "Ca": 200.0,
         "Mg": 80.0, "HCO3": 100.0, "SO4": 100.0},
        ph=5.5, salinity_ppm=6_000.0, wat_c=52.0, wax_pct=12.0,
        co2=0.0002, q_oil=8.0, q_water=12.0, surface_t=5.0, u_to=25.0,
    )
    corrosion = _case(
        "DEMO-CORROSION-01",
        {"Na": 8_000.0, "Cl": 13_000.0, "Ca": 800.0,
         "Mg": 200.0, "HCO3": 80.0, "SO4": 100.0},
        ph=5.2, salinity_ppm=22_000.0, wat_c=10.0, wax_pct=1.0,
        co2=0.04, inhibitor=0.0, q_oil=8.0, q_water=92.0, tubing=0.050,
    )
    mixed = _case(
        "DEMO-MIXED-CONFLICT-01",
        {"Na": 4_000.0, "Cl": 8_000.0, "Ca": 2_500.0,
         "Mg": 250.0, "HCO3": 2_600.0, "SO4": 120.0},
        ph=7.1, salinity_ppm=18_000.0, wat_c=36.0, wax_pct=7.0,
        co2=0.018, inhibitor=0.25, q_oil=10.0, q_water=70.0,
        surface_t=7.0, u_to=20.0,
    )
    return (
        DemoScenario("halite", "Преимущественно галит", "halite", halite,
                     "Скрининг пересыщенного Na-Cl рассола; модель Питцера всё ещё нужна."),
        DemoScenario("calcite", "Преимущественно кальцит", "calcite", calcite,
                     "Учебный карбонатный состав в области Stiff-Davis, не промысловая проба."),
        DemoScenario("wax", "Преимущественно АСПО", "wax", wax,
                     "Холодный малодебитный режим показывает пересечение T и WAT."),
        DemoScenario("corrosion", "Преимущественно CO2-коррозия", "corrosion", corrosion,
                     "Неингибированный screening-кейс; CO2 задан, а не измерен."),
        DemoScenario("mixed_conflict", "Смешанный конфликт технологий", "mixed", mixed,
                     "Несколько механизмов конкурируют; dominant сообщается отдельно."),
    )


def _co2_sensitivity(case: WellCase) -> tuple[SensitivityPoint, ...]:
    points = []
    for fraction in (0.002, 0.01, 0.03):
        result = diagnose(replace(case, co2_mol_frac=fraction))
        points.append(SensitivityPoint(
            fraction, result.corrosion["rate_mm_yr"],
            result.severity["corrosion"], result.dominant,
        ))
    return tuple(points)


def run_competition_scenarios() -> tuple[DemoScenarioResult, ...]:
    """Calculate all competition scenarios with the production diagnosis core."""
    output = []
    for scenario in competition_scenarios():
        diagnosis = diagnose(scenario.case)
        sensitivity = _co2_sensitivity(scenario.case) if scenario.key in {
            "corrosion", "mixed_conflict"
        } else ()
        counterfactual = None
        if scenario.key in {"corrosion", "mixed_conflict"}:
            treated = diagnose(replace(scenario.case, inhibitor_efficiency=0.90))
            counterfactual = Counterfactual(
                "CO2 corrosion inhibitor efficiency set to 90% (illustrative assumption)",
                diagnosis, treated,
            )
        output.append(DemoScenarioResult(
            scenario=scenario,
            diagnosis=diagnosis,
            actual_dominant=diagnosis.dominant,
            co2_sensitivity=sensitivity,
            counterfactual=counterfactual,
        ))
    return tuple(output)
