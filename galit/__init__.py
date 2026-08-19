"""ГАЛИТ -- интегрированный flow assurance для позднестадийного фонда.

Прогноз галита, кальцита, АСПО и CO2-коррозии на единой шкале риска
с подбором технологии из арсенала НГДУ и расчётом экономического эффекта.
"""

__version__ = "0.1.0"

from .corrosion import CorrosionConditions, corrosion_rate_1995, corrosion_severity
from .integrated import (
    DataProvenance,
    DataQuality,
    DataQualityError,
    DEFAULT_RISK_POLICY,
    DiagnosisResult,
    MECHANISM_WEIGHTS,
    RiskPolicy,
    ScenarioInterval,
    UncertaintyConfig,
    UncertaintyResult,
    WellCase,
    assess_quality,
    diagnose,
    rank_wells,
)
from .scale import WaterAnalysis, halite_saturation_index, stiff_davis_index
from .wax import WaxProperties, wax_onset_depth
from .wellbore import (
    FluidProperties,
    ProductionRate,
    ThermalParams,
    WellGeometry,
    pressure_profile,
    temperature_profile,
)

__all__ = [
    "CorrosionConditions",
    "DataProvenance",
    "DataQuality",
    "DataQualityError",
    "DEFAULT_RISK_POLICY",
    "DiagnosisResult",
    "MECHANISM_WEIGHTS",
    "RiskPolicy",
    "ScenarioInterval",
    "UncertaintyConfig",
    "UncertaintyResult",
    "FluidProperties",
    "ProductionRate",
    "ThermalParams",
    "WaterAnalysis",
    "WaxProperties",
    "WellCase",
    "WellGeometry",
    "assess_quality",
    "corrosion_rate_1995",
    "corrosion_severity",
    "diagnose",
    "halite_saturation_index",
    "pressure_profile",
    "rank_wells",
    "stiff_davis_index",
    "temperature_profile",
    "wax_onset_depth",
]
