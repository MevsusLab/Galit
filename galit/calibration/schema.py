"""Versioned tabular contracts for offline calibration history."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

SCHEMA_VERSION = "1.0"

# Every production snapshot is self-contained: constructor defaults are forbidden.
SNAPSHOT_COLUMNS = (
    "schema_version", "well_id", "timestamp", "source", "quality",
    "depth_m", "tubing_id_m", "inclination_deg", "roughness_m",
    "q_oil_m3d", "q_water_m3d", "gor_m3m3", "gamma_oil", "gamma_gas",
    "salinity_ppm", "surface_tension_n_m", "t_surface_c",
    "geothermal_grad_k_m", "k_earth_w_mk", "alpha_earth_m2_s",
    "u_to_w_m2k", "r_to_m", "r_wb_m", "cp_fluid_j_kgk",
    "production_days", "na_mg_l", "cl_mg_l", "ca_mg_l", "mg_mg_l",
    "k_mg_l", "hco3_mg_l", "so4_mg_l", "ph", "water_t_c",
    "water_p_pa", "wat_stock_tank_c", "wax_content_pct", "co2_mol_frac",
    "inhibitor_efficiency", "lift_type", "p_wellhead_pa",
)
OPTIONAL_SNAPSHOT_COLUMNS = (
    "target_temperature_c", "target_pressure_pa", "measurement_depth_m",
    "target_wax_onset_m", "target_corrosion_mm_y", "event_label",
    "risk_label", "is_synthetic",
)
LEAKAGE_COLUMNS = frozenset({
    "target_temperature_c", "target_pressure_pa", "target_wax_onset_m",
    "target_corrosion_mm_y", "event_label", "risk_label", "event_type",
    "treatment_type", "treatment_result", "post_temperature_c",
    "post_pressure_pa", "post_corrosion_mm_y", "outcome",
})

@dataclass(frozen=True)
class InputProvenance:
    source: str
    quality: str
    timestamp: datetime
    schema_version: str = SCHEMA_VERSION
    record_id: str | None = None

@dataclass(frozen=True)
class WellSnapshot:
    values: dict[str, Any]
    provenance: InputProvenance

    @property
    def well_id(self) -> str:
        return str(self.values["well_id"])

    @property
    def timestamp(self) -> datetime:
        return self.provenance.timestamp

@dataclass(frozen=True)
class Observation:
    well_id: str
    timestamp: datetime
    kind: str
    value: float
    unit: str
    source: str
    quality: str

@dataclass(frozen=True)
class Event:
    well_id: str
    timestamp: datetime
    event_type: str
    label: float | None
    source: str
    quality: str

@dataclass(frozen=True)
class Treatment:
    well_id: str
    timestamp: datetime
    treatment_type: str
    result: str | None
    source: str
    quality: str

@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("History validation failed:\n- " + "\n- ".join(self.errors))

@dataclass
class HistoryDataset:
    snapshots: list[WellSnapshot]
    observations: list[Observation] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    treatments: list[Treatment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset_hash: str = ""
    synthetic: bool = False


def parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return dt.astimezone(timezone.utc)


def finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
