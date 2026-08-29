"""Единая подготовка данных для карты фонда без привязки к UI.

Координаты необязательны. Некорректные точки и оценки потерь не мешают
показу остальных скважин и явно учитываются в сводке.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .integrated import DEFAULT_RISK_POLICY
from .master_plan import DiagnosedWell, generate_master_plan

STATUS_LABELS = {
    "green": "норма",
    "yellow": "растущий риск",
    "red": "критический",
}
STATUS_COLORS = {
    "green": "#2E7D32",
    "yellow": "#B26A00",
    "red": "#C62828",
}


@dataclass(frozen=True)
class FieldMapPoint:
    well: str
    latitude: float
    longitude: float
    site: str | None
    cluster: str | None
    risk: float
    status: str
    status_label: str
    color: str
    dominant: str
    possible_oil_loss_m3d: float | None
    marker_size: float


@dataclass(frozen=True)
class FieldMapSummary:
    total_wells: int
    mapped_wells: int
    missing_coordinates: int
    invalid_coordinates: int
    missing_losses: int
    counts_by_status: dict[str, int]
    possible_oil_loss_m3d: float | None


@dataclass(frozen=True)
class FieldMapData:
    points: tuple[FieldMapPoint, ...]
    summary: FieldMapSummary


def map_status(risk: object) -> str:
    """Translate integrated risk into the shared traffic-light map status."""
    value = _finite(risk)
    if value is None:
        return "green"
    if value >= DEFAULT_RISK_POLICY.risk_critical:
        return "red"
    if value >= DEFAULT_RISK_POLICY.risk_warn:
        return "yellow"
    return "green"


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def valid_coordinates(latitude: object, longitude: object) -> bool:
    lat, lon = _finite(latitude), _finite(longitude)
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def _marker_size(loss: float | None, max_loss: float) -> float:
    """Bounded area-like scaling: missing/zero loss remains visible."""
    if loss is None or loss <= 0 or max_loss <= 0:
        return 12.0
    return round(12.0 + 24.0 * math.sqrt(loss / max_loss), 2)


def prepare_field_map(items: Iterable[DiagnosedWell]) -> FieldMapData:
    """Create safe map points and an audit-friendly aggregate summary."""
    rows = list(items)
    plan = generate_master_plan(rows, include_low_risk=True)
    losses = {task.well.casefold(): _finite(task.possible_oil_loss.central_m3d)
              for task in plan.tasks}
    candidates: list[tuple[DiagnosedWell, float, float, float | None]] = []
    missing_coordinates = invalid_coordinates = 0
    for item in rows:
        lat, lon = getattr(item.case, "latitude", None), getattr(item.case, "longitude", None)
        if lat is None or lon is None:
            missing_coordinates += 1
            continue
        if not valid_coordinates(lat, lon):
            invalid_coordinates += 1
            continue
        candidates.append((item, float(lat), float(lon), losses.get(item.case.name.casefold())))

    usable_losses = [loss for *_, loss in candidates if loss is not None and loss >= 0]
    max_loss = max(usable_losses, default=0.0)
    points: list[FieldMapPoint] = []
    counts = {label: 0 for label in STATUS_LABELS.values()}
    for item, lat, lon, loss in candidates:
        risk = _finite(item.diagnosis.integrated_risk)
        bounded_risk = min(1.0, max(0.0, risk or 0.0))
        status = map_status(risk)
        counts[STATUS_LABELS[status]] += 1
        points.append(FieldMapPoint(
            well=item.case.name, latitude=lat, longitude=lon,
            site=getattr(item.case, "site", None), cluster=getattr(item.case, "cluster", None),
            risk=bounded_risk, status=status, status_label=STATUS_LABELS[status],
            color=STATUS_COLORS[status], dominant=item.diagnosis.dominant,
            possible_oil_loss_m3d=loss if loss is not None and loss >= 0 else None,
            marker_size=_marker_size(loss, max_loss),
        ))
    total_loss = round(sum(p.possible_oil_loss_m3d for p in points
                           if p.possible_oil_loss_m3d is not None), 1) if usable_losses else None
    return FieldMapData(
        points=tuple(points),
        summary=FieldMapSummary(
            total_wells=len(rows), mapped_wells=len(points),
            missing_coordinates=missing_coordinates, invalid_coordinates=invalid_coordinates,
            missing_losses=sum(point.possible_oil_loss_m3d is None for point in points),
            counts_by_status=counts, possible_oil_loss_m3d=total_loss,
        ),
    )


__all__ = [
    "FieldMapData", "FieldMapPoint", "FieldMapSummary", "STATUS_COLORS",
    "STATUS_LABELS", "map_status", "prepare_field_map", "valid_coordinates",
]
