"""Unified, explainable digital-twin timeline assembled from GALIT repositories.

The module is deliberately an aggregation/service layer: source repositories remain
authoritative. Only genuinely manual repair/failure/laboratory records are persisted
in the dedicated event store.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from enum import Enum
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable, Protocol
from uuid import uuid4

SCHEMA_VERSION = 1
MODEL_VERSION = "galit-digital-twin-1.0"
POLICY_VERSION = "galit-twin-health-1.0"
ASSOCIATION_DISCLAIMER = (
    "Временная последовательность показывает ассоциацию, а не причинность. "
    "Возможная связь требует инженерного подтверждения."
)


class EventCategory(str, Enum):
    PRODUCTION = "production"
    PRESSURE_TEMPERATURE = "pressure_temperature"
    WATERCUT = "watercut"
    REPAIR = "repair"
    EQUIPMENT_FAILURE = "equipment_failure"
    EQUIPMENT_TELEMETRY = "equipment_telemetry"
    TREATMENT = "treatment"
    LABORATORY = "laboratory"
    COMPLICATION = "complication"
    ECONOMIC_LOSS = "economic_loss"


class TwinNotFoundError(LookupError): pass
class TwinAmbiguousError(RuntimeError): pass
class TwinConflictError(RuntimeError): pass
class TwinStorageError(RuntimeError): pass


def aware(value: datetime | str, name: str = "timestamp") -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return value.astimezone(timezone.utc)


def normalize_well_name(value: str) -> str:
    """Strict display-name normalization; context is never discarded."""
    text = " ".join(str(value).strip().split()).casefold().replace("ё", "е")
    text = re.sub(r"\s*[-–—]\s*", "-", text)
    if not text:
        raise ValueError("well name must be non-empty")
    return text


def _context(value: str | None) -> str | None:
    return " ".join(value.strip().split()).casefold().replace("ё", "е") if value and value.strip() else None


@dataclass(frozen=True)
class WellIdentity:
    display_name: str
    field: str | None = None
    cluster: str | None = None
    site: str | None = None
    reservoir: str | None = None
    canonical_id: str | None = None
    normalized_name: str = dc_field(init=False)

    def __post_init__(self) -> None:
        display = " ".join(str(self.display_name).strip().split())
        normalized = normalize_well_name(display)
        object.__setattr__(self, "display_name", display)
        object.__setattr__(self, "normalized_name", normalized)
        for name in ("field", "cluster", "site", "reservoir"):
            value = getattr(self, name)
            object.__setattr__(self, name, " ".join(value.strip().split()) if value and value.strip() else None)
        context = "|".join(_context(getattr(self, x)) or "" for x in ("field", "cluster", "site", "reservoir"))
        digest = hashlib.sha256(f"{normalized}|{context}".encode("utf-8")).hexdigest()[:20]
        object.__setattr__(self, "canonical_id", self.canonical_id or f"well:{digest}")

    def matches_context(self, other: "WellIdentity") -> bool:
        for name in ("field", "cluster", "site", "reservoir"):
            left, right = _context(getattr(self, name)), _context(getattr(other, name))
            if left and right and left != right:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "normalized_name"}


@dataclass(frozen=True)
class MetricValue:
    value: float | int | str | bool | None
    unit: str | None = None
    quality: str = "unknown"
    as_of: datetime | None = None
    source: str | None = None
    freshness: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("metric values must be finite or null")
        if self.as_of is not None:
            object.__setattr__(self, "as_of", aware(self.as_of, "metric.as_of"))
        if self.unit is not None and not self.unit.strip():
            raise ValueError("metric unit must be non-empty when supplied")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["as_of"] = self.as_of.isoformat() if self.as_of else None
        return value


@dataclass(frozen=True)
class DigitalTwinEvent:
    event_id: str
    well: WellIdentity
    occurred_at: datetime
    recorded_at: datetime
    category: EventCategory
    event_type: str
    source: str
    source_record_id: str
    title: str
    summary: str
    severity: str = "info"
    status: str | None = None
    metrics: dict[str, MetricValue] = dc_field(default_factory=dict)
    provenance: dict[str, Any] = dc_field(default_factory=dict)
    data_quality: str = "unknown"
    links: tuple[str, ...] = ()
    metadata: dict[str, Any] = dc_field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    model_version: str = MODEL_VERSION

    def __post_init__(self) -> None:
        for name in ("event_id", "event_type", "source", "source_record_id", "title", "summary"):
            if not str(getattr(self, name)).strip(): raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "occurred_at", aware(self.occurred_at, "occurred_at"))
        object.__setattr__(self, "recorded_at", aware(self.recorded_at, "recorded_at"))
        object.__setattr__(self, "category", EventCategory(self.category))
        object.__setattr__(self, "metrics", {k: v if isinstance(v, MetricValue) else MetricValue(**v) for k, v in self.metrics.items()})
        json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "model_version": self.model_version,
            "event_id": self.event_id, "well": self.well.to_dict(),
            "occurred_at": self.occurred_at.isoformat(), "recorded_at": self.recorded_at.isoformat(),
            "category": self.category.value, "event_type": self.event_type,
            "source": self.source, "source_record_id": self.source_record_id,
            "severity": self.severity, "status": self.status, "title": self.title,
            "summary": self.summary, "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "provenance": self.provenance, "data_quality": self.data_quality,
            "links": list(self.links), "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DigitalTwinEvent":
        data = dict(value); data["well"] = WellIdentity(**data["well"])
        data["metrics"] = {k: MetricValue(**{**v, "as_of": aware(v["as_of"]) if v.get("as_of") else None}) for k, v in data.get("metrics", {}).items()}
        data["links"] = tuple(data.get("links", ()))
        return cls(**data)


TimelineEvent = DigitalTwinEvent


def stable_event_id(source: str, source_record_id: str, identity: WellIdentity) -> str:
    digest = hashlib.sha256(f"{source}|{source_record_id}|{identity.canonical_id}".encode("utf-8")).hexdigest()
    return f"evt:{digest[:32]}"


def _event(identity: WellIdentity, occurred_at: datetime, category: EventCategory,
           event_type: str, source: str, source_id: str, title: str, summary: str,
           *, metrics: dict[str, MetricValue] | None = None, severity: str = "info",
           status: str | None = None, quality: str = "unknown", metadata: dict[str, Any] | None = None,
           recorded_at: datetime | None = None) -> DigitalTwinEvent:
    return DigitalTwinEvent(stable_event_id(source, source_id, identity), identity, occurred_at,
                            recorded_at or occurred_at, category, event_type, source, source_id,
                            title, summary, severity, status, metrics or {}, {"adapter": source},
                            quality, (), metadata or {})


class ManualEventRepository:
    """Versioned UTF-8 JSON store with file locking, atomic replace and idempotency."""
    _locks: dict[str, threading.RLock] = {}; _guard = threading.Lock()
    def __init__(self, path: str | Path | None = None, lock_timeout: float = 5):
        self.path = Path(path or os.environ.get("GALIT_TWIN_EVENT_STORAGE", "data/digital_twin_events.json"))
        self.lock_timeout = lock_timeout
        with self._guard: self._lock = self._locks.setdefault(str(self.path.resolve()), threading.RLock())
    def _read(self) -> list[DigitalTwinEvent]:
        if not self.path.exists(): return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported schema")
            rows = [DigitalTwinEvent.from_dict(x) for x in payload.get("events", [])]
            if len({x.event_id for x in rows}) != len(rows): raise ValueError("duplicate event IDs")
            return rows
        except Exception as exc: raise TwinStorageError(f"digital twin storage is corrupt: {exc}") from exc
    def _file_lock(self) -> Path:
        lock = self.path.with_suffix(self.path.suffix + ".lock"); deadline = time.monotonic() + self.lock_timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); return lock
            except FileExistsError:
                if time.monotonic() >= deadline: raise TwinStorageError("timed out waiting for digital twin lock")
                time.sleep(.02)
    def _write(self, rows: Iterable[DigitalTwinEvent]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
                   "events": [x.to_dict() for x in sorted(rows, key=lambda x: (x.occurred_at, x.event_id))]}
        self.path.parent.mkdir(parents=True, exist_ok=True); temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, self.path)
        except OSError as exc: raise TwinStorageError(f"failed atomic digital twin write: {exc}") from exc
        finally:
            try: temp.unlink(missing_ok=True)
            except OSError: pass
    def add(self, event: DigitalTwinEvent) -> DigitalTwinEvent:
        with self._lock:
            lock = self._file_lock()
            try:
                rows = self._read(); existing = next((x for x in rows if x.event_id == event.event_id), None)
                if existing:
                    if existing.to_dict() == event.to_dict(): return existing
                    raise TwinConflictError(f"event {event.event_id} already exists with different content")
                rows.append(event); self._write(rows); return event
            finally:
                try: lock.unlink(missing_ok=True)
                except OSError: pass
    def list(self) -> list[DigitalTwinEvent]:
        with self._lock: return self._read()


class TwinAdapter(Protocol):
    def identities(self) -> Iterable[WellIdentity]: ...
    def events(self) -> Iterable[DigitalTwinEvent]: ...


class WatercutAdapter:
    def __init__(self, repository: Any): self.repository = repository
    def identities(self):
        for x in self.repository.list_metadata():
            yield WellIdentity(x.well, x.field_name, x.cluster, x.site, x.reservoir)
    def events(self):
        identities = list(self.identities())
        for x in self.repository.list_production():
            matches = [w for w in identities if w.normalized_name == normalize_well_name(x.well)]
            identity = matches[0] if len(matches) == 1 else WellIdentity(x.well)
            metrics = {"oil_rate": MetricValue(x.q_oil_m3d, "m3/d", "good", x.timestamp, "watercut.production"),
                       "water_rate": MetricValue(x.q_water_m3d, "m3/d", "good", x.timestamp, "watercut.production"),
                       "water_cut": MetricValue(x.water_cut, "fraction", "questionable" if x.quality_flags else "good", x.timestamp, "watercut.production")}
            if x.pressure is not None: metrics["pressure"] = MetricValue(x.pressure, x.pressure_unit, "good", x.timestamp, "watercut.production")
            yield _event(identity, x.timestamp, EventCategory.PRODUCTION, "production_snapshot", "watercut.production", x.id,
                         "Производственный замер", "Дебиты и обводнённость", metrics=metrics,
                         status=x.status, quality="questionable" if x.quality_flags else "good",
                         metadata={"quality_flags": list(x.quality_flags)})


class EquipmentAdapter:
    def __init__(self, repository: Any): self.repository = repository
    def identities(self):
        for x in self.repository.list_equipment(): yield WellIdentity(x.well)
    def events(self):
        identities = {x.normalized_name: x for x in self.identities()}
        for x in self.repository.list_telemetry():
            identity = identities.get(normalize_well_name(x.well), WellIdentity(x.well)); metrics = {}
            for name, unit in (("intake_pressure", x.pressure_unit), ("discharge_pressure", x.pressure_unit),
                               ("wellhead_pressure", x.pressure_unit), ("motor_temperature_c", "degC"),
                               ("fluid_temperature_c", "degC"), ("bearing_temperature_c", "degC"),
                               ("vibration_mm_s", "mm/s"), ("current_a", "A"), ("rod_load_kn", "kN")):
                value = getattr(x, name)
                if value is not None: metrics[name] = MetricValue(value, unit, "unknown", x.timestamp, "equipment.telemetry")
            yield _event(identity, x.timestamp, EventCategory.EQUIPMENT_TELEMETRY, "telemetry_snapshot", "equipment.telemetry", x.id,
                         "Телеметрия оборудования", f"Snapshot {x.lift_type}", metrics=metrics, quality="unknown")


class TreatmentAdapter:
    def __init__(self, repository: Any): self.repository = repository
    def identities(self):
        for x in self.repository.list(include_archived=True): yield WellIdentity(x.well_name, x.field_name, x.cluster, x.site)
    def events(self):
        for x in self.repository.list(include_archived=True):
            identity = WellIdentity(x.well_name, x.field_name, x.cluster, x.site, canonical_id=x.well_id if x.well_id.startswith("well:") else None)
            metrics = {"cost": MetricValue(x.cost, x.currency, "good", x.event_at, "treatments")}
            if x.rate_before_m3_day is not None: metrics["rate_before"] = MetricValue(x.rate_before_m3_day, "m3/d", "good", x.event_at, "treatments")
            if x.rate_after_m3_day is not None: metrics["rate_after"] = MetricValue(x.rate_after_m3_day, "m3/d", "good", x.event_at, "treatments")
            yield _event(identity, x.event_at, EventCategory.TREATMENT, "treatment", "treatments", x.id,
                         x.treatment_type, x.description, metrics=metrics, status=x.status.value,
                         severity="info" if x.success is not False else "warning", quality="good",
                         metadata={"effect": x.actual_result, "success": x.success, "revision": x.revision})


class PassportAdapter:
    CATEGORY = {"repair": EventCategory.REPAIR, "lab_report": EventCategory.LABORATORY,
                "parameters_analysis": EventCategory.LABORATORY, "complication": EventCategory.COMPLICATION,
                "risk_snapshot": EventCategory.COMPLICATION, "rate_change": EventCategory.PRODUCTION,
                "reagent_effectiveness": EventCategory.TREATMENT, "deposit_photo": EventCategory.COMPLICATION}
    def __init__(self, repository: Any): self.repository = repository
    def identities(self):
        for x in self.repository.list(): yield WellIdentity(x.well_name, canonical_id=x.well_id if x.well_id.startswith("well:") else None)
    def events(self):
        for x in self.repository.list():
            identity = WellIdentity(x.well_name, canonical_id=x.well_id if x.well_id.startswith("well:") else None)
            metrics = {k: MetricValue(v, None, "unknown", x.event_at, "passport") for k, v in x.data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            category = self.CATEGORY[x.event_type.value]
            yield _event(identity, x.event_at, category, x.event_type.value, "passport", x.id,
                         x.title, x.notes or x.title, metrics=metrics, quality="unknown", recorded_at=x.created_at,
                         metadata={"data": x.data, "revision": x.revision})


class ManualAdapter:
    def __init__(self, repository: ManualEventRepository): self.repository = repository
    def identities(self):
        for x in self.repository.list(): yield x.well
    def events(self): return self.repository.list()


@dataclass(frozen=True)
class TwinHealthPolicy:
    version: str = POLICY_VERSION
    stale_after_days: int = 30
    watercut_watch: float = .70
    watercut_critical: float = .90
    equipment_watch: float = .35
    equipment_critical: float = .70


@dataclass(frozen=True)
class TwinSnapshot:
    well: WellIdentity
    as_of: datetime
    state: str
    health_score: float | None
    drivers: tuple[str, ...]
    changed_since_previous: tuple[str, ...]
    indicators: dict[str, MetricValue]
    active_complications: tuple[str, ...]
    latest_treatment: dict[str, Any] | None
    recent_repairs: tuple[dict[str, Any], ...]
    latest_laboratory: dict[str, Any] | None
    economic_losses: dict[str, float]
    missing_sources: tuple[str, ...]
    stale_sources: tuple[str, ...]
    policy_version: str = POLICY_VERSION
    disclaimer: str = ASSOCIATION_DISCLAIMER
    def to_dict(self):
        return {"well": self.well.to_dict(), "as_of": self.as_of.isoformat(), "state": self.state,
                "health_score": self.health_score, "drivers": list(self.drivers),
                "changed_since_previous": list(self.changed_since_previous),
                "indicators": {k: v.to_dict() for k, v in self.indicators.items()},
                "active_complications": list(self.active_complications), "latest_treatment": self.latest_treatment,
                "recent_repairs": list(self.recent_repairs), "latest_laboratory": self.latest_laboratory,
                "economic_losses": self.economic_losses, "missing_sources": list(self.missing_sources),
                "stale_sources": list(self.stale_sources), "policy_version": self.policy_version,
                "disclaimer": self.disclaimer}


@dataclass(frozen=True)
class ChangeExplanation:
    event_id: str
    occurred_at: datetime
    title: str
    statement: str
    before: dict[str, Any]
    after: dict[str, Any]
    evidence: tuple[str, ...]
    confidence: str
    alternative_explanations: tuple[str, ...]
    disclaimer: str = ASSOCIATION_DISCLAIMER
    def to_dict(self):
        value = asdict(self); value["occurred_at"] = self.occurred_at.isoformat(); return value


class DigitalTwinService:
    def __init__(self, adapters: Iterable[TwinAdapter], policy: TwinHealthPolicy | None = None):
        self.adapters = tuple(adapters); self.policy = policy or TwinHealthPolicy()
    def _all_identities(self) -> list[WellIdentity]:
        by_id: dict[str, WellIdentity] = {}
        for adapter in self.adapters:
            try:
                for item in adapter.identities():
                    by_id[item.canonical_id] = item
            except Exception:
                # One unavailable/corrupt optional source must not make the
                # aggregate twin unusable when other repositories are healthy.
                continue
        # Context-free source aliases (equipment/passport) may safely attach only
        # when exactly one contextual identity exists. With two fields they stay
        # visible and resolution remains explicitly ambiguous.
        grouped: dict[str, list[WellIdentity]] = {}
        for item in by_id.values(): grouped.setdefault(item.normalized_name, []).append(item)
        result: list[WellIdentity] = []
        for items in grouped.values():
            contextual = [x for x in items if any((x.field, x.cluster, x.site, x.reservoir))]
            if len(contextual) == 1:
                result.extend(contextual)
            else:
                result.extend(items)
        return sorted({x.canonical_id: x for x in result}.values(),
                      key=lambda x: (x.normalized_name, x.field or "", x.canonical_id))
    def list_wells(self) -> list[dict[str, Any]]:
        groups: dict[str, int] = {}
        for x in self._all_identities(): groups[x.normalized_name] = groups.get(x.normalized_name, 0) + 1
        return [{**x.to_dict(), "ambiguous_name": groups[x.normalized_name] > 1} for x in self._all_identities()]
    def resolve(self, query: str, *, field: str | None = None, cluster: str | None = None,
                site: str | None = None, reservoir: str | None = None) -> WellIdentity:
        q = query.strip(); identities = self._all_identities()
        exact_id = [x for x in identities if x.canonical_id == q]
        if exact_id: return exact_id[0]
        normalized = normalize_well_name(q); candidates = [x for x in identities if x.normalized_name == normalized]
        requested = WellIdentity(q, field, cluster, site, reservoir)
        candidates = [x for x in candidates if x.matches_context(requested)]
        if not candidates: raise TwinNotFoundError(f"well {query} not found")
        if len(candidates) > 1: raise TwinAmbiguousError(f"well {query} is ambiguous; specify canonical_id or field/site/reservoir")
        return candidates[0]
    def _events_for(self, identity: WellIdentity) -> list[DigitalTwinEvent]:
        unique: dict[str, DigitalTwinEvent] = {}
        for adapter in self.adapters:
            for event in adapter.events():
                if event.well.canonical_id == identity.canonical_id or (event.well.normalized_name == identity.normalized_name and event.well.matches_context(identity)):
                    current = unique.get(event.event_id)
                    if current and current.to_dict() != event.to_dict(): raise TwinConflictError(f"event ID collision: {event.event_id}")
                    unique[event.event_id] = event
        return sorted(unique.values(), key=lambda x: (x.occurred_at, x.recorded_at, x.event_id), reverse=True)
    def timeline(self, query: str, *, date_from: datetime | None = None, date_to: datetime | None = None,
                 categories: Iterable[str | EventCategory] | None = None, limit: int = 100,
                 cursor: str | None = None, **context: Any) -> dict[str, Any]:
        if not 1 <= limit <= 1000: raise ValueError("limit must be 1..1000")
        identity = self.resolve(query, **context); start = aware(date_from) if date_from else None; end = aware(date_to) if date_to else None
        if start and end and start > end: raise ValueError("from cannot be after to")
        allowed = {EventCategory(x) for x in categories} if categories else None
        rows = [x for x in self._events_for(identity) if (not start or x.occurred_at >= start) and (not end or x.occurred_at <= end) and (not allowed or x.category in allowed)]
        offset = 0
        if cursor:
            try: offset = int(cursor)
            except ValueError as exc: raise ValueError("cursor is invalid") from exc
            if offset < 0: raise ValueError("cursor is invalid")
        page = rows[offset:offset + limit]; next_cursor = str(offset + limit) if offset + limit < len(rows) else None
        return {"well": identity.to_dict(), "items": [x.to_dict() for x in page], "next_cursor": next_cursor,
                "total": len(rows), "schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION}
    def snapshot(self, query: str, *, as_of: datetime | None = None, **context: Any) -> TwinSnapshot:
        when = aware(as_of or datetime.now(timezone.utc), "as_of"); identity = self.resolve(query, **context)
        rows = [x for x in self._events_for(identity) if x.occurred_at <= when]
        indicators: dict[str, MetricValue] = {}; source_latest: dict[str, datetime] = {}
        for event in rows:
            source_latest[event.source] = max(source_latest.get(event.source, event.occurred_at), event.occurred_at)
            for key, metric in event.metrics.items():
                if key not in indicators:
                    age = when - event.occurred_at; freshness = "fresh" if age <= timedelta(days=self.policy.stale_after_days) else "stale"
                    indicators[key] = MetricValue(metric.value, metric.unit, metric.quality, event.occurred_at, event.source, freshness)
        drivers: list[str] = []; scores: list[float] = []
        wc = indicators.get("water_cut")
        if wc and isinstance(wc.value, (float, int)):
            score = min(max(float(wc.value), 0), 1); scores.append(score)
            if score >= self.policy.watercut_critical: drivers.append("Критическая обводнённость")
            elif score >= self.policy.watercut_watch: drivers.append("Высокая обводнённость")
        complication_rows = [x for x in rows if x.category in {EventCategory.COMPLICATION, EventCategory.EQUIPMENT_FAILURE}]
        if complication_rows: scores.append(max(.7 if x.severity in {"critical", "high"} else .4 for x in complication_rows[:5])); drivers.append(complication_rows[0].title)
        state = "insufficient_data" if not indicators and not complication_rows else "critical" if any(x >= .7 for x in scores) else "watch" if any(x >= .35 for x in scores) else "normal"
        missing = tuple(x for x in ("production", "equipment.telemetry", "laboratory") if not any(e.category.value.startswith(x.split(".")[0]) or e.source.startswith(x) for e in rows))
        stale = tuple(sorted(source for source, stamp in source_latest.items() if when - stamp > timedelta(days=self.policy.stale_after_days)))
        latest_treatment = next((x.to_dict() for x in rows if x.category is EventCategory.TREATMENT), None)
        latest_lab = next((x.to_dict() for x in rows if x.category is EventCategory.LABORATORY), None)
        repairs = tuple(x.to_dict() for x in rows if x.category is EventCategory.REPAIR)[:10]
        active = tuple(dict.fromkeys(x.title for x in complication_rows if x.status not in {"closed", "resolved"}))
        economic: dict[str, float] = {}
        for event in rows:
            if event.category is EventCategory.ECONOMIC_LOSS:
                for metric in event.metrics.values():
                    if isinstance(metric.value, (int, float)) and metric.unit: economic[metric.unit] = economic.get(metric.unit, 0) + float(metric.value)
        previous = self._snapshot_state(rows[1:], when) if len(rows) > 1 else None
        changed = () if previous == state or previous is None else (f"Состояние изменилось: {previous} → {state}",)
        return TwinSnapshot(identity, when, state, max(scores) if scores else None, tuple(dict.fromkeys(drivers)), changed,
                            indicators, active, latest_treatment, repairs, latest_lab, economic, missing, stale)
    def _snapshot_state(self, rows: list[DigitalTwinEvent], when: datetime) -> str:
        if not rows: return "insufficient_data"
        wc = next((m.value for e in rows for k, m in e.metrics.items() if k == "water_cut"), None)
        if isinstance(wc, (int, float)) and wc >= self.policy.watercut_critical: return "critical"
        if isinstance(wc, (int, float)) and wc >= self.policy.watercut_watch: return "watch"
        if any(e.severity in {"critical", "high"} for e in rows if e.category in {EventCategory.COMPLICATION, EventCategory.EQUIPMENT_FAILURE}): return "critical"
        return "normal"
    def changes(self, query: str, *, as_of: datetime | None = None, limit: int = 20, **context: Any) -> list[ChangeExplanation]:
        when = aware(as_of or datetime.now(timezone.utc)); identity = self.resolve(query, **context)
        rows = [x for x in reversed(self._events_for(identity)) if x.occurred_at <= when]
        significant = {EventCategory.REPAIR, EventCategory.TREATMENT, EventCategory.EQUIPMENT_FAILURE, EventCategory.LABORATORY, EventCategory.COMPLICATION}
        results = []
        for event in (x for x in rows if x.category in significant):
            before_event = next((x for x in reversed(rows) if x.occurred_at < event.occurred_at and x.metrics), None)
            after_event = next((x for x in rows if x.occurred_at > event.occurred_at and x.metrics), None)
            before = {k: v.to_dict() for k, v in (before_event.metrics.items() if before_event else [])}
            after = {k: v.to_dict() for k, v in (after_event.metrics.items() if after_event else [])}
            evidence = [f"Событие зарегистрировано {event.occurred_at.isoformat()}"]
            if after_event: evidence.append(f"Следующий замер {after_event.occurred_at.isoformat()}")
            confidence = "medium" if before and after else "low"
            results.append(ChangeExplanation(event.event_id, event.occurred_at, event.title,
                f"Изменение последовало после «{event.title}»; возможная связь требует подтверждения.",
                before, after, tuple(evidence), confidence,
                ("изменение режима эксплуатации", "другое одновременное мероприятие", "ошибка или несопоставимость измерений")))
        return list(reversed(results[-limit:]))


def manual_event(*, well: WellIdentity, occurred_at: datetime, category: EventCategory | str,
                 event_type: str, title: str, summary: str, source_record_id: str | None = None,
                 severity: str = "info", status: str | None = None,
                 metrics: dict[str, MetricValue | dict[str, Any]] | None = None,
                 metadata: dict[str, Any] | None = None, recorded_at: datetime | None = None) -> DigitalTwinEvent:
    source_id = source_record_id or str(uuid4()); now = aware(recorded_at or datetime.now(timezone.utc))
    return _event(well, aware(occurred_at), EventCategory(category), event_type, "manual", source_id,
                  title, summary, metrics={k: v if isinstance(v, MetricValue) else MetricValue(**v) for k, v in (metrics or {}).items()},
                  severity=severity, status=status, quality="manual", metadata=metadata, recorded_at=now)


def manual_csv_template() -> str:
    return "source_record_id,well,field,cluster,site,reservoir,occurred_at,category,event_type,title,summary,severity,status,metrics_json\n"


def manual_events_from_csv(text: str) -> tuple[list[DigitalTwinEvent], list[str]]:
    rows, errors = [], []
    for index, value in enumerate(csv.DictReader(io.StringIO(text)), 2):
        try:
            metrics_raw = json.loads(value.get("metrics_json") or "{}")
            rows.append(manual_event(well=WellIdentity(value["well"], value.get("field"), value.get("cluster"), value.get("site"), value.get("reservoir")),
                occurred_at=aware(value["occurred_at"]), category=value["category"], event_type=value["event_type"],
                title=value["title"], summary=value["summary"], source_record_id=value.get("source_record_id") or None,
                severity=value.get("severity") or "info", status=value.get("status") or None, metrics=metrics_raw))
        except Exception as exc: errors.append(f"row {index}: {exc}")
    return rows, errors


def build_default_service(*, watercut: Any = None, equipment: Any = None, treatments: Any = None,
                          passports: Any = None, manual: ManualEventRepository | None = None) -> DigitalTwinService:
    adapters: list[TwinAdapter] = []
    if watercut is not None: adapters.append(WatercutAdapter(watercut))
    if equipment is not None: adapters.append(EquipmentAdapter(equipment))
    if treatments is not None: adapters.append(TreatmentAdapter(treatments))
    if passports is not None: adapters.append(PassportAdapter(passports))
    if manual is not None: adapters.append(ManualAdapter(manual))
    return DigitalTwinService(adapters)
