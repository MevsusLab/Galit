"""Journal of interventions and observed effects.

The module deliberately keeps plans separate from assessed outcomes. JSON storage is
versioned and replaced atomically; analytics only uses explicitly assessed records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Iterable
from uuid import uuid4

SCHEMA_VERSION = 1
DEFAULT_MIN_SAMPLE_SIZE = 5
VALID_DOSAGE_UNITS = {"mg/l", "g/l", "kg/m3", "l/m3", "l", "kg", "g", "ml"}
VALID_CURRENCIES = {"BYN", "RUB", "USD", "EUR"}


class TreatmentStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ASSESSED = "assessed"


ALLOWED_TRANSITIONS = {
    TreatmentStatus.PLANNED: {TreatmentStatus.IN_PROGRESS},
    TreatmentStatus.IN_PROGRESS: {TreatmentStatus.COMPLETED},
    TreatmentStatus.COMPLETED: {TreatmentStatus.ASSESSED},
    TreatmentStatus.ASSESSED: set(),
}


class TreatmentStorageError(RuntimeError):
    """Storage is unavailable, corrupt, or could not be updated safely."""


class TreatmentNotFoundError(LookupError):
    pass


class TreatmentConflictError(RuntimeError):
    """A stale revision or immutable assessed record was submitted."""


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return value.astimezone(timezone.utc)


def _text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _finite_non_negative(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class TreatmentRecord:
    id: str
    well_id: str
    well_name: str
    event_at: datetime
    complication_type: str
    description: str
    reagent_name: str
    reagent_id: str | None
    dosage: float
    dosage_unit: str
    cost: float
    currency: str
    treatment_type: str
    status: TreatmentStatus = TreatmentStatus.PLANNED
    baseline_risk: float | None = None
    baseline_state: str | None = None
    expected_result: str | None = None
    actual_result: str | None = None
    result_metrics: dict[str, float] = field(default_factory=dict)
    success: bool | None = None
    effect_duration_days: float | None = None
    recurrence: bool | None = None
    recurrence_date: datetime | None = None
    comment: str | None = None
    source: str = "manual"
    well_group: str | None = None
    revision: int = 1
    archived: bool = False
    archived_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", TreatmentStatus(self.status))
        for name in ("id", "well_id", "well_name", "complication_type", "description",
                     "reagent_name", "treatment_type", "source"):
            object.__setattr__(self, name, _text(str(getattr(self, name)), name))
        for name in ("reagent_id", "baseline_state", "expected_result", "actual_result",
                     "comment", "well_group"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(str(value), name))
        object.__setattr__(self, "event_at", _aware(self.event_at, "event_at"))
        object.__setattr__(self, "created_at", _aware(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _aware(self.updated_at, "updated_at"))
        if self.archived_at is not None:
            object.__setattr__(self, "archived_at", _aware(self.archived_at, "archived_at"))
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if self.archived != (self.archived_at is not None):
            raise ValueError("archived and archived_at must be set together")
        object.__setattr__(self, "dosage", _finite_non_negative(self.dosage, "dosage"))
        unit = self.dosage_unit.strip().lower().replace("³", "3")
        if unit not in VALID_DOSAGE_UNITS:
            raise ValueError("unsupported dosage_unit; use mg/l, g/l, kg/m3, l/m3, l, kg, g, or ml")
        object.__setattr__(self, "dosage_unit", unit)
        object.__setattr__(self, "cost", _finite_non_negative(self.cost, "cost"))
        currency = self.currency.strip().upper()
        if currency not in VALID_CURRENCIES:
            raise ValueError("currency must be one of BYN, RUB, USD, EUR")
        object.__setattr__(self, "currency", currency)
        if self.baseline_risk is not None:
            risk = float(self.baseline_risk)
            if not math.isfinite(risk) or not 0 <= risk <= 1:
                raise ValueError("baseline_risk must be finite and within [0, 1]")
            object.__setattr__(self, "baseline_risk", risk)
        metrics = {str(k).strip(): _finite_non_negative(v, f"result_metrics.{k}")
                   for k, v in self.result_metrics.items()}
        if any(not key for key in metrics):
            raise ValueError("result metric names must be non-empty")
        object.__setattr__(self, "result_metrics", metrics)
        if self.effect_duration_days is not None:
            object.__setattr__(self, "effect_duration_days", _finite_non_negative(
                self.effect_duration_days, "effect_duration_days"))
        if self.recurrence_date is not None:
            object.__setattr__(self, "recurrence_date", _aware(self.recurrence_date, "recurrence_date"))
            if self.recurrence is not True:
                raise ValueError("recurrence_date requires recurrence=true")
            if self.recurrence_date < self.event_at:
                raise ValueError("recurrence_date cannot precede event_at")
        actual_supplied = bool(self.actual_result or self.result_metrics or self.success is not None or
                               self.effect_duration_days is not None or self.recurrence is not None or
                               self.recurrence_date is not None)
        if actual_supplied and self.status not in {TreatmentStatus.COMPLETED, TreatmentStatus.ASSESSED}:
            raise ValueError("actual result fields require completed or assessed status")
        if self.status is TreatmentStatus.ASSESSED:
            missing = [name for name, value in (
                ("actual_result", self.actual_result), ("success", self.success),
                ("effect_duration_days", self.effect_duration_days), ("recurrence", self.recurrence),
            ) if value is None]
            if missing:
                raise ValueError("assessed record requires " + ", ".join(missing))
            if not self.result_metrics:
                raise ValueError("assessed record requires at least one measurable result metric")
        if self.recurrence is True and self.recurrence_date is None:
            raise ValueError("recurrence=true requires recurrence_date")
        if self.recurrence is False and self.recurrence_date is not None:
            raise ValueError("recurrence=false cannot have recurrence_date")

    def transition(self, status: TreatmentStatus, *, now: datetime | None = None,
                   **changes: Any) -> "TreatmentRecord":
        target = TreatmentStatus(status)
        if self.archived:
            raise TreatmentConflictError("archived treatment cannot be changed")
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid treatment transition: {self.status.value} -> {target.value}")
        return replace(self, status=target, updated_at=now or datetime.now(timezone.utc), **changes)

    def edit(self, *, now: datetime | None = None, **changes: Any) -> "TreatmentRecord":
        if self.archived:
            raise TreatmentConflictError("archived treatment cannot be changed")
        if self.status is TreatmentStatus.ASSESSED:
            raise TreatmentConflictError("assessed treatment is immutable")
        forbidden = {"id", "status", "revision", "archived", "archived_at", "created_at"}
        if forbidden.intersection(changes):
            raise ValueError("identity, lifecycle, revision and archive fields cannot be edited")
        return replace(self, updated_at=now or datetime.now(timezone.utc), **changes)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        for name in ("event_at", "recurrence_date", "archived_at", "created_at", "updated_at"):
            if data[name] is not None:
                data[name] = data[name].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreatmentRecord":
        values = dict(data)
        for name in ("event_at", "recurrence_date", "archived_at", "created_at", "updated_at"):
            if values.get(name) is not None:
                values[name] = datetime.fromisoformat(values[name])
        return cls(**values)


class TreatmentRepository:
    """Small local JSON repository with process/thread locking and atomic replacement."""
    _locks: dict[str, threading.RLock] = {}
    _guard = threading.Lock()

    def __init__(self, path: str | Path = "data/treatments.json", lock_timeout: float = 5.0):
        self.path = Path(path)
        self.lock_timeout = lock_timeout
        key = str(self.path.resolve())
        with self._guard:
            self._lock = self._locks.setdefault(key, threading.RLock())

    def _read_unlocked(self) -> list[TreatmentRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("records"), list):
                raise ValueError("unsupported schema or missing records array")
            records = [TreatmentRecord.from_dict(item) for item in payload["records"]]
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            raise TreatmentStorageError(f"treatment storage is corrupt or unreadable: {exc}") from exc
        ids = [item.id for item in records]
        if len(ids) != len(set(ids)):
            raise TreatmentStorageError("treatment storage contains duplicate IDs")
        return records

    def _file_lock(self) -> Path:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        deadline = time.monotonic() + self.lock_timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lock_path
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TreatmentStorageError("timed out waiting for treatment storage lock")
                time.sleep(0.02)

    def _write_unlocked(self, records: Iterable[TreatmentRecord]) -> None:
        ordered = sorted(records, key=lambda item: (item.created_at, item.id))
        payload = {"schema_version": SCHEMA_VERSION, "records": [item.to_dict() for item in ordered]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        except OSError as exc:
            raise TreatmentStorageError(f"failed to write treatment storage atomically: {exc}") from exc
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def _mutate(self, action):
        with self._lock:
            lock_path = self._file_lock()
            try:
                records = self._read_unlocked()
                result = action(records)
                self._write_unlocked(records)
                return result
            finally:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def create(self, record: TreatmentRecord) -> TreatmentRecord:
        def action(records):
            if any(item.id == record.id for item in records):
                raise ValueError(f"treatment {record.id} already exists")
            records.append(record)
            return record
        return self._mutate(action)

    def get(self, record_id: str) -> TreatmentRecord:
        with self._lock:
            for item in self._read_unlocked():
                if item.id == record_id:
                    return item
        raise TreatmentNotFoundError(f"treatment {record_id} not found")

    def list(self, *, well: str | None = None, status: TreatmentStatus | None = None,
             complication_type: str | None = None, reagent: str | None = None,
             currency: str | None = None, well_group: str | None = None,
             include_archived: bool = False, offset: int = 0,
             limit: int | None = None) -> list[TreatmentRecord]:
        if offset < 0 or (limit is not None and limit < 1):
            raise ValueError("offset must be non-negative and limit must be positive")
        with self._lock:
            records = self._read_unlocked()
        def same(value: str, query: str | None) -> bool:
            return query is None or value.casefold() == query.strip().casefold()
        filtered = [item for item in records
                    if (include_archived or not item.archived)
                    and (well is None or same(item.well_id, well) or same(item.well_name, well))
                    and (status is None or item.status is TreatmentStatus(status))
                    and same(item.complication_type, complication_type)
                    and (reagent is None or same(item.reagent_name, reagent) or
                         (item.reagent_id is not None and same(item.reagent_id, reagent)))
                    and (currency is None or item.currency == currency.strip().upper())
                    and (well_group is None or (item.well_group is not None and same(item.well_group, well_group)))]
        filtered.sort(key=lambda item: (item.event_at, item.id), reverse=True)
        page = filtered[offset:]
        return page[:limit] if limit is not None else page

    def update(self, record: TreatmentRecord, *, expected_revision: int | None = None) -> TreatmentRecord:
        def action(records):
            for index, item in enumerate(records):
                if item.id == record.id:
                    expected = record.revision if expected_revision is None else expected_revision
                    if expected != item.revision:
                        raise TreatmentConflictError(
                            f"revision conflict: expected {expected}, current {item.revision}")
                    if record.created_at != item.created_at:
                        raise ValueError("created_at is immutable")
                    if item.status is TreatmentStatus.ASSESSED or item.archived:
                        raise TreatmentConflictError("assessed or archived treatment is immutable")
                    saved = replace(record, revision=item.revision + 1)
                    records[index] = saved
                    return saved
            raise TreatmentNotFoundError(f"treatment {record.id} not found")
        return self._mutate(action)

    def archive(self, record_id: str, *, expected_revision: int,
                now: datetime | None = None) -> TreatmentRecord:
        timestamp = now or datetime.now(timezone.utc)
        def action(records):
            for index, item in enumerate(records):
                if item.id == record_id:
                    if expected_revision != item.revision:
                        raise TreatmentConflictError(
                            f"revision conflict: expected {expected_revision}, current {item.revision}")
                    if item.archived:
                        raise TreatmentConflictError("treatment is already archived")
                    saved = replace(item, archived=True, archived_at=timestamp,
                                    updated_at=timestamp, revision=item.revision + 1)
                    records[index] = saved
                    return saved
            raise TreatmentNotFoundError(f"treatment {record_id} not found")
        return self._mutate(action)


def new_treatment(*, now: datetime | None = None, **values: Any) -> TreatmentRecord:
    timestamp = now or datetime.now(timezone.utc)
    return TreatmentRecord(id=str(uuid4()), created_at=timestamp, updated_at=timestamp, **values)


def _group_key(item: TreatmentRecord, group_by: str) -> str:
    if group_by == "well":
        return item.well_name
    if group_by == "complication_type":
        return item.complication_type
    if group_by == "reagent":
        return item.reagent_name
    if group_by == "well_group":
        return item.well_group or "unknown"
    raise ValueError("group_by must be well, complication_type, reagent, or well_group")


def treatment_summary(records: Iterable[TreatmentRecord], group_by: str = "reagent") -> dict[str, Any]:
    groups: dict[str, list[TreatmentRecord]] = {}
    all_records = list(records)
    for item in all_records:
        groups.setdefault(_group_key(item, group_by), []).append(item)
    rows = []
    for key in sorted(groups):
        items = groups[key]
        assessed = [item for item in items if item.status is TreatmentStatus.ASSESSED]
        successes = [item for item in assessed if item.success is True]
        durations = [float(item.effect_duration_days) for item in assessed
                     if item.effect_duration_days is not None]
        recurrences = [item.recurrence for item in assessed if item.recurrence is not None]
        currencies = sorted({item.currency for item in assessed})
        costs: dict[str, dict[str, float | None]] = {}
        for currency in currencies:
            same_currency = [item for item in assessed if item.currency == currency]
            total = sum(item.cost for item in same_currency)
            successful = sum(item.success is True for item in same_currency)
            days = sum(float(item.effect_duration_days or 0) for item in same_currency)
            costs[currency] = {
                "total": total,
                "per_success": total / successful if successful else None,
                "per_effect_day": total / days if days > 0 else None,
            }
        assessed_n = len(assessed)
        quality = "high" if assessed_n >= 20 else "medium" if assessed_n >= DEFAULT_MIN_SAMPLE_SIZE else "low"
        rows.append({
            "group": key, "treatments": len(items), "assessed_observations": assessed_n,
            "success_rate": len(successes) / assessed_n if assessed_n else None,
            "effect_days_mean": statistics.fmean(durations) if durations else None,
            "effect_days_median": statistics.median(durations) if durations else None,
            "recurrence_rate": sum(value is True for value in recurrences) / len(recurrences) if recurrences else None,
            "costs_by_currency": costs,
            "confidence": quality,
            "data_quality": {"assessed_fraction": assessed_n / len(items) if items else 0,
                             "missing_metric_records": sum(not item.result_metrics for item in assessed)},
        })
    return {"status": "available" if rows else "insufficient_data", "group_by": group_by,
            "groups": rows, "observational_warning": "Наблюдательная связь не доказывает причинность."}


def compare_reagents(records: Iterable[TreatmentRecord], reagent_a: str, reagent_b: str, *,
                     metric: str = "success_rate", min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
                     complication_type: str | None = None, well_group: str | None = None) -> dict[str, Any]:
    if min_sample_size < 2:
        raise ValueError("min_sample_size must be at least 2")
    if metric not in {"success_rate", "mean_effect_days"}:
        raise ValueError("metric must be success_rate or mean_effect_days")
    if not complication_type or not well_group:
        raise ValueError("complication_type and well_group are required for a comparable cohort")
    comparable = [item for item in records if item.status is TreatmentStatus.ASSESSED
                  and (complication_type is None or item.complication_type.casefold() == complication_type.casefold())
                  and (well_group is None or (item.well_group or "").casefold() == well_group.casefold())]
    def sample(name: str) -> list[TreatmentRecord]:
        return [item for item in comparable if item.reagent_name.casefold() == name.casefold()]
    a, b = sample(reagent_a), sample(reagent_b)
    def value(items: list[TreatmentRecord]) -> float | None:
        if not items:
            return None
        if metric == "success_rate":
            return sum(item.success is True for item in items) / len(items)
        durations = [float(item.effect_duration_days) for item in items if item.effect_duration_days is not None]
        return statistics.fmean(durations) if durations else None
    av, bv = value(a), value(b)
    reason = None
    if len(a) < min_sample_size or len(b) < min_sample_size:
        reason = "minimum comparable sample not reached"
    elif av is None or bv is None:
        reason = "metric is unavailable"
    elif av == 0:
        reason = "zero baseline; relative uplift is undefined"
    uplift = None if reason else (bv - av) / av
    return {
        "status": "insufficient_data" if reason else "available",
        "metric": metric, "formula": "(metric_B - metric_A) / metric_A",
        "reagent_a": {"name": reagent_a, "n": len(a), "value": av},
        "reagent_b": {"name": reagent_b, "n": len(b), "value": bv},
        "relative_uplift": uplift, "min_sample_size": min_sample_size,
        "comparability": {"complication_type": complication_type, "well_group": well_group,
                           "rule": "same complication_type and, when supplied, same explicit well_group"},
        "confidence": "medium" if not reason and min(len(a), len(b)) < 20 else "high" if not reason else "low",
        "reason": reason,
        "warning": "Наблюдательная связь не доказывает причинность; скрытые различия групп не контролируются.",
    }
