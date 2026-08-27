"""Versioned digital well passport with safe local attachments and treatment aggregation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable
from uuid import uuid4

PASSPORT_SCHEMA_VERSION = 1
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024
ALLOWED_ATTACHMENT_MIMES = {
    "image/jpeg": {".jpg", ".jpeg"}, "image/png": {".png"},
    "application/pdf": {".pdf"}, "text/plain": {".txt"},
    "text/csv": {".csv"},
}
_SAFE_NAME = re.compile(r"[^A-Za-zА-Яа-яЁё0-9._ -]+")


class PassportEventType(str, Enum):
    PARAMETERS_ANALYSIS = "parameters_analysis"
    RISK_SNAPSHOT = "risk_snapshot"
    REPAIR = "repair"
    RATE_CHANGE = "rate_change"
    COMPLICATION = "complication"
    DEPOSIT_PHOTO = "deposit_photo"
    LAB_REPORT = "lab_report"
    REAGENT_EFFECTIVENESS = "reagent_effectiveness"


class PassportStorageError(RuntimeError):
    """Passport data cannot be read or safely persisted."""


class PassportNotFoundError(LookupError):
    pass


class PassportConflictError(RuntimeError):
    pass


def _text(value: object, name: str, maximum: int = 2000) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(result) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return result


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return value.astimezone(timezone.utc)


def _number(value: object, name: str, *, nonnegative: bool = False,
            upper: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0) or (upper is not None and result > upper):
        suffix = " finite and non-negative" if nonnegative else " finite"
        if upper is not None:
            suffix += f" and <= {upper}"
        raise ValueError(f"{name} must be{suffix}")
    return result


def safe_attachment_name(value: str) -> str:
    raw = Path(str(value).replace("\\", "/")).name.strip()
    cleaned = _SAFE_NAME.sub("_", raw).strip(" .")
    if not cleaned or cleaned in {".", ".."} or len(cleaned) > 160:
        raise ValueError("attachment filename is empty, unsafe, or too long")
    return cleaned


@dataclass(frozen=True)
class AttachmentMetadata:
    id: str
    filename: str
    mime_type: str
    size: int
    sha256: str
    relative_path: str
    uploaded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "attachment.id", 100))
        object.__setattr__(self, "filename", safe_attachment_name(self.filename))
        if self.mime_type not in ALLOWED_ATTACHMENT_MIMES:
            raise ValueError("unsupported attachment MIME type")
        if Path(self.filename).suffix.lower() not in ALLOWED_ATTACHMENT_MIMES[self.mime_type]:
            raise ValueError("attachment extension does not match MIME type")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or not 0 < self.size <= MAX_ATTACHMENT_SIZE:
            raise ValueError(f"attachment size must be 1..{MAX_ATTACHMENT_SIZE} bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("attachment sha256 is invalid")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("attachment path must be repository-relative")
        object.__setattr__(self, "uploaded_at", _aware(self.uploaded_at, "uploaded_at"))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self); result["uploaded_at"] = self.uploaded_at.isoformat(); return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttachmentMetadata":
        data = dict(value); data["uploaded_at"] = datetime.fromisoformat(data["uploaded_at"]); return cls(**data)


@dataclass(frozen=True)
class PassportEvent:
    id: str
    well_id: str
    well_name: str
    event_type: PassportEventType
    event_at: datetime
    title: str
    data: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    source: str = "manual"
    attachment: AttachmentMetadata | None = None
    revision: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", PassportEventType(self.event_type))
        for name, maximum in (("id", 100), ("well_id", 200), ("well_name", 200),
                              ("title", 500), ("source", 100)):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum))
        if self.notes is not None:
            object.__setattr__(self, "notes", _text(self.notes, "notes", 10000))
        for name in ("event_at", "created_at", "updated_at"):
            object.__setattr__(self, name, _aware(getattr(self, name), name))
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if not isinstance(self.data, dict):
            raise ValueError("data must be an object")
        clean = _validate_event_data(self.event_type, self.data)
        object.__setattr__(self, "data", clean)
        needs_attachment = self.event_type in {PassportEventType.DEPOSIT_PHOTO, PassportEventType.LAB_REPORT}
        if needs_attachment != (self.attachment is not None):
            raise ValueError(f"{self.event_type.value} requires exactly one attachment")

    def edit(self, *, now: datetime | None = None, **changes: Any) -> "PassportEvent":
        if {"id", "created_at", "revision"}.intersection(changes):
            raise ValueError("identity, created_at and revision cannot be edited")
        return replace(self, updated_at=now or datetime.now(timezone.utc), **changes)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        for key in ("event_at", "created_at", "updated_at"):
            value[key] = getattr(self, key).isoformat()
        value["attachment"] = self.attachment.to_dict() if self.attachment else None
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PassportEvent":
        data = dict(value)
        for key in ("event_at", "created_at", "updated_at"):
            data[key] = datetime.fromisoformat(data[key])
        if data.get("attachment"):
            data["attachment"] = AttachmentMetadata.from_dict(data["attachment"])
        return cls(**data)


def _validate_event_data(kind: PassportEventType, data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    numeric: dict[PassportEventType, dict[str, tuple[bool, float | None]]] = {
        PassportEventType.RISK_SNAPSHOT: {"integrated_risk": (True, 1), "wax_risk": (True, 1), "halite_risk": (True, 1), "calcite_risk": (True, 1), "corrosion_risk": (True, 1)},
        PassportEventType.RATE_CHANGE: {"oil_rate_m3d": (True, None), "water_rate_m3d": (True, None), "gas_rate_m3d": (True, None), "delta_oil_m3d": (False, None)},
        PassportEventType.REAGENT_EFFECTIVENESS: {"efficiency": (True, 1), "dosage": (True, None), "effect_duration_days": (True, None)},
    }
    for key, (nonnegative, upper) in numeric.get(kind, {}).items():
        if key in result and result[key] is not None:
            result[key] = _number(result[key], f"data.{key}", nonnegative=nonnegative, upper=upper)
    required_any = {
        PassportEventType.PARAMETERS_ANALYSIS: {"parameters", "analysis"},
        PassportEventType.RISK_SNAPSHOT: {"integrated_risk", "wax_risk", "halite_risk", "calcite_risk", "corrosion_risk"},
        PassportEventType.RATE_CHANGE: {"oil_rate_m3d", "water_rate_m3d", "gas_rate_m3d", "delta_oil_m3d"},
        PassportEventType.REPAIR: {"description", "work_type"},
        PassportEventType.COMPLICATION: {"description", "complication_type"},
        PassportEventType.REAGENT_EFFECTIVENESS: {"efficiency", "result", "reagent_name"},
    }
    choices = required_any.get(kind)
    if choices and not choices.intersection(result):
        raise ValueError(f"{kind.value} data requires at least one of: {', '.join(sorted(choices))}")
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data must contain JSON-compatible finite values: {exc}") from exc
    return result


class PassportRepository:
    """Thread/process-safe JSON event store with atomic replacement and safe files."""
    _locks: dict[str, threading.RLock] = {}
    _guard = threading.Lock()

    def __init__(self, path: str | Path = "data/well_passports.json", *,
                 attachment_root: str | Path | None = None, lock_timeout: float = 5):
        self.path = Path(path)
        self.attachment_root = Path(attachment_root) if attachment_root else self.path.parent / "passport_attachments"
        self.lock_timeout = lock_timeout
        with self._guard:
            self._lock = self._locks.setdefault(str(self.path.resolve()), threading.RLock())

    def _read(self) -> list[PassportEvent]:
        if not self.path.exists(): return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("schema_version") != PASSPORT_SCHEMA_VERSION or not isinstance(value.get("events"), list):
                raise ValueError("unsupported schema or missing events array")
            events = [PassportEvent.from_dict(row) for row in value["events"]]
            if len({row.id for row in events}) != len(events): raise ValueError("duplicate event IDs")
            return events
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise PassportStorageError(f"passport storage is corrupt or unreadable: {exc}") from exc

    def _file_lock(self) -> Path:
        lock = self.path.with_suffix(self.path.suffix + ".lock"); deadline = time.monotonic() + self.lock_timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); return lock
            except FileExistsError:
                if time.monotonic() >= deadline: raise PassportStorageError("timed out waiting for passport storage lock")
                time.sleep(.02)

    def _write(self, events: Iterable[PassportEvent]) -> None:
        payload = {"schema_version": PASSPORT_SCHEMA_VERSION, "events": [row.to_dict() for row in sorted(events, key=lambda x: (x.created_at, x.id))]}
        self.path.parent.mkdir(parents=True, exist_ok=True); temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, self.path)
        except OSError as exc: raise PassportStorageError(f"failed to write passport atomically: {exc}") from exc
        finally:
            try: temp.unlink(missing_ok=True)
            except OSError: pass

    def _mutate(self, action):
        with self._lock:
            lock = self._file_lock()
            try:
                rows = self._read(); result = action(rows); self._write(rows); return result
            finally:
                try: lock.unlink(missing_ok=True)
                except OSError: pass

    def create(self, event: PassportEvent) -> PassportEvent:
        def action(rows):
            if any(x.id == event.id for x in rows): raise ValueError(f"passport event {event.id} already exists")
            rows.append(event); return event
        return self._mutate(action)

    def get(self, event_id: str) -> PassportEvent:
        with self._lock:
            for row in self._read():
                if row.id == event_id: return row
        raise PassportNotFoundError(f"passport event {event_id} not found")

    def list(self, *, well: str | None = None, event_type: PassportEventType | None = None,
             date_from: datetime | None = None, date_to: datetime | None = None,
             offset: int = 0, limit: int | None = None) -> list[PassportEvent]:
        if offset < 0 or (limit is not None and not 1 <= limit <= 1000): raise ValueError("offset must be non-negative and limit 1..1000")
        start = _aware(date_from, "date_from") if date_from else None; end = _aware(date_to, "date_to") if date_to else None
        if start and end and start > end: raise ValueError("date_from cannot be after date_to")
        query = well.strip().casefold() if well else None
        with self._lock: rows = self._read()
        result = [x for x in rows if (not query or x.well_id.casefold() == query or x.well_name.casefold() == query)
                  and (event_type is None or x.event_type is PassportEventType(event_type))
                  and (start is None or x.event_at >= start) and (end is None or x.event_at <= end)]
        result.sort(key=lambda x: (x.event_at, x.id), reverse=True); result = result[offset:]
        return result[:limit] if limit else result

    def update(self, event: PassportEvent, *, expected_revision: int) -> PassportEvent:
        def action(rows):
            for index, current in enumerate(rows):
                if current.id == event.id:
                    if current.revision != expected_revision: raise PassportConflictError(f"revision conflict: expected {expected_revision}, current {current.revision}")
                    if event.created_at != current.created_at: raise ValueError("created_at is immutable")
                    saved = replace(event, revision=current.revision + 1); rows[index] = saved; return saved
            raise PassportNotFoundError(f"passport event {event.id} not found")
        return self._mutate(action)

    def delete(self, event_id: str, *, expected_revision: int) -> PassportEvent:
        def action(rows):
            for index, current in enumerate(rows):
                if current.id == event_id:
                    if current.revision != expected_revision: raise PassportConflictError(f"revision conflict: expected {expected_revision}, current {current.revision}")
                    rows.pop(index); return current
            raise PassportNotFoundError(f"passport event {event_id} not found")
        removed = self._mutate(action)
        if removed.attachment:
            try: self.attachment_path(removed.attachment).unlink(missing_ok=True)
            except OSError: pass
        return removed

    def save_attachment(self, filename: str, mime_type: str, content: bytes) -> AttachmentMetadata:
        name = safe_attachment_name(filename)
        if mime_type not in ALLOWED_ATTACHMENT_MIMES: raise ValueError("unsupported attachment MIME type")
        if Path(name).suffix.lower() not in ALLOWED_ATTACHMENT_MIMES[mime_type]: raise ValueError("attachment extension does not match MIME type")
        if not 0 < len(content) <= MAX_ATTACHMENT_SIZE: raise ValueError(f"attachment size must be 1..{MAX_ATTACHMENT_SIZE} bytes")
        digest = hashlib.sha256(content).hexdigest(); attachment_id = str(uuid4())
        relative = Path(digest[:2]) / f"{attachment_id}-{name}"; target = self.attachment_root / relative
        target.parent.mkdir(parents=True, exist_ok=True); temp = target.with_name(f".{target.name}.tmp")
        try:
            with temp.open("xb") as stream: stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, target)
        except OSError as exc: raise PassportStorageError(f"failed to save attachment atomically: {exc}") from exc
        finally:
            try: temp.unlink(missing_ok=True)
            except OSError: pass
        return AttachmentMetadata(attachment_id, name, mime_type, len(content), digest, relative.as_posix(), datetime.now(timezone.utc))

    def attachment_path(self, metadata: AttachmentMetadata) -> Path:
        root = self.attachment_root.resolve(); path = (root / metadata.relative_path).resolve()
        if root not in path.parents: raise PassportStorageError("attachment path escapes storage root")
        return path


def new_passport_event(*, now: datetime | None = None, **values: Any) -> PassportEvent:
    timestamp = now or datetime.now(timezone.utc)
    return PassportEvent(id=str(uuid4()), created_at=timestamp, updated_at=timestamp, **values)


def passport_timeline(events: Iterable[PassportEvent], treatments: Iterable[Any] = ()) -> list[dict[str, Any]]:
    rows = [{"origin": "passport", **event.to_dict()} for event in events]
    for item in treatments:
        rows.append({"origin": "treatment", "id": item.id, "well_id": item.well_id,
                     "well_name": item.well_name, "event_type": "treatment", "event_at": item.event_at.isoformat(),
                     "title": item.treatment_type, "data": item.to_dict(), "notes": item.description})
    return sorted(rows, key=lambda row: (row["event_at"], row["id"]), reverse=True)


def passport_summary(events: Iterable[PassportEvent], treatments: Iterable[Any] = ()) -> dict[str, Any]:
    rows = list(events); treatment_rows = list(treatments)
    counts = {kind.value: sum(x.event_type is kind for x in rows) for kind in PassportEventType}
    latest_rate = next((x.data for x in sorted(rows, key=lambda x: x.event_at, reverse=True) if x.event_type is PassportEventType.RATE_CHANGE), None)
    latest_risk = next((x.data for x in sorted(rows, key=lambda x: x.event_at, reverse=True) if x.event_type is PassportEventType.RISK_SNAPSHOT), None)
    assessed = [x for x in treatment_rows if getattr(getattr(x, "status", None), "value", None) == "assessed"]
    return {"event_count": len(rows), "treatment_count": len(treatment_rows), "assessed_treatments": len(assessed),
            "counts": counts, "latest_rate": latest_rate, "latest_risk": latest_risk,
            "first_event_at": min((x.event_at for x in rows), default=None), "last_event_at": max((x.event_at for x in rows), default=None)}
