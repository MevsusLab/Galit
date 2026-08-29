"""Explainable equipment-failure screening for ESP (ЭЦН) and rod pumps (ШГН).

This is a deterministic engineering baseline, not a field-validated ML model.
All thresholds and weights are explicit so they can later be calibrated or replaced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import csv
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable
from uuid import uuid4

EQUIPMENT_SCHEMA_VERSION = 1
MODEL_VERSION = "galit-equipment-screening-1.0"
DISCLAIMER = ("Инженерная baseline/screening-оценка, не подтверждённый ML-прогноз. "
              "Требуется калибровка и валидация на размеченной истории отказов Белоруснефти; "
              "результат не является командой автоматического управления.")
PRESSURE_UNITS = {"pa", "kpa", "mpa", "bar"}
TEMPERATURE_UNITS = {"c"}
VIBRATION_UNITS = {"mm/s"}
LOAD_UNITS = {"kn"}


class LiftType(str, Enum):
    ESP = "ESP"
    ROD_PUMP = "ROD_PUMP"
    UNSUPPORTED = "UNSUPPORTED"


class EquipmentStorageError(RuntimeError): pass
class EquipmentNotFoundError(LookupError): pass
class EquipmentConflictError(RuntimeError): pass


def normalize_lift(value: str) -> LiftType:
    key = str(value).strip().casefold().replace("ё", "е")
    if key in {"esp", "эцн"}: return LiftType.ESP
    if key in {"rod_pump", "rod pump", "шгн", "сшну"}: return LiftType.ROD_PUMP
    return LiftType.UNSUPPORTED


def _text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result: raise ValueError(f"{name} must be non-empty")
    return result


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return value.astimezone(timezone.utc)


def _number(value: float | None, name: str, *, positive: bool = False) -> float | None:
    if value is None: return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return result


@dataclass(frozen=True)
class EquipmentMetadata:
    well: str
    lift_type: str
    equipment_id: str | None = None
    installed_at: datetime | None = None
    runtime_days: float | None = None
    nominal_current_a: float | None = None
    temperature_limit_c: float | None = None
    vibration_limit_mm_s: float | None = None
    load_limit_kn: float | None = None
    pressure_unit: str = "mpa"
    timezone_name: str = "UTC"

    def __post_init__(self) -> None:
        object.__setattr__(self, "well", _text(self.well, "well"))
        object.__setattr__(self, "lift_type", normalize_lift(self.lift_type).value)
        if self.equipment_id is not None: object.__setattr__(self, "equipment_id", _text(self.equipment_id, "equipment_id"))
        if self.installed_at is not None: object.__setattr__(self, "installed_at", _aware(self.installed_at, "installed_at"))
        for name in ("runtime_days", "nominal_current_a", "temperature_limit_c", "vibration_limit_mm_s", "load_limit_kn"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=name != "runtime_days"))
        unit = self.pressure_unit.strip().lower()
        if unit not in PRESSURE_UNITS: raise ValueError("pressure_unit must be pa, kpa, mpa, or bar")
        object.__setattr__(self, "pressure_unit", unit)
        if self.installed_at is None and self.runtime_days is None:
            raise ValueError("installed_at or runtime_days is required")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.installed_at: value["installed_at"] = self.installed_at.isoformat()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EquipmentMetadata":
        data = dict(value)
        if data.get("installed_at"): data["installed_at"] = datetime.fromisoformat(data["installed_at"])
        return cls(**data)


@dataclass(frozen=True)
class TelemetrySnapshot:
    well: str
    timestamp: datetime
    lift_type: str
    id: str | None = None
    current_a: float | None = None
    nominal_current_a: float | None = None
    intake_pressure: float | None = None
    discharge_pressure: float | None = None
    wellhead_pressure: float | None = None
    baseline_intake_pressure: float | None = None
    baseline_discharge_pressure: float | None = None
    pressure_unit: str = "mpa"
    motor_temperature_c: float | None = None
    fluid_temperature_c: float | None = None
    bearing_temperature_c: float | None = None
    temperature_limit_c: float | None = None
    vibration_mm_s: float | None = None
    vibration_limit_mm_s: float | None = None
    rod_load_kn: float | None = None
    load_limit_kn: float | None = None
    strokes_per_min: float | None = None
    dynamic_level_m: float | None = None
    baseline_dynamic_level_m: float | None = None
    fillage_fraction: float | None = None
    sand_fraction: float | None = None
    halite_risk: float | None = None
    calcite_risk: float | None = None
    wax_risk: float | None = None
    corrosion_risk: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "well", _text(self.well, "well"))
        object.__setattr__(self, "timestamp", _aware(self.timestamp, "timestamp"))
        object.__setattr__(self, "lift_type", normalize_lift(self.lift_type).value)
        object.__setattr__(self, "id", self.id or f"{self.well}|{self.timestamp.isoformat()}")
        unit = self.pressure_unit.strip().lower()
        if unit not in PRESSURE_UNITS: raise ValueError("pressure_unit must be pa, kpa, mpa, or bar")
        object.__setattr__(self, "pressure_unit", unit)
        for name in ("current_a", "nominal_current_a", "intake_pressure", "discharge_pressure", "wellhead_pressure",
                     "baseline_intake_pressure", "baseline_discharge_pressure", "motor_temperature_c", "fluid_temperature_c",
                     "bearing_temperature_c", "temperature_limit_c", "vibration_mm_s", "vibration_limit_mm_s", "rod_load_kn",
                     "load_limit_kn", "strokes_per_min", "dynamic_level_m", "baseline_dynamic_level_m"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in ("fillage_fraction", "sand_fraction", "halite_risk", "calcite_risk", "wax_risk", "corrosion_risk"):
            value = _number(getattr(self, name), name)
            if value is not None and value > 1: raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        if self.intake_pressure is not None and self.discharge_pressure is not None and self.discharge_pressure < self.intake_pressure:
            raise ValueError("discharge_pressure cannot be below intake_pressure in the same unit")
        if normalize_lift(self.lift_type) is LiftType.ESP and self.rod_load_kn is not None:
            raise ValueError("rod_load_kn is not applicable to ESP")
        if normalize_lift(self.lift_type) is LiftType.ROD_PUMP and self.current_a is not None:
            raise ValueError("current_a is not applicable to rod pump")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self); value["timestamp"] = self.timestamp.isoformat(); return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TelemetrySnapshot":
        data = dict(value); data["timestamp"] = datetime.fromisoformat(data["timestamp"]); return cls(**data)


@dataclass(frozen=True)
class EquipmentRiskPolicy:
    version: str = MODEL_VERSION
    telemetry_weight: float = .75
    complication_weight: float = .25
    warn_threshold: float = .35
    critical_threshold: float = .70
    normal_rul_days: tuple[int, int] = (120, 365)


@dataclass(frozen=True)
class RiskCause:
    code: str
    label: str
    group: str
    indicator: float
    weight: float
    contribution: float
    explanation: str


@dataclass(frozen=True)
class EquipmentForecast:
    well: str
    lift_type: str
    status: str
    baseline_failure_risk: float | None
    risk_level: str
    rul_days: tuple[int, int] | None
    causes: tuple[RiskCause, ...]
    anomalies: dict[str, float]
    maintenance_window_start: datetime | None
    maintenance_window_end: datetime | None
    urgency: str
    recommended_action: str
    data_completeness: float
    confidence: str
    missing_fields: tuple[str, ...]
    model_version: str
    assumptions: tuple[str, ...]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("maintenance_window_start", "maintenance_window_end"):
            if value[key] is not None: value[key] = value[key].isoformat()
        return value


def _ratio(value: float | None, baseline: float | None, tolerance: float = .1) -> float:
    if value is None or baseline is None or baseline <= 0: return 0.0
    return min(max(abs(value / baseline - 1) - tolerance, 0.0) / .5, 1.0)


def _high(value: float | None, limit: float | None, start: float = .8) -> float:
    if value is None or limit is None or limit <= 0: return 0.0
    return min(max(value / limit - start, 0.0) / (1.0 - start), 1.0)


def _trend(history: list[TelemetrySnapshot], field: str) -> float:
    points = [(x.timestamp, getattr(x, field)) for x in history if getattr(x, field) is not None]
    if len(points) < 3: return 0.0
    points.sort(); first, last = float(points[0][1]), float(points[-1][1])
    baseline = max(abs(first), 1e-9)
    return min(max((last - first) / baseline, 0.0) / .30, 1.0)


def forecast_equipment(metadata: EquipmentMetadata, history: Iterable[TelemetrySnapshot], *,
                       as_of: datetime | None = None, policy: EquipmentRiskPolicy | None = None) -> EquipmentForecast:
    policy = policy or EquipmentRiskPolicy()
    now = _aware(as_of or datetime.now(timezone.utc), "as_of")
    lift = normalize_lift(metadata.lift_type)
    rows = sorted([x for x in history if x.well.casefold() == metadata.well.casefold()], key=lambda x: x.timestamp)
    if lift is LiftType.UNSUPPORTED:
        return EquipmentForecast(metadata.well, lift.value, "not_applicable", None, "not_applicable", None, (), {}, None, None,
                                 "none", "Прогноз применим только к ЭЦН и ШГН.", 0, "unavailable", (), policy.version,
                                 ("Способ эксплуатации не поддерживается baseline-моделью.",))
    if not rows:
        return EquipmentForecast(metadata.well, lift.value, "insufficient_data", None, "unavailable", None, (), {}, None, None,
                                 "unknown", "Загрузите хотя бы один телеметрический snapshot.", 0, "unavailable",
                                 ("telemetry",), policy.version, ("Телеметрия отсутствует.",))
    latest = rows[-1]
    if normalize_lift(latest.lift_type) is not lift: raise ValueError("metadata and telemetry lift_type mismatch")
    indicators: list[tuple[str, str, str, float, float, str]] = []
    missing: list[str] = []
    if lift is LiftType.ESP:
        nominal = latest.nominal_current_a or metadata.nominal_current_a
        current = _ratio(latest.current_a, nominal)
        if latest.current_a is None or nominal is None: missing.append("current_a/nominal_current_a")
        indicators.append(("current", "Аномалия тока", "telemetry", max(current, _trend(rows, "current_a")), .25, "Отклонение тока от номинала/рост по истории."))
    else:
        load_limit = latest.load_limit_kn or metadata.load_limit_kn
        rod = max(_high(latest.rod_load_kn, load_limit), _trend(rows, "rod_load_kn"))
        fill = 0 if latest.fillage_fraction is None else min(max((.8-latest.fillage_fraction)/.5, 0), 1)
        level = _ratio(latest.dynamic_level_m, latest.baseline_dynamic_level_m)
        if latest.rod_load_kn is None: missing.append("rod_load_kn")
        indicators += [("rod_load", "Нагрузка ШГН", "telemetry", rod, .18, "Рост/превышение нагрузки на штанги."),
                       ("fillage", "Наполнение насоса", "telemetry", fill, .12, "Снижение коэффициента наполнения."),
                       ("dynamic_level", "Динамический уровень", "telemetry", level, .08, "Отклонение динамического уровня от baseline.")]
    pressure = max(_ratio(latest.intake_pressure, latest.baseline_intake_pressure),
                   _ratio(latest.discharge_pressure, latest.baseline_discharge_pressure),
                   _trend(rows, "discharge_pressure"))
    if latest.intake_pressure is None and latest.discharge_pressure is None: missing.append("pressure")
    temp_limit = latest.temperature_limit_c or metadata.temperature_limit_c
    temp = max(_high(v, temp_limit) for v in (latest.motor_temperature_c, latest.fluid_temperature_c, latest.bearing_temperature_c))
    temp = max(temp, _trend(rows, "motor_temperature_c"), _trend(rows, "bearing_temperature_c"))
    if all(v is None for v in (latest.motor_temperature_c, latest.fluid_temperature_c, latest.bearing_temperature_c)): missing.append("temperature")
    vibration = max(_high(latest.vibration_mm_s, latest.vibration_limit_mm_s or metadata.vibration_limit_mm_s), _trend(rows, "vibration_mm_s"))
    if latest.vibration_mm_s is None: missing.append("vibration_mm_s")
    sand = latest.sand_fraction or 0.0
    indicators += [("pressure", "Аномалия давления", "telemetry", pressure, .17, "Отклонение intake/discharge от baseline или неблагоприятный тренд."),
                   ("temperature", "Аномалия температуры", "telemetry", temp, .17, "Приближение к пределу или рост температуры."),
                   ("vibration", "Аномалия вибрации", "telemetry", vibration, .16, "Приближение к пределу или рост вибрации."),
                   ("sand", "Песок/твёрдые частицы", "process", sand, .07, "Индикатор абразивного износа.")]
    complications = {"halite": latest.halite_risk, "calcite": latest.calcite_risk,
                     "wax": latest.wax_risk, "corrosion": latest.corrosion_risk}
    for key, value in complications.items():
        if value is not None:
            indicators.append((key, {"halite":"Галит", "calcite":"Кальцит", "wax":"АСПО", "corrosion":"Коррозия"}[key],
                               "process", value, .0625, "Отдельный процессный индикатор GALIT; общий лимит веса исключает double counting."))
    telemetry_raw = sum(score * weight for _,_,group,score,weight,_ in indicators if group == "telemetry")
    telemetry_den = sum(weight for *_, group, score, weight, explanation in []) if False else sum(x[4] for x in indicators if x[2] == "telemetry")
    process_raw = sum(score * weight for _,_,group,score,weight,_ in indicators if group == "process")
    process_den = sum(x[4] for x in indicators if x[2] == "process")
    telemetry_score = telemetry_raw / telemetry_den if telemetry_den else 0
    process_score = process_raw / process_den if process_den else 0
    risk = min(max(policy.telemetry_weight * telemetry_score + policy.complication_weight * process_score, 0), 1)
    causes = tuple(sorted((RiskCause(code, label, group, score, weight,
                                     (policy.telemetry_weight if group == "telemetry" else policy.complication_weight) * score * weight / (telemetry_den if group == "telemetry" else process_den), explanation)
                                  for code,label,group,score,weight,explanation in indicators if score > 0),
                                 key=lambda x: (-x.contribution, x.code)))
    expected = ["current", "pressure", "temperature", "vibration"] if lift is LiftType.ESP else ["rod_load", "pressure", "temperature", "vibration", "fillage"]
    available = len([x for x in expected if not any(x in item for item in missing)])
    completeness = available / len(expected)
    confidence = "high" if len(rows) >= 5 and completeness >= .8 else "medium" if completeness >= .6 else "low"
    if risk >= policy.critical_threshold: level, urgency, days = "critical", "immediate_engineering_review", (0, 2)
    elif risk >= policy.warn_threshold: level, urgency, days = "warning", "plan_soon", (7, 30)
    else: level, urgency, days = "normal", "planned", policy.normal_rul_days
    if confidence == "low":
        rul = None if completeness < .4 else (days[0], max(days[1], 180))
    else: rul = days
    start = now + timedelta(days=days[0]); end = now + timedelta(days=days[1])
    critical_safety = temp >= 1 or vibration >= 1 or (lift is LiftType.ROD_PUMP and any(x.code == "rod_load" and x.indicator >= 1 for x in causes))
    if critical_safety:
        level, urgency, days, rul = "critical", "immediate_engineering_review", (0, 2), (0, 2)
        start, end = now, now + timedelta(days=2)
    action = ("Немедленная инженерная проверка и безопасная остановка по действующим регламентам; не выполнять автоматическое управление."
              if critical_safety else "Проверить режим, тренды и первопричины; спланировать диагностику оборудования в указанном окне.")
    assumptions = ("Нормализованные пороги — инженерная политика, а не обученная вероятность.",
                   "Telemetry и process complications агрегируются раздельно (75%/25%).",
                   "Одиночный snapshot снижает confidence; тренд используется при наличии истории.")
    return EquipmentForecast(metadata.well, lift.value, "screening", round(risk, 6), level, rul, causes,
                             {x.code: x.indicator for x in causes if x.group == "telemetry"}, start, end, urgency,
                             action, completeness, confidence, tuple(sorted(set(missing))), policy.version, assumptions)


class EquipmentRepository:
    _locks: dict[str, threading.RLock] = {}; _guard = threading.Lock()
    def __init__(self, path: str | Path = "data/equipment.json", lock_timeout: float = 5):
        self.path = Path(path); self.lock_timeout = lock_timeout
        with self._guard: self._lock = self._locks.setdefault(str(self.path.resolve()), threading.RLock())
    def _read(self) -> tuple[list[EquipmentMetadata], list[TelemetrySnapshot]]:
        if not self.path.exists(): return [], []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != EQUIPMENT_SCHEMA_VERSION: raise ValueError("unsupported schema")
            equipment = [EquipmentMetadata.from_dict(x) for x in payload.get("equipment", [])]
            telemetry = [TelemetrySnapshot.from_dict(x) for x in payload.get("telemetry", [])]
            if len({x.well.casefold() for x in equipment}) != len(equipment): raise ValueError("duplicate equipment well")
            if len({x.id for x in telemetry}) != len(telemetry): raise ValueError("duplicate telemetry id")
            return equipment, telemetry
        except Exception as exc:
            if isinstance(exc, EquipmentStorageError): raise
            raise EquipmentStorageError(f"equipment storage corrupt or unreadable: {exc}") from exc
    def _write(self, equipment, telemetry):
        self.path.parent.mkdir(parents=True, exist_ok=True); temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        payload = {"schema_version": EQUIPMENT_SCHEMA_VERSION, "equipment": [x.to_dict() for x in equipment], "telemetry": [x.to_dict() for x in telemetry]}
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, self.path)
        except OSError as exc: raise EquipmentStorageError(f"atomic write failed: {exc}") from exc
        finally:
            try: temp.unlink(missing_ok=True)
            except OSError: pass
    def _mutate(self, fn):
        with self._lock:
            lock = self.path.with_suffix(self.path.suffix + ".lock"); deadline = time.monotonic() + self.lock_timeout
            self.path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                try: fd=os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd); break
                except FileExistsError:
                    if time.monotonic() >= deadline: raise EquipmentStorageError("timed out waiting for lock")
                    time.sleep(.02)
            try:
                equipment, telemetry = self._read(); result=fn(equipment, telemetry); self._write(equipment, telemetry); return result
            finally:
                try: lock.unlink(missing_ok=True)
                except OSError: pass
    def upsert_equipment(self, item: EquipmentMetadata) -> EquipmentMetadata:
        def action(equipment, telemetry):
            for i, old in enumerate(equipment):
                if old.well.casefold() == item.well.casefold(): equipment[i]=item; return item
            equipment.append(item); return item
        return self._mutate(action)
    def ingest(self, item: TelemetrySnapshot, *, idempotent: bool = True) -> TelemetrySnapshot:
        def action(equipment, telemetry):
            if any(x.id == item.id for x in telemetry):
                if idempotent and next(x for x in telemetry if x.id == item.id) == item: return item
                raise EquipmentConflictError(f"telemetry {item.id} already exists")
            telemetry.append(item); return item
        return self._mutate(action)
    def list_equipment(self) -> list[EquipmentMetadata]: return self._read()[0]
    def get_equipment(self, well: str) -> EquipmentMetadata:
        for x in self.list_equipment():
            if x.well.casefold() == well.strip().casefold(): return x
        raise EquipmentNotFoundError(f"equipment for {well} not found")
    def list_telemetry(self, well: str | None = None) -> list[TelemetrySnapshot]:
        rows=self._read()[1]
        if well: rows=[x for x in rows if x.well.casefold() == well.strip().casefold()]
        return sorted(rows, key=lambda x:(x.timestamp,x.id or ""))
    def forecast(self, well: str, *, as_of: datetime | None = None) -> EquipmentForecast:
        return forecast_equipment(self.get_equipment(well), self.list_telemetry(well), as_of=as_of)
    def portfolio(self, *, as_of: datetime | None = None) -> list[EquipmentForecast]:
        return sorted((forecast_equipment(x, self.list_telemetry(x.well), as_of=as_of) for x in self.list_equipment()),
                      key=lambda x: (-(x.baseline_failure_risk or -1), x.well.casefold()))


CSV_COLUMNS = tuple(TelemetrySnapshot.__dataclass_fields__)
def telemetry_csv_template() -> bytes:
    stream=io.StringIO(); writer=csv.DictWriter(stream, fieldnames=CSV_COLUMNS); writer.writeheader();
    writer.writerow({"well":"Well-1","timestamp":"2026-08-23T12:00:00+00:00","lift_type":"ESP","pressure_unit":"mpa"})
    return ("\ufeff"+stream.getvalue()).encode("utf-8")
def telemetry_to_csv(rows: Iterable[TelemetrySnapshot]) -> bytes:
    stream=io.StringIO(); writer=csv.DictWriter(stream, fieldnames=CSV_COLUMNS); writer.writeheader()
    for row in rows: writer.writerow(row.to_dict())
    return ("\ufeff"+stream.getvalue()).encode("utf-8")
def telemetry_from_csv(data: bytes | str) -> list[TelemetrySnapshot]:
    text=data.decode("utf-8-sig") if isinstance(data, bytes) else data.lstrip("\ufeff")
    result=[]
    for index,row in enumerate(csv.DictReader(io.StringIO(text)),2):
        values={k:v for k,v in row.items() if v not in (None, "")}
        try:
            values["timestamp"]=datetime.fromisoformat(values["timestamp"].replace("Z","+00:00"))
            numeric=set(CSV_COLUMNS)-{"well","timestamp","lift_type","id","pressure_unit"}
            for key in numeric:
                if key in values: values[key]=float(values[key])
            result.append(TelemetrySnapshot(**values))
        except Exception as exc: raise ValueError(f"telemetry CSV row {index}: {exc}") from exc
    return result
