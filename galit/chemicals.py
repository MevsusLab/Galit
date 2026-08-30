"""Evidence-gated chemical selection and append-only local inventory ledger.

This is a deterministic, single-process prototype.  Efficacy is never inferred:
recommendations require validated, referenced, tested dose-response evidence for
one product covering every actionable hazard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Sequence
from uuid import uuid4

SCHEMA_VERSION = 1
CANONICAL_DOSE_UNIT = "kg/m3"
VALID_CURRENCIES = {"BYN", "RUB", "USD", "EUR"}
TRANSACTION_KINDS = {"receipt", "consumption", "adjustment", "expiry", "release"}


class ChemicalStorageError(RuntimeError):
    pass


class ChemicalNotFoundError(LookupError):
    pass


class ChemicalConflictError(RuntimeError):
    pass


def _text(value: str, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _decimal(value: Any, name: str, *, positive: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return value.astimezone(timezone.utc)


def convert_quantity(value: Any, from_unit: str, to_unit: str, *,
                     density_kg_l: Any | None = None) -> Decimal:
    """Convert chemical quantities exactly through kg; density gates mass/volume."""
    amount = _decimal(value, "value")
    source, target = from_unit.lower().strip(), to_unit.lower().strip()
    factors = {"kg": Decimal("1"), "g": Decimal("0.001"), "mg": Decimal("0.000001")}
    volumes = {"l": Decimal("1"), "ml": Decimal("0.001"), "m3": Decimal("1000")}
    if source == target:
        return amount
    if source in factors and target in factors:
        return amount * factors[source] / factors[target]
    if source in volumes and target in volumes:
        return amount * volumes[source] / volumes[target]
    if density_kg_l is None:
        raise ValueError("density_kg_l is required for mass-volume conversion")
    density = _decimal(density_kg_l, "density_kg_l", positive=True)
    kg = amount * factors[source] if source in factors else amount * volumes[source] * density if source in volumes else None
    if kg is None or (target not in factors and target not in volumes):
        raise ValueError("supported units are mg, g, kg, ml, l, m3")
    return kg / factors[target] if target in factors else kg / density / volumes[target]


def dose_to_kg_m3(value: Any, unit: str) -> Decimal:
    """mg/L and kg/m3 are dimensionally identical; no rounding is performed."""
    unit = unit.lower().strip().replace("³", "3")
    amount = _decimal(value, "dose")
    if unit in {"kg/m3", "mg/l"}:
        return amount
    if unit == "g/m3":
        return amount / Decimal("1000")
    raise ValueError("dose unit must be kg/m3, mg/L, or g/m3")


@dataclass(frozen=True)
class ChemicalProduct:
    id: str
    name: str
    manufacturer: str
    hazards: tuple[str, ...]
    compatible_with: tuple[str, ...] = ()
    density_kg_l: Decimal | None = None
    price_per_kg: Decimal | None = None
    currency: str | None = None
    active: bool = True
    notes: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "name", "manufacturer"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        hazards = tuple(sorted({_text(x, "hazard").lower() for x in self.hazards}))
        if not hazards:
            raise ValueError("hazards must not be empty")
        object.__setattr__(self, "hazards", hazards)
        object.__setattr__(self, "compatible_with", tuple(sorted({_text(x, "compatible_with") for x in self.compatible_with})))
        if self.density_kg_l is not None:
            object.__setattr__(self, "density_kg_l", _decimal(self.density_kg_l, "density_kg_l", positive=True))
        if self.price_per_kg is not None:
            object.__setattr__(self, "price_per_kg", _decimal(self.price_per_kg, "price_per_kg"))
            if self.currency is None:
                raise ValueError("currency is required with price_per_kg")
        if self.currency is not None:
            currency = self.currency.upper().strip()
            if currency not in VALID_CURRENCIES:
                raise ValueError("unsupported currency")
            object.__setattr__(self, "currency", currency)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("density_kg_l", "price_per_kg"):
            data[key] = str(data[key]) if data[key] is not None else None
        data["hazards"], data["compatible_with"] = list(self.hazards), list(self.compatible_with)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChemicalProduct":
        return cls(**data)


@dataclass(frozen=True)
class DoseResponsePoint:
    dose_kg_m3: Decimal
    effective: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "dose_kg_m3", _decimal(self.dose_kg_m3, "dose_kg_m3"))
        if not isinstance(self.effective, bool):
            raise ValueError("effective must be boolean")


@dataclass(frozen=True)
class DoseResponseEnvelope:
    id: str
    product_id: str
    hazard: str
    points: tuple[DoseResponsePoint, ...]
    validated: bool
    validation_reference: str | None
    conditions: str
    revision: int = 1

    def __post_init__(self) -> None:
        for name in ("id", "product_id", "hazard", "conditions"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "hazard", self.hazard.lower())
        points = tuple(sorted((p if isinstance(p, DoseResponsePoint) else DoseResponsePoint(**p) for p in self.points), key=lambda p: p.dose_kg_m3))
        if not points or len({p.dose_kg_m3 for p in points}) != len(points):
            raise ValueError("points require unique tested doses")
        object.__setattr__(self, "points", points)
        if self.validated and not (self.validation_reference and self.validation_reference.strip()):
            raise ValueError("validated evidence requires validation_reference")
        if self.validation_reference is not None:
            object.__setattr__(self, "validation_reference", _text(self.validation_reference, "validation_reference"))
        if self.revision < 1:
            raise ValueError("revision must be positive")

    @property
    def minimum_effective_dose(self) -> Decimal | None:
        return next((p.dose_kg_m3 for p in self.points if p.effective), None) if self.validated else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [{"dose_kg_m3": str(p.dose_kg_m3), "effective": p.effective} for p in self.points]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoseResponseEnvelope":
        values = dict(data)
        values["points"] = tuple(DoseResponsePoint(**p) for p in values["points"])
        return cls(**values)


@dataclass(frozen=True)
class ChemicalRecommendation:
    status: str
    hazards: tuple[str, ...]
    product_id: str | None = None
    product_name: str | None = None
    dose_kg_m3: Decimal | None = None
    daily_consumption_kg: Decimal | None = None
    daily_cost: Decimal | None = None
    cost_per_m3_oil: Decimal | None = None
    currency: str | None = None
    evidence_ids: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("dose_kg_m3", "daily_consumption_kg", "daily_cost", "cost_per_m3_oil"):
            data[key] = str(data[key]) if data[key] is not None else None
        data["hazards"], data["evidence_ids"] = list(self.hazards), list(self.evidence_ids)
        return data


def recommend_products(products: Iterable[ChemicalProduct], envelopes: Iterable[DoseResponseEnvelope],
                       hazards: Sequence[str], treated_fluid_m3_day: Any,
                       oil_m3_day: Any) -> list[ChemicalRecommendation]:
    wanted = tuple(sorted({_text(h, "hazard").lower() for h in hazards}))
    treated, oil = _decimal(treated_fluid_m3_day, "treated_fluid_m3_day"), _decimal(oil_m3_day, "oil_m3_day")
    if not wanted:
        return [ChemicalRecommendation("unavailable", wanted, reason="no actionable hazards supplied")]
    evidence = list(envelopes)
    ranked: list[ChemicalRecommendation] = []
    for product in products:
        if not product.active or not set(wanted).issubset(product.hazards):
            continue
        selected: list[DoseResponseEnvelope] = []
        for hazard in wanted:
            candidates = [e for e in evidence if e.product_id == product.id and e.hazard == hazard and e.validated and e.minimum_effective_dose is not None]
            if not candidates:
                selected = []
                break
            selected.append(sorted(candidates, key=lambda e: (e.minimum_effective_dose, e.id))[0])
        if not selected:
            continue
        # A multi-hazard use is allowed only when compatibility is explicitly known.
        if len(wanted) > 1 and not set(wanted).issubset(set(product.compatible_with)):
            continue
        dose = max(e.minimum_effective_dose for e in selected if e.minimum_effective_dose is not None)
        daily = dose * treated
        cost = daily * product.price_per_kg if product.price_per_kg is not None else None
        ranked.append(ChemicalRecommendation(
            "available", wanted, product.id, product.name, dose, daily, cost,
            None if oil == 0 or cost is None else cost / oil,
            product.currency, tuple(sorted(e.id for e in selected)), None,
        ))
    if not ranked:
        return [ChemicalRecommendation("unavailable", wanted, reason="no single compatible product has validated referenced effective tests for every hazard")]
    ranked.sort(key=lambda r: (r.cost_per_m3_oil is None, r.cost_per_m3_oil or Decimal(0), r.dose_kg_m3 or Decimal(0), r.product_id or ""))
    return ranked


@dataclass(frozen=True)
class StockLot:
    id: str
    product_id: str
    received_at: datetime
    expires_on: date
    initial_quantity_kg: Decimal
    unit_cost: Decimal | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "product_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "received_at", _aware(self.received_at, "received_at"))
        object.__setattr__(self, "initial_quantity_kg", _decimal(self.initial_quantity_kg, "initial_quantity_kg", positive=True))
        if self.expires_on < self.received_at.date():
            raise ValueError("expires_on cannot precede received_at")
        if self.unit_cost is not None:
            object.__setattr__(self, "unit_cost", _decimal(self.unit_cost, "unit_cost"))
            if not self.currency:
                raise ValueError("currency is required with unit_cost")
        if self.currency:
            currency = self.currency.upper()
            if currency not in VALID_CURRENCIES:
                raise ValueError("unsupported currency")
            object.__setattr__(self, "currency", currency)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["received_at"] = self.received_at.isoformat(); data["expires_on"] = self.expires_on.isoformat()
        data["initial_quantity_kg"] = str(self.initial_quantity_kg); data["unit_cost"] = str(self.unit_cost) if self.unit_cost is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StockLot":
        values = dict(data); values["received_at"] = datetime.fromisoformat(values["received_at"]); values["expires_on"] = date.fromisoformat(values["expires_on"])
        return cls(**values)


@dataclass(frozen=True)
class StockTransaction:
    id: str
    idempotency_key: str
    product_id: str
    lot_id: str
    kind: str
    quantity_kg: Decimal
    occurred_at: datetime
    reference: str

    def __post_init__(self) -> None:
        for name in ("id", "idempotency_key", "product_id", "lot_id", "reference"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.kind not in TRANSACTION_KINDS:
            raise ValueError("unsupported transaction kind")
        object.__setattr__(self, "quantity_kg", _decimal(self.quantity_kg, "quantity_kg", positive=True))
        object.__setattr__(self, "occurred_at", _aware(self.occurred_at, "occurred_at"))

    @property
    def signed_quantity(self) -> Decimal:
        return self.quantity_kg if self.kind in {"receipt", "adjustment", "release"} else -self.quantity_kg

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["quantity_kg"] = str(self.quantity_kg); data["occurred_at"] = self.occurred_at.isoformat(); return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StockTransaction":
        values = dict(data); values["occurred_at"] = datetime.fromisoformat(values["occurred_at"]); return cls(**values)


@dataclass(frozen=True)
class StockReservation:
    id: str
    idempotency_key: str
    product_id: str
    quantity_kg: Decimal
    created_at: datetime
    required_on: date
    allocations: tuple[tuple[str, Decimal], ...]
    status: str = "active"
    revision: int = 1

    def __post_init__(self) -> None:
        for name in ("id", "idempotency_key", "product_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "quantity_kg", _decimal(self.quantity_kg, "quantity_kg", positive=True))
        object.__setattr__(self, "created_at", _aware(self.created_at, "created_at"))
        allocations = tuple((str(lot), _decimal(qty, "allocation", positive=True)) for lot, qty in self.allocations)
        if sum((q for _, q in allocations), Decimal(0)) != self.quantity_kg:
            raise ValueError("allocations must equal reservation quantity")
        object.__setattr__(self, "allocations", allocations)
        if self.status not in {"active", "released", "consumed"} or self.revision < 1:
            raise ValueError("invalid reservation status or revision")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["quantity_kg"] = str(self.quantity_kg); data["created_at"] = self.created_at.isoformat(); data["required_on"] = self.required_on.isoformat()
        data["allocations"] = [{"lot_id": lot, "quantity_kg": str(qty)} for lot, qty in self.allocations]; return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StockReservation":
        values = dict(data); values["created_at"] = datetime.fromisoformat(values["created_at"]); values["required_on"] = date.fromisoformat(values["required_on"])
        values["allocations"] = tuple((x["lot_id"], x["quantity_kg"]) for x in values["allocations"]); return cls(**values)


class ChemicalRepository:
    """Versioned atomic JSON store. Transactions are immutable and append-only."""
    def __init__(self, path: str | Path = "data/chemicals.json"):
        self.path = Path(path)
        self._lock = threading.RLock()

    def _empty(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "revision": 0, "products": [], "envelopes": [], "lots": [], "transactions": [], "reservations": []}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists(): return self._empty()
        try: data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc: raise ChemicalStorageError(f"chemical storage unreadable: {exc}") from exc
        if data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("revision"), int): raise ChemicalStorageError("unsupported chemical storage schema")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True); temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, self.path)
        except OSError as exc: raise ChemicalStorageError(f"atomic chemical storage write failed: {exc}") from exc
        finally: temp.unlink(missing_ok=True)

    def _mutate(self, expected_revision: int | None, action):
        with self._lock:
            data = self._read()
            if expected_revision is not None and data["revision"] != expected_revision: raise ChemicalConflictError(f"revision conflict: expected {expected_revision}, current {data['revision']}")
            result, changed = action(data)
            if changed: data["revision"] += 1; self._write(data)
            return result

    @property
    def revision(self) -> int:
        with self._lock: return self._read()["revision"]

    def list_products(self) -> list[ChemicalProduct]:
        with self._lock: return [ChemicalProduct.from_dict(x) for x in self._read()["products"]]

    def put_product(self, item: ChemicalProduct, *, expected_revision: int | None = None) -> ChemicalProduct:
        def action(data):
            for i, old in enumerate(data["products"]):
                if old["id"] == item.id: data["products"][i] = item.to_dict(); return item, old != item.to_dict()
            data["products"].append(item.to_dict()); return item, True
        return self._mutate(expected_revision, action)

    def list_envelopes(self, product_id: str | None = None) -> list[DoseResponseEnvelope]:
        with self._lock: rows = [DoseResponseEnvelope.from_dict(x) for x in self._read()["envelopes"]]
        return [x for x in rows if product_id is None or x.product_id == product_id]

    def put_envelope(self, item: DoseResponseEnvelope, *, expected_revision: int | None = None) -> DoseResponseEnvelope:
        def action(data):
            if not any(x["id"] == item.product_id for x in data["products"]): raise ChemicalNotFoundError(f"product {item.product_id} not found")
            for i, old in enumerate(data["envelopes"]):
                if old["id"] == item.id:
                    current = DoseResponseEnvelope.from_dict(old)
                    if item.revision != current.revision: raise ChemicalConflictError("envelope revision conflict")
                    saved = DoseResponseEnvelope(**{**asdict(item), "revision": current.revision + 1}); data["envelopes"][i] = saved.to_dict(); return saved, True
            data["envelopes"].append(item.to_dict()); return item, True
        return self._mutate(expected_revision, action)

    def list_lots(self, product_id: str | None = None) -> list[StockLot]:
        with self._lock: rows = [StockLot.from_dict(x) for x in self._read()["lots"]]
        return [x for x in rows if product_id is None or x.product_id == product_id]

    def add_lot(self, lot: StockLot, *, idempotency_key: str, expected_revision: int | None = None) -> StockLot:
        key = _text(idempotency_key, "idempotency_key")
        def action(data):
            previous = next((x for x in data["transactions"] if x["idempotency_key"] == key), None)
            if previous:
                old_lot = next((x for x in data["lots"] if x["id"] == previous["lot_id"]), None)
                if old_lot == lot.to_dict() and previous["kind"] == "receipt": return lot, False
                raise ChemicalConflictError("idempotency key already used with different payload")
            if not any(x["id"] == lot.product_id for x in data["products"]):
                raise ChemicalNotFoundError(f"product {lot.product_id} not found")
            if any(x["id"] == lot.id for x in data["lots"]): raise ChemicalConflictError(f"lot {lot.id} already exists")
            data["lots"].append(lot.to_dict())
            tx = StockTransaction(str(uuid4()), key, lot.product_id, lot.id, "receipt", lot.initial_quantity_kg, lot.received_at, f"lot:{lot.id}")
            data["transactions"].append(tx.to_dict()); return lot, True
        return self._mutate(expected_revision, action)

    def list_transactions(self, product_id: str | None = None) -> list[StockTransaction]:
        with self._lock: rows = [StockTransaction.from_dict(x) for x in self._read()["transactions"]]
        return [x for x in rows if product_id is None or x.product_id == product_id]

    def list_reservations(self, product_id: str | None = None) -> list[StockReservation]:
        with self._lock: rows = [StockReservation.from_dict(x) for x in self._read()["reservations"]]
        return [x for x in rows if product_id is None or x.product_id == product_id]

    def append_transaction(self, item: StockTransaction, *, expected_revision: int | None = None) -> StockTransaction:
        """Append an idempotent adjustment/expiry/release entry; existing entries are immutable."""
        if item.kind in {"receipt", "consumption"}:
            raise ValueError("receipt and consumption must use lot and consume operations")
        def action(data):
            previous = next((StockTransaction.from_dict(x) for x in data["transactions"]
                             if x["idempotency_key"] == item.idempotency_key), None)
            if previous:
                if previous.to_dict() == item.to_dict():
                    return previous, False
                raise ChemicalConflictError("idempotency key already used with different payload")
            lot_raw = next((x for x in data["lots"] if x["id"] == item.lot_id), None)
            if lot_raw is None:
                raise ChemicalNotFoundError(f"lot {item.lot_id} not found")
            if lot_raw["product_id"] != item.product_id:
                raise ChemicalConflictError("transaction product does not match lot product")
            current = sum((StockTransaction.from_dict(x).signed_quantity
                           for x in data["transactions"] if x["lot_id"] == item.lot_id), Decimal(0))
            if current + item.signed_quantity < 0:
                raise ChemicalConflictError("transaction would make lot stock negative")
            data["transactions"].append(item.to_dict())
            return item, True
        return self._mutate(expected_revision, action)

    def release_reservation(self, reservation_id: str, *, revision: int,
                            expected_revision: int | None = None) -> StockReservation:
        def action(data):
            for index, raw in enumerate(data["reservations"]):
                if raw["id"] != reservation_id:
                    continue
                current = StockReservation.from_dict(raw)
                if current.revision != revision:
                    raise ChemicalConflictError("reservation revision conflict")
                if current.status == "released":
                    return current, False
                if current.status != "active":
                    raise ChemicalConflictError("only active reservations can be released")
                saved = StockReservation(**{**asdict(current), "status": "released",
                                            "revision": current.revision + 1})
                data["reservations"][index] = saved.to_dict()
                return saved, True
            raise ChemicalNotFoundError(f"reservation {reservation_id} not found")
        return self._mutate(expected_revision, action)

    def _balances(self, data: dict[str, Any], product_id: str, as_of: date, include_reserved: bool = True) -> list[tuple[StockLot, Decimal]]:
        txs = [StockTransaction.from_dict(x) for x in data["transactions"]]
        reservations = [StockReservation.from_dict(x) for x in data["reservations"]]
        rows = []
        for raw in data["lots"]:
            lot = StockLot.from_dict(raw)
            if lot.product_id != product_id or lot.expires_on < as_of: continue
            balance = sum((x.signed_quantity for x in txs if x.lot_id == lot.id), Decimal(0))
            if include_reserved:
                balance -= sum((q for r in reservations if r.status == "active" for lid, q in r.allocations if lid == lot.id), Decimal(0))
            if balance > 0: rows.append((lot, balance))
        return sorted(rows, key=lambda x: (x[0].expires_on, x[0].received_at, x[0].id))

    def stock(self, product_id: str, *, as_of: date | None = None) -> dict[str, Any]:
        day = as_of or date.today()
        with self._lock: data = self._read(); rows = self._balances(data, product_id, day)
        return {"product_id": product_id, "as_of": day.isoformat(), "available_kg": str(sum((q for _, q in rows), Decimal(0))), "lots": [{"lot_id": lot.id, "expires_on": lot.expires_on.isoformat(), "available_kg": str(q)} for lot, q in rows], "revision": data["revision"]}

    def reserve(self, product_id: str, quantity_kg: Any, required_on: date, *, idempotency_key: str,
                expected_revision: int | None = None, now: datetime | None = None) -> StockReservation:
        qty, key, timestamp = _decimal(quantity_kg, "quantity_kg", positive=True), _text(idempotency_key, "idempotency_key"), _aware(now or datetime.now(timezone.utc), "now")
        def action(data):
            old = next((StockReservation.from_dict(x) for x in data["reservations"] if x["idempotency_key"] == key), None)
            if old:
                if old.product_id == product_id and old.quantity_kg == qty and old.required_on == required_on: return old, False
                raise ChemicalConflictError("idempotency key already used with different payload")
            remaining, allocations = qty, []
            for lot, available in self._balances(data, product_id, required_on):
                take = min(remaining, available)
                if take: allocations.append((lot.id, take)); remaining -= take
                if remaining == 0: break
            if remaining: raise ChemicalConflictError("insufficient non-expired FEFO stock")
            item = StockReservation(str(uuid4()), key, product_id, qty, timestamp, required_on, tuple(allocations))
            data["reservations"].append(item.to_dict()); return item, True
        return self._mutate(expected_revision, action)

    def consume(self, product_id: str, quantity_kg: Any, occurred_at: datetime, *, idempotency_key: str,
                reference: str, expected_revision: int | None = None) -> list[StockTransaction]:
        qty, key, when = _decimal(quantity_kg, "quantity_kg", positive=True), _text(idempotency_key, "idempotency_key"), _aware(occurred_at, "occurred_at")
        def action(data):
            olds = [StockTransaction.from_dict(x) for x in data["transactions"] if x["idempotency_key"].startswith(key + ":")]
            if olds:
                if sum((x.quantity_kg for x in olds), Decimal(0)) == qty and all(x.product_id == product_id for x in olds): return olds, False
                raise ChemicalConflictError("idempotency key already used with different payload")
            rows, remaining = [], qty
            for lot, available in self._balances(data, product_id, when.date()):
                take = min(remaining, available)
                if take:
                    tx = StockTransaction(str(uuid4()), f"{key}:{len(rows)}", product_id, lot.id, "consumption", take, when, reference)
                    data["transactions"].append(tx.to_dict()); rows.append(tx); remaining -= take
                if remaining == 0: break
            if remaining: raise ChemicalConflictError("insufficient non-expired FEFO stock")
            return rows, True
        return self._mutate(expected_revision, action)


def deterministic_consumption_forecast(history: Sequence[tuple[date, Any]], *, horizon_days: int,
                                       as_of: date) -> dict[str, Any]:
    if horizon_days < 1: raise ValueError("horizon_days must be positive")
    rows = sorted((day, _decimal(qty, "quantity_kg")) for day, qty in history if day <= as_of)
    if not rows: return {"status": "unavailable", "reason": "no consumption history", "as_of": as_of.isoformat(), "horizon_days": horizon_days, "daily_kg": None, "required_kg": None}
    first = rows[0][0]; elapsed = max(1, (as_of - first).days + 1); daily = sum((q for _, q in rows), Decimal(0)) / Decimal(elapsed)
    return {"status": "available", "method": "calendar-day arithmetic mean", "as_of": as_of.isoformat(), "horizon_days": horizon_days, "history_start": first.isoformat(), "daily_kg": str(daily), "required_kg": str(daily * horizon_days)}


def shortage_report(available_kg: Any, daily_consumption_kg: Any, *, lead_time_days: int,
                    safety_stock_days: int, as_of: date) -> dict[str, Any]:
    if lead_time_days < 0 or safety_stock_days < 0: raise ValueError("lead_time_days and safety_stock_days must be non-negative")
    available, daily = _decimal(available_kg, "available_kg"), _decimal(daily_consumption_kg, "daily_consumption_kg")
    required = daily * Decimal(lead_time_days + safety_stock_days); shortage = max(Decimal(0), required - available)
    days_cover = None if daily == 0 else available / daily
    return {"as_of": as_of.isoformat(), "lead_time_days": lead_time_days, "safety_stock_days": safety_stock_days, "available_kg": str(available), "required_kg": str(required), "shortage_kg": str(shortage), "risk": shortage > 0, "days_cover": str(days_cover) if days_cover is not None else None}
