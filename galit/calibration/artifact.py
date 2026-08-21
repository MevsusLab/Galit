"""Strict, versioned GALIT calibration artifact and runtime adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

ARTIFACT_SCHEMA_VERSION = "2.0"
MODEL_VERSION = "galit-0.1.0"
MODEL_COMPATIBILITY = "galit-0.1.x"
VALIDATION_STATUSES = {
    "baseline", "calibrated-not-field-validated", "holdout-validated", "blocked"
}
SUPPORTED_PARAMETERS = {"thermal.u_to", "weight.halite", "weight.calcite", "weight.wax", "weight.corrosion"}
MECHANISMS = ("halite", "calcite", "wax", "corrosion")


class ArtifactValidationError(ValueError):
    """Artifact is malformed, unsafe, or incompatible with this runtime."""


@dataclass(frozen=True)
class ParameterSet:
    # First fields preserve the legacy Python constructor; persisted v1 JSON is rejected.
    kind: str
    parameters: Mapping[str, float]
    model_version: str
    created_at: str
    dataset_hash: str
    train_wells: tuple[str, ...]
    test_wells: tuple[str, ...]
    metrics: Mapping[str, Any]
    synthetic: bool = False
    artifact_id: str = "legacy-in-memory"
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    model_compatibility: str = MODEL_COMPATIBILITY
    split: Mapping[str, Any] = field(default_factory=dict)
    risk_policy: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    validation_status: str = "calibrated-not-field-validated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "split", MappingProxyType(dict(self.split)))
        object.__setattr__(self, "risk_policy", MappingProxyType(dict(self.risk_policy)))
        object.__setattr__(self, "train_wells", tuple(self.train_wells))
        object.__setattr__(self, "test_wells", tuple(self.test_wells))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        self.validate()

    def validate(self) -> "ParameterSet":
        errors: list[str] = []
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            errors.append(f"unsupported schema_version {self.schema_version!r}; expected {ARTIFACT_SCHEMA_VERSION!r}")
        if self.artifact_id != "legacy-in-memory" and (
            self.model_version != MODEL_VERSION or self.model_compatibility != MODEL_COMPATIBILITY
        ):
            errors.append(
                f"incompatible model version/compatibility: {self.model_version!r}/{self.model_compatibility!r}; "
                f"runtime is {MODEL_VERSION!r}/{MODEL_COMPATIBILITY!r}"
            )
        if self.kind not in {"physical", "risk-policy", "blocked"}:
            errors.append("kind must be physical, risk-policy, or blocked")
        if self.validation_status not in VALIDATION_STATUSES:
            errors.append(f"unknown validation_status {self.validation_status!r}")
        if not self.artifact_id or not self.dataset_hash:
            errors.append("artifact_id and dataset_hash must be non-empty")
        if self.artifact_id != "legacy-in-memory":
            try:
                created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    errors.append("created_at must include timezone")
            except (TypeError, ValueError):
                errors.append("created_at must be ISO-8601")
        unknown = sorted(set(self.parameters) - SUPPORTED_PARAMETERS)
        if unknown:
            errors.append("unknown/unsupported calibrated parameters: " + ", ".join(unknown))
        for name, value in self.parameters.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"parameter {name} must be finite")
        u_to = self.parameters.get("thermal.u_to")
        if u_to is not None and not 2.0 <= float(u_to) <= 80.0:
            errors.append("thermal.u_to must be within [2, 80]")
        if self.kind == "physical" and set(self.parameters) != {"thermal.u_to"}:
            errors.append("physical artifact must contain exactly thermal.u_to")
        policy = dict(self.risk_policy)
        if self.kind == "risk-policy":
            weights = policy.get("weights")
            if not isinstance(weights, dict) or set(weights) != set(MECHANISMS):
                errors.append("risk_policy.weights must contain exactly halite, calcite, wax, corrosion")
            else:
                values = list(weights.values())
                if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0 for v in values):
                    errors.append("risk weights must be finite and non-negative")
                elif not math.isclose(sum(float(v) for v in values), 1.0, abs_tol=1e-12):
                    errors.append("risk weights must sum to 1; artifacts are never normalized implicitly")
            if not policy.get("id") or not policy.get("version"):
                errors.append("risk_policy requires id and version")
        if self.validation_status == "holdout-validated" and not _holdout_available(self.metrics):
            errors.append("holdout-validated requires available holdout/test calibrated metrics")
        if self.validation_status == "blocked" and self.parameters:
            errors.append("blocked artifact cannot contain runtime parameters")
        if errors:
            raise ArtifactValidationError("Invalid calibration artifact:\n- " + "\n- ".join(errors))
        return self

    @property
    def production_ready(self) -> bool:
        return self.validation_status == "holdout-validated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "model": {"version": self.model_version, "compatibility": self.model_compatibility},
            "created_at": self.created_at,
            "data": {"fingerprint": self.dataset_hash, "split": dict(self.split),
                     "train_wells": list(self.train_wells), "holdout_wells": list(self.test_wells),
                     "synthetic": self.synthetic},
            "calibration": {"kind": self.kind, "parameters": dict(self.parameters)},
            "risk_policy": dict(self.risk_policy),
            "metrics": dict(self.metrics),
            "limitations": list(self.limitations),
            "labels": {"validation_status": self.validation_status,
                       "production_ready": self.production_ready},
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ParameterSet":
        # v1 had a flat unversioned shape. Reject with an actionable migration error.
        if "schema_version" not in raw:
            raise ArtifactValidationError(
                "Legacy ParameterSet artifact has no schema_version. Re-run calibration_cli.py calibrate "
                "to migrate it to schema 2.0; unsafe implicit migration is not supported."
            )
        allowed = {"schema_version", "artifact_id", "model", "created_at", "data", "calibration",
                   "risk_policy", "metrics", "limitations", "labels"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ArtifactValidationError("Unknown artifact fields: " + ", ".join(unknown))
        try:
            model, data, calibration, labels = raw["model"], raw["data"], raw["calibration"], raw["labels"]
            return cls(
                artifact_id=raw["artifact_id"], kind=calibration["kind"],
                parameters=calibration.get("parameters", {}), model_version=model["version"],
                model_compatibility=model["compatibility"], created_at=raw["created_at"],
                dataset_hash=data["fingerprint"], train_wells=tuple(data.get("train_wells", [])),
                test_wells=tuple(data.get("holdout_wells", [])), metrics=raw.get("metrics", {}),
                synthetic=bool(data.get("synthetic", False)), schema_version=raw["schema_version"],
                split=data.get("split", {}), risk_policy=raw.get("risk_policy", {}),
                limitations=tuple(raw.get("limitations", [])),
                validation_status=labels["validation_status"],
            )
        except (KeyError, TypeError) as exc:
            raise ArtifactValidationError(f"Missing or invalid artifact field: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "ParameterSet":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"Cannot read calibration artifact: {exc}") from exc
        if not isinstance(raw, dict):
            raise ArtifactValidationError("Calibration artifact root must be a JSON object")
        return cls.from_dict(raw)

    def to_runtime(self):
        from galit.integrated import RiskPolicy, RuntimeCalibration
        if self.validation_status == "blocked":
            raise ArtifactValidationError("Blocked calibration artifact cannot be applied at runtime")
        policy = None
        if self.kind == "risk-policy":
            policy = RiskPolicy(policy_id=str(self.risk_policy["id"]),
                                version=str(self.risk_policy["version"]),
                                weights=dict(self.risk_policy["weights"]))
        return RuntimeCalibration(
            calibration_id=self.artifact_id,
            artifact_version=self.schema_version,
            validation_status=self.validation_status,
            thermal_u_to=self.parameters.get("thermal.u_to"),
            risk_policy=policy,
        )


def _reject_constant(value: str) -> None:
    raise ArtifactValidationError(f"Non-finite JSON value {value} is forbidden")


def _holdout_available(metrics: Mapping[str, Any]) -> bool:
    holdout = metrics.get("holdout", metrics.get("test", {}))
    if not isinstance(holdout, dict):
        return False
    calibrated = holdout.get("calibrated", {})
    return isinstance(calibrated, dict) and calibrated.get("available") is True and calibrated.get("n", 0) > 0


def validation_status_for(metrics: Mapping[str, Any], *, synthetic: bool) -> str:
    if synthetic:
        return "calibrated-not-field-validated"
    return "holdout-validated" if _holdout_available(metrics) else "calibrated-not-field-validated"


def blocked_parameter_set(dataset_hash: str, limitations: list[str], *, artifact_id: str = "blocked") -> ParameterSet:
    return ParameterSet(
        artifact_id=artifact_id, kind="blocked", parameters={}, model_version=MODEL_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(), dataset_hash=dataset_hash,
        train_wells=(), test_wells=(), metrics={"train": {}, "holdout": {}},
        limitations=tuple(limitations), validation_status="blocked",
    )
