"""Leakage-safe deterministic train/test splitting."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
from typing import Any, Mapping, Sequence
from .schema import WellSnapshot, parse_time


def split_by_well(snapshots: list[WellSnapshot], *, test_fraction: float = .2,
                  seed: int = 0) -> tuple[list[WellSnapshot], list[WellSnapshot]]:
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    wells = sorted({s.well_id for s in snapshots})
    if len(wells) < 2: raise ValueError("at least two wells are required for group split")
    ranked = sorted(wells, key=lambda w: hashlib.sha256(f"{seed}:{w}".encode()).digest())
    n_test = max(1, min(len(wells)-1, round(len(wells)*test_fraction)))
    test_wells = set(ranked[:n_test])
    train = [s for s in snapshots if s.well_id not in test_wells]
    test = [s for s in snapshots if s.well_id in test_wells]
    assert not ({s.well_id for s in train} & {s.well_id for s in test})
    return train, test


def temporal_group_holdout(snapshots: list[WellSnapshot], *, test_fraction: float=.2,
                           seed: int=0) -> tuple[list[WellSnapshot], list[WellSnapshot]]:
    """Hold out whole latest wells; preserves time ordering and zero well overlap."""
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    latest = {}
    for s in snapshots: latest[s.well_id] = max(latest.get(s.well_id, s.timestamp), s.timestamp)
    wells = sorted(latest, key=lambda w: (latest[w], hashlib.sha256(f"{seed}:{w}".encode()).digest()))
    if len(wells) < 2: raise ValueError("at least two wells are required for temporal holdout")
    n_test=max(1,min(len(wells)-1,round(len(wells)*test_fraction))); test_wells=set(wells[-n_test:])
    return ([s for s in snapshots if s.well_id not in test_wells],
            [s for s in snapshots if s.well_id in test_wells])


def _pilot_identity(row: Mapping[str, Any] | WellSnapshot) -> tuple[str, datetime]:
    if isinstance(row, WellSnapshot):
        return row.well_id, row.timestamp
    well = str(row.get("well_id", "")).strip()
    if not well:
        raise ValueError("pilot split row is missing well_id")
    return well, parse_time(row.get("timestamp"))


def validate_pilot_split(
    train: Sequence[Mapping[str, Any] | WellSnapshot],
    calibration: Sequence[Mapping[str, Any] | WellSnapshot],
    holdout: Sequence[Mapping[str, Any] | WellSnapshot],
) -> dict[str, Any]:
    """Reject grouped or chronological leakage across pilot partitions."""
    names = ("train", "calibration", "holdout")
    partitions = tuple(train_cal_hold for train_cal_hold in (train, calibration, holdout))
    if any(not part for part in partitions):
        raise ValueError("train, calibration and holdout must all be non-empty")
    identities = [[_pilot_identity(row) for row in part] for part in partitions]
    wells = [set(well for well, _ in part) for part in identities]
    for left in range(len(wells)):
        for right in range(left + 1, len(wells)):
            overlap = wells[left] & wells[right]
            if overlap:
                raise ValueError(
                    f"well leakage between {names[left]} and {names[right]}: "
                    + ", ".join(sorted(overlap))
                )
    bounds = [(min(ts for _, ts in part), max(ts for _, ts in part)) for part in identities]
    if not (bounds[0][1] < bounds[1][0] and bounds[1][1] < bounds[2][0]):
        raise ValueError("time leakage: require max(train) < min(calibration) and max(calibration) < min(holdout)")
    return {
        "valid": True,
        "chronology": "train < calibration < holdout",
        "partitions": {
            name: {"rows": len(part), "wells": len(group),
                   "start": bound[0].isoformat(), "end": bound[1].isoformat()}
            for name, part, group, bound in zip(names, partitions, wells, bounds)
        },
        "well_overlap": 0,
    }
