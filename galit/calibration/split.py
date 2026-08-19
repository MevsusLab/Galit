"""Leakage-safe deterministic train/test splitting."""
from __future__ import annotations
import hashlib
from .schema import WellSnapshot


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
