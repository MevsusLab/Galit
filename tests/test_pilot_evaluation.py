from __future__ import annotations
from datetime import datetime, timezone

import pytest

from galit.calibration.metrics import ranking_metrics
from galit.calibration.split import validate_pilot_split
from galit.evaluation import PILOT_LABELS, compare_strategies, evaluate_uploaded_rows


def rows():
    return [
        {"well_id":"A","timestamp":"2025-01-01T00:00:00+00:00","event_outcome":1,"calendar_score":.9,"independent_score":.2,"galit_score":.8},
        {"well_id":"B","timestamp":"2025-02-01T00:00:00+00:00","event_outcome":0,"calendar_score":.8,"independent_score":.9,"galit_score":.7},
        {"well_id":"C","timestamp":"2025-03-01T00:00:00+00:00","event_outcome":1,"calendar_score":.1,"independent_score":.8,"galit_score":.6},
    ]


def test_ranking_known_example_and_k_bounds():
    result = ranking_metrics([1, 0, 1], [.9, .8, .1], k=2)
    assert result["precision_at_k"] == .5
    assert result["recall_at_k"] == .5
    assert result["missed_events"] == 1
    assert result["unnecessary_interventions"] == 1
    assert ranking_metrics([1], [.2], k=99)["k"] == 1
    with pytest.raises(ValueError):
        ranking_metrics([1], [.2], k=0)


def test_tie_boundary_uses_expected_fraction_not_input_order():
    first = ranking_metrics([1, 0, 1], [.9, .5, .5], k=2)
    second = ranking_metrics([1, 1, 0], [.9, .5, .5], k=2)
    assert first["precision_at_k"] == second["precision_at_k"] == .75
    assert first["tie_policy"] == "expected value across tied boundary"


def test_no_positives_and_missing_targets():
    result = ranking_metrics([0, 0], [.9, .1], k=1)
    assert result["empty_positives"] and result["recall_at_k"] == 0
    assert not ranking_metrics([None, 1], [.9, .8], k=1)["available"]
    blocked = evaluate_uploaded_rows([{**rows()[0], "event_outcome": None}])
    assert blocked["status"] == "blocked" and "event_outcome" in blocked["reason"]


def test_three_strategy_comparison_and_illustrative_labels():
    result = compare_strategies(rows(), k=2)
    assert result["status"] == "evaluated" and len(result["strategies"]) == 3
    illustrative = compare_strategies(rows(), k=2, illustrative=True)
    assert illustrative["status"] == "illustrative"
    assert set(illustrative["labels"]) == set(PILOT_LABELS)
    assert "accuracy" in illustrative["disclaimer"]


def test_split_validation_rejects_well_and_time_leakage():
    train=[{"well_id":"A","timestamp":"2025-01-01T00:00:00+00:00"}]
    calibration=[{"well_id":"B","timestamp":"2025-02-01T00:00:00+00:00"}]
    holdout=[{"well_id":"C","timestamp":"2025-03-01T00:00:00+00:00"}]
    summary=validate_pilot_split(train,calibration,holdout)
    assert summary["valid"] and summary["chronology"] == "train < calibration < holdout"
    with pytest.raises(ValueError, match="well leakage"):
        validate_pilot_split(train,[{"well_id":"A","timestamp":"2025-02-01T00:00:00+00:00"}],holdout)
    with pytest.raises(ValueError, match="time leakage"):
        validate_pilot_split(train,[{"well_id":"B","timestamp":"2024-12-01T00:00:00+00:00"}],holdout)
