"""Baseline comparison for a leakage-safe, shadow-mode GALIT pilot.

The module never manufactures target evidence. Field evaluation requires an
explicit event outcome for every evaluated row. Synthetic scenarios can only
be reported as illustrative and not field validated.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from galit.calibration.metrics import ranking_metrics

PILOT_LABELS = ("synthetic", "illustrative", "not field validated")
STRATEGIES = (
    "calendar_fixed_schedule",
    "independent_mechanism_threshold",
    "galit_integrated_ranking",
)
PILOT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("well_id", "text", "Stable well identifier; grouping key for leakage control."),
    ("timestamp", "ISO-8601 UTC", "Decision timestamp, including timezone."),
    ("calendar_score", "0..1", "Pre-registered fixed-schedule priority, e.g. normalized days overdue."),
    ("independent_score", "0..1", "Maximum independent mechanism score, before integration."),
    ("galit_score", "0..1", "Frozen GALIT integrated ranking score available at decision time."),
    ("event_outcome", "0/1", "Target: qualifying event within the pre-registered target horizon."),
    ("event_type", "text", "Qualifying event definition/category."),
    ("target_horizon_days", "days", "Fixed forward horizon from decision timestamp."),
    ("intervention", "0/1", "Whether an intervention occurred during the horizon."),
    ("intervention_cost", "BYN", "Fully loaded intervention cost; currency must remain fixed."),
    ("downtime_hours", "hours", "Observed downtime in the target horizon."),
    ("oil_loss_m3", "m3", "Observed oil loss in the target horizon."),
    ("oil_recovery_m3", "m3", "Observed incremental recovery; requires agreed counterfactual."),
)


def _missing(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    return not rows or any(row.get(field) in (None, "") for row in rows)


def _float_column(rows: Sequence[Mapping[str, Any]], field: str) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        value = row.get(field)
        values.append(None if value in (None, "") else float(value))
    return values


def compare_strategies(
    rows: Sequence[Mapping[str, Any]], *, k: int = 5,
    assumptions: Mapping[str, float] | None = None,
    illustrative: bool = False,
    split_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare three frozen strategies against explicitly supplied outcomes.

    ``assumptions`` is optional. Business value is reported only when the
    caller explicitly supplies ``event_prevention_fraction`` and any required
    unit values. Ranking metrics are observational; prevented loss is a
    scenario calculation, not a causal estimate.
    """
    labels = list(PILOT_LABELS) if illustrative else ["field outcomes supplied"]
    base: dict[str, Any] = {
        "status": "blocked", "labels": labels, "n": len(rows),
        "assumptions": dict(assumptions or {}), "split_summary": dict(split_summary or {}),
        "strategies": [],
    }
    if _missing(rows, "event_outcome"):
        base["reason"] = (
            "Missing event_outcome. Supply a binary outcome observed within the "
            "pre-registered target_horizon_days; scores alone cannot validate performance."
        )
        return base
    missing_scores = [name for name in ("calendar_score", "independent_score", "galit_score")
                      if _missing(rows, name)]
    if missing_scores:
        base["reason"] = "Missing pre-decision strategy score columns: " + ", ".join(missing_scores)
        return base

    outcomes = _float_column(rows, "event_outcome")
    if any(value not in (0.0, 1.0) for value in outcomes):
        base["reason"] = "event_outcome must contain only 0/1 values"
        return base

    strategy_fields = dict(zip(STRATEGIES, ("calendar_score", "independent_score", "galit_score")))
    results = []
    for strategy, field in strategy_fields.items():
        metrics = ranking_metrics(outcomes, _float_column(rows, field), k=k)
        metrics["strategy"] = strategy
        results.append(metrics)
    base.update(status="illustrative" if illustrative else "evaluated", strategies=results)
    base["comparison_name"] = "illustrative comparison" if illustrative else "holdout outcome comparison"
    if illustrative:
        base["disclaimer"] = "SYNTHETIC / ILLUSTRATIVE / NOT FIELD VALIDATED; not accuracy."

    if assumptions:
        required = {"event_prevention_fraction", "oil_value_per_m3", "downtime_cost_per_hour"}
        if required <= set(assumptions) and not _missing(rows, "oil_loss_m3") and not _missing(rows, "downtime_hours"):
            effectiveness = float(assumptions["event_prevention_fraction"])
            if not 0 <= effectiveness <= 1:
                raise ValueError("event_prevention_fraction must be between 0 and 1")
            for result, (_, field) in zip(results, strategy_fields.items()):
                selected = _tie_selection_weights(_float_column(rows, field), result["k"])
                prevented = effectiveness * sum(
                    weight * float(row["event_outcome"]) * (
                        float(row["oil_loss_m3"]) * float(assumptions["oil_value_per_m3"])
                        + float(row["downtime_hours"]) * float(assumptions["downtime_cost_per_hour"])
                    ) for weight, row in zip(selected, rows)
                )
                intervention_cost = sum(
                    weight * float(row.get("intervention_cost") or 0.0)
                    for weight, row in zip(selected, rows)
                )
                result.update(prevented_loss=prevented, intervention_cost=intervention_cost,
                              net_value=prevented-intervention_cost,
                              business_value_label="assumption-based scenario; not causal proof")
        else:
            base["business_metrics_reason"] = (
                "Prevented loss/net value blocked: explicitly provide event_prevention_fraction, "
                "oil_value_per_m3, downtime_cost_per_hour and complete oil_loss_m3/downtime_hours."
            )
    return base


def _tie_selection_weights(scores: Sequence[float | None], k: int) -> list[float]:
    """Expected selection weights when the K boundary cuts a score tie."""
    if not scores:
        return []
    kk = min(max(int(k), 0), len(scores))
    indexed = sorted(enumerate(float(value) for value in scores), key=lambda item: item[1], reverse=True)
    weights = [0.0] * len(scores)
    used = 0
    start = 0
    while start < len(indexed) and used < kk:
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        take = min(kk-used, end-start)
        fraction = take / (end-start)
        for index, _ in indexed[start:end]:
            weights[index] = fraction
        used += take
        start = end
    return weights


def evaluate_uploaded_rows(rows: Sequence[Mapping[str, Any]], *, k: int = 5,
                           assumptions: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Field-upload entry point. It is blocked unless real targets are present."""
    return compare_strategies(rows, k=k, assumptions=assumptions, illustrative=False)


def pilot_contract_frame() -> list[dict[str, str]]:
    return [{"field": field, "unit/type": unit, "definition": definition}
            for field, unit, definition in PILOT_COLUMNS]
