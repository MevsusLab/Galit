"""Honest baseline comparison and measurable shadow-pilot API."""
from .pilot import (
    PILOT_COLUMNS,
    PILOT_LABELS,
    STRATEGIES,
    compare_strategies,
    evaluate_uploaded_rows,
    pilot_contract_frame,
)

__all__ = [
    "PILOT_COLUMNS", "PILOT_LABELS", "STRATEGIES", "compare_strategies",
    "evaluate_uploaded_rows", "pilot_contract_frame",
]
