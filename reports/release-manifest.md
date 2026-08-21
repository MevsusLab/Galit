# GALIT release evidence

- Version: `0.1.0`
- Generated (UTC): `2026-08-21T10:27:59.781182+00:00`
- Git commit: `e19767f6ce54927da8930f2485ab2a7a3d479dab`
- Git dirty: `True`
- Python: `3.12.10` (CPython)

## Software verification

- Result: **passed**
- Pytest: `178 passed, 1 warning in 9.34s`
- Passed: 178; failed: 0; errors: 0

This is software verification, not model validation on independent field data.

## Reproducible demo scenario

- Data label: **synthetic**
- Interpretation: **scenario illustration; not field data**
- Seed: `20260806`; wells: 40
- Risk > 0.35: **14 / 40**
- Dominant mechanisms: calcite=1, corrosion=39
- Top well: Золотухинское 185 (risk 0.526505, corrosion)
- Historical economic envelope: 9,695,249.36–54,423,847.99 BYN/year; midpoint 27,836,532.02 (**scenario envelope; not forecast; not field validated; not KPI**)

## Measurable shadow pilot

- Strategies: calendar/fixed schedule; independent-mechanism threshold; GALIT integrated ranking.
- Primary metric: NDCG@K on an untouched, leakage-safe holdout.
- Outcome evaluation: **blocked until real event outcomes are supplied**.
- Protocol: `reports/pre-registered-pilot-protocol.md`.
- Data contract: `reports/pilot-data-contract.md`.

## API integration prototype

- Versioned single and bounded bulk diagnosis contracts.
- Non-root container with healthcheck and restricted build context.
- Authentication/authorization: **not implemented; roadmap**.
- Positioning: **integration prototype; not production-ready**.

## Evidence labels

- `synthetic`: generated inputs, not customer/field data.
- `scenario`: illustrative output under stated assumptions.
- `model_validation`: **not validated on independent field data**.
- Allowed use: screening and decision-support only.
