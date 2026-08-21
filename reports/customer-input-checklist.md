# GALIT — customer input checklist for pilot economics and validation

> Do not substitute missing facts with public averages or zeros. Record owner, period, currency, tax treatment, source system/document and confidence for every value.

## A. Pilot boundary and cost

- [ ] Pilot scope, wells, start/end dates and shadow-only mode.
- [ ] Total pilot cost and currency: internal labour, integration, data preparation, infrastructure, licences, support and contingency; identify sunk costs.
- [ ] Current intervention capacity `K` and decision cadence.
- [ ] Baseline process frozen before scoring: calendar/fixed schedule and independent-mechanism threshold.

## B. Unit economics — customer-approved values

For each value provide central/low/high, source, approval owner and whether channels overlap.

- [ ] One treatment: reagent, crew/equipment, logistics, laboratory and direct downtime separately.
- [ ] One qualifying failure: repair/TKRS, equipment, logistics and direct downtime separately.
- [ ] One day of downtime: net contribution margin or another approved value — **not public oil price by default**.
- [ ] One saved tonne: approved net value — **not gross revenue unless finance approves it**.
- [ ] Rules preventing double counting between failure, downtime and saved-tonne channels.
- [ ] Taxes, transfer prices, discounting and reporting currency.

Break-even is reported separately as treatments-only, failures-only, downtime-days-only and saved-tonnes-only, plus a mixed scenario. No BYN result is published until these inputs are approved.

## C. Model inputs and provenance

- [ ] Well geometry, NKT, lift type and operating regime.
- [ ] Timestamped oil/water/gas rates and pressure/temperature measurements.
- [ ] Full water analysis with sample T/P and pH; laboratory WAT/wax.
- [ ] CO₂ and inhibitor history/efficiency; corrosion measurements.
- [ ] Source and quality for every critical field (`measured`, `derived`, `default`, `synthetic`).

## D. Outcomes and intervention history

- [ ] Timestamped treatments, reason, technology, dosage, duration and measured before/after outcome.
- [ ] Failures with agreed taxonomy, cause, repair dates and downtime.
- [ ] Complication events/onset intervals and adjudication rules.
- [ ] Missingness/exclusion reasons; no target-derived features.

## E. Pre-registered validation targets

Approve before opening holdout outcomes:

- [ ] Train/calibration/holdout dates and disjoint well lists.
- [ ] Primary metric NDCG@K and K.
- [ ] Required improvement versus calendar baseline.
- [ ] Missed-event non-inferiority limit.
- [ ] Maximum unnecessary interventions.
- [ ] Minimum data/target completeness.
- [ ] Statistical uncertainty method and tie handling.
- [ ] Named outcome adjudicators and sign-off route.

Protocol: [`pre-registered-pilot-protocol.md`](pre-registered-pilot-protocol.md). Data contract: [`pilot-data-contract.md`](pilot-data-contract.md).
