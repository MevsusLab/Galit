# GALIT pre-registered measurable pilot protocol

## Design

- Population: **20–30 wells**, eligibility and exclusions agreed before start.
- Mode: **shadow only**; no automatic control and no production recommendation executed by GALIT.
- Comparators: (1) calendar/fixed schedule, (2) independent-mechanism threshold, (3) frozen GALIT integrated ranking.
- Prospective scoring: all three scores timestamped before the outcome horizon; outcomes adjudicated after horizon closure.
- Primary metric: **NDCG@K on the untouched holdout**, with K fixed by the customer's intervention capacity before start.
- Secondary metrics: Precision@K, Recall@K, missed events, unnecessary interventions.
- Safety/operational metrics: total qualifying events, downtime hours, oil loss, intervention count, adverse/conflicting interventions and data completeness.
- Business scenario metrics: prevented loss and net value only with customer-approved unit values, costs and prevention assumptions; not treated as causal evidence.

## Acceptance criteria (pre-registration placeholders)

- Primary-metric improvement versus calendar baseline: **CUSTOMER TO SET**.
- Non-inferiority/safety limit for missed events: **CUSTOMER TO SET**.
- Maximum unnecessary interventions: **CUSTOMER TO SET**.
- Minimum target/data completeness: **CUSTOMER TO SET**.
- Statistical uncertainty/reporting method: **CUSTOMER TO APPROVE**.

No thresholds are inferred from synthetic demonstrations or selected after viewing holdout outcomes.

## Chronology and leakage control

1. **Train period:** historical inputs/outcomes; used for model development only.
2. **Calibration period:** later, disjoint wells; used to freeze policy, thresholds, K and assumptions.
3. **Holdout/shadow period:** latest, disjoint wells; opened once for the pre-registered comparison.

Validation rejects any well appearing in more than one partition, any timestamp overlap/order violation, missing timezone, target-derived score input, or missing holdout outcomes. Ties at K use the expected value across the tied boundary. All exclusions and missingness are reported.

## Reporting labels

Demo output must read **SYNTHETIC / ILLUSTRATIVE / NOT FIELD VALIDATED** and is never called accuracy. Supplied data without targets is reported as **BLOCKED**, with missing fields listed.
