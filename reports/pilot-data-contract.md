# GALIT pilot data contract

**Purpose:** prospective, shadow-mode comparison. Scores are frozen at the decision timestamp; outcomes are joined only after the target horizon closes. Missing outcomes block evaluation.

| Field | Unit/type | Definition |
|---|---|---|
| `well_id` | text | Stable well identifier and leakage-control group. |
| `timestamp` | ISO-8601 UTC | Decision timestamp with timezone. |
| `calendar_score` | 0–1 | Pre-registered fixed-schedule priority (for example, normalized days overdue). |
| `independent_score` | 0–1 | Maximum independent mechanism score available before outcome. |
| `galit_score` | 0–1 | Frozen GALIT integrated score available before outcome. |
| `event_outcome` | 0/1 | Qualifying event observed inside the agreed forward target horizon. |
| `event_type` | controlled text | Agreed event taxonomy (plugging, corrosion failure, production-impacting deposition, etc.). |
| `target_horizon_days` | days | One customer-approved horizon, measured forward from `timestamp`. |
| `intervention` | 0/1 | Whether an intervention occurred inside the horizon. |
| `intervention_cost` | BYN | Fully loaded intervention cost; one agreed currency basis. |
| `downtime_hours` | h | Observed well downtime in the horizon. |
| `oil_loss_m3` | m³ | Observed oil loss in the horizon. |
| `oil_recovery_m3` | m³ | Incremental recovery only under an agreed counterfactual method. |

## Target definition to complete before data collection

- Qualifying event(s): **CUSTOMER TO DEFINE**
- Target horizon: **CUSTOMER TO DEFINE** days
- Outcome adjudication source and owner: **CUSTOMER TO DEFINE**
- Calendar schedule rule and normalization: **CUSTOMER TO DEFINE**
- Independent-mechanism thresholds: **CUSTOMER TO DEFINE**
- Currency/base date and oil-value source: **CUSTOMER TO DEFINE**

Do not backfill scores using post-event information. A well may belong to only one train/calibration/holdout partition, and the chronology must be strictly `train < calibration < holdout`.
