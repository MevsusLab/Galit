# GALIT — decision support for the complicated well stock

## Problem

Four mechanisms — halite, calcite, wax and CO₂ corrosion — compete for the same intervention, downtime and budget. Separate calculations do not provide a comparable queue and may miss technology conflicts.

## What GALIT does now

A Python decision-support prototype calculates four screening mechanisms, ranks wells under a versioned risk policy, explains contributions and warnings, and supports what-if analysis. The trust panel exposes input provenance, applicability limits, scenario intervals and calibration status. Five synthetic archetypes provide a live teaching demo. Single and bounded bulk API contracts are available.

**It is not automatic control, not a field-validated predictor and not merely a calculator:** the product claim is a reproducible, auditable way to decide which complicated wells require attention first and why.

## Live demo

1. Run five labelled archetypes: halite, calcite, wax, corrosion and mixed conflict.
2. Show the calculated dominant mechanism, T/WAT crossing, CO₂ sensitivity and intervention conflict.
3. Open the trust gate: synthetic label, model warnings, provenance, baseline weights and blocked field-validation status.

## Evidence and trust gate

Software verification status and current test count are generated in [`release-manifest.md`](release-manifest.md); passing tests do not establish field accuracy. Runtime calibration safely supports `thermal.u_to` and four normalized mechanism weights only. Baseline evaluation compares calendar, independent-threshold and GALIT ranking on a leakage-safe holdout.

## Shadow pilot

20–30 wells, no automatic control and no GALIT recommendation executed. Scores are timestamped prospectively; outcomes are opened once after the horizon. Primary metric: pre-registered NDCG@K, with Precision@K, Recall@K, missed events and unnecessary interventions.

## Unit economics

Pilot cost and customer-approved value of one treatment, one failure, one downtime day and one saved tonne are explicit inputs with source/confidence. Break-even is reported separately per channel and for a mixed scenario. **No BYN benefit is claimed without customer inputs.** The historical 9.7–54.4 million BYN range (27.8 million midpoint) is retained only as an unvalidated scenario envelope, not a forecast or KPI.

## Ask / next step

Approve the shadow protocol and provide the checklist inputs: well history and provenance, interventions/outcomes, CO₂/WAT/water chemistry, pilot cost and finance-approved unit values. Then freeze targets, baseline, K and split before examining holdout outcomes.

Customer checklist: [`customer-input-checklist.md`](customer-input-checklist.md).
