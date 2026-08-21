# GALIT pitch — 3–5 minute script and slide outline

## Slide 1 — Problem (0:00–0:40)

“GALIT is decision support for the complicated well stock. Halite, calcite, wax and CO₂ corrosion are usually assessed separately, but they compete for one intervention and can create technology conflicts. The decision is not ‘what is the risk number?’; it is ‘which well needs attention first, why, and what evidence limits that recommendation?’”

## Slide 2 — Solution and live demo (0:40–1:50)

Run `python demo.py --competition`. Show five clearly labelled synthetic archetypes. Point to the recalculated dominant mechanism, wax onset, CO₂ sensitivity and mixed-conflict counterfactual. “These are calculation paths, not accuracy claims. Single and bulk API contracts expose the same core for integration.”

## Slide 3 — Trust gate (1:50–2:35)

Show provenance/applicability warnings, scenario intervals and calibration status. “Production mode blocks default or synthetic critical inputs. Baseline weights are expert assumptions. Runtime calibration accepts only `thermal.u_to` and four normalized weights. The release manifest separates software verification from field validation.”

## Slide 4 — Shadow pilot and baseline (2:35–3:25)

“On 20–30 wells GALIT runs in shadow: no automatic control and no recommendation is executed. Before outcomes are opened we freeze calendar and independent-threshold comparators, K, split and acceptance criteria. We score NDCG@K, missed events and unnecessary interventions on an untouched holdout.”

## Slide 5 — Unit economics and next step (3:25–4:30)

“Economics starts with your inputs: total pilot cost and approved value of one treatment, one failure, one downtime day and one saved tonne. We show how many of each are required for break-even separately and in a mixed scenario, with sensitivity and no double counting. We do not present a BYN forecast before finance-approved inputs.”

Close: “The next step is not autonomous deployment. It is approval of the shadow protocol and the customer input checklist, followed by a frozen prospective comparison. If GALIT does not beat the baseline within agreed safety limits, it does not pass the gate.”

## Backup / disclosure slide

- Synthetic/illustrative/not field validated.
- Four screening mechanisms; high-salinity carbonate calculations remain outside simple-index calibration.
- No detailed public competition criteria were found; confirmed dates: submission through 31.08.2026, review through 30.09.2026, final date announced after 01.10.2026.
- Historical 9.7–54.4 million BYN envelope and 27.8 million midpoint: scenario only, not forecast, not field validated, not headline KPI.
