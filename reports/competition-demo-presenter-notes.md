# GALIT competition demo — presenter notes (3–5 minutes)

> **SYNTHETIC · ILLUSTRATIVE · NOT FIELD VALIDATED**
>
> These scenarios demonstrate calculation paths and decision-support behavior. They do not promise diagnostic accuracy and are not customer or field observations.

## 0:00–0:40 — establish the evidence boundary

Run `python demo.py --competition`. Point to the large labels first. Explain that the original seed-based unbiased synthetic fund remains in the same run; the five archetypes are a separate teaching mode. Every number shown is recalculated through `diagnose`, not stored as an expected answer.

## 0:40–1:40 — mechanisms with interpretable physics

Use the halite, calcite and wax rows. For halite, mention the explicit screening limitation and need for a full Pitzer model. For calcite, point to the Stiff–Davis teaching input. For wax, point to the calculated T/WAT crossing and onset depth. Do not describe any of these as validated predictions.

## 1:40–2:40 — corrosion sensitivity

Use the corrosion row. Read the CO2 sweep from low to high and show that the model response is monotonic for this fixed case. Say explicitly that CO2 is an assumed input; the sweep shows input sensitivity, not uncertainty calibration or accuracy.

## 2:40–3:40 — mixed conflict and counterfactual

Use the mixed row. Contrast **educational focus: mixed** with **actual dominant: corrosion** (or whatever the core reports). Point out that calcite and corrosion can both be material, so a single-technology story is unsafe. Show the before/after inhibitor counterfactual. Clarify that 90% efficiency is an illustrative assumption and that the result supports a hypothesis to test, not a guaranteed field outcome.

## 3:40–4:30 — close honestly

State the product claim narrowly: GALIT combines four screening calculations, exposes applicability warnings, compares mechanisms under one risk policy, and supports reproducible what-if analysis. Next evidence required is independent field data, measured CO2/inhibitor performance, laboratory water/oil properties, and prospective validation.
