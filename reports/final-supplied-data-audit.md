# Supplied GALIT calibration data audit

## Source preservation

- Raw file: `data\raw\galit-supplied-pasted.csv`
- SHA-256: `f9e1cbe47eaa1467153072984abca085dc8650b42d65c1b39469edd3a933c568`
- Bytes: 25251

## Inventory and completeness

- Rows: **500**
- Distinct non-empty wells: **49**
- Columns: `well_id, date, depth_m, flow_rate_m3h, ph, iron_mg_l, hardness_meq_l, tds_mg_l`

| Column | Missing | Missing % |
|---|---:|---:|
| `well_id` | 0 | 0.0% |
| `date` | 0 | 0.0% |
| `depth_m` | 0 | 0.0% |
| `flow_rate_m3h` | 22 | 4.4% |
| `ph` | 22 | 4.4% |
| `iron_mg_l` | 21 | 4.2% |
| `hardness_meq_l` | 22 | 4.4% |
| `tds_mg_l` | 0 | 0.0% |

## Duplicates and dates

- Exact duplicate rows: **0**
- Duplicate `(well_id, date)` records beyond the first: **2**
- Parsed date range: **2023-01-02** to **2026-07-21**
- Invalid dates: **0**
- Dates after audit date 2026-08-21: **0**

## Numeric ranges and IQR outliers

| Column | Count | Min | Median | Max | IQR outliers |
|---|---:|---:|---:|---:|---:|
| `well_id` | 500 | 101 | 124 | 149 | 0 |
| `depth_m` | 500 | 30.6 | 92.85 | 149.8 | 0 |
| `flow_rate_m3h` | 478 | 0.67 | 5.005 | 9.45 | 3 |
| `ph` | 478 | 5.76 | 7.2 | 8.57 | 5 |
| `iron_mg_l` | 479 | 0 | 0.56 | 4.9 | 24 |
| `hardness_meq_l` | 478 | 4.145 | 10.63 | 16.745 | 0 |
| `tds_mg_l` | 500 | 121.75 | 716.5 | 1239 | 3 |

IQR outliers are screening flags, not automatic errors.

## Units plausibility

- `depth_m`: plausible positive depth values.
- `flow_rate_m3h`: plausible positive magnitudes for the stated m3/h unit; not a canonical GALIT oil/water split.
- `ph`: within the physical pH scale 0-14.
- `iron_mg_l`: non-negative for stated mg/L; iron is not one of the seven required charge-balance ions.
- `hardness_meq_l`: non-negative for stated meq/L; aggregate hardness cannot recover separate Ca and Mg concentrations.
- `tds_mg_l`: non-negative for stated mg/L; TDS cannot be safely treated as salinity_ppm without an agreed conversion/definition.

## Ionic/chemical charge balance

**Calculable: no.** Complete individual cation/anion concentrations are required; pH, iron, aggregate hardness, and TDS are insufficient.
Required ion columns: `na_mg_l, k_mg_l, ca_mg_l, mg_mg_l, cl_mg_l, hco3_mg_l, so4_mg_l`.
Missing: `na_mg_l, k_mg_l, ca_mg_l, mg_mg_l, cl_mg_l, hco3_mg_l, so4_mg_l`.

## Calibration compatibility decision

- GALIT physical calibration: **blocked**
- GALIT risk-policy calibration: **blocked**
- Independent unseen-well/time evaluation: **blocked** (there is no valid canonical model input plus measured target pair to score).

No mapping, target, ion, calibration parameter, or accuracy metric was fabricated. The 49 wells could support a leakage-safe group/time holdout only after compatible inputs and measured targets are supplied.

### Exact blockers

- canonical snapshot inputs absent (38/41): schema_version, timestamp, source, quality, tubing_id_m, inclination_deg, roughness_m, q_oil_m3d, q_water_m3d, gor_m3m3, gamma_oil, gamma_gas, salinity_ppm, surface_tension_n_m, t_surface_c, geothermal_grad_k_m, k_earth_w_mk, alpha_earth_m2_s, u_to_w_m2k, r_to_m, r_wb_m, cp_fluid_j_kgk, production_days, na_mg_l, cl_mg_l, ca_mg_l, mg_mg_l, k_mg_l, hco3_mg_l, so4_mg_l, water_t_c, water_p_pa, wat_stock_tank_c, wax_content_pct, co2_mol_frac, inhibitor_efficiency, lift_type, p_wellhead_pa.
- date is not canonical timezone-aware timestamp.
- flow_rate_m3h does not identify q_oil_m3d and q_water_m3d and no conversion/split is supplied.
- tds_mg_l is not an explicit salinity_ppm measurement/mapping.
- physical calibration target target_temperature_c is absent.
- risk calibration target risk_label is absent.
- measurement_depth_m required to locate temperature observations is absent.

### Required data

Supply the canonical snapshot schema columns listed in `galit/calibration/schema.py`, including timezone-aware `timestamp`, source and quality, explicit geometry/rates/fluid/thermal values, all seven ions, water conditions, wax/CO2/inhibitor/lift/wellhead inputs, without substituting aggregates.
For physical `thermal.u_to` calibration and unseen evaluation, also supply measured `target_temperature_c` and `measurement_depth_m` on both training and held-out wells. For risk-policy calibration, supply an agreed `risk_label`. Other optional measured targets remain target-specific and must not be inferred.
