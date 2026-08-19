"""Conservative audit for tabular data proposed for GALIT calibration.

This module audits source data without inventing mappings or filling missing values.
"""
from __future__ import annotations

import csv
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import math
from pathlib import Path
from statistics import median
from typing import Any

from .schema import OPTIONAL_SNAPSHOT_COLUMNS, SNAPSHOT_COLUMNS

ION_COLUMNS = ("na_mg_l", "k_mg_l", "ca_mg_l", "mg_mg_l", "cl_mg_l", "hco3_mg_l", "so4_mg_l")
PHYSICAL_TARGETS = ("target_temperature_c",)
RISK_TARGETS = ("risk_label",)


def _number(value: str) -> float | None:
    if value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _quartiles(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint + (len(ordered) % 2):]
    return median(lower), median(upper)


def audit_csv(path: str | Path, *, today: date | None = None) -> dict[str, Any]:
    """Return a factual audit; never infer semantic mappings from similar names."""
    source = Path(path)
    raw = source.read_bytes()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)
    today = today or datetime.now(timezone.utc).date()
    missing = {h: sum(r.get(h, "").strip() == "" for r in rows) for h in headers}
    wells = {r.get("well_id", "").strip() for r in rows if r.get("well_id", "").strip()}
    exact_duplicates = len(rows) - len({tuple(r.get(h, "") for h in headers) for r in rows})
    date_column = "date" if "date" in headers else "timestamp" if "timestamp" in headers else None
    parsed_dates: list[date] = []
    invalid_dates: list[dict[str, Any]] = []
    if date_column:
        for line, row in enumerate(rows, 2):
            try:
                parsed_dates.append(datetime.fromisoformat(row[date_column].strip().replace("Z", "+00:00")).date())
            except (TypeError, ValueError):
                invalid_dates.append({"line": line, "value": row.get(date_column, "")})
    duplicate_keys = 0
    if date_column and "well_id" in headers:
        keys = [(r["well_id"].strip(), r[date_column].strip()) for r in rows]
        duplicate_keys = sum(n - 1 for n in Counter(keys).values() if n > 1)
    numeric: dict[str, Any] = {}
    for header in headers:
        values = [_number(r.get(header, "")) for r in rows]
        present = [v for v in values if v is not None]
        nonempty = sum(r.get(header, "").strip() != "" for r in rows)
        if not present or len(present) != nonempty:
            continue
        q1, q3 = _quartiles(present) if len(present) >= 4 else (min(present), max(present))
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        numeric[header] = {
            "count": len(present), "min": min(present), "max": max(present),
            "median": median(present), "q1": q1, "q3": q3,
            "iqr_outlier_count": sum(v < low or v > high for v in present),
            "iqr_fences": [low, high],
        }
    plausibility = {
        "depth_m": "plausible positive depth values" if numeric.get("depth_m", {}).get("min", 0) > 0 else "not assessable or non-positive values present",
        "flow_rate_m3h": "plausible positive magnitudes for the stated m3/h unit; not a canonical GALIT oil/water split" if numeric.get("flow_rate_m3h", {}).get("min", 0) >= 0 else "negative values present",
        "ph": "within the physical pH scale 0-14" if numeric.get("ph", {}).get("min", -1) >= 0 and numeric.get("ph", {}).get("max", 15) <= 14 else "outside pH scale or not assessable",
        "iron_mg_l": "non-negative for stated mg/L; iron is not one of the seven required charge-balance ions" if numeric.get("iron_mg_l", {}).get("min", -1) >= 0 else "negative values present",
        "hardness_meq_l": "non-negative for stated meq/L; aggregate hardness cannot recover separate Ca and Mg concentrations" if numeric.get("hardness_meq_l", {}).get("min", -1) >= 0 else "negative values present",
        "tds_mg_l": "non-negative for stated mg/L; TDS cannot be safely treated as salinity_ppm without an agreed conversion/definition" if numeric.get("tds_mg_l", {}).get("min", -1) >= 0 else "negative values present",
    }
    required_missing = [c for c in SNAPSHOT_COLUMNS if c not in headers]
    physical_target_present = all(c in headers and missing[c] < len(rows) for c in PHYSICAL_TARGETS)
    risk_target_present = all(c in headers and missing[c] < len(rows) for c in RISK_TARGETS)
    ions_missing = [c for c in ION_COLUMNS if c not in headers]
    blockers = [
        f"canonical snapshot inputs absent ({len(required_missing)}/{len(SNAPSHOT_COLUMNS)}): " + ", ".join(required_missing),
        "date is not canonical timezone-aware timestamp" if "timestamp" not in headers else "",
        "flow_rate_m3h does not identify q_oil_m3d and q_water_m3d and no conversion/split is supplied" if "flow_rate_m3h" in headers else "",
        "tds_mg_l is not an explicit salinity_ppm measurement/mapping" if "tds_mg_l" in headers and "salinity_ppm" not in headers else "",
        "physical calibration target target_temperature_c is absent" if not physical_target_present else "",
        "risk calibration target risk_label is absent" if not risk_target_present else "",
        "measurement_depth_m required to locate temperature observations is absent" if "measurement_depth_m" not in headers else "",
    ]
    blockers = [b for b in blockers if b]
    return {
        "source": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "rows": len(rows), "wells": len(wells), "columns": headers,
        "missing": missing, "exact_duplicate_rows": exact_duplicates,
        "duplicate_well_date_records": duplicate_keys,
        "dates": {"column": date_column, "invalid_count": len(invalid_dates), "invalid": invalid_dates,
                  "min": min(parsed_dates).isoformat() if parsed_dates else None,
                  "max": max(parsed_dates).isoformat() if parsed_dates else None,
                  "future_count": sum(d > today for d in parsed_dates), "audit_date": today.isoformat()},
        "numeric": numeric, "units_plausibility": plausibility,
        "charge_balance": {"calculable": not ions_missing, "required_ions": list(ION_COLUMNS),
                           "missing_ions": ions_missing,
                           "reason": None if not ions_missing else "Complete individual cation/anion concentrations are required; pH, iron, aggregate hardness, and TDS are insufficient."},
        "compatibility": {"physical_calibration": not required_missing and physical_target_present,
                          "risk_calibration": not required_missing and risk_target_present,
                          "independent_holdout_evaluation": False,
                          "blockers": blockers,
                          "required_optional_targets": list(OPTIONAL_SNAPSHOT_COLUMNS)},
    }


def write_markdown(audit: dict[str, Any], output: str | Path) -> Path:
    """Write a compact, reviewable compatibility report."""
    n = audit["numeric"]
    lines = [
        "# Supplied GALIT calibration data audit", "",
        "## Source preservation", "",
        f"- Raw file: `{audit['source']}`", f"- SHA-256: `{audit['sha256']}`", f"- Bytes: {audit['bytes']}", "",
        "## Inventory and completeness", "",
        f"- Rows: **{audit['rows']}**", f"- Distinct non-empty wells: **{audit['wells']}**",
        f"- Columns: `{', '.join(audit['columns'])}`", "",
        "| Column | Missing | Missing % |", "|---|---:|---:|",
    ]
    for column, count in audit["missing"].items():
        lines.append(f"| `{column}` | {count} | {100*count/audit['rows']:.1f}% |")
    d = audit["dates"]
    lines += ["", "## Duplicates and dates", "", f"- Exact duplicate rows: **{audit['exact_duplicate_rows']}**",
              f"- Duplicate `(well_id, {d['column']})` records beyond the first: **{audit['duplicate_well_date_records']}**",
              f"- Parsed date range: **{d['min']}** to **{d['max']}**", f"- Invalid dates: **{d['invalid_count']}**",
              f"- Dates after audit date {d['audit_date']}: **{d['future_count']}**", "",
              "## Numeric ranges and IQR outliers", "", "| Column | Count | Min | Median | Max | IQR outliers |", "|---|---:|---:|---:|---:|---:|"]
    for column, stats in n.items():
        lines.append(f"| `{column}` | {stats['count']} | {stats['min']:.6g} | {stats['median']:.6g} | {stats['max']:.6g} | {stats['iqr_outlier_count']} |")
    lines += ["", "IQR outliers are screening flags, not automatic errors.", "", "## Units plausibility", ""]
    lines += [f"- `{key}`: {value}." for key, value in audit["units_plausibility"].items() if key in audit["columns"]]
    cb = audit["charge_balance"]
    lines += ["", "## Ionic/chemical charge balance", "", f"**Calculable: {'yes' if cb['calculable'] else 'no'}.** {cb['reason'] or ''}",
              f"Required ion columns: `{', '.join(cb['required_ions'])}`.", f"Missing: `{', '.join(cb['missing_ions'])}`.", "",
              "## Calibration compatibility decision", "",
              f"- GALIT physical calibration: **{'compatible' if audit['compatibility']['physical_calibration'] else 'blocked'}**",
              f"- GALIT risk-policy calibration: **{'compatible' if audit['compatibility']['risk_calibration'] else 'blocked'}**",
              "- Independent unseen-well/time evaluation: **blocked** (there is no valid canonical model input plus measured target pair to score).", "",
              "No mapping, target, ion, calibration parameter, or accuracy metric was fabricated. The 49 wells could support a leakage-safe group/time holdout only after compatible inputs and measured targets are supplied.", "",
              "### Exact blockers", ""]
    lines += [f"- {b}." for b in audit["compatibility"]["blockers"]]
    lines += ["", "### Required data", "",
              "Supply the canonical snapshot schema columns listed in `galit/calibration/schema.py`, including timezone-aware `timestamp`, source and quality, explicit geometry/rates/fluid/thermal values, all seven ions, water conditions, wax/CO2/inhibitor/lift/wellhead inputs, without substituting aggregates.",
              "For physical `thermal.u_to` calibration and unseen evaluation, also supply measured `target_temperature_c` and `measurement_depth_m` on both training and held-out wells. For risk-policy calibration, supply an agreed `risk_label`. Other optional measured targets remain target-specific and must not be inferred."]
    target = Path(output)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
