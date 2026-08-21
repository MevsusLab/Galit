"""Generate reproducible GALIT release evidence in JSON and Markdown."""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from galit import __version__
from galit.demo_scenarios import DEMO_LABELS, run_competition_scenarios
from galit.economics import compute_effect, default_assumptions, scenario_bounds
from galit.evaluation import STRATEGIES
from galit.integrated import rank_wells
from galit.synthetic import make_fund

DEFAULT_SEED = 20260806
DEFAULT_WELL_COUNT = 40
RISK_THRESHOLD = 0.35


def run_command(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def collect_git(root: Path) -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], root)
    status = run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], root,
    )
    available = commit.returncode == 0 and status.returncode == 0
    lines = [line for line in status.stdout.splitlines() if line.strip()] if available else []
    return {
        "available": available,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(lines) if available else None,
    }


def parse_pytest_summary(output: str, returncode: int) -> dict[str, Any]:
    passed = re.search(r"(\d+) passed", output)
    failed = re.search(r"(\d+) failed", output)
    errors = re.search(r"(\d+) errors?", output)
    return {
        "command": f"{sys.executable} -m pytest -q",
        "result": "passed" if returncode == 0 else "failed",
        "exit_code": returncode,
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "errors": int(errors.group(1)) if errors else 0,
        "summary": next(
            (line.strip() for line in reversed(output.splitlines()) if " passed" in line or " failed" in line),
            "pytest summary unavailable",
        ),
    }


def collect_tests(root: Path) -> dict[str, Any]:
    completed = run_command([sys.executable, "-m", "pytest", "-q"], root)
    return parse_pytest_summary(completed.stdout, completed.returncode)


def collect_demo(seed: int = DEFAULT_SEED, well_count: int = DEFAULT_WELL_COUNT) -> dict[str, Any]:
    ranked = rank_wells(make_fund(well_count, seed=seed))
    dominant = Counter(result.dominant for result in ranked)
    high = sum(result.integrated_risk > RISK_THRESHOLD for result in ranked)
    top = ranked[0]
    assumptions = default_assumptions()
    economics = compute_effect(assumptions)
    envelope = scenario_bounds(assumptions)
    return {
        "dataset_label": "synthetic",
        "interpretation_label": "scenario illustration; not field data",
        "seed": seed,
        "well_count": well_count,
        "risk_threshold": RISK_THRESHOLD,
        "above_threshold_count": high,
        "dominant_mechanism_counts": dict(sorted(dominant.items())),
        "top_well": {
            "name": top.well,
            "integrated_risk": round(top.integrated_risk, 6),
            "dominant_mechanism": top.dominant,
        },
        "historical_economic_scenario_envelope": {
            "label": "scenario envelope; not forecast; not field validated; not KPI",
            "currency": "BYN/year",
            "low": round(envelope["консервативный"], 2),
            "midpoint": round(economics.total, 2),
            "high": round(envelope["оптимистичный"], 2),
        },
    }


def collect_competition_demo() -> dict[str, Any]:
    results = run_competition_scenarios()
    return {
        "labels": list(DEMO_LABELS),
        "calculation": "galit.integrated.diagnose",
        "scenario_count": len(results),
        "scenarios": [
            {
                "key": item.scenario.key,
                "educational_focus": item.scenario.educational_focus,
                "actual_dominant": item.actual_dominant,
                "integrated_risk": round(item.diagnosis.integrated_risk, 6),
                "has_co2_sensitivity": bool(item.co2_sensitivity),
                "has_counterfactual": item.counterfactual is not None,
            }
            for item in results
        ],
    }


def build_manifest(root: Path, tests: dict[str, Any], generated_at: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "project": "GALIT",
        "version": __version__,
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "git": collect_git(root),
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation()},
        "software_verification": tests,
        "demo": collect_demo(),
        "competition_demo": collect_competition_demo(),
        "pilot_unit_economics": {
            "model": "explicit customer-input break-even",
            "required_inputs": [
                "pilot_cost", "treatment_value", "failure_value",
                "downtime_day_value", "saved_tonne_value",
            ],
            "outputs": "separate channel thresholds and mixed break-even",
            "customer_checklist": "reports/customer-input-checklist.md",
            "currency_claim": "none until customer-approved unit values are supplied",
        },
        "pilot_evidence": {
            "mode": "shadow; no automatic control",
            "strategies": list(STRATEGIES),
            "primary_metric": "NDCG@K on untouched holdout",
            "outcome_evaluation": "blocked until real event_outcome targets are supplied",
            "data_contract": "reports/pilot-data-contract.md",
            "pre_registered_protocol": "reports/pre-registered-pilot-protocol.md",
            "leakage_controls": "disjoint wells and strict train < calibration < holdout chronology",
        },
        "api_integration_prototype": {
            "positioning": "integration prototype; not production-ready",
            "contract": "versioned single and bounded bulk diagnosis",
            "authentication": "not implemented; roadmap",
            "container": "non-root with healthcheck and restricted build context",
            "postman_collection": "postman/collections/GALIT API",
        },
        "evidence_labels": {
            "demo_data": "synthetic",
            "economic_output": "scenario",
            "software_verification": tests["result"],
            "model_validation": "not validated on independent field data",
            "allowed_use": "screening and decision-support only",
        },
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    git = manifest["git"]
    tests = manifest["software_verification"]
    demo = manifest["demo"]
    counts = ", ".join(f"{key}={value}" for key, value in demo["dominant_mechanism_counts"].items())
    envelope = demo["historical_economic_scenario_envelope"]
    return f"""# GALIT release evidence

- Version: `{manifest['version']}`
- Generated (UTC): `{manifest['generated_at_utc']}`
- Git commit: `{git['commit'] or 'unavailable'}`
- Git dirty: `{git['dirty']}`
- Python: `{manifest['runtime']['python']}` ({manifest['runtime']['implementation']})

## Software verification

- Result: **{tests['result']}**
- Pytest: `{tests['summary']}`
- Passed: {tests['passed']}; failed: {tests['failed']}; errors: {tests['errors']}

This is software verification, not model validation on independent field data.

## Reproducible demo scenario

- Data label: **synthetic**
- Interpretation: **scenario illustration; not field data**
- Seed: `{demo['seed']}`; wells: {demo['well_count']}
- Risk > {demo['risk_threshold']}: **{demo['above_threshold_count']} / {demo['well_count']}**
- Dominant mechanisms: {counts}
- Top well: {demo['top_well']['name']} (risk {demo['top_well']['integrated_risk']}, {demo['top_well']['dominant_mechanism']})
- Historical economic envelope: {envelope['low']:,.2f}–{envelope['high']:,.2f} BYN/year; midpoint {envelope['midpoint']:,.2f} (**scenario envelope; not forecast; not field validated; not KPI**)

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
"""


def write_reports(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "release-manifest.json"
    md_path = reports / "release-manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="record tests as not run")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    tests = ({"command": None, "result": "not_run", "exit_code": None, "passed": 0,
              "failed": 0, "errors": 0, "summary": "tests were explicitly skipped"}
             if args.skip_tests else collect_tests(root))
    manifest = build_manifest(root, tests)
    json_path, md_path = write_reports(root, manifest)
    print(f"Wrote {json_path.relative_to(root)} and {md_path.relative_to(root)}")
    print(tests["summary"])
    return 0 if tests["result"] in {"passed", "not_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
