"""JSON and Markdown evaluation reports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .metrics import coverage, regression_metrics, classification_metrics, ranking_metrics
from .calibrator import ParameterSet, _temperature_prediction
from .schema import HistoryDataset, WellSnapshot

TARGETS=["target_temperature_c","target_pressure_pa","target_wax_onset_m","target_corrosion_mm_y","event_label","risk_label"]
def evaluate_parameter_set(params:ParameterSet,dataset:HistoryDataset,rows:list[WellSnapshot]|None=None)->dict[str,Any]:
    rows=rows if rows is not None else dataset.snapshots; raw=[s.values for s in rows]
    report={"model_version":params.model_version,"dataset_hash":dataset.dataset_hash,"synthetic":dataset.synthetic,
            "synthetic_disclaimer":"Synthetic smoke test only; not evidence of accuracy." if dataset.synthetic else None,
            "coverage":coverage(raw,TARGETS),"metrics":{},"warnings":list(dataset.warnings)}
    if len({s.well_id for s in rows})<10: report["warnings"].append("Small sample: fewer than 10 wells; metrics are unstable.")
    if params.kind=="physical" and "thermal.u_to" in params.parameters:
        ys=[float(s.values["target_temperature_c"]) if s.values.get("target_temperature_c") not in (None,"") else None for s in rows]
        ps=[_temperature_prediction(s,params.parameters["thermal.u_to"]) if y is not None else None for s,y in zip(rows,ys)]
        report["metrics"]["temperature"]=regression_metrics("temperature_c",ys,ps)
    else: report["metrics"]["temperature"]={"name":"temperature_c","available":False,"reason":"not evaluated by this parameter set","n":0}
    for key,name in (("target_pressure_pa","pressure"),("target_wax_onset_m","onset"),("target_corrosion_mm_y","corrosion")):
        report["metrics"][name]=regression_metrics(name,[r.get(key) for r in raw],[None]*len(raw))
    report["metrics"]["events"]=classification_metrics("events",[r.get("event_label") for r in raw],[None]*len(raw))
    report["metrics"]["risk_ranking"]=ranking_metrics([r.get("risk_label") for r in raw],[None]*len(raw))
    return report

def write_report(report:dict[str,Any],json_path:str|Path,markdown_path:str|Path)->None:
    Path(json_path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# GALIT calibration accuracy report","",f"- Model: `{report['model_version']}`",f"- Dataset: `{report['dataset_hash']}`"]
    if report.get("synthetic_disclaimer"): lines += ["",f"> **{report['synthetic_disclaimer']}**"]
    lines += ["","## Metrics","","| Metric | Available | n | Details |","|---|---:|---:|---|"]
    for name,m in report["metrics"].items():
        details=", ".join(f"{k}={v:.4g}" if isinstance(v,float) else f"{k}={v}" for k,v in m.items() if k not in {"name","available","n"})
        lines.append(f"| {name} | {m.get('available',False)} | {m.get('n',0)} | {details} |")
    lines += ["","## Warnings"]+[f"- {w}" for w in report.get("warnings",[])]
    Path(markdown_path).write_text("\n".join(lines)+"\n",encoding="utf-8")
