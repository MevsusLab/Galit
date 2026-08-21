"""Windows-friendly CLI for GALIT offline calibration."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from galit.calibration import (ParameterSet, audit_csv, blocked_parameter_set,
 calibrate_physical, calibrate_risk_policy, evaluate_parameter_set, generate_template,
 load_history, split_by_well, temporal_group_holdout, write_audit_markdown, write_report)

def main(argv=None)->int:
 p=argparse.ArgumentParser(prog="calibration_cli.py"); sub=p.add_subparsers(dest="command",required=True)
 g=sub.add_parser("generate-template");g.add_argument("output");g.add_argument("--example",action="store_true")
 v=sub.add_parser("validate");v.add_argument("input")
 d=sub.add_parser("audit");d.add_argument("input");d.add_argument("--json",default="data-audit.json");d.add_argument("--markdown",default="data-audit.md")
 c=sub.add_parser("calibrate");c.add_argument("input");c.add_argument("output");c.add_argument("--kind",choices=["physical","risk-policy"],default="physical");c.add_argument("--split",choices=["group","temporal"],default="group");c.add_argument("--test-fraction",type=float,default=.2);c.add_argument("--seed",type=int,default=0);c.add_argument("--method",choices=["grid","random","robust"],default="robust")
 e=sub.add_parser("evaluate");e.add_argument("input");e.add_argument("parameters");e.add_argument("--json",default="calibration-report.json");e.add_argument("--markdown",default="calibration-report.md")
 a=p.parse_args(argv)
 if a.command=="generate-template": generate_template(a.output,example=a.example);print(a.output);return 0
 if a.command=="audit":
  report=audit_csv(a.input);Path(a.json).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");write_audit_markdown(report,a.markdown);print(a.json);print(a.markdown);return 0
 if a.command=="calibrate":
  try:
   data=load_history(a.input)
   split=split_by_well if a.split=="group" else temporal_group_holdout; train,test=split(data.snapshots,test_fraction=a.test_fraction,seed=a.seed)
   ps=calibrate_physical(train,test,data.dataset_hash,method=a.method,seed=a.seed,synthetic=data.synthetic) if a.kind=="physical" else calibrate_risk_policy(train,test,data.dataset_hash,seed=a.seed,synthetic=data.synthetic)
  except ValueError as exc:
   digest=hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
   ps=blocked_parameter_set(digest,[str(exc)],artifact_id=f"blocked-{digest[:12]}")
  ps.save(a.output);print(json.dumps({"artifact":a.output,"status":ps.validation_status,"metrics":dict(ps.metrics),"limitations":list(ps.limitations)},ensure_ascii=False,indent=2));return 0
 data=load_history(a.input)
 if a.command=="validate": print(json.dumps({"valid":True,"snapshots":len(data.snapshots),"wells":len({s.well_id for s in data.snapshots}),"dataset_hash":data.dataset_hash,"warnings":data.warnings},ensure_ascii=False,indent=2));return 0
 ps=ParameterSet.load(a.parameters);report=evaluate_parameter_set(ps,data);write_report(report,a.json,a.markdown);print(a.json);print(a.markdown);return 0
if __name__=="__main__": raise SystemExit(main())
