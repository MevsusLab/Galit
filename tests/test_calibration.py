from __future__ import annotations
import csv
from dataclasses import replace
from pathlib import Path
import pytest
from galit.calibration import *
from galit.calibration.loader import template_row

def rows(n=4):
 out=[]
 for i in range(n):
  r=template_row(example=True);r["well_id"]=f"SYN-{i//2}";r["timestamp"]=f"2025-01-{i%2+1:02d}T00:00:00+00:00";r["target_temperature_c"]=25+i;r["is_synthetic"]=True;out.append(r)
 return out

def snapshots(n=4):
 v=validate_rows(rows(n));return [WellSnapshot(r,InputProvenance(r["source"],r["quality"],r["timestamp"])) for r in v.rows]

def test_schema_strict_missing_range_duplicate_order_and_charge_warning():
 r=rows(2);r[0].pop("depth_m");r[1]["ph"]=99
 result=validate_rows(r,strict=False);assert result.errors
 with pytest.raises(ValueError):validate_rows(r)
 dup=rows(2);dup[1]["timestamp"]=dup[0]["timestamp"]
 assert any("duplicate" in x for x in validate_rows(dup,strict=False).errors)
 imbalanced=rows(1);imbalanced[0]["cl_mg_l"]=1
 assert validate_rows(imbalanced,strict=False).warnings

def test_csv_and_xlsx_roundtrip(tmp_path):
 for ext in ("csv","xlsx"):
  p=tmp_path/f"t.{ext}";generate_template(p,example=True);d=load_history(p);assert len(d.snapshots)==1 and d.synthetic

def test_mapper_explicit_provenance_no_defaults():
 c=snapshot_to_well_case(snapshots(1)[0]);assert c.provenance.sources["thermal.u_to"]=="measured";assert c.geometry.roughness_m==4.6e-5

def test_split_no_overlap_reproducible_and_temporal():
 ss=snapshots(8);a,b=split_by_well(ss,test_fraction=.5,seed=7);a2,b2=split_by_well(ss,test_fraction=.5,seed=7)
 assert [x.well_id for x in b]==[x.well_id for x in b2];assert not ({x.well_id for x in a}&{x.well_id for x in b})
 a,b=temporal_group_holdout(ss,test_fraction=.5);assert not ({x.well_id for x in a}&{x.well_id for x in b})

def test_leakage_prevention():
 assert_no_leakage(["depth_m"])
 with pytest.raises(ValueError):assert_no_leakage(["depth_m","target_corrosion_mm_y"])

def test_optimizer_train_only_and_serialization(tmp_path):
 ss=snapshots(8);train,test=split_by_well(ss,test_fraction=.5)
 original=[s.values["target_temperature_c"] for s in test]
 ps=calibrate_physical(train,test,"abc",bounds=(5,20),method="grid",synthetic=True)
 assert original==[s.values["target_temperature_c"] for s in test]
 p=tmp_path/"p.json";ps.save(p);assert ParameterSet.load(p).parameters==ps.parameters;assert set(ps.train_wells).isdisjoint(ps.test_wells)

def test_risk_policy_calibration_uses_public_severity_scores():
 ss=snapshots(8);train,test=split_by_well(ss,test_fraction=.5)
 for index,snapshot in enumerate(ss): snapshot.values["risk_label"]=index/10
 ps=calibrate_risk_policy(train,test,"abc",seed=7,synthetic=True)
 assert ps.kind=="risk-policy";assert sum(ps.parameters.values())==pytest.approx(1.0)

def test_metrics_unavailable_and_synthetic_disclaimer():
 m=regression_metrics("x",[None],[None]);assert not m["available"]
 d=HistoryDataset(snapshots(1),dataset_hash="x",synthetic=True);p=ParameterSet("physical",{"thermal.u_to":15},"v","now","x",["a"],[],{})
 report=evaluate_parameter_set(p,d);assert "not evidence" in report["synthetic_disclaimer"];assert not report["metrics"]["pressure"]["available"]
