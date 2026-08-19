"""Bounded physical and risk-policy calibration using train records only."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
import json, math, random
from pathlib import Path
from typing import Any, Callable

from .loader import snapshot_to_well_case
from .metrics import regression_metrics, classification_metrics
from .schema import WellSnapshot

MODEL_VERSION="galit-0.1.0"
@dataclass
class ParameterSet:
    kind:str; parameters:dict[str,float]; model_version:str; created_at:str
    dataset_hash:str; train_wells:list[str]; test_wells:list[str]; metrics:dict[str,Any]
    synthetic:bool=False
    def save(self,path:str|Path)->Path:
        p=Path(path); p.write_text(json.dumps(asdict(self),ensure_ascii=False,indent=2),encoding="utf-8"); return p
    @classmethod
    def load(cls,path:str|Path)->"ParameterSet": return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

def _bounded_search(objective:Callable[[float],float], bounds:tuple[float,float], *, method:str="grid", seed:int=0, iterations:int=81)->tuple[float,float]:
    lo,hi=bounds
    if not (math.isfinite(lo) and math.isfinite(hi) and lo<hi): raise ValueError("invalid bounds")
    if method not in {"grid","random","robust"}: raise ValueError("method must be grid, random, or robust")
    if method=="random":
        rng=random.Random(seed); candidates=[lo,hi]+[rng.uniform(lo,hi) for _ in range(max(1,iterations-2))]
    else: candidates=[lo+(hi-lo)*i/(iterations-1) for i in range(iterations)]
    scored=[(objective(x),x) for x in candidates]; scored=[p for p in scored if math.isfinite(p[0])]
    if not scored: raise ValueError("objective produced no finite values")
    return min(scored)[1],min(scored)[0]

def _temperature_prediction(snapshot:WellSnapshot,u_to:float)->float:
    from galit.wellbore import temperature_profile
    case=snapshot_to_well_case(snapshot); case.thermal=replace(case.thermal,u_to=u_to)
    depths,temps,_=temperature_profile(case.geometry,case.rate,case.fluid,case.thermal)
    target_depth=float(snapshot.values.get("measurement_depth_m",0) or 0)
    return min(zip(depths,temps),key=lambda x:abs(x[0]-target_depth))[1]

def calibrate_physical(train:list[WellSnapshot], test:list[WellSnapshot], dataset_hash:str,
                       *, parameter:str="thermal.u_to", bounds:tuple[float,float]=(2,80),
                       method:str="robust", seed:int=0, synthetic:bool=False)->ParameterSet:
    if parameter!="thermal.u_to": raise ValueError("only identifiable site parameter thermal.u_to is supported")
    usable=[s for s in train if s.values.get("target_temperature_c") not in (None,"")]
    if not usable: raise ValueError("thermal.u_to requires train target_temperature_c measurements")
    def objective(value:float)->float:
        errors=[_temperature_prediction(s,value)-float(s.values["target_temperature_c"]) for s in usable]
        if method=="robust": return sorted(abs(e) for e in errors)[len(errors)//2]
        return sum(e*e for e in errors)/len(errors)
    best,_=_bounded_search(objective,bounds,method=method,seed=seed)
    def measure(rows):
        targets=[float(s.values["target_temperature_c"]) if s.values.get("target_temperature_c") not in (None,"") else None for s in rows]
        baseline=[_temperature_prediction(s,float(s.values["u_to_w_m2k"])) if y is not None else None for s,y in zip(rows,targets)]
        calibrated=[_temperature_prediction(s,best) if y is not None else None for s,y in zip(rows,targets)]
        return {"baseline":regression_metrics("temperature_c",targets,baseline),"calibrated":regression_metrics("temperature_c",targets,calibrated)}
    metrics={"train":measure(train),"test":measure(test)}
    return ParameterSet("physical",{parameter:best},MODEL_VERSION,datetime.now(timezone.utc).isoformat(),dataset_hash,
      sorted({s.well_id for s in train}),sorted({s.well_id for s in test}),metrics,synthetic)

def calibrate_risk_policy(train:list[WellSnapshot],test:list[WellSnapshot],dataset_hash:str,*,seed:int=0,synthetic:bool=False)->ParameterSet:
    """Fit non-negative simplex risk weights to labels/rankings; never physical parameters."""
    from galit.integrated import diagnose
    mechanisms=("halite","calcite","wax","corrosion")
    usable=[s for s in train if s.values.get("risk_label") not in (None,"")]
    if not usable: raise ValueError("risk-policy calibration requires train risk_label")
    features=[]
    for s in usable:
        d=diagnose(snapshot_to_well_case(s)); features.append(([d.severity[m] for m in mechanisms],float(s.values["risk_label"])))
    rng=random.Random(seed); candidates=[[.3,.15,.3,.25]]
    for _ in range(500):
        raw=[rng.random() for _ in mechanisms]; total=sum(raw); candidates.append([v/total for v in raw])
    weights=min(candidates,key=lambda w:sum((sum(a*b for a,b in zip(x,w))-y)**2 for x,y in features))
    params={f"weight.{m}":w for m,w in zip(mechanisms,weights)}
    def evaluate(rows):
        ys=[]; base=[]; cal=[]
        for s in rows:
            if s.values.get("risk_label") in (None,""): continue
            d=diagnose(snapshot_to_well_case(s)); x=[d.severity[m] for m in mechanisms]
            ys.append(float(s.values["risk_label"]));base.append(sum(a*b for a,b in zip(x,[.3,.15,.3,.25])));cal.append(sum(a*b for a,b in zip(x,weights)))
        return {"baseline":classification_metrics("risk",ys,base),"calibrated":classification_metrics("risk",ys,cal)}
    return ParameterSet("risk-policy",params,MODEL_VERSION,datetime.now(timezone.utc).isoformat(),dataset_hash,
      sorted({s.well_id for s in train}),sorted({s.well_id for s in test}),{"train":evaluate(train),"test":evaluate(test)},synthetic)
