"""Metrics that never invent unavailable targets."""
from __future__ import annotations
import math
from typing import Any

def _available(name: str, values: list[tuple[float|None,float|None]]) -> dict[str,Any]:
    pairs=[(float(y),float(p)) for y,p in values if y is not None and p is not None]
    if not pairs: return {"name":name,"available":False,"reason":"target unavailable","n":0}
    errors=[p-y for y,p in pairs]
    return {"name":name,"available":True,"n":len(pairs),
            "mae":sum(abs(e) for e in errors)/len(errors),
            "rmse":math.sqrt(sum(e*e for e in errors)/len(errors))}

def regression_metrics(name:str, targets:list[float|None], predictions:list[float|None])->dict[str,Any]:
    return _available(name,list(zip(targets,predictions)))

def classification_metrics(name:str, labels:list[float|None], probabilities:list[float|None])->dict[str,Any]:
    pairs=[(float(y),float(p)) for y,p in zip(labels,probabilities) if y is not None and p is not None]
    if not pairs:return {"name":name,"available":False,"reason":"target unavailable","n":0}
    return {"name":name,"available":True,"n":len(pairs),
            "brier":sum((p-y)**2 for y,p in pairs)/len(pairs),
            "accuracy":sum((p>=.5)==(y>=.5) for y,p in pairs)/len(pairs)}

def ranking_metrics(labels:list[float|None], scores:list[float|None],k:int=5)->dict[str,Any]:
    pairs=[(float(y),float(s)) for y,s in zip(labels,scores) if y is not None and s is not None]
    if not pairs:return {"name":"risk_ranking","available":False,"reason":"target unavailable","n":0}
    ranked=sorted(pairs,key=lambda x:x[1],reverse=True); kk=min(k,len(ranked)); rel=[y for y,_ in ranked[:kk]]
    precision=sum(y>0 for y in rel)/kk
    dcg=sum((2**y-1)/math.log2(i+2) for i,y in enumerate(rel))
    ideal=sorted((y for y,_ in pairs),reverse=True)[:kk]; idcg=sum((2**y-1)/math.log2(i+2) for i,y in enumerate(ideal))
    return {"name":"risk_ranking","available":True,"n":len(pairs),"k":kk,"precision_at_k":precision,"ndcg_at_k":dcg/idcg if idcg else 0.0}

def coverage(rows:list[dict[str,Any]], targets:list[str])->dict[str,Any]:
    n=len(rows); return {t:{"available":sum(r.get(t) not in (None,"") for r in rows),
                           "missing":sum(r.get(t) in (None,"") for r in rows),
                           "coverage":(sum(r.get(t) not in (None,"") for r in rows)/n if n else 0.0)} for t in targets}
