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
    """Tie-safe binary ranking metrics.

    Missing labels block the complete evaluation rather than silently changing
    the cohort. If K cuts a score tie, metrics are expected values across all
    orderings within that tied group, so input row order cannot change results.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if not labels or any(y is None for y in labels):
        return {"name":"risk_ranking","available":False,"reason":"target unavailable","n":0}
    if any(s is None for s in scores):
        return {"name":"risk_ranking","available":False,"reason":"score unavailable","n":0}
    pairs=[(float(y),float(s)) for y,s in zip(labels,scores)]
    if any(y not in (0.0,1.0) for y,_ in pairs):
        raise ValueError("ranking labels must be binary 0/1")
    kk=min(k,len(pairs)); ranked=sorted(pairs,key=lambda x:x[1],reverse=True)
    selected_positive=0.0; selected_count=0; position=0; dcg=0.0
    while position < len(ranked) and selected_count < kk:
        end=position+1
        while end < len(ranked) and ranked[end][1] == ranked[position][1]: end+=1
        group=ranked[position:end]; take=min(kk-selected_count,len(group))
        positive_fraction=sum(y for y,_ in group)/len(group)
        selected_positive += take*positive_fraction
        for rank_index in range(selected_count,selected_count+take):
            dcg += positive_fraction/math.log2(rank_index+2)
        selected_count += take; position=end
    positives=sum(y for y,_ in pairs)
    ideal_positive_slots=min(int(positives),kk)
    idcg=sum(1/math.log2(i+2) for i in range(ideal_positive_slots))
    return {
        "name":"risk_ranking","available":True,"n":len(pairs),"k":kk,
        "precision_at_k":selected_positive/kk,
        "recall_at_k":selected_positive/positives if positives else 0.0,
        "ndcg_at_k":dcg/idcg if idcg else 0.0,
        "missed_events":positives-selected_positive,
        "unnecessary_interventions":kk-selected_positive,
        "empty_positives":positives == 0,
        "tie_policy":"expected value across tied boundary",
    }

def coverage(rows:list[dict[str,Any]], targets:list[str])->dict[str,Any]:
    n=len(rows); return {t:{"available":sum(r.get(t) not in (None,"") for r in rows),
                           "missing":sum(r.get(t) in (None,"") for r in rows),
                           "coverage":(sum(r.get(t) not in (None,"") for r in rows)/n if n else 0.0)} for t in targets}
