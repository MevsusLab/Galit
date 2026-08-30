"""Explainable water-cut and injector-influence screening.

The deterministic baseline is intentionally non-causal: without tracers, a calibrated
reservoir model and labelled history it only ranks possible associations for review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import csv, io, json, math, os, statistics, threading, time
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

SCHEMA_VERSION = 1
POLICY_VERSION = "galit-watercut-screening-1.0"
DISCLAIMER = ("Инженерная screening-диагностика возможного влияния/прорыва, не доказательство причинности. "
              "Подтвердите гипотезу промысловыми замерами, пробами/ионным fingerprint, трассером и ГДИС; "
              "изменение закачки допустимо только после инженерного подтверждения.")
PRESSURE_UNITS = {"pa", "kpa", "mpa", "bar"}
ROLES = {"producer", "injector"}

class WatercutStorageError(RuntimeError): pass
class WatercutConflictError(RuntimeError): pass
class WatercutNotFoundError(LookupError): pass


def _text(value: Any, name: str) -> str:
    value = str(value).strip()
    if not value: raise ValueError(f"{name} must be non-empty")
    return value

def _timestamp(value: datetime | date | str) -> datetime:
    if isinstance(value, str): value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, date) and not isinstance(value, datetime): value = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None: value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

def _number(value: Any, name: str, *, nonnegative: bool = False) -> float | None:
    if value is None or value == "": return None
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0): raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return result

def _optional_text(value: Any) -> str | None:
    return None if value is None or str(value).strip() == "" else str(value).strip()

@dataclass(frozen=True)
class WellMetadata:
    well: str
    role: str
    latitude: float | None = None
    longitude: float | None = None
    field_name: str | None = None
    cluster: str | None = None
    site: str | None = None
    reservoir: str | None = None
    zone: str | None = None
    layer: str | None = None
    commissioning_date: date | None = None
    def __post_init__(self):
        object.__setattr__(self, "well", _text(self.well, "well")); role = self.role.strip().lower()
        if role not in ROLES: raise ValueError("role must be producer or injector")
        object.__setattr__(self, "role", role)
        lat, lon = _number(self.latitude, "latitude"), _number(self.longitude, "longitude")
        if lat is not None and not -90 <= lat <= 90: raise ValueError("latitude must be within [-90, 90]")
        if lon is not None and not -180 <= lon <= 180: raise ValueError("longitude must be within [-180, 180]")
        object.__setattr__(self, "latitude", lat); object.__setattr__(self, "longitude", lon)
        for name in ("field_name","cluster","site","reservoir","zone","layer"): object.__setattr__(self,name,_optional_text(getattr(self,name)))
        if isinstance(self.commissioning_date, str): object.__setattr__(self,"commissioning_date",date.fromisoformat(self.commissioning_date))
    def to_dict(self):
        value=asdict(self); value["commissioning_date"]=self.commissioning_date.isoformat() if self.commissioning_date else None; return value
    @classmethod
    def from_dict(cls, value): return cls(**value)

@dataclass(frozen=True)
class ProductionHistory:
    well: str
    timestamp: datetime
    q_oil_m3d: float
    q_water_m3d: float
    id: str | None = None
    liquid_rate_m3d: float | None = None
    water_cut: float | None = None
    water_cut_unit: str | None = None
    pressure: float | None = None
    pressure_unit: str | None = None
    choke: float | None = None
    downtime_hours: float | None = None
    status: str | None = None
    quality_flags: tuple[str,...] = ()
    def __post_init__(self):
        object.__setattr__(self,"well",_text(self.well,"well")); object.__setattr__(self,"timestamp",_timestamp(self.timestamp))
        for name in ("q_oil_m3d","q_water_m3d","liquid_rate_m3d","pressure","choke","downtime_hours"):
            object.__setattr__(self,name,_number(getattr(self,name),name,nonnegative=True))
        total=self.q_oil_m3d+self.q_water_m3d
        calculated=(self.q_water_m3d/total) if total>0 else None
        flags=list(self.quality_flags)
        supplied=None
        if self.water_cut is not None:
            unit=(self.water_cut_unit or "").strip().lower()
            if unit not in {"fraction","percent","%"}: raise ValueError("water_cut_unit must be fraction or percent")
            supplied=_number(self.water_cut,"water_cut",nonnegative=True)/(100 if unit in {"percent","%"} else 1)
            if supplied is not None and not 0<=supplied<=1: raise ValueError("water_cut must be within its explicit unit range")
            if calculated is not None and abs(supplied-calculated)>.03: flags.append("water_cut_conflicts_with_rates")
        if self.liquid_rate_m3d is not None and abs(self.liquid_rate_m3d-total)>max(1,.03*max(total,1)): flags.append("liquid_rate_conflicts_with_components")
        object.__setattr__(self,"water_cut",calculated if calculated is not None else supplied)
        object.__setattr__(self,"water_cut_unit","fraction")
        if self.pressure is not None:
            unit=(self.pressure_unit or "").strip().lower()
            if unit not in PRESSURE_UNITS: raise ValueError("pressure_unit must be pa, kpa, mpa, or bar")
            object.__setattr__(self,"pressure_unit",unit)
        object.__setattr__(self,"status",_optional_text(self.status)); object.__setattr__(self,"quality_flags",tuple(sorted(set(flags))))
        object.__setattr__(self,"id",self.id or f"{self.well}|{self.timestamp.isoformat()}")
    def to_dict(self):
        value=asdict(self); value["timestamp"]=self.timestamp.isoformat(); value["quality_flags"]=list(self.quality_flags); return value
    @classmethod
    def from_dict(cls,value): return cls(**value)

@dataclass(frozen=True)
class InjectionHistory:
    well: str
    timestamp: datetime
    injection_rate_m3d: float
    id: str | None = None
    injection_pressure: float | None = None
    pressure_unit: str | None = None
    status: str | None = None
    def __post_init__(self):
        object.__setattr__(self,"well",_text(self.well,"well")); object.__setattr__(self,"timestamp",_timestamp(self.timestamp))
        object.__setattr__(self,"injection_rate_m3d",_number(self.injection_rate_m3d,"injection_rate_m3d",nonnegative=True))
        object.__setattr__(self,"injection_pressure",_number(self.injection_pressure,"injection_pressure",nonnegative=True))
        if self.injection_pressure is not None:
            unit=(self.pressure_unit or "").strip().lower()
            if unit not in PRESSURE_UNITS: raise ValueError("pressure_unit must be pa, kpa, mpa, or bar")
            object.__setattr__(self,"pressure_unit",unit)
        object.__setattr__(self,"status",_optional_text(self.status)); object.__setattr__(self,"id",self.id or f"{self.well}|{self.timestamp.isoformat()}")
    def to_dict(self): value=asdict(self); value["timestamp"]=self.timestamp.isoformat(); return value
    @classmethod
    def from_dict(cls,value): return cls(**value)

@dataclass(frozen=True)
class WatercutPolicy:
    version: str = POLICY_VERSION
    min_points: int = 5
    baseline_points: int = 3
    growing_change_pp: float = .08
    critical_change_pp: float = .20
    max_distance_km: float = 15
    min_link_score: float = .42
    max_lag_days: int = 180

@dataclass(frozen=True)
class OilForecastPoint:
    days: int; estimate_m3d: float; low_m3d: float; high_m3d: float
@dataclass(frozen=True)
class InjectorCandidate:
    injector: str; score: float; distance_km: float; lag_days: int | None; confidence: str; contributions: dict[str,float]; reasons: tuple[str,...]
@dataclass(frozen=True)
class WatercutDiagnosis:
    well: str; status: str; severity: str; current_water_cut: float | None; absolute_change_pp: float | None; relative_change: float | None
    slope_pp_day: float | None; onset_date: str | None; onset_window: tuple[str,str] | None; confidence: str; data_quality: float
    reasons: tuple[str,...]; missing_data: tuple[str,...]; oil_forecast: tuple[OilForecastPoint,...]; possible_oil_loss_m3d: float | None
    candidate_injectors: tuple[InjectorCandidate,...]; evidence: dict[str,float]; alternative_explanations: tuple[str,...]
    policy_version: str = POLICY_VERSION; assumptions: tuple[str,...] = ("Связь оценивается по расстоянию, геологии и временному совпадению, не по причинности.",); disclaimer: str = DISCLAIMER
    def to_dict(self): return asdict(self)
@dataclass(frozen=True)
class WatercutLink:
    injector: str; producer: str; score: float; status: str; distance_km: float; lag_days: int | None; confidence: str
    injector_latitude: float; injector_longitude: float; producer_latitude: float; producer_longitude: float; bearing_deg: float; label: str
    def to_dict(self): return asdict(self)

def haversine_km(lat1,lon1,lat2,lon2):
    if not all(math.isfinite(float(v)) for v in (lat1,lon1,lat2,lon2)): raise ValueError("coordinates must be finite")
    p1,p2=math.radians(lat1),math.radians(lat2); dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 6371.0088*2*math.atan2(math.sqrt(a),math.sqrt(1-a))
def bearing_deg(lat1,lon1,lat2,lon2):
    p1,p2=math.radians(lat1),math.radians(lat2); dl=math.radians(lon2-lon1)
    return (math.degrees(math.atan2(math.sin(dl)*math.cos(p2),math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)))+360)%360

def _median_slope(points, field):
    pairs=[]
    for i,a in enumerate(points):
        for b in points[i+1:]:
            days=(b.timestamp-a.timestamp).total_seconds()/86400
            if days>0: pairs.append((getattr(b,field)-getattr(a,field))/days)
    return statistics.median(pairs) if pairs else 0

def _same_geology(a,b):
    contributions={}; mismatch=False
    for name in ("field_name","reservoir","zone","layer"):
        av,bv=getattr(a,name),getattr(b,name)
        if av and bv: contributions[name]=1.0 if av.casefold()==bv.casefold() else 0.0; mismatch |= name in {"reservoir","zone","layer"} and contributions[name]==0
        else: contributions[name]=.35
    return contributions,mismatch

def _injector_candidates(producer, injectors, production, injections, policy):
    result=[]
    for injector in injectors:
        if None in (producer.latitude,producer.longitude,injector.latitude,injector.longitude): continue
        distance=haversine_km(producer.latitude,producer.longitude,injector.latitude,injector.longitude)
        if distance>policy.max_distance_km: continue
        geology,mismatch=_same_geology(producer,injector)
        if mismatch: continue
        rows=sorted([x for x in injections if x.well.casefold()==injector.well.casefold()],key=lambda x:x.timestamp)
        inj_change=0; lag=None; temporal=.15; overlap=0
        if len(rows)>=3 and len(production)>=policy.min_points:
            half=max(1,len(rows)//2); old=statistics.median(x.injection_rate_m3d for x in rows[:half]); new=statistics.median(x.injection_rate_m3d for x in rows[-half:])
            inj_change=max(0,min(1,(new-old)/max(old,1)/.5)); change_at=rows[half].timestamp
            rising=[x for x in production if x.timestamp>=change_at and x.water_cut is not None]
            if rising: lag=max(0,int((rising[0].timestamp-change_at).total_seconds()/86400)); temporal=max(0,1-lag/policy.max_lag_days)
            overlap=min(1,min(len(rows),len(production))/8)
        distance_score=max(0,1-distance/policy.max_distance_km); shared=statistics.mean(geology.values())
        parts={"distance":distance_score,"shared_reservoir":shared,"injection_change":inj_change,"lag_correlation":temporal,"data_overlap":overlap}
        score=.30*distance_score+.25*shared+.20*inj_change+.15*temporal+.10*overlap
        confidence="medium" if overlap>=.6 and lag is not None else "low"
        reasons=(f"геодезическое расстояние {distance:.2f} км", "общая геология учтена без отождествления разных пластов", "корреляция не вычисляется при малой истории")
        result.append(InjectorCandidate(injector.well,round(score,4),round(distance,3),lag,confidence,{k:round(v,4) for k,v in parts.items()},reasons))
    return tuple(sorted(result,key=lambda x:(-x.score,x.injector.casefold())))

def diagnose_watercut(metadata, production_history, injector_metadata=(), injection_history=(), *, policy=None):
    policy=policy or WatercutPolicy()
    if metadata.role!="producer": raise ValueError("water-cut diagnosis requires producer role")
    rows=sorted([x for x in production_history if x.well.casefold()==metadata.well.casefold() and x.water_cut is not None],key=lambda x:x.timestamp)
    missing=[]; reasons=[]
    if len(rows)<policy.min_points:
        return WatercutDiagnosis(metadata.well,"insufficient_data","low",rows[-1].water_cut if rows else None,None,None,None,None,None,"unavailable",len(rows)/policy.min_points,tuple(reasons),("production_history",),(),None,(),{},("неполная история",))
    base=statistics.median(x.water_cut for x in rows[:policy.baseline_points]); current=statistics.median(x.water_cut for x in rows[-min(3,len(rows)):])
    change=current-base; relative=change/max(base,.01); slope=_median_slope(rows,"water_cut")
    threshold=base+max(.04,change*.35); onset=next((x.timestamp for x in rows[policy.baseline_points:] if x.water_cut>=threshold),None)
    conflicts=sum(bool(x.quality_flags) for x in rows); quality=max(0,min(1,len(rows)/10))*(1-.15*conflicts/max(len(rows),1))
    if change>=policy.critical_change_pp: severity="critical"
    elif change>=policy.growing_change_pp: severity="growing"
    else: severity="low"
    confidence="high" if quality>=.8 and (rows[-1].timestamp-rows[0].timestamp).days>=30 else "medium" if quality>=.5 else "low"
    reasons += [f"robust baseline {base:.1%}",f"absolute change {change:+.1%}",f"Theil–Sen-like median slope {slope:+.4f}/day"]
    oil=[x for x in rows if not (x.status and x.status.casefold() in {"down","offline","stopped","простой"}) and (x.downtime_hours or 0)<20]
    forecasts=[]; loss=None
    if len(oil)>=policy.min_points:
        oil_slope=_median_slope(oil,"q_oil_m3d"); residuals=[abs(oil[i].q_oil_m3d-oil[i-1].q_oil_m3d) for i in range(1,len(oil))]; uncertainty=statistics.median(residuals) if residuals else 0
        last=statistics.median(x.q_oil_m3d for x in oil[-min(3,len(oil)):])
        for days in (7,30,90):
            estimate=max(0,last+oil_slope*days); spread=uncertainty*math.sqrt(max(days,1)/7)+abs(oil_slope)*days*.35
            forecasts.append(OilForecastPoint(days,round(estimate,3),round(max(0,estimate-spread),3),round(estimate+spread,3)))
        loss=max(0,oil[0].q_oil_m3d-forecasts[1].estimate_m3d)
    else: missing.append("oil_forecast_history")
    candidates=_injector_candidates(metadata,[x for x in injector_metadata if x.role=="injector"],rows,injection_history,policy)
    candidates=tuple(x for x in candidates if x.score>=policy.min_link_score)
    acceleration=min(1,max(0,change/policy.critical_change_pp)); oil_response=0 if loss is None else min(1,loss/max(rows[-1].q_oil_m3d,1)); association=candidates[0].score if candidates else 0
    evidence={"water_cut_acceleration":round(acceleration,4),"oil_response":round(oil_response,4),"injector_association":round(association,4)}
    combined=.5*acceleration+.25*oil_response+.25*association
    if combined>=.7 and severity!="low": severity="critical"
    elif combined>=.35 and severity!="low": severity="growing"
    alternatives=("изменение режима/насоса или штуцера","утечка или канал высокой проводимости","ошибка/несогласованность измерений","снижение жидкости без замещения нефти водой")
    window=(rows[max(0,policy.baseline_points-1)].timestamp.date().isoformat(),onset.date().isoformat()) if onset else None
    return WatercutDiagnosis(metadata.well,"screening",severity,round(current,6),round(change,6),round(relative,6),round(slope,8),onset.date().isoformat() if onset else None,window,confidence,round(quality,3),tuple(reasons),tuple(missing),tuple(forecasts),round(loss,3) if loss is not None else None,candidates,evidence,alternatives)

def build_watercut_links(metadata, production, injections, *, policy=None, top_n=50):
    policy=policy or WatercutPolicy(); producers=[x for x in metadata if x.role=="producer"]; injectors=[x for x in metadata if x.role=="injector"]
    links={}
    for p in producers:
        diagnosis=diagnose_watercut(p,production,injectors,injections,policy=policy)
        for candidate in diagnosis.candidate_injectors:
            i=next(x for x in injectors if x.well==candidate.injector)
            if None in (i.latitude,i.longitude,p.latitude,p.longitude): continue
            key=(i.well.casefold(),p.well.casefold())
            links[key]=WatercutLink(i.well,p.well,candidate.score,diagnosis.severity,candidate.distance_km,candidate.lag_days,candidate.confidence,i.latitude,i.longitude,p.latitude,p.longitude,round(bearing_deg(i.latitude,i.longitude,p.latitude,p.longitude),2),f"{i.well} → возможное влияние → {p.well}")
    return tuple(sorted(links.values(),key=lambda x:(-x.score,x.injector,x.producer))[:top_n])

class WatercutRepository:
    _locks={}; _guard=threading.Lock()
    def __init__(self,path="data/watercut.json",lock_timeout=5):
        self.path=Path(path); self.lock_timeout=lock_timeout
        with self._guard: self._lock=self._locks.setdefault(str(self.path.resolve()),threading.RLock())
    def _read(self):
        if not self.path.exists(): return {"metadata":[],"production":[],"injection":[]}
        try:
            p=json.loads(self.path.read_text(encoding="utf-8"))
            if p.get("schema_version")!=SCHEMA_VERSION: raise ValueError("unsupported schema")
            return {"metadata":[WellMetadata.from_dict(x) for x in p.get("metadata",[])],"production":[ProductionHistory.from_dict(x) for x in p.get("production",[])],"injection":[InjectionHistory.from_dict(x) for x in p.get("injection",[])]}
        except Exception as exc: raise WatercutStorageError(f"watercut storage is corrupt: {exc}") from exc
    def _lockfile(self):
        path=self.path.with_suffix(self.path.suffix+".lock"); self.path.parent.mkdir(parents=True,exist_ok=True); deadline=time.monotonic()+self.lock_timeout
        while True:
            try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd); return path
            except FileExistsError:
                if time.monotonic()>=deadline: raise WatercutStorageError("timed out waiting for lock")
                time.sleep(.02)
    def _write(self,data):
        payload={"schema_version":SCHEMA_VERSION,**{k:[x.to_dict() for x in v] for k,v in data.items()}}; temp=self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temp.open("w",encoding="utf-8",newline="\n") as f: json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
            os.replace(temp,self.path)
        except OSError as exc: raise WatercutStorageError(str(exc)) from exc
        finally: temp.unlink(missing_ok=True)
    def _mutate(self,action):
        with self._lock:
            lock=self._lockfile()
            try: data=self._read(); result=action(data); self._write(data); return result
            finally: lock.unlink(missing_ok=True)
    def upsert_metadata(self,item):
        def action(data):
            for n,old in enumerate(data["metadata"]):
                if old.well.casefold()==item.well.casefold(): data["metadata"][n]=item; return item
            data["metadata"].append(item); return item
        return self._mutate(action)
    def ingest_production(self,items): return self._ingest("production",list(items))
    def ingest_injection(self,items): return self._ingest("injection",list(items))
    def _ingest(self,key,items):
        def action(data):
            known={x.id:x for x in data[key]}; created=0
            for x in items:
                if x.id in known:
                    if x!=known[x.id]: raise WatercutConflictError(f"duplicate id {x.id} has different payload")
                else: data[key].append(x); known[x.id]=x; created+=1
            return {"received":len(items),"created":created,"idempotent":len(items)-created}
        return self._mutate(action)
    def list_metadata(self,role=None): return [x for x in self._read()["metadata"] if role is None or x.role==role]
    def list_production(self,well=None): return [x for x in self._read()["production"] if well is None or x.well.casefold()==well.casefold()]
    def list_injection(self,well=None): return [x for x in self._read()["injection"] if well is None or x.well.casefold()==well.casefold()]

def _csv(rows,fields):
    out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=fields,lineterminator="\n"); writer.writeheader()
    for x in rows: writer.writerow({k:("|".join(v) if isinstance(v,(list,tuple)) else v) for k,v in x.to_dict().items() if k in fields})
    return out.getvalue()
def metadata_csv_template(): return _csv([WellMetadata("Добывающая 139","producer",52.4,30.7,"Речицкое","Куст 3",reservoir="D3",zone="A",layer="1")],list(WellMetadata.__dataclass_fields__))
def production_csv_template(): return _csv([ProductionHistory("Добывающая 139",datetime(2026,1,1,tzinfo=timezone.utc),12,28)],list(ProductionHistory.__dataclass_fields__))
def injection_csv_template(): return _csv([InjectionHistory("Нагнетательная 82",datetime(2026,1,1,tzinfo=timezone.utc),180, injection_pressure=12,pressure_unit="mpa",status="active")],list(InjectionHistory.__dataclass_fields__))
def _from_csv(text,cls):
    rows=[]; errors=[]
    for n,row in enumerate(csv.DictReader(io.StringIO(text)),2):
        try: rows.append(cls(**{k:v for k,v in row.items() if v not in (None,"") and k in cls.__dataclass_fields__}))
        except Exception as exc: errors.append(f"row {n}: {exc}")
    return rows,errors
def metadata_from_csv(text): return _from_csv(text,WellMetadata)
def production_from_csv(text): return _from_csv(text,ProductionHistory)
def injection_from_csv(text): return _from_csv(text,InjectionHistory)
