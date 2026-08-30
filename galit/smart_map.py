"""Smart Map 2.0 geospatial screening service.

Existing domain repositories remain authoritative.  This store contains only explicit
historical risk observations and user-supplied GIS infrastructure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
import csv, hashlib, io, json, math, os, threading, time
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .digital_twin import WellIdentity, aware
from .watercut import haversine_km, bearing_deg

SCHEMA_VERSION = 1
MODEL_VERSION = "galit-smart-map-2.0"
DISCLAIMER = ("Объяснимый пространственно-временной screening, не доказательство причинности. "
              "Проверьте общий режим, качество измерений и инфраструктурные события.")
MECHANISMS = ("integrated", "halite", "calcite", "wax", "corrosion", "watercut", "equipment")
ROLES = {"producer", "injector", "unknown"}
ASSET_TYPES = {"pipeline", "flowline", "waterline", "road", "other"}
FACILITY_TYPES = {"gathering", "treatment", "injection", "metering", "other"}
STATUSES = {"active", "inactive", "planned", "unknown"}
QUALITY = {"good": 1.0, "questionable": .65, "poor": .35, "unknown": .2}

class SmartMapStorageError(RuntimeError): pass
class SmartMapConflictError(RuntimeError): pass
class SmartMapNotFoundError(LookupError): pass


def _finite(value: Any, name: str, low: float | None = None, high: float | None = None) -> float | None:
    if value is None or value == "": return None
    result = float(value)
    if not math.isfinite(result) or low is not None and result < low or high is not None and result > high:
        raise ValueError(f"{name} must be finite" + (f" within [{low}, {high}]" if low is not None else ""))
    return result


def _text(value: Any, name: str, optional: bool = False) -> str | None:
    if value is None or not str(value).strip():
        if optional: return None
        raise ValueError(f"{name} must be non-empty")
    return " ".join(str(value).strip().split())


def _stable(prefix: str, *parts: Any) -> str:
    raw = "|".join("" if x is None else str(x).strip().casefold() for x in parts)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


def _coordinates(value: Any, geometry: str) -> tuple:
    if geometry == "Point":
        if not isinstance(value, (list, tuple)) or len(value) != 2: raise ValueError("Point coordinates must be [longitude, latitude]")
        lon, lat = _finite(value[0], "longitude", -180, 180), _finite(value[1], "latitude", -90, 90)
        return (lon, lat)
    if geometry == "LineString":
        if not isinstance(value, (list, tuple)) or len(value) < 2: raise ValueError("LineString requires at least two positions")
        return tuple(_coordinates(x, "Point") for x in value)
    if geometry == "MultiLineString":
        if not isinstance(value, (list, tuple)) or not value: raise ValueError("MultiLineString requires segments")
        return tuple(_coordinates(x, "LineString") for x in value)
    raise ValueError("unsupported geometry; use Point, LineString or MultiLineString")


@dataclass(frozen=True)
class RiskObservation:
    well: WellIdentity
    occurred_at: datetime
    latitude: float
    longitude: float
    severities: dict[str, float | None]
    integrated_risk: float | None = None
    well_role: str = "unknown"
    economic_loss: float | None = None
    economic_currency: str | None = None
    economic_unit: str | None = None
    source: str = "manual"
    source_record_id: str | None = None
    source_quality: str = "unknown"
    provenance: dict[str, Any] = field(default_factory=dict)
    observation_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    def __post_init__(self):
        object.__setattr__(self, "occurred_at", aware(self.occurred_at, "occurred_at"))
        object.__setattr__(self, "latitude", _finite(self.latitude, "latitude", -90, 90))
        object.__setattr__(self, "longitude", _finite(self.longitude, "longitude", -180, 180))
        role = str(self.well_role).lower()
        if role not in ROLES: raise ValueError("well_role must be producer, injector or unknown")
        object.__setattr__(self, "well_role", role)
        values = {k: _finite(v, f"severities.{k}", 0, 1) for k, v in self.severities.items() if k in MECHANISMS and k != "integrated"}
        object.__setattr__(self, "severities", dict(sorted(values.items())))
        risk = _finite(self.integrated_risk, "integrated_risk", 0, 1)
        object.__setattr__(self, "integrated_risk", risk)
        loss = _finite(self.economic_loss, "economic_loss", 0)
        object.__setattr__(self, "economic_loss", loss)
        if loss is not None and (not self.economic_currency or not self.economic_unit):
            raise ValueError("economic loss requires explicit currency and unit")
        object.__setattr__(self, "economic_currency", self.economic_currency.upper() if self.economic_currency else None)
        quality = str(self.source_quality).lower()
        if quality not in QUALITY: raise ValueError("unsupported source_quality")
        object.__setattr__(self, "source_quality", quality)
        source = _text(self.source, "source"); object.__setattr__(self, "source", source)
        source_id = self.source_record_id or _stable("source", self.well.canonical_id, self.occurred_at.isoformat(), source)
        object.__setattr__(self, "source_record_id", source_id)
        object.__setattr__(self, "observation_id", self.observation_id or _stable("risk", source, source_id, self.well.canonical_id))
        json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
    def severity(self, mechanism: str) -> float | None:
        if mechanism == "integrated": return self.integrated_risk
        if mechanism == "salts":
            vals = [self.severities.get(x) for x in ("halite", "calcite") if self.severities.get(x) is not None]
            return max(vals) if vals else None
        return self.severities.get(mechanism)
    def to_dict(self):
        data = asdict(self); data["well"] = self.well.to_dict(); data["occurred_at"] = self.occurred_at.isoformat(); return data
    @classmethod
    def from_dict(cls, value):
        data = dict(value); data["well"] = WellIdentity(**data["well"]); return cls(**data)


@dataclass(frozen=True)
class InfrastructureAsset:
    asset_id: str
    name: str
    asset_type: str
    status: str
    geometry_type: str
    coordinates: tuple
    diameter: float | None = None
    diameter_unit: str | None = None
    capacity: float | None = None
    capacity_unit: str | None = None
    source: str = "user_import"
    provenance: dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False
    def __post_init__(self):
        for name in ("asset_id", "name", "source"): object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.asset_type not in ASSET_TYPES | FACILITY_TYPES: raise ValueError("unsupported asset_type")
        if self.status not in STATUSES: raise ValueError("unsupported status")
        coords = _coordinates(self.coordinates, self.geometry_type); object.__setattr__(self, "coordinates", coords)
        if self.geometry_type == "Point" and self.asset_type not in FACILITY_TYPES: raise ValueError("Point requires a facility type")
        if self.geometry_type != "Point" and self.asset_type not in ASSET_TYPES: raise ValueError("line geometry requires a pipeline type")
        object.__setattr__(self, "diameter", _finite(self.diameter, "diameter", 0))
        object.__setattr__(self, "capacity", _finite(self.capacity, "capacity", 0))
        if self.diameter is not None and not self.diameter_unit: raise ValueError("diameter_unit is required")
        if self.capacity is not None and not self.capacity_unit: raise ValueError("capacity_unit is required")
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, value): return cls(**value)
    def feature(self):
        return {"type":"Feature", "id":self.asset_id, "geometry":{"type":self.geometry_type,"coordinates":self.coordinates},
                "properties":{k:v for k,v in self.to_dict().items() if k not in {"geometry_type","coordinates"}}}


@dataclass(frozen=True)
class SmartMapPolicy:
    version: str = "galit-smart-map-screening-2.0"
    warn: float = .35
    critical: float = .60
    stale_after_days: int = 45
    hotspot_distance_km: float = 3.0
    hotspot_window_days: int = 30
    hotspot_min_wells: int = 3
    max_frames: int = 24
    max_points: int = 3000
    heat_radius: int = 28
    heat_intensity: float = .8


def risk_status(value: float | None, policy: SmartMapPolicy | None = None) -> str:
    p=policy or SmartMapPolicy()
    return "missing" if value is None else "critical" if value >= p.critical else "growing" if value >= p.warn else "normal"


class SmartMapRepository:
    """Versioned UTF-8 atomic repository for manual observations and user GIS."""
    _locks: dict[str, threading.RLock]={}; _guard=threading.Lock()
    def __init__(self, path: str | Path | None = None, lock_timeout: float = 5):
        self.path=Path(path or os.environ.get("GALIT_SMART_MAP_STORAGE","data/smart_map.json")); self.lock_timeout=lock_timeout
        with self._guard: self._lock=self._locks.setdefault(str(self.path.resolve()),threading.RLock())
    def _read(self):
        if not self.path.exists(): return [],[]
        try:
            payload=json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported schema")
            return [RiskObservation.from_dict(x) for x in payload.get("observations",[])],[InfrastructureAsset.from_dict(x) for x in payload.get("infrastructure",[])]
        except Exception as exc: raise SmartMapStorageError(f"smart-map storage is corrupt: {exc}") from exc
    def _file_lock(self):
        lock=self.path.with_suffix(self.path.suffix+".lock"); deadline=time.monotonic()+self.lock_timeout; self.path.parent.mkdir(parents=True,exist_ok=True)
        while True:
            try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd); return lock
            except FileExistsError:
                if time.monotonic()>=deadline: raise SmartMapStorageError("timed out waiting for smart-map lock")
                time.sleep(.02)
    def _write(self, observations, assets):
        payload={"schema_version":SCHEMA_VERSION,"model_version":MODEL_VERSION,
                 "observations":[x.to_dict() for x in sorted(observations,key=lambda x:(x.occurred_at,x.well.canonical_id,x.observation_id))],
                 "infrastructure":[x.to_dict() for x in sorted(assets,key=lambda x:x.asset_id)]}
        self.path.parent.mkdir(parents=True,exist_ok=True); temp=self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temp.open("w",encoding="utf-8",newline="\n") as stream:
                json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp,self.path)
        except OSError as exc: raise SmartMapStorageError(f"failed atomic smart-map write: {exc}") from exc
        finally:
            try: temp.unlink(missing_ok=True)
            except OSError: pass
    def add_observation(self,item):
        with self._lock:
            lock=self._file_lock()
            try:
                rows,assets=self._read(); old=next((x for x in rows if x.observation_id==item.observation_id),None)
                if old:
                    if old.to_dict()==item.to_dict(): return old
                    raise SmartMapConflictError(f"observation {item.observation_id} conflicts")
                rows.append(item); self._write(rows,assets); return item
            finally: lock.unlink(missing_ok=True)
    def list_observations(self):
        with self._lock: return self._read()[0]
    def upsert_asset(self,item):
        with self._lock:
            lock=self._file_lock()
            try:
                rows,assets=self._read(); old=next((x for x in assets if x.asset_id==item.asset_id),None)
                if old and old.to_dict()!=item.to_dict(): raise SmartMapConflictError(f"asset {item.asset_id} already exists with different content")
                if not old: assets.append(item); self._write(rows,assets)
                return old or item
            finally: lock.unlink(missing_ok=True)
    def list_assets(self):
        with self._lock: return self._read()[1]
    def delete_asset(self,asset_id):
        with self._lock:
            lock=self._file_lock()
            try:
                rows,assets=self._read(); old=next((x for x in assets if x.asset_id==asset_id),None)
                if not old: raise SmartMapNotFoundError(f"asset {asset_id} not found")
                self._write(rows,[x for x in assets if x.asset_id!=asset_id]); return old
            finally: lock.unlink(missing_ok=True)


class SmartMapService:
    def __init__(self, repository: SmartMapRepository, policy: SmartMapPolicy | None=None, observations: Iterable[RiskObservation]=()):
        self.repository=repository; self.policy=policy or SmartMapPolicy(); self.supplied=tuple(observations)
    def observations(self, **filters):
        rows={x.observation_id:x for x in (*self.repository.list_observations(),*self.supplied)}.values()
        date_from, date_to, as_of = filters.get("date_from"),filters.get("date_to"),filters.get("as_of")
        if date_from: date_from=aware(date_from,"from")
        if date_to: date_to=aware(date_to,"to")
        if as_of: as_of=aware(as_of,"as_of")
        if date_from and date_to and date_from>date_to: raise ValueError("from must not be after to")
        def ok(x):
            return (not date_from or x.occurred_at>=date_from) and (not date_to or x.occurred_at<=date_to) and (not as_of or x.occurred_at<=as_of) and all(
                not filters.get(k) or str(getattr(x.well,k,None) or getattr(x,k,None) or "").casefold()==str(filters[k]).casefold()
                for k in ("field","cluster","site","reservoir","well_role"))
        return tuple(sorted((x for x in rows if ok(x)),key=lambda x:(x.occurred_at,x.well.canonical_id,x.observation_id)))
    def snapshot(self, *, as_of: datetime | None=None, mechanism="integrated", **filters):
        end=aware(as_of,"as_of") if as_of else datetime.now(timezone.utc); rows=self.observations(as_of=end,**filters); latest={}
        for x in rows:
            if (end-x.occurred_at).days<=self.policy.stale_after_days: latest[x.well.canonical_id]=x
        points=[]
        for x in sorted(latest.values(),key=lambda x:x.well.canonical_id):
            severity=x.severity(mechanism)
            if severity is None: continue
            points.append({**x.to_dict(),"selected_mechanism":mechanism,"selected_severity":severity,"status":risk_status(severity,self.policy),
                           "heat_weight":round(severity/max(len(latest),1),8),"coverage":QUALITY[x.source_quality]})
        return {"as_of":end.isoformat(),"mechanism":mechanism,"points":points,"sample_size":len(points),
                "coverage":round(sum(x["coverage"] for x in points)/max(len(latest),1),3),"stale_after_days":self.policy.stale_after_days,
                "disclaimer":DISCLAIMER,"alternatives":["общий режим","качество измерений","инфраструктурное событие"]}
    def groups(self, level="cluster", **filters):
        if level not in {"cluster","site"}: raise ValueError("level must be cluster or site")
        snap=self.snapshot(**filters); grouped={}
        for p in snap["points"]:
            well=p["well"]; key=(well.get("field") or "",well.get(level) or "Не задано")
            grouped.setdefault(key,[]).append(p)
        result=[]
        for (field_name,name),rows in sorted(grouped.items()):
            vals=[x["selected_severity"] for x in rows]; mechs={}
            losses={}
            for x in rows:
                for m,v in x["severities"].items():
                    if v is not None: mechs[m]=mechs.get(m,0)+v
                if x["economic_loss"] is not None:
                    key=(x["economic_currency"],x["economic_unit"]); losses[key]=losses.get(key,0)+x["economic_loss"]
            result.append({"field":field_name or None,"level":level,"name":name,"count":len(rows),
                "normal":sum(x["status"]=="normal" for x in rows),"growing":sum(x["status"]=="growing" for x in rows),"critical":sum(x["status"]=="critical" for x in rows),
                "avg_risk":round(sum(vals)/len(vals),4),"max_risk":max(vals),"dominant_mechanisms":[k for k,_ in sorted(mechs.items(),key=lambda x:(-x[1],x[0]))[:3]],
                "economic_loss_buckets":[{"currency":k[0],"unit":k[1],"total":round(v,3)} for k,v in sorted(losses.items())],
                "coverage":round(sum(x["coverage"] for x in rows)/len(rows),3),"sample_size":len(rows)})
        return result
    def frames(self, mechanism="integrated", max_frames=None, **filters):
        rows=self.observations(**filters); dates=sorted({x.occurred_at.date() for x in rows}); limit=max_frames or self.policy.max_frames
        if len(dates)>limit:
            indexes=sorted({round(i*(len(dates)-1)/(limit-1)) for i in range(limit)}); dates=[dates[i] for i in indexes]
        return [{"date":d.isoformat(),**self.snapshot(as_of=datetime(d.year,d.month,d.day,23,59,59,tzinfo=timezone.utc),mechanism=mechanism,**filters)} for d in dates]
    def hotspots(self, *, days=None, min_wells=None, distance_km=None, mechanism="integrated", **filters):
        window=days or self.policy.hotspot_window_days; threshold=min_wells or self.policy.hotspot_min_wells; radius=distance_km or self.policy.hotspot_distance_km
        supplied_as_of=filters.pop("as_of",None)
        end=aware(supplied_as_of,"as_of") if supplied_as_of else datetime.now(timezone.utc)
        current=self.snapshot(as_of=end,mechanism=mechanism,date_from=end-timedelta(days=window),**filters)["points"]
        bad=[x for x in current if x["status"] in {"growing","critical"}]; candidates=[]
        for seed in bad:
            members=[x for x in bad if haversine_km(seed["latitude"],seed["longitude"],x["latitude"],x["longitude"])<=radius]
            if len(members)>=threshold:
                # star neighbourhood prevents single-link chain artefacts
                ids=tuple(sorted({x["well"]["canonical_id"] for x in members})); candidates.append((ids,members))
        selected=[]; used=set()
        for ids,members in sorted(candidates,key=lambda x:(-len(x[0]),x[0])):
            unique=[x for x in members if x["well"]["canonical_id"] not in used]
            if len(unique)<threshold: continue
            used.update(x["well"]["canonical_id"] for x in unique); selected.append(unique)
        zones=[]
        for members in selected:
            lat=sum(x["latitude"] for x in members)/len(members); lon=sum(x["longitude"] for x in members)/len(members)
            mechs={m:sum((x["severities"].get(m) or 0) for x in members) for m in MECHANISMS[1:]}
            losses={}
            for x in members:
                if x["economic_loss"] is not None:
                    k=(x["economic_currency"],x["economic_unit"]); losses[k]=losses.get(k,0)+x["economic_loss"]
            first=min(x["occurred_at"] for x in members); last=max(x["occurred_at"] for x in members)
            ids=tuple(sorted(x["well"]["canonical_id"] for x in members)); zones.append({"zone_id":_stable("zone",*ids,end.date()),"centroid":{"latitude":lat,"longitude":lon},
                "radius_km":round(max(haversine_km(lat,lon,x["latitude"],x["longitude"]) for x in members),3),"member_wells":[x["well"] for x in sorted(members,key=lambda x:x["well"]["canonical_id"])],
                "common_mechanisms":[k for k,v in sorted(mechs.items(),key=lambda x:(-x[1],x[0])) if v>0][:3],"risk_trend":"deteriorating_screening",
                "economic_exposure":[{"currency":k[0],"unit":k[1],"total":round(v,3)} for k,v in sorted(losses.items())],"first_seen":first,"last_seen":last,
                "confidence":"medium" if len(members)>=threshold+2 else "low","coverage":round(sum(x["coverage"] for x in members)/len(members),3),
                "evidence":{"member_count":len(members),"distance_km":radius,"window_days":window,"method":"non-overlapping star neighbourhood"},"disclaimer":DISCLAIMER})
        return sorted(zones,key=lambda x:(-len(x["member_wells"]),x["zone_id"]))
    def spread(self, mechanism="integrated", **filters):
        frames=self.frames(mechanism=mechanism,**filters); centers=[]
        for frame in frames:
            pts=[x for x in frame["points"] if x["selected_severity"]>=self.policy.warn]
            if len(pts)>=2: centers.append((frame["date"],sum(x["latitude"] for x in pts)/len(pts),sum(x["longitude"] for x in pts)/len(pts),len(pts)))
        if len(centers)<3: return {"available":False,"confidence":"low","reason":"need at least 3 frames with 2 wells","disclaimer":DISCLAIMER}
        a,b=centers[0],centers[-1]; days=max(1,(date.fromisoformat(b[0])-date.fromisoformat(a[0])).days); distance=haversine_km(a[1],a[2],b[1],b[2])
        return {"available":True,"mechanism":mechanism,"bearing_deg":round(bearing_deg(a[1],a[2],b[1],b[2]),1),"distance_km":round(distance,3),
                "speed_km_day_range":[round(distance/days*.6,3),round(distance/days*1.4,3)],"frames":len(centers),"confidence":"medium" if len(centers)>=5 else "low","causal":False,"disclaimer":DISCLAIMER}
    def infrastructure_geojson(self): return {"type":"FeatureCollection","features":[x.feature() for x in self.repository.list_assets()]}


def observation_from_diagnosed(item: Any, occurred_at: datetime, *, role="unknown", field_name=None, reservoir=None, source="diagnosis"):
    case=item.case; identity=WellIdentity(case.name,field_name,getattr(case,"cluster",None),getattr(case,"site",None),reservoir)
    return RiskObservation(identity,occurred_at,case.latitude,case.longitude,item.diagnosis.severity,item.diagnosis.integrated_risk,role,
                           source=source,source_quality="good" if item.diagnosis.quality.production_ready else "questionable",
                           provenance={"policy_id":item.diagnosis.policy_id,"policy_version":item.diagnosis.policy_version},
                           source_record_id=_stable("diagnosis",identity.canonical_id,occurred_at.isoformat(),item.diagnosis.integrated_risk))


def assets_from_geojson(payload: dict[str,Any], *, source="user_geojson"):
    if payload.get("type")!="FeatureCollection" or not isinstance(payload.get("features"),list): raise ValueError("GeoJSON must be a FeatureCollection")
    rows=[]
    for feature in payload["features"]:
        if feature.get("type")!="Feature" or not isinstance(feature.get("geometry"),dict): raise ValueError("each item must be a GeoJSON Feature")
        p=feature.get("properties") or {}; g=feature["geometry"]
        rows.append(InfrastructureAsset(str(feature.get("id") or p.get("asset_id") or ""),p.get("name"),p.get("asset_type") or p.get("type"),p.get("status","unknown"),g.get("type"),g.get("coordinates"),
            p.get("diameter"),p.get("diameter_unit"),p.get("capacity"),p.get("capacity_unit"),source,{"import":"geojson"},bool(p.get("synthetic",False))))
    return tuple(rows)


def risk_csv_template():
    return "well,occurred_at,latitude,longitude,field,cluster,site,reservoir,well_role,integrated_risk,halite,calcite,wax,corrosion,watercut,equipment,economic_loss,economic_currency,economic_unit,source,source_record_id,source_quality\n".encode()

__all__=["RiskObservation","InfrastructureAsset","SmartMapPolicy","SmartMapRepository","SmartMapService","SmartMapStorageError","SmartMapConflictError","SmartMapNotFoundError","DISCLAIMER","MECHANISMS","risk_status","observation_from_diagnosed","assets_from_geojson","risk_csv_template"]
