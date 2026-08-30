from datetime import datetime, timedelta, timezone
import json
import pytest
import galit

NOW=datetime(2026,8,23,tzinfo=timezone.utc)

def obs(name,lat,lon,risk=.7,days=0,field="F",cluster="C",mechanism=.8):
    return galit.RiskObservation(galit.WellIdentity(name,field,cluster,"S","R"),NOW-timedelta(days=days),lat,lon,
        {"wax":mechanism,"halite":.1},risk,"producer",10,"BYN","day","test",f"{name}-{days}","good")

def test_model_timezone_wgs84_identity_and_economics():
    with pytest.raises(ValueError): obs("A",91,30)
    with pytest.raises(ValueError): galit.RiskObservation(galit.WellIdentity("A"),datetime(2026,1,1),52,30,{},.2)
    with pytest.raises(ValueError): galit.RiskObservation(galit.WellIdentity("A"),NOW,52,30,{},.2,economic_loss=1)
    assert galit.WellIdentity("1","F1").canonical_id != galit.WellIdentity("1","F2").canonical_id

def test_repository_idempotent_roundtrip_and_conflict(tmp_path):
    repo=galit.SmartMapRepository(tmp_path/"map.json"); item=obs("A",52,30)
    assert repo.add_observation(item)==repo.add_observation(item)
    assert repo.list_observations()[0].to_dict()==item.to_dict()
    changed=galit.RiskObservation(**{**item.__dict__,"integrated_risk":.8})
    with pytest.raises(galit.SmartMapConflictError): repo.add_observation(changed)
    assert not list(tmp_path.glob("*.tmp"))

def test_asof_staleness_mechanism_heat_and_group_isolation(tmp_path):
    service=galit.SmartMapService(galit.SmartMapRepository(tmp_path/"x.json"),observations=[obs("A",52,30,.9),obs("B",52.01,30.01,.2,field="F2"),obs("OLD",52,30,.9,days=60)])
    snap=service.snapshot(as_of=NOW,mechanism="wax")
    assert {x["well"]["display_name"] for x in snap["points"]}=={"A","B"}
    assert next(x for x in snap["points"] if x["well"]["display_name"]=="B")["selected_severity"]==.8
    assert sum(x["heat_weight"] for x in snap["points"])==pytest.approx(.8)
    assert len(service.groups())==2

def test_hotspot_deterministic_no_double_count_and_spread(tmp_path):
    rows=[]
    for frame in (20,10,0):
        rows += [obs(f"W{i}",52+i*.005+frame*.0001,30+i*.005,.75,frame) for i in range(3)]
    service=galit.SmartMapService(galit.SmartMapRepository(tmp_path/"x.json"),observations=rows)
    zones=service.hotspots(as_of=NOW,days=30,min_wells=3,distance_km=3)
    assert len(zones)==1 and len(zones[0]["member_wells"])==3 and zones==service.hotspots(as_of=NOW,days=30,min_wells=3,distance_km=3)
    assert service.spread()["available"]
    assert not galit.SmartMapService(galit.SmartMapRepository(tmp_path/"y.json"),observations=rows[:3]).spread()["available"]

def test_geojson_validation_and_assets(tmp_path):
    payload={"type":"FeatureCollection","features":[{"type":"Feature","id":"p1","geometry":{"type":"LineString","coordinates":[[30,52],[30.1,52.1]]},"properties":{"name":"USER pipeline","asset_type":"pipeline","status":"active"}},{"type":"Feature","id":"f1","geometry":{"type":"Point","coordinates":[30,52]},"properties":{"name":"USER facility","asset_type":"treatment","status":"unknown"}}]}
    rows=galit.assets_from_geojson(payload); assert len(rows)==2
    repo=galit.SmartMapRepository(tmp_path/"x.json")
    for row in rows: repo.upsert_asset(row)
    assert len(galit.SmartMapService(repo).infrastructure_geojson()["features"])==2
    payload["features"][0]["geometry"]["coordinates"]=[[300,52],[30,52]]
    with pytest.raises(ValueError): galit.assets_from_geojson(payload)
