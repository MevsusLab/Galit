from datetime import datetime, timezone
from fastapi.testclient import TestClient
import api, galit

def test_smart_map_api_happy_filters_geojson_and_errors(tmp_path,monkeypatch):
    repo=galit.SmartMapRepository(tmp_path/"map.json"); monkeypatch.setattr(api,"SMART_MAP",repo)
    client=TestClient(api.app)
    body={"well":{"display_name":"A","field":"F","cluster":"C"},"occurred_at":"2026-08-23T00:00:00Z","latitude":52,"longitude":30,"severities":{"wax":.8},"integrated_risk":.7,"well_role":"producer","source_record_id":"a1","source_quality":"good"}
    assert client.post("/api/v1/smart-map/observations",json=body).status_code==201
    assert client.post("/api/v1/smart-map/observations",json=body).status_code==201
    assert client.get("/api/v1/smart-map/observations?field=F&limit=1").json()["total"]==1
    assert client.get("/api/v1/smart-map/snapshot?mechanism=wax&as_of=2026-08-23T12:00:00Z").json()["points"][0]["selected_severity"]==.8
    assert client.get("/api/v1/smart-map/groups").status_code==200
    assert client.get("/api/v1/smart-map/hotspots?min_wells=2").status_code==200
    assert client.get("/api/v1/smart-map/frames").status_code==200
    gis={"type":"FeatureCollection","features":[{"type":"Feature","id":"f","geometry":{"type":"Point","coordinates":[30,52]},"properties":{"name":"User","asset_type":"treatment","status":"active"}}]}
    assert client.post("/api/v1/smart-map/infrastructure/import",json=gis).status_code==201
    assert client.get("/api/v1/smart-map/geojson").json()["type"]=="FeatureCollection"
    assert client.delete("/api/v1/smart-map/infrastructure/missing").status_code==404
    body["occurred_at"]="2026-08-23T00:00:00"; assert client.post("/api/v1/smart-map/observations",json=body).status_code==422
