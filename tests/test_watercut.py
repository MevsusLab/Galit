from datetime import datetime, timedelta, timezone
import pytest
import galit


def histories(well="P", values=(.2,.21,.22,.35,.5,.6)):
    start=datetime(2026,1,1,tzinfo=timezone.utc); rows=[]
    for i,wc in enumerate(values):
        liquid=100; rows.append(galit.ProductionHistory(well,start+timedelta(days=i*10),liquid*(1-wc),liquid*wc))
    return rows


def test_watercut_units_conflict_and_validation():
    row=galit.ProductionHistory("P",datetime.now(timezone.utc),20,80,water_cut=20,water_cut_unit="percent")
    assert row.water_cut==.8 and "water_cut_conflicts_with_rates" in row.quality_flags
    with pytest.raises(ValueError): galit.ProductionHistory("P",datetime.now(timezone.utc),1,-1)
    with pytest.raises(ValueError): galit.WellMetadata("I","injector",latitude=100)


def test_growing_deterministic_forecast_nonnegative():
    meta=galit.WellMetadata("P","producer",52.4,30.7,reservoir="D3",zone="A",layer="1")
    one=galit.diagnose_watercut(meta,histories()); two=galit.diagnose_watercut(meta,histories())
    assert one==two and one.severity in {"growing","critical"}
    assert one.absolute_change_pp>0 and all(x.low_m3d>=0 for x in one.oil_forecast)


def test_insufficient_and_wrong_role():
    producer=galit.WellMetadata("P","producer")
    assert galit.diagnose_watercut(producer,histories(values=(.2,.3))).status=="insufficient_data"
    with pytest.raises(ValueError): galit.diagnose_watercut(galit.WellMetadata("I","injector"),[])


def test_haversine_reservoir_exclusion_link_dedupe_and_bearing():
    p=galit.WellMetadata("P","producer",52.4,30.7,reservoir="D3",zone="A",layer="1")
    good=galit.WellMetadata("I","injector",52.41,30.71,reservoir="D3",zone="A",layer="1")
    bad=galit.WellMetadata("X","injector",52.405,30.705,reservoir="D2",zone="A",layer="1")
    start=datetime(2025,10,1,tzinfo=timezone.utc)
    inj=[galit.InjectionHistory("I",start+timedelta(days=i*20),100+i*30) for i in range(6)]
    links=galit.build_watercut_links([p,good,bad],histories(),inj)
    assert galit.haversine_km(52.4,30.7,52.41,30.71)>0
    assert len(links)<=1 and all(x.injector!="X" and 0<=x.bearing_deg<360 for x in links)


def test_repository_roundtrip_idempotency_conflict_and_csv(tmp_path):
    repo=galit.WatercutRepository(tmp_path/"wc.json"); meta=galit.WellMetadata("P","producer")
    repo.upsert_metadata(meta); row=histories(values=(.2,))[0]
    assert repo.ingest_production([row])["created"]==1
    assert repo.ingest_production([row])["idempotent"]==1
    with pytest.raises(galit.WatercutConflictError): repo.ingest_production([galit.ProductionHistory("P",row.timestamp,10,90)])
    assert repo.list_metadata()==[meta] and repo.list_production()==[row]
    parsed,errors=galit.production_from_csv(galit.production_csv_template())
    assert parsed and not errors
