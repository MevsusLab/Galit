from datetime import datetime, timedelta, timezone
from dataclasses import replace
import pytest
import galit

NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)

def meta(lift="ESP"):
    return galit.EquipmentMetadata(well="W-1", lift_type=lift, runtime_days=100,
        nominal_current_a=100, temperature_limit_c=100, vibration_limit_mm_s=10, load_limit_kn=50)

def esp(**kw):
    values=dict(well="W-1", timestamp=NOW, lift_type="ESP", current_a=100,
        nominal_current_a=100, intake_pressure=5, discharge_pressure=10,
        baseline_intake_pressure=5, baseline_discharge_pressure=10,
        motor_temperature_c=60, temperature_limit_c=100,
        vibration_mm_s=2, vibration_limit_mm_s=10)
    values.update(kw); return galit.TelemetrySnapshot(**values)

def test_esp_normal_warn_critical_and_monotonic_rul():
    normal=galit.forecast_equipment(meta(), [esp()], as_of=NOW)
    warn=galit.forecast_equipment(meta(), [esp(current_a=135, vibration_mm_s=8.8)], as_of=NOW)
    critical=galit.forecast_equipment(meta(), [esp(current_a=160, vibration_mm_s=11)], as_of=NOW)
    assert normal.baseline_failure_risk <= warn.baseline_failure_risk <= critical.baseline_failure_risk
    assert normal.rul_days[1] >= warn.rul_days[1] >= critical.rul_days[1]
    assert critical.urgency == "immediate_engineering_review"

def test_rod_pump_signals_and_unsupported():
    rod=galit.TelemetrySnapshot(well="W-1", timestamp=NOW, lift_type="ШГН", rod_load_kn=55,
        load_limit_kn=50, fillage_fraction=.4, dynamic_level_m=800, baseline_dynamic_level_m=500,
        discharge_pressure=8, baseline_discharge_pressure=6, motor_temperature_c=80,
        temperature_limit_c=100, vibration_mm_s=9, vibration_limit_mm_s=10)
    result=galit.forecast_equipment(meta("ШГН"), [rod], as_of=NOW)
    assert {"rod_load", "fillage", "dynamic_level"} <= {x.code for x in result.causes}
    unsupported=galit.forecast_equipment(galit.EquipmentMetadata("F", "фонтан", runtime_days=1), [], as_of=NOW)
    assert unsupported.status == "not_applicable" and unsupported.baseline_failure_risk is None

def test_complications_sand_missing_deterministic_and_validation():
    base=galit.forecast_equipment(meta(), [esp()], as_of=NOW)
    bad=esp(sand_fraction=1, halite_risk=1, calcite_risk=1, wax_risk=1, corrosion_risk=1)
    one=galit.forecast_equipment(meta(), [bad], as_of=NOW)
    two=galit.forecast_equipment(meta(), [bad], as_of=NOW)
    assert one == two and one.baseline_failure_risk >= base.baseline_failure_risk
    assert {"sand", "halite", "calcite", "wax", "corrosion"} <= {x.code for x in one.causes}
    partial=galit.forecast_equipment(meta(), [galit.TelemetrySnapshot("W-1", NOW, "ESP")], as_of=NOW)
    assert partial.confidence == "low" and partial.missing_fields
    with pytest.raises(ValueError): galit.TelemetrySnapshot("W", NOW.replace(tzinfo=None), "ESP")
    with pytest.raises(ValueError): galit.TelemetrySnapshot("W", NOW, "ESP", intake_pressure=10, discharge_pressure=5)

def test_history_trend_repository_idempotency_conflict_and_csv(tmp_path):
    history=[esp(timestamp=NOW-timedelta(days=2), current_a=80), esp(timestamp=NOW-timedelta(days=1), current_a=90), esp(current_a=110)]
    assert galit.forecast_equipment(meta(), history, as_of=NOW).anomalies["current"] > 0
    repo=galit.EquipmentRepository(tmp_path/"equipment.json")
    repo.upsert_equipment(meta()); item=esp(id="snapshot-1"); repo.ingest(item); repo.ingest(item)
    assert repo.get_equipment("w-1") == meta() and repo.list_telemetry("W-1") == [item]
    with pytest.raises(galit.EquipmentConflictError): repo.ingest(replace(item, current_a=120))
    encoded=galit.telemetry_to_csv([item]); assert galit.telemetry_from_csv(encoded) == [item]
    assert b"timestamp" in galit.telemetry_csv_template()
