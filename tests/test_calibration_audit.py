from datetime import date

from galit.calibration.audit import audit_csv, write_markdown


def test_audit_is_conservative_about_mappings_targets_and_charge_balance(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "well_id,date,depth_m,flow_rate_m3h,ph,iron_mg_l,hardness_meq_l,tds_mg_l\n"
        "A,2026-01-01,100,4,7.0,1,10,700\n"
        "A,2026-01-01,100,,7.0,1,10,700\n",
        encoding="utf-8",
    )
    audit = audit_csv(source, today=date(2026, 8, 19))
    assert audit["rows"] == 2
    assert audit["wells"] == 1
    assert audit["missing"]["flow_rate_m3h"] == 1
    assert audit["duplicate_well_date_records"] == 1
    assert not audit["charge_balance"]["calculable"]
    assert not audit["compatibility"]["physical_calibration"]
    assert not audit["compatibility"]["risk_calibration"]
    blockers = " ".join(audit["compatibility"]["blockers"])
    assert "target_temperature_c" in blockers
    assert "risk_label" in blockers
    assert "q_oil_m3d" in blockers
    report = write_markdown(audit, tmp_path / "audit.md")
    assert "No mapping, target, ion" in report.read_text(encoding="utf-8")


def test_audit_flags_future_dates_and_exact_duplicates(tmp_path):
    source = tmp_path / "source.csv"
    row = "A,2027-01-01,100,4,7,1,10,700\n"
    source.write_text("well_id,date,depth_m,flow_rate_m3h,ph,iron_mg_l,hardness_meq_l,tds_mg_l\n" + row + row, encoding="utf-8")
    audit = audit_csv(source, today=date(2026, 8, 19))
    assert audit["dates"]["future_count"] == 2
    assert audit["exact_duplicate_rows"] == 1
