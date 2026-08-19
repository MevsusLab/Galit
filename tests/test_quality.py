"""Качество данных, provenance и промышленный режим."""
from dataclasses import replace

import pandas as pd
import pytest

import dashboard
from galit import DataProvenance, DataQualityError, assess_quality, diagnose


def test_legacy_case_is_backward_compatible_and_ready():
    case = dashboard.galit.synthetic.make_fund(1)[0]
    result = diagnose(case)
    assert result.quality.production_ready
    assert result.quality.completeness == 1.0
    assert result.quality.grade == "A"


def test_quality_tracks_default_synthetic_and_missing():
    quality = assess_quality(DataProvenance(
        sources={"co2_mol_frac": "default", "water.ions_mg_l": "synthetic"},
        missing_fields=["wax.wax_content_pct"],
    ))
    assert 0.0 <= quality.completeness < 1.0
    assert not quality.production_ready
    assert "co2_mol_frac" in quality.defaulted_fields
    assert "water.ions_mg_l" in quality.synthetic_fields
    assert "wax.wax_content_pct" in quality.missing_fields
    assert len(quality.reasons) == 3


def test_screening_warns_and_strict_rejects_with_all_reasons():
    case = dashboard.galit.synthetic.make_fund(1)[0]
    case = replace(case, provenance=DataProvenance(sources={
        "water.ions_mg_l": "synthetic",
        "co2_mol_frac": "default",
    }))
    result = diagnose(case)
    assert not result.quality.production_ready
    assert any("Screening" in warning for warning in result.warnings)
    with pytest.raises(DataQualityError) as caught:
        diagnose(case, production_mode=True)
    assert len(caught.value.reasons) == 2


def test_dashboard_marks_defaults_and_typical_brine():
    cases, errors = dashboard.frame_to_cases(pd.DataFrame([{
        "name": "minimal", "depth_m": 3000.0, "tubing_id_m": 0.062,
        "q_oil_m3d": 10.0, "q_water_m3d": 50.0, "gor_m3m3": 60.0,
        "wat_stock_tank_c": 32.0,
    }]))
    assert errors == []
    provenance = cases[0].provenance
    assert provenance.sources["water.ions_mg_l"] == "synthetic"
    assert provenance.sources["co2_mol_frac"] == "default"
    results, strict_errors = dashboard.diagnose_frame(pd.DataFrame([{
        "name": "minimal", "depth_m": 3000.0, "tubing_id_m": 0.062,
        "q_oil_m3d": 10.0, "q_water_m3d": 50.0, "gor_m3m3": 60.0,
        "wat_stock_tank_c": 32.0,
    }]), production_mode=True)
    assert results == []
    assert any("не ранжируется" in error for error in strict_errors)
