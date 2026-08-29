from __future__ import annotations

from dataclasses import replace

import galit


def _item(case, risk):
    result = galit.diagnose(case)
    result.integrated_risk = risk
    return galit.DiagnosedWell(case, result)


def test_map_status_thresholds_match_shared_policy():
    assert galit.map_status(0.0) == "green"
    assert galit.map_status(galit.DEFAULT_RISK_POLICY.risk_warn) == "yellow"
    assert galit.map_status(galit.DEFAULT_RISK_POLICY.risk_critical) == "red"


def test_prepare_field_map_skips_bad_coordinates_and_scales_losses():
    base = galit.synthetic.make_fund(3, seed=41)
    good_small = replace(base[0], latitude=52.3, longitude=30.2)
    good_large = replace(base[1], latitude=52.4, longitude=30.3,
                         rate=replace(base[1].rate, q_oil_m3d=100.0))
    missing = replace(base[2], latitude=None, longitude=None)
    data = galit.prepare_field_map([
        _item(good_small, .1), _item(good_large, .7), _item(missing, .4),
    ])
    assert data.summary.total_wells == 3
    assert data.summary.mapped_wells == 2
    assert data.summary.missing_coordinates == 1
    assert [point.status_label for point in data.points] == ["норма", "критический"]
    assert data.points[1].marker_size > data.points[0].marker_size
    assert data.summary.possible_oil_loss_m3d is not None


def test_invalid_coordinates_do_not_break_other_points():
    cases = galit.synthetic.make_fund(2, seed=42)
    invalid = replace(cases[0], latitude=91, longitude=30)
    valid = replace(cases[1], latitude=52, longitude=30)
    data = galit.prepare_field_map([_item(invalid, .2), _item(valid, .4)])
    assert data.summary.invalid_coordinates == 1
    assert [point.well for point in data.points] == [valid.name]
