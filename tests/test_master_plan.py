"""Контрактные тесты доменного модуля «План мастера на сегодня»."""
from dataclasses import replace
from datetime import date, datetime, timezone
import re

import pytest

from galit import (
    DataQuality,
    DiagnosedWell,
    DiagnosisResult,
    FluidProperties,
    ProductionRate,
    ScenarioInterval,
    ThermalParams,
    UncertaintyResult,
    WaterAnalysis,
    WaxProperties,
    WellCase,
    WellGeometry,
    generate_master_plan,
)


def case(name="A-1", oil=100.0):
    return WellCase(
        name=name,
        geometry=WellGeometry(2000.0, 0.062),
        rate=ProductionRate(oil, 50.0, 60.0),
        fluid=FluidProperties(), thermal=ThermalParams(),
        water=WaterAnalysis(
            {"Na": 1, "Cl": 1, "Ca": 1, "Mg": 1, "HCO3": 1},
            ph=7.0, t_c=60.0, p_pa=8e6,
        ),
        wax=WaxProperties(35.0),
    )


def result(name="A-1", risk=0.5, dominant="wax", ready=True, interval=None):
    uncertainty = None if interval is None else UncertaintyResult(
        samples=20, integrated_risk=ScenarioInterval(*interval)
    )
    return DiagnosisResult(
        well=name, depths=[], temps=[], pressures=[], wat_profile=[],
        integrated_risk=risk, dominant=dominant,
        severity={"halite": .1, "calcite": .1, "wax": .1, "corrosion": .1, dominant: .8},
        quality=DataQuality(
            grade="A" if ready else "D", production_ready=ready,
            reasons=[] if ready else ["wax: нет фактических данных"],
        ),
        warnings=[] if ready else ["Screening only"], uncertainty=uncertainty,
    )


def diagnosed(name="A-1", risk=.5, dominant="wax", **kwargs):
    return DiagnosedWell(case(name, kwargs.pop("oil", 100.0)), result(name, risk, dominant, **kwargs))


def test_sorting_is_priority_then_risk_then_well_and_limit_applies_after_sort():
    plan = generate_master_plan([
        diagnosed("B", .61), diagnosed("C", .8), diagnosed("A", .65), diagnosed("D", .4)
    ], generated_at=date(2026, 1, 2), limit=3)
    assert [task.well for task in plan.tasks] == ["C", "A", "B"]
    assert plan.summary.excluded_by_limit == 1


def test_duplicate_names_ignore_case_and_whitespace_and_keep_highest_risk():
    plan = generate_master_plan([
        diagnosed(" Well  7 ", .4), diagnosed("well 7", .7, "corrosion")
    ], generated_at=date(2026, 1, 2))
    assert plan.summary.diagnosed_wells == 2
    assert plan.summary.unique_wells == 1
    assert len(plan.tasks) == 1
    assert plan.tasks[0].risk == .7


@pytest.mark.parametrize("risk,deadline", [
    (.80, "немедленно"), (.60, "24ч"), (.45, "48ч"), (.35, "72ч"), (.10, "планово")
])
def test_deadline_boundaries(risk, deadline):
    task = generate_master_plan([diagnosed(risk=risk)], generated_at=date(2026, 1, 2)).tasks[0]
    assert task.response_deadline == deadline


@pytest.mark.parametrize("mechanism,phrase", [
    ("halite", "соле"), ("calcite", "карбонат"),
    ("wax", "АСПО"), ("corrosion", "коррозион"),
])
def test_each_dominant_mechanism_has_safe_specific_playbook(mechanism, phrase):
    task = generate_master_plan(
        [diagnosed(dominant=mechanism)], generated_at=date(2026, 1, 2)
    ).tasks[0]
    assert phrase.casefold() in task.recommended_action.casefold()
    assert task.pre_trip_checklist and task.materials and task.equipment
    text = " ".join((task.recommended_action, *task.materials))
    assert not re.search(r"\b\d+(?:[.,]\d+)?\s*(?:мг|г|кг|л)/(?:л|м3|сут)\b", text.casefold())
    assert "утвержд" in text.casefold() or "лаборатор" in text.casefold()


def test_low_risk_filtering_can_be_disabled():
    hidden = generate_master_plan([diagnosed(risk=.09)], generated_at=date(2026, 1, 2))
    shown = generate_master_plan(
        [diagnosed(risk=.09)], generated_at=date(2026, 1, 2), include_low_risk=True
    )
    assert hidden.tasks == () and hidden.summary.filtered_low_risk == 1
    assert len(shown.tasks) == 1


def test_loss_bounds_use_scenario_interval_and_are_bounded_by_rate():
    task = generate_master_plan([
        diagnosed(risk=.5, oil=80.0, interval=(.2, .4, .9))
    ], generated_at=date(2026, 1, 2)).tasks[0]
    loss = task.possible_oil_loss
    assert (loss.lower_m3d, loss.central_m3d, loss.upper_m3d) == (16.0, 32.0, 72.0)
    assert 0 <= loss.lower_m3d <= loss.central_m3d <= loss.upper_m3d <= 80.0
    assert "не прогноз" in loss.limitations


def test_loss_screening_without_uncertainty_has_honest_bounds():
    loss = generate_master_plan(
        [diagnosed(risk=.6, oil=100)], generated_at=date(2026, 1, 2)
    ).tasks[0].possible_oil_loss
    assert (loss.lower_m3d, loss.central_m3d, loss.upper_m3d) == (30.0, 60.0, 90.0)
    assert "screening" in loss.method


def test_poor_quality_blocks_action_and_surfaces_warnings():
    task = generate_master_plan([
        diagnosed(risk=.8, ready=False)
    ], generated_at=date(2026, 1, 2)).tasks[0]
    assert not task.production_ready and not task.safe_to_act
    assert task.response_deadline == "немедленно"
    assert "Не выполнять воздействие" in task.recommended_action
    assert any("production-ready" in warning for warning in task.quality_warnings)


def test_stable_id_is_order_and_date_independent_but_mechanism_sensitive():
    first = generate_master_plan([diagnosed()], generated_at=date(2025, 1, 1)).tasks[0]
    second = generate_master_plan([diagnosed()], generated_at=date(2030, 1, 1)).tasks[0]
    other = generate_master_plan([diagnosed(dominant="halite")], generated_at=date(2025, 1, 1)).tasks[0]
    assert first.id == second.id
    assert first.id != other.id


def test_empty_input_has_zero_summary_and_no_fake_loss_total():
    plan = generate_master_plan([], generated_at=date(2026, 1, 2))
    assert plan.tasks == ()
    assert plan.summary.task_count == 0
    assert plan.summary.possible_oil_loss_central_m3d is None


def test_explicit_date_and_datetime_are_deterministic():
    by_date = generate_master_plan([], generated_at=date(2026, 3, 4))
    stamp = datetime(2026, 3, 4, 12, 30, tzinfo=timezone.utc)
    by_time = generate_master_plan([], generated_at=stamp)
    assert by_date.generated_at == datetime(2026, 3, 4, tzinfo=timezone.utc)
    assert by_date.plan_date == date(2026, 3, 4)
    assert by_time.generated_at == stamp and by_time.plan_date == stamp.date()


def test_case_and_result_names_must_match_after_normalization():
    with pytest.raises(ValueError):
        DiagnosedWell(case("A"), result("B"))


def test_negative_limit_is_rejected_and_zero_limit_is_valid():
    with pytest.raises(ValueError):
        generate_master_plan([], limit=-1)
    plan = generate_master_plan([diagnosed()], generated_at=date(2026, 1, 2), limit=0)
    assert plan.tasks == () and plan.summary.excluded_by_limit == 1
