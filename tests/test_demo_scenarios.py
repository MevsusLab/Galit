"""Competition demo scenarios must remain honest and reproducible."""
from dataclasses import replace

import galit.demo_scenarios as scenarios
from galit.integrated import DiagnosisResult, WellCase, diagnose


def test_five_archetypes_and_required_labels():
    definitions = scenarios.competition_scenarios()
    assert {item.key for item in definitions} == {
        "halite", "calcite", "wax", "corrosion", "mixed_conflict",
    }
    assert all(isinstance(item.case, WellCase) for item in definitions)
    assert set(scenarios.DEMO_LABELS) == {
        "synthetic", "illustrative", "not field validated",
    }


def test_results_are_deterministic_and_calculated_by_core(monkeypatch):
    calls = 0
    real = scenarios.diagnose

    def counted(case):
        nonlocal calls
        calls += 1
        return real(case)

    monkeypatch.setattr(scenarios, "diagnose", counted)
    first = scenarios.run_competition_scenarios()
    assert calls >= len(first)
    second = scenarios.run_competition_scenarios()
    assert [x.diagnosis.severity for x in first] == [x.diagnosis.severity for x in second]
    assert all(isinstance(x.diagnosis, DiagnosisResult) for x in first)
    for item in first:
        direct = diagnose(item.scenario.case)
        assert item.diagnosis.severity == direct.severity
        assert item.actual_dominant == direct.dominant


def test_clean_archetypes_demonstrate_claimed_mechanism():
    results = {item.scenario.key: item for item in scenarios.run_competition_scenarios()}
    for key in ("halite", "calcite", "wax", "corrosion"):
        assert results[key].actual_dominant == key
    mixed = results["mixed_conflict"]
    assert mixed.scenario.educational_focus == "mixed"
    assert mixed.diagnosis.severity["calcite"] >= 0.5
    assert mixed.diagnosis.severity["corrosion"] >= 0.5
    assert mixed.actual_dominant in mixed.diagnosis.severity


def test_co2_sensitivity_is_monotonic_for_supported_cases():
    results = scenarios.run_competition_scenarios()
    for item in results:
        if item.scenario.key not in {"corrosion", "mixed_conflict"}:
            assert not item.co2_sensitivity
            continue
        rates = [point.corrosion_rate_mm_yr for point in item.co2_sensitivity]
        assert rates == sorted(rates)
        assert all(rate >= 0.0 for rate in rates)


def test_inhibitor_counterfactual_is_safe_and_reduces_corrosion():
    for item in scenarios.run_competition_scenarios():
        if item.counterfactual is None:
            continue
        before = item.counterfactual.before
        after = item.counterfactual.after
        assert 0.0 <= after.severity["corrosion"] <= before.severity["corrosion"] <= 1.0
        assert 0.0 <= after.corrosion["rate_mm_yr"] < before.corrosion["rate_mm_yr"]
        # The prevention action must not silently change unrelated input physics.
        direct = diagnose(replace(item.scenario.case, inhibitor_efficiency=0.90))
        assert after.severity == direct.severity
