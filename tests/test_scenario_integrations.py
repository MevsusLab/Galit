import copy

from fastapi.testclient import TestClient

from api import app
import telegram_bot as tb


client = TestClient(app)
WELL = {
    "name": "Well A", "geometry": {"depth_m": 3200, "tubing_id_m": .062},
    "rate": {"q_oil_m3d": 8, "q_water_m3d": 72, "gor_m3m3": 65},
    "fluid": {"gamma_oil": .86, "gamma_gas": .78, "salinity_ppm": 290000},
    "thermal": {"t_surface_c": 8, "geothermal_grad": .033, "u_to": 15, "production_days": 400},
    "water": {"ions_mg_l": {"Na": 95000, "Cl": 205000, "Ca": 28000,
        "Mg": 3100, "K": 1800, "HCO3": 130, "SO4": 250},
        "ph": 6, "t_c": 40, "p_pa": 5000000},
    "wax": {"wat_stock_tank_c": 34, "wax_content_pct": 6.5},
    "co2_mol_frac": .012, "inhibitor_efficiency": 0,
    "lift_type": "ЭЦН", "p_wellhead_pa": 1400000,
}


def test_scenario_api_happy_partial_and_stable_contract():
    response = client.post("/api/v1/scenarios/compare", json={
        "well": copy.deepcopy(WELL),
        "changes": {"oil_rate_relative_change": -.1, "wash_treatment": True},
    })
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "well", "before", "after", "delta", "economics",
                         "applied_changes", "formulas", "assumptions", "warnings",
                         "missing_inputs", "audit_trail"}
    assert body["status"] == "partial"
    assert body["before"]["forecast_oil_rate_m3_day"] == 8
    assert body["after"]["forecast_oil_rate_m3_day"] == 7.2
    assert body["economics"] is None


def test_scenario_api_full_economics_and_validation():
    payload = {
        "well": copy.deepcopy(WELL),
        "changes": {"inhibitor_dosage_delta_mg_l": 20, "effect_override": {
            "inhibitor_efficiency": .8, "source": "field trial"
        }},
        "economics": {"horizon_days": 30, "event_probability": .4,
            "treatment_efficiency": .7, "product_price_per_m3": 100,
            "operating_loss_per_day": 0, "treatment_cost": 500, "currency": "BYN"},
    }
    response = client.post("/api/v1/scenarios/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "available"
    bad = copy.deepcopy(payload)
    bad["changes"]["oil_rate_delta_m3_day"] = 1
    bad["changes"]["oil_rate_relative_change"] = .1
    assert client.post("/api/v1/scenarios/compare", json=bad).status_code == 400


def test_bot_scenario_parser_format_and_help():
    values, errors = tb.parse_scenario_args([
        "oil_pct=-10", "temperature=2", "wash=yes", "well=<A>"
    ])
    assert errors == [] and values["oil_rate_relative_change"] == -.1
    item = tb._diagnosed("<A>") if hasattr(tb, "_diagnosed") else None
    if item is None:
        params, _ = tb.parse_args(tb.POSITIONAL_KEYS if False else ["3200", "62", "8", "72", "65", "34", "скважина=<A>"])
        case = tb.build_case(params)
        import galit
        item = galit.DiagnosedWell(case, galit.diagnose(case))
    text = "\n".join(tb.format_scenario_messages(item, values))
    assert "<A>" not in text and "&lt;A&gt;" in text
    assert "Риск:" in text and "Дебит нефти:" in text
    assert all(len(chunk) <= tb.TELEGRAM_TEXT_LIMIT for chunk in tb.format_scenario_messages(item, values))
    assert "/scenario" in tb.HELP_TEXT and "/scenario" in tb.START_TEXT


def test_dashboard_scenario_adapter_is_partial_without_economics():
    import dashboard
    case = dashboard.galit.synthetic.make_fund(1, seed=4)[0]
    result = dashboard.scenario_for_dashboard(
        case, dashboard.galit.ScenarioChanges(surface_temperature_delta_c=2)
    )
    assert result.status.value == "partial"
    assert result.before.parameters["t_surface_c"] + 2 == result.after.parameters["t_surface_c"]
