"""API, dashboard and bot integration tests for risk economics."""
import copy

import pytest
from fastapi.testclient import TestClient

import dashboard
import telegram_bot as tb
from api import app
from tests.test_api import WELL


client = TestClient(app)


def economics_payload():
    well = copy.deepcopy(WELL)
    well["rate"]["q_oil_m3d"] = 1
    return {"well": well, "economics": {
        "event_probability": 1, "horizon_days": 31,
        "treatment_efficiency": 1, "event_downtime_days": 0,
        "treatment_downtime_days": 0, "product_price_per_m3": 1000,
        "operating_loss_per_day": 0, "treatment_cost": 8000, "currency": "BYN",
    }}


def test_api_typed_contract_and_legacy_endpoint_unchanged():
    response = client.post("/api/v1/risk-economics", json=economics_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available" and body["currency"] == "BYN"
    assert body["breakdown"]["potential_avoided_damage"] == 31_000
    assert body["breakdown"]["net_expected_effect"] == 23_000
    assert "formulas" in body and body["forecast_link"]["used"] is False
    assert set(client.post("/api/v1/diagnose", json=WELL).json()) == {
        "integrated_risk", "dominant", "wax_onset_m", "recommendation", "warnings"
    }


def test_api_validation_and_unavailable_money():
    bad = economics_payload()
    bad["economics"]["treatment_cost"] = -1
    assert client.post("/api/v1/risk-economics", json=bad).status_code == 422
    partial = economics_payload()
    for key in ("product_price_per_m3", "operating_loss_per_day", "treatment_cost", "currency"):
        partial["economics"].pop(key)
    body = client.post("/api/v1/risk-economics", json=partial).json()
    assert body["status"] == "partial" and body["breakdown"]["net_expected_effect"] is None


def test_dashboard_adapter_and_optional_parser():
    case = dashboard.galit.synthetic.make_fund(1)[0]
    case.rate.q_oil_m3d = 1
    result = dashboard.risk_economics_for_dashboard(
        case, probability=1, horizon_days=31, efficiency=1,
        event_downtime_days=0, treatment_downtime_days=0,
        price=1000, operating_loss=0, treatment_cost=8000, currency="byn",
    )
    assert result.breakdown.net_expected_effect == 23_000
    assert dashboard.parse_optional_nonnegative("", "x") is None
    with pytest.raises(ValueError):
        dashboard.parse_optional_nonnegative("-1", "x")


def test_bot_explicit_inputs_report_and_missing_data():
    values, errors = tb.parse_economics_args([
        "probability=1", "horizon=31", "efficiency=1", "event_days=0",
        "treatment_days=0", "price=1000", "operating=0", "cost=8000", "currency=BYN",
    ])
    assert errors == []
    item = tb._diagnosed("Economic <well>") if hasattr(tb, "_diagnosed") else None
    if item is None:
        params, _ = tb.parse_args(tb.POSITIONAL_KEYS if False else ["3200", "62", "1", "72", "65", "34"])
        case = tb.build_case(params | {"name": "Economic <well>"})
        item = tb.DiagnosedWell(case, tb.diagnose(case))
    chunks = tb.format_economics_messages(item, values)
    assert all(len(chunk) <= tb.TELEGRAM_TEXT_LIMIT for chunk in chunks)
    text = "\n".join(chunks)
    assert "23 000.00 BYN" in text and "&lt;well&gt;" in text
    _, missing = tb.parse_economics_args(["currency=BYN"])
    assert missing
