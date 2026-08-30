from datetime import date, datetime, timezone
from decimal import Decimal
import json
import pytest
import galit

NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


def product(**kw):
    values = dict(id="p1", name="P", manufacturer="M", hazards=("halite",),
                  density_kg_l=Decimal("1.2"), price_per_kg=Decimal("2.5"), currency="BYN")
    return galit.ChemicalProduct(**(values | kw))


def envelope(**kw):
    values = dict(id="e1", product_id="p1", hazard="halite",
                  points=(galit.ChemicalDoseResponsePoint("0.01", False),
                          galit.ChemicalDoseResponsePoint("0.025", True),
                          galit.ChemicalDoseResponsePoint("0.05", True)),
                  validated=True, validation_reference="Lab report 42", conditions="80 C")
    return galit.ChemicalDoseResponseEnvelope(**(values | kw))


def seeded(tmp_path):
    repo = galit.ChemicalRepository(tmp_path / "chemicals.json")
    repo.put_product(product())
    return repo


def test_evidence_gate_exact_tested_dose_cost_and_zero_oil():
    valid = galit.recommend_products([product()], [envelope()], ["halite"], 100, 10)[0]
    assert valid.dose_kg_m3 == Decimal("0.025")
    assert valid.daily_consumption_kg == Decimal("2.500")
    assert valid.daily_cost == Decimal("6.2500")
    assert valid.cost_per_m3_oil == Decimal("0.62500")
    assert valid.evidence_ids == ("e1",)
    zero = galit.recommend_products([product()], [envelope()], ["halite"], 100, 0)[0]
    assert zero.status == "available" and zero.cost_per_m3_oil is None
    for bad in (envelope(validated=False, validation_reference=None),
                envelope(points=(galit.ChemicalDoseResponsePoint("0.025", False),))):
        assert galit.recommend_products([product()], [bad], ["halite"], 100, 10)[0].status == "unavailable"
    with pytest.raises(ValueError, match="validation_reference"):
        envelope(validation_reference=None)


def test_multi_hazard_requires_explicit_compatibility():
    p = product(hazards=("halite", "calcite"), compatible_with=())
    e2 = envelope(id="e2", hazard="calcite")
    assert galit.recommend_products([p], [envelope(), e2], ["halite", "calcite"], 1, 1)[0].status == "unavailable"
    compatible = product(hazards=("halite", "calcite"), compatible_with=("halite", "calcite"))
    assert galit.recommend_products([compatible], [envelope(), e2], ["halite", "calcite"], 1, 1)[0].status == "available"


def test_conversions_and_density_gate():
    assert galit.convert_quantity("1200", "g", "kg") == Decimal("1.200")
    assert galit.convert_quantity("10", "l", "kg", density_kg_l="1.2") == Decimal("12.0")
    assert galit.convert_quantity("12", "kg", "l", density_kg_l="1.2") == Decimal("1E+1")
    assert galit.dose_to_kg_m3("25", "g/m3") == Decimal("0.025")
    with pytest.raises(ValueError, match="density"):
        galit.convert_quantity(1, "l", "kg")


def test_fefo_expiry_idempotency_revision_reservation_and_persistence(tmp_path):
    repo = seeded(tmp_path)
    early = galit.StockLot("l1", "p1", NOW, date(2026, 9, 1), 10)
    late = galit.StockLot("l2", "p1", NOW, date(2026, 10, 1), 20)
    repo.add_lot(late, idempotency_key="receipt-2")
    repo.add_lot(early, idempotency_key="receipt-1")
    revision = repo.revision
    assert repo.add_lot(early, idempotency_key="receipt-1") == early
    assert repo.revision == revision
    txs = repo.consume("p1", 12, NOW, idempotency_key="use-1", reference="job")
    assert [(x.lot_id, x.quantity_kg) for x in txs] == [("l1", Decimal("10")), ("l2", Decimal("2"))]
    assert repo.consume("p1", 12, NOW, idempotency_key="use-1", reference="job") == txs
    reserved = repo.reserve("p1", 5, date(2026, 9, 10), idempotency_key="r1", now=NOW)
    assert reserved.allocations == (("l2", Decimal("5")),)
    assert repo.stock("p1", as_of=date(2026, 9, 10))["available_kg"] == "13"
    released = repo.release_reservation(reserved.id, revision=1)
    assert released.status == "released" and released.revision == 2
    assert repo.stock("p1", as_of=date(2026, 9, 10))["available_kg"] == "18"
    with pytest.raises(galit.ChemicalConflictError):
        repo.release_reservation(reserved.id, revision=1)
    assert galit.ChemicalRepository(repo.path).list_transactions() == repo.list_transactions()
    assert json.loads(repo.path.read_text())["revision"] == repo.revision


def test_append_only_adjustment_and_conflicts(tmp_path):
    repo = seeded(tmp_path)
    repo.add_lot(galit.StockLot("l1", "p1", NOW, date(2027, 1, 1), 10), idempotency_key="receipt")
    item = galit.StockTransaction("t1", "adjust-1", "p1", "l1", "adjustment", 2, NOW, "count")
    assert repo.append_transaction(item) == item
    assert repo.append_transaction(item) == item
    with pytest.raises(galit.ChemicalConflictError):
        repo.append_transaction(galit.StockTransaction("t2", "adjust-1", "p1", "l1", "expiry", 1, NOW, "x"))
    with pytest.raises(ValueError):
        repo.append_transaction(galit.StockTransaction("t3", "x", "p1", "l1", "consumption", 1, NOW, "x"))


def test_forecast_and_shortage_contracts():
    history = [(date(2026, 8, 21), 3), (date(2026, 8, 23), 6)]
    forecast = galit.deterministic_consumption_forecast(history, horizon_days=10, as_of=date(2026, 8, 23))
    assert forecast["daily_kg"] == "3" and forecast["required_kg"] == "30"
    report = galit.shortage_report(20, forecast["daily_kg"], lead_time_days=5, safety_stock_days=3, as_of=date(2026, 8, 23))
    assert report["risk"] is True and report["shortage_kg"] == "4"
    assert galit.deterministic_consumption_forecast([], horizon_days=1, as_of=date.today())["status"] == "unavailable"
