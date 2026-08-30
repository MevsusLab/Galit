from datetime import datetime, timedelta, timezone

import pytest

import galit


def stamp(days=0):
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=days)


def event(well, record, days, category="repair", metrics=None):
    return galit.manual_event(
        well=well, occurred_at=stamp(days), category=category,
        event_type=category, title=record, summary="test", source_record_id=record,
        metrics=metrics or {}, recorded_at=stamp(days),
    )


def test_identity_normalization_and_no_unsafe_field_merge(tmp_path):
    assert galit.normalize_well_name("  WELL — 12 ") == "well-12"
    repo = galit.ManualEventRepository(tmp_path / "events.json")
    repo.add(event(galit.WellIdentity("Well 12", "Field A"), "a", 0))
    repo.add(event(galit.WellIdentity(" well   12 ", "Field B"), "b", 0))
    service = galit.build_default_service(manual=repo)
    with pytest.raises(galit.TwinAmbiguousError): service.resolve("WELL 12")
    assert service.resolve("well 12", field="Field A").field == "Field A"


def test_timezone_units_roundtrip_idempotency_and_conflict(tmp_path):
    repo = galit.ManualEventRepository(tmp_path / "events.json")
    identity = galit.WellIdentity("W-1")
    with pytest.raises(ValueError):
        galit.manual_event(well=identity, occurred_at=datetime(2026, 1, 1),
            category="repair", event_type="repair", title="x", summary="x")
    row = event(identity, "repair-1", 0, metrics={"duration": galit.MetricValue(2, "h")})
    assert repo.add(row).event_id == repo.add(row).event_id
    assert repo.list()[0].metrics["duration"].unit == "h"
    changed = galit.manual_event(well=identity, occurred_at=stamp(), category="repair",
        event_type="repair", title="changed", summary="test", source_record_id="repair-1", recorded_at=stamp())
    with pytest.raises(galit.TwinConflictError): repo.add(changed)
    assert not list(tmp_path.glob("*.tmp"))


def test_timeline_filter_pagination_snapshot_and_changes(tmp_path):
    repo = galit.ManualEventRepository(tmp_path / "events.json")
    well = galit.WellIdentity("W-2")
    repo.add(event(well, "before", 0, "production", {"water_cut": galit.MetricValue(.2, "fraction")}))
    repo.add(event(well, "work", 1, "repair"))
    repo.add(event(well, "after", 2, "production", {"water_cut": galit.MetricValue(.92, "fraction")}))
    service = galit.build_default_service(manual=repo)
    first = service.timeline("w-2", limit=2)
    assert len(first["items"]) == 2 and first["next_cursor"] == "2"
    assert len(service.timeline("w-2", categories=["repair"], limit=10)["items"]) == 1
    snapshot = service.snapshot("w-2", as_of=stamp(3))
    assert snapshot.state == "critical" and snapshot.indicators["water_cut"].unit == "fraction"
    explanation = service.changes("w-2")[0]
    assert explanation.confidence == "medium"
    assert "требует подтверждения" in explanation.statement
    assert explanation.alternative_explanations


def test_partial_source_and_csv_validation(tmp_path):
    rows, errors = galit.manual_events_from_csv(
        "source_record_id,well,field,cluster,site,reservoir,occurred_at,category,event_type,title,summary,severity,status,metrics_json\n"
        "lab-1,W-3,F,,,,2026-01-01T00:00:00+00:00,laboratory,water,Lab,Result,info,,{\"ph\":{\"value\":6.2}}\n"
    )
    assert len(rows) == 1 and not errors
    repo = galit.ManualEventRepository(tmp_path / "events.json"); repo.add(rows[0])
    snapshot = galit.build_default_service(manual=repo).snapshot("W-3")
    assert snapshot.latest_laboratory is not None


def test_dashboard_and_bot_helpers_escape_and_annotate():
    import dashboard
    import telegram_bot
    item = event(galit.WellIdentity("<W>"), "repair", 0).to_dict()
    fig = dashboard.fig_twin_timeline([item])
    assert len(fig.data) == 1 and fig.data[0].text[0] == "repair"
    well, days, errors = telegram_bot.parse_twin_command('/timeline "Well 1" 90', allow_days=True)
    assert (well, days, errors) == ("Well 1", 90, [])
    snapshot = galit.TwinSnapshot(galit.WellIdentity("<W>"), stamp(), "normal", 0, (), (), {}, (), None, (), None, {}, (), ())
    assert "&lt;W&gt;" in "".join(telegram_bot.format_twin_snapshot(snapshot))
    assert "/twin" in telegram_bot.HELP_TEXT
