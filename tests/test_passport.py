from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import api
import dashboard
import galit
import telegram_bot as bot

NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


def event(**changes):
    values = dict(well_id="w-1", well_name="Well 1", event_type="rate_change",
                  event_at=NOW, title="Rate", data={"oil_rate_m3d": 10})
    values.update(changes)
    return galit.new_passport_event(now=NOW, **values)


def test_repository_crud_filter_revision_and_schema(tmp_path):
    repo = galit.PassportRepository(tmp_path / "passport.json")
    created = repo.create(event())
    assert repo.list(well="WELL 1")[0] == created
    updated = repo.update(created.edit(title="Updated", now=NOW), expected_revision=1)
    assert updated.revision == 2
    with pytest.raises(galit.PassportConflictError):
        repo.update(updated.edit(title="stale", now=NOW), expected_revision=1)
    assert repo.delete(updated.id, expected_revision=2).id == updated.id
    assert repo.list() == []


def test_validation_timezone_numbers_types_and_attachment_security(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        event(event_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        event(data={"oil_rate_m3d": -1})
    with pytest.raises(ValueError):
        event(event_type="unknown")
    repo = galit.PassportRepository(tmp_path / "p.json")
    with pytest.raises(ValueError, match="MIME"):
        repo.save_attachment("x.exe", "application/octet-stream", b"x")
    attachment = repo.save_attachment("../report.pdf", "application/pdf", b"pdf")
    assert attachment.filename == "report.pdf"
    assert repo.attachment_path(attachment).read_bytes() == b"pdf"
    with pytest.raises(ValueError, match="requires exactly one"):
        event(event_type="lab_report", data={})


def test_summary_timeline_and_dashboard_frames():
    rate = event()
    risk = event(event_type="risk_snapshot", title="Risk", data={"integrated_risk": .5})
    summary = galit.passport_summary([rate, risk])
    assert summary["event_count"] == 2 and summary["latest_rate"]["oil_rate_m3d"] == 10
    assert {row["event_type"] for row in galit.passport_timeline([rate, risk])} == {"risk_snapshot", "rate_change"}
    assert not dashboard.passport_event_frame([rate]).empty
    assert list(dashboard.passport_series([rate], galit.PassportEventType.RATE_CHANGE,
                                          ("oil_rate_m3d",))["oil_rate_m3d"]) == [10]


def test_api_crud_summary_and_attachment(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PASSPORTS", galit.PassportRepository(tmp_path / "api.json"))
    monkeypatch.setattr(api, "TREATMENTS", galit.TreatmentRepository(tmp_path / "treatments.json"))
    client = TestClient(api.app)
    payload = {"well_id": "w-1", "well_name": "Well 1", "event_type": "rate_change",
               "event_at": NOW.isoformat(), "title": "Rate", "data": {"oil_rate_m3d": 10}}
    response = client.post("/api/v1/passport/events", json=payload)
    assert response.status_code == 201
    item = response.json()
    assert client.get("/api/v1/passport/events", params={"well": "w-1"}).json()[0]["id"] == item["id"]
    changed = client.patch(f"/api/v1/passport/events/{item['id']}",
                           json={"revision": 1, "title": "Updated"})
    assert changed.status_code == 200 and changed.json()["revision"] == 2
    aggregate = client.get("/api/v1/passport/w-1").json()
    assert aggregate["summary"]["event_count"] == 1 and aggregate["timeline"][0]["origin"] == "passport"
    upload = client.post("/api/v1/passport/attachments", params={
        "well_id": "w-1", "well_name": "Well 1", "event_type": "lab_report", "title": "Lab"},
        headers={"Content-Type": "application/pdf", "X-File-Name": "lab.pdf"}, content=b"pdf")
    assert upload.status_code == 201 and upload.json()["attachment"]["filename"] == "lab.pdf"
    assert client.delete(f"/api/v1/passport/events/{item['id']}", params={"revision": 2}).status_code == 200


def test_bot_parser_and_safe_formats():
    values, errors = bot.parse_passport_command('/passport_rate well="<Well>" oil=10,5 water=20')
    assert errors == [] and values["oil_rate_m3d"] == "10,5"
    rate = event(well_name="<Well>")
    text = "".join(bot.format_passport_summary("<Well>", [rate], []))
    assert "<Well>" not in text and "&lt;Well&gt;" in text
    assert "/passport_rate" in bot.HELP_TEXT
