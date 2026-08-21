"""Static integration-prototype packaging checks."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_excludes_sensitive_artifacts_and_runs_non_root():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "USER galit" in dockerfile and "HEALTHCHECK" in dockerfile
    assert {".env", "data", "reports"} <= set(ignored)
    assert "COPY ." not in dockerfile


def test_sample_payload_is_json_and_has_no_secrets():
    path = ROOT / "examples" / "sample-well.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"].startswith("Synthetic")
    assert not ({"token", "password", "secret"} & set(payload))


def test_local_postman_collection_has_four_safe_requests():
    collection = ROOT / "postman" / "collections" / "GALIT API"
    requests = sorted(collection.glob("*.request.yaml"))
    assert len(requests) == 4
    combined = "\n".join(path.read_text(encoding="utf-8") for path in requests)
    for endpoint in ("/health", "/readiness", "/diagnose", "/diagnose/bulk"):
        assert endpoint in combined
    assert "{{base_url}}" in combined
    assert "token" not in combined.lower()
