"""Unit tests for ML telemetry anomaly detection (Isolation Forest)."""
from datetime import datetime, timedelta, timezone
import pytest
import galit
from galit.anomaly import (
    detect_anomalies,
    AnomalyReport,
    AnomalyPoint,
    ESP_FEATURES,
    ROD_PUMP_FEATURES,
    MIN_TRAINING_SAMPLES,
)

NOW = datetime(2026, 8, 23, 12, 0, 0, tzinfo=timezone.utc)


def make_esp_telemetry(count: int = 15, *, anomaly_at_end: bool = False) -> list[galit.TelemetrySnapshot]:
    """Генерирует нормальную телеметрию ЭЦН с опциональной предаварийной аномалией в конце."""
    snapshots = []
    for i in range(count):
        t = NOW - timedelta(hours=(count - i))
        # Небольшие нормальные флуктуации
        curr = 98.0 + (i % 3) * 1.5
        vibr = 2.0 + (i % 2) * 0.3
        temp = 62.0 + (i % 4) * 0.8
        p_intake = 5.2 - (i % 3) * 0.1
        p_disch = 10.1 + (i % 2) * 0.2

        if anomaly_at_end and i >= count - 2:
            # Предаварийный режим: резкий рост тока, вибрации, температуры ПЭД при падении давления на приеме
            curr = 145.0 + (i - (count - 2)) * 15.0
            vibr = 11.5 + (i - (count - 2)) * 2.0
            temp = 96.0 + (i - (count - 2)) * 5.0
            p_intake = 2.1 - (i - (count - 2)) * 0.5
            p_disch = 8.5

        snapshots.append(galit.TelemetrySnapshot(
            well="Well-ESP-1",
            timestamp=t,
            lift_type="ESP",
            current_a=curr,
            nominal_current_a=100.0,
            intake_pressure=p_intake,
            discharge_pressure=p_disch,
            motor_temperature_c=temp,
            temperature_limit_c=100.0,
            vibration_mm_s=vibr,
            vibration_limit_mm_s=10.0,
        ))
    return snapshots


def test_esp_normal_telemetry():
    """Нормальная телеметрия без выбросов должна давать статус ok и низкий anomaly score."""
    telemetry = make_esp_telemetry(15, anomaly_at_end=False)
    report = detect_anomalies(telemetry, "ESP", random_state=42)

    assert isinstance(report, AnomalyReport)
    assert report.well == "Well-ESP-1"
    assert report.total_points == 15
    assert report.training_samples == 15
    assert report.status in ("ok", "warning")
    # Последняя точка должна быть нормальной
    assert report.points[-1].level == "normal"
    assert report.latest_anomaly_score is not None
    assert report.latest_anomaly_score < 0.65


def test_esp_pre_failure_anomaly_detected():
    """Предаварийный дрейф токов, вибрации и температуры надежно детектируется ML."""
    telemetry = make_esp_telemetry(20, anomaly_at_end=True)
    report = detect_anomalies(telemetry, "ESP", contamination=0.1, random_state=42)

    assert report.status in ("warning", "critical")
    assert report.anomaly_count >= 1
    assert report.max_anomaly_score > 0.60
    # Последние точки должны иметь повышенный anomaly score
    last_point = report.points[-1]
    assert last_point.anomaly_score > 0.60
    assert last_point.level in ("warning", "critical")
    # Вклад признаков: ток, вибрация или температура должны лидировать
    assert any(
        feat in ("current_a", "vibration_mm_s", "motor_temperature_c")
        for feat in last_point.feature_contributions
    )
    assert report.dominant_anomaly_feature is not None


def test_insufficient_data():
    """Меньше 5 точек возвращает статус insufficient_data."""
    telemetry = make_esp_telemetry(3)
    report = detect_anomalies(telemetry, "ESP")

    assert report.status == "insufficient_data"
    assert report.anomaly_count == 0
    assert len(report.points) == 0


def test_rod_pump_anomaly_detection():
    """ШГН детектирует аномалии нагрузки штанг и динамического уровня."""
    count = 12
    snapshots = []
    for i in range(count):
        t = NOW - timedelta(hours=(count - i))
        load = 35.0 if i < 10 else 65.0  # перегрузка на последних 2 точках
        snapshots.append(galit.TelemetrySnapshot(
            well="Well-ROD-1",
            timestamp=t,
            lift_type="ROD_PUMP",
            rod_load_kn=load,
            load_limit_kn=50.0,
            intake_pressure=4.0,
            discharge_pressure=8.0,
            motor_temperature_c=50.0,
            vibration_mm_s=2.0,
            strokes_per_min=6.0,
        ))

    report = detect_anomalies(snapshots, "ROD_PUMP", contamination=0.15, random_state=42)
    assert report.total_points == count
    assert report.anomaly_count >= 1
    # Проверяем вклад: rod_load_kn должен лидировать в аномалии
    anom_points = [p for p in report.points if p.is_anomaly]
    assert len(anom_points) > 0
    assert "rod_load_kn" in anom_points[-1].feature_contributions


def test_equipment_repository_integration(tmp_path):
    """EquipmentRepository.detect_anomalies бесшовно работает с хранилищем."""
    repo = galit.EquipmentRepository(tmp_path / "equipment.json")
    meta = galit.EquipmentMetadata(
        well="W-ML-1", lift_type="ESP", runtime_days=120,
        nominal_current_a=100.0, temperature_limit_c=100.0,
    )
    repo.upsert_equipment(meta)

    # Пустая телеметрия -> insufficient_data
    empty_rep = repo.detect_anomalies("W-ML-1")
    assert empty_rep.status == "insufficient_data"

    # Загружаем нормальную + аномальную телеметрию
    telemetry = make_esp_telemetry(15, anomaly_at_end=True)
    for sn in telemetry:
        repo.ingest(galit.TelemetrySnapshot(
            well="W-ML-1",
            timestamp=sn.timestamp,
            lift_type="ESP",
            current_a=sn.current_a,
            intake_pressure=sn.intake_pressure,
            discharge_pressure=sn.discharge_pressure,
            motor_temperature_c=sn.motor_temperature_c,
            vibration_mm_s=sn.vibration_mm_s,
        ))

    rep = repo.detect_anomalies("W-ML-1")
    assert rep.status in ("warning", "critical")
    assert rep.total_points == 15

    portfolio = repo.portfolio_anomalies()
    assert "W-ML-1" in portfolio
    assert portfolio["W-ML-1"].total_points == 15


def test_reproducibility():
    """Два запуска с одинаковым seed дают идентичный результат."""
    telemetry = make_esp_telemetry(12, anomaly_at_end=True)
    rep1 = detect_anomalies(telemetry, "ESP", random_state=42)
    rep2 = detect_anomalies(telemetry, "ESP", random_state=42)

    assert rep1.anomaly_count == rep2.anomaly_count
    assert rep1.max_anomaly_score == rep2.max_anomaly_score
    for p1, p2 in zip(rep1.points, rep2.points):
        assert p1.anomaly_score == p2.anomaly_score
        assert p1.is_anomaly == p2.is_anomaly
