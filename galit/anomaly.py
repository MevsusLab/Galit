"""Локальный ML-детектор предаварийных аномалий телеметрии ЭЦН/ШГН.

Использует Isolation Forest (scikit-learn) для unsupervised обнаружения
аномалий в многомерном потоке телеметрии. Обучается на лету на истории
нормальной работы конкретной скважины — разметка аварий не нужна.

Ключевая идея: ML ловит ФАКТ зарождения аномалии по датчикам,
а физическое ядро ГАЛИТа объясняет её ПРИЧИНУ и глубину.

Всё работает локально, без внешних API и облаков.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Sequence

# Lazy import sklearn — модуль не падает если библиотеки нет,
# а возвращает graceful degradation с предупреждением.
_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

MODEL_VERSION = "galit-anomaly-iforest-1.0"
DISCLAIMER = (
    "ML-детектор аномалий — unsupervised screening-инструмент. "
    "Обучается на лету на истории нормальной работы; разметка аварий не нужна. "
    "Anomaly score не является вероятностью отказа. Требуется инженерная "
    "интерпретация совместно с физическими моделями ГАЛИТа."
)

# Каналы телеметрии, по которым ведётся детекция.
# Порядок важен: индекс в массиве соответствует feature importance.
ESP_FEATURES = (
    "current_a",
    "intake_pressure",
    "discharge_pressure",
    "motor_temperature_c",
    "vibration_mm_s",
)
ROD_PUMP_FEATURES = (
    "rod_load_kn",
    "intake_pressure",
    "discharge_pressure",
    "motor_temperature_c",
    "vibration_mm_s",
    "strokes_per_min",
)

# Минимум наблюдений для обучения Isolation Forest.
# Меньше — модель не обучается, возвращается insufficient_data.
MIN_TRAINING_SAMPLES = 5

# Доля «нормальных» наблюдений в обучающей выборке.
# contamination=0.05 означает: ~5% точек в обучении считаются аномальными.
DEFAULT_CONTAMINATION = 0.05

# Пороги перевода raw anomaly score в категорию
SCORE_WARN_THRESHOLD = 0.50
SCORE_CRITICAL_THRESHOLD = 0.70


@dataclass(frozen=True)
class AnomalyPoint:
    """Результат ML-детекции для одного телеметрического snapshot."""
    timestamp: datetime
    anomaly_score: float          # 0..1, где 1 = максимальная аномальность
    is_anomaly: bool              # True если модель считает точку аномальной
    level: str                    # "normal" | "warning" | "critical"
    feature_contributions: dict[str, float]  # вклад каждого канала в аномалию
    raw_score: float              # сырой score из Isolation Forest (чем < -0.5, тем нормальнее)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass(frozen=True)
class AnomalyReport:
    """Сводный отчёт ML-аномалий по скважине."""
    well: str
    status: str                   # "ok" | "warning" | "critical" | "insufficient_data" | "unavailable"
    total_points: int
    anomaly_count: int
    anomaly_rate: float           # доля аномальных точек
    max_anomaly_score: float
    latest_anomaly_score: float | None
    trending_up: bool             # True если последние 3+ точки ухудшаются
    points: tuple[AnomalyPoint, ...]
    dominant_anomaly_feature: str | None  # канал с максимальным вкладом в последнюю аномалию
    training_samples: int
    model_version: str = MODEL_VERSION
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["points"] = [p.to_dict() for p in self.points]
        return result


def _extract_feature_matrix(snapshots: Sequence, features: tuple[str, ...]) -> tuple[list[list[float]], list[datetime], list[list[bool]]]:
    """Извлекает матрицу признаков из списка TelemetrySnapshot.

    Returns:
        (matrix, timestamps, mask) где mask[i][j] = True если значение было реальным (не заполненным).
    """
    matrix: list[list[float]] = []
    timestamps: list[datetime] = []
    masks: list[list[bool]] = []

    for snap in snapshots:
        row: list[float] = []
        mask: list[bool] = []
        for feat in features:
            value = getattr(snap, feat, None)
            if value is not None and math.isfinite(value):
                row.append(float(value))
                mask.append(True)
            else:
                row.append(0.0)  # placeholder, будет заменён медианой
                mask.append(False)
        matrix.append(row)
        timestamps.append(snap.timestamp)
        masks.append(mask)

    return matrix, timestamps, masks


def _impute_median(matrix: list[list[float]], masks: list[list[bool]]) -> list[list[float]]:
    """Замена пропущенных значений медианой столбца."""
    if not matrix:
        return matrix
    n_features = len(matrix[0])
    for j in range(n_features):
        real_values = [matrix[i][j] for i in range(len(matrix)) if masks[i][j]]
        if real_values:
            real_values.sort()
            median = real_values[len(real_values) // 2]
        else:
            median = 0.0
        for i in range(len(matrix)):
            if not masks[i][j]:
                matrix[i][j] = median
    return matrix


def _compute_feature_contributions(
    point: list[float],
    baseline_mean: list[float],
    baseline_std: list[float],
    features: tuple[str, ...],
) -> dict[str, float]:
    """Вычисляет вклад каждого канала в аномалию как нормированное отклонение от базовой линии.

    Чем больше Z-score, тем больше канал «виноват» в аномалии.
    """
    contributions: dict[str, float] = {}
    total = 0.0
    raw_scores: list[float] = []

    for i, feat in enumerate(features):
        std = baseline_std[i] if baseline_std[i] > 1e-9 else 1.0
        z = abs(point[i] - baseline_mean[i]) / std
        raw_scores.append(z)
        total += z

    for i, feat in enumerate(features):
        contributions[feat] = round(raw_scores[i] / total, 4) if total > 0 else round(1.0 / len(features), 4)

    return contributions


def _normalize_anomaly_score(raw_score: float) -> float:
    """Преобразует raw score из IsolationForest (-1..0..+1) в шкалу 0..1.

    IsolationForest.decision_function возвращает:
      < 0 — аномалия (чем меньше, тем аномальнее)
      > 0 — нормально
    score_samples возвращает отрицательные значения, аномалии ближе к 0.

    Мы используем decision_function и нормализуем:
    score = 0.5 - decision_function_value  (clamped to 0..1)
    """
    # decision_function: положительные = нормальные, отрицательные = аномалии
    # Типичный диапазон: -0.3 .. +0.3
    normalized = 0.5 - raw_score
    return min(max(normalized, 0.0), 1.0)


def _score_to_level(score: float) -> str:
    if score >= SCORE_CRITICAL_THRESHOLD:
        return "critical"
    if score >= SCORE_WARN_THRESHOLD:
        return "warning"
    return "normal"


def _is_trending_up(scores: list[float], window: int = 3) -> bool:
    """Проверяет, растут ли последние `window` anomaly scores."""
    if len(scores) < window:
        return False
    tail = scores[-window:]
    return all(tail[i] < tail[i + 1] for i in range(len(tail) - 1))


def detect_anomalies(
    snapshots: Sequence,
    lift_type: str,
    *,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = 42,
) -> AnomalyReport:
    """Запускает Isolation Forest на истории телеметрии одной скважины.

    Args:
        snapshots: Отсортированные по времени TelemetrySnapshot одной скважины.
        lift_type: "ESP" или "ROD_PUMP".
        contamination: Ожидаемая доля аномалий в обучающей выборке (0..0.5).
        random_state: Seed для воспроизводимости.

    Returns:
        AnomalyReport с anomaly score для каждой точки и сводкой.
    """
    well = snapshots[0].well if snapshots else "unknown"

    if not _SKLEARN_AVAILABLE:
        return AnomalyReport(
            well=well, status="unavailable", total_points=0,
            anomaly_count=0, anomaly_rate=0.0, max_anomaly_score=0.0,
            latest_anomaly_score=None, trending_up=False, points=(),
            dominant_anomaly_feature=None, training_samples=0,
        )

    # Выбираем каналы по типу оборудования
    key = lift_type.strip().upper()
    if key in ("ESP", "ЭЦН"):
        features = ESP_FEATURES
    elif key in ("ROD_PUMP", "ШГН", "СШНУ"):
        features = ROD_PUMP_FEATURES
    else:
        return AnomalyReport(
            well=well, status="unavailable", total_points=len(snapshots),
            anomaly_count=0, anomaly_rate=0.0, max_anomaly_score=0.0,
            latest_anomaly_score=None, trending_up=False, points=(),
            dominant_anomaly_feature=None, training_samples=0,
        )

    if len(snapshots) < MIN_TRAINING_SAMPLES:
        return AnomalyReport(
            well=well, status="insufficient_data",
            total_points=len(snapshots), anomaly_count=0,
            anomaly_rate=0.0, max_anomaly_score=0.0,
            latest_anomaly_score=None, trending_up=False, points=(),
            dominant_anomaly_feature=None, training_samples=len(snapshots),
        )

    # Извлекаем матрицу признаков
    matrix, timestamps, masks = _extract_feature_matrix(snapshots, features)
    matrix = _impute_median(matrix, masks)

    X = np.array(matrix, dtype=np.float64)

    # Стандартизация (Z-score нормализация)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучаем Isolation Forest
    n_estimators = min(100, max(10, len(X) * 2))
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,  # локально, без параллелизма для стабильности
    )
    model.fit(X_scaled)

    # Получаем предсказания
    predictions = model.predict(X_scaled)       # +1 = normal, -1 = anomaly
    raw_scores = model.decision_function(X_scaled)  # чем < 0, тем аномальнее

    # Базовые статистики для вычисления feature contributions
    baseline_mean = X.mean(axis=0).tolist()
    baseline_std = X.std(axis=0).tolist()

    # Строим результат по каждой точке
    anomaly_points: list[AnomalyPoint] = []
    anomaly_scores_list: list[float] = []

    for i in range(len(X)):
        raw = float(raw_scores[i])
        score = _normalize_anomaly_score(raw)
        is_anom = int(predictions[i]) == -1
        level = _score_to_level(score)

        contributions = _compute_feature_contributions(
            X[i].tolist(), baseline_mean, baseline_std, features,
        )

        anomaly_points.append(AnomalyPoint(
            timestamp=timestamps[i],
            anomaly_score=round(score, 4),
            is_anomaly=is_anom,
            level=level,
            feature_contributions=contributions,
            raw_score=round(raw, 6),
        ))
        anomaly_scores_list.append(score)

    anomaly_count = sum(1 for p in anomaly_points if p.is_anomaly)
    max_score = max(anomaly_scores_list) if anomaly_scores_list else 0.0
    latest_score = anomaly_scores_list[-1] if anomaly_scores_list else None
    trending = _is_trending_up(anomaly_scores_list)

    # Определяем общий статус
    if any(p.level == "critical" for p in anomaly_points[-3:]):
        status = "critical"
    elif any(p.level == "warning" or p.is_anomaly for p in anomaly_points[-3:]):
        status = "warning"
    else:
        status = "ok"

    # Доминирующий канал в последней аномальной точке
    dominant = None
    anomalous_points = [p for p in anomaly_points if p.is_anomaly]
    if anomalous_points:
        last_anom = anomalous_points[-1]
        dominant = max(last_anom.feature_contributions, key=last_anom.feature_contributions.get)

    return AnomalyReport(
        well=well,
        status=status,
        total_points=len(anomaly_points),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(anomaly_points), 4) if anomaly_points else 0.0,
        max_anomaly_score=round(max_score, 4),
        latest_anomaly_score=round(latest_score, 4) if latest_score is not None else None,
        trending_up=trending,
        points=tuple(anomaly_points),
        dominant_anomaly_feature=dominant,
        training_samples=len(X),
    )


# --------------------------------------------------------------------------- #
# Human-readable labels for features (для UI)
# --------------------------------------------------------------------------- #
FEATURE_LABELS: dict[str, str] = {
    "current_a": "Ток, А",
    "intake_pressure": "Давление приёма",
    "discharge_pressure": "Давление нагнетания",
    "motor_temperature_c": "Температура ПЭД, °C",
    "vibration_mm_s": "Вибрация, мм/с",
    "rod_load_kn": "Нагрузка ШГН, кН",
    "strokes_per_min": "Качаний/мин",
}

FEATURE_PHYSICS_HINTS: dict[str, str] = {
    "current_a": "Рост тока → солевая пробка на приёмной сетке или заклинивание вала",
    "intake_pressure": "Падение давления приёма → снижение притока или пробка",
    "discharge_pressure": "Рост давления нагнетания → сужение НКТ отложениями",
    "motor_temperature_c": "Рост температуры ПЭД → перегрузка или обрыв охлаждения",
    "vibration_mm_s": "Рост вибрации → дисбаланс, износ подшипников, песок",
    "rod_load_kn": "Рост нагрузки → отложения на плунжере или зависание штанг",
    "strokes_per_min": "Изменение частоты → нестабильность привода",
}
