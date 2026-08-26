"""Аудируемая экономика риска для одной скважины.

Модуль не содержит цен, валютных курсов или нормативов стоимости. Денежные
показатели появляются только при явных входах в одной валюте.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class RiskEconomicsStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RiskEconomicsInput:
    """Явные входы расчёта; длительности задаются в сутках."""

    event_probability: float | None
    horizon_days: float
    treatment_efficiency: float
    event_downtime_days: float
    treatment_downtime_days: float
    oil_rate_m3_day: float
    product_price_per_m3: float | None
    operating_loss_per_day: float | None
    treatment_cost: float | None
    currency: str | None
    production_loss_fraction: float = 1.0
    probability_source: str = "explicit_input"

    def __post_init__(self) -> None:
        bounded = ("event_probability", "treatment_efficiency", "production_loss_fraction")
        for name in bounded:
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name in ("horizon_days", "event_downtime_days", "treatment_downtime_days",
                     "oil_rate_m3_day", "product_price_per_m3",
                     "operating_loss_per_day", "treatment_cost"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.currency is not None:
            normalized = self.currency.strip().upper()
            if not normalized:
                raise ValueError("currency must be non-empty when supplied")
            if len(normalized) > 12:
                raise ValueError("currency must contain at most 12 characters")
            object.__setattr__(self, "currency", normalized)
        if not self.probability_source.strip():
            raise ValueError("probability_source must be non-empty")


@dataclass(frozen=True)
class RiskEconomicsBreakdown:
    gross_production_at_risk_m3: float
    expected_production_loss_m3: float | None
    expected_production_loss_money: float | None
    expected_event_downtime_cost: float | None
    treatment_downtime_cost: float | None
    recommended_treatment_cost: float | None
    total_treatment_cost: float | None
    expected_damage_without_treatment: float | None
    potential_avoided_damage: float | None
    net_expected_effect: float | None
    roi_ratio: float | None
    payback_ratio: float | None


@dataclass(frozen=True)
class RiskEconomicsResult:
    status: RiskEconomicsStatus
    currency: str | None
    data_sufficient: bool
    missing_inputs: tuple[str, ...]
    assumptions: tuple[str, ...]
    formulas: dict[str, str]
    breakdown: RiskEconomicsBreakdown
    limitations: tuple[str, ...] = field(default_factory=tuple)


FORMULAS = {
    "gross_production_at_risk_m3": "oil_rate_m3_day × horizon_days × production_loss_fraction",
    "expected_production_loss_m3": "gross_production_at_risk_m3 × event_probability",
    "expected_production_loss_money": "expected_production_loss_m3 × product_price_per_m3",
    "expected_event_downtime_cost": "event_probability × event_downtime_days × (oil_rate_m3_day × product_price_per_m3 + operating_loss_per_day)",
    "treatment_downtime_cost": "treatment_downtime_days × (oil_rate_m3_day × product_price_per_m3 + operating_loss_per_day)",
    "expected_damage_without_treatment": "expected_production_loss_money + expected_event_downtime_cost",
    "potential_avoided_damage": "expected_damage_without_treatment × treatment_efficiency",
    "total_treatment_cost": "recommended_treatment_cost + treatment_downtime_cost",
    "net_expected_effect": "potential_avoided_damage − total_treatment_cost",
    "roi_ratio": "net_expected_effect ÷ total_treatment_cost",
    "payback_ratio": "total_treatment_cost ÷ potential_avoided_damage",
}


def calculate_risk_economics(inputs: RiskEconomicsInput) -> RiskEconomicsResult:
    """Рассчитать unit economics без скрытых ставок и смешения валют."""
    gross = inputs.oil_rate_m3_day * inputs.horizon_days * inputs.production_loss_fraction
    probability = inputs.event_probability
    expected_m3 = gross * probability if probability is not None else None

    missing = []
    for name in ("event_probability", "product_price_per_m3", "operating_loss_per_day",
                 "treatment_cost", "currency"):
        if getattr(inputs, name) is None:
            missing.append(name)

    price = inputs.product_price_per_m3
    operating = inputs.operating_loss_per_day
    production_money = expected_m3 * price if expected_m3 is not None and price is not None else None
    event_downtime = (
        probability * inputs.event_downtime_days
        * (inputs.oil_rate_m3_day * price + operating)
        if probability is not None and price is not None and operating is not None else None
    )
    treatment_downtime = (
        inputs.treatment_downtime_days * (inputs.oil_rate_m3_day * price + operating)
        if price is not None and operating is not None else None
    )
    expected_damage = (
        production_money + event_downtime
        if production_money is not None and event_downtime is not None else None
    )
    avoided = expected_damage * inputs.treatment_efficiency if expected_damage is not None else None
    total_treatment = (
        inputs.treatment_cost + treatment_downtime
        if inputs.treatment_cost is not None and treatment_downtime is not None else None
    )
    net = avoided - total_treatment if avoided is not None and total_treatment is not None else None
    roi = net / total_treatment if net is not None and total_treatment not in (None, 0.0) else None
    payback = total_treatment / avoided if total_treatment is not None and avoided not in (None, 0.0) else None

    monetary_values = (production_money, event_downtime, treatment_downtime, total_treatment,
                       expected_damage, avoided, net)
    if not missing:
        status = RiskEconomicsStatus.AVAILABLE
    elif expected_m3 is not None or any(value is not None for value in monetary_values):
        status = RiskEconomicsStatus.PARTIAL
    else:
        status = RiskEconomicsStatus.UNAVAILABLE

    assumptions = (
        f"Вероятность события: {probability:.4g} ({inputs.probability_source})."
        if probability is not None else "Вероятность события не задана; ожидаемые значения недоступны.",
        f"Горизонт: {inputs.horizon_days:g} сут.; доля потери дебита при событии: {inputs.production_loss_fraction:.4g}.",
        f"Эффективность обработки: {inputs.treatment_efficiency:.4g}; считается долей предотвращаемого ущерба.",
        "Дебит, цена и режим считаются постоянными на горизонте; дисконтирование не применяется.",
        "Все денежные входы должны быть выражены в одной указанной валюте; конвертация не выполняется.",
    )
    limitations = (
        "Результат является ожидаемой стоимостью сценария, а не гарантированным денежным потоком.",
        "ROI и payback не выводятся при нулевом знаменателе, чтобы не создавать бесконечность или ложную точность.",
    )
    return RiskEconomicsResult(
        status=status,
        currency=inputs.currency,
        data_sufficient=not missing,
        missing_inputs=tuple(missing),
        assumptions=assumptions,
        formulas=dict(FORMULAS),
        breakdown=RiskEconomicsBreakdown(
            gross_production_at_risk_m3=gross,
            expected_production_loss_m3=expected_m3,
            expected_production_loss_money=production_money,
            expected_event_downtime_cost=event_downtime,
            treatment_downtime_cost=treatment_downtime,
            recommended_treatment_cost=inputs.treatment_cost,
            total_treatment_cost=total_treatment,
            expected_damage_without_treatment=expected_damage,
            potential_avoided_damage=avoided,
            net_expected_effect=net,
            roi_ratio=roi,
            payback_ratio=payback,
        ),
        limitations=limitations,
    )
