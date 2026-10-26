"""Deterministic user-configured price and cost calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Literal, TypedDict, cast

from multi_agent_debate.usage import UsageTotals


class CostError(RuntimeError):
    """Base error for pricing and cost validation."""


class CostValidationError(CostError):
    """Pricing or cost data violates the deterministic contract."""


@dataclass(frozen=True, slots=True)
class Pricing:
    input_cost_per_million: Decimal = Decimal("0")
    output_cost_per_million: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_cost_per_million",
            validate_price(self.input_cost_per_million),
        )
        object.__setattr__(
            self,
            "output_cost_per_million",
            validate_price(self.output_cost_per_million),
        )


class PricingData(TypedDict):
    input_cost_per_million: str
    output_cost_per_million: str
    pricing_basis: Literal["local_zero_default", "user_supplied_rates"]


class CostEstimate(TypedDict):
    status: Literal["available", "unavailable"]
    estimated_cost: str | None
    pricing_basis: Literal["local_zero_default", "user_supplied_rates"]
    input_cost_per_million: str
    output_cost_per_million: str
    unavailable_reason: str | None


def validate_price(value: object) -> Decimal:
    """Validate a finite, nonnegative, non-Boolean numeric rate."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise CostValidationError("price must be a nonnegative finite number")
    if isinstance(value, float) and not isfinite(value):
        raise CostValidationError("price must be a nonnegative finite number")
    try:
        resolved = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise CostValidationError("price must be a nonnegative finite number") from None
    if not resolved.is_finite() or resolved < 0:
        raise CostValidationError("price must be a nonnegative finite number")
    return resolved


def pricing_to_data(pricing: Pricing) -> PricingData:
    resolved = validate_pricing(pricing)
    zero = (
        resolved.input_cost_per_million == 0
        and resolved.output_cost_per_million == 0
    )
    return {
        "input_cost_per_million": decimal_to_string(
            resolved.input_cost_per_million
        ),
        "output_cost_per_million": decimal_to_string(
            resolved.output_cost_per_million
        ),
        "pricing_basis": "local_zero_default" if zero else "user_supplied_rates",
    }


def validate_pricing(value: object) -> Pricing:
    if not isinstance(value, Pricing):
        raise CostValidationError("pricing configuration is invalid")
    return Pricing(
        input_cost_per_million=value.input_cost_per_million,
        output_cost_per_million=value.output_cost_per_million,
    )


def validate_pricing_data(value: object) -> PricingData:
    if not isinstance(value, dict) or set(value) != set(PricingData.__required_keys__):
        raise CostValidationError("pricing data has invalid fields")
    input_rate = decimal_from_string(value["input_cost_per_million"])
    output_rate = decimal_from_string(value["output_cost_per_million"])
    expected = pricing_to_data(Pricing(input_rate, output_rate))
    if value != expected:
        raise CostValidationError("pricing data is inconsistent")
    return expected


def calculate_cost(
    totals: UsageTotals,
    *,
    pricing: Pricing,
) -> CostEstimate:
    """Calculate exact decimal cost, or mark incomplete usage unavailable."""

    resolved_pricing = validate_pricing(pricing)
    pricing_data = pricing_to_data(resolved_pricing)
    # Entries are unavailable here, so validate the totals' primitive shape.
    _validate_totals_shape(totals)
    if not totals["all_usage_available"]:
        return {
            "status": "unavailable",
            "estimated_cost": None,
            **pricing_data,
            "unavailable_reason": "token_usage_unavailable",
        }
    estimate = (
        Decimal(totals["input_tokens"])
        * resolved_pricing.input_cost_per_million
        / Decimal(1_000_000)
        + Decimal(totals["output_tokens"])
        * resolved_pricing.output_cost_per_million
        / Decimal(1_000_000)
    )
    return {
        "status": "available",
        "estimated_cost": decimal_to_string(estimate),
        **pricing_data,
        "unavailable_reason": None,
    }


def validate_cost_estimate(
    value: object,
    *,
    totals: UsageTotals,
    pricing: Pricing,
) -> CostEstimate:
    if not isinstance(value, dict) or set(value) != set(CostEstimate.__required_keys__):
        raise CostValidationError("cost estimate has invalid fields")
    expected = calculate_cost(totals, pricing=pricing)
    if value != expected:
        raise CostValidationError("cost estimate is inconsistent")
    return cast(CostEstimate, expected.copy())


def decimal_to_string(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise CostValidationError("decimal value is invalid")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def decimal_from_string(value: object) -> Decimal:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CostValidationError("serialized decimal is invalid")
    try:
        resolved = Decimal(value)
    except InvalidOperation:
        raise CostValidationError("serialized decimal is invalid") from None
    if not resolved.is_finite() or resolved < 0 or decimal_to_string(resolved) != value:
        raise CostValidationError("serialized decimal is invalid")
    return resolved


def _validate_totals_shape(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(UsageTotals.__required_keys__):
        raise CostValidationError("usage totals are invalid")
    for field in (
        "model_calls",
        "available_calls",
        "unavailable_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        if type(value[field]) is not int or value[field] < 0:
            raise CostValidationError("usage totals are invalid")
    if type(value["all_usage_available"]) is not bool:
        raise CostValidationError("usage totals are invalid")
    if value["total_tokens"] != value["input_tokens"] + value["output_tokens"]:
        raise CostValidationError("usage totals are inconsistent")
    if value["available_calls"] + value["unavailable_calls"] != value["model_calls"]:
        raise CostValidationError("usage totals are inconsistent")
    if value["all_usage_available"] != (value["unavailable_calls"] == 0):
        raise CostValidationError("usage totals are inconsistent")
