"""Provider-neutral, content-free model usage telemetry."""

from __future__ import annotations

import copy
import math
import threading
from collections.abc import Mapping
from typing import Literal, TypedDict, cast

USAGE_EXTRACTION_VERSION = "1.0"

UsageStatus = Literal["available", "unavailable"]
UsageSource = Literal[
    "usage_metadata",
    "ollama_response_metadata",
    "unavailable",
]

USAGE_STAGES = frozenset(
    {
        "decomposition",
        "research",
        "team_a_argument",
        "team_b_argument",
        "neutral_synthesis",
        "evidence_analysis",
        "argument_draft",
        "review_revision",
        "debate_judge",
    }
)
USAGE_TEAMS = frozenset({"team_a", "team_b", "system"})
_VARIANT_ORDER = {
    "single_round_one_shot": 0,
    "two_round_one_shot": 1,
    "hierarchical_single_round": 2,
    "final_product": 3,
}
_STAGE_ORDER = {
    "decomposition": 0,
    "research": 1,
    "evidence_analysis": 2,
    "argument_draft": 3,
    "review_revision": 4,
    "team_a_argument": 5,
    "team_b_argument": 6,
    "neutral_synthesis": 7,
    "debate_judge": 8,
}


class UsageError(RuntimeError):
    """Base error for bounded usage telemetry."""


class UsageValidationError(UsageError):
    """Usage metadata or a normalized usage record is malformed."""


class NormalizedUsage(TypedDict):
    usage_status: UsageStatus
    usage_source: UsageSource
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    unavailable_reason: str | None


class ModelUsageEntry(NormalizedUsage):
    stage: str
    model: str
    duration_seconds: float
    variant: str | None
    round_number: int | None
    team: str | None


class UsageTotals(TypedDict):
    model_calls: int
    available_calls: int
    unavailable_calls: int
    all_usage_available: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int


def extract_message_usage(message: object) -> NormalizedUsage:
    """Extract trustworthy counts from one AI message without estimating."""

    standard_raw = getattr(message, "usage_metadata", None)
    response_raw = getattr(message, "response_metadata", None)

    standard = None
    if standard_raw is not None:
        standard = _usage_from_standard_metadata(standard_raw)

    response = None
    if response_raw is not None:
        if not isinstance(response_raw, Mapping):
            raise UsageValidationError("response usage metadata must be a mapping")
        prompt_present = "prompt_eval_count" in response_raw
        output_present = "eval_count" in response_raw
        if prompt_present != output_present:
            raise UsageValidationError("Ollama usage metadata is incomplete")
        if prompt_present:
            response = _available_usage(
                response_raw["prompt_eval_count"],
                response_raw["eval_count"],
                source="ollama_response_metadata",
            )

    if standard is not None and response is not None:
        if _token_tuple(standard) != _token_tuple(response):
            raise UsageValidationError("model usage metadata representations disagree")
        return standard
    if standard is not None:
        return standard
    if response is not None:
        return response
    return {
        "usage_status": "unavailable",
        "usage_source": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "unavailable_reason": "metadata_unavailable",
    }


def validate_usage_entry(value: object) -> ModelUsageEntry:
    """Strictly validate one content-free telemetry entry and return a copy."""

    if not isinstance(value, dict) or set(value) != set(
        ModelUsageEntry.__required_keys__
    ):
        raise UsageValidationError("usage entry has invalid fields")
    stage = value["stage"]
    if stage not in USAGE_STAGES:
        raise UsageValidationError("usage stage is invalid")
    model = _bounded_text(value["model"], "usage model", maximum=200)
    duration = _finite_nonnegative(value["duration_seconds"], "usage duration")
    variant = _optional_bounded_text(value["variant"], "usage variant", 100)
    team = value["team"]
    if team is not None and team not in USAGE_TEAMS:
        raise UsageValidationError("usage team is invalid")
    round_number = value["round_number"]
    if round_number is not None and (
        type(round_number) is not int or round_number <= 0
    ):
        raise UsageValidationError("usage round number is invalid")

    status = value["usage_status"]
    source = value["usage_source"]
    unavailable_reason = value["unavailable_reason"]
    tokens = (
        value["input_tokens"],
        value["output_tokens"],
        value["total_tokens"],
    )
    if status == "available":
        if source not in ("usage_metadata", "ollama_response_metadata"):
            raise UsageValidationError("available usage source is invalid")
        input_tokens, output_tokens, total_tokens = _validate_token_tuple(tokens)
        if unavailable_reason is not None:
            raise UsageValidationError("available usage cannot have a reason")
    elif status == "unavailable":
        if source != "unavailable" or tokens != (None, None, None):
            raise UsageValidationError("unavailable usage must not contain tokens")
        if unavailable_reason != "metadata_unavailable":
            raise UsageValidationError("unavailable usage reason is invalid")
        input_tokens = output_tokens = total_tokens = None
    else:
        raise UsageValidationError("usage status is invalid")

    return {
        "stage": cast(str, stage),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "duration_seconds": duration,
        "usage_source": cast(UsageSource, source),
        "usage_status": cast(UsageStatus, status),
        "unavailable_reason": cast(str | None, unavailable_reason),
        "variant": variant,
        "round_number": cast(int | None, round_number),
        "team": cast(str | None, team),
    }


def aggregate_usage(entries: object) -> UsageTotals:
    """Sum only available entries while retaining explicit incompleteness."""

    if type(entries) is not list:
        raise UsageValidationError("usage entries must be a list")
    validated = [validate_usage_entry(entry) for entry in entries]
    available = [
        entry for entry in validated if entry["usage_status"] == "available"
    ]
    return {
        "model_calls": len(validated),
        "available_calls": len(available),
        "unavailable_calls": len(validated) - len(available),
        "all_usage_available": len(available) == len(validated),
        "input_tokens": sum(cast(int, entry["input_tokens"]) for entry in available),
        "output_tokens": sum(
            cast(int, entry["output_tokens"]) for entry in available
        ),
        "total_tokens": sum(cast(int, entry["total_tokens"]) for entry in available),
    }


def validate_usage_totals(value: object, *, entries: list[ModelUsageEntry]) -> UsageTotals:
    if not isinstance(value, dict) or set(value) != set(UsageTotals.__required_keys__):
        raise UsageValidationError("usage totals have invalid fields")
    expected = aggregate_usage(entries)
    if value != expected or any(
        type(value[field]) is not int
        for field in (
            "model_calls",
            "available_calls",
            "unavailable_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
    ) or type(value["all_usage_available"]) is not bool:
        raise UsageValidationError("usage totals are inconsistent")
    return expected.copy()


class UsageRecorder:
    """Thread-safe recorder returning canonical defensive snapshots."""

    def __init__(self) -> None:
        self._entries: list[ModelUsageEntry] = []
        self._lock = threading.Lock()

    def record(self, entry: ModelUsageEntry) -> None:
        validated = validate_usage_entry(entry)
        with self._lock:
            self._entries.append(copy.deepcopy(validated))

    def record_message(
        self,
        *,
        stage: str,
        model: str,
        message: object,
        duration_seconds: float,
        variant: str | None = None,
        round_number: int | None = None,
        team: str | None = None,
    ) -> None:
        normalized = extract_message_usage(message)
        self.record(
            {
                "stage": stage,
                "model": model,
                "duration_seconds": duration_seconds,
                "variant": variant,
                "round_number": round_number,
                "team": team,
                **normalized,
            }
        )

    def snapshot(self) -> list[ModelUsageEntry]:
        with self._lock:
            copied = copy.deepcopy(self._entries)
        return sorted(copied, key=_usage_sort_key)


def _usage_from_standard_metadata(value: object) -> NormalizedUsage:
    if not isinstance(value, Mapping):
        raise UsageValidationError("standard usage metadata must be a mapping")
    required = ("input_tokens", "output_tokens", "total_tokens")
    if any(field not in value for field in required):
        raise UsageValidationError("standard usage metadata is incomplete")
    input_tokens, output_tokens, total_tokens = _validate_token_tuple(
        tuple(value[field] for field in required)
    )
    return {
        "usage_status": "available",
        "usage_source": "usage_metadata",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "unavailable_reason": None,
    }


def _available_usage(
    input_tokens: object,
    output_tokens: object,
    *,
    source: UsageSource,
) -> NormalizedUsage:
    validated_input, validated_output, total = _validate_token_tuple(
        (input_tokens, output_tokens, _strict_token(input_tokens) + _strict_token(output_tokens))
    )
    return {
        "usage_status": "available",
        "usage_source": source,
        "input_tokens": validated_input,
        "output_tokens": validated_output,
        "total_tokens": total,
        "unavailable_reason": None,
    }


def _validate_token_tuple(values: tuple[object, object, object]) -> tuple[int, int, int]:
    input_tokens, output_tokens, total_tokens = (
        _strict_token(value) for value in values
    )
    if total_tokens != input_tokens + output_tokens:
        raise UsageValidationError("usage token total is inconsistent")
    return input_tokens, output_tokens, total_tokens


def _strict_token(value: object) -> int:
    if type(value) is not int or value < 0:
        raise UsageValidationError("usage tokens must be nonnegative integers")
    return value


def _token_tuple(value: NormalizedUsage) -> tuple[int | None, int | None, int | None]:
    return value["input_tokens"], value["output_tokens"], value["total_tokens"]


def _finite_nonnegative(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise UsageValidationError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise UsageValidationError(f"{label} must be finite and nonnegative")
    return resolved


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise UsageValidationError(f"{label} is invalid")
    return value.strip()


def _optional_bounded_text(value: object, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, label, maximum=maximum)


def _usage_sort_key(entry: ModelUsageEntry) -> tuple[object, ...]:
    variant = entry["variant"] or ""
    variant_rank = _VARIANT_ORDER.get(variant, len(_VARIANT_ORDER))
    stage = entry["stage"]
    round_number = entry["round_number"] or 0
    team_rank = {None: 0, "system": 0, "team_a": 1, "team_b": 2}[entry["team"]]
    return (
        variant_rank,
        variant,
        0 if stage == "decomposition" else 2 if stage == "neutral_synthesis" else 1,
        round_number,
        team_rank,
        _STAGE_ORDER[stage],
        entry["model"],
        entry["usage_status"],
        entry["usage_source"],
        entry["input_tokens"] if entry["input_tokens"] is not None else -1,
        entry["output_tokens"] if entry["output_tokens"] is not None else -1,
        entry["total_tokens"] if entry["total_tokens"] is not None else -1,
        entry["duration_seconds"],
    )
