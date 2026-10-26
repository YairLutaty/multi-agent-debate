"""Thread-safe bounded traces for the final multi-round debate loop."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from math import isfinite
from threading import Lock
from typing import Literal, TypeAlias, TypedDict

from multi_agent_debate.termination import (
    MultiRoundModelInvocationError,
    MultiRoundModelResponseError,
    MultiRoundValidationError,
)

MultiRoundStage = Literal[
    "team_a_argument",
    "team_b_argument",
    "termination",
    "neutral_synthesis",
]
TraceTeam = Literal["team_a", "team_b", "system"]
TraceStatus = Literal["success", "error"]
MonotonicClock: TypeAlias = Callable[[], float]

STAGE_ORDER: tuple[MultiRoundStage, ...] = (
    "team_a_argument",
    "team_b_argument",
    "termination",
    "neutral_synthesis",
)
ERROR_CATEGORIES = frozenset(
    {
        "model_invocation_error",
        "model_response_error",
        "validation_error",
        "unexpected_error",
    }
)


class MultiRoundTrace(TypedDict):
    round_number: int
    team: TraceTeam
    stage: MultiRoundStage
    duration_seconds: float
    status: TraceStatus
    error_category: str | None


class MultiRoundTraceError(MultiRoundValidationError):
    """Raised when successful-loop traces are incomplete or malformed."""


class MultiRoundTraceRecorder:
    """Own one lock-protected trace list and return defensive snapshots."""

    def __init__(self) -> None:
        self._traces: list[MultiRoundTrace] = []
        self._lock = Lock()

    def record(self, trace: MultiRoundTrace) -> None:
        validated = validate_multi_round_trace(trace)
        with self._lock:
            self._traces.append(copy.deepcopy(validated))

    def snapshot(self) -> list[MultiRoundTrace]:
        with self._lock:
            return copy.deepcopy(self._traces)


def validate_multi_round_trace(value: object) -> MultiRoundTrace:
    if not isinstance(value, dict) or set(value) != set(
        MultiRoundTrace.__required_keys__
    ):
        raise MultiRoundTraceError("multi-round trace has invalid fields")
    round_number = value["round_number"]
    if type(round_number) is not int or round_number <= 0:
        raise MultiRoundTraceError("trace round number must be positive")
    team = value["team"]
    stage = value["stage"]
    if team not in ("team_a", "team_b", "system") or stage not in STAGE_ORDER:
        raise MultiRoundTraceError("multi-round trace identity is invalid")
    expected_team: TraceTeam = (
        "team_a"
        if stage == "team_a_argument"
        else "team_b" if stage == "team_b_argument" else "system"
    )
    if team != expected_team:
        raise MultiRoundTraceError("trace team and stage disagree")
    duration = _finite_nonnegative(value["duration_seconds"])
    status = value["status"]
    category = value["error_category"]
    if status == "success" and category is None:
        pass
    elif status == "error" and category in ERROR_CATEGORIES:
        pass
    else:
        raise MultiRoundTraceError("trace status and error category disagree")
    return {
        "round_number": round_number,
        "team": team,
        "stage": stage,
        "duration_seconds": duration,
        "status": status,
        "error_category": category,
    }


def canonicalize_successful_traces(
    *,
    recorder: MultiRoundTraceRecorder,
    completed_rounds: int,
) -> list[MultiRoundTrace]:
    """Require exactly three round traces plus one final synthesis trace."""

    if type(completed_rounds) is not int or completed_rounds <= 0:
        raise MultiRoundTraceError("completed round count must be positive")
    if completed_rounds > 3:
        raise MultiRoundTraceError("completed round count exceeds the product maximum")
    expected = [
        (round_number, stage)
        for round_number in range(1, completed_rounds + 1)
        for stage in STAGE_ORDER[:3]
    ]
    expected.append((completed_rounds, "neutral_synthesis"))
    by_identity: dict[tuple[int, MultiRoundStage], MultiRoundTrace] = {}
    for raw in recorder.snapshot():
        trace = validate_multi_round_trace(raw)
        identity = (trace["round_number"], trace["stage"])
        if identity not in expected:
            raise MultiRoundTraceError("unexpected successful-run trace")
        if identity in by_identity:
            raise MultiRoundTraceError("duplicate successful-run trace")
        if trace["status"] != "success" or trace["error_category"] is not None:
            raise MultiRoundTraceError("completed run contains an error trace")
        by_identity[identity] = trace
    if set(by_identity) != set(expected):
        raise MultiRoundTraceError("completed run is missing required traces")
    return [copy.deepcopy(by_identity[identity]) for identity in expected]


def clock_value(clock: MonotonicClock) -> float:
    if not callable(clock):
        raise MultiRoundTraceError("monotonic clock must be callable")
    try:
        value = float(clock())
    except Exception:
        raise MultiRoundTraceError("monotonic clock failed") from None
    if not isfinite(value):
        raise MultiRoundTraceError("monotonic clock returned a nonfinite value")
    return value


def elapsed_seconds(clock: MonotonicClock, started: float) -> float:
    elapsed = clock_value(clock) - started
    return _finite_nonnegative(elapsed)


def bounded_multi_round_error_category(error: BaseException) -> str:
    if isinstance(error, MultiRoundModelInvocationError):
        return "model_invocation_error"
    if isinstance(error, MultiRoundModelResponseError):
        return "model_response_error"
    if isinstance(error, MultiRoundValidationError):
        return "validation_error"
    return "unexpected_error"


def _finite_nonnegative(value: object) -> float:
    if type(value) not in (int, float):
        raise MultiRoundTraceError("trace duration must be numeric")
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise MultiRoundTraceError("trace duration must be finite and nonnegative")
    return resolved
