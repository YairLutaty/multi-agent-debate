"""Pure deterministic termination policy for bounded debate rounds."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypedDict

from multi_agent_debate.metrics import SHINGLE_SIZE, normalize_text

DEFAULT_MAX_ROUNDS = 2
MAX_SUPPORTED_ROUNDS = 3
DEFAULT_NOVELTY_THRESHOLD = 0.15

TerminationReason = Literal["continue", "low_novelty", "max_rounds"]


class TerminationDecision(TypedDict):
    stop: bool
    reason: TerminationReason
    team_a_novelty: float | None
    team_b_novelty: float | None


class MultiRoundError(RuntimeError):
    """Base error for the final bounded multi-round product."""


class MultiRoundValidationError(MultiRoundError):
    """Raised when multi-round state or policy input is inconsistent."""


class MultiRoundModelInvocationError(MultiRoundError):
    """Raised when an unknown debate-model failure crosses the graph boundary."""


class MultiRoundModelResponseError(MultiRoundError):
    """Raised when a debate model returns an invalid public product."""


def validate_max_rounds(value: object) -> int:
    """Return a strict supported round count."""

    if type(value) is not int or not 1 <= value <= MAX_SUPPORTED_ROUNDS:
        raise MultiRoundValidationError("maximum rounds must be an integer from 1 to 3")
    return value


def validate_novelty_threshold(value: object) -> float:
    """Return a finite non-Boolean threshold in the closed unit interval."""

    if type(value) not in (int, float):
        raise MultiRoundValidationError("novelty threshold must be a finite number")
    resolved = float(value)
    if not isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise MultiRoundValidationError("novelty threshold must be between 0 and 1")
    return resolved


def validate_argument(value: object, *, field_name: str = "argument") -> str:
    """Validate one nonblank argument that contains lexical content."""

    if not isinstance(value, str) or not value.strip() or not normalize_text(value):
        raise MultiRoundValidationError(f"{field_name} must be a substantive string")
    return value.strip()


def calculate_same_team_novelty(previous: str, current: str) -> float:
    """Calculate normalized Jaccard novelty between consecutive team turns.

    Three-token shingles are used when both arguments can produce them. If
    either argument is shorter, both sides use token sets so short products are
    compared in the same feature space.
    """

    previous_tokens = normalize_text(
        validate_argument(previous, field_name="previous argument")
    )
    current_tokens = normalize_text(
        validate_argument(current, field_name="current argument")
    )
    if len(previous_tokens) >= SHINGLE_SIZE and len(current_tokens) >= SHINGLE_SIZE:
        previous_features = _shingles(previous_tokens)
        current_features = _shingles(current_tokens)
    else:
        previous_features = set(previous_tokens)
        current_features = set(current_tokens)

    union = previous_features | current_features
    similarity = len(previous_features & current_features) / len(union)
    novelty = 1.0 - similarity
    if not isfinite(novelty) or not 0.0 <= novelty <= 1.0:
        raise MultiRoundValidationError("calculated novelty is invalid")
    return novelty


def decide_termination(
    *,
    team_a_history: Sequence[str],
    team_b_history: Sequence[str],
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
) -> TerminationDecision:
    """Assess a completed round using only deterministic policy inputs."""

    resolved_max = validate_max_rounds(max_rounds)
    threshold = validate_novelty_threshold(novelty_threshold)
    team_a = _validate_history(team_a_history, "Team A history")
    team_b = _validate_history(team_b_history, "Team B history")
    if len(team_a) != len(team_b):
        raise MultiRoundValidationError("team histories must have equal lengths")
    if not team_a:
        raise MultiRoundValidationError("termination requires a completed round")
    if len(team_a) > resolved_max:
        raise MultiRoundValidationError("completed rounds exceed the configured maximum")

    if len(team_a) == 1:
        if resolved_max == 1:
            return _decision(True, "max_rounds", None, None)
        return _decision(False, "continue", None, None)

    team_a_novelty = calculate_same_team_novelty(team_a[-2], team_a[-1])
    team_b_novelty = calculate_same_team_novelty(team_b[-2], team_b[-1])
    if team_a_novelty <= threshold and team_b_novelty <= threshold:
        return _decision(
            True,
            "low_novelty",
            team_a_novelty,
            team_b_novelty,
        )
    if len(team_a) == resolved_max:
        return _decision(True, "max_rounds", team_a_novelty, team_b_novelty)
    return _decision(False, "continue", team_a_novelty, team_b_novelty)


def validate_termination_decision(value: object) -> TerminationDecision:
    """Strictly validate and copy a termination decision."""

    if not isinstance(value, dict) or set(value) != set(
        TerminationDecision.__required_keys__
    ):
        raise MultiRoundValidationError("termination decision has invalid fields")
    if type(value["stop"]) is not bool:
        raise MultiRoundValidationError("termination stop flag must be Boolean")
    reason = value["reason"]
    if reason not in ("continue", "low_novelty", "max_rounds"):
        raise MultiRoundValidationError("termination reason is invalid")
    if value["stop"] is (reason == "continue"):
        raise MultiRoundValidationError("termination stop flag and reason disagree")

    novelties: list[float | None] = []
    for field in ("team_a_novelty", "team_b_novelty"):
        raw = value[field]
        if raw is None:
            novelties.append(None)
            continue
        if type(raw) not in (int, float):
            raise MultiRoundValidationError("termination novelty must be numeric")
        number = float(raw)
        if not isfinite(number) or not 0.0 <= number <= 1.0:
            raise MultiRoundValidationError("termination novelty must be within [0, 1]")
        novelties.append(number)
    if (novelties[0] is None) != (novelties[1] is None):
        raise MultiRoundValidationError("team novelty values must be present together")
    if reason == "low_novelty" and novelties[0] is None:
        raise MultiRoundValidationError(
            "low-novelty termination requires both novelty values"
        )
    return _decision(
        value["stop"],
        reason,
        novelties[0],
        novelties[1],
    )


def _validate_history(value: object, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MultiRoundValidationError(f"{context} must be a sequence")
    return [
        validate_argument(argument, field_name=f"{context} argument")
        for argument in value
    ]


def _decision(
    stop: bool,
    reason: TerminationReason,
    team_a_novelty: float | None,
    team_b_novelty: float | None,
) -> TerminationDecision:
    return {
        "stop": stop,
        "reason": reason,
        "team_a_novelty": team_a_novelty,
        "team_b_novelty": team_b_novelty,
    }


def _shingles(tokens: Sequence[str]) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + SHINGLE_SIZE])
        for index in range(len(tokens) - SHINGLE_SIZE + 1)
    }
