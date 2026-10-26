"""Provider-neutral contracts and Ollama adapter for advisory debate judging."""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable, Sequence
from math import isfinite
from typing import Literal, Protocol, TypedDict, cast

from langchain_ollama import ChatOllama

from multi_agent_debate.judge_prompts import build_judge_messages
from multi_agent_debate.ollama_model import DEFAULT_MODEL_NAME
from multi_agent_debate.research import ResearchFinding, validate_research_finding
from multi_agent_debate.usage import UsageRecorder

JUDGE_VERSION = "1.0"
DRAW_THRESHOLD = 3
RUBRIC_CRITERIA = (
    "evidence_use",
    "rebuttal_quality",
    "stance_consistency",
    "clarity",
    "argument_coherence",
)
_SCORE_FIELDS = (*RUBRIC_CRITERIA, "overall_score")
_POINT_LIST_FIELDS = (
    "decisive_points",
    "team_a_strengths",
    "team_b_strengths",
    "team_a_weaknesses",
    "team_b_weaknesses",
    "limitations",
)

JudgeWinner = Literal["team_a", "team_b", "draw"]
JudgeResponseCategory = Literal[
    "missing_field",
    "unknown_field",
    "invalid_score",
    "invalid_confidence",
    "invalid_list",
    "invalid_text",
    "invalid_stance_drift",
    "invalid_json",
    "invalid_structure",
    "invalid_record",
]

_INVALID_STRUCTURED_OUTPUT = "The advisory judge returned invalid structured output."


class JudgeError(RuntimeError):
    """Base bounded error for advisory judging."""


class JudgeInvocationError(JudgeError):
    """The local judge model could not be constructed or invoked."""


class JudgeResponseError(JudgeError):
    """The structured judge response violates the public contract."""

    def __init__(
        self,
        message: str = _INVALID_STRUCTURED_OUTPUT,
        *,
        category: JudgeResponseCategory = "invalid_text",
        validation_path: str | None = None,
        diagnostic_message: str = "Structured judge validation failed.",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.validation_path = validation_path
        self.diagnostic_message = diagnostic_message


class PublicDebateRound(TypedDict):
    round_number: int
    team_a_argument: str
    team_b_argument: str


class JudgeDebateMaterial(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str
    consolidated_evidence: list[ResearchFinding]
    debate_rounds: list[PublicDebateRound]
    final_team_a_argument: str
    final_team_b_argument: str
    neutral_synthesis: str


class TeamJudgeScores(TypedDict):
    evidence_use: int
    rebuttal_quality: int
    stance_consistency: int
    clarity: int
    argument_coherence: int
    overall_score: int


class RawTeamJudgeScores(TypedDict):
    evidence_use: object
    rebuttal_quality: object
    stance_consistency: object
    clarity: object
    argument_coherence: object


class StanceDrift(TypedDict):
    team_a: bool
    team_b: bool


class JudgeAssessment(TypedDict):
    winner: JudgeWinner
    team_a_scores: TeamJudgeScores
    team_b_scores: TeamJudgeScores
    confidence: int
    decisive_points: list[str]
    team_a_strengths: list[str]
    team_b_strengths: list[str]
    team_a_weaknesses: list[str]
    team_b_weaknesses: list[str]
    stance_drift_detected: StanceDrift
    explanation: str
    limitations: list[str]


class RawJudgeAssessment(TypedDict):
    team_a_scores: RawTeamJudgeScores
    team_b_scores: RawTeamJudgeScores
    confidence: object
    decisive_points: list[object]
    team_a_strengths: list[object]
    team_b_strengths: list[object]
    team_a_weaknesses: list[object]
    team_b_weaknesses: list[object]
    stance_drift_detected: StanceDrift
    explanation: object
    limitations: list[object]


class DebateJudge(Protocol):
    def judge(self, material: JudgeDebateMaterial) -> JudgeAssessment:
        """Return one validated advisory judgment."""
        ...


class OllamaDebateJudge:
    """Make exactly one JSON-mode local-model call for a completed debate."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        usage_recorder: UsageRecorder | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        try:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError
            self._model_name = model_name
            self._usage_recorder = usage_recorder
            self._clock = clock
            self._model = ChatOllama(
                model=model_name,
                temperature=0.0,
                reasoning=False,
                format="json",
            )
        except Exception:
            raise JudgeInvocationError(
                "The advisory judge could not be initialized."
            ) from None

    def judge(self, material: JudgeDebateMaterial) -> JudgeAssessment:
        validated_material = validate_judge_material(material)
        reference_data = copy.deepcopy(cast(dict[str, object], validated_material))
        started = self._clock()
        try:
            response = self._model.invoke(build_judge_messages(reference_data))
        except Exception:
            raise JudgeInvocationError(
                "The advisory judge could not complete its assessment."
            ) from None

        if self._usage_recorder is not None:
            try:
                self._usage_recorder.record_message(
                    stage="debate_judge",
                    model=self._model_name,
                    message=response,
                    duration_seconds=self._clock() - started,
                    variant="ui_judge",
                    round_number=None,
                    team="system",
                )
            except Exception:
                raise _response_error(
                    "invalid_record",
                    diagnostic_message="Judge usage metadata could not be recorded.",
                ) from None

        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise _response_error(
                "invalid_structure",
                diagnostic_message="Expected a text JSON response.",
            )
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            raise _response_error(
                "invalid_json",
                diagnostic_message="Expected exactly one valid JSON object.",
            ) from None
        if not isinstance(parsed, dict):
            raise _response_error(
                "invalid_structure",
                diagnostic_message="Expected a JSON object.",
            )
        normalized = normalize_raw_judge_assessment(parsed)
        return validate_judge_assessment(normalized)


def compute_winner(team_a_overall: int, team_b_overall: int) -> JudgeWinner:
    """Apply the fixed, model-independent three-point draw rule."""

    team_a = _strict_score(team_a_overall, "Team A overall score")
    team_b = _strict_score(team_b_overall, "Team B overall score")
    if abs(team_a - team_b) <= DRAW_THRESHOLD:
        return "draw"
    return "team_a" if team_a > team_b else "team_b"


def validate_judge_material(value: object) -> JudgeDebateMaterial:
    """Validate and copy the complete bounded material sent to the judge."""

    if not isinstance(value, dict) or set(value) != set(
        JudgeDebateMaterial.__required_keys__
    ):
        raise JudgeResponseError("Judge input has invalid fields.")
    question = _text(value["question"], "question", maximum=5000)
    team_a_stance = _text(value["team_a_stance"], "Team A stance", maximum=1000)
    team_b_stance = _text(value["team_b_stance"], "Team B stance", maximum=1000)
    raw_evidence = value["consolidated_evidence"]
    if type(raw_evidence) is not list or not 1 <= len(raw_evidence) <= 100:
        raise JudgeResponseError("Judge evidence is invalid.")
    try:
        evidence = [validate_research_finding(item) for item in raw_evidence]
    except Exception:
        raise JudgeResponseError("Judge evidence is invalid.") from None
    raw_rounds = value["debate_rounds"]
    if type(raw_rounds) is not list or not 1 <= len(raw_rounds) <= 3:
        raise JudgeResponseError("Judge debate rounds are invalid.")
    rounds: list[PublicDebateRound] = []
    for expected, raw in enumerate(raw_rounds, start=1):
        if not isinstance(raw, dict) or set(raw) != set(
            PublicDebateRound.__required_keys__
        ):
            raise JudgeResponseError("Judge debate round fields are invalid.")
        if type(raw["round_number"]) is not int or raw["round_number"] != expected:
            raise JudgeResponseError("Judge debate round order is invalid.")
        rounds.append(
            {
                "round_number": expected,
                "team_a_argument": _argument(raw["team_a_argument"]),
                "team_b_argument": _argument(raw["team_b_argument"]),
            }
        )
    final_a = _argument(value["final_team_a_argument"])
    final_b = _argument(value["final_team_b_argument"])
    if final_a != rounds[-1]["team_a_argument"] or final_b != rounds[-1][
        "team_b_argument"
    ]:
        raise JudgeResponseError("Judge final arguments do not match the final round.")
    synthesis = _text(
        value["neutral_synthesis"], "neutral synthesis", maximum=20_000
    )
    return {
        "question": question,
        "team_a_stance": team_a_stance,
        "team_b_stance": team_b_stance,
        "consolidated_evidence": copy.deepcopy(evidence),
        "debate_rounds": copy.deepcopy(rounds),
        "final_team_a_argument": final_a,
        "final_team_b_argument": final_b,
        "neutral_synthesis": synthesis,
    }


def normalize_raw_judge_assessment(value: object) -> JudgeAssessment:
    """Normalize one model assessment and derive all deterministic fields."""

    data = _raw_fields(value, set(RawJudgeAssessment.__required_keys__))
    team_a_scores = _normalize_raw_scores(
        data["team_a_scores"], path="team_a_scores"
    )
    team_b_scores = _normalize_raw_scores(
        data["team_b_scores"], path="team_b_scores"
    )
    confidence = _normalize_raw_score(
        data["confidence"],
        category="invalid_confidence",
        path="confidence",
    )
    drift = _normalize_raw_stance_drift(data["stance_drift_detected"])
    lists = {
        field: _normalize_raw_point_list(data[field], field)
        for field in _POINT_LIST_FIELDS
    }
    explanation = _normalize_explanation(data["explanation"])
    winner = compute_winner(
        team_a_scores["overall_score"], team_b_scores["overall_score"]
    )
    return {
        "winner": winner,
        "team_a_scores": team_a_scores,
        "team_b_scores": team_b_scores,
        "confidence": confidence,
        "decisive_points": lists["decisive_points"],
        "team_a_strengths": lists["team_a_strengths"],
        "team_b_strengths": lists["team_b_strengths"],
        "team_a_weaknesses": lists["team_a_weaknesses"],
        "team_b_weaknesses": lists["team_b_weaknesses"],
        "stance_drift_detected": {
            "team_a": drift["team_a"],
            "team_b": drift["team_b"],
        },
        "explanation": explanation,
        "limitations": lists["limitations"],
    }


def validate_judge_assessment(value: object) -> JudgeAssessment:
    """Strictly validate a canonical, already normalized judgment."""

    if not isinstance(value, dict) or set(value) != set(
        JudgeAssessment.__required_keys__
    ):
        raise JudgeResponseError("The advisory judge response has invalid fields.")
    team_a_scores = _validate_scores(value["team_a_scores"])
    team_b_scores = _validate_scores(value["team_b_scores"])
    confidence = _strict_score(value["confidence"], "judge confidence")
    expected_winner = compute_winner(
        team_a_scores["overall_score"], team_b_scores["overall_score"]
    )
    if value["winner"] != expected_winner:
        raise JudgeResponseError("The advisory judge winner is inconsistent.")
    drift = value["stance_drift_detected"]
    if not isinstance(drift, dict) or set(drift) != {"team_a", "team_b"} or any(
        type(drift[team]) is not bool for team in ("team_a", "team_b")
    ):
        raise JudgeResponseError("The advisory judge stance-drift result is invalid.")
    lists = {
        field: _validate_point_list(value[field], field)
        for field in _POINT_LIST_FIELDS
    }
    explanation = _normalize_explanation(value["explanation"])
    return {
        "winner": expected_winner,
        "team_a_scores": team_a_scores,
        "team_b_scores": team_b_scores,
        "confidence": confidence,
        "decisive_points": lists["decisive_points"],
        "team_a_strengths": lists["team_a_strengths"],
        "team_b_strengths": lists["team_b_strengths"],
        "team_a_weaknesses": lists["team_a_weaknesses"],
        "team_b_weaknesses": lists["team_b_weaknesses"],
        "stance_drift_detected": {
            "team_a": drift["team_a"],
            "team_b": drift["team_b"],
        },
        "explanation": explanation,
        "limitations": lists["limitations"],
    }


def compute_overall_score(value: object) -> int:
    """Return ``round(sum(criteria) / 5)`` using Python's integer rounding."""

    if not isinstance(value, dict) or set(value) != set(RUBRIC_CRITERIA):
        raise JudgeResponseError("The advisory judge score fields are invalid.")
    scores = [_strict_score(value[field], field) for field in RUBRIC_CRITERIA]
    return round(sum(scores) / len(scores))


def _normalize_raw_scores(value: object, *, path: str) -> TeamJudgeScores:
    data = _raw_fields(value, set(RUBRIC_CRITERIA), path=path)
    criteria = {
        field: _normalize_raw_score(
            data[field], category="invalid_score", path=f"{path}.{field}"
        )
        for field in RUBRIC_CRITERIA
    }
    return cast(
        TeamJudgeScores,
        {**criteria, "overall_score": compute_overall_score(criteria)},
    )


def _validate_scores(value: object) -> TeamJudgeScores:
    if not isinstance(value, dict) or set(value) != set(_SCORE_FIELDS):
        raise JudgeResponseError("The advisory judge score fields are invalid.")
    scores = cast(
        TeamJudgeScores,
        {field: _strict_score(value[field], field) for field in _SCORE_FIELDS},
    )
    criteria = {field: scores[field] for field in RUBRIC_CRITERIA}
    if scores["overall_score"] != compute_overall_score(criteria):
        raise JudgeResponseError("The advisory judge overall score is inconsistent.")
    return scores


def _normalize_raw_score(
    value: object,
    *,
    category: Literal["invalid_score", "invalid_confidence"],
    path: str,
) -> int:
    normalized: int
    if type(value) is int:
        normalized = value
    elif type(value) is float and isfinite(value) and value.is_integer():
        normalized = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        normalized = int(value)
    else:
        raise _response_error(
            category,
            validation_path=path,
            diagnostic_message="Expected an integer-compatible value between 0 and 100.",
        )
    if not 0 <= normalized <= 100:
        raise _response_error(
            category,
            validation_path=path,
            diagnostic_message="Expected an integer-compatible value between 0 and 100.",
        )
    return normalized


def _strict_score(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= 100:
        raise JudgeResponseError(f"{label} must be an integer from 0 through 100.")
    return value


def _validate_point_list(value: object, label: str) -> list[str]:
    if type(value) is not list:
        raise _response_error(
            "invalid_list",
            validation_path=label,
            diagnostic_message="Expected a list containing at most five items.",
        )
    copied: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(cast(Sequence[object], value)):
        if not isinstance(item, str):
            raise _response_error(
                "invalid_list",
                validation_path=f"{label}[{index}]",
                diagnostic_message="Expected a text list item.",
            )
        normalized = item.strip()
        if not normalized:
            continue
        if len(normalized) > 500:
            raise _response_error(
                "invalid_list",
                validation_path=f"{label}[{index}]",
                diagnostic_message="Expected a list item containing at most 500 characters.",
            )
        if normalized not in seen:
            copied.append(normalized)
            seen.add(normalized)
    if len(copied) > 5:
        raise _response_error(
            "invalid_list",
            validation_path=label,
            diagnostic_message="Expected at most five nonblank distinct items.",
        )
    return copied


def _normalize_raw_point_list(value: object, label: str) -> list[str]:
    """Accept one raw string without weakening canonical list validation."""

    candidate = [value] if isinstance(value, str) else value
    return _validate_point_list(candidate, label)


def _normalize_raw_stance_drift(value: object) -> StanceDrift:
    """Normalize an unambiguous raw ``false`` without guessing team identity."""

    if type(value) is bool:
        if value is False:
            return {"team_a": False, "team_b": False}
        raise _response_error(
            "invalid_stance_drift",
            validation_path="stance_drift_detected",
            diagnostic_message="A true stance-drift value is ambiguous.",
        )
    drift = _raw_fields(
        value,
        {"team_a", "team_b"},
        path="stance_drift_detected",
    )
    if any(type(drift[team]) is not bool for team in ("team_a", "team_b")):
        invalid_team = next(
            team for team in ("team_a", "team_b") if type(drift[team]) is not bool
        )
        raise _response_error(
            "invalid_stance_drift",
            validation_path=f"stance_drift_detected.{invalid_team}",
            diagnostic_message="Expected a Boolean stance-drift value.",
        )
    return {"team_a": drift["team_a"], "team_b": drift["team_b"]}


def _normalize_explanation(value: object) -> str:
    if not isinstance(value, str):
        raise _response_error(
            "invalid_text",
            validation_path="explanation",
            diagnostic_message="Expected a substantive explanation.",
        )
    normalized = value.strip()
    if len(normalized) > 2000 or len(normalized.split()) < 3:
        raise _response_error(
            "invalid_text",
            validation_path="explanation",
            diagnostic_message="Expected a substantive explanation of at most 2,000 characters.",
        )
    return normalized


def _raw_fields(
    value: object, required: set[str], *, path: str | None = None
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _response_error(
            "invalid_structure",
            validation_path=path,
            diagnostic_message="Expected a structured object.",
        )
    keys = set(value)
    missing = sorted(required - keys)
    if missing:
        field_path = f"{path}.{missing[0]}" if path else missing[0]
        raise _response_error(
            "missing_field",
            validation_path=field_path,
            diagnostic_message="Required field is missing.",
        )
    unknown = sorted(str(key) for key in keys - required)
    if unknown:
        field_path = f"{path}.{unknown[0]}" if path else unknown[0]
        raise _response_error(
            "unknown_field",
            validation_path=field_path,
            diagnostic_message="Unexpected field is not allowed.",
        )
    return cast(dict[str, object], value)


def _response_error(
    category: JudgeResponseCategory,
    *,
    validation_path: str | None = None,
    diagnostic_message: str = "Structured judge validation failed.",
) -> JudgeResponseError:
    return JudgeResponseError(
        category=category,
        validation_path=validation_path,
        diagnostic_message=diagnostic_message,
    )


def _argument(value: object) -> str:
    text = _text(value, "debate argument", maximum=20_000)
    if len(text.split()) < 4:
        raise JudgeResponseError("Judge debate argument must be substantive.")
    return text


def _text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise JudgeResponseError(f"{label} is invalid.")
    return value.strip()
