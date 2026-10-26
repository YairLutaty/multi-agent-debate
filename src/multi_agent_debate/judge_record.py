"""Versioned, cross-validated records for advisory debate judgments."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import TypedDict, cast

from multi_agent_debate.costs import (
    CostEstimate,
    Pricing,
    PricingData,
    calculate_cost,
    decimal_from_string,
    pricing_to_data,
    validate_cost_estimate,
    validate_pricing_data,
)
from multi_agent_debate.judge import (
    DRAW_THRESHOLD,
    JUDGE_VERSION,
    RUBRIC_CRITERIA,
    JudgeAssessment,
    JudgeDebateMaterial,
    JudgeWinner,
    StanceDrift,
    TeamJudgeScores,
    compute_overall_score,
    compute_winner,
    validate_judge_assessment,
    validate_judge_material,
)
from multi_agent_debate.run_record import (
    format_utc_timestamp,
    normalize_implementation_commit,
)
from multi_agent_debate.usage import (
    USAGE_EXTRACTION_VERSION,
    ModelUsageEntry,
    UsageTotals,
    aggregate_usage,
    validate_usage_entry,
    validate_usage_totals,
)

JUDGE_RECORD_VERSION = "1.0"


class JudgeRecordError(RuntimeError):
    """Base bounded judgment-record error."""


class JudgeRecordValidationError(JudgeRecordError):
    """A judgment record violates its exact public contract."""


class JudgeModelConfiguration(TypedDict):
    name: str
    temperature: float
    reasoning: bool
    structured_output_method: str


class JudgeProvenance(TypedDict):
    run_timestamp_utc: str
    implementation_commit: str
    judge_version: str
    model: JudgeModelConfiguration
    pricing: PricingData


class JudgeInputSummary(TypedDict):
    debate_result_sha256: str
    question: str
    team_a_stance: str
    team_b_stance: str
    completed_rounds: int
    evidence_count: int


class JudgeRubric(TypedDict):
    criteria: list[str]
    score_minimum: int
    score_maximum: int
    draw_threshold: int
    advisory_only: bool


class JudgeScores(TypedDict):
    team_a: TeamJudgeScores
    team_b: TeamJudgeScores


class TeamPointGroups(TypedDict):
    team_a: list[str]
    team_b: list[str]


class JudgeUsage(TypedDict):
    usage_extraction_version: str
    entries: list[ModelUsageEntry]
    totals: UsageTotals


class JudgeRecord(TypedDict):
    judge_record_version: str
    provenance: JudgeProvenance
    input_summary: JudgeInputSummary
    rubric: JudgeRubric
    scores: JudgeScores
    winner: JudgeWinner
    confidence: int
    decisive_points: list[str]
    strengths: TeamPointGroups
    weaknesses: TeamPointGroups
    stance_drift: StanceDrift
    explanation: str
    limitations: list[str]
    usage: JudgeUsage
    cost_estimate: CostEstimate


def build_judge_record(
    *,
    assessment: JudgeAssessment,
    debate_material: JudgeDebateMaterial,
    debate_result: object,
    run_timestamp_utc: object,
    implementation_commit: str,
    model_name: str,
    pricing: Pricing,
    usage_entries: list[ModelUsageEntry],
) -> JudgeRecord:
    """Build one advisory record referencing the exact completed debate."""

    validated_assessment = validate_judge_assessment(assessment)
    material = validate_judge_material(debate_material)
    if not isinstance(model_name, str) or not model_name.strip():
        raise JudgeRecordValidationError("Judge model configuration is invalid.")
    try:
        timestamp = format_utc_timestamp(run_timestamp_utc)
        commit = normalize_implementation_commit(implementation_commit)
        entries = [validate_usage_entry(item) for item in usage_entries]
        totals = aggregate_usage(entries)
        price_data = pricing_to_data(pricing)
    except Exception:
        raise JudgeRecordValidationError("Judge provenance is invalid.") from None
    record: JudgeRecord = {
        "judge_record_version": JUDGE_RECORD_VERSION,
        "provenance": {
            "run_timestamp_utc": timestamp,
            "implementation_commit": commit,
            "judge_version": JUDGE_VERSION,
            "model": {
                "name": model_name.strip(),
                "temperature": 0.0,
                "reasoning": False,
                "structured_output_method": "json",
            },
            "pricing": price_data,
        },
        "input_summary": {
            "debate_result_sha256": debate_result_sha256(debate_result),
            "question": material["question"],
            "team_a_stance": material["team_a_stance"],
            "team_b_stance": material["team_b_stance"],
            "completed_rounds": len(material["debate_rounds"]),
            "evidence_count": len(material["consolidated_evidence"]),
        },
        "rubric": {
            "criteria": list(RUBRIC_CRITERIA),
            "score_minimum": 0,
            "score_maximum": 100,
            "draw_threshold": DRAW_THRESHOLD,
            "advisory_only": True,
        },
        "scores": {
            "team_a": copy.deepcopy(validated_assessment["team_a_scores"]),
            "team_b": copy.deepcopy(validated_assessment["team_b_scores"]),
        },
        "winner": validated_assessment["winner"],
        "confidence": validated_assessment["confidence"],
        "decisive_points": validated_assessment["decisive_points"].copy(),
        "strengths": {
            "team_a": validated_assessment["team_a_strengths"].copy(),
            "team_b": validated_assessment["team_b_strengths"].copy(),
        },
        "weaknesses": {
            "team_a": validated_assessment["team_a_weaknesses"].copy(),
            "team_b": validated_assessment["team_b_weaknesses"].copy(),
        },
        "stance_drift": copy.deepcopy(
            validated_assessment["stance_drift_detected"]
        ),
        "explanation": validated_assessment["explanation"],
        "limitations": validated_assessment["limitations"].copy(),
        "usage": {
            "usage_extraction_version": USAGE_EXTRACTION_VERSION,
            "entries": copy.deepcopy(entries),
            "totals": totals,
        },
        "cost_estimate": calculate_cost(totals, pricing=pricing),
    }
    return validate_judge_record(
        record,
        debate_result=debate_result,
        debate_material=material,
    )


def validate_judge_record(
    value: object,
    *,
    debate_result: object,
    debate_material: JudgeDebateMaterial,
) -> JudgeRecord:
    """Cross-validate a record against its debate and return a deep copy."""

    try:
        if not isinstance(value, dict) or set(value) != set(
            JudgeRecord.__required_keys__
        ):
            raise JudgeRecordValidationError("Judgment record has invalid fields.")
        if value["judge_record_version"] != JUDGE_RECORD_VERSION:
            raise JudgeRecordValidationError("Judgment record version is invalid.")
        material = validate_judge_material(debate_material)
        provenance = _provenance(value["provenance"])
        summary = _input_summary(
            value["input_summary"], debate_result=debate_result, material=material
        )
        rubric = _rubric(value["rubric"])
        scores = _scores(value["scores"])
        expected_winner = compute_winner(
            scores["team_a"]["overall_score"], scores["team_b"]["overall_score"]
        )
        if value["winner"] != expected_winner:
            raise JudgeRecordValidationError("Judgment winner is inconsistent.")
        confidence = _score(value["confidence"])
        decisive = _points(value["decisive_points"])
        strengths = _point_groups(value["strengths"])
        weaknesses = _point_groups(value["weaknesses"])
        drift = value["stance_drift"]
        if not isinstance(drift, dict) or set(drift) != {"team_a", "team_b"} or any(
            type(drift[team]) is not bool for team in ("team_a", "team_b")
        ):
            raise JudgeRecordValidationError("Judgment stance drift is invalid.")
        explanation = _bounded_text(value["explanation"], maximum=2000)
        limitations = _points(value["limitations"])
        usage = _usage(value["usage"], model_name=provenance["model"]["name"])
        pricing = Pricing(
            decimal_from_string(
                provenance["pricing"]["input_cost_per_million"]
            ),
            decimal_from_string(
                provenance["pricing"]["output_cost_per_million"]
            ),
        )
        cost = validate_cost_estimate(
            value["cost_estimate"], totals=usage["totals"], pricing=pricing
        )
    except JudgeRecordValidationError:
        raise
    except Exception:
        raise JudgeRecordValidationError("Judgment record is invalid.") from None
    return {
        "judge_record_version": JUDGE_RECORD_VERSION,
        "provenance": provenance,
        "input_summary": summary,
        "rubric": rubric,
        "scores": scores,
        "winner": expected_winner,
        "confidence": confidence,
        "decisive_points": decisive,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "stance_drift": {"team_a": drift["team_a"], "team_b": drift["team_b"]},
        "explanation": explanation,
        "limitations": limitations,
        "usage": usage,
        "cost_estimate": cost,
    }


def debate_result_sha256(value: object) -> str:
    """Hash canonical public JSON without retaining another full debate copy."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        raise JudgeRecordValidationError("Debate result cannot be hashed.") from None
    return hashlib.sha256(payload).hexdigest()


def _provenance(value: object) -> JudgeProvenance:
    if not isinstance(value, dict) or set(value) != set(
        JudgeProvenance.__required_keys__
    ):
        raise JudgeRecordValidationError("Judgment provenance is invalid.")
    raw_timestamp = value["run_timestamp_utc"]
    if not isinstance(raw_timestamp, str) or not raw_timestamp.endswith("Z"):
        raise JudgeRecordValidationError("Judgment timestamp is invalid.")
    try:
        parsed = datetime.fromisoformat(raw_timestamp[:-1] + "+00:00")
    except ValueError:
        raise JudgeRecordValidationError("Judgment timestamp is invalid.") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise JudgeRecordValidationError("Judgment timestamp is invalid.")
    timestamp = format_utc_timestamp(parsed)
    commit = normalize_implementation_commit(value["implementation_commit"])
    if timestamp != value["run_timestamp_utc"] or commit != value["implementation_commit"]:
        raise JudgeRecordValidationError("Judgment provenance is not normalized.")
    if value["judge_version"] != JUDGE_VERSION:
        raise JudgeRecordValidationError("Judge version is invalid.")
    model = value["model"]
    if not isinstance(model, dict) or set(model) != set(
        JudgeModelConfiguration.__required_keys__
    ):
        raise JudgeRecordValidationError("Judge model settings are invalid.")
    if (
        not isinstance(model["name"], str)
        or not model["name"].strip()
        or type(model["temperature"]) is not float
        or model["temperature"] != 0.0
        or model["reasoning"] is not False
        or model["structured_output_method"] != "json"
    ):
        raise JudgeRecordValidationError("Judge model settings are invalid.")
    pricing = validate_pricing_data(value["pricing"])
    return {
        "run_timestamp_utc": timestamp,
        "implementation_commit": commit,
        "judge_version": JUDGE_VERSION,
        "model": {
            "name": model["name"],
            "temperature": 0.0,
            "reasoning": False,
            "structured_output_method": "json",
        },
        "pricing": pricing,
    }


def _input_summary(
    value: object, *, debate_result: object, material: JudgeDebateMaterial
) -> JudgeInputSummary:
    if not isinstance(value, dict) or set(value) != set(
        JudgeInputSummary.__required_keys__
    ):
        raise JudgeRecordValidationError("Judgment input summary is invalid.")
    expected: JudgeInputSummary = {
        "debate_result_sha256": debate_result_sha256(debate_result),
        "question": material["question"],
        "team_a_stance": material["team_a_stance"],
        "team_b_stance": material["team_b_stance"],
        "completed_rounds": len(material["debate_rounds"]),
        "evidence_count": len(material["consolidated_evidence"]),
    }
    if value != expected:
        raise JudgeRecordValidationError("Judgment input summary is inconsistent.")
    return copy.deepcopy(expected)


def _rubric(value: object) -> JudgeRubric:
    expected: JudgeRubric = {
        "criteria": list(RUBRIC_CRITERIA),
        "score_minimum": 0,
        "score_maximum": 100,
        "draw_threshold": DRAW_THRESHOLD,
        "advisory_only": True,
    }
    if not isinstance(value, dict) or value != expected or any(
        type(value[field]) is not expected_type
        for field, expected_type in (
            ("score_minimum", int),
            ("score_maximum", int),
            ("draw_threshold", int),
            ("advisory_only", bool),
        )
    ):
        raise JudgeRecordValidationError("Judgment rubric is invalid.")
    return copy.deepcopy(expected)


def _scores(value: object) -> JudgeScores:
    if not isinstance(value, dict) or set(value) != {"team_a", "team_b"}:
        raise JudgeRecordValidationError("Judgment scores are invalid.")
    return {
        "team_a": _team_scores(value["team_a"]),
        "team_b": _team_scores(value["team_b"]),
    }


def _team_scores(value: object) -> TeamJudgeScores:
    required = {*RUBRIC_CRITERIA, "overall_score"}
    if not isinstance(value, dict) or set(value) != required:
        raise JudgeRecordValidationError("Judgment team scores are invalid.")
    scores = cast(
        TeamJudgeScores, {field: _score(value[field]) for field in required}
    )
    criteria = {field: scores[field] for field in RUBRIC_CRITERIA}
    if scores["overall_score"] != compute_overall_score(criteria):
        raise JudgeRecordValidationError("Judgment overall score is inconsistent.")
    return scores


def _score(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 100:
        raise JudgeRecordValidationError("Judgment score is invalid.")
    return value


def _points(value: object) -> list[str]:
    if type(value) is not list or len(value) > 5:
        raise JudgeRecordValidationError("Judgment point collection is invalid.")
    resolved = [_bounded_text(item, maximum=500) for item in value]
    if len(set(resolved)) != len(resolved):
        raise JudgeRecordValidationError("Judgment points must be distinct.")
    return resolved


def _point_groups(value: object) -> TeamPointGroups:
    if not isinstance(value, dict) or set(value) != {"team_a", "team_b"}:
        raise JudgeRecordValidationError("Judgment team points are invalid.")
    return {"team_a": _points(value["team_a"]), "team_b": _points(value["team_b"])}


def _bounded_text(value: object, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or len(value.split()) < 3
    ):
        raise JudgeRecordValidationError("Judgment text is invalid.")
    return value.strip()


def _usage(value: object, *, model_name: str) -> JudgeUsage:
    if not isinstance(value, dict) or set(value) != set(JudgeUsage.__required_keys__):
        raise JudgeRecordValidationError("Judgment usage is invalid.")
    if value["usage_extraction_version"] != USAGE_EXTRACTION_VERSION:
        raise JudgeRecordValidationError("Judgment usage version is invalid.")
    if type(value["entries"]) is not list or len(value["entries"]) != 1:
        raise JudgeRecordValidationError("Judgment requires one usage entry.")
    entry = validate_usage_entry(value["entries"][0])
    if (
        entry["stage"] != "debate_judge"
        or entry["model"] != model_name
        or entry["variant"] != "ui_judge"
        or entry["team"] != "system"
        or entry["round_number"] is not None
    ):
        raise JudgeRecordValidationError("Judgment usage identity is invalid.")
    totals = validate_usage_totals(value["totals"], entries=[entry])
    return {
        "usage_extraction_version": USAGE_EXTRACTION_VERSION,
        "entries": [copy.deepcopy(entry)],
        "totals": totals,
    }
