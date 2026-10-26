"""Validated, presentation-neutral records returned by the local debate UI."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from math import isfinite
from datetime import datetime
from typing import Literal, TypedDict, cast
from urllib.parse import urlsplit

from multi_agent_debate.costs import (
    CostEstimate,
    Pricing,
    PricingData,
    decimal_from_string,
    decimal_to_string,
    validate_cost_estimate,
    validate_pricing_data,
)
from multi_agent_debate.efficiency_report import (
    EfficiencyMeasurements,
    calculate_efficiency,
    count_generated_words,
)
from multi_agent_debate.expanded_graph import validate_decomposition_result
from multi_agent_debate.judge import JudgeDebateMaterial, RUBRIC_CRITERIA
from multi_agent_debate.judge_record import JudgeRecord, validate_judge_record
from multi_agent_debate.metrics import scorecard_to_json
from multi_agent_debate.multi_round_run_record import (
    MultiRoundRunRecordV21,
    validate_multi_round_run_record,
)
from multi_agent_debate.multi_round_state import DebateRound
from multi_agent_debate.research import (
    ResearchFinding,
    consolidate_research_findings,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.run_record import (
    ScorecardData,
    SearchCallEvidence,
    SearchResultEvidence,
    normalize_implementation_commit,
    sanitized_search_settings,
    score_expanded_result,
)
from multi_agent_debate.team_model import TeamWorkProduct, validate_team_work_product
from multi_agent_debate.termination import (
    decide_termination,
    validate_max_rounds,
    validate_novelty_threshold,
    validate_termination_decision,
)
from multi_agent_debate.usage import (
    USAGE_EXTRACTION_VERSION,
    ModelUsageEntry,
    UsageTotals,
    aggregate_usage,
    validate_usage_entry,
    validate_usage_totals,
)

UI_DEBATE_RESULT_VERSION = "1.0"
UI_RESULT_VERSION = "1.0"
UI_MODES = ("single-round", "multi-round", "hierarchical")
JUDGMENT_UNAVAILABLE_NOTICE = "Advisory judgment is unavailable for this run."

UiMode = Literal["single-round", "multi-round", "hierarchical"]
UiJudgmentErrorCategory = Literal[
    "invocation_failed",
    "invalid_json",
    "invalid_structure",
    "invalid_scores",
    "invalid_narrative",
    "invalid_record",
]
UI_JUDGMENT_ERROR_CATEGORIES = (
    "invocation_failed",
    "invalid_json",
    "invalid_structure",
    "invalid_scores",
    "invalid_narrative",
    "invalid_record",
)


class UiResultError(RuntimeError):
    """Base bounded UI-result error."""


class UiResultValidationError(UiResultError):
    """A public debate or downloadable UI result is inconsistent."""


class UiModelSettings(TypedDict):
    name: str
    temperature: float
    reasoning: bool


class UiDebateProvenance(TypedDict):
    run_timestamp_utc: str
    implementation_commit: str
    architecture_version: str
    source_run_record_version: str
    model: UiModelSettings
    search_provider: str
    search_settings: dict[str, object]
    pricing: PricingData


class UiDebateInput(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str
    max_rounds: int
    novelty_threshold: float


class UiDecomposition(TypedDict):
    question_complexity: Literal["simple", "complex"]
    subquestions: list[str]


class UiEvidence(TypedDict):
    raw_research_findings: list[ResearchFinding]
    consolidated_evidence: list[ResearchFinding]
    research_findings: list[str]


class UiFinalOutputs(TypedDict):
    team_a_argument: str
    team_b_argument: str
    neutral_synthesis: str


class UiModelCallCounts(TypedDict):
    decomposition: int
    team_a_argument: int
    team_b_argument: int
    team_a_analysis: int
    team_a_draft: int
    team_a_review: int
    team_b_analysis: int
    team_b_draft: int
    team_b_review: int
    synthesis: int
    research_provider_calls: int
    total_model_calls: int


class UiUsage(TypedDict):
    usage_extraction_version: str
    entries: list[ModelUsageEntry]
    totals: UsageTotals


class UiTeamWorkProducts(TypedDict):
    team_a: TeamWorkProduct | None
    team_b: TeamWorkProduct | None


class UiDebateResult(TypedDict):
    debate_result_version: str
    mode: UiMode
    provenance: UiDebateProvenance
    input: UiDebateInput
    decomposition: UiDecomposition
    search_evidence: list[SearchCallEvidence]
    evidence: UiEvidence
    debate_rounds: list[DebateRound]
    termination: dict[str, object]
    final_outputs: UiFinalOutputs
    team_work_products: UiTeamWorkProducts
    timings: dict[str, object]
    model_call_counts: UiModelCallCounts
    metrics: ScorecardData
    usage: UiUsage
    cost_estimate: CostEstimate
    efficiency: EfficiencyMeasurements


class UiDisplayMetadata(TypedDict):
    selected_mode: UiMode
    requested_rounds: int
    model_name: str
    created_utc: str
    judgment_label: str


class UiDisplayRow(TypedDict):
    """One homogeneous row rendered by a Streamlit metrics table."""

    Metric: str
    Value: str


class UiJudgmentStatus(TypedDict):
    status: Literal["available", "unavailable"]
    error_category: UiJudgmentErrorCategory | None


class UiResult(TypedDict):
    ui_result_version: str
    debate_result: UiDebateResult
    judgment_status: UiJudgmentStatus
    judgment: JudgeRecord | None
    display_metadata: UiDisplayMetadata


def public_debate_from_product_record(value: object) -> UiDebateResult:
    """Normalize a validated v2.1 product record without changing its meaning."""

    record = validate_multi_round_run_record(value)
    if record["run_record_version"] != "2.1":
        raise UiResultValidationError("The UI requires telemetry-bearing debate output.")
    v21 = cast(MultiRoundRunRecordV21, record)
    mode = cast(UiMode, v21["provenance"]["product_mode"])
    counts = v21["model_call_counts"]
    public: UiDebateResult = {
        "debate_result_version": UI_DEBATE_RESULT_VERSION,
        "mode": mode,
        "provenance": {
            "run_timestamp_utc": v21["provenance"]["run_timestamp_utc"],
            "implementation_commit": v21["provenance"]["implementation_commit"],
            "architecture_version": v21["provenance"]["architecture_version"],
            "source_run_record_version": "2.1",
            "model": {
                "name": v21["provenance"]["model"]["name"],
                "temperature": 0.0,
                "reasoning": False,
            },
            "search_provider": v21["provenance"]["search"]["provider"],
            "search_settings": copy.deepcopy(
                v21["provenance"]["search"]["settings"]
            ),
            "pricing": {
                "input_cost_per_million": v21["cost_estimate"][
                    "input_cost_per_million"
                ],
                "output_cost_per_million": v21["cost_estimate"][
                    "output_cost_per_million"
                ],
                "pricing_basis": v21["cost_estimate"]["pricing_basis"],
            },
        },
        "input": copy.deepcopy(v21["input"]),
        "decomposition": copy.deepcopy(v21["decomposition"]),
        "search_evidence": copy.deepcopy(v21["search_evidence"]),
        "evidence": {
            "raw_research_findings": copy.deepcopy(
                v21["graph_evidence"]["raw_research_findings"]
            ),
            "consolidated_evidence": copy.deepcopy(
                v21["graph_evidence"]["consolidated_evidence"]
            ),
            "research_findings": v21["graph_evidence"]["research_findings"].copy(),
        },
        "debate_rounds": copy.deepcopy(v21["debate_rounds"]),
        "termination": copy.deepcopy(v21["termination"]),
        "final_outputs": {
            "team_a_argument": v21["graph_evidence"]["final_team_a_argument"],
            "team_b_argument": v21["graph_evidence"]["final_team_b_argument"],
            "neutral_synthesis": v21["graph_evidence"]["neutral_synthesis"],
        },
        "team_work_products": {"team_a": None, "team_b": None},
        "timings": copy.deepcopy(v21["timings"]),
        "model_call_counts": {
            "decomposition": counts["decomposition"],
            "team_a_argument": counts["team_a_argument"],
            "team_b_argument": counts["team_b_argument"],
            "team_a_analysis": 0,
            "team_a_draft": 0,
            "team_a_review": 0,
            "team_b_analysis": 0,
            "team_b_draft": 0,
            "team_b_review": 0,
            "synthesis": counts["synthesis"],
            "research_provider_calls": counts["research_provider_calls"],
            "total_model_calls": counts["total_model_calls"],
        },
        "metrics": copy.deepcopy(v21["metrics"]),
        "usage": copy.deepcopy(v21["usage"]),
        "cost_estimate": copy.deepcopy(v21["cost_estimate"]),
        "efficiency": copy.deepcopy(v21["efficiency"]),
    }
    return validate_ui_debate_result(public)


def validate_ui_debate_result(value: object) -> UiDebateResult:
    """Strictly cross-validate one normalized public debate result."""

    try:
        data = _exact(value, UiDebateResult, "debate result")
        if data["debate_result_version"] != UI_DEBATE_RESULT_VERSION:
            raise UiResultValidationError("Debate result version is invalid.")
        mode = data["mode"]
        if mode not in UI_MODES:
            raise UiResultValidationError("Debate mode is invalid.")
        provenance = _validate_provenance(data["provenance"])
        expected_provenance = {
            "single-round": ("final-multi-round-grounded", "2.1"),
            "multi-round": ("final-multi-round-grounded", "2.1"),
            "hierarchical": ("hierarchical-grounded-ui", "hierarchical-live-1.0"),
        }[mode]
        if (
            provenance["architecture_version"],
            provenance["source_run_record_version"],
        ) != expected_provenance:
            raise UiResultValidationError("Debate provenance is inconsistent.")
        run_input = _validate_input(data["input"], mode=mode)
        decomposition = validate_decomposition_result(data["decomposition"])
        search_evidence = _validate_search_evidence(data["search_evidence"])
        evidence = _validate_evidence(
            data["evidence"],
            question=run_input["question"],
            complexity=decomposition["question_complexity"],
            subquestions=decomposition["subquestions"],
        )
        _cross_validate_search_findings(search_evidence, evidence)
        rounds = _validate_rounds(
            data["debate_rounds"],
            max_rounds=run_input["max_rounds"],
            novelty_threshold=run_input["novelty_threshold"],
        )
        termination = validate_termination_decision(data["termination"])
        if rounds[-1]["termination_decision"] != termination:
            raise UiResultValidationError("Final termination is inconsistent.")
        outputs = _validate_outputs(data["final_outputs"], rounds=rounds)
        products = _validate_products(
            data["team_work_products"],
            evidence=evidence["consolidated_evidence"],
            input_data=run_input,
            outputs=outputs,
            mode=mode,
        )
        timings = _validate_json_object(data["timings"], "timings")
        expected_timing_keys = (
            {"decomposition", "research", "team", "total_duration_seconds"}
            if mode == "hierarchical"
            else {"decomposition", "research", "debate", "total_duration_seconds"}
        )
        if set(timings) != expected_timing_keys:
            raise UiResultValidationError("Debate timings have invalid fields.")
        if "total_duration_seconds" not in timings:
            raise UiResultValidationError("Total latency is missing.")
        _finite_nonnegative(timings["total_duration_seconds"])
        counts = _validate_call_counts(
            data["model_call_counts"], mode=mode, rounds=len(rounds)
        )
        if counts["research_provider_calls"] != len(search_evidence):
            raise UiResultValidationError("Research call count is inconsistent.")
        usage = _validate_usage(
            data["usage"],
            expected_calls=counts["total_model_calls"],
            mode=mode,
            rounds=len(rounds),
            model_name=provenance["model"]["name"],
        )
        pricing = _pricing_from_data(provenance["pricing"])
        cost = validate_cost_estimate(
            data["cost_estimate"], totals=usage["totals"], pricing=pricing
        )
        metrics = _validate_metrics(
            data["metrics"],
            input_data=run_input,
            decomposition=decomposition,
            evidence=evidence,
            outputs=outputs,
        )
        generated_words = calculate_ui_generated_words(
            mode=mode,
            decomposition=decomposition,
            rounds=rounds,
            products=products,
            synthesis=outputs["neutral_synthesis"],
        )
        expected_efficiency = calculate_efficiency(
            usage_totals=usage["totals"],
            visible_output_words=sum(
                count_generated_words(text) for text in outputs.values()
            ),
            all_generated_words=generated_words,
            latency_seconds=cast(float, timings["total_duration_seconds"]),
            completed_rounds=len(rounds),
        )
        if data["efficiency"] != expected_efficiency:
            raise UiResultValidationError("Debate efficiency is inconsistent.")
    except UiResultValidationError:
        raise
    except Exception:
        raise UiResultValidationError("Public debate result is invalid.") from None
    return {
        "debate_result_version": UI_DEBATE_RESULT_VERSION,
        "mode": cast(UiMode, mode),
        "provenance": provenance,
        "input": run_input,
        "decomposition": copy.deepcopy(decomposition),
        "search_evidence": search_evidence,
        "evidence": evidence,
        "debate_rounds": rounds,
        "termination": copy.deepcopy(termination),
        "final_outputs": outputs,
        "team_work_products": products,
        "timings": timings,
        "model_call_counts": counts,
        "metrics": metrics,
        "usage": usage,
        "cost_estimate": cost,
        "efficiency": expected_efficiency,
    }


def judge_material_from_debate(value: object) -> JudgeDebateMaterial:
    debate = validate_ui_debate_result(value)
    return {
        "question": debate["input"]["question"],
        "team_a_stance": debate["input"]["team_a_stance"],
        "team_b_stance": debate["input"]["team_b_stance"],
        "consolidated_evidence": copy.deepcopy(
            debate["evidence"]["consolidated_evidence"]
        ),
        "debate_rounds": [
            {
                "round_number": item["round_number"],
                "team_a_argument": item["team_a_argument"],
                "team_b_argument": item["team_b_argument"],
            }
            for item in debate["debate_rounds"]
        ],
        "final_team_a_argument": debate["final_outputs"]["team_a_argument"],
        "final_team_b_argument": debate["final_outputs"]["team_b_argument"],
        "neutral_synthesis": debate["final_outputs"]["neutral_synthesis"],
    }


def build_ui_result(
    *,
    debate_result: UiDebateResult,
    judgment: JudgeRecord | None,
    judgment_error_category: UiJudgmentErrorCategory | None = None,
) -> UiResult:
    debate = validate_ui_debate_result(debate_result)
    if judgment is None:
        if judgment_error_category is None:
            raise UiResultValidationError("Unavailable judgment requires an error category.")
        judge = None
        status: UiJudgmentStatus = {
            "status": "unavailable",
            "error_category": judgment_error_category,
        }
        created_utc = debate["provenance"]["run_timestamp_utc"]
        judgment_label = "Advisory judgment unavailable"
    else:
        if judgment_error_category is not None:
            raise UiResultValidationError("Available judgment cannot have an error category.")
        material = judge_material_from_debate(debate)
        judge = validate_judge_record(
            judgment, debate_result=debate, debate_material=material
        )
        status = {"status": "available", "error_category": None}
        created_utc = judge["provenance"]["run_timestamp_utc"]
        judgment_label = "LLM-generated judgment (advisory, not objective)"
    value: UiResult = {
        "ui_result_version": UI_RESULT_VERSION,
        "debate_result": debate,
        "judgment_status": status,
        "judgment": judge,
        "display_metadata": {
            "selected_mode": debate["mode"],
            "requested_rounds": debate["input"]["max_rounds"],
            "model_name": debate["provenance"]["model"]["name"],
            "created_utc": created_utc,
            "judgment_label": judgment_label,
        },
    }
    return validate_ui_result(value)


def validate_ui_result(value: object, *, forbidden_secret: str | None = None) -> UiResult:
    try:
        data = _exact(value, UiResult, "UI result")
        if data["ui_result_version"] != UI_RESULT_VERSION:
            raise UiResultValidationError("UI result version is invalid.")
        debate = validate_ui_debate_result(data["debate_result"])
        status = _validate_judgment_status(data["judgment_status"])
        if status["status"] == "available":
            if data["judgment"] is None:
                raise UiResultValidationError("Available judgment is missing.")
            material = judge_material_from_debate(debate)
            judgment = validate_judge_record(
                data["judgment"], debate_result=debate, debate_material=material
            )
            created_utc = judgment["provenance"]["run_timestamp_utc"]
            judgment_label = "LLM-generated judgment (advisory, not objective)"
        else:
            if data["judgment"] is not None:
                raise UiResultValidationError("Unavailable judgment must be null.")
            judgment = None
            created_utc = debate["provenance"]["run_timestamp_utc"]
            judgment_label = "Advisory judgment unavailable"
        metadata = _exact(data["display_metadata"], UiDisplayMetadata, "display metadata")
        expected = {
            "selected_mode": debate["mode"],
            "requested_rounds": debate["input"]["max_rounds"],
            "model_name": debate["provenance"]["model"]["name"],
            "created_utc": created_utc,
            "judgment_label": judgment_label,
        }
        if metadata != expected:
            raise UiResultValidationError("UI display metadata is inconsistent.")
        result: UiResult = {
            "ui_result_version": UI_RESULT_VERSION,
            "debate_result": debate,
            "judgment_status": status,
            "judgment": judgment,
            "display_metadata": cast(UiDisplayMetadata, copy.deepcopy(expected)),
        }
        _assert_forbidden_fields_absent(result)
        if forbidden_secret:
            serialized = _serialize(result)
            if forbidden_secret in serialized or _contains_string(result, forbidden_secret):
                raise UiResultValidationError("Sensitive configuration was detected.")
        return result
    except UiResultValidationError:
        raise
    except Exception:
        raise UiResultValidationError("Downloadable UI result is invalid.") from None


def ui_result_to_json(value: object, *, forbidden_secret: str | None = None) -> str:
    return _serialize(validate_ui_result(value, forbidden_secret=forbidden_secret))


def debate_overview(value: object) -> dict[str, object]:
    result = validate_ui_result(value)
    debate = result["debate_result"]
    usage = debate["usage"]["totals"]
    judgment = result["judgment"]
    total_tokens: object = (
        usage["total_tokens"] if usage["all_usage_available"] else "unavailable"
    )
    debate_cost = debate["cost_estimate"]
    total_cost: object = (
        debate_cost["estimated_cost"]
        if debate_cost["status"] == "available"
        else "unavailable"
    )
    judge_duration = 0.0
    winner: object = None
    confidence: object = None
    team_a_overall: object = None
    team_b_overall: object = None
    if judgment is not None:
        judge_usage = judgment["usage"]["totals"]
        if usage["all_usage_available"] and judge_usage["all_usage_available"]:
            total_tokens = usage["total_tokens"] + judge_usage["total_tokens"]
        judge_cost = judgment["cost_estimate"]
        if debate_cost["status"] == "available" and judge_cost["status"] == "available":
            total_cost = decimal_to_string(
                decimal_from_string(debate_cost["estimated_cost"])
                + decimal_from_string(judge_cost["estimated_cost"])
            )
        judge_duration = sum(
            entry["duration_seconds"] for entry in judgment["usage"]["entries"]
        )
        winner = judgment["winner"]
        confidence = judgment["confidence"]
        team_a_overall = judgment["scores"]["team_a"]["overall_score"]
        team_b_overall = judgment["scores"]["team_b"]["overall_score"]
    return {
        "judgment_status": result["judgment_status"]["status"],
        "winner": winner,
        "confidence": confidence,
        "team_a_overall": team_a_overall,
        "team_b_overall": team_b_overall,
        "termination_reason": debate["termination"]["reason"],
        "completed_rounds": len(debate["debate_rounds"]),
        "source_count": sum(len(call["results"]) for call in debate["search_evidence"]),
        "total_tokens": total_tokens,
        "estimated_cost": total_cost,
        "total_latency_seconds": debate["timings"]["total_duration_seconds"] + judge_duration,
    }


def debate_round_rows(value: object) -> list[dict[str, object]]:
    debate = validate_ui_debate_result(value)
    final_number = len(debate["debate_rounds"])
    return [
        {
            **copy.deepcopy(item),
            "team_a_novelty_display": (
                "n/a" if item["team_a_novelty"] is None else item["team_a_novelty"]
            ),
            "team_b_novelty_display": (
                "n/a" if item["team_b_novelty"] is None else item["team_b_novelty"]
            ),
            "is_final_round": item["round_number"] == final_number,
        }
        for item in debate["debate_rounds"]
    ]


def judge_score_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, dict) or "scores" not in value:
        raise UiResultValidationError("Judgment data is invalid.")
    scores = value["scores"]
    if not isinstance(scores, dict):
        raise UiResultValidationError("Judgment data is invalid.")
    return [
        {
            "criterion": criterion.replace("_", " ").title(),
            "team_a": scores["team_a"][criterion],
            "team_b": scores["team_b"][criterion],
        }
        for criterion in (*RUBRIC_CRITERIA, "overall_score")
    ]


def evidence_rows(value: object) -> list[dict[str, object]]:
    debate = validate_ui_debate_result(value)
    rows: list[dict[str, object]] = []
    for item in debate["evidence"]["consolidated_evidence"]:
        link = source_link_presentation(item.get("source_url"))
        rows.append(
            {
                "subquestion_index": item["subquestion_index"],
                "finding_index": item["finding_index"],
                "subquestion": item["subquestion"],
                "claim": item["claim"],
                "source_title": item["source_title"],
                "source_url": link["source_url"],
                "source_link_label": link["source_link_label"],
                "domain": link["domain"],
                "source_type": item["source_type"],
            }
        )
    return rows


def validate_renderable_source_url(value: object) -> str | None:
    """Return a browser-safe HTTP(S) source URL or ``None``."""

    if not isinstance(value, str) or not value:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.casefold()
        host = parsed.hostname
    except (TypeError, ValueError):
        return None
    if scheme not in {"http", "https"}:
        return None
    if not parsed.netloc or not isinstance(host, str) or not host.strip():
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return value


def source_link_presentation(value: object) -> dict[str, str | None]:
    """Build a link-safe evidence-row fragment for Streamlit rendering."""

    source_url = validate_renderable_source_url(value)
    if source_url is None:
        return {
            "source_url": None,
            "source_link_label": "Unavailable link",
            "domain": "unknown",
        }
    parsed = urlsplit(source_url)
    return {
        "source_url": source_url,
        "source_link_label": "Open source",
        "domain": parsed.hostname or "unknown",
    }


def metrics_and_cost(value: object) -> dict[str, object]:
    debate = validate_ui_debate_result(value)
    usage = debate["usage"]["totals"]
    return {
        "research_utilization": debate["metrics"]["research_utilization"]["score"],
        "team_novelty": debate["metrics"]["team_novelty"]["score"],
        "synthesis_representation": debate["metrics"]["synthesis_representation"]["score"],
        "summary_policy_compliance": debate["metrics"]["summary_policy_compliance"]["score"],
        "model_calls": debate["model_call_counts"]["total_model_calls"],
        "input_tokens": usage["input_tokens"] if usage["all_usage_available"] else "unavailable",
        "output_tokens": usage["output_tokens"] if usage["all_usage_available"] else "unavailable",
        "total_tokens": usage["total_tokens"] if usage["all_usage_available"] else "unavailable",
        "tokens_per_call": debate["efficiency"]["tokens_per_call"],
        "tokens_per_visible_output_word": debate["efficiency"]["tokens_per_visible_output_word"],
        "latency_seconds": debate["timings"]["total_duration_seconds"],
        "input_cost_per_million": debate["provenance"]["pricing"]["input_cost_per_million"],
        "output_cost_per_million": debate["provenance"]["pricing"]["output_cost_per_million"],
        "estimated_cost": debate["cost_estimate"]["estimated_cost"],
        "pricing_basis": debate["cost_estimate"]["pricing_basis"],
    }


def judgment_usage_and_cost(value: object) -> dict[str, object]:
    result = validate_ui_result(value)
    judgment = result["judgment"]
    if judgment is None:
        return {
            "status": "unavailable",
            "model_calls": "unavailable",
            "input_tokens": "unavailable",
            "output_tokens": "unavailable",
            "total_tokens": "unavailable",
            "latency_seconds": "unavailable",
            "estimated_cost": "unavailable",
            "pricing_basis": "unavailable",
        }
    usage = judgment["usage"]["totals"]
    cost = judgment["cost_estimate"]
    return {
        "status": "available",
        "model_calls": usage["model_calls"],
        "input_tokens": usage["input_tokens"] if usage["all_usage_available"] else "unavailable",
        "output_tokens": usage["output_tokens"] if usage["all_usage_available"] else "unavailable",
        "total_tokens": usage["total_tokens"] if usage["all_usage_available"] else "unavailable",
        "latency_seconds": sum(
            entry["duration_seconds"] for entry in judgment["usage"]["entries"]
        ),
        "estimated_cost": cost["estimated_cost"] if cost["status"] == "available" else "unavailable",
        "pricing_basis": cost["pricing_basis"],
    }


_GROUPED_INTEGER_DISPLAY_FIELDS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens"}
)


def display_metric_rows(values: Mapping[str, object]) -> list[UiDisplayRow]:
    """Format mixed typed values as homogeneous strings for table display only."""

    rows: list[UiDisplayRow] = []
    for metric, value in values.items():
        if not isinstance(metric, str) or not metric:
            raise UiResultValidationError("A display metric name is invalid.")
        rows.append(
            {
                "Metric": metric.replace("_", " ").title(),
                "Value": _format_display_value(metric, value),
            }
        )
    return rows


def _format_display_value(metric: str, value: object) -> str:
    if value is None or (
        isinstance(value, str) and value.casefold() == "unavailable"
    ):
        return "n/a"
    if type(value) is bool:
        return "Yes" if value else "No"
    if type(value) is int:
        if metric == "summary_policy_compliance" and value in (0, 1):
            return "Yes" if value else "No"
        if metric in _GROUPED_INTEGER_DISPLAY_FIELDS:
            return f"{value:,}"
        return str(value)
    if type(value) is float:
        if not isfinite(value):
            raise UiResultValidationError("A numeric display value is invalid.")
        if metric == "summary_policy_compliance" and value in (0.0, 1.0):
            return "Yes" if value else "No"
        return f"{value:.6f}"
    if isinstance(value, str):
        return value
    raise UiResultValidationError("A display value has an unsupported type.")


def judgment_presentation(value: object) -> dict[str, object]:
    """Return bounded Judge-tab data without fabricating unavailable results."""

    result = validate_ui_result(value)
    if result["judgment"] is None:
        return {
            "available": False,
            "notice": JUDGMENT_UNAVAILABLE_NOTICE,
            "judgment": None,
        }
    return {
        "available": True,
        "notice": None,
        "judgment": copy.deepcopy(result["judgment"]),
    }


def _validate_judgment_status(value: object) -> UiJudgmentStatus:
    data = _exact(value, UiJudgmentStatus, "judgment status")
    status = data["status"]
    category = data["error_category"]
    if status == "available" and category is None:
        return {"status": "available", "error_category": None}
    if status == "unavailable" and category in UI_JUDGMENT_ERROR_CATEGORIES:
        return {
            "status": "unavailable",
            "error_category": cast(UiJudgmentErrorCategory, category),
        }
    raise UiResultValidationError("Judgment status is invalid.")


def _validate_provenance(value: object) -> UiDebateProvenance:
    data = _exact(value, UiDebateProvenance, "debate provenance")
    for field in (
        "run_timestamp_utc",
        "architecture_version",
        "source_run_record_version",
        "search_provider",
    ):
        if not isinstance(data[field], str) or not data[field].strip():
            raise UiResultValidationError("Debate provenance is invalid.")
    timestamp = data["run_timestamp_utc"]
    if not timestamp.endswith("Z"):
        raise UiResultValidationError("Debate timestamp is invalid.")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError:
        raise UiResultValidationError("Debate timestamp is invalid.") from None
    if (
        parsed_timestamp.utcoffset() is None
        or parsed_timestamp.utcoffset().total_seconds() != 0
    ):
        raise UiResultValidationError("Debate timestamp is invalid.")
    if data["search_provider"] != "tavily":
        raise UiResultValidationError("Debate search provider is invalid.")
    commit = normalize_implementation_commit(data["implementation_commit"])
    if commit != data["implementation_commit"]:
        raise UiResultValidationError("Implementation commit is not normalized.")
    model = data["model"]
    if not isinstance(model, dict) or set(model) != set(UiModelSettings.__required_keys__):
        raise UiResultValidationError("Debate model settings are invalid.")
    if (
        not isinstance(model["name"], str)
        or not model["name"].strip()
        or type(model["temperature"]) is not float
        or model["temperature"] != 0.0
        or model["reasoning"] is not False
    ):
        raise UiResultValidationError("Debate model settings are invalid.")
    settings = _validate_json_object(data["search_settings"], "search settings")
    pricing = validate_pricing_data(data["pricing"])
    return {
        "run_timestamp_utc": data["run_timestamp_utc"],
        "implementation_commit": commit,
        "architecture_version": data["architecture_version"],
        "source_run_record_version": data["source_run_record_version"],
        "model": {"name": model["name"], "temperature": 0.0, "reasoning": False},
        "search_provider": data["search_provider"],
        "search_settings": settings,
        "pricing": pricing,
    }


def _validate_input(value: object, *, mode: object) -> UiDebateInput:
    data = _exact(value, UiDebateInput, "debate input")
    for field in ("question", "team_a_stance", "team_b_stance"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise UiResultValidationError("Debate input text is invalid.")
    rounds = validate_max_rounds(data["max_rounds"])
    threshold = validate_novelty_threshold(data["novelty_threshold"])
    if mode in ("single-round", "hierarchical") and rounds != 1:
        raise UiResultValidationError("Selected mode requires exactly one round.")
    return {
        "question": data["question"],
        "team_a_stance": data["team_a_stance"],
        "team_b_stance": data["team_b_stance"],
        "max_rounds": rounds,
        "novelty_threshold": threshold,
    }


def _validate_search_evidence(value: object) -> list[SearchCallEvidence]:
    if type(value) is not list or not 1 <= len(value) <= 4:
        raise UiResultValidationError("Search evidence is invalid.")
    expected_settings = sanitized_search_settings()
    calls: list[SearchCallEvidence] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != set(
            SearchCallEvidence.__required_keys__
        ):
            raise UiResultValidationError("Search call has invalid fields.")
        if (
            type(raw["subquestion_index"]) is not int
            or raw["subquestion_index"] < 0
            or not isinstance(raw["query"], str)
            or not raw["query"].strip()
            or type(raw["requested_max_results"]) is not int
            or raw["requested_max_results"] <= 0
            or raw["settings"] != expected_settings
            or raw["status"] != "success"
            or raw["error_category"] is not None
        ):
            raise UiResultValidationError("Search call is inconsistent.")
        backend_duration = _finite_nonnegative(raw["backend_duration_seconds"])
        research_duration = _finite_nonnegative(raw["research_duration_seconds"])
        raw_results = raw["results"]
        if type(raw_results) is not list or len(raw_results) > raw["requested_max_results"]:
            raise UiResultValidationError("Search results are invalid.")
        results: list[SearchResultEvidence] = []
        for rank, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict) or set(item) != set(
                SearchResultEvidence.__required_keys__
            ):
                raise UiResultValidationError("Search result has invalid fields.")
            if type(item["rank"]) is not int or item["rank"] != rank:
                raise UiResultValidationError("Search result rank is invalid.")
            for field in ("title", "url", "snippet"):
                if not isinstance(item[field], str) or not item[field].strip():
                    raise UiResultValidationError("Search result text is invalid.")
            score = item["provider_score"]
            if (
                type(score) not in (int, float)
                or not isfinite(float(score))
                or item["source_type"] != "web"
            ):
                raise UiResultValidationError("Search result metadata is invalid.")
            results.append(
                {
                    "rank": rank,
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                    "provider_score": float(score),
                    "source_type": "web",
                }
            )
        calls.append(
            {
                "subquestion_index": raw["subquestion_index"],
                "query": raw["query"],
                "requested_max_results": raw["requested_max_results"],
                "settings": copy.deepcopy(expected_settings),
                "status": "success",
                "error_category": None,
                "backend_duration_seconds": backend_duration,
                "research_duration_seconds": research_duration,
                "results": results,
            }
        )
    return calls


def _validate_evidence(
    value: object,
    *,
    question: str,
    complexity: str,
    subquestions: list[str],
) -> UiEvidence:
    data = _exact(value, UiEvidence, "evidence")
    if type(data["raw_research_findings"]) is not list or type(
        data["consolidated_evidence"]
    ) is not list or type(data["research_findings"]) is not list:
        raise UiResultValidationError("Evidence collections are invalid.")
    raw = [validate_research_finding(item) for item in data["raw_research_findings"]]
    consolidated = [
        validate_research_finding(item) for item in data["consolidated_evidence"]
    ]
    expected = consolidate_research_findings(
        question=question,
        question_complexity=cast(Literal["simple", "complex"], complexity),
        subquestions=subquestions,
        raw_findings=raw,
    )
    if consolidated != expected:
        raise UiResultValidationError("Consolidated evidence is inconsistent.")
    formatted = [format_research_finding(item) for item in consolidated]
    if data["research_findings"] != formatted:
        raise UiResultValidationError("Formatted evidence is inconsistent.")
    return {
        "raw_research_findings": copy.deepcopy(raw),
        "consolidated_evidence": copy.deepcopy(consolidated),
        "research_findings": formatted,
    }


def _cross_validate_search_findings(
    calls: list[SearchCallEvidence], evidence: UiEvidence
) -> None:
    expected = []
    for call in calls:
        if not isinstance(call, dict) or "subquestion_index" not in call or "query" not in call or "results" not in call:
            raise UiResultValidationError("Search evidence is malformed.")
        for item in call["results"]:
            expected.append(
                {
                    "subquestion_index": call["subquestion_index"],
                    "subquestion": call["query"],
                    "finding_index": item["rank"],
                    "claim": item["snippet"],
                    "source_title": item["title"],
                    "source_url": item["url"],
                    "source_type": item["source_type"],
                    "supporting_excerpt": item["snippet"],
                }
            )
    if sorted(expected, key=_finding_key) != sorted(
        evidence["raw_research_findings"], key=_finding_key
    ):
        raise UiResultValidationError("Search evidence and findings are inconsistent.")


def _validate_rounds(
    value: object, *, max_rounds: int, novelty_threshold: float
) -> list[DebateRound]:
    if type(value) is not list or not 1 <= len(value) <= max_rounds:
        raise UiResultValidationError("Debate rounds are invalid.")
    rounds = copy.deepcopy(cast(list[DebateRound], value))
    team_a_history: list[str] = []
    team_b_history: list[str] = []
    for number, item in enumerate(rounds, start=1):
        if not isinstance(item, dict) or set(item) != set(DebateRound.__required_keys__):
            raise UiResultValidationError("Debate round fields are invalid.")
        if type(item["round_number"]) is not int or item["round_number"] != number:
            raise UiResultValidationError("Debate round order is invalid.")
        for field in ("team_a_argument", "team_b_argument", "team_b_opponent_argument"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise UiResultValidationError("Debate argument is invalid.")
        if number == 1:
            if item["team_a_opponent_argument"] is not None or item["team_a_novelty"] is not None or item["team_b_novelty"] is not None:
                raise UiResultValidationError("Round-one routing or novelty is invalid.")
        elif item["team_a_opponent_argument"] != rounds[number - 2]["team_b_argument"]:
            raise UiResultValidationError("Team A opponent routing is invalid.")
        if item["team_b_opponent_argument"] != item["team_a_argument"]:
            raise UiResultValidationError("Team B opponent routing is invalid.")
        for field in ("team_a_duration_seconds", "team_b_duration_seconds", "termination_duration_seconds"):
            _finite_nonnegative(item[field])
        decision = validate_termination_decision(item["termination_decision"])
        if decision["team_a_novelty"] != item["team_a_novelty"] or decision["team_b_novelty"] != item["team_b_novelty"]:
            raise UiResultValidationError("Round novelty is inconsistent.")
        team_a_history.append(item["team_a_argument"])
        team_b_history.append(item["team_b_argument"])
        expected_decision = decide_termination(
            team_a_history=team_a_history,
            team_b_history=team_b_history,
            max_rounds=max_rounds,
            novelty_threshold=novelty_threshold,
        )
        if decision != expected_decision:
            raise UiResultValidationError("Round termination policy is inconsistent.")
        if number < len(rounds) and decision["stop"]:
            raise UiResultValidationError("Debate continued after stopping.")
        if number == len(rounds) and not decision["stop"]:
            raise UiResultValidationError("Final round did not stop.")
    return rounds


def _validate_outputs(value: object, *, rounds: list[DebateRound]) -> UiFinalOutputs:
    data = _exact(value, UiFinalOutputs, "final outputs")
    for field in UiFinalOutputs.__required_keys__:
        if not isinstance(data[field], str) or not data[field].strip():
            raise UiResultValidationError("Final output is invalid.")
    if data["team_a_argument"] != rounds[-1]["team_a_argument"] or data[
        "team_b_argument"
    ] != rounds[-1]["team_b_argument"]:
        raise UiResultValidationError("Final arguments are inconsistent.")
    return cast(UiFinalOutputs, copy.deepcopy(data))


def _validate_products(
    value: object,
    *,
    evidence: list[ResearchFinding],
    input_data: UiDebateInput,
    outputs: UiFinalOutputs,
    mode: object,
) -> UiTeamWorkProducts:
    data = _exact(value, UiTeamWorkProducts, "team work products")
    if mode != "hierarchical":
        if data != {"team_a": None, "team_b": None}:
            raise UiResultValidationError("One-shot debates cannot contain team work products.")
        return {"team_a": None, "team_b": None}
    if data["team_a"] is None or data["team_b"] is None:
        raise UiResultValidationError("Hierarchical team work products are missing.")
    team_a = validate_team_work_product(
        data["team_a"],
        consolidated_evidence=evidence,
        expected_team="team_a",
        expected_stance=input_data["team_a_stance"],
    )
    team_b = validate_team_work_product(
        data["team_b"],
        consolidated_evidence=evidence,
        expected_team="team_b",
        expected_stance=input_data["team_b_stance"],
    )
    if team_a["final_argument"] != outputs["team_a_argument"] or team_b[
        "final_argument"
    ] != outputs["team_b_argument"]:
        raise UiResultValidationError("Team work products do not match final arguments.")
    return {"team_a": copy.deepcopy(team_a), "team_b": copy.deepcopy(team_b)}


def _validate_call_counts(value: object, *, mode: object, rounds: int) -> UiModelCallCounts:
    data = _exact(value, UiModelCallCounts, "model call counts")
    if any(type(data[field]) is not int or data[field] < 0 for field in data):
        raise UiResultValidationError("Model call counts are invalid.")
    if mode == "hierarchical":
        expected = {
            "decomposition": 1,
            "team_a_argument": 0,
            "team_b_argument": 0,
            "team_a_analysis": 1,
            "team_a_draft": 1,
            "team_a_review": 1,
            "team_b_analysis": 1,
            "team_b_draft": 1,
            "team_b_review": 1,
            "synthesis": 1,
            "research_provider_calls": data["research_provider_calls"],
            "total_model_calls": 8,
        }
    else:
        expected = {
            "decomposition": 1,
            "team_a_argument": rounds,
            "team_b_argument": rounds,
            "team_a_analysis": 0,
            "team_a_draft": 0,
            "team_a_review": 0,
            "team_b_analysis": 0,
            "team_b_draft": 0,
            "team_b_review": 0,
            "synthesis": 1,
            "research_provider_calls": data["research_provider_calls"],
            "total_model_calls": 2 * rounds + 2,
        }
    if data != expected or data["research_provider_calls"] <= 0:
        raise UiResultValidationError("Model call counts are inconsistent.")
    return cast(UiModelCallCounts, copy.deepcopy(data))


def _validate_usage(
    value: object,
    *,
    expected_calls: int,
    mode: object,
    rounds: int,
    model_name: str,
) -> UiUsage:
    data = _exact(value, UiUsage, "usage")
    if data["usage_extraction_version"] != USAGE_EXTRACTION_VERSION or type(data["entries"]) is not list:
        raise UiResultValidationError("Usage telemetry is invalid.")
    entries = [validate_usage_entry(item) for item in data["entries"]]
    totals = validate_usage_totals(data["totals"], entries=entries)
    if totals["model_calls"] != expected_calls:
        raise UiResultValidationError("Usage count does not match model calls.")
    if any(entry["model"] != model_name for entry in entries):
        raise UiResultValidationError("Usage model is inconsistent.")
    if mode == "hierarchical":
        expected_identities = [
            ("decomposition", "system", None),
            ("evidence_analysis", "team_a", 1),
            ("argument_draft", "team_a", 1),
            ("review_revision", "team_a", 1),
            ("evidence_analysis", "team_b", 1),
            ("argument_draft", "team_b", 1),
            ("review_revision", "team_b", 1),
            ("neutral_synthesis", "system", 1),
        ]
        expected_variant = "ui_hierarchical"
    else:
        expected_identities = [("decomposition", "system", None)]
        expected_identities.extend(
            (stage, team, round_number)
            for round_number in range(1, rounds + 1)
            for stage, team in (
                ("team_a_argument", "team_a"),
                ("team_b_argument", "team_b"),
            )
        )
        expected_identities.append(("neutral_synthesis", "system", rounds))
        expected_variant = "final_product"
    observed = [
        (entry["stage"], entry["team"], entry["round_number"])
        for entry in entries
    ]
    if observed != expected_identities or any(
        entry["variant"] != expected_variant for entry in entries
    ):
        raise UiResultValidationError("Usage stages are inconsistent.")
    return {
        "usage_extraction_version": USAGE_EXTRACTION_VERSION,
        "entries": copy.deepcopy(entries),
        "totals": totals,
    }


def _validate_metrics(
    value: object,
    *,
    input_data: UiDebateInput,
    decomposition: dict,
    evidence: UiEvidence,
    outputs: UiFinalOutputs,
) -> ScorecardData:
    if not isinstance(value, dict):
        raise UiResultValidationError("Metrics are invalid.")
    metric_input = {
        "question": input_data["question"],
        "team_a_stance": input_data["team_a_stance"],
        "team_b_stance": input_data["team_b_stance"],
        "question_complexity": decomposition["question_complexity"],
        "subquestions": decomposition["subquestions"].copy(),
        "raw_research_findings": copy.deepcopy(evidence["raw_research_findings"]),
        "consolidated_evidence": copy.deepcopy(evidence["consolidated_evidence"]),
        "research_findings": evidence["research_findings"].copy(),
        "team_a_argument": outputs["team_a_argument"],
        "team_b_argument": outputs["team_b_argument"],
        "synthesis": outputs["neutral_synthesis"],
    }
    expected = json.loads(scorecard_to_json(score_expanded_result(metric_input)))
    if value != expected:
        raise UiResultValidationError("Metrics do not match the final debate outputs.")
    return copy.deepcopy(cast(ScorecardData, expected))


def calculate_ui_generated_words(
    *,
    mode: object,
    decomposition: dict,
    rounds: list[DebateRound],
    products: UiTeamWorkProducts,
    synthesis: str,
) -> int:
    total = count_generated_words(decomposition["question_complexity"]) + sum(
        count_generated_words(item) for item in decomposition["subquestions"]
    )
    if mode == "hierarchical":
        for product in (products["team_a"], products["team_b"]):
            if product is None:
                raise UiResultValidationError("Hierarchical work product is missing.")
            total += _count_strings(product["evidence_analysis"])
            total += count_generated_words(product["draft_argument"])
            total += _count_strings(product["internal_critique"])
            total += count_generated_words(product["final_argument"])
    else:
        total += sum(
            count_generated_words(item["team_a_argument"])
            + count_generated_words(item["team_b_argument"])
            for item in rounds
        )
    return total + count_generated_words(synthesis)


def _count_strings(value: object) -> int:
    if isinstance(value, str):
        return count_generated_words(value)
    if isinstance(value, dict):
        return sum(_count_strings(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_strings(item) for item in value)
    return 0


def _pricing_from_data(data: PricingData) -> Pricing:
    return Pricing(
        decimal_from_string(data["input_cost_per_million"]),
        decimal_from_string(data["output_cost_per_million"]),
    )


def _validate_json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UiResultValidationError(f"{label} is invalid.")
    _ensure_finite(value)
    return copy.deepcopy(cast(dict[str, object], value))


def _ensure_finite(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise UiResultValidationError("Nonfinite value is not allowed.")
        return
    if isinstance(value, list):
        for item in value:
            _ensure_finite(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _ensure_finite(item)
        return
    raise UiResultValidationError("Non-JSON value is not allowed.")


def _finite_nonnegative(value: object) -> float:
    if type(value) not in (int, float) or not isfinite(float(value)) or float(value) < 0:
        raise UiResultValidationError("Duration is invalid.")
    return float(value)


def _exact(value: object, schema: type[TypedDict], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(schema.__required_keys__):
        raise UiResultValidationError(f"{label} has invalid fields.")
    return value


def _finding_key(item: dict) -> tuple:
    return tuple(item[field] for field in (
        "subquestion_index", "subquestion", "finding_index", "claim",
        "source_title", "source_url", "source_type", "supporting_excerpt",
    ))


def _assert_forbidden_fields_absent(value: object) -> None:
    forbidden = {
        "api_key",
        "authorization",
        "environment",
        "secrets",
        "system_prompt",
        "prompt",
        "messages",
        "raw_response",
        "response_metadata",
        "chain_of_thought",
        "hidden_reasoning",
        "session_id",
    }
    if isinstance(value, dict):
        if any(str(key).casefold() in forbidden for key in value):
            raise UiResultValidationError("UI result contains a forbidden field.")
        for item in value.values():
            _assert_forbidden_fields_absent(item)
    elif isinstance(value, list):
        for item in value:
            _assert_forbidden_fields_absent(item)


def _contains_string(value: object, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(_contains_string(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains_string(item, needle) for item in value)
    return False


def _serialize(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
