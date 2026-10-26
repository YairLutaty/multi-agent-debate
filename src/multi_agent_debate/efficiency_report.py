"""Validated cost and efficiency records for controlled architecture replay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime
from decimal import Decimal
from typing import Literal, TypedDict, cast

from multi_agent_debate.costs import (
    CostEstimate,
    CostValidationError,
    Pricing,
    PricingData,
    calculate_cost,
    decimal_from_string,
    decimal_to_string,
    pricing_to_data,
    validate_cost_estimate,
    validate_pricing_data,
)
from multi_agent_debate.metrics import METRIC_VERSION, Scorecard, scorecard_to_json, tokenize_words
from multi_agent_debate.replay import ReplaySource, validate_replay_source
from multi_agent_debate.run_record import (
    ScorecardData,
    format_utc_timestamp,
    normalize_implementation_commit,
    score_expanded_result,
)
from multi_agent_debate.usage import (
    USAGE_EXTRACTION_VERSION,
    ModelUsageEntry,
    UsageError,
    UsageTotals,
    aggregate_usage,
    validate_usage_entry,
    validate_usage_totals,
)

EFFICIENCY_REPORT_VERSION = "1.0"
VARIANT_IDS = (
    "single_round_one_shot",
    "two_round_one_shot",
    "hierarchical_single_round",
)
VariantId = Literal[
    "single_round_one_shot",
    "two_round_one_shot",
    "hierarchical_single_round",
]

ARCHITECTURES: dict[VariantId, str] = {
    "single_round_one_shot": "one-shot-single-round-replay",
    "two_round_one_shot": "one-shot-two-round-replay",
    "hierarchical_single_round": "hierarchical-single-round-replay",
}
EXPECTED_CALL_BREAKDOWNS: dict[VariantId, dict[str, int]] = {
    "single_round_one_shot": {
        "team_a_argument": 1,
        "team_b_argument": 1,
        "neutral_synthesis": 1,
        "replay_decomposition": 0,
        "replay_research": 0,
        "total_model_calls": 3,
    },
    "two_round_one_shot": {
        "team_a_argument": 2,
        "team_b_argument": 2,
        "neutral_synthesis": 1,
        "replay_decomposition": 0,
        "replay_research": 0,
        "total_model_calls": 5,
    },
    "hierarchical_single_round": {
        "team_a_analysis": 1,
        "team_a_draft": 1,
        "team_a_review": 1,
        "team_b_analysis": 1,
        "team_b_draft": 1,
        "team_b_review": 1,
        "neutral_synthesis": 1,
        "replay_decomposition": 0,
        "replay_research": 0,
        "total_model_calls": 7,
    },
}
LIMITATIONS = [
    "Local zero pricing excludes hardware, electricity, and personnel cost.",
    "User-supplied rates are hypothetical estimates, not provider pricing claims.",
    "Token availability depends on metadata returned by Ollama and LangChain.",
    "One controlled topic is not statistically representative.",
    "Metric v1.0 does not measure factual correctness or persuasiveness.",
]


class EfficiencyReportError(RuntimeError):
    """Base bounded efficiency-report error."""


class EfficiencyReportValidationError(EfficiencyReportError):
    """A report or derived statistic is invalid or inconsistent."""


class GeneratedStageWordCount(TypedDict):
    stage: str
    team: str
    round_number: int | None
    word_count: int


class FinalVisibleOutputs(TypedDict):
    team_a_argument: str
    team_b_argument: str
    neutral_synthesis: str


class EfficiencyMeasurements(TypedDict):
    model_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    tokens_per_call: float | None
    visible_output_words: int
    all_generated_words: int
    tokens_per_visible_output_word: float | None
    latency_seconds: float
    completed_rounds: int


class MetricTokenEfficiency(TypedDict):
    research_utilization_per_1k_tokens: float | None
    team_novelty_per_1k_tokens: float | None
    synthesis_representation_per_1k_tokens: float | None
    summary_policy_compliance: int


class EfficiencyVariant(TypedDict):
    variant_id: VariantId
    architecture: str
    completed_rounds: int
    termination_reason: str | None
    model_call_breakdown: dict[str, int]
    usage_entries: list[ModelUsageEntry]
    usage_totals: UsageTotals
    generated_stage_word_counts: list[GeneratedStageWordCount]
    final_visible_outputs: FinalVisibleOutputs
    efficiency: EfficiencyMeasurements
    cost_estimate: CostEstimate
    metrics: ScorecardData
    metric_token_efficiency: MetricTokenEfficiency


class EfficiencyProvenance(TypedDict):
    run_timestamp_utc: str
    implementation_commit: str
    model: str
    temperature: float
    reasoning: bool
    metric_version: str
    usage_extraction_version: str
    efficiency_report_version: str


class EfficiencyReplaySource(TypedDict):
    source_artifact_path: str
    source_sha256: str
    question: str
    team_a_stance: str
    team_b_stance: str
    question_complexity: str
    subquestions: list[str]
    decomposition_sha256: str
    consolidated_evidence_sha256: str


class ComparisonDelta(TypedDict):
    baseline_variant: VariantId
    target_variant: VariantId
    model_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    estimated_cost: str | None
    latency_seconds: float
    research_utilization: float
    team_novelty: float
    synthesis_representation: float
    summary_policy_compliance: int


class EfficiencyReport(TypedDict):
    efficiency_report_version: str
    provenance: EfficiencyProvenance
    replay_source: EfficiencyReplaySource
    pricing: PricingData
    variants: list[EfficiencyVariant]
    comparison: list[ComparisonDelta]
    limitations: list[str]


def count_generated_words(value: str) -> int:
    if not isinstance(value, str):
        raise EfficiencyReportValidationError("generated output must be text")
    return len(tokenize_words(value))


def calculate_efficiency(
    *,
    usage_totals: UsageTotals,
    visible_output_words: int,
    all_generated_words: int,
    latency_seconds: float,
    completed_rounds: int,
) -> EfficiencyMeasurements:
    _validate_usage_totals_shape(usage_totals)
    visible = _strict_nonnegative_int(visible_output_words, "visible output words")
    generated = _strict_nonnegative_int(all_generated_words, "all generated words")
    latency = _finite_nonnegative(latency_seconds, "latency")
    rounds = _strict_positive_int(completed_rounds, "completed rounds")
    complete = usage_totals["all_usage_available"]
    total = usage_totals["total_tokens"] if complete else None
    return {
        "model_calls": usage_totals["model_calls"],
        "input_tokens": usage_totals["input_tokens"] if complete else None,
        "output_tokens": usage_totals["output_tokens"] if complete else None,
        "total_tokens": total,
        "tokens_per_call": (
            total / usage_totals["model_calls"]
            if total is not None and usage_totals["model_calls"] > 0
            else None
        ),
        "visible_output_words": visible,
        "all_generated_words": generated,
        "tokens_per_visible_output_word": (
            total / visible if total is not None and visible > 0 else None
        ),
        "latency_seconds": latency,
        "completed_rounds": rounds,
    }


def calculate_metric_token_efficiency(
    metrics: ScorecardData,
    *,
    total_tokens: int | None,
) -> MetricTokenEfficiency:
    validated_metrics = _validate_scorecard_shape(metrics)
    denominator = (
        total_tokens / 1000
        if type(total_tokens) is int and total_tokens > 0
        else None
    )
    if total_tokens is not None and (type(total_tokens) is not int or total_tokens < 0):
        raise EfficiencyReportValidationError("total tokens are invalid")
    return {
        "research_utilization_per_1k_tokens": (
            validated_metrics["research_utilization"]["score"] / denominator
            if denominator is not None
            else None
        ),
        "team_novelty_per_1k_tokens": (
            validated_metrics["team_novelty"]["score"] / denominator
            if denominator is not None
            else None
        ),
        "synthesis_representation_per_1k_tokens": (
            validated_metrics["synthesis_representation"]["score"] / denominator
            if denominator is not None
            else None
        ),
        "summary_policy_compliance": validated_metrics[
            "summary_policy_compliance"
        ]["score"],
    }


def build_efficiency_variant(
    *,
    variant_id: VariantId,
    completed_rounds: int,
    termination_reason: str | None,
    model_call_breakdown: dict[str, int],
    usage_entries: list[ModelUsageEntry],
    generated_stage_word_counts: list[GeneratedStageWordCount],
    final_visible_outputs: FinalVisibleOutputs,
    latency_seconds: float,
    pricing: Pricing,
    scorecard: Scorecard,
) -> EfficiencyVariant:
    metrics = cast(ScorecardData, json.loads(scorecard_to_json(scorecard)))
    raw: EfficiencyVariant = {
        "variant_id": variant_id,
        "architecture": ARCHITECTURES[variant_id],
        "completed_rounds": completed_rounds,
        "termination_reason": termination_reason,
        "model_call_breakdown": copy.deepcopy(model_call_breakdown),
        "usage_entries": copy.deepcopy(usage_entries),
        "usage_totals": aggregate_usage(usage_entries),
        "generated_stage_word_counts": copy.deepcopy(generated_stage_word_counts),
        "final_visible_outputs": copy.deepcopy(final_visible_outputs),
        "efficiency": cast(EfficiencyMeasurements, {}),
        "cost_estimate": cast(CostEstimate, {}),
        "metrics": metrics,
        "metric_token_efficiency": cast(MetricTokenEfficiency, {}),
    }
    visible_words = sum(count_generated_words(text) for text in final_visible_outputs.values())
    all_words = sum(item["word_count"] for item in generated_stage_word_counts)
    raw["efficiency"] = calculate_efficiency(
        usage_totals=raw["usage_totals"],
        visible_output_words=visible_words,
        all_generated_words=all_words,
        latency_seconds=latency_seconds,
        completed_rounds=completed_rounds,
    )
    raw["cost_estimate"] = calculate_cost(raw["usage_totals"], pricing=pricing)
    raw["metric_token_efficiency"] = calculate_metric_token_efficiency(
        metrics,
        total_tokens=raw["efficiency"]["total_tokens"],
    )
    return validate_efficiency_variant(raw, pricing=pricing)


def validate_efficiency_variant(
    value: object,
    *,
    pricing: Pricing,
    model_name: str | None = None,
) -> EfficiencyVariant:
    data = _require_exact(value, EfficiencyVariant, "efficiency variant")
    variant = data["variant_id"]
    if variant not in VARIANT_IDS:
        raise EfficiencyReportValidationError("efficiency variant ID is invalid")
    variant = cast(VariantId, variant)
    if data["architecture"] != ARCHITECTURES[variant]:
        raise EfficiencyReportValidationError("variant architecture is invalid")
    rounds = _strict_positive_int(data["completed_rounds"], "completed rounds")
    expected_rounds = 2 if variant == "two_round_one_shot" else 1
    if rounds != expected_rounds:
        raise EfficiencyReportValidationError("variant round count is invalid")
    reason = data["termination_reason"]
    if variant == "hierarchical_single_round":
        if reason is not None:
            raise EfficiencyReportValidationError("hierarchical termination is invalid")
    elif reason not in ("low_novelty", "max_rounds"):
        raise EfficiencyReportValidationError("one-shot termination is invalid")
    breakdown = data["model_call_breakdown"]
    if breakdown != EXPECTED_CALL_BREAKDOWNS[variant] or any(
        type(item) is not int for item in breakdown.values()
    ):
        raise EfficiencyReportValidationError("model call breakdown is invalid")

    if type(data["usage_entries"]) is not list:
        raise EfficiencyReportValidationError("usage entries must be a list")
    try:
        entries = [validate_usage_entry(item) for item in data["usage_entries"]]
    except UsageError:
        raise EfficiencyReportValidationError("usage entries are invalid") from None
    if len(entries) != breakdown["total_model_calls"]:
        raise EfficiencyReportValidationError("usage entry count does not match calls")
    if any(entry["variant"] != variant for entry in entries):
        raise EfficiencyReportValidationError("usage entries have the wrong variant")
    if model_name is not None and any(entry["model"] != model_name for entry in entries):
        raise EfficiencyReportValidationError("usage entries have the wrong model")
    _validate_usage_stage_counts(variant, entries)
    try:
        totals = validate_usage_totals(data["usage_totals"], entries=entries)
    except UsageError:
        raise EfficiencyReportValidationError("usage totals are invalid") from None

    word_counts = _validate_generated_word_counts(
        data["generated_stage_word_counts"], variant=variant
    )
    outputs = _validate_final_outputs(data["final_visible_outputs"])
    expected_efficiency = calculate_efficiency(
        usage_totals=totals,
        visible_output_words=sum(count_generated_words(text) for text in outputs.values()),
        all_generated_words=sum(item["word_count"] for item in word_counts),
        latency_seconds=_efficiency_latency(data["efficiency"]),
        completed_rounds=rounds,
    )
    if data["efficiency"] != expected_efficiency:
        raise EfficiencyReportValidationError("efficiency measurements are inconsistent")
    try:
        cost = validate_cost_estimate(data["cost_estimate"], totals=totals, pricing=pricing)
    except CostValidationError:
        raise EfficiencyReportValidationError("cost estimate is inconsistent") from None
    metrics = _validate_scorecard_shape(data["metrics"])
    expected_ratios = calculate_metric_token_efficiency(
        metrics,
        total_tokens=expected_efficiency["total_tokens"],
    )
    if data["metric_token_efficiency"] != expected_ratios:
        raise EfficiencyReportValidationError("metric token efficiency is inconsistent")
    _ensure_finite(data)
    return {
        "variant_id": variant,
        "architecture": ARCHITECTURES[variant],
        "completed_rounds": rounds,
        "termination_reason": cast(str | None, reason),
        "model_call_breakdown": breakdown.copy(),
        "usage_entries": copy.deepcopy(entries),
        "usage_totals": totals,
        "generated_stage_word_counts": copy.deepcopy(word_counts),
        "final_visible_outputs": outputs,
        "efficiency": expected_efficiency,
        "cost_estimate": cost,
        "metrics": copy.deepcopy(metrics),
        "metric_token_efficiency": expected_ratios,
    }


def build_efficiency_report(
    *,
    replay_source: ReplaySource,
    variants: list[EfficiencyVariant],
    pricing: Pricing,
    run_timestamp_utc: str,
    implementation_commit: str,
    model_name: str,
) -> EfficiencyReport:
    source = validate_replay_source(replay_source)
    source_record = source["source_record"]
    report: EfficiencyReport = {
        "efficiency_report_version": EFFICIENCY_REPORT_VERSION,
        "provenance": {
            "run_timestamp_utc": run_timestamp_utc,
            "implementation_commit": normalize_implementation_commit(
                implementation_commit
            ),
            "model": model_name,
            "temperature": 0.0,
            "reasoning": False,
            "metric_version": METRIC_VERSION,
            "usage_extraction_version": USAGE_EXTRACTION_VERSION,
            "efficiency_report_version": EFFICIENCY_REPORT_VERSION,
        },
        "replay_source": {
            "source_artifact_path": source["source_artifact_path"],
            "source_sha256": source["source_sha256"],
            "question": source_record["input"]["question"],
            "team_a_stance": source_record["input"]["team_a_stance"],
            "team_b_stance": source_record["input"]["team_b_stance"],
            "question_complexity": source_record["decomposition"][
                "question_complexity"
            ],
            "subquestions": source_record["decomposition"]["subquestions"].copy(),
            "decomposition_sha256": _json_sha256(source_record["decomposition"]),
            "consolidated_evidence_sha256": _json_sha256(
                source_record["graph_evidence"]["consolidated_evidence"]
            ),
        },
        "pricing": pricing_to_data(pricing),
        "variants": copy.deepcopy(variants),
        "comparison": [],
        "limitations": LIMITATIONS.copy(),
    }
    by_id = {item["variant_id"]: item for item in variants}
    if set(by_id) != set(VARIANT_IDS):
        raise EfficiencyReportValidationError("report variants are incomplete")
    report["comparison"] = [
        _comparison_delta(
            by_id["single_round_one_shot"],
            by_id[target],
        )
        for target in ("two_round_one_shot", "hierarchical_single_round")
    ]
    return validate_efficiency_report(report, replay_source=source)


def validate_efficiency_report(
    value: object,
    *,
    replay_source: ReplaySource,
) -> EfficiencyReport:
    source = validate_replay_source(replay_source)
    data = _require_exact(value, EfficiencyReport, "efficiency report")
    if data["efficiency_report_version"] != EFFICIENCY_REPORT_VERSION:
        raise EfficiencyReportValidationError("efficiency report version is invalid")
    provenance = _validate_provenance(data["provenance"])
    source_data = _validate_source_data(data["replay_source"], source)
    try:
        pricing_data = validate_pricing_data(data["pricing"])
        pricing = Pricing(
            decimal_from_string(pricing_data["input_cost_per_million"]),
            decimal_from_string(pricing_data["output_cost_per_million"]),
        )
    except CostValidationError:
        raise EfficiencyReportValidationError("report pricing is invalid") from None
    if type(data["variants"]) is not list or len(data["variants"]) != 3:
        raise EfficiencyReportValidationError("report variants are invalid")
    variants = [
        validate_efficiency_variant(item, pricing=pricing, model_name=provenance["model"])
        for item in data["variants"]
    ]
    if [item["variant_id"] for item in variants] != list(VARIANT_IDS):
        raise EfficiencyReportValidationError("report variants are not canonical")
    _validate_variant_metrics(variants, source)
    expected_comparison = [
        _comparison_delta(variants[0], variants[1]),
        _comparison_delta(variants[0], variants[2]),
    ]
    if data["comparison"] != expected_comparison:
        raise EfficiencyReportValidationError("report comparison is inconsistent")
    if data["limitations"] != LIMITATIONS:
        raise EfficiencyReportValidationError("report limitations are invalid")
    _ensure_finite(data)
    return {
        "efficiency_report_version": EFFICIENCY_REPORT_VERSION,
        "provenance": provenance,
        "replay_source": source_data,
        "pricing": pricing_data,
        "variants": copy.deepcopy(variants),
        "comparison": copy.deepcopy(expected_comparison),
        "limitations": LIMITATIONS.copy(),
    }


def efficiency_report_to_json(value: object, *, replay_source: ReplaySource) -> str:
    validated = validate_efficiency_report(value, replay_source=replay_source)
    return json.dumps(
        validated,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _validate_variant_metrics(variants: list[EfficiencyVariant], source: ReplaySource) -> None:
    record = source["source_record"]
    base = {
        "question": record["input"]["question"],
        "team_a_stance": record["input"]["team_a_stance"],
        "team_b_stance": record["input"]["team_b_stance"],
        "question_complexity": record["decomposition"]["question_complexity"],
        "subquestions": record["decomposition"]["subquestions"].copy(),
        "raw_research_findings": copy.deepcopy(
            record["graph_evidence"]["raw_research_findings"]
        ),
        "consolidated_evidence": copy.deepcopy(
            record["graph_evidence"]["consolidated_evidence"]
        ),
        "research_findings": record["graph_evidence"]["research_findings"].copy(),
    }
    for variant in variants:
        outputs = variant["final_visible_outputs"]
        result = {
            **copy.deepcopy(base),
            "team_a_argument": outputs["team_a_argument"],
            "team_b_argument": outputs["team_b_argument"],
            "synthesis": outputs["neutral_synthesis"],
        }
        expected = json.loads(scorecard_to_json(score_expanded_result(result)))
        if variant["metrics"] != expected:
            raise EfficiencyReportValidationError("variant metrics are inconsistent")


def _validate_usage_stage_counts(variant: VariantId, entries: list[ModelUsageEntry]) -> None:
    observed: dict[tuple[str, str | None], int] = {}
    for entry in entries:
        key = (entry["stage"], entry["team"])
        observed[key] = observed.get(key, 0) + 1
    if variant == "single_round_one_shot":
        expected = {
            ("team_a_argument", "team_a"): 1,
            ("team_b_argument", "team_b"): 1,
            ("neutral_synthesis", "system"): 1,
        }
    elif variant == "two_round_one_shot":
        expected = {
            ("team_a_argument", "team_a"): 2,
            ("team_b_argument", "team_b"): 2,
            ("neutral_synthesis", "system"): 1,
        }
    else:
        expected = {
            (stage, team): 1
            for team in ("team_a", "team_b")
            for stage in ("evidence_analysis", "argument_draft", "review_revision")
        }
        expected[("neutral_synthesis", "system")] = 1
    if observed != expected:
        raise EfficiencyReportValidationError("usage stages do not match variant calls")


def _validate_generated_word_counts(
    value: object,
    *,
    variant: VariantId,
) -> list[GeneratedStageWordCount]:
    if type(value) is not list:
        raise EfficiencyReportValidationError("generated word counts must be a list")
    copied = []
    for item in value:
        data = _require_exact(item, GeneratedStageWordCount, "generated word count")
        stage = data["stage"]
        team = data["team"]
        if not isinstance(stage, str) or not stage or team not in (
            "team_a", "team_b", "system"
        ):
            raise EfficiencyReportValidationError("generated word-count identity is invalid")
        round_number = data["round_number"]
        if round_number is not None and (
            type(round_number) is not int or round_number <= 0
        ):
            raise EfficiencyReportValidationError("generated word-count round is invalid")
        copied.append(
            {
                "stage": stage,
                "team": team,
                "round_number": round_number,
                "word_count": _strict_nonnegative_int(data["word_count"], "word count"),
            }
        )
    if variant == "single_round_one_shot":
        expected_identities = [
            ("team_a_argument", "team_a", 1),
            ("team_b_argument", "team_b", 1),
            ("neutral_synthesis", "system", 1),
        ]
    elif variant == "two_round_one_shot":
        expected_identities = [
            ("team_a_argument", "team_a", 1),
            ("team_b_argument", "team_b", 1),
            ("team_a_argument", "team_a", 2),
            ("team_b_argument", "team_b", 2),
            ("neutral_synthesis", "system", 2),
        ]
    else:
        expected_identities = [
            (stage, team, 1)
            for team in ("team_a", "team_b")
            for stage in (
                "evidence_analysis",
                "argument_draft",
                "review_revision",
            )
        ]
        expected_identities.append(("neutral_synthesis", "system", 1))
    observed_identities = [
        (item["stage"], item["team"], item["round_number"])
        for item in copied
    ]
    if observed_identities != expected_identities:
        raise EfficiencyReportValidationError(
            "generated word-count stages are incomplete"
        )
    return copied


def _validate_final_outputs(value: object) -> FinalVisibleOutputs:
    data = _require_exact(value, FinalVisibleOutputs, "final visible outputs")
    copied = {}
    for field in FinalVisibleOutputs.__required_keys__:
        text = data[field]
        if not isinstance(text, str) or not text.strip():
            raise EfficiencyReportValidationError("final visible output is invalid")
        copied[field] = text
    return cast(FinalVisibleOutputs, copied)


def _comparison_delta(
    baseline: EfficiencyVariant,
    target: EfficiencyVariant,
) -> ComparisonDelta:
    baseline_efficiency = baseline["efficiency"]
    target_efficiency = target["efficiency"]
    baseline_cost = baseline["cost_estimate"]["estimated_cost"]
    target_cost = target["cost_estimate"]["estimated_cost"]
    return {
        "baseline_variant": baseline["variant_id"],
        "target_variant": target["variant_id"],
        "model_calls": target_efficiency["model_calls"] - baseline_efficiency["model_calls"],
        "input_tokens": _optional_int_delta(target_efficiency["input_tokens"], baseline_efficiency["input_tokens"]),
        "output_tokens": _optional_int_delta(target_efficiency["output_tokens"], baseline_efficiency["output_tokens"]),
        "total_tokens": _optional_int_delta(target_efficiency["total_tokens"], baseline_efficiency["total_tokens"]),
        "estimated_cost": _optional_decimal_delta(target_cost, baseline_cost),
        "latency_seconds": target_efficiency["latency_seconds"] - baseline_efficiency["latency_seconds"],
        "research_utilization": target["metrics"]["research_utilization"]["score"] - baseline["metrics"]["research_utilization"]["score"],
        "team_novelty": target["metrics"]["team_novelty"]["score"] - baseline["metrics"]["team_novelty"]["score"],
        "synthesis_representation": target["metrics"]["synthesis_representation"]["score"] - baseline["metrics"]["synthesis_representation"]["score"],
        "summary_policy_compliance": target["metrics"]["summary_policy_compliance"]["score"] - baseline["metrics"]["summary_policy_compliance"]["score"],
    }


def _validate_provenance(value: object) -> EfficiencyProvenance:
    data = _require_exact(value, EfficiencyProvenance, "efficiency provenance")
    try:
        timestamp = format_utc_timestamp(
            datetime.fromisoformat(data["run_timestamp_utc"][:-1] + "+00:00")
        )
        commit = normalize_implementation_commit(data["implementation_commit"])
    except Exception:
        raise EfficiencyReportValidationError("efficiency provenance is invalid") from None
    if timestamp != data["run_timestamp_utc"] or commit != data["implementation_commit"]:
        raise EfficiencyReportValidationError("efficiency provenance is not canonical")
    if not isinstance(data["model"], str) or not data["model"].strip():
        raise EfficiencyReportValidationError("efficiency model is invalid")
    if (
        type(data["temperature"]) is not float
        or data["temperature"] != 0.0
        or data["reasoning"] is not False
        or data["metric_version"] != METRIC_VERSION
        or data["usage_extraction_version"] != USAGE_EXTRACTION_VERSION
        or data["efficiency_report_version"] != EFFICIENCY_REPORT_VERSION
    ):
        raise EfficiencyReportValidationError("efficiency provenance settings are invalid")
    return cast(EfficiencyProvenance, copy.deepcopy(data))


def _validate_source_data(value: object, source: ReplaySource) -> EfficiencyReplaySource:
    data = _require_exact(value, EfficiencyReplaySource, "efficiency replay source")
    record = source["source_record"]
    expected: EfficiencyReplaySource = {
        "source_artifact_path": source["source_artifact_path"],
        "source_sha256": source["source_sha256"],
        "question": record["input"]["question"],
        "team_a_stance": record["input"]["team_a_stance"],
        "team_b_stance": record["input"]["team_b_stance"],
        "question_complexity": record["decomposition"]["question_complexity"],
        "subquestions": record["decomposition"]["subquestions"].copy(),
        "decomposition_sha256": _json_sha256(record["decomposition"]),
        "consolidated_evidence_sha256": _json_sha256(
            record["graph_evidence"]["consolidated_evidence"]
        ),
    }
    if data != expected:
        raise EfficiencyReportValidationError("efficiency replay source is inconsistent")
    return copy.deepcopy(expected)


def _validate_scorecard_shape(value: object) -> ScorecardData:
    if not isinstance(value, dict) or set(value) != set(ScorecardData.__required_keys__):
        raise EfficiencyReportValidationError("metric scorecard has invalid fields")
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        copied = json.loads(encoded)
    except (TypeError, ValueError):
        raise EfficiencyReportValidationError("metric scorecard is invalid") from None
    return cast(ScorecardData, copied)


def _efficiency_latency(value: object) -> float:
    data = _require_exact(value, EfficiencyMeasurements, "efficiency measurements")
    return _finite_nonnegative(data["latency_seconds"], "latency")


def _validate_usage_totals_shape(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(UsageTotals.__required_keys__):
        raise EfficiencyReportValidationError("usage totals are invalid")
    for field in ("model_calls", "available_calls", "unavailable_calls", "input_tokens", "output_tokens", "total_tokens"):
        _strict_nonnegative_int(value[field], "usage total")
    if type(value["all_usage_available"]) is not bool:
        raise EfficiencyReportValidationError("usage totals are invalid")
    if value["total_tokens"] != value["input_tokens"] + value["output_tokens"]:
        raise EfficiencyReportValidationError("usage totals are inconsistent")


def _optional_int_delta(target: int | None, baseline: int | None) -> int | None:
    return target - baseline if target is not None and baseline is not None else None


def _optional_decimal_delta(target: str | None, baseline: str | None) -> str | None:
    if target is None or baseline is None:
        return None
    try:
        return decimal_to_string(decimal_from_string(target) - decimal_from_string(baseline))
    except CostValidationError:
        raise EfficiencyReportValidationError("cost delta is invalid") from None


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact(value: object, schema: type[TypedDict], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(schema.__required_keys__):
        raise EfficiencyReportValidationError(f"{label} has invalid fields")
    return value


def _strict_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EfficiencyReportValidationError(f"{label} must be a nonnegative integer")
    return value


def _strict_positive_int(value: object, label: str) -> int:
    resolved = _strict_nonnegative_int(value, label)
    if resolved == 0:
        raise EfficiencyReportValidationError(f"{label} must be positive")
    return resolved


def _finite_nonnegative(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise EfficiencyReportValidationError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise EfficiencyReportValidationError(f"{label} must be finite and nonnegative")
    return resolved


def _ensure_finite(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise EfficiencyReportValidationError("efficiency data is not finite JSON") from None
