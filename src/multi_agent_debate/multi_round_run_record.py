"""Strict version 2.0 records for the final grounded debate product."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from math import isfinite
from typing import Literal, TypedDict, cast

from multi_agent_debate.costs import (
    CostEstimate,
    CostValidationError,
    Pricing,
    calculate_cost,
    decimal_from_string,
    validate_cost_estimate,
)
from multi_agent_debate.efficiency_report import (
    EfficiencyMeasurements,
    MetricTokenEfficiency,
    calculate_efficiency,
    calculate_metric_token_efficiency,
    count_generated_words,
)

from multi_agent_debate.expanded_state import ExpandedDebateResult
from multi_agent_debate.metrics import METRIC_VERSION, Scorecard, scorecard_to_json
from multi_agent_debate.multi_round_graph import validate_multi_round_result
from multi_agent_debate.multi_round_state import DebateRound, MultiRoundDebateResult
from multi_agent_debate.multi_round_trace import (
    MultiRoundTrace,
    MultiRoundTraceRecorder,
    canonicalize_successful_traces,
    validate_multi_round_trace,
)
from multi_agent_debate.research import (
    ResearchFinding,
    finding_sort_key,
    validate_research_finding,
)
from multi_agent_debate.run_record import (
    ModelConfiguration,
    ScorecardData,
    SearchCallEvidence,
    format_utc_timestamp,
    normalize_implementation_commit,
    sanitized_search_settings,
    score_expanded_result,
)
from multi_agent_debate.run_trace import (
    BackendSearchTrace,
    ModelStageTrace,
    ResearchTrace,
    RunTraceRecorder,
)
from multi_agent_debate.tavily_search import TAVILY_SEARCH_URL
from multi_agent_debate.termination import (
    TerminationDecision,
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

RUN_RECORD_VERSION = "2.0"
RUN_RECORD_VERSION_2_1 = "2.1"
ARCHITECTURE_VERSION = "final-multi-round-grounded"
SEARCH_PROVIDER = "tavily"
SEARCH_TOOL = "search_web"

ProductMode = Literal["single-round", "multi-round"]


class MultiRoundRunRecordError(RuntimeError):
    """Base error for final-product run records."""


class MultiRoundRunRecordValidationError(MultiRoundRunRecordError):
    """Raised when a version 2.0 record violates its strict contract."""


class ProductSearchConfiguration(TypedDict):
    provider: str
    tool: str
    endpoint: str
    max_results: int
    settings: dict[str, object]


class ProductProvenance(TypedDict):
    run_timestamp_utc: str
    implementation_commit: str
    architecture_version: str
    metric_version: str
    product_mode: ProductMode
    model: ModelConfiguration
    search: ProductSearchConfiguration


class ProductInput(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str
    max_rounds: int
    novelty_threshold: float


class ProductDecomposition(TypedDict):
    question_complexity: Literal["simple", "complex"]
    subquestions: list[str]


class ProductGraphEvidence(TypedDict):
    raw_research_findings: list[ResearchFinding]
    consolidated_evidence: list[ResearchFinding]
    research_findings: list[str]
    final_team_a_argument: str
    final_team_b_argument: str
    neutral_synthesis: str


class ProductTimings(TypedDict):
    decomposition: ModelStageTrace
    research: list[ResearchTrace]
    debate: list[MultiRoundTrace]
    total_duration_seconds: float


class ProductModelCallCounts(TypedDict):
    decomposition: int
    team_a_argument: int
    team_b_argument: int
    synthesis: int
    research_provider_calls: int
    total_model_calls: int


class MultiRoundRunRecord(TypedDict):
    run_record_version: str
    provenance: ProductProvenance
    input: ProductInput
    decomposition: ProductDecomposition
    search_evidence: list[SearchCallEvidence]
    graph_evidence: ProductGraphEvidence
    debate_rounds: list[DebateRound]
    termination: TerminationDecision
    timings: ProductTimings
    model_call_counts: ProductModelCallCounts
    metrics: ScorecardData


class ProductUsage(TypedDict):
    usage_extraction_version: str
    entries: list[ModelUsageEntry]
    totals: UsageTotals


class MultiRoundRunRecordV21(MultiRoundRunRecord):
    usage: ProductUsage
    cost_estimate: CostEstimate
    efficiency: EfficiencyMeasurements
    metric_token_efficiency: MetricTokenEfficiency


class GroundingTraces(TypedDict):
    backend_searches: list[BackendSearchTrace]
    research_calls: list[ResearchTrace]
    decomposition: ModelStageTrace


def score_multi_round_result(result: MultiRoundDebateResult) -> Scorecard:
    """Apply metric v1.0 only to consolidated claims and final outputs."""

    validated = validate_multi_round_result(result)
    metric_input: ExpandedDebateResult = {
        "question": validated["question"],
        "team_a_stance": validated["team_a_stance"],
        "team_b_stance": validated["team_b_stance"],
        "question_complexity": validated["question_complexity"],
        "subquestions": validated["subquestions"].copy(),
        "raw_research_findings": copy.deepcopy(validated["raw_research_findings"]),
        "consolidated_evidence": copy.deepcopy(validated["consolidated_evidence"]),
        "research_findings": validated["research_findings"].copy(),
        "team_a_argument": validated["team_a_argument"],
        "team_b_argument": validated["team_b_argument"],
        "synthesis": validated["synthesis"],
    }
    return score_expanded_result(metric_input)


def canonicalize_grounding_traces(
    result: MultiRoundDebateResult,
    recorder: RunTraceRecorder,
) -> GroundingTraces:
    """Order one-pass search traces by decomposition and require one decomposition."""

    validated = validate_multi_round_result(result)
    expected_tasks = _expected_tasks(validated)
    backend = _group_unique(
        recorder.backend_searches,
        expected=[query for _, query in expected_tasks],
        identity=lambda trace: trace["query"],
        context="backend search",
    )
    research = _group_unique(
        recorder.research_calls,
        expected=expected_tasks,
        identity=lambda trace: (trace["subquestion_index"], trace["query"]),
        context="research provider",
    )
    model_stages = recorder.model_stages
    if len(model_stages) != 1 or model_stages[0]["stage"] != "decomposition":
        raise MultiRoundRunRecordValidationError(
            "grounding traces require exactly one decomposition call"
        )
    decomposition = _validate_successful_model_trace(
        model_stages[0], expected_stage="decomposition"
    )
    return {
        "backend_searches": [copy.deepcopy(backend[query]) for _, query in expected_tasks],
        "research_calls": [copy.deepcopy(research[task]) for task in expected_tasks],
        "decomposition": decomposition,
    }


def build_multi_round_run_record(
    *,
    result: MultiRoundDebateResult,
    grounding_traces: GroundingTraces,
    debate_traces: list[MultiRoundTrace],
    scorecard: Scorecard,
    run_timestamp_utc: str,
    implementation_commit: str,
    model_name: str,
    max_results: int,
    product_mode: ProductMode,
    total_duration_seconds: float,
    usage_entries: list[ModelUsageEntry] | None = None,
    pricing: Pricing | None = None,
) -> MultiRoundRunRecord | MultiRoundRunRecordV21:
    """Build and cross-validate one complete successful product record."""

    validated = validate_multi_round_result(result)
    if product_mode not in ("single-round", "multi-round"):
        raise MultiRoundRunRecordValidationError("product mode is invalid")
    if product_mode == "single-round" and validated["max_rounds"] != 1:
        raise MultiRoundRunRecordValidationError(
            "single-round mode requires one maximum round"
        )
    if type(max_results) is not int or max_results <= 0:
        raise MultiRoundRunRecordValidationError("search maximum is invalid")
    if not isinstance(model_name, str) or not model_name.strip():
        raise MultiRoundRunRecordValidationError("model name is invalid")
    completed_rounds = len(validated["debate_rounds"])
    expected_debate_traces = _validate_canonical_debate_traces(
        debate_traces, completed_rounds=completed_rounds
    )
    search_evidence = _build_search_evidence(
        validated,
        grounding_traces,
        max_results=max_results,
    )
    metrics = cast(ScorecardData, json.loads(scorecard_to_json(scorecard)))
    record: MultiRoundRunRecord = {
        "run_record_version": RUN_RECORD_VERSION,
        "provenance": {
            "run_timestamp_utc": run_timestamp_utc,
            "implementation_commit": _normalize_commit(implementation_commit),
            "architecture_version": ARCHITECTURE_VERSION,
            "metric_version": METRIC_VERSION,
            "product_mode": product_mode,
            "model": {
                "name": model_name,
                "temperature": 0.0,
                "reasoning": False,
                "structured_output_method": "json_schema",
            },
            "search": {
                "provider": SEARCH_PROVIDER,
                "tool": SEARCH_TOOL,
                "endpoint": TAVILY_SEARCH_URL,
                "max_results": max_results,
                "settings": copy.deepcopy(sanitized_search_settings()),
            },
        },
        "input": {
            "question": validated["question"],
            "team_a_stance": validated["team_a_stance"],
            "team_b_stance": validated["team_b_stance"],
            "max_rounds": validated["max_rounds"],
            "novelty_threshold": validated["novelty_threshold"],
        },
        "decomposition": {
            "question_complexity": validated["question_complexity"],
            "subquestions": validated["subquestions"].copy(),
        },
        "search_evidence": search_evidence,
        "graph_evidence": {
            "raw_research_findings": copy.deepcopy(validated["raw_research_findings"]),
            "consolidated_evidence": copy.deepcopy(validated["consolidated_evidence"]),
            "research_findings": validated["research_findings"].copy(),
            "final_team_a_argument": validated["team_a_argument"],
            "final_team_b_argument": validated["team_b_argument"],
            "neutral_synthesis": validated["synthesis"],
        },
        "debate_rounds": copy.deepcopy(validated["debate_rounds"]),
        "termination": copy.deepcopy(validated["termination_decision"]),
        "timings": {
            "decomposition": copy.deepcopy(grounding_traces["decomposition"]),
            "research": copy.deepcopy(grounding_traces["research_calls"]),
            "debate": copy.deepcopy(expected_debate_traces),
            "total_duration_seconds": float(total_duration_seconds),
        },
        "model_call_counts": {
            "decomposition": 1,
            "team_a_argument": completed_rounds,
            "team_b_argument": completed_rounds,
            "synthesis": 1,
            "research_provider_calls": len(_expected_tasks(validated)),
            "total_model_calls": 2 * completed_rounds + 2,
        },
        "metrics": metrics,
    }
    validated_v20 = cast(MultiRoundRunRecord, validate_multi_round_run_record(record))
    if usage_entries is None and pricing is None:
        return validated_v20
    if usage_entries is None or pricing is None:
        raise MultiRoundRunRecordValidationError(
            "usage entries and pricing must be supplied together"
        )
    entries = [validate_usage_entry(entry) for entry in usage_entries]
    totals = aggregate_usage(entries)
    visible_words = sum(
        count_generated_words(text)
        for text in (
            validated["team_a_argument"],
            validated["team_b_argument"],
            validated["synthesis"],
        )
    )
    generated_words = _all_product_generated_words(validated)
    v21: MultiRoundRunRecordV21 = {
        **copy.deepcopy(validated_v20),
        "run_record_version": RUN_RECORD_VERSION_2_1,
        "usage": {
            "usage_extraction_version": USAGE_EXTRACTION_VERSION,
            "entries": copy.deepcopy(entries),
            "totals": totals,
        },
        "cost_estimate": calculate_cost(totals, pricing=pricing),
        "efficiency": calculate_efficiency(
            usage_totals=totals,
            visible_output_words=visible_words,
            all_generated_words=generated_words,
            latency_seconds=total_duration_seconds,
            completed_rounds=completed_rounds,
        ),
        "metric_token_efficiency": calculate_metric_token_efficiency(
            metrics,
            total_tokens=(totals["total_tokens"] if totals["all_usage_available"] else None),
        ),
    }
    return validate_multi_round_run_record(v21)


def validate_multi_round_run_record(
    value: object,
) -> MultiRoundRunRecord | MultiRoundRunRecordV21:
    """Validate frozen v2.0 records or telemetry-bearing v2.1 records."""

    try:
        if isinstance(value, dict) and value.get("run_record_version") == RUN_RECORD_VERSION_2_1:
            return _validate_multi_round_run_record_v21(value)
        return _validate_multi_round_run_record(value)
    except MultiRoundRunRecordValidationError:
        raise
    except Exception:
        raise MultiRoundRunRecordValidationError(
            "multi-round run record is invalid"
        ) from None


def _validate_multi_round_run_record(value: object) -> MultiRoundRunRecord:
    """Implementation kept behind one bounded public validation boundary."""

    data = _require_dict(value, MultiRoundRunRecord, "multi-round run record")
    if data["run_record_version"] != RUN_RECORD_VERSION:
        raise MultiRoundRunRecordValidationError("run-record version is invalid")
    provenance = _validate_provenance(data["provenance"])
    run_input = _validate_input(data["input"], provenance["product_mode"])
    decomposition = _validate_decomposition(data["decomposition"])
    expected_tasks = (
        [(0, run_input["question"])]
        if decomposition["question_complexity"] == "simple"
        else list(enumerate(decomposition["subquestions"], start=1))
    )
    search_evidence, expected_raw = _validate_search_evidence(
        data["search_evidence"],
        expected_tasks=expected_tasks,
        max_results=provenance["search"]["max_results"],
    )
    graph_evidence = _validate_graph_evidence(data["graph_evidence"])
    if graph_evidence["raw_research_findings"] != expected_raw:
        raise MultiRoundRunRecordValidationError(
            "raw findings do not match recorded search evidence"
        )
    raw_rounds = data["debate_rounds"]
    if type(raw_rounds) is not list or any(
        not isinstance(item, dict) for item in raw_rounds
    ):
        raise MultiRoundRunRecordValidationError("debate rounds must be a list")
    synthetic_result: MultiRoundDebateResult = {
        "question": run_input["question"],
        "team_a_stance": run_input["team_a_stance"],
        "team_b_stance": run_input["team_b_stance"],
        "max_rounds": run_input["max_rounds"],
        "novelty_threshold": run_input["novelty_threshold"],
        "question_complexity": decomposition["question_complexity"],
        "subquestions": decomposition["subquestions"].copy(),
        "raw_research_findings": copy.deepcopy(graph_evidence["raw_research_findings"]),
        "consolidated_evidence": copy.deepcopy(graph_evidence["consolidated_evidence"]),
        "research_findings": graph_evidence["research_findings"].copy(),
        "current_round_number": len(raw_rounds),
        "team_a_history": [item.get("team_a_argument") for item in raw_rounds],
        "team_b_history": [item.get("team_b_argument") for item in raw_rounds],
        "debate_rounds": copy.deepcopy(raw_rounds),
        "termination_decision": copy.deepcopy(data["termination"]),
        "team_a_argument": graph_evidence["final_team_a_argument"],
        "team_b_argument": graph_evidence["final_team_b_argument"],
        "synthesis": graph_evidence["neutral_synthesis"],
        "synthesis_duration_seconds": _synthesis_duration_from_timings(
            data["timings"]
        ),
    }
    try:
        validated_result = validate_multi_round_result(synthetic_result)
    except Exception:
        raise MultiRoundRunRecordValidationError(
            "debate rounds or final outputs are inconsistent"
        ) from None
    if validate_termination_decision(data["termination"]) != validated_result[
        "termination_decision"
    ]:
        raise MultiRoundRunRecordValidationError("termination section is inconsistent")

    timings = _validate_timings(
        data["timings"],
        expected_tasks=expected_tasks,
        search_evidence=search_evidence,
        rounds=validated_result["debate_rounds"],
    )
    call_counts = _validate_call_counts(
        data["model_call_counts"],
        completed_rounds=len(validated_result["debate_rounds"]),
        research_calls=len(expected_tasks),
        debate_traces=timings["debate"],
    )
    metrics = _require_dict(data["metrics"], ScorecardData, "metrics")
    expected_metrics = json.loads(
        scorecard_to_json(score_multi_round_result(validated_result))
    )
    if _canonical_json(metrics) != _canonical_json(expected_metrics):
        raise MultiRoundRunRecordValidationError(
            "embedded metrics do not match metric version 1.0"
        )
    _ensure_json_finite(metrics, "metrics")
    return {
        "run_record_version": RUN_RECORD_VERSION,
        "provenance": provenance,
        "input": run_input,
        "decomposition": decomposition,
        "search_evidence": search_evidence,
        "graph_evidence": graph_evidence,
        "debate_rounds": copy.deepcopy(validated_result["debate_rounds"]),
        "termination": copy.deepcopy(validated_result["termination_decision"]),
        "timings": timings,
        "model_call_counts": call_counts,
        "metrics": copy.deepcopy(cast(ScorecardData, metrics)),
    }


def multi_round_run_record_to_json(value: object) -> str:
    validated = validate_multi_round_run_record(value)
    return json.dumps(
        validated,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _validate_multi_round_run_record_v21(value: object) -> MultiRoundRunRecordV21:
    if not isinstance(value, dict) or set(value) != set(
        MultiRoundRunRecordV21.__required_keys__
    ):
        raise MultiRoundRunRecordValidationError(
            "version 2.1 run record has invalid fields"
        )
    base = {
        key: copy.deepcopy(value[key])
        for key in MultiRoundRunRecord.__required_keys__
    }
    base["run_record_version"] = RUN_RECORD_VERSION
    validated_base = _validate_multi_round_run_record(base)
    usage_raw = value["usage"]
    if not isinstance(usage_raw, dict) or set(usage_raw) != set(
        ProductUsage.__required_keys__
    ):
        raise MultiRoundRunRecordValidationError("usage section has invalid fields")
    if usage_raw["usage_extraction_version"] != USAGE_EXTRACTION_VERSION:
        raise MultiRoundRunRecordValidationError("usage extraction version is invalid")
    if type(usage_raw["entries"]) is not list:
        raise MultiRoundRunRecordValidationError("usage entries must be a list")
    try:
        entries = [validate_usage_entry(item) for item in usage_raw["entries"]]
        totals = validate_usage_totals(usage_raw["totals"], entries=entries)
    except Exception:
        raise MultiRoundRunRecordValidationError("usage telemetry is invalid") from None
    _validate_product_usage_entries(entries, validated_base)
    if totals["model_calls"] != validated_base["model_call_counts"]["total_model_calls"]:
        raise MultiRoundRunRecordValidationError(
            "usage entry count does not match model calls"
        )
    try:
        cost_raw = value["cost_estimate"]
        if not isinstance(cost_raw, dict):
            raise CostValidationError
        pricing = Pricing(
            decimal_from_string(cost_raw.get("input_cost_per_million")),
            decimal_from_string(cost_raw.get("output_cost_per_million")),
        )
        cost = validate_cost_estimate(cost_raw, totals=totals, pricing=pricing)
    except Exception:
        raise MultiRoundRunRecordValidationError("cost estimate is invalid") from None
    expected_efficiency = calculate_efficiency(
        usage_totals=totals,
        visible_output_words=sum(
            count_generated_words(text)
            for text in (
                validated_base["graph_evidence"]["final_team_a_argument"],
                validated_base["graph_evidence"]["final_team_b_argument"],
                validated_base["graph_evidence"]["neutral_synthesis"],
            )
        ),
        all_generated_words=_all_record_generated_words(validated_base),
        latency_seconds=validated_base["timings"]["total_duration_seconds"],
        completed_rounds=len(validated_base["debate_rounds"]),
    )
    if value["efficiency"] != expected_efficiency:
        raise MultiRoundRunRecordValidationError("efficiency section is inconsistent")
    expected_ratios = calculate_metric_token_efficiency(
        validated_base["metrics"],
        total_tokens=expected_efficiency["total_tokens"],
    )
    if value["metric_token_efficiency"] != expected_ratios:
        raise MultiRoundRunRecordValidationError(
            "metric token efficiency is inconsistent"
        )
    return {
        **copy.deepcopy(validated_base),
        "run_record_version": RUN_RECORD_VERSION_2_1,
        "usage": {
            "usage_extraction_version": USAGE_EXTRACTION_VERSION,
            "entries": copy.deepcopy(entries),
            "totals": totals,
        },
        "cost_estimate": cost,
        "efficiency": copy.deepcopy(expected_efficiency),
        "metric_token_efficiency": copy.deepcopy(expected_ratios),
    }


def _validate_product_usage_entries(
    entries: list[ModelUsageEntry],
    record: MultiRoundRunRecord,
) -> None:
    rounds = len(record["debate_rounds"])
    expected = [("decomposition", "system", None)]
    expected.extend(
        (stage, team, round_number)
        for round_number in range(1, rounds + 1)
        for stage, team in (
            ("team_a_argument", "team_a"),
            ("team_b_argument", "team_b"),
        )
    )
    expected.append(("neutral_synthesis", "system", rounds))
    observed = [
        (entry["stage"], entry["team"], entry["round_number"])
        for entry in entries
    ]
    if observed != expected:
        raise MultiRoundRunRecordValidationError(
            "usage stages do not match graph traces"
        )
    model_name = record["provenance"]["model"]["name"]
    if any(
        entry["model"] != model_name or entry["variant"] != "final_product"
        for entry in entries
    ):
        raise MultiRoundRunRecordValidationError(
            "usage model or variant is inconsistent"
        )


def _all_product_generated_words(result: MultiRoundDebateResult) -> int:
    return (
        count_generated_words(result["question_complexity"])
        + sum(count_generated_words(item) for item in result["subquestions"])
        + sum(count_generated_words(item) for item in result["team_a_history"])
        + sum(count_generated_words(item) for item in result["team_b_history"])
        + count_generated_words(result["synthesis"])
    )


def _all_record_generated_words(record: MultiRoundRunRecord) -> int:
    return (
        count_generated_words(record["decomposition"]["question_complexity"])
        + sum(
            count_generated_words(item)
            for item in record["decomposition"]["subquestions"]
        )
        + sum(
            count_generated_words(round_record["team_a_argument"])
            + count_generated_words(round_record["team_b_argument"])
            for round_record in record["debate_rounds"]
        )
        + count_generated_words(record["graph_evidence"]["neutral_synthesis"])
    )


def canonicalize_product_debate_traces(
    *, recorder: MultiRoundTraceRecorder, completed_rounds: int
) -> list[MultiRoundTrace]:
    """Public convenience boundary used by the pure product executor."""

    return canonicalize_successful_traces(
        recorder=recorder, completed_rounds=completed_rounds
    )


def _build_search_evidence(
    result: MultiRoundDebateResult,
    traces: GroundingTraces,
    *,
    max_results: int,
) -> list[SearchCallEvidence]:
    expected_tasks = _expected_tasks(result)
    if len(traces["backend_searches"]) != len(expected_tasks) or len(
        traces["research_calls"]
    ) != len(expected_tasks):
        raise MultiRoundRunRecordValidationError("grounding traces are incomplete")
    _validate_successful_model_trace(
        traces["decomposition"], expected_stage="decomposition"
    )
    settings = sanitized_search_settings()
    calls: list[SearchCallEvidence] = []
    for task, backend, research in zip(
        expected_tasks,
        traces["backend_searches"],
        traces["research_calls"],
        strict=True,
    ):
        index, query = task
        if backend["query"] != query or (
            research["subquestion_index"], research["query"]
        ) != task:
            raise MultiRoundRunRecordValidationError(
                "grounding trace order is inconsistent"
            )
        if backend["status"] != "success" or research["status"] != "success":
            raise MultiRoundRunRecordValidationError(
                "completed record contains failed grounding traces"
            )
        if backend["error_category"] is not None or research["error_category"] is not None:
            raise MultiRoundRunRecordValidationError(
                "successful grounding traces contain error categories"
            )
        if backend["requested_max_results"] != max_results or backend["settings"] != settings:
            raise MultiRoundRunRecordValidationError(
                "grounding trace configuration is inconsistent"
            )
        results = []
        for rank, item in enumerate(backend["results"], start=1):
            if "provider_score" not in item:
                raise MultiRoundRunRecordValidationError(
                    "grounded evidence is missing provider score"
                )
            results.append(
                {
                    "rank": rank,
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                    "provider_score": item["provider_score"],
                    "source_type": "web",
                }
            )
        calls.append(
            {
                "subquestion_index": index,
                "query": query,
                "requested_max_results": max_results,
                "settings": copy.deepcopy(settings),
                "status": "success",
                "error_category": None,
                "backend_duration_seconds": backend["duration_seconds"],
                "research_duration_seconds": research["duration_seconds"],
                "results": results,
            }
        )
    return calls


def _validate_provenance(value: object) -> ProductProvenance:
    data = _require_dict(value, ProductProvenance, "provenance")
    timestamp = _validate_timestamp(data["run_timestamp_utc"])
    commit = _normalize_commit(data["implementation_commit"])
    if data["implementation_commit"] != commit:
        raise MultiRoundRunRecordValidationError("implementation commit must be lowercase")
    if (
        data["architecture_version"] != ARCHITECTURE_VERSION
        or data["metric_version"] != METRIC_VERSION
    ):
        raise MultiRoundRunRecordValidationError("provenance version is invalid")
    mode = data["product_mode"]
    if mode not in ("single-round", "multi-round"):
        raise MultiRoundRunRecordValidationError("product mode is invalid")
    model = _require_dict(data["model"], ModelConfiguration, "model configuration")
    if not isinstance(model["name"], str) or not model["name"].strip():
        raise MultiRoundRunRecordValidationError("model name is invalid")
    if type(model["temperature"]) is not float or model["temperature"] != 0.0:
        raise MultiRoundRunRecordValidationError("model temperature is invalid")
    if model["reasoning"] is not False or model["structured_output_method"] != "json_schema":
        raise MultiRoundRunRecordValidationError("model settings are invalid")
    search = _require_dict(data["search"], ProductSearchConfiguration, "search configuration")
    if (
        search["provider"] != SEARCH_PROVIDER
        or search["tool"] != SEARCH_TOOL
        or search["endpoint"] != TAVILY_SEARCH_URL
    ):
        raise MultiRoundRunRecordValidationError("search provenance is invalid")
    if type(search["max_results"]) is not int or search["max_results"] <= 0:
        raise MultiRoundRunRecordValidationError("search maximum is invalid")
    settings = search["settings"]
    expected_settings = sanitized_search_settings()
    if (
        settings != expected_settings
        or not isinstance(settings, dict)
        or set(settings) != set(expected_settings)
        or any(
            type(settings[field]) is not type(expected_settings[field])
            for field in expected_settings
        )
    ):
        raise MultiRoundRunRecordValidationError("search settings are invalid")
    return {
        "run_timestamp_utc": timestamp,
        "implementation_commit": commit,
        "architecture_version": ARCHITECTURE_VERSION,
        "metric_version": METRIC_VERSION,
        "product_mode": mode,
        "model": {
            "name": model["name"],
            "temperature": 0.0,
            "reasoning": False,
            "structured_output_method": "json_schema",
        },
        "search": {
            "provider": SEARCH_PROVIDER,
            "tool": SEARCH_TOOL,
            "endpoint": TAVILY_SEARCH_URL,
            "max_results": search["max_results"],
            "settings": copy.deepcopy(settings),
        },
    }


def _validate_input(value: object, mode: ProductMode) -> ProductInput:
    data = _require_dict(value, ProductInput, "input")
    for field in ("question", "team_a_stance", "team_b_stance"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise MultiRoundRunRecordValidationError(f"{field} is invalid")
    max_rounds = _policy_value(validate_max_rounds, data["max_rounds"])
    threshold = _policy_value(validate_novelty_threshold, data["novelty_threshold"])
    if mode == "single-round" and max_rounds != 1:
        raise MultiRoundRunRecordValidationError("single-round mode requires one round")
    return {
        "question": data["question"],
        "team_a_stance": data["team_a_stance"],
        "team_b_stance": data["team_b_stance"],
        "max_rounds": max_rounds,
        "novelty_threshold": threshold,
    }


def _validate_decomposition(value: object) -> ProductDecomposition:
    data = _require_dict(value, ProductDecomposition, "decomposition")
    complexity = data["question_complexity"]
    subquestions = data["subquestions"]
    if complexity not in ("simple", "complex") or type(subquestions) is not list:
        raise MultiRoundRunRecordValidationError("decomposition is invalid")
    if any(not isinstance(item, str) or not item.strip() for item in subquestions):
        raise MultiRoundRunRecordValidationError("decomposition subquestions are invalid")
    if complexity == "simple" and subquestions:
        raise MultiRoundRunRecordValidationError("simple decomposition is invalid")
    if complexity == "complex" and (
        len(subquestions) < 2 or len(set(subquestions)) != len(subquestions)
    ):
        raise MultiRoundRunRecordValidationError("complex decomposition is invalid")
    return {"question_complexity": complexity, "subquestions": subquestions.copy()}


def _validate_search_evidence(
    value: object,
    *,
    expected_tasks: list[tuple[int, str]],
    max_results: int,
) -> tuple[list[SearchCallEvidence], list[ResearchFinding]]:
    if type(value) is not list or len(value) != len(expected_tasks):
        raise MultiRoundRunRecordValidationError("search evidence does not match tasks")
    settings = sanitized_search_settings()
    calls: list[SearchCallEvidence] = []
    findings: list[ResearchFinding] = []
    expected_call_keys = set(SearchCallEvidence.__required_keys__)
    for raw, (index, query) in zip(value, expected_tasks, strict=True):
        if not isinstance(raw, dict) or set(raw) != expected_call_keys:
            raise MultiRoundRunRecordValidationError("search call has invalid fields")
        if (
            type(raw["subquestion_index"]) is not int
            or raw["subquestion_index"] != index
            or raw["query"] != query
        ):
            raise MultiRoundRunRecordValidationError("search call task is invalid")
        if (
            type(raw["requested_max_results"]) is not int
            or raw["requested_max_results"] != max_results
        ):
            raise MultiRoundRunRecordValidationError("search call maximum is invalid")
        if (
            raw["settings"] != settings
            or not isinstance(raw["settings"], dict)
            or any(
                type(raw["settings"].get(field)) is not type(expected)
                for field, expected in settings.items()
            )
            or raw["status"] != "success"
            or raw["error_category"] is not None
        ):
            raise MultiRoundRunRecordValidationError("search call status or settings are invalid")
        backend_duration = _finite_nonnegative(raw["backend_duration_seconds"], "backend duration")
        research_duration = _finite_nonnegative(
            raw["research_duration_seconds"], "research duration"
        )
        raw_results = raw["results"]
        if type(raw_results) is not list or len(raw_results) > max_results:
            raise MultiRoundRunRecordValidationError("search results are invalid")
        results = []
        for rank, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict) or set(item) != {
                "rank",
                "title",
                "url",
                "snippet",
                "provider_score",
                "source_type",
            }:
                raise MultiRoundRunRecordValidationError("search result has invalid fields")
            if type(item["rank"]) is not int or item["rank"] != rank:
                raise MultiRoundRunRecordValidationError("search result rank is invalid")
            for field in ("title", "url", "snippet"):
                if not isinstance(item[field], str) or not item[field].strip():
                    raise MultiRoundRunRecordValidationError("search result text is invalid")
            score = _finite_number(item["provider_score"], "provider score")
            if item["source_type"] != "web":
                raise MultiRoundRunRecordValidationError("search source type is invalid")
            result_item = {
                "rank": rank,
                "title": item["title"],
                "url": item["url"],
                "snippet": item["snippet"],
                "provider_score": score,
                "source_type": "web",
            }
            results.append(result_item)
            findings.append(
                {
                    "subquestion_index": index,
                    "subquestion": query,
                    "finding_index": rank,
                    "claim": item["snippet"],
                    "source_title": item["title"],
                    "source_url": item["url"],
                    "source_type": "web",
                    "supporting_excerpt": item["snippet"],
                }
            )
        calls.append(
            {
                "subquestion_index": index,
                "query": query,
                "requested_max_results": max_results,
                "settings": copy.deepcopy(settings),
                "status": "success",
                "error_category": None,
                "backend_duration_seconds": backend_duration,
                "research_duration_seconds": research_duration,
                "results": results,
            }
        )
    return calls, sorted(findings, key=finding_sort_key)


def _validate_graph_evidence(value: object) -> ProductGraphEvidence:
    data = _require_dict(value, ProductGraphEvidence, "graph evidence")
    raw = _finding_list(data["raw_research_findings"], "raw findings")
    consolidated = _finding_list(data["consolidated_evidence"], "consolidated evidence")
    formatted = data["research_findings"]
    if type(formatted) is not list or any(not isinstance(item, str) for item in formatted):
        raise MultiRoundRunRecordValidationError("formatted evidence is invalid")
    outputs = {}
    for field in ("final_team_a_argument", "final_team_b_argument", "neutral_synthesis"):
        item = data[field]
        if not isinstance(item, str) or not item.strip():
            raise MultiRoundRunRecordValidationError("final graph output is invalid")
        outputs[field] = item
    return {
        "raw_research_findings": raw,
        "consolidated_evidence": consolidated,
        "research_findings": formatted.copy(),
        "final_team_a_argument": outputs["final_team_a_argument"],
        "final_team_b_argument": outputs["final_team_b_argument"],
        "neutral_synthesis": outputs["neutral_synthesis"],
    }


def _validate_timings(
    value: object,
    *,
    expected_tasks: list[tuple[int, str]],
    search_evidence: list[SearchCallEvidence],
    rounds: list[DebateRound],
) -> ProductTimings:
    data = _require_dict(value, ProductTimings, "timings")
    decomposition = _validate_successful_model_trace(
        data["decomposition"], expected_stage="decomposition"
    )
    raw_research = data["research"]
    if type(raw_research) is not list or len(raw_research) != len(expected_tasks):
        raise MultiRoundRunRecordValidationError("research timings are invalid")
    research = []
    for item, task, search in zip(raw_research, expected_tasks, search_evidence, strict=True):
        if not isinstance(item, dict) or set(item) != set(ResearchTrace.__required_keys__):
            raise MultiRoundRunRecordValidationError("research timing has invalid fields")
        if (
            type(item["subquestion_index"]) is not int
            or (item["subquestion_index"], item["query"]) != task
            or item["status"] != "success"
            or item["error_category"] is not None
        ):
            raise MultiRoundRunRecordValidationError("research timing is inconsistent")
        duration = _finite_nonnegative(item["duration_seconds"], "research duration")
        if duration != search["research_duration_seconds"]:
            raise MultiRoundRunRecordValidationError("research duration does not match evidence")
        research.append({**item, "duration_seconds": duration})
    debate = _validate_canonical_debate_traces(data["debate"], completed_rounds=len(rounds))
    for round_record in rounds:
        number = round_record["round_number"]
        trace_by_stage = {
            trace["stage"]: trace
            for trace in debate
            if trace["round_number"] == number
        }
        if (
            trace_by_stage["team_a_argument"]["duration_seconds"]
            != round_record["team_a_duration_seconds"]
            or trace_by_stage["team_b_argument"]["duration_seconds"]
            != round_record["team_b_duration_seconds"]
            or trace_by_stage["termination"]["duration_seconds"]
            != round_record["termination_duration_seconds"]
        ):
            raise MultiRoundRunRecordValidationError("round timings do not match traces")
    synthesis = debate[-1]
    if synthesis["stage"] != "neutral_synthesis":
        raise MultiRoundRunRecordValidationError("synthesis timing is missing")
    total = _finite_nonnegative(data["total_duration_seconds"], "total duration")
    return {
        "decomposition": decomposition,
        "research": copy.deepcopy(research),
        "debate": copy.deepcopy(debate),
        "total_duration_seconds": total,
    }


def _validate_call_counts(
    value: object,
    *,
    completed_rounds: int,
    research_calls: int,
    debate_traces: list[MultiRoundTrace],
) -> ProductModelCallCounts:
    data = _require_dict(value, ProductModelCallCounts, "model call counts")
    expected = {
        "decomposition": 1,
        "team_a_argument": completed_rounds,
        "team_b_argument": completed_rounds,
        "synthesis": 1,
        "research_provider_calls": research_calls,
        "total_model_calls": 2 * completed_rounds + 2,
    }
    if any(type(data[field]) is not int for field in expected) or data != expected:
        raise MultiRoundRunRecordValidationError("model call counts are inconsistent")
    if (
        sum(trace["stage"] == "team_a_argument" for trace in debate_traces)
        != completed_rounds
        or sum(trace["stage"] == "team_b_argument" for trace in debate_traces)
        != completed_rounds
        or sum(trace["stage"] == "neutral_synthesis" for trace in debate_traces)
        != 1
    ):
        raise MultiRoundRunRecordValidationError("model call counts do not match traces")
    return cast(ProductModelCallCounts, expected.copy())


def _validate_canonical_debate_traces(
    value: object, *, completed_rounds: int
) -> list[MultiRoundTrace]:
    if type(value) is not list:
        raise MultiRoundRunRecordValidationError("debate traces must be a list")
    validated = [validate_multi_round_trace(item) for item in value]
    expected = [
        (round_number, stage)
        for round_number in range(1, completed_rounds + 1)
        for stage in ("team_a_argument", "team_b_argument", "termination")
    ]
    expected.append((completed_rounds, "neutral_synthesis"))
    if [(item["round_number"], item["stage"]) for item in validated] != expected:
        raise MultiRoundRunRecordValidationError("debate traces are not canonical")
    if any(
        item["status"] != "success" or item["error_category"] is not None
        for item in validated
    ):
        raise MultiRoundRunRecordValidationError("successful record has failed debate trace")
    return copy.deepcopy(validated)


def _synthesis_duration_from_timings(value: object) -> float:
    if not isinstance(value, dict):
        raise MultiRoundRunRecordValidationError("timings are invalid")
    debate = value.get("debate")
    if type(debate) is not list or not debate or not isinstance(debate[-1], dict):
        raise MultiRoundRunRecordValidationError("synthesis timing is missing")
    if debate[-1].get("stage") != "neutral_synthesis":
        raise MultiRoundRunRecordValidationError("synthesis timing is missing")
    return _finite_nonnegative(debate[-1].get("duration_seconds"), "synthesis duration")


def _validate_successful_model_trace(value: object, *, expected_stage: str) -> ModelStageTrace:
    if not isinstance(value, dict) or set(value) != set(ModelStageTrace.__required_keys__):
        raise MultiRoundRunRecordValidationError("model trace has invalid fields")
    if (
        value["stage"] != expected_stage
        or value["status"] != "success"
        or value["error_category"] is not None
    ):
        raise MultiRoundRunRecordValidationError("model trace is inconsistent")
    return {
        "stage": cast(object, value["stage"]),
        "duration_seconds": _finite_nonnegative(value["duration_seconds"], "model duration"),
        "status": "success",
        "error_category": None,
    }


def _expected_tasks(result: MultiRoundDebateResult) -> list[tuple[int, str]]:
    if result["question_complexity"] == "simple":
        return [(0, result["question"])]
    return list(enumerate(result["subquestions"], start=1))


def _group_unique(values: list, *, expected: list, identity, context: str) -> dict:
    expected_set = set(expected)
    grouped = {}
    for item in values:
        key = identity(item)
        if key not in expected_set or key in grouped:
            raise MultiRoundRunRecordValidationError(f"{context} traces are invalid")
        grouped[key] = item
    if set(grouped) != expected_set:
        raise MultiRoundRunRecordValidationError(f"{context} traces are incomplete")
    return grouped


def _require_dict(value: object, schema: type[TypedDict], context: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(schema.__required_keys__):
        raise MultiRoundRunRecordValidationError(f"{context} has invalid fields")
    return value


def _finding_list(value: object, context: str) -> list[ResearchFinding]:
    if type(value) is not list:
        raise MultiRoundRunRecordValidationError(f"{context} must be a list")
    try:
        return [validate_research_finding(item) for item in value]
    except ValueError:
        raise MultiRoundRunRecordValidationError(f"{context} is invalid") from None


def _normalize_commit(value: object) -> str:
    try:
        return normalize_implementation_commit(value)
    except Exception:
        raise MultiRoundRunRecordValidationError("implementation commit is invalid") from None


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MultiRoundRunRecordValidationError("UTC timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        normalized = format_utc_timestamp(parsed)
    except Exception:
        raise MultiRoundRunRecordValidationError("UTC timestamp is invalid") from None
    if normalized != value:
        raise MultiRoundRunRecordValidationError("UTC timestamp is not canonical")
    return value


def _policy_value(function, value):
    try:
        return function(value)
    except Exception:
        raise MultiRoundRunRecordValidationError("debate policy setting is invalid") from None


def _finite_number(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise MultiRoundRunRecordValidationError(f"{context} must be numeric")
    resolved = float(value)
    if not isfinite(resolved):
        raise MultiRoundRunRecordValidationError(f"{context} must be finite")
    return resolved


def _finite_nonnegative(value: object, context: str) -> float:
    resolved = _finite_number(value, context)
    if resolved < 0.0:
        raise MultiRoundRunRecordValidationError(f"{context} must be nonnegative")
    return resolved


def _ensure_json_finite(value: object, context: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise MultiRoundRunRecordValidationError(f"{context} is not finite JSON") from None


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise MultiRoundRunRecordValidationError(
            "record contains invalid JSON values"
        ) from None
