"""Versioned grounded-run records, deterministic scoring, and safe publication."""

import json
import os
import re
import tempfile
from collections.abc import Iterator, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Literal, TypedDict, cast

from multi_agent_debate.expanded_graph import validate_decomposition_result
from multi_agent_debate.expanded_state import ExpandedDebateResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.metrics import (
    METRIC_VERSION,
    Scorecard,
    calculate_research_utilization,
    calculate_summary_policy_compliance,
    calculate_synthesis_representation,
    calculate_team_novelty,
    count_output_words,
    scorecard_to_json,
)
from multi_agent_debate.research import (
    ResearchFinding,
    consolidate_research_findings,
    finding_sort_key,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.run_trace import CanonicalRunTraces
from multi_agent_debate.tavily_search import TAVILY_SEARCH_URL

RUN_RECORD_VERSION = "1.0"
ARCHITECTURE_VERSION = "milestone-2.1b2b"
SEARCH_PROVIDER = "tavily"
SEARCH_TOOL = "search_web"

_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


class ModelConfiguration(TypedDict):
    name: str
    temperature: float
    reasoning: bool
    structured_output_method: str


class SanitizedSearchSettings(TypedDict):
    search_depth: str
    include_answer: bool
    include_raw_content: bool
    include_images: bool


class SearchConfiguration(TypedDict):
    provider: str
    tool: str
    endpoint: str
    max_results: int
    settings: SanitizedSearchSettings


class RunProvenance(TypedDict):
    run_timestamp_utc: str
    implementation_commit: str
    architecture_version: str
    metric_version: str
    model: ModelConfiguration
    search: SearchConfiguration


class RunInput(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str


class DecompositionEvidence(TypedDict):
    question_complexity: Literal["simple", "complex"]
    subquestions: list[str]


class SearchResultEvidence(TypedDict):
    rank: int
    title: str
    url: str
    snippet: str
    provider_score: float
    source_type: str


class SearchCallEvidence(TypedDict):
    subquestion_index: int
    query: str
    requested_max_results: int
    settings: SanitizedSearchSettings
    status: Literal["success", "error"]
    error_category: str | None
    backend_duration_seconds: float
    research_duration_seconds: float
    results: list[SearchResultEvidence]


class GraphEvidence(TypedDict):
    raw_research_findings: list[ResearchFinding]
    consolidated_evidence: list[ResearchFinding]
    research_findings: list[str]
    team_a_argument: str
    team_b_argument: str
    synthesis: str


class StageTiming(TypedDict):
    stage: str
    duration_seconds: float
    status: Literal["success", "error"]
    error_category: str | None


class ResearchTiming(TypedDict):
    subquestion_index: int
    query: str
    duration_seconds: float
    status: Literal["success", "error"]
    error_category: str | None


class RunTimings(TypedDict):
    decomposition: StageTiming
    research: list[ResearchTiming]
    team_a: StageTiming
    team_b: StageTiming
    neutral_synthesis: StageTiming
    total_duration_seconds: float


class FindingUtilizationData(TypedDict):
    finding_index: int
    team_a_coverage: float
    team_b_coverage: float
    synthesis_coverage: float
    utilization: float


class ResearchUtilizationData(TypedDict):
    score: float
    findings: list[FindingUtilizationData]


class TeamNoveltyData(TypedDict):
    score: float
    jaccard_similarity: float
    team_a_shingle_count: int
    team_b_shingle_count: int
    shared_shingle_count: int
    union_shingle_count: int


class SynthesisRepresentationData(TypedDict):
    score: float
    team_a_coverage: float
    team_b_coverage: float
    mean_coverage: float
    balance: float


class SummaryPolicyComplianceData(TypedDict):
    score: int
    matched_pattern_ids: list[str]


class WordCountsData(TypedDict):
    research_finding_words: list[int]
    research_total_words: int
    team_a_argument_words: int
    team_b_argument_words: int
    synthesis_words: int
    total_output_words: int


class ScorecardData(TypedDict):
    metric_version: str
    research_utilization: ResearchUtilizationData
    team_novelty: TeamNoveltyData
    synthesis_representation: SynthesisRepresentationData
    summary_policy_compliance: SummaryPolicyComplianceData
    word_counts: WordCountsData


class RunRecord(TypedDict):
    run_record_version: str
    provenance: RunProvenance
    input: RunInput
    decomposition: DecompositionEvidence
    search_evidence: list[SearchCallEvidence]
    graph_evidence: GraphEvidence
    timings: RunTimings
    metrics: ScorecardData


class RunRecordError(RuntimeError):
    """Base exception for grounded-run evidence handling."""


class RunRecordValidationError(RunRecordError):
    """Raised when a run record violates its versioned contract."""


class RunRecordSecurityError(RunRecordError):
    """Raised when credential material is found in artifact content."""


class ArtifactPublicationError(RunRecordError):
    """Raised when validated artifact content cannot be safely published."""


class ArtifactExistsError(ArtifactPublicationError):
    """Raised when overwrite is disabled and an output already exists."""


def sanitized_search_settings() -> SanitizedSearchSettings:
    """Return a fresh copy of the fixed Tavily request settings."""

    return {
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }


def normalize_implementation_commit(value: object) -> str:
    """Validate a full Git object ID and normalize hexadecimal case."""

    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise RunRecordValidationError(
            "implementation commit must be exactly 40 hexadecimal characters"
        )
    return value.lower()


def format_utc_timestamp(value: object) -> str:
    """Serialize one timezone-aware UTC datetime with a trailing Z."""

    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RunRecordValidationError("run timestamp must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise RunRecordValidationError("run timestamp must be UTC")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def score_expanded_result(result: ExpandedDebateResult) -> Scorecard:
    """Score only consolidated claim text with frozen metric version 1.0."""

    claims = [finding["claim"] for finding in result["consolidated_evidence"]]
    return Scorecard(
        metric_version=METRIC_VERSION,
        research_utilization=calculate_research_utilization(
            claims,
            result["team_a_argument"],
            result["team_b_argument"],
            result["synthesis"],
        ),
        team_novelty=calculate_team_novelty(
            result["team_a_argument"],
            result["team_b_argument"],
        ),
        synthesis_representation=calculate_synthesis_representation(
            result["team_a_argument"],
            result["team_b_argument"],
            result["synthesis"],
        ),
        summary_policy_compliance=calculate_summary_policy_compliance(
            result["synthesis"]
        ),
        word_counts=count_output_words(
            claims,
            result["team_a_argument"],
            result["team_b_argument"],
            result["synthesis"],
        ),
    )


def build_run_record(
    *,
    result: ExpandedDebateResult,
    traces: CanonicalRunTraces,
    scorecard: Scorecard,
    run_timestamp_utc: str,
    implementation_commit: str,
    model_name: str,
    max_results: int,
    total_duration_seconds: float,
) -> RunRecord:
    """Construct and strictly validate one complete successful run record."""

    scorecard_json = scorecard_to_json(scorecard)
    embedded_metrics = cast(ScorecardData, json.loads(scorecard_json))
    settings = sanitized_search_settings()
    search_evidence: list[SearchCallEvidence] = []
    research_timings: list[ResearchTiming] = []
    expected_tasks = _expected_tasks(result)
    if (
        len(traces["backend_searches"]) != len(expected_tasks)
        or len(traces["research_calls"]) != len(expected_tasks)
    ):
        raise RunRecordValidationError("canonical research traces are incomplete")

    for (subquestion_index, query), backend_trace, research_trace in zip(
        expected_tasks,
        traces["backend_searches"],
        traces["research_calls"],
        strict=True,
    ):
        if (
            backend_trace["query"] != query
            or research_trace["subquestion_index"] != subquestion_index
            or research_trace["query"] != query
        ):
            raise RunRecordValidationError(
                "canonical research traces do not match decomposition"
            )
        if (
            backend_trace["status"] != "success"
            or research_trace["status"] != "success"
        ):
            raise RunRecordValidationError(
                "a completed run cannot contain failed search traces"
            )
        if backend_trace["settings"] != settings:
            raise RunRecordValidationError(
                "backend trace search settings are inconsistent"
            )
        result_records: list[SearchResultEvidence] = []
        for rank, backend_result in enumerate(backend_trace["results"], start=1):
            if "provider_score" not in backend_result:
                raise RunRecordValidationError(
                    "grounded search evidence must retain provider scores"
                )
            result_records.append(
                {
                    "rank": rank,
                    "title": backend_result["title"],
                    "url": backend_result["url"],
                    "snippet": backend_result["snippet"],
                    "provider_score": backend_result["provider_score"],
                    "source_type": "web",
                }
            )
        search_evidence.append(
            {
                "subquestion_index": subquestion_index,
                "query": query,
                "requested_max_results": backend_trace["requested_max_results"],
                "settings": deepcopy(settings),
                "status": "success",
                "error_category": None,
                "backend_duration_seconds": backend_trace["duration_seconds"],
                "research_duration_seconds": research_trace["duration_seconds"],
                "results": result_records,
            }
        )
        research_timings.append(
            {
                "subquestion_index": subquestion_index,
                "query": query,
                "duration_seconds": research_trace["duration_seconds"],
                "status": "success",
                "error_category": None,
            }
        )

    stages = {trace["stage"]: trace for trace in traces["model_stages"]}
    expected_stages = {
        "decomposition",
        "team_a",
        "team_b",
        "neutral_synthesis",
    }
    if len(traces["model_stages"]) != 4 or set(stages) != expected_stages:
        raise RunRecordValidationError("canonical model-stage traces are incomplete")
    record: RunRecord = {
        "run_record_version": RUN_RECORD_VERSION,
        "provenance": {
            "run_timestamp_utc": run_timestamp_utc,
            "implementation_commit": normalize_implementation_commit(
                implementation_commit
            ),
            "architecture_version": ARCHITECTURE_VERSION,
            "metric_version": METRIC_VERSION,
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
                "settings": deepcopy(settings),
            },
        },
        "input": {
            "question": result["question"],
            "team_a_stance": result["team_a_stance"],
            "team_b_stance": result["team_b_stance"],
        },
        "decomposition": {
            "question_complexity": result["question_complexity"],
            "subquestions": result["subquestions"].copy(),
        },
        "search_evidence": search_evidence,
        "graph_evidence": {
            "raw_research_findings": [
                finding.copy() for finding in result["raw_research_findings"]
            ],
            "consolidated_evidence": [
                finding.copy() for finding in result["consolidated_evidence"]
            ],
            "research_findings": result["research_findings"].copy(),
            "team_a_argument": result["team_a_argument"],
            "team_b_argument": result["team_b_argument"],
            "synthesis": result["synthesis"],
        },
        "timings": {
            "decomposition": _stage_timing(stages, "decomposition"),
            "research": research_timings,
            "team_a": _stage_timing(stages, "team_a"),
            "team_b": _stage_timing(stages, "team_b"),
            "neutral_synthesis": _stage_timing(stages, "neutral_synthesis"),
            "total_duration_seconds": float(total_duration_seconds),
        },
        "metrics": embedded_metrics,
    }
    return validate_run_record(record)


def validate_run_record(value: object) -> RunRecord:
    """Strictly validate, cross-check, and defensively copy run-record v1.0."""

    record = _require_dict(
        value,
        {
            "run_record_version",
            "provenance",
            "input",
            "decomposition",
            "search_evidence",
            "graph_evidence",
            "timings",
            "metrics",
        },
        "run record",
    )
    if record["run_record_version"] != RUN_RECORD_VERSION:
        raise RunRecordValidationError("run record version is invalid")

    provenance = _validate_provenance(record["provenance"])
    run_input = _validate_input(record["input"])
    decomposition = _validate_decomposition(record["decomposition"])
    expected_tasks = (
        [(0, run_input["question"])]
        if decomposition["question_complexity"] == "simple"
        else list(enumerate(decomposition["subquestions"], start=1))
    )
    search_evidence, expected_raw = _validate_search_evidence(
        record["search_evidence"],
        expected_tasks=expected_tasks,
        expected_max_results=provenance["search"]["max_results"],
    )
    graph_evidence = _validate_graph_evidence(
        record["graph_evidence"],
        run_input=run_input,
        decomposition=decomposition,
        expected_raw=expected_raw,
    )
    timings = _validate_timings(
        record["timings"],
        expected_tasks=expected_tasks,
        search_evidence=search_evidence,
    )

    reconstructed: ExpandedDebateResult = {
        "question": run_input["question"],
        "team_a_stance": run_input["team_a_stance"],
        "team_b_stance": run_input["team_b_stance"],
        "question_complexity": decomposition["question_complexity"],
        "subquestions": decomposition["subquestions"].copy(),
        "raw_research_findings": [
            finding.copy() for finding in graph_evidence["raw_research_findings"]
        ],
        "consolidated_evidence": [
            finding.copy() for finding in graph_evidence["consolidated_evidence"]
        ],
        "research_findings": graph_evidence["research_findings"].copy(),
        "team_a_argument": graph_evidence["team_a_argument"],
        "team_b_argument": graph_evidence["team_b_argument"],
        "synthesis": graph_evidence["synthesis"],
    }
    metrics = _validate_metrics(record["metrics"])
    expected_metrics = json.loads(scorecard_to_json(score_expanded_result(reconstructed)))
    if metrics != expected_metrics:
        raise RunRecordValidationError(
            "embedded metrics do not match frozen metric version 1.0"
        )

    return {
        "run_record_version": RUN_RECORD_VERSION,
        "provenance": provenance,
        "input": run_input,
        "decomposition": decomposition,
        "search_evidence": search_evidence,
        "graph_evidence": graph_evidence,
        "timings": timings,
        "metrics": metrics,
    }


def run_record_to_json(record: object) -> str:
    """Serialize a validated run record deterministically."""

    validated = validate_run_record(record)
    return (
        json.dumps(
            validated,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def assert_credential_absent(
    value: object,
    serialized: str,
    *,
    api_key: str,
) -> None:
    """Reject exact credential material without exposing it in the error."""

    if not isinstance(api_key, str) or not api_key.strip():
        raise RunRecordSecurityError("artifact credential guard is not configured")
    if any(api_key in item for item in _iter_strings(value)) or api_key in serialized:
        raise RunRecordSecurityError("artifact contains forbidden credential data")


def preflight_output_paths(
    run_output: Path,
    score_output: Path | None,
    *,
    overwrite: bool,
) -> None:
    """Reject invalid or occupied destinations before live construction."""

    destinations = [run_output, *([score_output] if score_output is not None else [])]
    resolved = [destination.resolve() for destination in destinations]
    if len(set(resolved)) != len(resolved):
        raise ArtifactPublicationError("artifact output paths must be distinct")
    for destination in destinations:
        if destination.exists() and destination.is_dir():
            raise ArtifactPublicationError("artifact output path is a directory")
        if destination.exists() and not overwrite:
            raise ArtifactExistsError("artifact output already exists")


def publish_serialized_artifacts(
    *,
    run_output: Path,
    run_json: str,
    score_output: Path | None,
    scorecard_json: str | None,
    overwrite: bool,
) -> None:
    """Stage complete documents, publish score first, and publish run last.

    If final run publication fails after score publication, restore or remove
    the score as appropriate. No broader cross-file transaction is promised.
    """

    if score_output is None and scorecard_json is not None:
        raise ArtifactPublicationError("score content requires a score output path")
    if score_output is not None and scorecard_json is None:
        raise ArtifactPublicationError("score output requires serialized score content")
    _validate_serialized_json(run_json)
    if scorecard_json is not None:
        _validate_serialized_json(scorecard_json)
    preflight_output_paths(run_output, score_output, overwrite=overwrite)

    score_existed = score_output is not None and score_output.exists()
    original_score_bytes: bytes | None = None
    if score_existed:
        try:
            original_score_bytes = score_output.read_bytes()
        except OSError:
            raise ArtifactPublicationError("artifact publication failed") from None

    staged: dict[Path, Path] = {}
    score_published = False
    try:
        staged[run_output] = _stage_serialized(run_output, run_json)
        if score_output is not None and scorecard_json is not None:
            staged[score_output] = _stage_serialized(score_output, scorecard_json)

        preflight_output_paths(run_output, score_output, overwrite=overwrite)
        if score_output is not None:
            os.replace(staged[score_output], score_output)
            score_published = True
        os.replace(staged[run_output], run_output)
    except Exception:
        if score_published and score_output is not None:
            try:
                if score_existed:
                    assert original_score_bytes is not None
                    _restore_file_bytes_atomically(
                        score_output,
                        original_score_bytes,
                    )
                else:
                    score_output.unlink(missing_ok=True)
            except Exception:
                raise ArtifactPublicationError(
                    "artifact publication failed and score rollback failed"
                ) from None
        raise ArtifactPublicationError("artifact publication failed") from None
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _stage_timing(
    stages: dict[str, object],
    stage: str,
) -> StageTiming:
    trace = cast(dict[str, object], stages[stage])
    return {
        "stage": stage,
        "duration_seconds": cast(float, trace["duration_seconds"]),
        "status": cast(Literal["success", "error"], trace["status"]),
        "error_category": cast(str | None, trace["error_category"]),
    }


def _expected_tasks(result: ExpandedDebateResult) -> list[tuple[int, str]]:
    if result["question_complexity"] == "simple":
        return [(0, result["question"])]
    return list(enumerate(result["subquestions"], start=1))


def _validate_provenance(value: object) -> RunProvenance:
    data = _require_dict(
        value,
        {
            "run_timestamp_utc",
            "implementation_commit",
            "architecture_version",
            "metric_version",
            "model",
            "search",
        },
        "provenance",
    )
    timestamp = _validate_timestamp(data["run_timestamp_utc"])
    commit = normalize_implementation_commit(data["implementation_commit"])
    if data["implementation_commit"] != commit:
        raise RunRecordValidationError("implementation commit must be lowercase")
    if data["architecture_version"] != ARCHITECTURE_VERSION:
        raise RunRecordValidationError("architecture version is invalid")
    if data["metric_version"] != METRIC_VERSION:
        raise RunRecordValidationError("metric version is invalid")

    model = _require_dict(
        data["model"],
        {"name", "temperature", "reasoning", "structured_output_method"},
        "model configuration",
    )
    name = _require_nonempty_text(model["name"], "model name")
    if type(model["temperature"]) is not float or model["temperature"] != 0.0:
        raise RunRecordValidationError("model temperature must be 0.0")
    if model["reasoning"] is not False:
        raise RunRecordValidationError("model reasoning must be disabled")
    if model["structured_output_method"] != "json_schema":
        raise RunRecordValidationError("structured-output method is invalid")

    search = _require_dict(
        data["search"],
        {"provider", "tool", "endpoint", "max_results", "settings"},
        "search configuration",
    )
    if search["provider"] != SEARCH_PROVIDER:
        raise RunRecordValidationError("search provider is invalid")
    if search["tool"] != SEARCH_TOOL:
        raise RunRecordValidationError("search tool is invalid")
    if search["endpoint"] != TAVILY_SEARCH_URL:
        raise RunRecordValidationError("search endpoint is invalid")
    max_results = _require_positive_int(search["max_results"], "max_results")
    settings = _validate_search_settings(search["settings"])
    return {
        "run_timestamp_utc": timestamp,
        "implementation_commit": commit,
        "architecture_version": ARCHITECTURE_VERSION,
        "metric_version": METRIC_VERSION,
        "model": {
            "name": name,
            "temperature": 0.0,
            "reasoning": False,
            "structured_output_method": "json_schema",
        },
        "search": {
            "provider": SEARCH_PROVIDER,
            "tool": SEARCH_TOOL,
            "endpoint": TAVILY_SEARCH_URL,
            "max_results": max_results,
            "settings": settings,
        },
    }


def _validate_input(value: object) -> RunInput:
    data = _require_dict(
        value,
        {"question", "team_a_stance", "team_b_stance"},
        "run input",
    )
    question = _require_nonempty_text(data["question"], "question")
    if data["team_a_stance"] != TEAM_A_STANCE:
        raise RunRecordValidationError("Team A stance does not match the graph")
    if data["team_b_stance"] != TEAM_B_STANCE:
        raise RunRecordValidationError("Team B stance does not match the graph")
    return {
        "question": question,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
    }


def _validate_decomposition(value: object) -> DecompositionEvidence:
    data = _require_dict(
        value,
        {"question_complexity", "subquestions"},
        "decomposition",
    )
    try:
        validated = validate_decomposition_result(data)
    except ValueError:
        raise RunRecordValidationError("decomposition is invalid") from None
    if len(validated["subquestions"]) > 4:
        raise RunRecordValidationError("decomposition exceeds four subquestions")
    return {
        "question_complexity": validated["question_complexity"],
        "subquestions": validated["subquestions"].copy(),
    }


def _validate_search_evidence(
    value: object,
    *,
    expected_tasks: list[tuple[int, str]],
    expected_max_results: int,
) -> tuple[list[SearchCallEvidence], list[ResearchFinding]]:
    if type(value) is not list or len(value) != len(expected_tasks):
        raise RunRecordValidationError(
            "search evidence must match every research task exactly"
        )
    validated_calls: list[SearchCallEvidence] = []
    expected_raw: list[ResearchFinding] = []
    for raw_call, (expected_index, expected_query) in zip(
        value,
        expected_tasks,
        strict=True,
    ):
        call = _require_dict(
            raw_call,
            {
                "subquestion_index",
                "query",
                "requested_max_results",
                "settings",
                "status",
                "error_category",
                "backend_duration_seconds",
                "research_duration_seconds",
                "results",
            },
            "search evidence call",
        )
        if (
            type(call["subquestion_index"]) is not int
            or call["subquestion_index"] != expected_index
        ):
            raise RunRecordValidationError("search subquestion index is invalid")
        if call["query"] != expected_query:
            raise RunRecordValidationError("search query does not match decomposition")
        if (
            type(call["requested_max_results"]) is not int
            or call["requested_max_results"] != expected_max_results
        ):
            raise RunRecordValidationError("search maximum is inconsistent")
        settings = _validate_search_settings(call["settings"])
        if call["status"] != "success" or call["error_category"] is not None:
            raise RunRecordValidationError(
                "successful artifacts may contain only successful searches"
            )
        backend_duration = _finite_nonnegative_float(
            call["backend_duration_seconds"], "backend duration"
        )
        research_duration = _finite_nonnegative_float(
            call["research_duration_seconds"], "research duration"
        )
        raw_results = call["results"]
        if type(raw_results) is not list or len(raw_results) > expected_max_results:
            raise RunRecordValidationError(
                "search results exceed the requested maximum"
            )
        results: list[SearchResultEvidence] = []
        for expected_rank, raw_result in enumerate(raw_results, start=1):
            item = _require_dict(
                raw_result,
                {"rank", "title", "url", "snippet", "provider_score", "source_type"},
                "search result",
            )
            if type(item["rank"]) is not int or item["rank"] != expected_rank:
                raise RunRecordValidationError(
                    "search result ranks must be consecutive and one-based"
                )
            title = _require_nonempty_text(item["title"], "search result title")
            url = _require_nonempty_text(item["url"], "search result URL")
            snippet = _require_nonempty_text(item["snippet"], "search result snippet")
            score = _finite_float(item["provider_score"], "provider score")
            if item["source_type"] != "web":
                raise RunRecordValidationError("search source type must be web")
            result: SearchResultEvidence = {
                "rank": expected_rank,
                "title": title,
                "url": url,
                "snippet": snippet,
                "provider_score": score,
                "source_type": "web",
            }
            results.append(result)
            expected_raw.append(
                {
                    "subquestion_index": expected_index,
                    "subquestion": expected_query,
                    "finding_index": expected_rank,
                    "claim": snippet,
                    "source_title": title,
                    "source_url": url,
                    "source_type": "web",
                    "supporting_excerpt": snippet,
                }
            )
        validated_calls.append(
            {
                "subquestion_index": expected_index,
                "query": expected_query,
                "requested_max_results": expected_max_results,
                "settings": settings,
                "status": "success",
                "error_category": None,
                "backend_duration_seconds": backend_duration,
                "research_duration_seconds": research_duration,
                "results": results,
            }
        )
    return validated_calls, sorted(expected_raw, key=finding_sort_key)


def _validate_graph_evidence(
    value: object,
    *,
    run_input: RunInput,
    decomposition: DecompositionEvidence,
    expected_raw: list[ResearchFinding],
) -> GraphEvidence:
    data = _require_dict(
        value,
        {
            "raw_research_findings",
            "consolidated_evidence",
            "research_findings",
            "team_a_argument",
            "team_b_argument",
            "synthesis",
        },
        "graph evidence",
    )
    raw_findings = _validate_finding_list(
        data["raw_research_findings"], "raw research findings"
    )
    if raw_findings != expected_raw:
        raise RunRecordValidationError(
            "raw findings do not match recorded search evidence"
        )
    try:
        expected_consolidated = consolidate_research_findings(
            question=run_input["question"],
            question_complexity=decomposition["question_complexity"],
            subquestions=decomposition["subquestions"],
            raw_findings=raw_findings,
        )
    except ValueError:
        raise RunRecordValidationError("research consolidation is invalid") from None
    consolidated = _validate_finding_list(
        data["consolidated_evidence"], "consolidated evidence"
    )
    if consolidated != expected_consolidated:
        raise RunRecordValidationError(
            "consolidated evidence does not match frozen consolidation"
        )
    research_findings = data["research_findings"]
    if type(research_findings) is not list or any(
        not isinstance(item, str) for item in research_findings
    ):
        raise RunRecordValidationError("formatted research findings are invalid")
    expected_formatted = [
        format_research_finding(finding) for finding in consolidated
    ]
    if research_findings != expected_formatted:
        raise RunRecordValidationError(
            "formatted research findings do not match frozen formatting"
        )
    return {
        "raw_research_findings": [finding.copy() for finding in raw_findings],
        "consolidated_evidence": [finding.copy() for finding in consolidated],
        "research_findings": cast(list[str], research_findings).copy(),
        "team_a_argument": _require_nonempty_text(
            data["team_a_argument"], "Team A argument"
        ),
        "team_b_argument": _require_nonempty_text(
            data["team_b_argument"], "Team B argument"
        ),
        "synthesis": _require_nonempty_text(data["synthesis"], "synthesis"),
    }


def _validate_timings(
    value: object,
    *,
    expected_tasks: list[tuple[int, str]],
    search_evidence: list[SearchCallEvidence],
) -> RunTimings:
    data = _require_dict(
        value,
        {
            "decomposition",
            "research",
            "team_a",
            "team_b",
            "neutral_synthesis",
            "total_duration_seconds",
        },
        "timings",
    )
    decomposition = _validate_stage(data["decomposition"], "decomposition")
    team_a = _validate_stage(data["team_a"], "team_a")
    team_b = _validate_stage(data["team_b"], "team_b")
    synthesis = _validate_stage(data["neutral_synthesis"], "neutral_synthesis")
    raw_research = data["research"]
    if type(raw_research) is not list or len(raw_research) != len(expected_tasks):
        raise RunRecordValidationError("research timings do not match tasks")
    research: list[ResearchTiming] = []
    for raw_timing, task, search_call in zip(
        raw_research,
        expected_tasks,
        search_evidence,
        strict=True,
    ):
        timing = _require_dict(
            raw_timing,
            {
                "subquestion_index",
                "query",
                "duration_seconds",
                "status",
                "error_category",
            },
            "research timing",
        )
        if (
            type(timing["subquestion_index"]) is not int
            or timing["subquestion_index"] != task[0]
            or timing["query"] != task[1]
        ):
            raise RunRecordValidationError("research timing task is invalid")
        duration = _finite_nonnegative_float(
            timing["duration_seconds"], "research timing duration"
        )
        if duration != search_call["research_duration_seconds"]:
            raise RunRecordValidationError(
                "research timing does not match search evidence"
            )
        if timing["status"] != "success" or timing["error_category"] is not None:
            raise RunRecordValidationError("research timing must be successful")
        research.append(
            {
                "subquestion_index": task[0],
                "query": task[1],
                "duration_seconds": duration,
                "status": "success",
                "error_category": None,
            }
        )
    total = _finite_nonnegative_float(
        data["total_duration_seconds"], "total duration"
    )
    return {
        "decomposition": decomposition,
        "research": research,
        "team_a": team_a,
        "team_b": team_b,
        "neutral_synthesis": synthesis,
        "total_duration_seconds": total,
    }


def _validate_metrics(value: object) -> ScorecardData:
    data = _require_dict(
        value,
        {
            "metric_version",
            "research_utilization",
            "team_novelty",
            "synthesis_representation",
            "summary_policy_compliance",
            "word_counts",
        },
        "metrics",
    )
    if data["metric_version"] != METRIC_VERSION:
        raise RunRecordValidationError("embedded metric version is invalid")
    research = _require_dict(
        data["research_utilization"], {"score", "findings"}, "research utilization"
    )
    research_score = _finite_float(research["score"], "research utilization score")
    raw_findings = research["findings"]
    if type(raw_findings) is not list:
        raise RunRecordValidationError("finding utilization must be a list")
    findings: list[FindingUtilizationData] = []
    for raw in raw_findings:
        finding = _require_dict(
            raw,
            {
                "finding_index",
                "team_a_coverage",
                "team_b_coverage",
                "synthesis_coverage",
                "utilization",
            },
            "finding utilization",
        )
        findings.append(
            {
                "finding_index": _require_positive_int(
                    finding["finding_index"], "metric finding index"
                ),
                "team_a_coverage": _finite_float(
                    finding["team_a_coverage"], "Team A finding coverage"
                ),
                "team_b_coverage": _finite_float(
                    finding["team_b_coverage"], "Team B finding coverage"
                ),
                "synthesis_coverage": _finite_float(
                    finding["synthesis_coverage"], "synthesis finding coverage"
                ),
                "utilization": _finite_float(
                    finding["utilization"], "finding utilization"
                ),
            }
        )

    novelty = _validate_float_and_count_object(
        data["team_novelty"],
        float_fields=("score", "jaccard_similarity"),
        count_fields=(
            "team_a_shingle_count",
            "team_b_shingle_count",
            "shared_shingle_count",
            "union_shingle_count",
        ),
        context="team novelty",
    )
    representation = _validate_float_and_count_object(
        data["synthesis_representation"],
        float_fields=(
            "score",
            "team_a_coverage",
            "team_b_coverage",
            "mean_coverage",
            "balance",
        ),
        count_fields=(),
        context="synthesis representation",
    )
    compliance = _require_dict(
        data["summary_policy_compliance"],
        {"score", "matched_pattern_ids"},
        "summary-policy compliance",
    )
    if type(compliance["score"]) is not int or compliance["score"] not in (0, 1):
        raise RunRecordValidationError("summary-policy score must be zero or one")
    matched = compliance["matched_pattern_ids"]
    if type(matched) is not list or any(not isinstance(item, str) for item in matched):
        raise RunRecordValidationError("matched pattern IDs must be strings")
    counts = _require_dict(
        data["word_counts"],
        {
            "research_finding_words",
            "research_total_words",
            "team_a_argument_words",
            "team_b_argument_words",
            "synthesis_words",
            "total_output_words",
        },
        "word counts",
    )
    per_finding = counts["research_finding_words"]
    if type(per_finding) is not list or any(
        type(item) is not int or item < 0 for item in per_finding
    ):
        raise RunRecordValidationError("research word counts must be nonnegative integers")
    count_names = (
        "research_total_words",
        "team_a_argument_words",
        "team_b_argument_words",
        "synthesis_words",
        "total_output_words",
    )
    validated_counts = {
        name: _require_nonnegative_int(counts[name], name) for name in count_names
    }
    return cast(
        ScorecardData,
        {
            "metric_version": METRIC_VERSION,
            "research_utilization": {
                "score": research_score,
                "findings": findings,
            },
            "team_novelty": novelty,
            "synthesis_representation": representation,
            "summary_policy_compliance": {
                "score": compliance["score"],
                "matched_pattern_ids": cast(list[str], matched).copy(),
            },
            "word_counts": {
                "research_finding_words": cast(list[int], per_finding).copy(),
                **validated_counts,
            },
        },
    )


def _validate_float_and_count_object(
    value: object,
    *,
    float_fields: Sequence[str],
    count_fields: Sequence[str],
    context: str,
) -> dict[str, object]:
    data = _require_dict(value, set(float_fields) | set(count_fields), context)
    return {
        **{name: _finite_float(data[name], name) for name in float_fields},
        **{name: _require_nonnegative_int(data[name], name) for name in count_fields},
    }


def _validate_stage(value: object, expected_stage: str) -> StageTiming:
    data = _require_dict(
        value,
        {"stage", "duration_seconds", "status", "error_category"},
        f"{expected_stage} timing",
    )
    if data["stage"] != expected_stage:
        raise RunRecordValidationError("timing stage is invalid")
    if data["status"] != "success" or data["error_category"] is not None:
        raise RunRecordValidationError("completed stage timing must be successful")
    return {
        "stage": expected_stage,
        "duration_seconds": _finite_nonnegative_float(
            data["duration_seconds"], f"{expected_stage} duration"
        ),
        "status": "success",
        "error_category": None,
    }


def _validate_search_settings(value: object) -> SanitizedSearchSettings:
    data = _require_dict(
        value,
        {"search_depth", "include_answer", "include_raw_content", "include_images"},
        "sanitized search settings",
    )
    expected = sanitized_search_settings()
    if data != expected or any(
        type(data[field]) is not type(expected_value)
        for field, expected_value in expected.items()
    ):
        raise RunRecordValidationError("sanitized search settings are invalid")
    return expected


def _validate_finding_list(value: object, context: str) -> list[ResearchFinding]:
    if type(value) is not list:
        raise RunRecordValidationError(f"{context} must be a list")
    try:
        return [validate_research_finding(finding) for finding in value]
    except ValueError:
        raise RunRecordValidationError(f"{context} contains an invalid finding") from None


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RunRecordValidationError("run timestamp must be a UTC string ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        raise RunRecordValidationError("run timestamp is invalid") from None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RunRecordValidationError("run timestamp must be UTC")
    return value


def _require_dict(
    value: object,
    expected_keys: set[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RunRecordValidationError(f"{context} has invalid fields")
    return cast(dict[str, object], value)


def _require_nonempty_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunRecordValidationError(f"{context} must be a nonempty string")
    return value


def _require_positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise RunRecordValidationError(f"{context} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise RunRecordValidationError(f"{context} must be a nonnegative integer")
    return value


def _finite_float(value: object, context: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise RunRecordValidationError(f"{context} must be a finite float")
    return value


def _finite_nonnegative_float(value: object, context: str) -> float:
    result = _finite_float(value, context)
    if result < 0:
        raise RunRecordValidationError(f"{context} must be nonnegative")
    return result


def _iter_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _validate_serialized_json(value: str) -> None:
    if not isinstance(value, str) or not value.endswith("\n") or value.endswith("\n\n"):
        raise ArtifactPublicationError(
            "serialized artifact must end with exactly one newline"
        )
    try:
        json.loads(value)
    except (TypeError, ValueError):
        raise ArtifactPublicationError("serialized artifact is not valid JSON") from None


def _stage_serialized(destination: Path, value: str) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            return Path(handle.name)
    except OSError:
        raise ArtifactPublicationError("artifact staging failed") from None


def _restore_file_bytes_atomically(destination: Path, original: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.restore.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except Exception:
        raise ArtifactPublicationError(
            "artifact publication failed and score restoration failed"
        ) from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
