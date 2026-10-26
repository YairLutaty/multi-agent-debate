"""Strict version 1.1 records for deterministic team-architecture replay."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from typing import Literal, TypedDict, cast

from multi_agent_debate.expanded_state import ExpandedDebateResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.metrics import METRIC_VERSION, Scorecard, scorecard_to_json
from multi_agent_debate.replay import ReplaySource, validate_replay_source
from multi_agent_debate.research import (
    ResearchFinding,
    consolidate_research_findings,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.run_record import (
    ModelConfiguration,
    ScorecardData,
    SearchCallEvidence,
    normalize_implementation_commit,
    score_expanded_result,
)
from multi_agent_debate.team_model import (
    EvidenceAnalysis,
    TeamCritique,
    TeamName,
    validate_team_work_product,
)
from multi_agent_debate.team_trace import TEAM_STAGES, TeamStageTrace

REPLAY_RUN_RECORD_VERSION = "1.1"
CONTROL_ARCHITECTURE_VERSION = "milestone-2.2b-control-replay"
HIERARCHICAL_ARCHITECTURE_VERSION = "milestone-2.2b-hierarchical-replay"

ReplayVariant = Literal["control", "hierarchical"]


class ReplayRunRecordError(RuntimeError):
    """Base error for replay run-record construction and validation."""


class ReplayRunRecordValidationError(ReplayRunRecordError):
    """A replay run record violates version 1.1 or its source evidence."""


class ReplayProvenance(TypedDict):
    run_timestamp_utc: str
    implementation_commit: str
    architecture_version: str
    metric_version: str
    model: ModelConfiguration


class ReplaySourceProvenance(TypedDict):
    source_artifact_path: str
    source_sha256: str
    source_run_record_version: str
    source_architecture_version: str
    source_implementation_commit: str
    source_run_timestamp_utc: str


class ReplayInput(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str


class ReplayDecomposition(TypedDict):
    question_complexity: Literal["simple", "complex"]
    subquestions: list[str]


class ReplayGraphEvidence(TypedDict):
    raw_research_findings: list[ResearchFinding]
    consolidated_evidence: list[ResearchFinding]
    research_findings: list[str]
    team_a_argument: str
    team_b_argument: str
    synthesis: str


class ControlTeamDetail(TypedDict):
    final_argument: str
    duration_seconds: float


class ControlTeamExecution(TypedDict):
    variant: Literal["control"]
    team_a: ControlTeamDetail
    team_b: ControlTeamDetail
    synthesis_duration_seconds: float


class HierarchicalTeamDetail(TypedDict):
    team: TeamName
    assigned_stance: str
    opponent_argument: str | None
    evidence_analysis: EvidenceAnalysis
    draft_argument: str
    internal_critique: TeamCritique
    final_argument: str
    stage_timings: list[TeamStageTrace]
    total_duration_seconds: float
    final_argument_equals_draft: bool


class HierarchicalTeamExecution(TypedDict):
    variant: Literal["hierarchical"]
    team_a: HierarchicalTeamDetail
    team_b: HierarchicalTeamDetail
    synthesis_duration_seconds: float


TeamExecution = ControlTeamExecution | HierarchicalTeamExecution


class ReplayTimings(TypedDict):
    team_a_duration_seconds: float
    team_b_duration_seconds: float
    synthesis_duration_seconds: float
    total_replay_duration_seconds: float
    team_stage_timings: list[TeamStageTrace]


class ModelCallCounts(TypedDict):
    replay_decomposition: int
    replay_research: int
    team_a_argument: int
    team_b_argument: int
    team_a_analysis: int
    team_a_draft: int
    team_a_review: int
    team_b_analysis: int
    team_b_draft: int
    team_b_review: int
    synthesis: int
    total_model_calls: int


class ReplayRunRecord(TypedDict):
    run_record_version: str
    provenance: ReplayProvenance
    replay_source: ReplaySourceProvenance
    input: ReplayInput
    decomposition: ReplayDecomposition
    source_search_evidence: list[SearchCallEvidence]
    graph_evidence: ReplayGraphEvidence
    team_execution: TeamExecution
    timings: ReplayTimings
    model_call_counts: ModelCallCounts
    metrics: ScorecardData


_TOP_LEVEL_KEYS = frozenset(ReplayRunRecord.__required_keys__)
_PROVENANCE_KEYS = frozenset(ReplayProvenance.__required_keys__)
_SOURCE_PROVENANCE_KEYS = frozenset(ReplaySourceProvenance.__required_keys__)
_INPUT_KEYS = frozenset(ReplayInput.__required_keys__)
_DECOMPOSITION_KEYS = frozenset(ReplayDecomposition.__required_keys__)
_GRAPH_EVIDENCE_KEYS = frozenset(ReplayGraphEvidence.__required_keys__)
_CONTROL_EXECUTION_KEYS = frozenset(ControlTeamExecution.__required_keys__)
_CONTROL_TEAM_KEYS = frozenset(ControlTeamDetail.__required_keys__)
_HIERARCHICAL_EXECUTION_KEYS = frozenset(
    HierarchicalTeamExecution.__required_keys__
)
_HIERARCHICAL_TEAM_KEYS = frozenset(HierarchicalTeamDetail.__required_keys__)
_TIMING_KEYS = frozenset(ReplayTimings.__required_keys__)
_CALL_COUNT_KEYS = frozenset(ModelCallCounts.__required_keys__)
_STAGE_TRACE_KEYS = frozenset(TeamStageTrace.__required_keys__)


def build_replay_run_record(
    *,
    variant: ReplayVariant,
    replay_source: ReplaySource,
    graph_result: object,
    team_execution: TeamExecution,
    timings: ReplayTimings,
    model_call_counts: ModelCallCounts,
    scorecard: Scorecard,
    run_timestamp_utc: str,
    implementation_commit: str,
    model_name: str,
) -> ReplayRunRecord:
    """Build and strictly validate one replay run record."""

    source = validate_replay_source(replay_source)
    source_record = source["source_record"]
    if variant not in ("control", "hierarchical"):
        raise ReplayRunRecordValidationError("replay variant is invalid")
    if not isinstance(graph_result, dict):
        raise ReplayRunRecordValidationError("replay graph result is invalid")
    architecture = (
        CONTROL_ARCHITECTURE_VERSION
        if variant == "control"
        else HIERARCHICAL_ARCHITECTURE_VERSION
    )
    metrics = cast(ScorecardData, json.loads(scorecard_to_json(scorecard)))
    try:
        graph_evidence: ReplayGraphEvidence = {
            "raw_research_findings": copy.deepcopy(
                graph_result["raw_research_findings"]
            ),
            "consolidated_evidence": copy.deepcopy(
                graph_result["consolidated_evidence"]
            ),
            "research_findings": copy.deepcopy(graph_result["research_findings"]),
            "team_a_argument": graph_result["team_a_argument"],
            "team_b_argument": graph_result["team_b_argument"],
            "synthesis": graph_result["synthesis"],
        }
    except KeyError:
        raise ReplayRunRecordValidationError(
            "replay graph result is incomplete"
        ) from None
    try:
        normalized_commit = normalize_implementation_commit(implementation_commit)
    except Exception:
        raise ReplayRunRecordValidationError(
            "replay implementation commit is invalid"
        ) from None
    record: ReplayRunRecord = {
        "run_record_version": REPLAY_RUN_RECORD_VERSION,
        "provenance": {
            "run_timestamp_utc": run_timestamp_utc,
            "implementation_commit": normalized_commit,
            "architecture_version": architecture,
            "metric_version": METRIC_VERSION,
            "model": {
                "name": model_name,
                "temperature": 0.0,
                "reasoning": False,
                "structured_output_method": "json_schema",
            },
        },
        "replay_source": {
            "source_artifact_path": source["source_artifact_path"],
            "source_sha256": source["source_sha256"],
            "source_run_record_version": source_record["run_record_version"],
            "source_architecture_version": source_record["provenance"][
                "architecture_version"
            ],
            "source_implementation_commit": source_record["provenance"][
                "implementation_commit"
            ],
            "source_run_timestamp_utc": source_record["provenance"][
                "run_timestamp_utc"
            ],
        },
        "input": copy.deepcopy(source_record["input"]),
        "decomposition": copy.deepcopy(source_record["decomposition"]),
        "source_search_evidence": copy.deepcopy(source_record["search_evidence"]),
        "graph_evidence": graph_evidence,
        "team_execution": copy.deepcopy(team_execution),
        "timings": copy.deepcopy(timings),
        "model_call_counts": copy.deepcopy(model_call_counts),
        "metrics": metrics,
    }
    return validate_replay_run_record(record, replay_source=source)


def validate_replay_run_record(
    value: object,
    *,
    replay_source: ReplaySource,
) -> ReplayRunRecord:
    """Validate all version 1.1 fields and cross-field replay invariants."""

    source = validate_replay_source(replay_source)
    source_record = source["source_record"]
    record = _require_dict(value, _TOP_LEVEL_KEYS, "replay run record")
    if record["run_record_version"] != REPLAY_RUN_RECORD_VERSION:
        raise ReplayRunRecordValidationError("replay run-record version is invalid")

    provenance, variant = _validate_provenance(record["provenance"])
    source_provenance = _validate_source_provenance(
        record["replay_source"], source
    )
    run_input = _require_dict(record["input"], _INPUT_KEYS, "replay input")
    if run_input != source_record["input"]:
        raise ReplayRunRecordValidationError("replay input differs from its source")
    decomposition = _require_dict(
        record["decomposition"], _DECOMPOSITION_KEYS, "replay decomposition"
    )
    if decomposition != source_record["decomposition"]:
        raise ReplayRunRecordValidationError(
            "replay decomposition differs from its source"
        )
    if record["source_search_evidence"] != source_record["search_evidence"]:
        raise ReplayRunRecordValidationError(
            "replay search evidence differs from its source"
        )
    source_search_evidence = copy.deepcopy(source_record["search_evidence"])
    graph_evidence = _validate_graph_evidence(record["graph_evidence"], source)
    team_execution = _validate_team_execution(
        record["team_execution"],
        variant=variant,
        graph_evidence=graph_evidence,
    )
    timings = _validate_timings(
        record["timings"], variant=variant, team_execution=team_execution
    )
    model_call_counts = _validate_model_call_counts(
        record["model_call_counts"], variant=variant, timings=timings
    )

    reconstructed: ExpandedDebateResult = {
        "question": source_record["input"]["question"],
        "team_a_stance": source_record["input"]["team_a_stance"],
        "team_b_stance": source_record["input"]["team_b_stance"],
        "question_complexity": source_record["decomposition"][
            "question_complexity"
        ],
        "subquestions": source_record["decomposition"]["subquestions"].copy(),
        "raw_research_findings": copy.deepcopy(
            graph_evidence["raw_research_findings"]
        ),
        "consolidated_evidence": copy.deepcopy(
            graph_evidence["consolidated_evidence"]
        ),
        "research_findings": graph_evidence["research_findings"].copy(),
        "team_a_argument": graph_evidence["team_a_argument"],
        "team_b_argument": graph_evidence["team_b_argument"],
        "synthesis": graph_evidence["synthesis"],
    }
    expected_metrics = json.loads(scorecard_to_json(score_expanded_result(reconstructed)))
    metrics = record["metrics"]
    if not isinstance(metrics, dict) or metrics != expected_metrics:
        raise ReplayRunRecordValidationError(
            "replay metrics do not match frozen metric version 1.0"
        )
    _reject_nonfinite_numbers(record)

    return {
        "run_record_version": REPLAY_RUN_RECORD_VERSION,
        "provenance": provenance,
        "replay_source": source_provenance,
        "input": copy.deepcopy(source_record["input"]),
        "decomposition": copy.deepcopy(source_record["decomposition"]),
        "source_search_evidence": source_search_evidence,
        "graph_evidence": graph_evidence,
        "team_execution": team_execution,
        "timings": timings,
        "model_call_counts": model_call_counts,
        "metrics": copy.deepcopy(cast(ScorecardData, metrics)),
    }


def replay_run_record_to_json(
    record: object,
    *,
    replay_source: ReplaySource,
) -> str:
    """Serialize a validated replay record with stable JSON formatting."""

    validated = validate_replay_run_record(record, replay_source=replay_source)
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


def _validate_provenance(
    value: object,
) -> tuple[ReplayProvenance, ReplayVariant]:
    data = _require_dict(value, _PROVENANCE_KEYS, "replay provenance")
    timestamp = _validate_timestamp(data["run_timestamp_utc"])
    try:
        commit = normalize_implementation_commit(data["implementation_commit"])
    except Exception:
        raise ReplayRunRecordValidationError(
            "replay implementation commit is invalid"
        ) from None
    if data["implementation_commit"] != commit:
        raise ReplayRunRecordValidationError(
            "replay implementation commit must be lowercase"
        )
    architecture = data["architecture_version"]
    if architecture == CONTROL_ARCHITECTURE_VERSION:
        variant: ReplayVariant = "control"
    elif architecture == HIERARCHICAL_ARCHITECTURE_VERSION:
        variant = "hierarchical"
    else:
        raise ReplayRunRecordValidationError("replay architecture is invalid")
    if data["metric_version"] != METRIC_VERSION:
        raise ReplayRunRecordValidationError("replay metric version is invalid")
    model = _require_dict(
        data["model"],
        {"name", "temperature", "reasoning", "structured_output_method"},
        "replay model configuration",
    )
    name = _require_nonblank(model["name"], "replay model name")
    if type(model["temperature"]) is not float or model["temperature"] != 0.0:
        raise ReplayRunRecordValidationError("replay temperature must be 0.0")
    if model["reasoning"] is not False:
        raise ReplayRunRecordValidationError("replay reasoning must be disabled")
    if model["structured_output_method"] != "json_schema":
        raise ReplayRunRecordValidationError(
            "replay structured-output method is invalid"
        )
    return (
        {
            "run_timestamp_utc": timestamp,
            "implementation_commit": commit,
            "architecture_version": architecture,
            "metric_version": METRIC_VERSION,
            "model": {
                "name": name,
                "temperature": 0.0,
                "reasoning": False,
                "structured_output_method": "json_schema",
            },
        },
        variant,
    )


def _validate_source_provenance(
    value: object,
    source: ReplaySource,
) -> ReplaySourceProvenance:
    data = _require_dict(
        value, _SOURCE_PROVENANCE_KEYS, "replay source provenance"
    )
    record = source["source_record"]
    expected: ReplaySourceProvenance = {
        "source_artifact_path": source["source_artifact_path"],
        "source_sha256": source["source_sha256"],
        "source_run_record_version": record["run_record_version"],
        "source_architecture_version": record["provenance"][
            "architecture_version"
        ],
        "source_implementation_commit": record["provenance"][
            "implementation_commit"
        ],
        "source_run_timestamp_utc": record["provenance"]["run_timestamp_utc"],
    }
    if data != expected:
        raise ReplayRunRecordValidationError("replay source provenance has drifted")
    return copy.deepcopy(expected)


def _validate_graph_evidence(
    value: object,
    source: ReplaySource,
) -> ReplayGraphEvidence:
    data = _require_dict(value, _GRAPH_EVIDENCE_KEYS, "replay graph evidence")
    source_graph = source["source_record"]["graph_evidence"]
    raw = _validate_finding_list(data["raw_research_findings"], "raw findings")
    consolidated = _validate_finding_list(
        data["consolidated_evidence"], "consolidated evidence"
    )
    if raw != source_graph["raw_research_findings"]:
        raise ReplayRunRecordValidationError("replay raw findings differ from source")
    if consolidated != source_graph["consolidated_evidence"]:
        raise ReplayRunRecordValidationError(
            "replay consolidated evidence differs from source"
        )
    independently_consolidated = consolidate_research_findings(
        question=source["source_record"]["input"]["question"],
        question_complexity=source["source_record"]["decomposition"][
            "question_complexity"
        ],
        subquestions=source["source_record"]["decomposition"]["subquestions"],
        raw_findings=raw,
    )
    if independently_consolidated != consolidated:
        raise ReplayRunRecordValidationError("replay consolidation is inconsistent")
    formatted = data["research_findings"]
    if not isinstance(formatted, list) or any(
        not isinstance(item, str) for item in formatted
    ):
        raise ReplayRunRecordValidationError("replay formatted findings are invalid")
    if formatted != [format_research_finding(item) for item in consolidated]:
        raise ReplayRunRecordValidationError("replay evidence formatting has drifted")
    team_a_argument = _require_nonblank(
        data["team_a_argument"], "Team A final argument"
    )
    team_b_argument = _require_nonblank(
        data["team_b_argument"], "Team B final argument"
    )
    synthesis = _require_nonblank(data["synthesis"], "neutral synthesis")
    return {
        "raw_research_findings": raw,
        "consolidated_evidence": consolidated,
        "research_findings": formatted.copy(),
        "team_a_argument": team_a_argument,
        "team_b_argument": team_b_argument,
        "synthesis": synthesis,
    }


def _validate_team_execution(
    value: object,
    *,
    variant: ReplayVariant,
    graph_evidence: ReplayGraphEvidence,
) -> TeamExecution:
    if variant == "control":
        data = _require_dict(
            value, _CONTROL_EXECUTION_KEYS, "control team execution"
        )
        if data["variant"] != "control":
            raise ReplayRunRecordValidationError("team-execution variant is invalid")
        team_a = _validate_control_team(
            data["team_a"], graph_evidence["team_a_argument"], "Team A"
        )
        team_b = _validate_control_team(
            data["team_b"], graph_evidence["team_b_argument"], "Team B"
        )
        return {
            "variant": "control",
            "team_a": team_a,
            "team_b": team_b,
            "synthesis_duration_seconds": _finite_nonnegative(
                data["synthesis_duration_seconds"], "synthesis duration"
            ),
        }

    data = _require_dict(
        value, _HIERARCHICAL_EXECUTION_KEYS, "hierarchical team execution"
    )
    if data["variant"] != "hierarchical":
        raise ReplayRunRecordValidationError("team-execution variant is invalid")
    team_a = _validate_hierarchical_team(
        data["team_a"],
        team="team_a",
        expected_stance=TEAM_A_STANCE,
        expected_argument=graph_evidence["team_a_argument"],
        expected_opponent=None,
        evidence=graph_evidence["consolidated_evidence"],
    )
    team_b = _validate_hierarchical_team(
        data["team_b"],
        team="team_b",
        expected_stance=TEAM_B_STANCE,
        expected_argument=graph_evidence["team_b_argument"],
        expected_opponent=graph_evidence["team_a_argument"],
        evidence=graph_evidence["consolidated_evidence"],
    )
    return {
        "variant": "hierarchical",
        "team_a": team_a,
        "team_b": team_b,
        "synthesis_duration_seconds": _finite_nonnegative(
            data["synthesis_duration_seconds"], "synthesis duration"
        ),
    }


def _validate_control_team(
    value: object,
    expected_argument: str,
    label: str,
) -> ControlTeamDetail:
    data = _require_dict(value, _CONTROL_TEAM_KEYS, f"{label} control execution")
    argument = _require_nonblank(data["final_argument"], f"{label} argument")
    if argument != expected_argument:
        raise ReplayRunRecordValidationError(
            f"{label} execution argument differs from graph output"
        )
    return {
        "final_argument": argument,
        "duration_seconds": _finite_nonnegative(
            data["duration_seconds"], f"{label} duration"
        ),
    }


def _validate_hierarchical_team(
    value: object,
    *,
    team: TeamName,
    expected_stance: str,
    expected_argument: str,
    expected_opponent: str | None,
    evidence: list[ResearchFinding],
) -> HierarchicalTeamDetail:
    data = _require_dict(
        value, _HIERARCHICAL_TEAM_KEYS, f"{team} hierarchical execution"
    )
    if data["team"] != team:
        raise ReplayRunRecordValidationError("team execution has the wrong team")
    if data["opponent_argument"] != expected_opponent:
        raise ReplayRunRecordValidationError(
            "team execution has the wrong opponent argument"
        )
    try:
        work_product = validate_team_work_product(
            {
                "team": data["team"],
                "assigned_stance": data["assigned_stance"],
                "evidence_analysis": data["evidence_analysis"],
                "draft_argument": data["draft_argument"],
                "internal_critique": data["internal_critique"],
                "final_argument": data["final_argument"],
            },
            consolidated_evidence=evidence,
            expected_team=team,
            expected_stance=expected_stance,
        )
    except Exception:
        raise ReplayRunRecordValidationError(
            "hierarchical team work product is invalid"
        ) from None
    if work_product["final_argument"] != expected_argument:
        raise ReplayRunRecordValidationError(
            "hierarchical final argument differs from graph output"
        )
    stages = _validate_stage_timings(data["stage_timings"], expected_team=team)
    equals_draft = data["final_argument_equals_draft"]
    if type(equals_draft) is not bool or equals_draft != (
        work_product["final_argument"] == work_product["draft_argument"]
    ):
        raise ReplayRunRecordValidationError(
            "draft-revision equality marker is inconsistent"
        )
    return {
        "team": team,
        "assigned_stance": work_product["assigned_stance"],
        "opponent_argument": expected_opponent,
        "evidence_analysis": copy.deepcopy(work_product["evidence_analysis"]),
        "draft_argument": work_product["draft_argument"],
        "internal_critique": copy.deepcopy(work_product["internal_critique"]),
        "final_argument": work_product["final_argument"],
        "stage_timings": stages,
        "total_duration_seconds": _finite_nonnegative(
            data["total_duration_seconds"], f"{team} total duration"
        ),
        "final_argument_equals_draft": equals_draft,
    }


def _validate_timings(
    value: object,
    *,
    variant: ReplayVariant,
    team_execution: TeamExecution,
) -> ReplayTimings:
    data = _require_dict(value, _TIMING_KEYS, "replay timings")
    team_a = _finite_nonnegative(data["team_a_duration_seconds"], "Team A timing")
    team_b = _finite_nonnegative(data["team_b_duration_seconds"], "Team B timing")
    synthesis = _finite_nonnegative(
        data["synthesis_duration_seconds"], "synthesis timing"
    )
    total = _finite_nonnegative(
        data["total_replay_duration_seconds"], "total replay timing"
    )
    if (
        team_a != team_execution["team_a"]["duration_seconds"]
        if variant == "control"
        else team_a != team_execution["team_a"]["total_duration_seconds"]
    ):
        raise ReplayRunRecordValidationError("Team A timing is inconsistent")
    if (
        team_b != team_execution["team_b"]["duration_seconds"]
        if variant == "control"
        else team_b != team_execution["team_b"]["total_duration_seconds"]
    ):
        raise ReplayRunRecordValidationError("Team B timing is inconsistent")
    if synthesis != team_execution["synthesis_duration_seconds"]:
        raise ReplayRunRecordValidationError("synthesis timing is inconsistent")

    raw_stages = data["team_stage_timings"]
    if variant == "control":
        if raw_stages != []:
            raise ReplayRunRecordValidationError(
                "control replay cannot contain hierarchical stage timings"
            )
        stages: list[TeamStageTrace] = []
    else:
        stages = [
            *copy.deepcopy(team_execution["team_a"]["stage_timings"]),
            *copy.deepcopy(team_execution["team_b"]["stage_timings"]),
        ]
        if raw_stages != stages:
            raise ReplayRunRecordValidationError(
                "hierarchical stage timings are inconsistent"
            )
    return {
        "team_a_duration_seconds": team_a,
        "team_b_duration_seconds": team_b,
        "synthesis_duration_seconds": synthesis,
        "total_replay_duration_seconds": total,
        "team_stage_timings": stages,
    }


def _validate_model_call_counts(
    value: object,
    *,
    variant: ReplayVariant,
    timings: ReplayTimings,
) -> ModelCallCounts:
    data = _require_dict(value, _CALL_COUNT_KEYS, "model call counts")
    copied: dict[str, int] = {}
    for field in ModelCallCounts.__required_keys__:
        item = data[field]
        if type(item) is not int or item < 0:
            raise ReplayRunRecordValidationError(
                "model call counts must be nonnegative integers"
            )
        copied[field] = item
    expected = expected_model_call_counts(variant)
    if copied != expected:
        raise ReplayRunRecordValidationError(
            "model call counts do not match successful replay execution"
        )
    if variant == "hierarchical" and len(timings["team_stage_timings"]) != 6:
        raise ReplayRunRecordValidationError(
            "hierarchical model calls lack successful stage traces"
        )
    return cast(ModelCallCounts, copied)


def expected_model_call_counts(variant: ReplayVariant) -> ModelCallCounts:
    """Return the fixed expected count shape for one successful replay."""

    if variant == "control":
        return {
            "replay_decomposition": 0,
            "replay_research": 0,
            "team_a_argument": 1,
            "team_b_argument": 1,
            "team_a_analysis": 0,
            "team_a_draft": 0,
            "team_a_review": 0,
            "team_b_analysis": 0,
            "team_b_draft": 0,
            "team_b_review": 0,
            "synthesis": 1,
            "total_model_calls": 3,
        }
    if variant == "hierarchical":
        return {
            "replay_decomposition": 0,
            "replay_research": 0,
            "team_a_argument": 0,
            "team_b_argument": 0,
            "team_a_analysis": 1,
            "team_a_draft": 1,
            "team_a_review": 1,
            "team_b_analysis": 1,
            "team_b_draft": 1,
            "team_b_review": 1,
            "synthesis": 1,
            "total_model_calls": 7,
        }
    raise ReplayRunRecordValidationError("replay variant is invalid")


def _validate_stage_timings(
    value: object,
    *,
    expected_team: TeamName,
) -> list[TeamStageTrace]:
    if not isinstance(value, list) or len(value) != len(TEAM_STAGES):
        raise ReplayRunRecordValidationError("team stage timings are incomplete")
    copied: list[TeamStageTrace] = []
    for raw, expected_stage in zip(value, TEAM_STAGES, strict=True):
        data = _require_dict(raw, _STAGE_TRACE_KEYS, "team stage timing")
        if data["team"] != expected_team or data["stage"] != expected_stage:
            raise ReplayRunRecordValidationError("team stage timing is misrouted")
        if data["status"] != "success" or data["error_category"] is not None:
            raise ReplayRunRecordValidationError("team stage timing is not successful")
        copied.append(
            {
                "team": expected_team,
                "stage": expected_stage,
                "duration_seconds": _finite_nonnegative(
                    data["duration_seconds"], "team stage duration"
                ),
                "status": "success",
                "error_category": None,
            }
        )
    return copied


def _validate_finding_list(value: object, context: str) -> list[ResearchFinding]:
    if not isinstance(value, list) or not value:
        raise ReplayRunRecordValidationError(f"{context} must be a nonempty list")
    copied: list[ResearchFinding] = []
    for item in value:
        try:
            copied.append(validate_research_finding(item))
        except ValueError:
            raise ReplayRunRecordValidationError(f"{context} is invalid") from None
    return copied


def _require_dict(
    value: object,
    expected_keys: frozenset[str] | set[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ReplayRunRecordValidationError(f"{context} has an invalid schema")
    return cast(dict[str, object], value)


def _require_nonblank(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayRunRecordValidationError(f"{context} must be nonblank")
    return value


def _finite_nonnegative(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayRunRecordValidationError(f"{context} must be numeric")
    copied = float(value)
    if not math.isfinite(copied) or copied < 0.0:
        raise ReplayRunRecordValidationError(
            f"{context} must be finite and nonnegative"
        )
    return copied


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ReplayRunRecordValidationError("replay timestamp must be UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ReplayRunRecordValidationError("replay timestamp is invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReplayRunRecordValidationError("replay timestamp must be UTC")
    return value


def _reject_nonfinite_numbers(value: object) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ReplayRunRecordValidationError("replay record contains nonfinite data")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite_numbers(item)
