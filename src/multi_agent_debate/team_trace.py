"""Thread-safe, credential-free tracing for hierarchical team subgraphs."""

from __future__ import annotations

import copy
import math
import threading
from typing import Literal, TypedDict

from multi_agent_debate.team_model import (
    HierarchicalTeamError,
    TeamGraphValidationError,
    TeamModelError,
    TeamModelInvocationError,
    TeamModelResponseError,
    TeamName,
    validate_team_name,
)

TeamStage = Literal["evidence_analysis", "argument_draft", "review_revision"]
TraceStatus = Literal["success", "error"]

TEAM_STAGES: tuple[TeamStage, ...] = (
    "evidence_analysis",
    "argument_draft",
    "review_revision",
)
TEAM_ERROR_CATEGORIES = frozenset(
    {
        "team_model_invocation_error",
        "team_model_response_error",
        "team_model_error",
        "team_graph_validation_error",
        "hierarchical_team_error",
        "project_error",
        "unexpected_provider_error",
    }
)


class TeamStageTrace(TypedDict):
    team: TeamName
    stage: TeamStage
    duration_seconds: float
    status: TraceStatus
    error_category: str | None


class TeamTotalTrace(TypedDict):
    team: TeamName
    duration_seconds: float
    status: Literal["success"]
    error_category: None


class CanonicalTeamTraces(TypedDict):
    stage_traces: list[TeamStageTrace]
    total_traces: list[TeamTotalTrace]


def validate_trace_duration(value: object) -> float:
    """Return a finite, nonnegative duration or reject the trace."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeamGraphValidationError("trace duration must be numeric")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0.0:
        raise TeamGraphValidationError(
            "trace duration must be finite and nonnegative"
        )
    return duration


def bounded_team_error_category(error: BaseException) -> str:
    """Map an exception type to bounded trace data without its message."""

    if isinstance(error, TeamModelInvocationError):
        return "team_model_invocation_error"
    if isinstance(error, TeamModelResponseError):
        return "team_model_response_error"
    if isinstance(error, TeamModelError):
        return "team_model_error"
    if isinstance(error, TeamGraphValidationError):
        return "team_graph_validation_error"
    if isinstance(error, HierarchicalTeamError):
        return "hierarchical_team_error"
    if type(error).__module__.startswith("multi_agent_debate."):
        return "project_error"
    return "unexpected_provider_error"


class TeamTraceRecorder:
    """Collect team-stage and team-total traces using defensive copies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage_traces: list[TeamStageTrace] = []
        self._total_traces: list[TeamTotalTrace] = []

    def record_stage(
        self,
        *,
        team: TeamName,
        stage: TeamStage,
        duration_seconds: float,
        status: TraceStatus,
        error_category: str | None,
    ) -> None:
        """Validate and append one bounded stage trace."""

        validated_team = validate_team_name(team)
        if stage not in TEAM_STAGES:
            raise TeamGraphValidationError("unknown team trace stage")
        if status not in ("success", "error"):
            raise TeamGraphValidationError("unknown team trace status")
        if status == "success" and error_category is not None:
            raise TeamGraphValidationError(
                "successful stage traces cannot contain an error category"
            )
        if status == "error" and (
            not isinstance(error_category, str) or not error_category.strip()
        ):
            raise TeamGraphValidationError(
                "failed stage traces require a bounded error category"
            )
        if status == "error" and error_category not in TEAM_ERROR_CATEGORIES:
            raise TeamGraphValidationError("unknown team trace error category")
        trace: TeamStageTrace = {
            "team": validated_team,
            "stage": stage,
            "duration_seconds": validate_trace_duration(duration_seconds),
            "status": status,
            "error_category": error_category,
        }
        with self._lock:
            self._stage_traces.append(copy.deepcopy(trace))

    def record_total_success(
        self,
        *,
        team: TeamName,
        duration_seconds: float,
    ) -> None:
        """Append the total duration emitted only by a reached finalizer."""

        trace: TeamTotalTrace = {
            "team": validate_team_name(team),
            "duration_seconds": validate_trace_duration(duration_seconds),
            "status": "success",
            "error_category": None,
        }
        with self._lock:
            self._total_traces.append(copy.deepcopy(trace))

    def stage_snapshot(self) -> list[TeamStageTrace]:
        """Return a deep snapshot that cannot mutate recorder state."""

        with self._lock:
            return copy.deepcopy(self._stage_traces)

    def total_snapshot(self) -> list[TeamTotalTrace]:
        """Return a deep snapshot that cannot mutate recorder state."""

        with self._lock:
            return copy.deepcopy(self._total_traces)


def canonicalize_successful_team_traces(
    recorder: TeamTraceRecorder,
) -> CanonicalTeamTraces:
    """Require and order exactly three stages plus one total per team."""

    stages = recorder.stage_snapshot()
    totals = recorder.total_snapshot()
    stage_by_key: dict[tuple[TeamName, TeamStage], TeamStageTrace] = {}
    for trace in stages:
        if trace["status"] != "success" or trace["error_category"] is not None:
            raise TeamGraphValidationError(
                "canonical successful traces cannot contain failures"
            )
        key = (trace["team"], trace["stage"])
        if key in stage_by_key:
            raise TeamGraphValidationError("duplicate team stage trace")
        stage_by_key[key] = copy.deepcopy(trace)

    total_by_team: dict[TeamName, TeamTotalTrace] = {}
    for trace in totals:
        team = trace["team"]
        if team in total_by_team:
            raise TeamGraphValidationError("duplicate team total trace")
        total_by_team[team] = copy.deepcopy(trace)

    expected_stage_keys = {
        (team, stage)
        for team in ("team_a", "team_b")
        for stage in TEAM_STAGES
    }
    if set(stage_by_key) != expected_stage_keys:
        raise TeamGraphValidationError("team stage traces are incomplete or unexpected")
    if set(total_by_team) != {"team_a", "team_b"}:
        raise TeamGraphValidationError("team total traces are incomplete or unexpected")

    return {
        "stage_traces": [
            copy.deepcopy(stage_by_key[(team, stage)])
            for team in ("team_a", "team_b")
            for stage in TEAM_STAGES
        ],
        "total_traces": [
            copy.deepcopy(total_by_team[team]) for team in ("team_a", "team_b")
        ],
    }
