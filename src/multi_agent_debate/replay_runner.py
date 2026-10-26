"""Pure provider-injected execution for paired deterministic replay variants."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal, TypedDict, cast

from multi_agent_debate.expanded_graph import run_expanded_debate
from multi_agent_debate.expanded_state import ExpandedDebateResult
from multi_agent_debate.hierarchical_graph import run_hierarchical_debate
from multi_agent_debate.model import DebateModel
from multi_agent_debate.metrics import scorecard_to_json
from multi_agent_debate.replay import (
    ReplayQuestionDecomposer,
    ReplayResearchProvider,
    ReplaySource,
    validate_replay_source,
)
from multi_agent_debate.replay_run_record import (
    ControlTeamExecution,
    HierarchicalTeamExecution,
    ModelCallCounts,
    ReplayRunRecord,
    ReplayTimings,
    ReplayVariant,
    build_replay_run_record,
    expected_model_call_counts,
    replay_run_record_to_json,
)
from multi_agent_debate.research import ResearchFinding
from multi_agent_debate.run_record import format_utc_timestamp, score_expanded_result
from multi_agent_debate.team_model import (
    EvidenceAnalysis,
    NeutralSynthesisModel,
    TeamModel,
    TeamName,
    TeamRevision,
)
from multi_agent_debate.team_state import HierarchicalDebateResult
from multi_agent_debate.team_trace import (
    CanonicalTeamTraces,
    TeamTraceRecorder,
    canonicalize_successful_team_traces,
)


class ReplayRunnerError(RuntimeError):
    """Base error for pure replay execution and trace validation."""


class ReplayRunnerValidationError(ReplayRunnerError):
    """Injected dependencies or successful call traces are inconsistent."""


class ReplayArtifacts(TypedDict):
    variant: ReplayVariant
    run_record: ReplayRunRecord
    run_record_json: str
    scorecard_json: str


class _ControlCall(TypedDict):
    team: Literal["team_a", "team_b"]
    duration_seconds: float


class _TimedSynthesisModel:
    def __init__(
        self,
        model: NeutralSynthesisModel,
        clock: Callable[[], float],
    ) -> None:
        self._model = model
        self._clock = clock
        self.durations: list[float] = []

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: Sequence[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        started = _clock_value(self._clock)
        result = self._model.synthesize_neutrally(
            question=question,
            research_findings=list(research_findings),
            team_a_argument=team_a_argument,
            team_b_argument=team_b_argument,
        )
        self.durations.append(_elapsed(self._clock, started))
        return result


class _TimedControlModel:
    def __init__(
        self,
        model: DebateModel,
        synthesis: _TimedSynthesisModel,
        clock: Callable[[], float],
    ) -> None:
        self._model = model
        self._synthesis = synthesis
        self._clock = clock
        self.argument_calls: list[_ControlCall] = []

    def research(self, *, question: str) -> list[str]:
        raise AssertionError("replay expanded graph must not use model research")

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        team: Literal["team_a", "team_b"] = (
            "team_a" if opponent_argument is None else "team_b"
        )
        started = _clock_value(self._clock)
        result = self._model.write_argument(
            question=question,
            stance=stance,
            research_findings=research_findings.copy(),
            opponent_argument=opponent_argument,
        )
        self.argument_calls.append(
            {"team": team, "duration_seconds": _elapsed(self._clock, started)}
        )
        return result

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: list[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        return self._synthesis.synthesize_neutrally(
            question=question,
            research_findings=research_findings.copy(),
            team_a_argument=team_a_argument,
            team_b_argument=team_b_argument,
        )


class _CountingTeamModel:
    def __init__(self, model: TeamModel) -> None:
        self._model = model
        self.calls: list[tuple[TeamName, str]] = []

    def analyze_evidence(
        self,
        *,
        question: str,
        team_name: TeamName,
        stance: str,
        consolidated_evidence: Sequence[ResearchFinding],
        opponent_argument: str | None,
    ) -> EvidenceAnalysis:
        result = self._model.analyze_evidence(
            question=question,
            team_name=team_name,
            stance=stance,
            consolidated_evidence=copy.deepcopy(consolidated_evidence),
            opponent_argument=opponent_argument,
        )
        self.calls.append((team_name, "evidence_analysis"))
        return result

    def draft_argument(
        self,
        *,
        question: str,
        team_name: TeamName,
        stance: str,
        research_findings: Sequence[str],
        evidence_analysis: EvidenceAnalysis,
        opponent_argument: str | None,
    ) -> str:
        result = self._model.draft_argument(
            question=question,
            team_name=team_name,
            stance=stance,
            research_findings=list(research_findings),
            evidence_analysis=copy.deepcopy(evidence_analysis),
            opponent_argument=opponent_argument,
        )
        self.calls.append((team_name, "argument_draft"))
        return result

    def review_and_revise(
        self,
        *,
        question: str,
        team_name: TeamName,
        stance: str,
        consolidated_evidence: Sequence[ResearchFinding],
        evidence_analysis: EvidenceAnalysis,
        draft_argument: str,
        opponent_argument: str | None,
    ) -> TeamRevision:
        result = self._model.review_and_revise(
            question=question,
            team_name=team_name,
            stance=stance,
            consolidated_evidence=copy.deepcopy(consolidated_evidence),
            evidence_analysis=copy.deepcopy(evidence_analysis),
            draft_argument=draft_argument,
            opponent_argument=opponent_argument,
        )
        self.calls.append((team_name, "review_revision"))
        return result


def run_replay_variant(
    *,
    variant: ReplayVariant,
    replay_source: ReplaySource,
    control_model: DebateModel | None,
    team_model: TeamModel | None,
    synthesis_model: NeutralSynthesisModel,
    team_trace_recorder: TeamTraceRecorder | None,
    clock: Callable[[], float],
    utc_now: Callable[[], datetime],
    implementation_commit: str,
    model_name: str,
) -> ReplayArtifacts:
    """Execute one replay variant without filesystem or live-provider access."""

    if variant not in ("control", "hierarchical"):
        raise ReplayRunnerValidationError("replay variant is invalid")
    if not callable(clock) or not callable(utc_now):
        raise ReplayRunnerValidationError("replay clocks must be callable")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ReplayRunnerValidationError("replay model name must be nonblank")
    source = validate_replay_source(replay_source)
    source_record = source["source_record"]
    decomposer = ReplayQuestionDecomposer(source)
    research_provider = ReplayResearchProvider(source)
    timed_synthesis = _TimedSynthesisModel(synthesis_model, clock)
    try:
        run_timestamp = format_utc_timestamp(utc_now())
    except Exception:
        raise ReplayRunnerValidationError("replay UTC timestamp is invalid") from None
    total_started = _clock_value(clock)

    if variant == "control":
        if control_model is None or team_model is not None:
            raise ReplayRunnerValidationError(
                "control replay requires only a one-shot debate model"
            )
        if team_trace_recorder is not None and (
            team_trace_recorder.stage_snapshot()
            or team_trace_recorder.total_snapshot()
        ):
            raise ReplayRunnerValidationError(
                "control replay cannot use populated team traces"
            )
        timed_control = _TimedControlModel(control_model, timed_synthesis, clock)
        result: ExpandedDebateResult | HierarchicalDebateResult = run_expanded_debate(
            source_record["input"]["question"],
            decomposer,
            research_provider,
            timed_control,
        )
        total_duration = _elapsed(clock, total_started)
        team_execution, timings, counts = _control_execution_data(
            result, timed_control, timed_synthesis, total_duration
        )
    else:
        if team_model is None or control_model is not None:
            raise ReplayRunnerValidationError(
                "hierarchical replay requires only a team model"
            )
        recorder = team_trace_recorder or TeamTraceRecorder()
        if recorder.stage_snapshot() or recorder.total_snapshot():
            raise ReplayRunnerValidationError(
                "hierarchical replay requires an empty team trace recorder"
            )
        counted_team = _CountingTeamModel(team_model)
        result = run_hierarchical_debate(
            source_record["input"]["question"],
            decomposer,
            research_provider,
            counted_team,
            timed_synthesis,
            team_trace_recorder=recorder,
            clock=clock,
        )
        total_duration = _elapsed(clock, total_started)
        canonical = canonicalize_successful_team_traces(recorder)
        team_execution, timings, counts = _hierarchical_execution_data(
            cast(HierarchicalDebateResult, result),
            canonical,
            counted_team,
            timed_synthesis,
            total_duration,
        )

    scorecard = score_expanded_result(cast(ExpandedDebateResult, result))
    scorecard_json = scorecard_to_json(scorecard)
    record = build_replay_run_record(
        variant=variant,
        replay_source=source,
        graph_result=result,
        team_execution=team_execution,
        timings=timings,
        model_call_counts=counts,
        scorecard=scorecard,
        run_timestamp_utc=run_timestamp,
        implementation_commit=implementation_commit,
        model_name=model_name,
    )
    record_json = replay_run_record_to_json(record, replay_source=source)
    return {
        "variant": variant,
        "run_record": copy.deepcopy(record),
        "run_record_json": record_json,
        "scorecard_json": scorecard_json,
    }


def _control_execution_data(
    result: ExpandedDebateResult,
    model: _TimedControlModel,
    synthesis: _TimedSynthesisModel,
    total_duration: float,
) -> tuple[ControlTeamExecution, ReplayTimings, ModelCallCounts]:
    if [call["team"] for call in model.argument_calls] != ["team_a", "team_b"]:
        raise ReplayRunnerValidationError(
            "control model calls do not match graph execution"
        )
    if len(synthesis.durations) != 1:
        raise ReplayRunnerValidationError("control synthesis call count is invalid")
    counts = expected_model_call_counts("control")
    if len(model.argument_calls) + len(synthesis.durations) != counts[
        "total_model_calls"
    ]:
        raise ReplayRunnerValidationError("control model call count is invalid")
    team_a_duration = model.argument_calls[0]["duration_seconds"]
    team_b_duration = model.argument_calls[1]["duration_seconds"]
    synthesis_duration = synthesis.durations[0]
    execution: ControlTeamExecution = {
        "variant": "control",
        "team_a": {
            "final_argument": result["team_a_argument"],
            "duration_seconds": team_a_duration,
        },
        "team_b": {
            "final_argument": result["team_b_argument"],
            "duration_seconds": team_b_duration,
        },
        "synthesis_duration_seconds": synthesis_duration,
    }
    timings: ReplayTimings = {
        "team_a_duration_seconds": team_a_duration,
        "team_b_duration_seconds": team_b_duration,
        "synthesis_duration_seconds": synthesis_duration,
        "total_replay_duration_seconds": total_duration,
        "team_stage_timings": [],
    }
    return execution, timings, counts


def _hierarchical_execution_data(
    result: HierarchicalDebateResult,
    traces: CanonicalTeamTraces,
    model: _CountingTeamModel,
    synthesis: _TimedSynthesisModel,
    total_duration: float,
) -> tuple[HierarchicalTeamExecution, ReplayTimings, ModelCallCounts]:
    expected_calls = [
        (team, stage)
        for team in ("team_a", "team_b")
        for stage in ("evidence_analysis", "argument_draft", "review_revision")
    ]
    if model.calls != expected_calls:
        raise ReplayRunnerValidationError(
            "hierarchical model calls do not match successful stage traces"
        )
    if len(synthesis.durations) != 1:
        raise ReplayRunnerValidationError(
            "hierarchical synthesis call count is invalid"
        )
    trace_calls = [
        (trace["team"], trace["stage"]) for trace in traces["stage_traces"]
    ]
    if trace_calls != expected_calls:
        raise ReplayRunnerValidationError(
            "hierarchical stage traces do not match model calls"
        )
    counts = expected_model_call_counts("hierarchical")
    if len(model.calls) + len(synthesis.durations) != counts["total_model_calls"]:
        raise ReplayRunnerValidationError("hierarchical model call count is invalid")

    totals = {trace["team"]: trace for trace in traces["total_traces"]}
    stages = {
        team: [
            copy.deepcopy(trace)
            for trace in traces["stage_traces"]
            if trace["team"] == team
        ]
        for team in ("team_a", "team_b")
    }
    synthesis_duration = synthesis.durations[0]
    team_a_product = result["team_a_work_product"]
    team_b_product = result["team_b_work_product"]
    execution: HierarchicalTeamExecution = {
        "variant": "hierarchical",
        "team_a": {
            "team": "team_a",
            "assigned_stance": team_a_product["assigned_stance"],
            "opponent_argument": None,
            "evidence_analysis": copy.deepcopy(team_a_product["evidence_analysis"]),
            "draft_argument": team_a_product["draft_argument"],
            "internal_critique": copy.deepcopy(
                team_a_product["internal_critique"]
            ),
            "final_argument": team_a_product["final_argument"],
            "stage_timings": stages["team_a"],
            "total_duration_seconds": totals["team_a"]["duration_seconds"],
            "final_argument_equals_draft": (
                team_a_product["final_argument"]
                == team_a_product["draft_argument"]
            ),
        },
        "team_b": {
            "team": "team_b",
            "assigned_stance": team_b_product["assigned_stance"],
            "opponent_argument": result["team_a_argument"],
            "evidence_analysis": copy.deepcopy(team_b_product["evidence_analysis"]),
            "draft_argument": team_b_product["draft_argument"],
            "internal_critique": copy.deepcopy(
                team_b_product["internal_critique"]
            ),
            "final_argument": team_b_product["final_argument"],
            "stage_timings": stages["team_b"],
            "total_duration_seconds": totals["team_b"]["duration_seconds"],
            "final_argument_equals_draft": (
                team_b_product["final_argument"]
                == team_b_product["draft_argument"]
            ),
        },
        "synthesis_duration_seconds": synthesis_duration,
    }
    timings: ReplayTimings = {
        "team_a_duration_seconds": totals["team_a"]["duration_seconds"],
        "team_b_duration_seconds": totals["team_b"]["duration_seconds"],
        "synthesis_duration_seconds": synthesis_duration,
        "total_replay_duration_seconds": total_duration,
        "team_stage_timings": copy.deepcopy(traces["stage_traces"]),
    }
    return execution, timings, counts


def _clock_value(clock: Callable[[], float]) -> float:
    try:
        value = clock()
    except Exception:
        raise ReplayRunnerValidationError("replay clock failed") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayRunnerValidationError("replay clock must return a number")
    copied = float(value)
    if not math.isfinite(copied) or copied < 0.0:
        raise ReplayRunnerValidationError(
            "replay clock must return finite nonnegative values"
        )
    return copied


def _elapsed(clock: Callable[[], float], started: float) -> float:
    duration = _clock_value(clock) - started
    if not math.isfinite(duration) or duration < 0.0:
        raise ReplayRunnerValidationError(
            "replay duration must be finite and nonnegative"
        )
    return duration
