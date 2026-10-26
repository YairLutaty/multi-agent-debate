"""Parameterized compiled LangGraph subgraph for one hierarchical debate team."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from typing import TypeVar, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_debate.research import (
    ResearchFinding,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.team_model import (
    EvidenceAnalysis,
    HierarchicalTeamError,
    TeamGraphValidationError,
    TeamModel,
    TeamModelInvocationError,
    TeamModelResponseError,
    TeamName,
    TeamRevision,
    TeamWorkProduct,
    validate_configured_stance,
    validate_evidence_analysis,
    validate_team_argument,
    validate_team_name,
    validate_team_revision,
    validate_team_work_product,
)
from multi_agent_debate.team_state import (
    TeamAInput,
    TeamAResult,
    TeamAState,
    TeamAnalysisUpdate,
    TeamBInput,
    TeamBResult,
    TeamBState,
    TeamDraftNodeInput,
    TeamDraftUpdate,
    TeamFinalizeNodeInput,
    TeamReviewNodeInput,
    TeamRevisionUpdate,
)
from multi_agent_debate.team_trace import (
    TeamStage,
    TeamTraceRecorder,
    bounded_team_error_category,
    validate_trace_duration,
)

_T = TypeVar("_T")


def _clock_value(clock: Callable[[], float]) -> float:
    try:
        value = clock()
    except Exception:
        raise TeamGraphValidationError("team clock failed") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeamGraphValidationError("team clock must return a number")
    copied = float(value)
    if not math.isfinite(copied) or copied < 0.0:
        raise TeamGraphValidationError(
            "team clock must return a finite nonnegative value"
        )
    return copied


def _copy_team_inputs(
    *,
    question: object,
    consolidated_evidence: object,
    research_findings: object,
) -> tuple[str, list[ResearchFinding], list[str]]:
    if not isinstance(question, str) or not question.strip():
        raise TeamGraphValidationError("team question must be nonblank")
    if not isinstance(consolidated_evidence, list) or not consolidated_evidence:
        raise TeamGraphValidationError(
            "team consolidated evidence must be a nonempty list"
        )
    copied_evidence: list[ResearchFinding] = []
    for finding in consolidated_evidence:
        try:
            copied_evidence.append(validate_research_finding(finding))
        except ValueError:
            raise TeamGraphValidationError(
                "team consolidated evidence contains an invalid finding"
            ) from None
    if not isinstance(research_findings, list):
        raise TeamGraphValidationError("formatted research must be a list")
    if any(
        not isinstance(finding, str) or not finding.strip()
        for finding in research_findings
    ):
        raise TeamGraphValidationError(
            "formatted research must contain only nonblank strings"
        )
    copied_formatted = research_findings.copy()
    expected_formatted = [
        format_research_finding(finding) for finding in copied_evidence
    ]
    if copied_formatted != expected_formatted:
        raise TeamGraphValidationError(
            "formatted research must exactly match consolidated evidence"
        )
    return question, copied_evidence, copied_formatted


def _invoke_model(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except HierarchicalTeamError:
        raise
    except Exception:
        raise TeamModelInvocationError("team model invocation failed") from None


def build_team_subgraph(
    *,
    team_name: TeamName,
    stance: str,
    model: TeamModel,
    trace_recorder: TeamTraceRecorder | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CompiledStateGraph:
    """Build one static four-node team graph for either parent role."""

    validated_team = validate_team_name(team_name)
    configured_stance = validate_configured_stance(stance)
    if not callable(clock):
        raise TeamGraphValidationError("team clock must be callable")
    if trace_recorder is not None and not isinstance(
        trace_recorder, TeamTraceRecorder
    ):
        raise TeamGraphValidationError("trace recorder has the wrong type")
    for method_name in (
        "analyze_evidence",
        "draft_argument",
        "review_and_revise",
    ):
        if not callable(getattr(model, method_name, None)):
            raise TeamGraphValidationError("team model is missing a required method")

    def run_stage(stage: TeamStage, operation: Callable[[], _T]) -> _T:
        started_at = _clock_value(clock)
        try:
            result = operation()
        except Exception as error:
            duration = validate_trace_duration(_clock_value(clock) - started_at)
            if trace_recorder is not None:
                trace_recorder.record_stage(
                    team=validated_team,
                    stage=stage,
                    duration_seconds=duration,
                    status="error",
                    error_category=bounded_team_error_category(error),
                )
            raise
        duration = validate_trace_duration(_clock_value(clock) - started_at)
        if trace_recorder is not None:
            trace_recorder.record_stage(
                team=validated_team,
                stage=stage,
                duration_seconds=duration,
                status="success",
                error_category=None,
            )
        return result

    def evidence_analyst(
        state: TeamAInput | TeamBInput,
    ) -> TeamAnalysisUpdate:
        team_started_at = _clock_value(clock)

        def execute() -> TeamAnalysisUpdate:
            try:
                parent_stance = (
                    state["team_a_stance"]
                    if validated_team == "team_a"
                    else state["team_b_stance"]
                )
                question, evidence, formatted = _copy_team_inputs(
                    question=state["question"],
                    consolidated_evidence=state["consolidated_evidence"],
                    research_findings=state["research_findings"],
                )
            except (KeyError, TypeError):
                raise TeamGraphValidationError("team input is incomplete") from None
            if parent_stance != configured_stance:
                raise TeamGraphValidationError(
                    "parent stance does not match the configured team stance"
                )
            opponent_argument: str | None
            if validated_team == "team_a":
                opponent_argument = None
            else:
                try:
                    opponent_argument = validate_team_argument(
                        state["team_a_argument"],  # type: ignore[typeddict-item]
                        field_name="opponent_argument",
                    )
                except KeyError:
                    raise TeamGraphValidationError(
                        "Team B input requires Team A's final argument"
                    ) from None

            raw_analysis = _invoke_model(
                lambda: model.analyze_evidence(
                    question=question,
                    team_name=validated_team,
                    stance=configured_stance,
                    consolidated_evidence=copy.deepcopy(evidence),
                    opponent_argument=opponent_argument,
                )
            )
            analysis = validate_evidence_analysis(
                raw_analysis,
                team_name=validated_team,
                consolidated_evidence=evidence,
            )
            # ``formatted`` is validated here even though it remains an input
            # channel for the next stage; this locks the evidence boundary.
            if formatted != [format_research_finding(item) for item in evidence]:
                raise TeamGraphValidationError("team evidence boundary changed")
            return {
                "assigned_stance": configured_stance,
                "opponent_argument": opponent_argument,
                "evidence_analysis": copy.deepcopy(analysis),
                "team_started_at": team_started_at,
            }

        return run_stage("evidence_analysis", execute)

    def argument_drafter(state: TeamDraftNodeInput) -> TeamDraftUpdate:
        def execute() -> TeamDraftUpdate:
            question, evidence, formatted = _copy_team_inputs(
                question=state["question"],
                consolidated_evidence=state["consolidated_evidence"],
                research_findings=state["research_findings"],
            )
            if state["assigned_stance"] != configured_stance:
                raise TeamGraphValidationError("private team stance changed")
            analysis = validate_evidence_analysis(
                state["evidence_analysis"],
                team_name=validated_team,
                consolidated_evidence=evidence,
            )
            raw_draft = _invoke_model(
                lambda: model.draft_argument(
                    question=question,
                    team_name=validated_team,
                    stance=configured_stance,
                    research_findings=formatted.copy(),
                    evidence_analysis=copy.deepcopy(analysis),
                    opponent_argument=state["opponent_argument"],
                )
            )
            return {
                "draft_argument": validate_team_argument(
                    raw_draft, field_name="draft_argument"
                )
            }

        return run_stage("argument_draft", execute)

    def internal_critic_reviser(
        state: TeamReviewNodeInput,
    ) -> TeamRevisionUpdate:
        def execute() -> TeamRevisionUpdate:
            question, evidence, _ = _copy_team_inputs(
                question=state["question"],
                consolidated_evidence=state["consolidated_evidence"],
                research_findings=[
                    format_research_finding(item)
                    for item in state["consolidated_evidence"]
                ],
            )
            if state["assigned_stance"] != configured_stance:
                raise TeamGraphValidationError("private team stance changed")
            analysis = validate_evidence_analysis(
                state["evidence_analysis"],
                team_name=validated_team,
                consolidated_evidence=evidence,
            )
            draft = validate_team_argument(
                state["draft_argument"], field_name="draft_argument"
            )
            raw_revision = _invoke_model(
                lambda: model.review_and_revise(
                    question=question,
                    team_name=validated_team,
                    stance=configured_stance,
                    consolidated_evidence=copy.deepcopy(evidence),
                    evidence_analysis=copy.deepcopy(analysis),
                    draft_argument=draft,
                    opponent_argument=state["opponent_argument"],
                )
            )
            revision: TeamRevision = validate_team_revision(raw_revision)
            return {
                "internal_critique": copy.deepcopy(revision["critique"]),
                "final_argument": revision["revised_argument"],
            }

        return run_stage("review_revision", execute)

    def finalize(state: TeamFinalizeNodeInput) -> TeamAResult | TeamBResult:
        work_product: TeamWorkProduct = validate_team_work_product(
            {
                "team": validated_team,
                "assigned_stance": state["assigned_stance"],
                "evidence_analysis": copy.deepcopy(state["evidence_analysis"]),
                "draft_argument": state["draft_argument"],
                "internal_critique": copy.deepcopy(state["internal_critique"]),
                "final_argument": state["final_argument"],
            },
            consolidated_evidence=state["consolidated_evidence"],
            expected_team=validated_team,
            expected_stance=configured_stance,
        )
        duration = validate_trace_duration(
            _clock_value(clock) - state["team_started_at"]
        )
        if trace_recorder is not None:
            trace_recorder.record_total_success(
                team=validated_team,
                duration_seconds=duration,
            )
        if validated_team == "team_a":
            return {
                "team_a_argument": work_product["final_argument"],
                "team_a_work_product": copy.deepcopy(work_product),
            }
        return {
            "team_b_argument": work_product["final_argument"],
            "team_b_work_product": copy.deepcopy(work_product),
        }

    if validated_team == "team_a":
        builder = StateGraph(
            TeamAState,
            input_schema=TeamAInput,
            output_schema=TeamAResult,
        )
        analyst_input_schema = TeamAInput
    else:
        builder = StateGraph(
            TeamBState,
            input_schema=TeamBInput,
            output_schema=TeamBResult,
        )
        analyst_input_schema = TeamBInput

    builder.add_node(
        "evidence_analyst",
        evidence_analyst,
        input_schema=analyst_input_schema,
    )
    builder.add_node(
        "argument_drafter",
        argument_drafter,
        input_schema=TeamDraftNodeInput,
    )
    builder.add_node(
        "internal_critic_reviser",
        internal_critic_reviser,
        input_schema=TeamReviewNodeInput,
    )
    builder.add_node(
        "finalize",
        finalize,
        input_schema=TeamFinalizeNodeInput,
    )
    builder.add_edge(START, "evidence_analyst")
    builder.add_edge("evidence_analyst", "argument_drafter")
    builder.add_edge("argument_drafter", "internal_critic_reviser")
    builder.add_edge("internal_critic_reviser", "finalize")
    builder.add_edge("finalize", END)
    return cast(CompiledStateGraph, builder.compile())
