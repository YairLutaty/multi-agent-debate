"""Additive grounded LangGraph with a bounded deterministic debate loop."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from math import isfinite
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_debate.expanded_graph import (
    QuestionDecomposer,
    route_research,
    validate_decomposition_result,
)
from multi_agent_debate.expanded_state import (
    ConsolidationNodeInput,
    ConsolidationUpdate,
    DecompositionNodeInput,
    DecompositionResult,
    ResearchWorkerInput,
    ResearchWorkerUpdate,
)
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.model import DebateModel
from multi_agent_debate.multi_round_state import (
    AssessRoundInput,
    AssessRoundUpdate,
    DebateRound,
    InitializeDebateInput,
    InitializeDebateUpdate,
    MultiRoundDebateInput,
    MultiRoundDebateResult,
    MultiRoundDebateState,
    MultiRoundSynthesisInput,
    MultiRoundSynthesisUpdate,
    TeamATurnInput,
    TeamATurnUpdate,
    TeamBTurnInput,
    TeamBTurnUpdate,
)
from multi_agent_debate.multi_round_trace import (
    MonotonicClock,
    MultiRoundStage,
    MultiRoundTraceRecorder,
    TraceTeam,
    bounded_multi_round_error_category,
    clock_value,
    elapsed_seconds,
)
from multi_agent_debate.research import (
    ResearchProvider,
    consolidate_research_findings,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.termination import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NOVELTY_THRESHOLD,
    MultiRoundError,
    MultiRoundModelInvocationError,
    MultiRoundModelResponseError,
    MultiRoundValidationError,
    TerminationDecision,
    decide_termination,
    validate_argument,
    validate_max_rounds,
    validate_novelty_threshold,
    validate_termination_decision,
)

MULTI_ROUND_RECURSION_LIMIT = 64


def build_multi_round_graph(
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    model: DebateModel,
    *,
    trace_recorder: MultiRoundTraceRecorder | None = None,
    clock: MonotonicClock = time.perf_counter,
) -> CompiledStateGraph:
    """Build the final graph while leaving every historical graph frozen."""

    recorder = trace_recorder if trace_recorder is not None else MultiRoundTraceRecorder()

    def decompose(state: DecompositionNodeInput) -> DecompositionResult:
        return validate_decomposition_result(
            decomposer.decompose(question=state["question"])
        )

    def research_worker(state: ResearchWorkerInput) -> ResearchWorkerUpdate:
        findings = research_provider.research(
            subquestion_index=state["subquestion_index"],
            subquestion=state["subquestion"],
        )
        return {
            "raw_research_findings": [
                validate_research_finding(
                    finding,
                    expected_subquestion_index=state["subquestion_index"],
                    expected_subquestion=state["subquestion"],
                )
                for finding in findings
            ]
        }

    def consolidate(state: ConsolidationNodeInput) -> ConsolidationUpdate:
        consolidated = consolidate_research_findings(
            question=state["question"],
            question_complexity=state["question_complexity"],
            subquestions=state["subquestions"],
            raw_findings=state["raw_research_findings"],
        )
        return {
            "consolidated_evidence": [finding.copy() for finding in consolidated],
            "research_findings": [
                format_research_finding(finding) for finding in consolidated
            ],
        }

    def initialize_debate(state: InitializeDebateInput) -> InitializeDebateUpdate:
        if state["team_a_stance"] != TEAM_A_STANCE:
            raise MultiRoundValidationError("Team A stance is invalid")
        if state["team_b_stance"] != TEAM_B_STANCE:
            raise MultiRoundValidationError("Team B stance is invalid")
        validate_max_rounds(state["max_rounds"])
        validate_novelty_threshold(state["novelty_threshold"])
        if not state["consolidated_evidence"]:
            raise MultiRoundValidationError("grounded debate requires evidence")
        expected_formatted = [
            format_research_finding(validate_research_finding(finding))
            for finding in state["consolidated_evidence"]
        ]
        if state["research_findings"] != expected_formatted:
            raise MultiRoundValidationError("formatted evidence is inconsistent")
        return {
            "current_round_number": 1,
            "team_a_history": [],
            "team_b_history": [],
            "debate_rounds": [],
        }

    def team_a_turn(state: TeamATurnInput) -> TeamATurnUpdate:
        round_number = _validate_round_number(state["current_round_number"])
        if type(state["team_b_history"]) is not list or len(
            state["team_b_history"]
        ) != round_number - 1:
            raise MultiRoundValidationError(
                "Team B history does not match the current round"
            )
        opponent = None if round_number == 1 else state["team_b_history"][-1]
        if opponent is not None:
            opponent = validate_argument(
                opponent, field_name="Team A opponent argument"
            )
        started = clock_value(clock)
        try:
            raw_argument = model.write_argument(
                question=state["question"],
                stance=state["team_a_stance"],
                research_findings=state["research_findings"].copy(),
                opponent_argument=opponent,
            )
            argument = _validate_model_text(raw_argument, "Team A argument")
        except MultiRoundError as error:
            _record_error(
                recorder,
                clock,
                started,
                round_number,
                "team_a",
                "team_a_argument",
                error,
            )
            raise
        except Exception:
            wrapped = MultiRoundModelInvocationError("Debate model invocation failed.")
            _record_error(
                recorder,
                clock,
                started,
                round_number,
                "team_a",
                "team_a_argument",
                wrapped,
            )
            raise wrapped from None
        duration = elapsed_seconds(clock, started)
        _record_success(
            recorder,
            round_number,
            "team_a",
            "team_a_argument",
            duration,
        )
        return {
            "current_team_a_opponent_argument": opponent,
            "current_team_a_argument": argument,
            "current_team_a_duration_seconds": duration,
        }

    def team_b_turn(state: TeamBTurnInput) -> TeamBTurnUpdate:
        round_number = _validate_round_number(state["current_round_number"])
        opponent = validate_argument(
            state["current_team_a_argument"],
            field_name="Team B opponent argument",
        )
        started = clock_value(clock)
        try:
            raw_argument = model.write_argument(
                question=state["question"],
                stance=state["team_b_stance"],
                research_findings=state["research_findings"].copy(),
                opponent_argument=opponent,
            )
            argument = _validate_model_text(raw_argument, "Team B argument")
        except MultiRoundError as error:
            _record_error(
                recorder,
                clock,
                started,
                round_number,
                "team_b",
                "team_b_argument",
                error,
            )
            raise
        except Exception:
            wrapped = MultiRoundModelInvocationError("Debate model invocation failed.")
            _record_error(
                recorder,
                clock,
                started,
                round_number,
                "team_b",
                "team_b_argument",
                wrapped,
            )
            raise wrapped from None
        duration = elapsed_seconds(clock, started)
        _record_success(
            recorder,
            round_number,
            "team_b",
            "team_b_argument",
            duration,
        )
        return {
            "current_team_b_opponent_argument": opponent,
            "current_team_b_argument": argument,
            "current_team_b_duration_seconds": duration,
        }

    def assess_round(state: AssessRoundInput) -> AssessRoundUpdate:
        round_number = _validate_round_number(state["current_round_number"])
        if (
            type(state["team_a_history"]) is not list
            or type(state["team_b_history"]) is not list
            or type(state["debate_rounds"]) is not list
            or len(state["team_a_history"]) != round_number - 1
            or len(state["team_b_history"]) != round_number - 1
            or len(state["debate_rounds"]) != round_number - 1
        ):
            raise MultiRoundValidationError(
                "completed histories do not match the current round"
            )
        started = clock_value(clock)
        team_a_history = [
            *state["team_a_history"],
            state["current_team_a_argument"],
        ]
        team_b_history = [
            *state["team_b_history"],
            state["current_team_b_argument"],
        ]
        try:
            decision = decide_termination(
                team_a_history=team_a_history,
                team_b_history=team_b_history,
                max_rounds=state["max_rounds"],
                novelty_threshold=state["novelty_threshold"],
            )
        except MultiRoundError as error:
            _record_error(
                recorder,
                clock,
                started,
                round_number,
                "system",
                "termination",
                error,
            )
            raise
        duration = elapsed_seconds(clock, started)
        _record_success(
            recorder,
            round_number,
            "system",
            "termination",
            duration,
        )
        round_record: DebateRound = {
            "round_number": round_number,
            "team_a_argument": state["current_team_a_argument"],
            "team_b_argument": state["current_team_b_argument"],
            "team_a_opponent_argument": state[
                "current_team_a_opponent_argument"
            ],
            "team_b_opponent_argument": state[
                "current_team_b_opponent_argument"
            ],
            "team_a_novelty": decision["team_a_novelty"],
            "team_b_novelty": decision["team_b_novelty"],
            "termination_decision": copy.deepcopy(decision),
            "team_a_duration_seconds": state["current_team_a_duration_seconds"],
            "team_b_duration_seconds": state["current_team_b_duration_seconds"],
            "termination_duration_seconds": duration,
        }
        return {
            "current_round_number": (
                round_number if decision["stop"] else round_number + 1
            ),
            "team_a_history": team_a_history.copy(),
            "team_b_history": team_b_history.copy(),
            "debate_rounds": [
                *(copy.deepcopy(state["debate_rounds"])),
                copy.deepcopy(round_record),
            ],
            "termination_decision": copy.deepcopy(decision),
            "team_a_argument": state["current_team_a_argument"],
            "team_b_argument": state["current_team_b_argument"],
        }

    def route_after_assessment(
        state: MultiRoundDebateState,
    ) -> Literal["continue", "stop"]:
        return "stop" if state["termination_decision"]["stop"] else "continue"

    def neutral_synthesis(
        state: MultiRoundSynthesisInput,
    ) -> MultiRoundSynthesisUpdate:
        started = clock_value(clock)
        try:
            raw_synthesis = model.synthesize_neutrally(
                question=state["question"],
                research_findings=state["research_findings"].copy(),
                team_a_argument=state["team_a_argument"],
                team_b_argument=state["team_b_argument"],
            )
            synthesis = _validate_model_text(raw_synthesis, "neutral synthesis")
        except MultiRoundError as error:
            _record_error(
                recorder,
                clock,
                started,
                state["current_round_number"],
                "system",
                "neutral_synthesis",
                error,
            )
            raise
        except Exception:
            wrapped = MultiRoundModelInvocationError("Debate model invocation failed.")
            _record_error(
                recorder,
                clock,
                started,
                state["current_round_number"],
                "system",
                "neutral_synthesis",
                wrapped,
            )
            raise wrapped from None
        duration = elapsed_seconds(clock, started)
        _record_success(
            recorder,
            state["current_round_number"],
            "system",
            "neutral_synthesis",
            duration,
        )
        return {"synthesis": synthesis, "synthesis_duration_seconds": duration}

    builder = StateGraph(
        MultiRoundDebateState,
        input_schema=MultiRoundDebateInput,
        output_schema=MultiRoundDebateResult,
    )
    builder.add_node("decompose", decompose, input_schema=DecompositionNodeInput)
    builder.add_node(
        "research_worker", research_worker, input_schema=ResearchWorkerInput
    )
    builder.add_node("consolidate", consolidate, input_schema=ConsolidationNodeInput)
    builder.add_node(
        "initialize_debate",
        initialize_debate,
        input_schema=InitializeDebateInput,
    )
    builder.add_node("team_a_turn", team_a_turn, input_schema=TeamATurnInput)
    builder.add_node("team_b_turn", team_b_turn, input_schema=TeamBTurnInput)
    builder.add_node("assess_round", assess_round, input_schema=AssessRoundInput)
    builder.add_node(
        "neutral_synthesis",
        neutral_synthesis,
        input_schema=MultiRoundSynthesisInput,
    )
    builder.add_edge(START, "decompose")
    builder.add_conditional_edges("decompose", route_research)
    builder.add_edge("research_worker", "consolidate")
    builder.add_edge("consolidate", "initialize_debate")
    builder.add_edge("initialize_debate", "team_a_turn")
    builder.add_edge("team_a_turn", "team_b_turn")
    builder.add_edge("team_b_turn", "assess_round")
    builder.add_conditional_edges(
        "assess_round",
        route_after_assessment,
        {"continue": "team_a_turn", "stop": "neutral_synthesis"},
    )
    builder.add_edge("neutral_synthesis", END)
    return builder.compile()


def run_multi_round_debate(
    question: str,
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    model: DebateModel,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    trace_recorder: MultiRoundTraceRecorder | None = None,
    clock: MonotonicClock = time.perf_counter,
) -> MultiRoundDebateResult:
    """Insert fixed stances and execute with policy and engine bounds."""

    if not isinstance(question, str) or not question.strip():
        raise MultiRoundValidationError("question must be a nonempty string")
    initial_state: MultiRoundDebateInput = {
        "question": question,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
        "max_rounds": validate_max_rounds(max_rounds),
        "novelty_threshold": validate_novelty_threshold(novelty_threshold),
    }
    result = build_multi_round_graph(
        decomposer,
        research_provider,
        model,
        trace_recorder=trace_recorder,
        clock=clock,
    ).invoke(
        initial_state,
        config={"recursion_limit": MULTI_ROUND_RECURSION_LIMIT},
    )
    return validate_multi_round_result(result)


def validate_multi_round_result(value: object) -> MultiRoundDebateResult:
    """Validate a completed graph result and return nested defensive copies."""

    if not isinstance(value, dict):
        raise MultiRoundValidationError("multi-round graph returned invalid state")
    expected_keys = set(MultiRoundDebateResult.__required_keys__)
    if set(value) != expected_keys:
        raise MultiRoundValidationError("multi-round result has invalid fields")
    try:
        question = value["question"]
        if not isinstance(question, str) or not question.strip():
            raise MultiRoundValidationError("result question is invalid")
        if value["team_a_stance"] != TEAM_A_STANCE:
            raise MultiRoundValidationError("result Team A stance is invalid")
        if value["team_b_stance"] != TEAM_B_STANCE:
            raise MultiRoundValidationError("result Team B stance is invalid")
        max_rounds = validate_max_rounds(value["max_rounds"])
        threshold = validate_novelty_threshold(value["novelty_threshold"])
        decomposition = validate_decomposition_result(
            {
                "question_complexity": value["question_complexity"],
                "subquestions": value["subquestions"],
            }
        )
        raw = [validate_research_finding(item) for item in value["raw_research_findings"]]
        consolidated = [
            validate_research_finding(item) for item in value["consolidated_evidence"]
        ]
        expected_consolidated = consolidate_research_findings(
            question=question,
            question_complexity=decomposition["question_complexity"],
            subquestions=decomposition["subquestions"],
            raw_findings=raw,
        )
        if consolidated != expected_consolidated:
            raise MultiRoundValidationError("result consolidation is inconsistent")
        formatted = value["research_findings"]
        if type(formatted) is not list or formatted != [
            format_research_finding(item) for item in consolidated
        ]:
            raise MultiRoundValidationError("result formatted evidence is inconsistent")
        team_a_history = _validate_history(value["team_a_history"], "Team A")
        team_b_history = _validate_history(value["team_b_history"], "Team B")
        raw_rounds = value["debate_rounds"]
        if type(raw_rounds) is not list or not raw_rounds:
            raise MultiRoundValidationError("result debate rounds are invalid")
        if len(raw_rounds) != len(team_a_history) or len(raw_rounds) != len(team_b_history):
            raise MultiRoundValidationError("round records and histories disagree")
        if len(raw_rounds) > max_rounds:
            raise MultiRoundValidationError("result exceeds maximum rounds")
        rounds: list[DebateRound] = []
        prior_decision: TerminationDecision | None = None
        for index, raw_round in enumerate(raw_rounds, start=1):
            if prior_decision is not None and prior_decision["stop"]:
                raise MultiRoundValidationError("round appears after termination")
            round_record = _validate_round(
                raw_round,
                round_number=index,
                team_a_history=team_a_history[:index],
                team_b_history=team_b_history[:index],
                max_rounds=max_rounds,
                novelty_threshold=threshold,
            )
            expected_a_opponent = None if index == 1 else team_b_history[index - 2]
            if round_record["team_a_opponent_argument"] != expected_a_opponent:
                raise MultiRoundValidationError("Team A opponent routing is invalid")
            if round_record["team_b_opponent_argument"] != round_record["team_a_argument"]:
                raise MultiRoundValidationError("Team B opponent routing is invalid")
            if round_record["team_a_argument"] != team_a_history[index - 1]:
                raise MultiRoundValidationError("Team A history is inconsistent")
            if round_record["team_b_argument"] != team_b_history[index - 1]:
                raise MultiRoundValidationError("Team B history is inconsistent")
            prior_decision = round_record["termination_decision"]
            rounds.append(copy.deepcopy(round_record))
        assert prior_decision is not None
        if not prior_decision["stop"]:
            raise MultiRoundValidationError("completed result did not terminate")
        final_decision = validate_termination_decision(value["termination_decision"])
        if final_decision != prior_decision:
            raise MultiRoundValidationError("final termination decision is inconsistent")
        current_round = value["current_round_number"]
        if type(current_round) is not int or current_round != len(rounds):
            raise MultiRoundValidationError("current round number is inconsistent")
        team_a_argument = validate_argument(
            value["team_a_argument"], field_name="Team A final argument"
        )
        team_b_argument = validate_argument(
            value["team_b_argument"], field_name="Team B final argument"
        )
        if team_a_argument != team_a_history[-1] or team_b_argument != team_b_history[-1]:
            raise MultiRoundValidationError("final arguments do not match final round")
        synthesis = validate_argument(value["synthesis"], field_name="synthesis")
        synthesis_duration = _finite_nonnegative(
            value["synthesis_duration_seconds"], "synthesis duration"
        )
    except KeyError:
        raise MultiRoundValidationError("multi-round result is incomplete") from None
    except (TypeError, ValueError):
        raise MultiRoundValidationError("multi-round result is malformed") from None

    return {
        "question": question,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
        "max_rounds": max_rounds,
        "novelty_threshold": threshold,
        "question_complexity": decomposition["question_complexity"],
        "subquestions": decomposition["subquestions"].copy(),
        "raw_research_findings": [item.copy() for item in raw],
        "consolidated_evidence": [item.copy() for item in consolidated],
        "research_findings": formatted.copy(),
        "current_round_number": current_round,
        "team_a_history": team_a_history.copy(),
        "team_b_history": team_b_history.copy(),
        "debate_rounds": copy.deepcopy(rounds),
        "termination_decision": copy.deepcopy(final_decision),
        "team_a_argument": team_a_argument,
        "team_b_argument": team_b_argument,
        "synthesis": synthesis,
        "synthesis_duration_seconds": synthesis_duration,
    }


def _validate_round(
    value: object,
    *,
    round_number: int,
    team_a_history: list[str],
    team_b_history: list[str],
    max_rounds: int,
    novelty_threshold: float,
) -> DebateRound:
    if not isinstance(value, dict) or set(value) != set(DebateRound.__required_keys__):
        raise MultiRoundValidationError("debate round has invalid fields")
    if type(value["round_number"]) is not int or value["round_number"] != round_number:
        raise MultiRoundValidationError("debate round number is invalid")
    team_a_argument = validate_argument(value["team_a_argument"], field_name="Team A argument")
    team_b_argument = validate_argument(value["team_b_argument"], field_name="Team B argument")
    a_opponent = value["team_a_opponent_argument"]
    if a_opponent is not None:
        a_opponent = validate_argument(a_opponent, field_name="Team A opponent argument")
    b_opponent = validate_argument(
        value["team_b_opponent_argument"],
        field_name="Team B opponent argument",
    )
    expected = decide_termination(
        team_a_history=team_a_history,
        team_b_history=team_b_history,
        max_rounds=max_rounds,
        novelty_threshold=novelty_threshold,
    )
    decision = validate_termination_decision(value["termination_decision"])
    if decision != expected:
        raise MultiRoundValidationError("round termination decision is inconsistent")
    raw_a_novelty = _validate_optional_novelty(value["team_a_novelty"])
    raw_b_novelty = _validate_optional_novelty(value["team_b_novelty"])
    if raw_a_novelty != decision["team_a_novelty"] or raw_b_novelty != decision["team_b_novelty"]:
        raise MultiRoundValidationError("round novelty values are inconsistent")
    return {
        "round_number": round_number,
        "team_a_argument": team_a_argument,
        "team_b_argument": team_b_argument,
        "team_a_opponent_argument": a_opponent,
        "team_b_opponent_argument": b_opponent,
        "team_a_novelty": decision["team_a_novelty"],
        "team_b_novelty": decision["team_b_novelty"],
        "termination_decision": copy.deepcopy(decision),
        "team_a_duration_seconds": _finite_nonnegative(
            value["team_a_duration_seconds"], "Team A duration"
        ),
        "team_b_duration_seconds": _finite_nonnegative(
            value["team_b_duration_seconds"], "Team B duration"
        ),
        "termination_duration_seconds": _finite_nonnegative(
            value["termination_duration_seconds"], "termination duration"
        ),
    }


def _validate_model_text(value: object, context: str) -> str:
    try:
        return validate_argument(value, field_name=context)
    except MultiRoundValidationError:
        raise MultiRoundModelResponseError(f"{context} is invalid.") from None


def _validate_history(value: object, team: str) -> list[str]:
    if type(value) is not list:
        raise MultiRoundValidationError(f"{team} history must be a list")
    return [
        validate_argument(item, field_name=f"{team} history argument")
        for item in value
    ]


def _finite_nonnegative(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise MultiRoundValidationError(f"{context} must be numeric")
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise MultiRoundValidationError(f"{context} must be finite and nonnegative")
    return resolved


def _validate_round_number(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 3:
        raise MultiRoundValidationError("current round number is invalid")
    return value


def _validate_optional_novelty(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise MultiRoundValidationError("round novelty must be numeric")
    resolved = float(value)
    if not isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise MultiRoundValidationError("round novelty must be within [0, 1]")
    return resolved


def _record_success(
    recorder: MultiRoundTraceRecorder,
    round_number: int,
    team: TraceTeam,
    stage: MultiRoundStage,
    duration: float,
) -> None:
    recorder.record(
        {
            "round_number": round_number,
            "team": team,
            "stage": stage,
            "duration_seconds": duration,
            "status": "success",
            "error_category": None,
        }
    )


def _record_error(
    recorder: MultiRoundTraceRecorder,
    clock: MonotonicClock,
    started: float,
    round_number: int,
    team: TraceTeam,
    stage: MultiRoundStage,
    error: BaseException,
) -> None:
    recorder.record(
        {
            "round_number": round_number,
            "team": team,
            "stage": stage,
            "duration_seconds": elapsed_seconds(clock, started),
            "status": "error",
            "error_category": bounded_multi_round_error_category(error),
        }
    )
