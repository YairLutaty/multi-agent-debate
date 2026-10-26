"""Additive parent graph with directly mounted hierarchical team subgraphs."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import cast

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
from multi_agent_debate.research import (
    ResearchFinding,
    ResearchProvider,
    consolidate_research_findings,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.state import NeutralSummarizerNodeInput, SynthesisUpdate
from multi_agent_debate.team_graph import build_team_subgraph
from multi_agent_debate.team_model import (
    NeutralSynthesisModel,
    TeamGraphValidationError,
    TeamModel,
    TeamWorkProduct,
    validate_team_argument,
    validate_team_work_product,
)
from multi_agent_debate.team_state import (
    HierarchicalDebateInput,
    HierarchicalDebateResult,
    HierarchicalDebateState,
)
from multi_agent_debate.team_trace import TeamTraceRecorder

HIERARCHICAL_RECURSION_LIMIT = 32


def build_hierarchical_graph(
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    team_model: TeamModel,
    synthesis_model: NeutralSynthesisModel,
    *,
    team_trace_recorder: TeamTraceRecorder | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CompiledStateGraph:
    """Build the M2.2 parent graph without modifying the frozen M2.1 graph."""

    team_a_subgraph = build_team_subgraph(
        team_name="team_a",
        stance=TEAM_A_STANCE,
        model=team_model,
        trace_recorder=team_trace_recorder,
        clock=clock,
    )
    team_b_subgraph = build_team_subgraph(
        team_name="team_b",
        stance=TEAM_B_STANCE,
        model=team_model,
        trace_recorder=team_trace_recorder,
        clock=clock,
    )

    def decompose(state: DecompositionNodeInput) -> DecompositionResult:
        return validate_decomposition_result(
            decomposer.decompose(question=state["question"])
        )

    def research_worker(state: ResearchWorkerInput) -> ResearchWorkerUpdate:
        findings = research_provider.research(
            subquestion_index=state["subquestion_index"],
            subquestion=state["subquestion"],
        )
        validated = [
            validate_research_finding(
                finding,
                expected_subquestion_index=state["subquestion_index"],
                expected_subquestion=state["subquestion"],
            )
            for finding in findings
        ]
        return {"raw_research_findings": validated}

    def consolidate(state: ConsolidationNodeInput) -> ConsolidationUpdate:
        consolidated = consolidate_research_findings(
            question=state["question"],
            question_complexity=state["question_complexity"],
            subquestions=state["subquestions"],
            raw_findings=state["raw_research_findings"],
        )
        return {
            "consolidated_evidence": [item.copy() for item in consolidated],
            "research_findings": [
                format_research_finding(item) for item in consolidated
            ],
        }

    def neutral_summarizer(
        state: NeutralSummarizerNodeInput,
    ) -> SynthesisUpdate:
        synthesis = synthesis_model.synthesize_neutrally(
            question=state["question"],
            research_findings=state["research_findings"].copy(),
            team_a_argument=state["team_a_argument"],
            team_b_argument=state["team_b_argument"],
        )
        if not isinstance(synthesis, str) or not synthesis.strip():
            raise TeamGraphValidationError("neutral synthesis must be nonblank")
        return {"synthesis": synthesis.strip()}

    builder = StateGraph(
        HierarchicalDebateState,
        input_schema=HierarchicalDebateInput,
        output_schema=HierarchicalDebateResult,
    )
    builder.add_node("decompose", decompose, input_schema=DecompositionNodeInput)
    builder.add_node(
        "research_worker",
        research_worker,
        input_schema=ResearchWorkerInput,
    )
    builder.add_node(
        "consolidate",
        consolidate,
        input_schema=ConsolidationNodeInput,
    )
    # Compiled subgraphs are mounted directly; no wrapper invokes either graph.
    builder.add_node("team_a", team_a_subgraph)
    builder.add_node("team_b", team_b_subgraph)
    builder.add_node(
        "neutral_summarizer",
        neutral_summarizer,
        input_schema=NeutralSummarizerNodeInput,
    )

    builder.add_edge(START, "decompose")
    builder.add_conditional_edges("decompose", route_research)
    builder.add_edge("research_worker", "consolidate")
    builder.add_edge("consolidate", "team_a")
    builder.add_edge("team_a", "team_b")
    builder.add_edge("team_b", "neutral_summarizer")
    builder.add_edge("neutral_summarizer", END)
    return builder.compile()


def _copy_hierarchical_result(value: object) -> HierarchicalDebateResult:
    if not isinstance(value, dict):
        raise TeamGraphValidationError("hierarchical graph returned invalid state")
    try:
        decomposition = validate_decomposition_result(
            {
                "question_complexity": value["question_complexity"],
                "subquestions": value["subquestions"],
            }
        )
        raw_findings = [
            validate_research_finding(item)
            for item in value["raw_research_findings"]
        ]
        consolidated = [
            validate_research_finding(item)
            for item in value["consolidated_evidence"]
        ]
        formatted = value["research_findings"]
        if not isinstance(formatted, list) or formatted != [
            format_research_finding(item) for item in consolidated
        ]:
            raise TeamGraphValidationError(
                "hierarchical result has inconsistent formatted research"
            )
        team_a_argument = validate_team_argument(
            value["team_a_argument"], field_name="team_a_argument"
        )
        team_b_argument = validate_team_argument(
            value["team_b_argument"], field_name="team_b_argument"
        )
        team_a_product: TeamWorkProduct = validate_team_work_product(
            value["team_a_work_product"],
            consolidated_evidence=consolidated,
            expected_team="team_a",
            expected_stance=TEAM_A_STANCE,
        )
        team_b_product: TeamWorkProduct = validate_team_work_product(
            value["team_b_work_product"],
            consolidated_evidence=consolidated,
            expected_team="team_b",
            expected_stance=TEAM_B_STANCE,
        )
        synthesis = value["synthesis"]
        if not isinstance(synthesis, str) or not synthesis.strip():
            raise TeamGraphValidationError("hierarchical result has blank synthesis")
        question = value["question"]
        team_a_stance = value["team_a_stance"]
        team_b_stance = value["team_b_stance"]
        if not all(
            isinstance(item, str)
            for item in (question, team_a_stance, team_b_stance)
        ):
            raise TeamGraphValidationError(
                "hierarchical result has invalid input fields"
            )
    except KeyError:
        raise TeamGraphValidationError(
            "hierarchical graph result is incomplete"
        ) from None
    except (TypeError, ValueError):
        raise TeamGraphValidationError(
            "hierarchical graph result is malformed"
        ) from None

    return {
        "question": question,
        "team_a_stance": team_a_stance,
        "team_b_stance": team_b_stance,
        "question_complexity": decomposition["question_complexity"],
        "subquestions": decomposition["subquestions"].copy(),
        "raw_research_findings": [item.copy() for item in raw_findings],
        "consolidated_evidence": [item.copy() for item in consolidated],
        "research_findings": formatted.copy(),
        "team_a_argument": team_a_argument,
        "team_b_argument": team_b_argument,
        "synthesis": synthesis.strip(),
        "team_a_work_product": copy.deepcopy(team_a_product),
        "team_b_work_product": copy.deepcopy(team_b_product),
    }


def run_hierarchical_debate(
    question: str,
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    team_model: TeamModel,
    synthesis_model: NeutralSynthesisModel,
    *,
    team_trace_recorder: TeamTraceRecorder | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> HierarchicalDebateResult:
    """Insert fixed stances and execute with an explicit recursion bound."""

    initial_state: HierarchicalDebateInput = {
        "question": question,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
    }
    result = build_hierarchical_graph(
        decomposer,
        research_provider,
        team_model,
        synthesis_model,
        team_trace_recorder=team_trace_recorder,
        clock=clock,
    ).invoke(
        initial_state,
        config={"recursion_limit": HIERARCHICAL_RECURSION_LIMIT},
    )
    return cast(HierarchicalDebateResult, _copy_hierarchical_result(result))
