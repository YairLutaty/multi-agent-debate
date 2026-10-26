"""Offline-verifiable LangGraph orchestration for Milestone 2.1A."""

from typing import Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from multi_agent_debate.expanded_state import (
    ConsolidationNodeInput,
    ConsolidationUpdate,
    DecompositionNodeInput,
    DecompositionResult,
    ExpandedDebateInput,
    ExpandedDebateResult,
    ExpandedDebateState,
    ResearchRoutingInput,
    ResearchWorkerInput,
    ResearchWorkerUpdate,
)
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.model import DebateModel
from multi_agent_debate.research import (
    ResearchProvider,
    consolidate_research_findings,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.state import (
    NeutralSummarizerNodeInput,
    SynthesisUpdate,
    TeamANodeInput,
    TeamAUpdate,
    TeamBNodeInput,
    TeamBUpdate,
)


class QuestionDecomposer(Protocol):
    """Provider-independent boundary for classifying and decomposing a question."""

    def decompose(self, *, question: str) -> DecompositionResult:
        """Return a simple classification or ordered complex subquestions."""
        ...


def validate_decomposition_result(value: object) -> DecompositionResult:
    """Validate and copy one decomposition result."""

    if not isinstance(value, dict):
        raise ValueError("decomposition result must be a dictionary")
    expected_keys = {"question_complexity", "subquestions"}
    if set(value) != expected_keys:
        raise ValueError("decomposition result must contain exactly two fields")

    question_complexity = value["question_complexity"]
    subquestions = value["subquestions"]
    if question_complexity not in ("simple", "complex"):
        raise ValueError("question_complexity must be simple or complex")
    if not isinstance(subquestions, list):
        raise ValueError("subquestions must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in subquestions):
        raise ValueError("subquestions must contain only nonempty strings")

    if question_complexity == "simple" and subquestions:
        raise ValueError("simple questions must not have subquestions")
    if question_complexity == "complex":
        if len(subquestions) < 2:
            raise ValueError("complex questions require at least two subquestions")
        if len(set(subquestions)) != len(subquestions):
            raise ValueError("complex subquestions must be distinct")

    return {
        "question_complexity": question_complexity,
        "subquestions": subquestions.copy(),
    }


def route_research(state: ResearchRoutingInput) -> list[Send]:
    """Create one narrow research-worker packet per required question."""

    if state["question_complexity"] == "simple":
        tasks = [(0, state["question"])]
    elif state["question_complexity"] == "complex":
        tasks = list(enumerate(state["subquestions"], start=1))
    else:
        raise ValueError("question_complexity must be simple or complex")

    return [
        Send(
            "research_worker",
            {
                "subquestion_index": index,
                "subquestion": subquestion,
            },
        )
        for index, subquestion in tasks
    ]


def build_expanded_graph(
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    model: DebateModel,
) -> CompiledStateGraph:
    """Build the expanded debate graph with injected offline-testable boundaries."""

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
            "consolidated_evidence": consolidated,
            "research_findings": [
                format_research_finding(finding) for finding in consolidated
            ],
        }

    def team_a(state: TeamANodeInput) -> TeamAUpdate:
        return {
            "team_a_argument": model.write_argument(
                question=state["question"],
                stance=state["team_a_stance"],
                research_findings=state["research_findings"],
                opponent_argument=None,
            )
        }

    def team_b(state: TeamBNodeInput) -> TeamBUpdate:
        return {
            "team_b_argument": model.write_argument(
                question=state["question"],
                stance=state["team_b_stance"],
                research_findings=state["research_findings"],
                opponent_argument=state["team_a_argument"],
            )
        }

    def neutral_summarizer(
        state: NeutralSummarizerNodeInput,
    ) -> SynthesisUpdate:
        return {
            "synthesis": model.synthesize_neutrally(
                question=state["question"],
                research_findings=state["research_findings"],
                team_a_argument=state["team_a_argument"],
                team_b_argument=state["team_b_argument"],
            )
        }

    builder = StateGraph(
        ExpandedDebateState,
        input_schema=ExpandedDebateInput,
        output_schema=ExpandedDebateResult,
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
    builder.add_node("team_a", team_a, input_schema=TeamANodeInput)
    builder.add_node("team_b", team_b, input_schema=TeamBNodeInput)
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


def run_expanded_debate(
    question: str,
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    model: DebateModel,
) -> ExpandedDebateResult:
    """Run the expanded graph after inserting the fixed opposing stances."""

    initial_state: ExpandedDebateInput = {
        "question": question,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
    }
    return cast(
        ExpandedDebateResult,
        build_expanded_graph(decomposer, research_provider, model).invoke(
            initial_state
        ),
    )
