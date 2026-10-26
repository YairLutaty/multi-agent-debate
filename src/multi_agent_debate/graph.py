"""LangGraph orchestration for the deterministic Milestone 0 debate."""

from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_debate.model import DebateModel
from multi_agent_debate.state import (
    DebateInput,
    DebateResult,
    DebateState,
    NeutralSummarizerNodeInput,
    ResearchNodeInput,
    ResearchUpdate,
    SynthesisUpdate,
    TeamANodeInput,
    TeamAUpdate,
    TeamBNodeInput,
    TeamBUpdate,
)

TEAM_A_STANCE = "Support the proposition stated in the question."
TEAM_B_STANCE = "Oppose the proposition stated in the question."


def build_graph(model: DebateModel) -> CompiledStateGraph:
    """Build the fixed, single-round debate graph using an injected model."""

    def research(state: ResearchNodeInput) -> ResearchUpdate:
        return {"research_findings": model.research(question=state["question"])}

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
        DebateState,
        input_schema=DebateInput,
        output_schema=DebateResult,
    )
    builder.add_node("research", research, input_schema=ResearchNodeInput)
    builder.add_node("team_a", team_a, input_schema=TeamANodeInput)
    builder.add_node("team_b", team_b, input_schema=TeamBNodeInput)
    builder.add_node(
        "neutral_summarizer",
        neutral_summarizer,
        input_schema=NeutralSummarizerNodeInput,
    )

    builder.add_edge(START, "research")
    builder.add_edge("research", "team_a")
    builder.add_edge("team_a", "team_b")
    builder.add_edge("team_b", "neutral_summarizer")
    builder.add_edge("neutral_summarizer", END)

    return builder.compile()


def run_debate(question: str, model: DebateModel) -> DebateResult:
    """Run one debate after inserting the two fixed opposing stances."""

    initial_state: DebateInput = {
        "question": question,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
    }
    return cast(DebateResult, build_graph(model).invoke(initial_state))
