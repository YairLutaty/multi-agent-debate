"""Deterministic end-to-end coverage for the Milestone 0 graph core."""

from typing import TypeAlias

from multi_agent_debate.graph import (
    TEAM_A_STANCE,
    TEAM_B_STANCE,
    run_debate,
)

RecordedCall: TypeAlias = tuple[object, ...]

QUESTION = "Should governments provide a universal basic income?"
RESEARCH_FINDINGS = [
    "Cash transfers can reduce short-term financial insecurity.",
    "Program cost depends on benefit size and tax design.",
    "Observed labor-market effects vary across trials.",
]
TEAM_A_ARGUMENT = "A universal basic income can strengthen the basic safety net."
TEAM_B_ARGUMENT = "A universal basic income may spend too broadly to be efficient."
SYNTHESIS = "The exchange contrasts broad income security with targeted efficiency."


class FakeDebateModel:
    """Fixed-response model that records every graph-to-model interaction."""

    def __init__(self) -> None:
        self.calls: list[RecordedCall] = []

    def research(self, *, question: str) -> list[str]:
        self.calls.append(("research", question))
        return RESEARCH_FINDINGS.copy()

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        self.calls.append(
            (
                "write_argument",
                question,
                stance,
                tuple(research_findings),
                opponent_argument,
            )
        )
        if opponent_argument is None:
            return TEAM_A_ARGUMENT
        return TEAM_B_ARGUMENT

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: list[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        self.calls.append(
            (
                "synthesize_neutrally",
                question,
                tuple(research_findings),
                team_a_argument,
                team_b_argument,
            )
        )
        return SYNTHESIS


def test_debate_graph_runs_end_to_end_with_deterministic_model() -> None:
    model = FakeDebateModel()

    result = run_debate(QUESTION, model)

    assert result == {
        "question": QUESTION,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
        "research_findings": RESEARCH_FINDINGS,
        "team_a_argument": TEAM_A_ARGUMENT,
        "team_b_argument": TEAM_B_ARGUMENT,
        "synthesis": SYNTHESIS,
    }

    expected_calls: list[RecordedCall] = [
        ("research", QUESTION),
        (
            "write_argument",
            QUESTION,
            TEAM_A_STANCE,
            tuple(RESEARCH_FINDINGS),
            None,
        ),
        (
            "write_argument",
            QUESTION,
            TEAM_B_STANCE,
            tuple(RESEARCH_FINDINGS),
            TEAM_A_ARGUMENT,
        ),
        (
            "synthesize_neutrally",
            QUESTION,
            tuple(RESEARCH_FINDINGS),
            TEAM_A_ARGUMENT,
            TEAM_B_ARGUMENT,
        ),
    ]
    assert model.calls == expected_calls

    argument_calls = [call for call in model.calls if call[0] == "write_argument"]
    assert len(argument_calls) == 2
