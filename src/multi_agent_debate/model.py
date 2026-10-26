"""Provider-independent model boundary for the debate graph."""

from typing import Protocol


class DebateModel(Protocol):
    """Operations required by the deterministic debate orchestration."""

    def research(self, *, question: str) -> list[str]:
        """Return a small set of findings relevant to the question."""
        ...

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        """Return one argument for a stance, optionally responding to an opponent."""
        ...

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: list[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        """Return a neutral synthesis of the completed exchange."""
        ...
