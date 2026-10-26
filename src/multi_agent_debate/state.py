"""Typed state contracts for the Milestone 0 debate graph."""

from typing import NotRequired, TypedDict


class DebateInput(TypedDict):
    """State fields that must exist before graph execution begins."""

    question: str
    team_a_stance: str
    team_b_stance: str


class DebateState(DebateInput):
    """Internal graph state while debate artifacts are produced."""

    research_findings: NotRequired[list[str]]
    team_a_argument: NotRequired[str]
    team_b_argument: NotRequired[str]
    synthesis: NotRequired[str]


class DebateResult(TypedDict):
    """Complete state returned after successful graph execution."""

    question: str
    team_a_stance: str
    team_b_stance: str
    research_findings: list[str]
    team_a_argument: str
    team_b_argument: str
    synthesis: str


class ResearchNodeInput(TypedDict):
    question: str


class TeamANodeInput(TypedDict):
    question: str
    team_a_stance: str
    research_findings: list[str]


class TeamBNodeInput(TypedDict):
    question: str
    team_b_stance: str
    research_findings: list[str]
    team_a_argument: str


class NeutralSummarizerNodeInput(TypedDict):
    question: str
    research_findings: list[str]
    team_a_argument: str
    team_b_argument: str


class ResearchUpdate(TypedDict):
    research_findings: list[str]


class TeamAUpdate(TypedDict):
    team_a_argument: str


class TeamBUpdate(TypedDict):
    team_b_argument: str


class SynthesisUpdate(TypedDict):
    synthesis: str
