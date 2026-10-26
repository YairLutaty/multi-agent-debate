"""Typed state contracts for the bounded multi-round grounded graph."""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from multi_agent_debate.expanded_state import QuestionComplexity
from multi_agent_debate.research import (
    ConsolidatedEvidence,
    RawResearchFindings,
    merge_raw_findings,
)
from multi_agent_debate.termination import TerminationDecision


class DebateRound(TypedDict):
    round_number: int
    team_a_argument: str
    team_b_argument: str
    team_a_opponent_argument: str | None
    team_b_opponent_argument: str
    team_a_novelty: float | None
    team_b_novelty: float | None
    termination_decision: TerminationDecision
    team_a_duration_seconds: float
    team_b_duration_seconds: float
    termination_duration_seconds: float


class MultiRoundDebateInput(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str
    max_rounds: int
    novelty_threshold: float


class MultiRoundDebateState(MultiRoundDebateInput):
    question_complexity: NotRequired[QuestionComplexity]
    subquestions: NotRequired[list[str]]
    raw_research_findings: NotRequired[
        Annotated[RawResearchFindings, merge_raw_findings]
    ]
    consolidated_evidence: NotRequired[ConsolidatedEvidence]
    research_findings: NotRequired[list[str]]
    current_round_number: NotRequired[int]
    team_a_history: NotRequired[list[str]]
    team_b_history: NotRequired[list[str]]
    debate_rounds: NotRequired[list[DebateRound]]
    current_team_a_opponent_argument: NotRequired[str | None]
    current_team_b_opponent_argument: NotRequired[str]
    current_team_a_argument: NotRequired[str]
    current_team_b_argument: NotRequired[str]
    current_team_a_duration_seconds: NotRequired[float]
    current_team_b_duration_seconds: NotRequired[float]
    termination_decision: NotRequired[TerminationDecision]
    team_a_argument: NotRequired[str]
    team_b_argument: NotRequired[str]
    synthesis: NotRequired[str]
    synthesis_duration_seconds: NotRequired[float]


class MultiRoundDebateResult(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str
    max_rounds: int
    novelty_threshold: float
    question_complexity: QuestionComplexity
    subquestions: list[str]
    raw_research_findings: RawResearchFindings
    consolidated_evidence: ConsolidatedEvidence
    research_findings: list[str]
    current_round_number: int
    team_a_history: list[str]
    team_b_history: list[str]
    debate_rounds: list[DebateRound]
    termination_decision: TerminationDecision
    team_a_argument: str
    team_b_argument: str
    synthesis: str
    synthesis_duration_seconds: float


class InitializeDebateInput(TypedDict):
    question: str
    team_a_stance: str
    team_b_stance: str
    max_rounds: int
    novelty_threshold: float
    consolidated_evidence: ConsolidatedEvidence
    research_findings: list[str]


class InitializeDebateUpdate(TypedDict):
    current_round_number: int
    team_a_history: list[str]
    team_b_history: list[str]
    debate_rounds: list[DebateRound]


class TeamATurnInput(TypedDict):
    question: str
    team_a_stance: str
    research_findings: list[str]
    current_round_number: int
    team_b_history: list[str]


class TeamATurnUpdate(TypedDict):
    current_team_a_opponent_argument: str | None
    current_team_a_argument: str
    current_team_a_duration_seconds: float


class TeamBTurnInput(TypedDict):
    question: str
    team_b_stance: str
    research_findings: list[str]
    current_round_number: int
    current_team_a_argument: str


class TeamBTurnUpdate(TypedDict):
    current_team_b_opponent_argument: str
    current_team_b_argument: str
    current_team_b_duration_seconds: float


class AssessRoundInput(TypedDict):
    max_rounds: int
    novelty_threshold: float
    current_round_number: int
    team_a_history: list[str]
    team_b_history: list[str]
    debate_rounds: list[DebateRound]
    current_team_a_opponent_argument: str | None
    current_team_b_opponent_argument: str
    current_team_a_argument: str
    current_team_b_argument: str
    current_team_a_duration_seconds: float
    current_team_b_duration_seconds: float


class AssessRoundUpdate(TypedDict):
    current_round_number: int
    team_a_history: list[str]
    team_b_history: list[str]
    debate_rounds: list[DebateRound]
    termination_decision: TerminationDecision
    team_a_argument: str
    team_b_argument: str


class MultiRoundSynthesisInput(TypedDict):
    question: str
    research_findings: list[str]
    team_a_argument: str
    team_b_argument: str
    current_round_number: int


class MultiRoundSynthesisUpdate(TypedDict):
    synthesis: str
    synthesis_duration_seconds: float
