"""Typed state contracts for the Milestone 2.1A expanded graph."""

from typing import Annotated, Literal, NotRequired, TypeAlias, TypedDict

from multi_agent_debate.research import (
    ConsolidatedEvidence,
    RawResearchFindings,
    merge_raw_findings,
)

QuestionComplexity: TypeAlias = Literal["simple", "complex"]


class ExpandedDebateInput(TypedDict):
    """Required input supplied before expanded graph execution."""

    question: str
    team_a_stance: str
    team_b_stance: str


class DecompositionResult(TypedDict):
    """Validated classification and ordered decomposition output."""

    question_complexity: QuestionComplexity
    subquestions: list[str]


class ExpandedDebateState(ExpandedDebateInput):
    """Partially populated state while the expanded graph executes."""

    question_complexity: NotRequired[QuestionComplexity]
    subquestions: NotRequired[list[str]]
    raw_research_findings: NotRequired[
        Annotated[RawResearchFindings, merge_raw_findings]
    ]
    consolidated_evidence: NotRequired[ConsolidatedEvidence]
    research_findings: NotRequired[list[str]]
    team_a_argument: NotRequired[str]
    team_b_argument: NotRequired[str]
    synthesis: NotRequired[str]


class ExpandedDebateResult(TypedDict):
    """Complete state returned by a successful expanded debate."""

    question: str
    team_a_stance: str
    team_b_stance: str
    question_complexity: QuestionComplexity
    subquestions: list[str]
    raw_research_findings: RawResearchFindings
    consolidated_evidence: ConsolidatedEvidence
    research_findings: list[str]
    team_a_argument: str
    team_b_argument: str
    synthesis: str


class DecompositionNodeInput(TypedDict):
    question: str


class ResearchRoutingInput(TypedDict):
    question: str
    question_complexity: QuestionComplexity
    subquestions: list[str]


class ResearchWorkerInput(TypedDict):
    subquestion_index: int
    subquestion: str


class ResearchWorkerUpdate(TypedDict):
    raw_research_findings: RawResearchFindings


class ConsolidationNodeInput(TypedDict):
    question: str
    question_complexity: QuestionComplexity
    subquestions: list[str]
    raw_research_findings: RawResearchFindings


class ConsolidationUpdate(TypedDict):
    consolidated_evidence: ConsolidatedEvidence
    research_findings: list[str]
