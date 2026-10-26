"""Typed state contracts for private team subgraphs and their parent graph."""

from typing import NotRequired, TypedDict

from multi_agent_debate.expanded_state import (
    ExpandedDebateInput,
    ExpandedDebateResult,
    ExpandedDebateState,
)
from multi_agent_debate.research import ConsolidatedEvidence
from multi_agent_debate.team_model import (
    EvidenceAnalysis,
    TeamCritique,
    TeamWorkProduct,
)


class TeamSharedInput(TypedDict):
    """Read-only data copied from the parent into either team subgraph."""

    question: str
    consolidated_evidence: ConsolidatedEvidence
    research_findings: list[str]


class TeamAInput(TeamSharedInput):
    """Team A has a stance but no opponent argument."""

    team_a_stance: str


class TeamBInput(TeamSharedInput):
    """Team B receives only Team A's final argument as opponent input."""

    team_b_stance: str
    team_a_argument: str


class TeamPrivateState(TypedDict, total=False):
    """Channels scoped to one compiled subgraph and absent from parent state."""

    assigned_stance: str
    opponent_argument: str | None
    evidence_analysis: EvidenceAnalysis
    draft_argument: str
    internal_critique: TeamCritique
    final_argument: str
    team_started_at: float


class TeamAState(TeamAInput, TeamPrivateState, total=False):
    """Complete internal Team A state, including its narrow exports."""

    team_a_argument: str
    team_a_work_product: TeamWorkProduct


class TeamBState(TeamBInput, TeamPrivateState, total=False):
    """Complete internal Team B state, including its narrow exports."""

    team_b_argument: str
    team_b_work_product: TeamWorkProduct


class TeamAResult(TypedDict):
    """The only Team A channels allowed back into parent state."""

    team_a_argument: str
    team_a_work_product: TeamWorkProduct


class TeamBResult(TypedDict):
    """The only Team B channels allowed back into parent state."""

    team_b_argument: str
    team_b_work_product: TeamWorkProduct


class TeamDraftNodeInput(TypedDict):
    question: str
    consolidated_evidence: ConsolidatedEvidence
    research_findings: list[str]
    assigned_stance: str
    opponent_argument: str | None
    evidence_analysis: EvidenceAnalysis


class TeamReviewNodeInput(TypedDict):
    question: str
    consolidated_evidence: ConsolidatedEvidence
    assigned_stance: str
    opponent_argument: str | None
    evidence_analysis: EvidenceAnalysis
    draft_argument: str


class TeamFinalizeNodeInput(TypedDict):
    consolidated_evidence: ConsolidatedEvidence
    assigned_stance: str
    evidence_analysis: EvidenceAnalysis
    draft_argument: str
    internal_critique: TeamCritique
    final_argument: str
    team_started_at: float


class TeamAnalysisUpdate(TypedDict):
    assigned_stance: str
    opponent_argument: str | None
    evidence_analysis: EvidenceAnalysis
    team_started_at: float


class TeamDraftUpdate(TypedDict):
    draft_argument: str


class TeamRevisionUpdate(TypedDict):
    internal_critique: TeamCritique
    final_argument: str


class HierarchicalDebateState(ExpandedDebateState):
    """Expanded parent state with only the two deliberate team exports."""

    team_a_work_product: NotRequired[TeamWorkProduct]
    team_b_work_product: NotRequired[TeamWorkProduct]


class HierarchicalDebateResult(ExpandedDebateResult):
    """Complete result returned by a successful hierarchical debate."""

    team_a_work_product: TeamWorkProduct
    team_b_work_product: TeamWorkProduct


HierarchicalDebateInput = ExpandedDebateInput
