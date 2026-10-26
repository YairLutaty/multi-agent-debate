"""Provider-neutral contracts and validation for hierarchical debate teams."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias, TypedDict, cast

from multi_agent_debate.research import ResearchFinding, validate_research_finding


class HierarchicalTeamError(RuntimeError):
    """Base error for deterministic hierarchical-team behavior."""


class TeamModelError(HierarchicalTeamError):
    """Base error for team-model invocation or response failures."""


class TeamModelInvocationError(TeamModelError):
    """A team-model call could not be completed."""


class TeamModelResponseError(TeamModelError):
    """A team-model response did not satisfy its typed contract."""


class TeamGraphValidationError(HierarchicalTeamError):
    """A graph configuration, input, or trace violated its contract."""


TeamName: TypeAlias = Literal["team_a", "team_b"]


class EvidenceReference(TypedDict):
    """A team-model reference to one consolidated research finding."""

    subquestion_index: int
    finding_index: int
    relevance: str


class EvidenceAnalysis(TypedDict):
    """Concise structured preparation for drafting one team's argument."""

    strongest_supporting_evidence: list[EvidenceReference]
    limitations_or_conflicting_evidence: list[EvidenceReference]
    key_claims_to_make: list[str]
    unsupported_claims_to_avoid: list[str]
    opponent_points_to_address: list[str]


class TeamCritique(TypedDict):
    """Explicit reviewer feedback; this is not hidden chain-of-thought."""

    stance_consistency: str
    evidence_usage: str
    unsupported_factual_claims: str
    opponent_response: str
    unnecessary_repetition: str
    clarity_and_concision: str


class TeamRevision(TypedDict):
    """The reviewer's critique and final revised argument."""

    critique: TeamCritique
    revised_argument: str


class TeamWorkProduct(TypedDict):
    """Deliberately exported evidence for inspecting one team subgraph."""

    team: TeamName
    assigned_stance: str
    evidence_analysis: EvidenceAnalysis
    draft_argument: str
    internal_critique: TeamCritique
    final_argument: str


class TeamModel(Protocol):
    """Provider-neutral model operations used by a team subgraph."""

    def analyze_evidence(
        self,
        *,
        question: str,
        team_name: TeamName,
        stance: str,
        consolidated_evidence: Sequence[ResearchFinding],
        opponent_argument: str | None,
    ) -> EvidenceAnalysis:
        """Select evidence, claims, limitations, and opponent points."""
        ...

    def draft_argument(
        self,
        *,
        question: str,
        team_name: TeamName,
        stance: str,
        research_findings: Sequence[str],
        evidence_analysis: EvidenceAnalysis,
        opponent_argument: str | None,
    ) -> str:
        """Draft an evidence-aware argument for the assigned stance."""
        ...

    def review_and_revise(
        self,
        *,
        question: str,
        team_name: TeamName,
        stance: str,
        consolidated_evidence: Sequence[ResearchFinding],
        evidence_analysis: EvidenceAnalysis,
        draft_argument: str,
        opponent_argument: str | None,
    ) -> TeamRevision:
        """Critique the draft and return its final revision."""
        ...


class NeutralSynthesisModel(Protocol):
    """Narrow provider boundary needed after both team subgraphs finish."""

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: Sequence[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        """Return one neutral synthesis of the final team arguments."""
        ...


EVIDENCE_REFERENCE_KEYS = frozenset(EvidenceReference.__required_keys__)
EVIDENCE_ANALYSIS_KEYS = frozenset(EvidenceAnalysis.__required_keys__)
TEAM_CRITIQUE_KEYS = frozenset(TeamCritique.__required_keys__)
TEAM_REVISION_KEYS = frozenset(TeamRevision.__required_keys__)
TEAM_WORK_PRODUCT_KEYS = frozenset(TeamWorkProduct.__required_keys__)
_TEAM_CRITIQUE_FIELDS = (
    "stance_consistency",
    "evidence_usage",
    "unsupported_factual_claims",
    "opponent_response",
    "unnecessary_repetition",
    "clarity_and_concision",
)

MIN_ARGUMENT_WORDS = 4
_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_PLACEHOLDER_TOKEN_FORMS = frozenset({"todo", "tbd", "n a", "placeholder"})


def validate_team_name(value: object) -> TeamName:
    """Validate one of the two fixed parent-graph team roles."""

    if value not in ("team_a", "team_b"):
        raise TeamGraphValidationError("team name must be team_a or team_b")
    return cast(TeamName, value)


def validate_configured_stance(value: object) -> str:
    """Validate and normalize the stance captured by a subgraph factory."""

    if not isinstance(value, str) or not value.strip():
        raise TeamGraphValidationError("configured stance must be nonblank")
    return value.strip()


def validate_team_argument(value: object, *, field_name: str = "argument") -> str:
    """Validate a substantive draft or final argument and return a copy."""

    if not isinstance(value, str) or not value.strip():
        raise TeamModelResponseError(f"{field_name} must be nonblank")
    copied = value.strip()
    tokens = _WORD_PATTERN.findall(unicodedata.normalize("NFKC", copied).casefold())
    if " ".join(tokens) in _PLACEHOLDER_TOKEN_FORMS:
        raise TeamModelResponseError(f"{field_name} must not be a placeholder")
    if len(tokens) < MIN_ARGUMENT_WORDS:
        raise TeamModelResponseError(
            f"{field_name} must contain at least {MIN_ARGUMENT_WORDS} words"
        )
    return copied


def _require_exact_keys(
    value: object,
    *,
    expected: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TeamModelResponseError(f"{label} must be a dictionary")
    if set(value) != expected:
        raise TeamModelResponseError(
            f"{label} must contain exactly the required fields"
        )
    return cast(dict[str, object], value)


def _validate_text_list(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list):
        raise TeamModelResponseError(f"{label} must be a list")
    if not minimum <= len(value) <= maximum:
        raise TeamModelResponseError(
            f"{label} must contain between {minimum} and {maximum} items"
        )
    copied: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TeamModelResponseError(f"{label} must contain nonblank strings")
        copied.append(item.strip())
    if len(set(copied)) != len(copied):
        raise TeamModelResponseError(f"{label} must not contain exact duplicates")
    return copied


def _validated_evidence_locators(
    evidence: Sequence[ResearchFinding],
) -> set[tuple[int, int]]:
    if isinstance(evidence, (str, bytes)):
        raise TeamModelResponseError("consolidated evidence must be a sequence")
    locators: set[tuple[int, int]] = set()
    for item in evidence:
        try:
            finding = validate_research_finding(item)
        except ValueError as error:
            raise TeamModelResponseError(
                "consolidated evidence contains an invalid finding"
            ) from error
        locator = (finding["subquestion_index"], finding["finding_index"])
        if locator in locators:
            raise TeamModelResponseError(
                "consolidated evidence contains duplicate evidence locators"
            )
        locators.add(locator)
    return locators


def _validate_reference_list(
    value: object,
    *,
    label: str,
    evidence_locators: set[tuple[int, int]],
) -> list[EvidenceReference]:
    if not isinstance(value, list):
        raise TeamModelResponseError(f"{label} must be a list")
    if len(value) > 3:
        raise TeamModelResponseError(f"{label} must contain no more than three items")

    copied: list[EvidenceReference] = []
    for raw_reference in value:
        reference = _require_exact_keys(
            raw_reference,
            expected=EVIDENCE_REFERENCE_KEYS,
            label="evidence reference",
        )
        subquestion_index = reference["subquestion_index"]
        finding_index = reference["finding_index"]
        relevance = reference["relevance"]
        if type(subquestion_index) is not int or subquestion_index < 0:
            raise TeamModelResponseError(
                "evidence-reference subquestion_index must be a nonnegative integer"
            )
        if type(finding_index) is not int or finding_index <= 0:
            raise TeamModelResponseError(
                "evidence-reference finding_index must be a positive integer"
            )
        if not isinstance(relevance, str) or not relevance.strip():
            raise TeamModelResponseError(
                "evidence-reference relevance must be nonblank"
            )
        locator = (subquestion_index, finding_index)
        if locator not in evidence_locators:
            raise TeamModelResponseError(
                "evidence reference does not resolve to consolidated evidence"
            )
        copied.append(
            {
                "subquestion_index": subquestion_index,
                "finding_index": finding_index,
                "relevance": relevance.strip(),
            }
        )
    return copied


def validate_evidence_analysis(
    value: object,
    *,
    team_name: TeamName,
    consolidated_evidence: Sequence[ResearchFinding],
) -> EvidenceAnalysis:
    """Validate and deeply copy an evidence-analysis model product."""

    validated_team = validate_team_name(team_name)
    analysis = _require_exact_keys(
        value,
        expected=EVIDENCE_ANALYSIS_KEYS,
        label="evidence analysis",
    )
    evidence_locators = _validated_evidence_locators(consolidated_evidence)
    supporting = _validate_reference_list(
        analysis["strongest_supporting_evidence"],
        label="strongest_supporting_evidence",
        evidence_locators=evidence_locators,
    )
    limitations = _validate_reference_list(
        analysis["limitations_or_conflicting_evidence"],
        label="limitations_or_conflicting_evidence",
        evidence_locators=evidence_locators,
    )
    all_locators = [
        (item["subquestion_index"], item["finding_index"])
        for item in (*supporting, *limitations)
    ]
    if not all_locators:
        raise TeamModelResponseError(
            "evidence analysis must reference at least one finding"
        )
    if len(set(all_locators)) != len(all_locators):
        raise TeamModelResponseError(
            "evidence analysis must not repeat evidence references"
        )

    key_claims = _validate_text_list(
        analysis["key_claims_to_make"],
        label="key_claims_to_make",
        minimum=1,
        maximum=4,
    )
    unsupported = _validate_text_list(
        analysis["unsupported_claims_to_avoid"],
        label="unsupported_claims_to_avoid",
        minimum=0,
        maximum=4,
    )
    opponent_minimum = 0 if validated_team == "team_a" else 1
    opponent_maximum = 0 if validated_team == "team_a" else 3
    opponent_points = _validate_text_list(
        analysis["opponent_points_to_address"],
        label="opponent_points_to_address",
        minimum=opponent_minimum,
        maximum=opponent_maximum,
    )
    return {
        "strongest_supporting_evidence": supporting,
        "limitations_or_conflicting_evidence": limitations,
        "key_claims_to_make": key_claims,
        "unsupported_claims_to_avoid": unsupported,
        "opponent_points_to_address": opponent_points,
    }


def validate_team_critique(value: object) -> TeamCritique:
    """Validate and copy a six-part internal critique."""

    critique = _require_exact_keys(
        value,
        expected=TEAM_CRITIQUE_KEYS,
        label="team critique",
    )
    copied_values: dict[str, str] = {}
    for field in _TEAM_CRITIQUE_FIELDS:
        field_value = critique[field]
        if not isinstance(field_value, str) or not field_value.strip():
            raise TeamModelResponseError(f"critique field {field} must be nonblank")
        copied_values[field] = field_value.strip()
    return {
        "stance_consistency": copied_values["stance_consistency"],
        "evidence_usage": copied_values["evidence_usage"],
        "unsupported_factual_claims": copied_values[
            "unsupported_factual_claims"
        ],
        "opponent_response": copied_values["opponent_response"],
        "unnecessary_repetition": copied_values["unnecessary_repetition"],
        "clarity_and_concision": copied_values["clarity_and_concision"],
    }


def validate_team_revision(value: object) -> TeamRevision:
    """Validate and deeply copy a reviewer response."""

    revision = _require_exact_keys(
        value,
        expected=TEAM_REVISION_KEYS,
        label="team revision",
    )
    return {
        "critique": validate_team_critique(revision["critique"]),
        "revised_argument": validate_team_argument(
            revision["revised_argument"], field_name="revised_argument"
        ),
    }


def validate_team_work_product(
    value: object,
    *,
    consolidated_evidence: Sequence[ResearchFinding],
    expected_team: TeamName | None = None,
    expected_stance: str | None = None,
) -> TeamWorkProduct:
    """Validate and deeply copy an exported team work product."""

    product = _require_exact_keys(
        value,
        expected=TEAM_WORK_PRODUCT_KEYS,
        label="team work product",
    )
    team = validate_team_name(product["team"])
    stance = validate_configured_stance(product["assigned_stance"])
    if expected_team is not None and team != validate_team_name(expected_team):
        raise TeamModelResponseError("team work product has the wrong team")
    if expected_stance is not None and stance != expected_stance:
        raise TeamModelResponseError("team work product has the wrong stance")
    return {
        "team": team,
        "assigned_stance": stance,
        "evidence_analysis": validate_evidence_analysis(
            product["evidence_analysis"],
            team_name=team,
            consolidated_evidence=consolidated_evidence,
        ),
        "draft_argument": validate_team_argument(
            product["draft_argument"], field_name="draft_argument"
        ),
        "internal_critique": validate_team_critique(product["internal_critique"]),
        "final_argument": validate_team_argument(
            product["final_argument"], field_name="final_argument"
        ),
    }
