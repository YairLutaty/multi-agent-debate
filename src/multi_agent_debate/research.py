"""Provider-independent research contracts and deterministic evidence helpers."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias, TypedDict


class ResearchFinding(TypedDict):
    """One research claim with its routing and source provenance."""

    subquestion_index: int
    subquestion: str
    finding_index: int
    claim: str
    source_title: str
    source_url: str
    source_type: str
    supporting_excerpt: str


RawResearchFindings: TypeAlias = list[ResearchFinding]
ConsolidatedEvidence: TypeAlias = list[ResearchFinding]

RESEARCH_FINDING_KEYS = frozenset(ResearchFinding.__required_keys__)
_TEXT_FIELDS = (
    "subquestion",
    "claim",
    "source_title",
    "source_url",
    "source_type",
    "supporting_excerpt",
)


class ResearchProvider(Protocol):
    """Return structured evidence for one research question.

    Synchronous LangGraph ``Send`` workers may call ``research`` concurrently
    from separate thread-pool threads. Implementations must therefore be
    thread-safe or serialize access internally. A future asyncio-native MCP
    client must not be shared and called directly from multiple worker threads;
    an async MCP adapter will need a deliberate synchronous, thread-safe bridge,
    such as a dedicated event-loop thread.
    """

    def research(
        self,
        *,
        subquestion_index: int,
        subquestion: str,
    ) -> Sequence[ResearchFinding]:
        """Return zero or more findings for one routed research question."""
        ...


def validate_research_finding(
    value: object,
    *,
    expected_subquestion_index: int | None = None,
    expected_subquestion: str | None = None,
) -> ResearchFinding:
    """Validate and copy one finding at a worker or consolidation boundary."""

    if not isinstance(value, dict):
        raise ValueError("research finding must be a dictionary")

    keys = set(value)
    if keys != RESEARCH_FINDING_KEYS:
        missing = sorted(RESEARCH_FINDING_KEYS - keys)
        unexpected = sorted(repr(key) for key in keys - RESEARCH_FINDING_KEYS)
        raise ValueError(
            "research finding must contain exactly the required keys; "
            f"missing={missing}, unexpected={unexpected}"
        )

    subquestion_index = value["subquestion_index"]
    if type(subquestion_index) is not int:
        raise ValueError("subquestion_index must be an integer")
    if (
        expected_subquestion_index is not None
        and subquestion_index != expected_subquestion_index
    ):
        raise ValueError("subquestion_index does not match the research task")

    finding_index = value["finding_index"]
    if type(finding_index) is not int:
        raise ValueError("finding_index must be an integer")
    if finding_index <= 0:
        raise ValueError("finding_index must be positive")

    for field in _TEXT_FIELDS:
        field_value = value[field]
        if not isinstance(field_value, str):
            raise ValueError(f"{field} must be a string")
        if not field_value.strip():
            raise ValueError(f"{field} must be nonempty")

    subquestion = value["subquestion"]
    if expected_subquestion is not None and subquestion != expected_subquestion:
        raise ValueError("subquestion does not match the research task")

    return {
        "subquestion_index": subquestion_index,
        "subquestion": subquestion,
        "finding_index": finding_index,
        "claim": value["claim"],
        "source_title": value["source_title"],
        "source_url": value["source_url"],
        "source_type": value["source_type"],
        "supporting_excerpt": value["supporting_excerpt"],
    }


def finding_sort_key(
    finding: ResearchFinding,
) -> tuple[int, int, str, str, str, str, str, str]:
    """Return a total canonical key containing all finding fields."""

    return (
        finding["subquestion_index"],
        finding["finding_index"],
        finding["source_url"],
        finding["claim"],
        finding["subquestion"],
        finding["source_title"],
        finding["source_type"],
        finding["supporting_excerpt"],
    )


def merge_raw_findings(
    left: RawResearchFindings,
    right: RawResearchFindings,
) -> RawResearchFindings:
    """Copy, retain, and canonically order all concurrent worker findings."""

    combined = [
        *(finding.copy() for finding in left),
        *(finding.copy() for finding in right),
    ]
    return sorted(combined, key=finding_sort_key)


def consolidate_research_findings(
    *,
    question: str,
    question_complexity: str,
    subquestions: Sequence[str],
    raw_findings: Sequence[ResearchFinding],
) -> ConsolidatedEvidence:
    """Revalidate, sort, and exactly deduplicate accumulated findings."""

    if not raw_findings:
        raise ValueError("research produced no findings")

    validated: list[ResearchFinding] = []
    for raw_finding in raw_findings:
        finding = validate_research_finding(raw_finding)
        if question_complexity == "simple":
            expected_index = 0
            expected_question = question
        elif question_complexity == "complex":
            expected_index = finding["subquestion_index"]
            if not 1 <= expected_index <= len(subquestions):
                raise ValueError(
                    "subquestion_index does not identify a generated subquestion"
                )
            expected_question = subquestions[expected_index - 1]
        else:
            raise ValueError("question_complexity must be simple or complex")

        validated.append(
            validate_research_finding(
                finding,
                expected_subquestion_index=expected_index,
                expected_subquestion=expected_question,
            )
        )

    consolidated: list[ResearchFinding] = []
    seen: set[tuple[int, int, str, str, str, str, str, str]] = set()
    for finding in sorted(validated, key=finding_sort_key):
        identity = finding_sort_key(finding)
        if identity not in seen:
            seen.add(identity)
            consolidated.append(finding.copy())
    return consolidated


def format_research_finding(finding: ResearchFinding) -> str:
    """Format one validated finding as a readable, fixed-order text block."""

    return "\n".join(
        (
            f"Claim: {finding['claim']}",
            f"Source: {finding['source_title']}",
            f"Source type: {finding['source_type']}",
            f"URL: {finding['source_url']}",
            f"Supporting excerpt: {finding['supporting_excerpt']}",
            f"Research question: {finding['subquestion']}",
            f"Subquestion index: {finding['subquestion_index']}",
            f"Finding index: {finding['finding_index']}",
        )
    )
