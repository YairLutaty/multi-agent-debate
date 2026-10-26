"""Strict parser for saved fixed-format debate transcripts."""

import re
from dataclasses import dataclass
from pathlib import Path

SECTION_HEADERS = (
    "Question:",
    "Team A stance:",
    "Team B stance:",
    "Simulated research findings:",
    "Team A argument:",
    "Team B argument:",
    "Neutral synthesis:",
)

_FINDING_PATTERN = re.compile(r"^([1-3])\.\s+(.+)$")
_TIMING_FOOTER_PATTERN = re.compile(
    r"^\S.*\s+\d+(?:\.\d+)?s\s+user\s+"
    r"\d+(?:\.\d+)?s\s+system\s+"
    r"\d+(?:\.\d+)?%\s+cpu\s+"
    r"\d+(?:\.\d+)?\s+total\s*$"
)


class TranscriptParseError(ValueError):
    """Raised when a saved transcript violates the fixed format."""


@dataclass(frozen=True, slots=True)
class DebateTranscript:
    question: str
    team_a_stance: str
    team_b_stance: str
    research_findings: tuple[str, str, str]
    team_a_argument: str
    team_b_argument: str
    synthesis: str


def load_transcript(path: str | Path) -> DebateTranscript:
    return parse_transcript(Path(path).read_text(encoding="utf-8"))


def parse_transcript(text: str) -> DebateTranscript:
    lines = text.splitlines()
    question_indexes = [
        index for index, line in enumerate(lines) if line == SECTION_HEADERS[0]
    ]
    if not question_indexes:
        raise TranscriptParseError("missing required section: Question")

    section_start = question_indexes[0]
    section_lines = lines[section_start:]
    positions: list[int] = []
    for header in SECTION_HEADERS:
        indexes = [index for index, line in enumerate(section_lines) if line == header]
        if not indexes:
            raise TranscriptParseError(f"missing required section: {header[:-1]}")
        if len(indexes) > 1:
            raise TranscriptParseError(f"duplicate section header: {header[:-1]}")
        positions.append(indexes[0])

    if positions != sorted(positions):
        raise TranscriptParseError("section headers are out of order")

    section_values: list[str] = []
    for index, header in enumerate(SECTION_HEADERS):
        content_start = positions[index] + 1
        content_end = positions[index + 1] if index + 1 < len(positions) else len(
            section_lines
        )
        content_lines = section_lines[content_start:content_end]
        if header == "Neutral synthesis:":
            content_lines = _remove_timing_footer(content_lines)
        content = "\n".join(content_lines).strip()
        if not content:
            raise TranscriptParseError(f"required section is empty: {header[:-1]}")
        section_values.append(content)

    findings = _parse_findings(section_values[3])
    return DebateTranscript(
        question=section_values[0],
        team_a_stance=section_values[1],
        team_b_stance=section_values[2],
        research_findings=findings,
        team_a_argument=section_values[4],
        team_b_argument=section_values[5],
        synthesis=section_values[6],
    )


def _parse_findings(content: str) -> tuple[str, str, str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) != 3:
        raise TranscriptParseError(
            "research findings must contain exactly three numbered lines"
        )

    findings: list[str] = []
    for expected_number, line in enumerate(lines, start=1):
        match = _FINDING_PATTERN.fullmatch(line)
        if match is None or int(match.group(1)) != expected_number:
            raise TranscriptParseError("research findings must be numbered 1 through 3")
        finding = match.group(2).strip()
        if not finding:
            raise TranscriptParseError("research findings must be nonempty")
        findings.append(finding)

    return findings[0], findings[1], findings[2]


def _remove_timing_footer(lines: list[str]) -> list[str]:
    footer_indexes = [
        index
        for index, line in enumerate(lines)
        if _TIMING_FOOTER_PATTERN.fullmatch(line)
    ]
    if len(footer_indexes) > 1:
        raise TranscriptParseError("multiple shell timing footers found")
    if not footer_indexes:
        return lines

    footer_index = footer_indexes[0]
    if any(line.strip() for line in lines[footer_index + 1 :]):
        raise TranscriptParseError("unexpected content after shell timing footer")
    return lines[:footer_index]
