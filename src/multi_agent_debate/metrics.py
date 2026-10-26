"""Deterministic metric version 1.0 for saved debate transcripts."""

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean

from multi_agent_debate.transcript import DebateTranscript

METRIC_VERSION = "1.0"
SHINGLE_SIZE = 3
WORD_REGEX = r"[^\W_]+"

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "d",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "ll",
        "m",
        "may",
        "might",
        "of",
        "on",
        "or",
        "our",
        "re",
        "s",
        "she",
        "should",
        "t",
        "than",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "those",
        "to",
        "ve",
        "was",
        "we",
        "were",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

VERDICT_PATTERN_SPECS = (
    ("team_a_wins", r"\bteam\s+a\s+wins\b"),
    ("team_b_wins", r"\bteam\s+b\s+wins\b"),
    (
        "team_a_more_convincing",
        r"\bteam\s+a\s+is\s+more\s+convincing\b",
    ),
    (
        "team_b_more_convincing",
        r"\bteam\s+b\s+is\s+more\s+convincing\b",
    ),
    ("stronger_argument", r"\bthe\s+stronger\s+argument\s+is\b"),
    ("winning_side", r"\bthe\s+winning\s+side\s+is\b"),
    ("my_verdict", r"\bmy(?:\s+final)?\s+verdict\s+is\b"),
    ("i_conclude", r"\bi(?:\s+therefore)?\s+conclude\s+that\b"),
    ("i_recommend", r"\bi(?:\s+would)?\s+recommend\b"),
)

_WORD_PATTERN = re.compile(WORD_REGEX, re.UNICODE)
_VERDICT_PATTERNS = tuple(
    (pattern_id, re.compile(pattern, re.IGNORECASE))
    for pattern_id, pattern in VERDICT_PATTERN_SPECS
)


@dataclass(frozen=True, slots=True)
class FindingUtilization:
    finding_index: int
    team_a_coverage: float
    team_b_coverage: float
    synthesis_coverage: float
    utilization: float


@dataclass(frozen=True, slots=True)
class ResearchUtilizationScore:
    score: float
    findings: tuple[FindingUtilization, ...]


@dataclass(frozen=True, slots=True)
class TeamNoveltyScore:
    score: float
    jaccard_similarity: float
    team_a_shingle_count: int
    team_b_shingle_count: int
    shared_shingle_count: int
    union_shingle_count: int


@dataclass(frozen=True, slots=True)
class SynthesisRepresentationScore:
    score: float
    team_a_coverage: float
    team_b_coverage: float
    mean_coverage: float
    balance: float


@dataclass(frozen=True, slots=True)
class SummaryPolicyComplianceScore:
    score: int
    matched_pattern_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WordCounts:
    research_finding_words: tuple[int, ...]
    research_total_words: int
    team_a_argument_words: int
    team_b_argument_words: int
    synthesis_words: int
    total_output_words: int


@dataclass(frozen=True, slots=True)
class Scorecard:
    metric_version: str
    research_utilization: ResearchUtilizationScore
    team_novelty: TeamNoveltyScore
    synthesis_representation: SynthesisRepresentationScore
    summary_policy_compliance: SummaryPolicyComplianceScore
    word_counts: WordCounts


def tokenize_words(text: str) -> tuple[str, ...]:
    """Return NFKC-normalized, case-folded word tokens including stop words."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_WORD_PATTERN.findall(normalized))


def normalize_text(text: str) -> tuple[str, ...]:
    """Return lexical tokens after removing the frozen stop-word list."""

    return tuple(token for token in tokenize_words(text) if token not in STOP_WORDS)


def calculate_research_utilization(
    research_findings: Sequence[str],
    team_a_argument: str,
    team_b_argument: str,
    synthesis: str,
) -> ResearchUtilizationScore:
    downstream_token_sets = (
        set(normalize_text(team_a_argument)),
        set(normalize_text(team_b_argument)),
        set(normalize_text(synthesis)),
    )
    finding_scores: list[FindingUtilization] = []

    for index, finding in enumerate(research_findings, start=1):
        finding_tokens = set(normalize_text(finding))
        coverages = tuple(
            _set_coverage(finding_tokens, output_tokens)
            for output_tokens in downstream_token_sets
        )
        finding_scores.append(
            FindingUtilization(
                finding_index=index,
                team_a_coverage=coverages[0],
                team_b_coverage=coverages[1],
                synthesis_coverage=coverages[2],
                utilization=max(coverages),
            )
        )

    score = (
        fmean(item.utilization for item in finding_scores)
        if finding_scores
        else 0.0
    )
    return ResearchUtilizationScore(score=score, findings=tuple(finding_scores))


def calculate_team_novelty(
    team_a_argument: str,
    team_b_argument: str,
) -> TeamNoveltyScore:
    team_a_tokens = normalize_text(team_a_argument)
    team_b_tokens = normalize_text(team_b_argument)
    team_a_shingles = _shingles(team_a_tokens)
    team_b_shingles = _shingles(team_b_tokens)
    shared_shingles = team_a_shingles & team_b_shingles
    union_shingles = team_a_shingles | team_b_shingles

    if not team_a_tokens or not team_b_tokens or not union_shingles:
        similarity = 1.0
        score = 0.0
    else:
        similarity = len(shared_shingles) / len(union_shingles)
        score = 1.0 - similarity

    return TeamNoveltyScore(
        score=score,
        jaccard_similarity=similarity,
        team_a_shingle_count=len(team_a_shingles),
        team_b_shingle_count=len(team_b_shingles),
        shared_shingle_count=len(shared_shingles),
        union_shingle_count=len(union_shingles),
    )


def calculate_synthesis_representation(
    team_a_argument: str,
    team_b_argument: str,
    synthesis: str,
) -> SynthesisRepresentationScore:
    team_a_tokens = set(normalize_text(team_a_argument))
    team_b_tokens = set(normalize_text(team_b_argument))
    synthesis_tokens = set(normalize_text(synthesis))

    if not team_a_tokens or not team_b_tokens or not synthesis_tokens:
        return SynthesisRepresentationScore(
            score=0.0,
            team_a_coverage=0.0,
            team_b_coverage=0.0,
            mean_coverage=0.0,
            balance=0.0,
        )

    team_a_coverage = _set_coverage(team_a_tokens, synthesis_tokens)
    team_b_coverage = _set_coverage(team_b_tokens, synthesis_tokens)
    mean_coverage = (team_a_coverage + team_b_coverage) / 2
    balance = 1.0 - abs(team_a_coverage - team_b_coverage)
    return SynthesisRepresentationScore(
        score=mean_coverage * balance,
        team_a_coverage=team_a_coverage,
        team_b_coverage=team_b_coverage,
        mean_coverage=mean_coverage,
        balance=balance,
    )


def calculate_summary_policy_compliance(
    synthesis: str,
) -> SummaryPolicyComplianceScore:
    normalized = unicodedata.normalize("NFKC", synthesis)
    matched_pattern_ids = tuple(
        pattern_id
        for pattern_id, pattern in _VERDICT_PATTERNS
        if pattern.search(normalized)
    )
    return SummaryPolicyComplianceScore(
        score=0 if matched_pattern_ids else 1,
        matched_pattern_ids=matched_pattern_ids,
    )


def count_output_words(
    research_findings: Sequence[str],
    team_a_argument: str,
    team_b_argument: str,
    synthesis: str,
) -> WordCounts:
    research_finding_words = tuple(
        len(tokenize_words(finding)) for finding in research_findings
    )
    research_total_words = sum(research_finding_words)
    team_a_argument_words = len(tokenize_words(team_a_argument))
    team_b_argument_words = len(tokenize_words(team_b_argument))
    synthesis_words = len(tokenize_words(synthesis))
    return WordCounts(
        research_finding_words=research_finding_words,
        research_total_words=research_total_words,
        team_a_argument_words=team_a_argument_words,
        team_b_argument_words=team_b_argument_words,
        synthesis_words=synthesis_words,
        total_output_words=(
            research_total_words
            + team_a_argument_words
            + team_b_argument_words
            + synthesis_words
        ),
    )


def score_transcript(transcript: DebateTranscript) -> Scorecard:
    return Scorecard(
        metric_version=METRIC_VERSION,
        research_utilization=calculate_research_utilization(
            transcript.research_findings,
            transcript.team_a_argument,
            transcript.team_b_argument,
            transcript.synthesis,
        ),
        team_novelty=calculate_team_novelty(
            transcript.team_a_argument,
            transcript.team_b_argument,
        ),
        synthesis_representation=calculate_synthesis_representation(
            transcript.team_a_argument,
            transcript.team_b_argument,
            transcript.synthesis,
        ),
        summary_policy_compliance=calculate_summary_policy_compliance(
            transcript.synthesis
        ),
        word_counts=count_output_words(
            transcript.research_findings,
            transcript.team_a_argument,
            transcript.team_b_argument,
            transcript.synthesis,
        ),
    )


def scorecard_to_json(scorecard: Scorecard) -> str:
    """Serialize a scorecard with a stable order and exactly one final newline."""

    return (
        json.dumps(
            asdict(scorecard),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _set_coverage(source: set[str], target: set[str]) -> float:
    if not source:
        return 0.0
    return len(source & target) / len(source)


def _shingles(tokens: Sequence[str]) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + SHINGLE_SIZE])
        for index in range(len(tokens) - SHINGLE_SIZE + 1)
    }
