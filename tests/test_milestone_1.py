"""Offline tests for deterministic metric version 1.0."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from multi_agent_debate.metrics import (
    METRIC_VERSION,
    calculate_research_utilization,
    calculate_summary_policy_compliance,
    calculate_synthesis_representation,
    calculate_team_novelty,
    count_output_words,
    normalize_text,
    score_transcript,
    scorecard_to_json,
    tokenize_words,
)
from multi_agent_debate.score import main as score_main
from multi_agent_debate.transcript import (
    TranscriptParseError,
    load_transcript,
    parse_transcript,
)

BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "baselines"
    / "m0_ubi_qwen3_8b.txt"
)


def _transcript_text(
    *,
    findings: str = "1. Support finding.\n2. Limitation finding.\n3. Uncertainty.",
    synthesis: str = "Neutral synthesis.",
    footer: str | None = None,
    after_footer: str = "",
) -> str:
    timing_footer = f"\n{footer}" if footer is not None else ""
    trailing_content = f"\n{after_footer}" if after_footer else ""
    return f"""\
Milestone: 0
Model: local
Team A stance:
This exact header is metadata and must be ignored before the Question section.

Question:
Should a policy be adopted?

Team A stance:
Support the proposition.

Team B stance:
Oppose the proposition.

Simulated research findings:
{findings}

Team A argument:
Team A argument text.

Team B argument:
Team B argument text.

Neutral synthesis:
{synthesis}{timing_footer}{trailing_content}
"""


def test_tokenize_words_retains_stop_words_and_normalizes_unicode() -> None:
    text = "ＴＨＥ Café—Straße, and UBI_Design UBI2024!"

    assert tokenize_words(text) == (
        "the",
        "café",
        "strasse",
        "and",
        "ubi",
        "design",
        "ubi2024",
    )


def test_normalize_text_removes_frozen_stop_words() -> None:
    assert normalize_text("The policy and its design") == ("policy", "design")
    assert normalize_text("Team B's response is not empty") == (
        "team",
        "b",
        "response",
        "not",
        "empty",
    )


def test_research_utilization_uses_best_set_coverage() -> None:
    result = calculate_research_utilization(
        ("alpha beta", "gamma delta"),
        "alpha",
        "gamma",
        "beta delta",
    )

    assert result.score == 0.5
    assert result.findings[0].team_a_coverage == 0.5
    assert result.findings[0].synthesis_coverage == 0.5
    assert result.findings[0].utilization == 0.5
    assert result.findings[1].team_b_coverage == 0.5
    assert result.findings[1].utilization == 0.5


def test_research_utilization_empty_inputs_score_zero() -> None:
    empty_finding = calculate_research_utilization(("the and",), "", "", "")
    no_findings = calculate_research_utilization((), "alpha", "beta", "gamma")

    assert empty_finding.score == 0.0
    assert empty_finding.findings[0].utilization == 0.0
    assert no_findings.score == 0.0
    assert no_findings.findings == ()


def test_team_novelty_identical_partial_and_disjoint_arguments() -> None:
    identical = calculate_team_novelty(
        "alpha beta gamma delta",
        "alpha beta gamma delta",
    )
    partial = calculate_team_novelty(
        "alpha beta gamma delta",
        "beta gamma delta epsilon",
    )
    disjoint = calculate_team_novelty(
        "alpha beta gamma",
        "delta epsilon zeta",
    )

    assert identical.jaccard_similarity == 1.0
    assert identical.score == 0.0
    assert partial.shared_shingle_count == 1
    assert partial.union_shingle_count == 3
    assert partial.jaccard_similarity == pytest.approx(1 / 3)
    assert partial.score == pytest.approx(2 / 3)
    assert disjoint.jaccard_similarity == 0.0
    assert disjoint.score == 1.0


def test_team_novelty_empty_and_short_input_rules() -> None:
    empty = calculate_team_novelty("", "alpha beta gamma")
    both_short = calculate_team_novelty("alpha beta", "gamma delta")
    one_short = calculate_team_novelty("alpha beta", "gamma delta epsilon")

    assert empty.score == 0.0
    assert empty.jaccard_similarity == 1.0
    assert both_short.union_shingle_count == 0
    assert both_short.score == 0.0
    assert one_short.union_shingle_count == 1
    assert one_short.score == 1.0


def test_synthesis_representation_rewards_coverage_and_balance() -> None:
    balanced = calculate_synthesis_representation(
        "alpha beta",
        "gamma delta",
        "alpha gamma",
    )
    imbalanced = calculate_synthesis_representation(
        "alpha beta",
        "gamma delta",
        "alpha beta gamma",
    )

    assert balanced.team_a_coverage == 0.5
    assert balanced.team_b_coverage == 0.5
    assert balanced.mean_coverage == 0.5
    assert balanced.balance == 1.0
    assert balanced.score == 0.5
    assert imbalanced.team_a_coverage == 1.0
    assert imbalanced.team_b_coverage == 0.5
    assert imbalanced.mean_coverage == 0.75
    assert imbalanced.balance == 0.5
    assert imbalanced.score == 0.375


@pytest.mark.parametrize(
    ("team_a", "team_b", "synthesis"),
    [
        ("", "beta", "beta"),
        ("alpha", "", "alpha"),
        ("alpha", "beta", ""),
    ],
)
def test_synthesis_representation_empty_inputs_score_zero(
    team_a: str,
    team_b: str,
    synthesis: str,
) -> None:
    result = calculate_synthesis_representation(team_a, team_b, synthesis)

    assert result.score == 0.0
    assert result.team_a_coverage == 0.0
    assert result.team_b_coverage == 0.0
    assert result.mean_coverage == 0.0
    assert result.balance == 0.0


@pytest.mark.parametrize(
    ("text", "pattern_id"),
    [
        ("TEAM A WINS this debate.", "team_a_wins"),
        ("Team B wins this debate.", "team_b_wins"),
        ("Team A is more convincing.", "team_a_more_convincing"),
        ("Team B is more convincing.", "team_b_more_convincing"),
        ("The stronger argument is the first.", "stronger_argument"),
        ("The winning side is Team B.", "winning_side"),
        ("My final verdict is against it.", "my_verdict"),
        ("I therefore conclude that it should pass.", "i_conclude"),
        ("I would recommend adoption.", "i_recommend"),
    ],
)
def test_every_verdict_pattern_is_detected(text: str, pattern_id: str) -> None:
    result = calculate_summary_policy_compliance(text)

    assert result.score == 0
    assert pattern_id in result.matched_pattern_ids


def test_verdict_variants_and_all_matches_are_reported() -> None:
    result = calculate_summary_policy_compliance(
        "My verdict is clear. I conclude that Team A wins, and I recommend it."
    )

    assert result.score == 0
    assert result.matched_pattern_ids == (
        "team_a_wins",
        "my_verdict",
        "i_conclude",
        "i_recommend",
    )


def test_neutral_and_empty_summaries_are_policy_compliant() -> None:
    neutral = calculate_summary_policy_compliance("Both positions involve tradeoffs.")
    empty = calculate_summary_policy_compliance("")

    assert neutral.score == 1
    assert neutral.matched_pattern_ids == ()
    assert empty.score == 1
    assert empty.matched_pattern_ids == ()


def test_word_counts_include_stop_words_and_report_each_section() -> None:
    counts = count_output_words(
        ("The alpha", "Beta and gamma", ""),
        "The team responds",
        "",
        "A final synthesis",
    )

    assert counts.research_finding_words == (2, 3, 0)
    assert counts.research_total_words == 5
    assert counts.team_a_argument_words == 3
    assert counts.team_b_argument_words == 0
    assert counts.synthesis_words == 3
    assert counts.total_output_words == 11


def test_transcript_parser_ignores_metadata_and_extracts_all_sections() -> None:
    transcript = parse_transcript(_transcript_text())

    assert transcript.question == "Should a policy be adopted?"
    assert transcript.team_a_stance == "Support the proposition."
    assert transcript.team_b_stance == "Oppose the proposition."
    assert transcript.research_findings == (
        "Support finding.",
        "Limitation finding.",
        "Uncertainty.",
    )
    assert transcript.team_a_argument == "Team A argument text."
    assert transcript.team_b_argument == "Team B argument text."
    assert transcript.synthesis == "Neutral synthesis."


def test_transcript_parser_rejects_empty_synthesis() -> None:
    with pytest.raises(
        TranscriptParseError,
        match="required section is empty: Neutral synthesis",
    ):
        parse_transcript(_transcript_text(synthesis=""))


def test_transcript_parser_removes_one_generic_timing_footer() -> None:
    footer = "arbitrary command --flag 0.01s user 0.02s system 3% cpu 0.030 total"
    transcript = parse_transcript(_transcript_text(footer=footer))

    assert transcript.synthesis == "Neutral synthesis."


def test_transcript_parser_rejects_content_after_timing_footer() -> None:
    footer = "some command 0.01s user 0.02s system 3% cpu 0.030 total"

    with pytest.raises(TranscriptParseError, match="unexpected content"):
        parse_transcript(
            _transcript_text(footer=footer, after_footer="unexpected trailing text")
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda text: text.replace("Team B argument:\nTeam B argument text.\n", ""),
            "missing required section",
        ),
        (
            lambda text: text.replace(
                "Team A stance:\nSupport the proposition.",
                "Team A stance:\nSupport the proposition.\n\n"
                "Team A stance:\nDuplicate.",
            ),
            "duplicate section header",
        ),
        (
            lambda text: text.replace(
                "Team A stance:\nSupport the proposition.\n\n"
                "Team B stance:\nOppose the proposition.",
                "Team B stance:\nOppose the proposition.\n\n"
                "Team A stance:\nSupport the proposition.",
            ),
            "out of order",
        ),
        (
            lambda text: text.replace(
                "Team A argument:\nTeam A argument text.",
                "Team A argument:",
            ),
            "required section is empty",
        ),
    ],
)
def test_transcript_parser_rejects_missing_duplicate_out_of_order_and_empty(
    mutation: Callable[[str], str],
    message: str,
) -> None:
    with pytest.raises(TranscriptParseError, match=message):
        parse_transcript(mutation(_transcript_text()))


@pytest.mark.parametrize(
    "findings",
    [
        "1. One.\n2. Two.",
        "1. One.\n2. Two.\n3. Three.\n4. Four.",
        "1. One.\n3. Three.\n2. Two.",
        "1. One.\n2. Two.\n4. Four.",
        "1. One.\nTwo without a number.\n3. Three.",
    ],
)
def test_transcript_parser_requires_exactly_three_numbered_findings(
    findings: str,
) -> None:
    with pytest.raises(TranscriptParseError, match="research findings"):
        parse_transcript(_transcript_text(findings=findings))


def test_scorecard_and_json_are_deterministic() -> None:
    transcript = parse_transcript(_transcript_text())

    first = score_transcript(transcript)
    second = score_transcript(transcript)
    first_json = scorecard_to_json(first)
    second_json = scorecard_to_json(second)

    assert first == second
    assert first_json == second_json
    assert first_json.endswith("\n")
    assert not first_json.endswith("\n\n")
    assert "timestamp" not in first_json
    assert "source_path" not in first_json
    assert json.loads(first_json)["metric_version"] == METRIC_VERSION


def test_scoring_cli_human_and_json_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(_transcript_text(), encoding="utf-8")

    assert score_main([str(transcript_path)]) == 0
    human_output = capsys.readouterr().out
    assert "Metric version: 1.0" in human_output
    assert "Research utilization:" in human_output
    assert "Output words:" in human_output

    assert score_main(["--format", "json", str(transcript_path)]) == 0
    json_output = capsys.readouterr().out
    parsed = json.loads(json_output)
    assert parsed["metric_version"] == METRIC_VERSION
    assert json_output.endswith("\n")
    assert not json_output.endswith("\n\n")


def test_scoring_cli_reports_missing_transcript_without_partial_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing-transcript.txt"

    assert score_main(["--format", "json", str(missing_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "metric_version" not in captured.err


def test_existing_baseline_loads_and_scores_without_regeneration() -> None:
    transcript = load_transcript(BASELINE_PATH)
    scorecard = score_transcript(transcript)

    assert transcript.question == (
        "Should governments provide a universal basic income to all adult citizens?"
    )
    assert len(transcript.research_findings) == 3
    assert "s user" not in transcript.synthesis
    assert scorecard.metric_version == METRIC_VERSION
    assert scorecard.research_utilization.score == pytest.approx(
        0.6986111111111111
    )
    assert scorecard.team_novelty.score == pytest.approx(0.9885057471264368)
    assert scorecard.synthesis_representation.score == pytest.approx(
        0.6734537956462252
    )
    assert scorecard.summary_policy_compliance.score == 1
    assert scorecard.word_counts.research_finding_words == (18, 23, 26)
    assert scorecard.word_counts.research_total_words == 67
    assert scorecard.word_counts.team_a_argument_words == 74
    assert scorecard.word_counts.team_b_argument_words == 60
    assert scorecard.word_counts.synthesis_words == 172
    assert scorecard.word_counts.total_output_words == 373
