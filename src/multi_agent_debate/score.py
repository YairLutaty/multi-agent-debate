"""Offline command-line scoring for saved debate transcripts."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from multi_agent_debate.metrics import Scorecard, score_transcript, scorecard_to_json
from multi_agent_debate.transcript import TranscriptParseError, load_transcript


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a saved debate transcript without running a model."
    )
    parser.add_argument("transcript", type=Path, help="Saved transcript to score.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Defaults to concise text.",
    )
    return parser.parse_args(argv)


def format_scorecard(scorecard: Scorecard) -> str:
    research = scorecard.research_utilization
    novelty = scorecard.team_novelty
    representation = scorecard.synthesis_representation
    compliance = scorecard.summary_policy_compliance
    counts = scorecard.word_counts
    finding_values = ", ".join(
        f"F{item.finding_index}={item.utilization:.6f}"
        for item in research.findings
    )
    matches = ", ".join(compliance.matched_pattern_ids) or "none"
    return "\n".join(
        (
            f"Metric version: {scorecard.metric_version}",
            f"Research utilization: {research.score:.6f}",
            f"  Finding utilization: {finding_values or 'none'}",
            f"Team novelty: {novelty.score:.6f}",
            f"  Jaccard similarity: {novelty.jaccard_similarity:.6f}",
            f"Synthesis representation: {representation.score:.6f}",
            f"  Team A coverage: {representation.team_a_coverage:.6f}",
            f"  Team B coverage: {representation.team_b_coverage:.6f}",
            f"Summary-policy compliance: {compliance.score}",
            f"  Matched verdict patterns: {matches}",
            "Output words:",
            f"  Research: {counts.research_total_words}",
            f"  Team A: {counts.team_a_argument_words}",
            f"  Team B: {counts.team_b_argument_words}",
            f"  Synthesis: {counts.synthesis_words}",
            f"  Total: {counts.total_output_words}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        scorecard = score_transcript(load_transcript(args.transcript))
    except (OSError, UnicodeError, TranscriptParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        sys.stdout.write(scorecard_to_json(scorecard))
    else:
        print(format_scorecard(scorecard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
