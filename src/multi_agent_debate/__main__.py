"""Command-line entry point for the local-model debate pipeline."""

import argparse
from collections.abc import Sequence

from multi_agent_debate.graph import run_debate
from multi_agent_debate.ollama_model import DEFAULT_MODEL_NAME, OllamaDebateModel
from multi_agent_debate.state import DebateResult


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-round debate pipeline with a local Ollama model."
    )
    parser.add_argument("question", help="A genuinely contested question.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Name of an already-installed Ollama model.",
    )
    return parser.parse_args(argv)


def _print_result(result: DebateResult) -> None:
    print("Question:")
    print(result["question"])
    print("\nTeam A stance:")
    print(result["team_a_stance"])
    print("\nTeam B stance:")
    print(result["team_b_stance"])
    print("\nSimulated research findings:")
    for index, finding in enumerate(result["research_findings"], start=1):
        print(f"{index}. {finding}")
    print("\nTeam A argument:")
    print(result["team_a_argument"])
    print("\nTeam B argument:")
    print(result["team_b_argument"])
    print("\nNeutral synthesis:")
    print(result["synthesis"])


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model = OllamaDebateModel(model_name=args.model)
    result = run_debate(args.question, model)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
