"""Paired frozen-evidence replay composition and command-line interface."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast

from multi_agent_debate.model import DebateModel
from multi_agent_debate.ollama_model import DEFAULT_MODEL_NAME, OllamaDebateModel
from multi_agent_debate.ollama_team_model import OllamaTeamModel
from multi_agent_debate.replay import (
    ReplayError,
    ReplaySource,
    load_replay_source,
    validate_replay_source,
)
from multi_agent_debate.replay_artifacts import (
    ReplayArtifactError,
    ReplayArtifactExistsError,
    ReplayArtifactPaths,
    preflight_replay_artifacts,
    publish_replay_artifacts,
)
from multi_agent_debate.replay_run_record import ReplayRunRecordError
from multi_agent_debate.replay_runner import (
    ReplayArtifacts,
    ReplayRunnerError,
    run_replay_variant,
)
from multi_agent_debate.run_record import (
    ScorecardData,
    normalize_implementation_commit,
)
from multi_agent_debate.team_model import (
    NeutralSynthesisModel,
    TeamModel,
    TeamModelError,
)
from multi_agent_debate.team_trace import TeamTraceRecorder


class ReplayCLIError(RuntimeError):
    """Base error for paired replay composition and CLI validation."""


class ReplayCLIArgumentError(ReplayCLIError):
    """Command-line arguments violate the fixed replay contract."""


class ReplayDependencyError(ReplayCLIError):
    """The local model adapters could not be safely constructed."""


class ReplayPairValidationError(ReplayCLIError):
    """The two successful variants do not form one controlled pair."""


class PairedReplayArtifacts(TypedDict):
    control: ReplayArtifacts
    hierarchical: ReplayArtifacts


class MetricDeltas(TypedDict):
    research_utilization: float
    team_novelty: float
    synthesis_representation: float
    summary_policy_compliance: int
    research_total_words: int
    team_a_argument_words: int
    team_b_argument_words: int
    synthesis_words: int
    total_output_words: int


@dataclass(frozen=True, slots=True)
class ReplayDependencies:
    control_model: DebateModel
    team_model: TeamModel
    synthesis_model: NeutralSynthesisModel


_SOURCE_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def build_replay_dependencies(*, model_name: str) -> ReplayDependencies:
    """Construct only the two local-model adapters needed by paired replay."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ReplayDependencyError("replay model name is invalid")
    try:
        debate_model = OllamaDebateModel(model_name=model_name)
        team_model = OllamaTeamModel(model_name=model_name)
    except Exception:
        raise ReplayDependencyError("replay models could not be initialized") from None
    return ReplayDependencies(
        control_model=debate_model,
        team_model=team_model,
        synthesis_model=debate_model,
    )


def execute_paired_replay(
    *,
    replay_source: ReplaySource,
    control_model: DebateModel,
    team_model: TeamModel,
    synthesis_model: NeutralSynthesisModel,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    implementation_commit: str,
    model_name: str,
    publication: Callable[[PairedReplayArtifacts], None] | None = None,
) -> PairedReplayArtifacts:
    """Execute control then treatment, optionally passing the validated pair out."""

    source = validate_replay_source(replay_source)
    if not callable(clock) or not callable(utc_now):
        raise ReplayPairValidationError("replay clocks must be callable")
    try:
        paired_timestamp = utc_now()
    except Exception:
        raise ReplayPairValidationError("replay UTC timestamp failed") from None
    fixed_utc_now = lambda: paired_timestamp

    control = run_replay_variant(
        variant="control",
        replay_source=source,
        control_model=control_model,
        team_model=None,
        synthesis_model=synthesis_model,
        team_trace_recorder=None,
        clock=clock,
        utc_now=fixed_utc_now,
        implementation_commit=implementation_commit,
        model_name=model_name,
    )
    hierarchical = run_replay_variant(
        variant="hierarchical",
        replay_source=source,
        control_model=None,
        team_model=team_model,
        synthesis_model=synthesis_model,
        team_trace_recorder=TeamTraceRecorder(),
        clock=clock,
        utc_now=fixed_utc_now,
        implementation_commit=implementation_commit,
        model_name=model_name,
    )
    pair: PairedReplayArtifacts = {
        "control": copy.deepcopy(control),
        "hierarchical": copy.deepcopy(hierarchical),
    }
    validated = _validate_paired_artifacts(pair, source)
    if publication is not None:
        if not callable(publication):
            raise ReplayPairValidationError(
                "replay publication callback must be callable"
            )
        publication(copy.deepcopy(validated))
    return copy.deepcopy(validated)


def calculate_metric_deltas(pair: PairedReplayArtifacts) -> MetricDeltas:
    """Return raw hierarchical-minus-control metric and word-count deltas."""

    control = cast(ScorecardData, pair["control"]["run_record"]["metrics"])
    hierarchical = cast(
        ScorecardData,
        pair["hierarchical"]["run_record"]["metrics"],
    )
    control_words = control["word_counts"]
    hierarchical_words = hierarchical["word_counts"]
    return {
        "research_utilization": (
            hierarchical["research_utilization"]["score"]
            - control["research_utilization"]["score"]
        ),
        "team_novelty": (
            hierarchical["team_novelty"]["score"]
            - control["team_novelty"]["score"]
        ),
        "synthesis_representation": (
            hierarchical["synthesis_representation"]["score"]
            - control["synthesis_representation"]["score"]
        ),
        "summary_policy_compliance": (
            hierarchical["summary_policy_compliance"]["score"]
            - control["summary_policy_compliance"]["score"]
        ),
        "research_total_words": (
            hierarchical_words["research_total_words"]
            - control_words["research_total_words"]
        ),
        "team_a_argument_words": (
            hierarchical_words["team_a_argument_words"]
            - control_words["team_a_argument_words"]
        ),
        "team_b_argument_words": (
            hierarchical_words["team_b_argument_words"]
            - control_words["team_b_argument_words"]
        ),
        "synthesis_words": (
            hierarchical_words["synthesis_words"]
            - control_words["synthesis_words"]
        ),
        "total_output_words": (
            hierarchical_words["total_output_words"]
            - control_words["total_output_words"]
        ),
    }


class _BoundedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ReplayCLIArgumentError("replay command arguments are invalid")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _BoundedArgumentParser(
        description="Run a controlled one-shot versus hierarchical replay pair."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--source-sha256",
        required=True,
        type=_source_sha_argument,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, type=_nonblank_argument)
    parser.add_argument(
        "--implementation-commit",
        required=True,
        type=_commit_argument,
    )
    parser.add_argument("--control-output", required=True, type=Path)
    parser.add_argument("--control-score-output", required=True, type=Path)
    parser.add_argument("--hierarchical-output", required=True, type=Path)
    parser.add_argument("--hierarchical-score-output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    dependency_builder: Callable[..., ReplayDependencies] = build_replay_dependencies,
    paired_executor: Callable[..., PairedReplayArtifacts] = execute_paired_replay,
    publisher: Callable[..., None] = publish_replay_artifacts,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    try:
        args = _parse_args(argv)
        paths = ReplayArtifactPaths(
            control_score=args.control_score_output,
            control_run=args.control_output,
            hierarchical_score=args.hierarchical_score_output,
            hierarchical_run=args.hierarchical_output,
        )
        preflight_replay_artifacts(paths, overwrite=args.overwrite)
        _reject_source_output_overlap(args.source, paths)
        replay_source = load_replay_source(
            args.source,
            expected_sha256=args.source_sha256,
        )
    except ReplayCLIArgumentError:
        print("error: replay configuration is invalid", file=sys.stderr)
        return 2
    except ReplayArtifactExistsError:
        print("error: replay output configuration is invalid", file=sys.stderr)
        return 2
    except ReplayArtifactError:
        print("error: replay output configuration is invalid", file=sys.stderr)
        return 2
    except ReplayError:
        print("error: replay source validation failed", file=sys.stderr)
        return 2

    try:
        dependencies = dependency_builder(model_name=args.model)

        def publication(pair: PairedReplayArtifacts) -> None:
            publisher(
                paths=paths,
                control_run_json=pair["control"]["run_record_json"],
                control_scorecard_json=pair["control"]["scorecard_json"],
                hierarchical_run_json=pair["hierarchical"]["run_record_json"],
                hierarchical_scorecard_json=pair["hierarchical"][
                    "scorecard_json"
                ],
                overwrite=args.overwrite,
            )

        pair = paired_executor(
            replay_source=replay_source,
            control_model=dependencies.control_model,
            team_model=dependencies.team_model,
            synthesis_model=dependencies.synthesis_model,
            clock=clock,
            utc_now=utc_now,
            implementation_commit=args.implementation_commit,
            model_name=args.model,
            publication=publication,
        )
    except ReplayDependencyError:
        print("error: replay model initialization failed", file=sys.stderr)
        return 1
    except ReplayArtifactError:
        print("error: replay artifact publication failed", file=sys.stderr)
        return 1
    except (TeamModelError, ReplayRunnerError, ReplayRunRecordError, ReplayCLIError):
        print("error: paired replay execution failed", file=sys.stderr)
        return 1
    except Exception:
        print("error: paired replay execution failed", file=sys.stderr)
        return 1

    if args.summary:
        _print_summary(
            pair,
            source_sha256=args.source_sha256,
            model_name=args.model,
            paths=paths,
        )
    return 0


def _validate_paired_artifacts(
    value: PairedReplayArtifacts,
    replay_source: ReplaySource,
) -> PairedReplayArtifacts:
    if not isinstance(value, dict) or set(value) != {"control", "hierarchical"}:
        raise ReplayPairValidationError("paired replay artifacts are invalid")
    control = value["control"]
    hierarchical = value["hierarchical"]
    if control["variant"] != "control" or hierarchical["variant"] != "hierarchical":
        raise ReplayPairValidationError("paired replay execution order is invalid")
    control_record = control["run_record"]
    hierarchical_record = hierarchical["run_record"]
    if control_record["replay_source"] != hierarchical_record["replay_source"]:
        raise ReplayPairValidationError("paired replay source provenance differs")
    for field in (
        "input",
        "decomposition",
        "source_search_evidence",
    ):
        if control_record[field] != hierarchical_record[field]:
            raise ReplayPairValidationError("paired replay inputs differ")
    for field in (
        "raw_research_findings",
        "consolidated_evidence",
        "research_findings",
    ):
        if (
            control_record["graph_evidence"][field]
            != hierarchical_record["graph_evidence"][field]
        ):
            raise ReplayPairValidationError("paired replay evidence differs")
    if control_record["replay_source"]["source_sha256"] != replay_source[
        "source_sha256"
    ]:
        raise ReplayPairValidationError("paired replay source hash differs")
    if control_record["model_call_counts"]["total_model_calls"] != 3:
        raise ReplayPairValidationError("control model-call count is invalid")
    if hierarchical_record["model_call_counts"]["total_model_calls"] != 7:
        raise ReplayPairValidationError("hierarchical model-call count is invalid")
    return copy.deepcopy(value)


def _source_sha_argument(value: str) -> str:
    if _SOURCE_SHA_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "must be exactly 64 hexadecimal characters"
        )
    return value.lower()


def _commit_argument(value: str) -> str:
    try:
        return normalize_implementation_commit(value)
    except Exception:
        raise argparse.ArgumentTypeError(
            "must be exactly 40 hexadecimal characters"
        ) from None


def _nonblank_argument(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be nonblank")
    return value.strip()


def _reject_source_output_overlap(
    source: Path,
    paths: ReplayArtifactPaths,
) -> None:
    try:
        source_path = source.resolve()
        output_paths = {path.resolve() for path in paths.in_publication_order()}
    except OSError:
        raise ReplayCLIArgumentError("replay paths are invalid") from None
    if source_path in output_paths:
        raise ReplayCLIArgumentError("replay source cannot be an output")


def _print_summary(
    pair: PairedReplayArtifacts,
    *,
    source_sha256: str,
    model_name: str,
    paths: ReplayArtifactPaths,
) -> None:
    control = pair["control"]["run_record"]
    hierarchical = pair["hierarchical"]["run_record"]
    control_metrics = control["metrics"]
    hierarchical_metrics = hierarchical["metrics"]
    deltas = calculate_metric_deltas(pair)
    print(f"Source SHA-256: {source_sha256}")
    print(f"Model: {model_name}")
    for label, key in (
        ("Research utilization", "research_utilization"),
        ("Team novelty", "team_novelty"),
        ("Synthesis representation", "synthesis_representation"),
        ("Summary-policy compliance", "summary_policy_compliance"),
    ):
        print(f"Control {label}: {control_metrics[key]['score']}")
        print(f"Hierarchical {label}: {hierarchical_metrics[key]['score']}")
        print(f"Delta {label}: {deltas[key]}")
    print(
        "Operational word-count deltas: "
        + json.dumps(
            {
                key: deltas[key]
                for key in (
                    "research_total_words",
                    "team_a_argument_words",
                    "team_b_argument_words",
                    "synthesis_words",
                    "total_output_words",
                )
            },
            sort_keys=True,
        )
    )
    print(
        "Control latency: "
        f"{control['timings']['total_replay_duration_seconds']}s"
    )
    print(
        "Hierarchical latency: "
        f"{hierarchical['timings']['total_replay_duration_seconds']}s"
    )
    print(
        "Model calls: control="
        f"{control['model_call_counts']['total_model_calls']}, "
        "hierarchical="
        f"{hierarchical['model_call_counts']['total_model_calls']}"
    )
    print(f"Control run output: {paths.control_run}")
    print(f"Control score output: {paths.control_score}")
    print(f"Hierarchical run output: {paths.hierarchical_run}")
    print(f"Hierarchical score output: {paths.hierarchical_score}")


if __name__ == "__main__":
    raise SystemExit(main())
