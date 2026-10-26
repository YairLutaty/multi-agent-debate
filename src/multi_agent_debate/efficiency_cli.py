"""CLI for the controlled three-architecture cost and efficiency replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from multi_agent_debate.costs import CostValidationError, Pricing, validate_price
from multi_agent_debate.efficiency_replay import (
    EfficiencyReplayArtifacts,
    EfficiencyReplayError,
    execute_efficiency_replay,
)
from multi_agent_debate.efficiency_report import EfficiencyReportError
from multi_agent_debate.model import DebateModel
from multi_agent_debate.ollama_model import DEFAULT_MODEL_NAME, OllamaDebateModel
from multi_agent_debate.ollama_team_model import OllamaTeamModel
from multi_agent_debate.replay import ReplayError, ReplaySource, load_replay_source
from multi_agent_debate.run_record import normalize_implementation_commit
from multi_agent_debate.team_model import NeutralSynthesisModel, TeamModel
from multi_agent_debate.usage import UsageRecorder


class EfficiencyCLIError(RuntimeError):
    """Base bounded efficiency CLI error."""


class EfficiencyCLIArgumentError(EfficiencyCLIError):
    """Efficiency CLI arguments are invalid."""


class EfficiencyDependencyError(EfficiencyCLIError):
    """Ollama efficiency dependencies could not be initialized."""


class EfficiencyArtifactError(EfficiencyCLIError):
    """Efficiency artifact publication failed."""


class EfficiencyArtifactExistsError(EfficiencyArtifactError):
    """The requested output exists and overwrite is disabled."""


@dataclass(frozen=True, slots=True)
class EfficiencyDependencies:
    single_round_model: DebateModel
    two_round_model: DebateModel
    hierarchical_team_model: TeamModel
    hierarchical_synthesis_model: NeutralSynthesisModel
    single_round_usage: UsageRecorder
    two_round_usage: UsageRecorder
    hierarchical_usage: UsageRecorder


def build_efficiency_dependencies(
    *,
    model_name: str,
    clock: Callable[[], float] = time.perf_counter,
) -> EfficiencyDependencies:
    """Build only local-model adapters; replay performs no live research."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise EfficiencyDependencyError("efficiency model configuration is invalid")
    try:
        single_usage = UsageRecorder()
        two_usage = UsageRecorder()
        hierarchical_usage = UsageRecorder()
        single_model = OllamaDebateModel(
            model_name=model_name,
            usage_recorder=single_usage,
            usage_variant="single_round_one_shot",
            clock=clock,
        )
        two_model = OllamaDebateModel(
            model_name=model_name,
            usage_recorder=two_usage,
            usage_variant="two_round_one_shot",
            clock=clock,
        )
        team_model = OllamaTeamModel(
            model_name=model_name,
            usage_recorder=hierarchical_usage,
            usage_variant="hierarchical_single_round",
            clock=clock,
        )
        synthesis_model = OllamaDebateModel(
            model_name=model_name,
            usage_recorder=hierarchical_usage,
            usage_variant="hierarchical_single_round",
            clock=clock,
        )
    except Exception:
        raise EfficiencyDependencyError(
            "efficiency model dependencies could not be initialized"
        ) from None
    return EfficiencyDependencies(
        single_round_model=single_model,
        two_round_model=two_model,
        hierarchical_team_model=team_model,
        hierarchical_synthesis_model=synthesis_model,
        single_round_usage=single_usage,
        two_round_usage=two_usage,
        hierarchical_usage=hierarchical_usage,
    )


def preflight_efficiency_output(output: Path, *, overwrite: bool) -> None:
    if not isinstance(output, Path):
        raise EfficiencyArtifactError("efficiency output path is invalid")
    if output.exists() and output.is_dir():
        raise EfficiencyArtifactError("efficiency output path is a directory")
    if output.exists() and not overwrite:
        raise EfficiencyArtifactExistsError("efficiency output already exists")


def publish_efficiency_report(
    *,
    output: Path,
    report_json: str,
    overwrite: bool,
) -> None:
    _validate_serialized_report(report_json)
    preflight_efficiency_output(output, overwrite=overwrite)
    try:
        original = output.read_bytes() if output.exists() else None
    except OSError:
        raise EfficiencyArtifactError("efficiency artifact publication failed") from None
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(report_json)
            handle.flush()
            os.fsync(handle.fileno())
        preflight_efficiency_output(output, overwrite=overwrite)
        os.replace(temporary, output)
        temporary = None
    except Exception:
        try:
            if original is None:
                output.unlink(missing_ok=True)
            elif not output.exists() or output.read_bytes() != original:
                _restore_bytes(output, original)
        except Exception:
            raise EfficiencyArtifactError(
                "efficiency artifact publication and restoration failed"
            ) from None
        raise EfficiencyArtifactError("efficiency artifact publication failed") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class _BoundedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise EfficiencyCLIArgumentError("efficiency command arguments are invalid")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _BoundedArgumentParser(
        description="Compare token and cost efficiency on frozen replay evidence."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True, type=_sha256_argument)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, type=_nonblank_argument)
    parser.add_argument(
        "--implementation-commit", required=True, type=_commit_argument
    )
    parser.add_argument(
        "--input-cost-per-million", type=_price_argument, default=Decimal("0")
    )
    parser.add_argument(
        "--output-cost-per-million", type=_price_argument, default=Decimal("0")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    dependency_builder: Callable[..., EfficiencyDependencies] = build_efficiency_dependencies,
    executor: Callable[..., EfficiencyReplayArtifacts] = execute_efficiency_replay,
    publisher: Callable[..., None] = publish_efficiency_report,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    try:
        args = _parse_args(argv)
        pricing = Pricing(
            args.input_cost_per_million,
            args.output_cost_per_million,
        )
        preflight_efficiency_output(args.output, overwrite=args.overwrite)
        if args.source.resolve() == args.output.resolve():
            raise EfficiencyCLIArgumentError(
                "replay source and efficiency output must differ"
            )
        replay_source = load_replay_source(
            args.source,
            expected_sha256=args.source_sha256,
        )
    except (
        EfficiencyCLIArgumentError,
        EfficiencyArtifactError,
        CostValidationError,
        ReplayError,
        OSError,
    ):
        print("error: efficiency replay configuration is invalid", file=sys.stderr)
        return 2

    try:
        dependencies = dependency_builder(model_name=args.model, clock=clock)
        artifacts = executor(
            replay_source=replay_source,
            single_round_model=dependencies.single_round_model,
            two_round_model=dependencies.two_round_model,
            hierarchical_team_model=dependencies.hierarchical_team_model,
            hierarchical_synthesis_model=dependencies.hierarchical_synthesis_model,
            single_round_usage=dependencies.single_round_usage,
            two_round_usage=dependencies.two_round_usage,
            hierarchical_usage=dependencies.hierarchical_usage,
            pricing=pricing,
            clock=clock,
            utc_now=utc_now,
            implementation_commit=args.implementation_commit,
            model_name=args.model,
        )
        publisher(
            output=args.output,
            report_json=artifacts["report_json"],
            overwrite=args.overwrite,
        )
    except EfficiencyDependencyError:
        print("error: efficiency model initialization failed", file=sys.stderr)
        return 1
    except EfficiencyArtifactError:
        print("error: efficiency artifact publication failed", file=sys.stderr)
        return 1
    except (EfficiencyReplayError, EfficiencyReportError):
        print("error: efficiency replay execution failed", file=sys.stderr)
        return 1
    except Exception:
        print("error: efficiency replay execution failed", file=sys.stderr)
        return 1

    if args.summary:
        _print_summary(artifacts, output=args.output)
    return 0


def _print_summary(artifacts: EfficiencyReplayArtifacts, *, output: Path) -> None:
    for variant in artifacts["report"]["variants"]:
        efficiency = variant["efficiency"]
        cost = variant["cost_estimate"]
        metrics = variant["metrics"]
        print(f"Variant: {variant['variant_id']}")
        print(f"  Model calls: {efficiency['model_calls']}")
        print(f"  Input tokens: {_display(efficiency['input_tokens'])}")
        print(f"  Output tokens: {_display(efficiency['output_tokens'])}")
        print(f"  Total tokens: {_display(efficiency['total_tokens'])}")
        print(f"  Latency: {efficiency['latency_seconds']:.6f}s")
        print(f"  Estimated cost: {_display(cost['estimated_cost'])}")
        print(
            "  Research utilization: "
            f"{metrics['research_utilization']['score']:.6f}"
        )
        print(
            "  Synthesis representation: "
            f"{metrics['synthesis_representation']['score']:.6f}"
        )
    print(f"Efficiency artifact: {output}")


def _sha256_argument(value: str) -> str:
    lowered = value.casefold() if isinstance(value, str) else ""
    if len(lowered) != 64 or any(character not in "0123456789abcdef" for character in lowered):
        raise argparse.ArgumentTypeError("must be exactly 64 hexadecimal characters")
    return lowered


def _commit_argument(value: str) -> str:
    try:
        return normalize_implementation_commit(value)
    except Exception:
        raise argparse.ArgumentTypeError(
            "must be exactly 40 hexadecimal characters"
        ) from None


def _price_argument(value: str) -> Decimal:
    try:
        if not isinstance(value, str) or not value.strip() or value.casefold() in {
            "true",
            "false",
        }:
            raise CostValidationError
        return validate_price(Decimal(value))
    except (InvalidOperation, CostValidationError, ValueError):
        raise argparse.ArgumentTypeError(
            "must be a finite nonnegative number"
        ) from None


def _nonblank_argument(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("must be nonblank")
    return value


def _validate_serialized_report(value: object) -> None:
    if not isinstance(value, str) or not value.endswith("\n") or value.endswith("\n\n"):
        raise EfficiencyArtifactError("serialized efficiency report is invalid")
    try:
        json.loads(value, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (TypeError, ValueError, json.JSONDecodeError):
        raise EfficiencyArtifactError("serialized efficiency report is invalid") from None


def _restore_bytes(output: Path, original: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.restore.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    except Exception:
        raise EfficiencyArtifactError("efficiency artifact restoration failed") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _display(value: object) -> str:
    return "unavailable" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
