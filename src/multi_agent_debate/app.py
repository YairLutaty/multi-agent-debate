"""Final single-command product for grounded one- or multi-round debates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from multi_agent_debate.expanded_graph import QuestionDecomposer
from multi_agent_debate.costs import CostValidationError, Pricing, validate_price
from multi_agent_debate.mcp_backend_search_server import (
    create_backend_search_server_factory,
)
from multi_agent_debate.mcp_research import MCPResearchError, MCPResearchProvider
from multi_agent_debate.metrics import Scorecard, scorecard_to_json
from multi_agent_debate.model import DebateModel
from multi_agent_debate.multi_round_graph import run_multi_round_debate
from multi_agent_debate.multi_round_run_record import (
    MultiRoundRunRecord,
    MultiRoundRunRecordV21,
    MultiRoundRunRecordError,
    build_multi_round_run_record,
    canonicalize_grounding_traces,
    canonicalize_product_debate_traces,
    multi_round_run_record_to_json,
    score_multi_round_result,
)
from multi_agent_debate.multi_round_trace import (
    MonotonicClock,
    MultiRoundTraceRecorder,
    clock_value,
    elapsed_seconds,
)
from multi_agent_debate.ollama_decomposer import (
    OllamaDecompositionError,
    OllamaQuestionDecomposer,
)
from multi_agent_debate.ollama_model import DEFAULT_MODEL_NAME, OllamaDebateModel
from multi_agent_debate.research import ResearchProvider
from multi_agent_debate.run_record import (
    ArtifactExistsError,
    ArtifactPublicationError,
    RunRecordError,
    assert_credential_absent,
    format_utc_timestamp,
    normalize_implementation_commit,
    preflight_output_paths,
    publish_serialized_artifacts,
    sanitized_search_settings,
)
from multi_agent_debate.run_trace import (
    RecordingSearchBackend,
    RunTraceRecorder,
    TimedQuestionDecomposer,
    TimedResearchProvider,
)
from multi_agent_debate.tavily_search import TAVILY_API_KEY_ENV, TavilySearchBackend
from multi_agent_debate.termination import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NOVELTY_THRESHOLD,
    MultiRoundError,
    MultiRoundValidationError,
    validate_max_rounds,
    validate_novelty_threshold,
)
from multi_agent_debate.usage import UsageRecorder

DEFAULT_MAX_RESULTS = 3


class ProductCLIError(RuntimeError):
    """Base bounded product-composition error."""


class ProductArgumentError(ProductCLIError):
    """Command arguments violate the final product contract."""


class ProductDependencyError(ProductCLIError):
    """Live product dependencies could not be constructed safely."""


@dataclass(frozen=True, slots=True)
class ProductLiveDependencies:
    decomposer: QuestionDecomposer
    research_provider: ResearchProvider
    debate_model: DebateModel
    grounding_recorder: RunTraceRecorder
    debate_recorder: MultiRoundTraceRecorder
    usage_recorder: UsageRecorder | None = None


@dataclass(frozen=True, slots=True)
class ProductExecution:
    run_record: MultiRoundRunRecord | MultiRoundRunRecordV21
    scorecard: Scorecard
    scorecard_json: str


def build_live_dependencies(
    *,
    model_name: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    environ: Mapping[str, str] | None = None,
    clock: MonotonicClock = time.perf_counter,
) -> tuple[ProductLiveDependencies, str]:
    """Read the Tavily key first, then construct the existing live adapters."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ProductDependencyError("product model configuration is invalid")
    if type(max_results) is not int or max_results <= 0:
        raise ProductDependencyError("product search configuration is invalid")
    source = os.environ if environ is None else environ
    api_key = source.get(TAVILY_API_KEY_ENV, "")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ProductDependencyError("Tavily configuration is missing")

    try:
        backend = TavilySearchBackend.from_environment(
            environ={TAVILY_API_KEY_ENV: api_key}
        )
        grounding_recorder = RunTraceRecorder()
        recording_backend = RecordingSearchBackend(
            backend,
            grounding_recorder,
            settings=sanitized_search_settings(),
            clock=clock,
        )
        server_factory = create_backend_search_server_factory(recording_backend)
        research_provider = MCPResearchProvider(
            server_factory,
            max_results=max_results,
        )
        usage_recorder = UsageRecorder()
        decomposer = OllamaQuestionDecomposer(
            model_name=model_name,
            usage_recorder=usage_recorder,
            usage_variant="final_product",
            clock=clock,
        )
        debate_model = OllamaDebateModel(
            model_name=model_name,
            usage_recorder=usage_recorder,
            usage_variant="final_product",
            clock=clock,
        )
    except Exception:
        raise ProductDependencyError(
            "live product dependencies could not be initialized"
        ) from None
    return (
        ProductLiveDependencies(
            decomposer=decomposer,
            research_provider=research_provider,
            debate_model=debate_model,
            grounding_recorder=grounding_recorder,
            debate_recorder=MultiRoundTraceRecorder(),
            usage_recorder=usage_recorder,
        ),
        api_key,
    )


def execute_product_run(
    *,
    question: str,
    mode: str,
    model_name: str,
    max_rounds: int,
    novelty_threshold: float,
    max_results: int,
    implementation_commit: str,
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    debate_model: DebateModel,
    grounding_recorder: RunTraceRecorder,
    debate_recorder: MultiRoundTraceRecorder,
    clock: MonotonicClock = time.perf_counter,
    utc_timestamp_provider: Callable[[], datetime] = lambda: datetime.now(
        timezone.utc
    ),
    usage_recorder: UsageRecorder | None = None,
    pricing: Pricing | None = None,
) -> ProductExecution:
    """Pure provider-injected execution, scoring, and record construction."""

    if not isinstance(question, str) or not question.strip():
        raise MultiRoundValidationError("question must be a nonempty string")
    if mode not in ("single-round", "multi-round"):
        raise MultiRoundValidationError("product mode is invalid")
    rounds = validate_max_rounds(max_rounds)
    if mode == "single-round" and rounds != 1:
        raise MultiRoundValidationError("single-round mode requires one round")
    threshold = validate_novelty_threshold(novelty_threshold)
    if not isinstance(model_name, str) or not model_name.strip():
        raise MultiRoundValidationError("model name must be nonempty")
    if type(max_results) is not int or max_results <= 0:
        raise MultiRoundValidationError("search maximum must be positive")
    try:
        commit = normalize_implementation_commit(implementation_commit)
        timestamp = format_utc_timestamp(utc_timestamp_provider())
    except RunRecordError:
        raise MultiRoundValidationError("product provenance is invalid") from None
    except Exception:
        raise MultiRoundValidationError("product provenance is invalid") from None

    timed_decomposer = TimedQuestionDecomposer(
        decomposer, grounding_recorder, clock=clock
    )
    timed_research = TimedResearchProvider(
        research_provider, grounding_recorder, clock=clock
    )
    started = clock_value(clock)
    result = run_multi_round_debate(
        question,
        timed_decomposer,
        timed_research,
        debate_model,
        max_rounds=rounds,
        novelty_threshold=threshold,
        trace_recorder=debate_recorder,
        clock=clock,
    )
    grounding = canonicalize_grounding_traces(result, grounding_recorder)
    debate_traces = canonicalize_product_debate_traces(
        recorder=debate_recorder,
        completed_rounds=len(result["debate_rounds"]),
    )
    scorecard = score_multi_round_result(result)
    total_duration = elapsed_seconds(clock, started)
    record = build_multi_round_run_record(
        result=result,
        grounding_traces=grounding,
        debate_traces=debate_traces,
        scorecard=scorecard,
        run_timestamp_utc=timestamp,
        implementation_commit=commit,
        model_name=model_name,
        max_results=max_results,
        product_mode=mode,
        total_duration_seconds=total_duration,
        usage_entries=(usage_recorder.snapshot() if usage_recorder is not None else None),
        pricing=pricing,
    )
    return ProductExecution(
        run_record=record,
        scorecard=scorecard,
        scorecard_json=scorecard_to_json(scorecard),
    )


class _BoundedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ProductArgumentError("product command arguments are invalid")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _BoundedArgumentParser(
        description="Run the final grounded bounded multi-round debate product."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--mode",
        choices=("single-round", "multi-round"),
        default="multi-round",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-rounds", type=_round_count_argument, default=None)
    parser.add_argument(
        "--novelty-threshold",
        type=_threshold_argument,
        default=DEFAULT_NOVELTY_THRESHOLD,
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=_price_argument,
        default=Decimal("0"),
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=_price_argument,
        default=Decimal("0"),
    )
    parser.add_argument(
        "--implementation-commit",
        required=True,
        type=_commit_argument,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    dependency_builder: Callable[
        ..., tuple[ProductLiveDependencies, str]
    ] = build_live_dependencies,
    executor: Callable[..., ProductExecution] = execute_product_run,
    publisher: Callable[..., None] = publish_serialized_artifacts,
    clock: MonotonicClock = time.perf_counter,
    utc_timestamp_provider: Callable[[], datetime] = lambda: datetime.now(
        timezone.utc
    ),
) -> int:
    try:
        args = _parse_args(argv)
        if not args.question.strip() or not args.model.strip():
            raise ProductArgumentError("question and model must be nonempty")
        explicit_rounds = args.max_rounds
        if args.mode == "single-round":
            if explicit_rounds is not None and explicit_rounds != 1:
                raise ProductArgumentError(
                    "single-round mode cannot use multiple rounds"
                )
            max_rounds = 1
        else:
            max_rounds = (
                DEFAULT_MAX_ROUNDS if explicit_rounds is None else explicit_rounds
            )
        pricing = Pricing(
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        )
        preflight_output_paths(
            args.output,
            args.score_output,
            overwrite=args.overwrite,
        )
    except (
        ProductArgumentError,
        ArtifactExistsError,
        ArtifactPublicationError,
        CostValidationError,
    ):
        print("error: product configuration is invalid", file=sys.stderr)
        return 2

    try:
        dependencies, api_key = dependency_builder(
            model_name=args.model,
            max_results=DEFAULT_MAX_RESULTS,
            environ=environ,
            clock=clock,
        )
        execution = executor(
            question=args.question,
            mode=args.mode,
            model_name=args.model,
            max_rounds=max_rounds,
            novelty_threshold=args.novelty_threshold,
            max_results=DEFAULT_MAX_RESULTS,
            implementation_commit=args.implementation_commit,
            decomposer=dependencies.decomposer,
            research_provider=dependencies.research_provider,
            debate_model=dependencies.debate_model,
            grounding_recorder=dependencies.grounding_recorder,
            debate_recorder=dependencies.debate_recorder,
            clock=clock,
            utc_timestamp_provider=utc_timestamp_provider,
            usage_recorder=dependencies.usage_recorder,
            pricing=(pricing if dependencies.usage_recorder is not None else None),
        )
        run_json = multi_round_run_record_to_json(execution.run_record)
        assert_credential_absent(execution.run_record, run_json, api_key=api_key)
        assert_credential_absent(
            json.loads(execution.scorecard_json),
            execution.scorecard_json,
            api_key=api_key,
        )
        publisher(
            run_output=args.output,
            run_json=run_json,
            score_output=args.score_output,
            scorecard_json=(
                execution.scorecard_json if args.score_output is not None else None
            ),
            overwrite=args.overwrite,
        )
    except ProductDependencyError:
        print("error: live product configuration is invalid", file=sys.stderr)
        return 2
    except (
        MultiRoundError,
        MultiRoundRunRecordError,
        RunRecordError,
        OllamaDecompositionError,
        MCPResearchError,
        ArtifactPublicationError,
    ):
        print("error: grounded debate product execution failed", file=sys.stderr)
        return 1
    except Exception:
        print("error: grounded debate product execution failed", file=sys.stderr)
        return 1

    if args.summary:
        _print_summary(
            execution,
            run_output=args.output,
            score_output=args.score_output,
        )
    return 0


def _round_count_argument(value: str) -> int:
    try:
        return validate_max_rounds(int(value))
    except Exception:
        raise argparse.ArgumentTypeError("must be an integer from 1 to 3") from None


def _threshold_argument(value: str) -> float:
    try:
        return validate_novelty_threshold(float(value))
    except Exception:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1") from None


def _commit_argument(value: str) -> str:
    try:
        return normalize_implementation_commit(value)
    except Exception:
        raise argparse.ArgumentTypeError(
            "must be exactly 40 hexadecimal characters"
        ) from None


def _price_argument(value: str) -> Decimal:
    try:
        if not isinstance(value, str) or not value.strip():
            raise CostValidationError
        lowered = value.strip().casefold()
        if lowered in {"true", "false"}:
            raise CostValidationError
        return validate_price(Decimal(value))
    except (InvalidOperation, CostValidationError, ValueError):
        raise argparse.ArgumentTypeError(
            "must be a finite nonnegative number"
        ) from None


def _print_summary(
    execution: ProductExecution,
    *,
    run_output: Path,
    score_output: Path | None,
) -> None:
    record = execution.run_record
    metrics = record["metrics"]
    print(f"Mode: {record['provenance']['product_mode']}")
    print(f"Complexity: {record['decomposition']['question_complexity']}")
    print(
        "Source count: "
        f"{sum(len(call['results']) for call in record['search_evidence'])}"
    )
    print(f"Completed rounds: {len(record['debate_rounds'])}")
    print(f"Termination reason: {record['termination']['reason']}")
    for round_record in record["debate_rounds"]:
        print(
            f"Round {round_record['round_number']} novelty: "
            f"Team A={round_record['team_a_novelty']}, "
            f"Team B={round_record['team_b_novelty']}"
        )
    print(
        "Metrics: "
        f"research_utilization={metrics['research_utilization']['score']:.6f}, "
        f"team_novelty={metrics['team_novelty']['score']:.6f}, "
        "synthesis_representation="
        f"{metrics['synthesis_representation']['score']:.6f}, "
        "summary_policy_compliance="
        f"{metrics['summary_policy_compliance']['score']}"
    )
    print(f"Total latency: {record['timings']['total_duration_seconds']:.6f}s")
    print(f"Model calls: {record['model_call_counts']['total_model_calls']}")
    if record["run_record_version"] == "2.1":
        usage = record["usage"]["totals"]
        cost = record["cost_estimate"]
        if usage["all_usage_available"]:
            print(f"Input tokens: {usage['input_tokens']}")
            print(f"Output tokens: {usage['output_tokens']}")
            print(f"Total tokens: {usage['total_tokens']}")
        else:
            print("Input tokens: unavailable")
            print("Output tokens: unavailable")
            print("Total tokens: unavailable")
        print(
            "Estimated cost: "
            f"{cost['estimated_cost'] if cost['status'] == 'available' else 'unavailable'}"
        )
        print(f"Pricing basis: {cost['pricing_basis']}")
    print(f"Run artifact: {run_output}")
    if score_output is not None:
        print(f"Scorecard: {score_output}")


if __name__ == "__main__":
    raise SystemExit(main())
