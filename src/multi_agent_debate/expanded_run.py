"""Offline-testable composition and CLI for one grounded expanded debate."""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from multi_agent_debate.expanded_graph import (
    QuestionDecomposer,
    run_expanded_debate,
)
from multi_agent_debate.mcp_backend_search_server import (
    create_backend_search_server_factory,
)
from multi_agent_debate.mcp_research import MCPResearchError, MCPResearchProvider
from multi_agent_debate.metrics import Scorecard, scorecard_to_json
from multi_agent_debate.model import DebateModel
from multi_agent_debate.ollama_decomposer import (
    OllamaDecompositionError,
    OllamaQuestionDecomposer,
)
from multi_agent_debate.ollama_model import DEFAULT_MODEL_NAME, OllamaDebateModel
from multi_agent_debate.research import ResearchProvider
from multi_agent_debate.run_record import (
    ArtifactExistsError,
    ArtifactPublicationError,
    RunRecord,
    RunRecordError,
    RunRecordValidationError,
    assert_credential_absent,
    build_run_record,
    format_utc_timestamp,
    normalize_implementation_commit,
    preflight_output_paths,
    publish_serialized_artifacts,
    run_record_to_json,
    sanitized_search_settings,
    score_expanded_result,
)
from multi_agent_debate.run_trace import (
    MonotonicClock,
    RecordingSearchBackend,
    RunTraceRecorder,
    TimedDebateModel,
    TimedQuestionDecomposer,
    TimedResearchProvider,
    canonicalize_completed_traces,
)
from multi_agent_debate.search_backend import SearchBackendConfigurationError
from multi_agent_debate.tavily_search import (
    TAVILY_API_KEY_ENV,
    TavilySearchBackend,
)


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    decomposer: QuestionDecomposer
    research_provider: ResearchProvider
    debate_model: DebateModel
    recorder: RunTraceRecorder


@dataclass(frozen=True, slots=True)
class ExpandedRunExecution:
    run_record: RunRecord
    scorecard: Scorecard
    scorecard_json: str


def build_live_dependencies(
    *,
    model_name: str,
    max_results: int,
    environ: Mapping[str, str] | None = None,
    clock: MonotonicClock = time.perf_counter,
) -> tuple[LiveDependencies, str]:
    """Read live configuration once and compose the frozen production path."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise SearchBackendConfigurationError("model name must be nonempty")
    if type(max_results) is not int or max_results <= 0:
        raise SearchBackendConfigurationError(
            "max_results must be a positive integer"
        )

    source = os.environ if environ is None else environ
    api_key = source.get(TAVILY_API_KEY_ENV, "")
    if not isinstance(api_key, str) or not api_key.strip():
        raise SearchBackendConfigurationError(
            "Tavily API key must be a nonempty string"
        )

    backend = TavilySearchBackend.from_environment(
        environ={TAVILY_API_KEY_ENV: api_key}
    )
    recorder = RunTraceRecorder()
    recording_backend = RecordingSearchBackend(
        backend,
        recorder,
        settings=sanitized_search_settings(),
        clock=clock,
    )
    server_factory = create_backend_search_server_factory(recording_backend)
    research_provider = MCPResearchProvider(
        server_factory,
        max_results=max_results,
    )
    decomposer = OllamaQuestionDecomposer(model_name=model_name)
    debate_model = OllamaDebateModel(model_name=model_name)
    return (
        LiveDependencies(
            decomposer=decomposer,
            research_provider=research_provider,
            debate_model=debate_model,
            recorder=recorder,
        ),
        api_key,
    )


def execute_expanded_run(
    *,
    question: str,
    model_name: str,
    max_results: int,
    implementation_commit: str,
    decomposer: QuestionDecomposer,
    research_provider: ResearchProvider,
    debate_model: DebateModel,
    recorder: RunTraceRecorder,
    clock: MonotonicClock = time.perf_counter,
    utc_timestamp_provider: Callable[[], datetime] = lambda: datetime.now(
        timezone.utc
    ),
) -> ExpandedRunExecution:
    """Run, trace, score, construct, and validate without live construction."""

    if not isinstance(question, str) or not question.strip():
        raise RunRecordValidationError("question must be a nonempty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise RunRecordValidationError("model name must be a nonempty string")
    if type(max_results) is not int or max_results <= 0:
        raise RunRecordValidationError("max_results must be a positive integer")
    commit = normalize_implementation_commit(implementation_commit)
    run_timestamp = format_utc_timestamp(utc_timestamp_provider())

    timed_decomposer = TimedQuestionDecomposer(
        decomposer,
        recorder,
        clock=clock,
    )
    timed_provider = TimedResearchProvider(
        research_provider,
        recorder,
        clock=clock,
    )
    timed_model = TimedDebateModel(
        debate_model,
        recorder,
        clock=clock,
    )

    started = clock()
    result = run_expanded_debate(
        question,
        timed_decomposer,
        timed_provider,
        timed_model,
    )
    traces = canonicalize_completed_traces(result, recorder)
    scorecard = score_expanded_result(result)
    total_duration = float(clock() - started)
    record = build_run_record(
        result=result,
        traces=traces,
        scorecard=scorecard,
        run_timestamp_utc=run_timestamp,
        implementation_commit=commit,
        model_name=model_name,
        max_results=max_results,
        total_duration_seconds=total_duration,
    )
    return ExpandedRunExecution(
        run_record=record,
        scorecard=scorecard,
        scorecard_json=scorecard_to_json(scorecard),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one grounded expanded debate and save structured evidence."
    )
    parser.add_argument("question", help="A genuinely contested question.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Name of an already-installed Ollama model.",
    )
    parser.add_argument(
        "--implementation-commit",
        required=True,
        type=_commit_argument,
        help="Committed implementation SHA (exactly 40 hexadecimal characters).",
    )
    parser.add_argument(
        "--max-results",
        type=_positive_int_argument,
        default=3,
        help="Maximum Tavily results per research question. Defaults to 3.",
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
    clock: MonotonicClock = time.perf_counter,
    utc_timestamp_provider: Callable[[], datetime] = lambda: datetime.now(
        timezone.utc
    ),
) -> int:
    args = _parse_args(argv)
    try:
        if not args.question.strip() or not args.model.strip():
            raise RunRecordValidationError(
                "question and model must be nonempty strings"
            )
        preflight_output_paths(
            args.output,
            args.score_output,
            overwrite=args.overwrite,
        )
        dependencies, api_key = build_live_dependencies(
            model_name=args.model,
            max_results=args.max_results,
            environ=environ,
            clock=clock,
        )
        execution = execute_expanded_run(
            question=args.question,
            model_name=args.model,
            max_results=args.max_results,
            implementation_commit=args.implementation_commit,
            decomposer=dependencies.decomposer,
            research_provider=dependencies.research_provider,
            debate_model=dependencies.debate_model,
            recorder=dependencies.recorder,
            clock=clock,
            utc_timestamp_provider=utc_timestamp_provider,
        )
        run_json = run_record_to_json(execution.run_record)
        assert_credential_absent(
            execution.run_record,
            run_json,
            api_key=api_key,
        )
        assert_credential_absent(
            json.loads(execution.scorecard_json),
            execution.scorecard_json,
            api_key=api_key,
        )
        publish_serialized_artifacts(
            run_output=args.output,
            run_json=run_json,
            score_output=args.score_output,
            scorecard_json=(
                execution.scorecard_json if args.score_output is not None else None
            ),
            overwrite=args.overwrite,
        )
    except (ArtifactExistsError, SearchBackendConfigurationError):
        print("error: live run configuration is invalid", file=sys.stderr)
        return 2
    except (RunRecordError, OllamaDecompositionError, MCPResearchError):
        print("error: grounded debate execution failed", file=sys.stderr)
        return 1
    except Exception:
        print("error: grounded debate execution failed", file=sys.stderr)
        return 1

    print(f"Run artifact: {args.output}")
    if args.score_output is not None:
        print(f"Scorecard: {args.score_output}")
    if args.summary:
        _print_summary(execution)
    return 0


def _commit_argument(value: str) -> str:
    try:
        return normalize_implementation_commit(value)
    except RunRecordValidationError:
        raise argparse.ArgumentTypeError(
            "must be exactly 40 hexadecimal characters"
        ) from None


def _positive_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _print_summary(execution: ExpandedRunExecution) -> None:
    record = execution.run_record
    metrics = record["metrics"]
    print(f"Question complexity: {record['decomposition']['question_complexity']}")
    print(f"Subquestions: {len(record['decomposition']['subquestions'])}")
    print(
        "Grounded sources: "
        f"{sum(len(call['results']) for call in record['search_evidence'])}"
    )
    print(
        "Research utilization: "
        f"{metrics['research_utilization']['score']:.6f}"
    )
    print(f"Team novelty: {metrics['team_novelty']['score']:.6f}")
    print(
        "Synthesis representation: "
        f"{metrics['synthesis_representation']['score']:.6f}"
    )
    print(
        "Summary-policy compliance: "
        f"{metrics['summary_policy_compliance']['score']}"
    )
    print(f"Total duration: {record['timings']['total_duration_seconds']:.6f}s")


if __name__ == "__main__":
    raise SystemExit(main())
