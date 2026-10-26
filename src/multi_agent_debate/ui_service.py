"""Pure coordination layer between the local UI and frozen debate engines."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict, cast

from multi_agent_debate.app import (
    DEFAULT_MAX_RESULTS,
    build_live_dependencies,
    execute_product_run,
)
from multi_agent_debate.costs import Pricing, calculate_cost, pricing_to_data
from multi_agent_debate.efficiency_report import calculate_efficiency, count_generated_words
from multi_agent_debate.expanded_graph import QuestionDecomposer
from multi_agent_debate.hierarchical_graph import run_hierarchical_debate
from multi_agent_debate.judge import (
    DebateJudge,
    JudgeError,
    JudgeInvocationError,
    JudgeResponseError,
    OllamaDebateJudge,
)
from multi_agent_debate.judge_record import JudgeRecordValidationError, build_judge_record
from multi_agent_debate.mcp_backend_search_server import create_backend_search_server_factory
from multi_agent_debate.mcp_research import MCPResearchProvider
from multi_agent_debate.metrics import scorecard_to_json
from multi_agent_debate.model import DebateModel
from multi_agent_debate.multi_round_state import DebateRound
from multi_agent_debate.multi_round_trace import MultiRoundTraceRecorder
from multi_agent_debate.ollama_decomposer import OllamaQuestionDecomposer
from multi_agent_debate.ollama_model import OllamaDebateModel
from multi_agent_debate.ollama_team_model import OllamaTeamModel
from multi_agent_debate.research import ResearchProvider
from multi_agent_debate.run_record import (
    format_utc_timestamp,
    normalize_implementation_commit,
    sanitized_search_settings,
    score_expanded_result,
)
from multi_agent_debate.run_trace import (
    RecordingSearchBackend,
    RunTraceRecorder,
    TimedQuestionDecomposer,
    TimedResearchProvider,
)
from multi_agent_debate.tavily_search import TAVILY_API_KEY_ENV, TavilySearchBackend
from multi_agent_debate.team_model import NeutralSynthesisModel, TeamModel
from multi_agent_debate.team_trace import (
    TeamTraceRecorder,
    canonicalize_successful_team_traces,
)
from multi_agent_debate.termination import (
    validate_max_rounds,
    validate_novelty_threshold,
)
from multi_agent_debate.ui_result import (
    UI_DEBATE_RESULT_VERSION,
    UiDebateResult,
    UiMode,
    UiResult,
    UiJudgmentErrorCategory,
    build_ui_result,
    calculate_ui_generated_words,
    judge_material_from_debate,
    public_debate_from_product_record,
    ui_result_to_json,
    validate_ui_debate_result,
    validate_ui_result,
)
from multi_agent_debate.usage import USAGE_EXTRACTION_VERSION, UsageRecorder, aggregate_usage

DEFAULT_QUESTION_MAXIMUM = 5000
DEFAULT_UI_MODE: UiMode = "multi-round"
DEFAULT_UI_ROUNDS = 2


class UiServiceError(RuntimeError):
    """Base bounded UI-service error."""


class UiConfigurationError(UiServiceError):
    """User input or required local configuration is invalid."""


class UiDependencyError(UiServiceError):
    """Live local dependencies could not be constructed safely."""


class UiExecutionError(UiServiceError):
    """The debate engine did not complete successfully."""


class UiProgressEvent(TypedDict):
    stage: str
    label: str


_PROGRESS = {
    "validate": "Validating configuration",
    "research": "Researching sources",
    "debate": "Running debate",
    "synthesis": "Creating neutral synthesis",
    "judge": "Evaluating debate",
    "finalize": "Finalizing result",
}


@dataclass(frozen=True, slots=True)
class UiDependencies:
    mode: UiMode
    decomposer: QuestionDecomposer
    research_provider: ResearchProvider
    debate_model: DebateModel
    grounding_recorder: RunTraceRecorder
    debate_usage_recorder: UsageRecorder
    judge: DebateJudge
    judge_usage_recorder: UsageRecorder
    debate_recorder: MultiRoundTraceRecorder | None = None
    team_model: TeamModel | None = None
    team_trace_recorder: TeamTraceRecorder | None = None


@dataclass(frozen=True, slots=True)
class UiServiceOutput:
    result: UiResult
    serialized_json: str


def resolve_tavily_api_key(
    *,
    streamlit_secrets: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve one local secret without retaining or exposing its source."""

    if streamlit_secrets is not None:
        try:
            value = streamlit_secrets.get(TAVILY_API_KEY_ENV)
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = os.environ if environ is None else environ
    try:
        value = source.get(TAVILY_API_KEY_ENV, "")
    except Exception:
        value = ""
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise UiConfigurationError("Tavily is not configured.")


def build_ui_dependencies(
    *,
    mode: UiMode,
    model_name: str,
    tavily_api_key: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    clock: Callable[[], float] = time.perf_counter,
) -> UiDependencies:
    """Construct existing live adapters after secret validation succeeds."""

    if not isinstance(tavily_api_key, str) or not tavily_api_key.strip():
        raise UiConfigurationError("Tavily is not configured.")
    if mode not in ("single-round", "multi-round", "hierarchical"):
        raise UiConfigurationError("The selected debate mode is invalid.")
    try:
        judge_usage = UsageRecorder()
        judge = OllamaDebateJudge(
            model_name=model_name, usage_recorder=judge_usage, clock=clock
        )
        if mode in ("single-round", "multi-round"):
            product, _ = build_live_dependencies(
                model_name=model_name,
                max_results=max_results,
                environ={TAVILY_API_KEY_ENV: tavily_api_key},
                clock=clock,
            )
            if product.usage_recorder is None:
                raise ValueError
            return UiDependencies(
                mode=mode,
                decomposer=product.decomposer,
                research_provider=product.research_provider,
                debate_model=product.debate_model,
                grounding_recorder=product.grounding_recorder,
                debate_recorder=product.debate_recorder,
                debate_usage_recorder=product.usage_recorder,
                judge=judge,
                judge_usage_recorder=judge_usage,
            )

        backend = TavilySearchBackend.from_environment(
            environ={TAVILY_API_KEY_ENV: tavily_api_key}
        )
        grounding = RunTraceRecorder()
        recording_backend = RecordingSearchBackend(
            backend,
            grounding,
            settings=sanitized_search_settings(),
            clock=clock,
        )
        research = MCPResearchProvider(
            create_backend_search_server_factory(recording_backend),
            max_results=max_results,
        )
        debate_usage = UsageRecorder()
        decomposer = OllamaQuestionDecomposer(
            model_name=model_name,
            usage_recorder=debate_usage,
            usage_variant="ui_hierarchical",
            clock=clock,
        )
        synthesis = OllamaDebateModel(
            model_name=model_name,
            usage_recorder=debate_usage,
            usage_variant="ui_hierarchical",
            clock=clock,
        )
        team_model = OllamaTeamModel(
            model_name=model_name,
            usage_recorder=debate_usage,
            usage_variant="ui_hierarchical",
            clock=clock,
        )
        return UiDependencies(
            mode="hierarchical",
            decomposer=decomposer,
            research_provider=research,
            debate_model=synthesis,
            grounding_recorder=grounding,
            debate_usage_recorder=debate_usage,
            judge=judge,
            judge_usage_recorder=judge_usage,
            team_model=team_model,
            team_trace_recorder=TeamTraceRecorder(),
        )
    except UiServiceError:
        raise
    except Exception:
        raise UiDependencyError("Local debate dependencies could not be initialized.") from None


def run_ui_debate(
    *,
    question: str,
    mode: str,
    max_rounds: int,
    novelty_threshold: float,
    model_name: str,
    implementation_commit: str,
    pricing: Pricing,
    tavily_api_key: str | None = None,
    dependencies: UiDependencies | None = None,
    progress_callback: Callable[[UiProgressEvent], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> UiServiceOutput:
    """Run a grounded debate, judge it once, and return validated download JSON."""

    _emit(progress_callback, "validate")
    selected_mode, rounds, threshold, commit = validate_ui_configuration(
        question=question,
        mode=mode,
        max_rounds=max_rounds,
        novelty_threshold=novelty_threshold,
        model_name=model_name,
        implementation_commit=implementation_commit,
        pricing=pricing,
    )
    if dependencies is None:
        if tavily_api_key is None:
            raise UiConfigurationError("Tavily is not configured.")
        dependencies = build_ui_dependencies(
            mode=selected_mode,
            model_name=model_name,
            tavily_api_key=tavily_api_key,
            clock=clock,
        )
    if dependencies.mode != selected_mode:
        raise UiConfigurationError("Injected UI dependencies do not match the mode.")

    try:
        _emit(progress_callback, "research")
        _emit(progress_callback, "debate")
        if selected_mode == "hierarchical":
            debate = _execute_hierarchical(
                question=question,
                model_name=model_name,
                implementation_commit=commit,
                novelty_threshold=threshold,
                pricing=pricing,
                dependencies=dependencies,
                clock=clock,
                utc_now=utc_now,
            )
        else:
            if dependencies.debate_recorder is None:
                raise UiExecutionError("Debate tracing is unavailable.")
            execution = execute_product_run(
                question=question,
                mode=selected_mode,
                model_name=model_name,
                max_rounds=rounds,
                novelty_threshold=threshold,
                max_results=DEFAULT_MAX_RESULTS,
                implementation_commit=commit,
                decomposer=dependencies.decomposer,
                research_provider=dependencies.research_provider,
                debate_model=dependencies.debate_model,
                grounding_recorder=dependencies.grounding_recorder,
                debate_recorder=dependencies.debate_recorder,
                clock=clock,
                utc_timestamp_provider=utc_now,
                usage_recorder=dependencies.debate_usage_recorder,
                pricing=pricing,
            )
            debate = public_debate_from_product_record(execution.run_record)
        _emit(progress_callback, "synthesis")
    except UiServiceError:
        raise
    except Exception:
        raise UiExecutionError("The debate could not be completed.") from None

    _emit(progress_callback, "judge")
    judgment = None
    judgment_error_category: UiJudgmentErrorCategory | None = None
    try:
        material = judge_material_from_debate(debate)
        assessment = dependencies.judge.judge(copy.deepcopy(material))
    except JudgeInvocationError:
        judgment_error_category = "invocation_failed"
    except JudgeResponseError as error:
        judgment_error_category = _judgment_error_category(error)
    except JudgeError:
        judgment_error_category = "invalid_structure"
    except Exception:
        judgment_error_category = "invocation_failed"
    else:
        try:
            judgment = build_judge_record(
                assessment=assessment,
                debate_material=material,
                debate_result=debate,
                run_timestamp_utc=utc_now(),
                implementation_commit=commit,
                model_name=model_name,
                pricing=pricing,
                usage_entries=dependencies.judge_usage_recorder.snapshot(),
            )
        except JudgeRecordValidationError:
            judgment_error_category = "invalid_record"
        except Exception:
            judgment_error_category = "invalid_record"

    try:
        _emit(progress_callback, "finalize")
        result = build_ui_result(
            debate_result=debate,
            judgment=judgment,
            judgment_error_category=judgment_error_category,
        )
        result = validate_ui_result(result, forbidden_secret=tavily_api_key)
        serialized = ui_result_to_json(result, forbidden_secret=tavily_api_key)
    except Exception:
        raise UiExecutionError("The downloadable result could not be finalized.") from None
    return UiServiceOutput(result=result, serialized_json=serialized)


def _judgment_error_category(error: JudgeResponseError) -> UiJudgmentErrorCategory:
    if error.category == "invalid_json":
        return "invalid_json"
    if error.category in {"invalid_score", "invalid_confidence"}:
        return "invalid_scores"
    if error.category in {
        "invalid_list",
        "invalid_text",
        "invalid_stance_drift",
    }:
        return "invalid_narrative"
    if error.category == "invalid_record":
        return "invalid_record"
    return "invalid_structure"


def validate_ui_configuration(
    *,
    question: object,
    mode: object,
    max_rounds: object,
    novelty_threshold: object,
    model_name: object,
    implementation_commit: object,
    pricing: object,
) -> tuple[UiMode, int, float, str]:
    if not isinstance(question, str) or not question.strip() or len(question) > DEFAULT_QUESTION_MAXIMUM:
        raise UiConfigurationError("Enter a nonblank debate question.")
    if mode not in ("single-round", "multi-round", "hierarchical"):
        raise UiConfigurationError("The selected debate mode is invalid.")
    try:
        rounds = validate_max_rounds(max_rounds)
        threshold = validate_novelty_threshold(novelty_threshold)
        commit = normalize_implementation_commit(implementation_commit)
    except Exception:
        raise UiConfigurationError("Debate settings are invalid.") from None
    if mode in ("single-round", "hierarchical") and rounds != 1:
        raise UiConfigurationError("The selected mode supports exactly one round.")
    if not isinstance(model_name, str) or not model_name.strip() or len(model_name) > 200:
        raise UiConfigurationError("The model name is invalid.")
    if not isinstance(pricing, Pricing):
        raise UiConfigurationError("Pricing settings are invalid.")
    return cast(UiMode, mode), rounds, threshold, commit


def _execute_hierarchical(
    *,
    question: str,
    model_name: str,
    implementation_commit: str,
    novelty_threshold: float,
    pricing: Pricing,
    dependencies: UiDependencies,
    clock: Callable[[], float],
    utc_now: Callable[[], datetime],
) -> UiDebateResult:
    if dependencies.team_model is None or dependencies.team_trace_recorder is None:
        raise UiExecutionError("Hierarchical debate dependencies are incomplete.")
    timed_decomposer = TimedQuestionDecomposer(
        dependencies.decomposer, dependencies.grounding_recorder, clock=clock
    )
    timed_research = TimedResearchProvider(
        dependencies.research_provider, dependencies.grounding_recorder, clock=clock
    )
    started = float(clock())
    result = run_hierarchical_debate(
        question,
        timed_decomposer,
        timed_research,
        dependencies.team_model,
        cast(NeutralSynthesisModel, dependencies.debate_model),
        team_trace_recorder=dependencies.team_trace_recorder,
        clock=clock,
    )
    total_duration = float(clock()) - started
    if total_duration < 0:
        raise UiExecutionError("Hierarchical timing is invalid.")
    team_traces = canonicalize_successful_team_traces(
        dependencies.team_trace_recorder
    )
    grounding = _canonical_grounding(result, dependencies.grounding_recorder)
    search_evidence = _search_evidence_from_grounding(result, grounding)
    totals_by_team = {
        item["team"]: item["duration_seconds"] for item in team_traces["total_traces"]
    }
    decision = {
        "stop": True,
        "reason": "max_rounds",
        "team_a_novelty": None,
        "team_b_novelty": None,
    }
    round_record: DebateRound = {
        "round_number": 1,
        "team_a_argument": result["team_a_argument"],
        "team_b_argument": result["team_b_argument"],
        "team_a_opponent_argument": None,
        "team_b_opponent_argument": result["team_a_argument"],
        "team_a_novelty": None,
        "team_b_novelty": None,
        "termination_decision": cast(dict, decision),
        "team_a_duration_seconds": totals_by_team["team_a"],
        "team_b_duration_seconds": totals_by_team["team_b"],
        "termination_duration_seconds": 0.0,
    }
    scorecard = score_expanded_result(result)
    metrics = json.loads(scorecard_to_json(scorecard))
    entries = dependencies.debate_usage_recorder.snapshot()
    usage_totals = aggregate_usage(entries)
    outputs = {
        "team_a_argument": result["team_a_argument"],
        "team_b_argument": result["team_b_argument"],
        "neutral_synthesis": result["synthesis"],
    }
    decomposition = {
        "question_complexity": result["question_complexity"],
        "subquestions": result["subquestions"].copy(),
    }
    products = {
        "team_a": copy.deepcopy(result["team_a_work_product"]),
        "team_b": copy.deepcopy(result["team_b_work_product"]),
    }
    efficiency = calculate_efficiency(
        usage_totals=usage_totals,
        visible_output_words=sum(count_generated_words(text) for text in outputs.values()),
        all_generated_words=calculate_ui_generated_words(
            mode="hierarchical",
            decomposition=decomposition,
            rounds=[round_record],
            products=products,
            synthesis=result["synthesis"],
        ),
        latency_seconds=total_duration,
        completed_rounds=1,
    )
    value: UiDebateResult = {
        "debate_result_version": UI_DEBATE_RESULT_VERSION,
        "mode": "hierarchical",
        "provenance": {
            "run_timestamp_utc": format_utc_timestamp(utc_now()),
            "implementation_commit": implementation_commit,
            "architecture_version": "hierarchical-grounded-ui",
            "source_run_record_version": "hierarchical-live-1.0",
            "model": {"name": model_name, "temperature": 0.0, "reasoning": False},
            "search_provider": "tavily",
            "search_settings": copy.deepcopy(sanitized_search_settings()),
            "pricing": pricing_to_data(pricing),
        },
        "input": {
            "question": question,
            "team_a_stance": result["team_a_stance"],
            "team_b_stance": result["team_b_stance"],
            "max_rounds": 1,
            "novelty_threshold": novelty_threshold,
        },
        "decomposition": cast(dict, decomposition),
        "search_evidence": search_evidence,
        "evidence": {
            "raw_research_findings": copy.deepcopy(result["raw_research_findings"]),
            "consolidated_evidence": copy.deepcopy(result["consolidated_evidence"]),
            "research_findings": result["research_findings"].copy(),
        },
        "debate_rounds": [round_record],
        "termination": cast(dict, decision),
        "final_outputs": cast(dict, outputs),
        "team_work_products": cast(dict, products),
        "timings": {
            "decomposition": copy.deepcopy(grounding["decomposition"]),
            "research": copy.deepcopy(grounding["research_calls"]),
            "team": copy.deepcopy(team_traces),
            "total_duration_seconds": total_duration,
        },
        "model_call_counts": {
            "decomposition": 1,
            "team_a_argument": 0,
            "team_b_argument": 0,
            "team_a_analysis": 1,
            "team_a_draft": 1,
            "team_a_review": 1,
            "team_b_analysis": 1,
            "team_b_draft": 1,
            "team_b_review": 1,
            "synthesis": 1,
            "research_provider_calls": len(grounding["research_calls"]),
            "total_model_calls": 8,
        },
        "metrics": metrics,
        "usage": {
            "usage_extraction_version": USAGE_EXTRACTION_VERSION,
            "entries": copy.deepcopy(entries),
            "totals": usage_totals,
        },
        "cost_estimate": calculate_cost(usage_totals, pricing=pricing),
        "efficiency": efficiency,
    }
    return validate_ui_debate_result(value)


def _canonical_grounding(result: dict, recorder: RunTraceRecorder) -> dict:
    tasks = (
        [(0, result["question"])]
        if result["question_complexity"] == "simple"
        else list(enumerate(result["subquestions"], start=1))
    )
    backend_by_query: dict[str, dict] = {}
    for trace in recorder.backend_searches:
        query = trace["query"]
        if query in backend_by_query or query not in {item[1] for item in tasks}:
            raise UiExecutionError("Grounding search traces are inconsistent.")
        if trace["status"] != "success" or trace["error_category"] is not None:
            raise UiExecutionError("Grounding search did not complete.")
        backend_by_query[query] = copy.deepcopy(trace)
    research_by_task: dict[tuple[int, str], dict] = {}
    for trace in recorder.research_calls:
        task = (trace["subquestion_index"], trace["query"])
        if task in research_by_task or task not in tasks:
            raise UiExecutionError("Grounding research traces are inconsistent.")
        if trace["status"] != "success" or trace["error_category"] is not None:
            raise UiExecutionError("Grounding research did not complete.")
        research_by_task[task] = copy.deepcopy(trace)
    model_stages = recorder.model_stages
    if (
        len(model_stages) != 1
        or model_stages[0]["stage"] != "decomposition"
        or model_stages[0]["status"] != "success"
        or model_stages[0]["error_category"] is not None
    ):
        raise UiExecutionError("Grounding decomposition trace is inconsistent.")
    if set(backend_by_query) != {item[1] for item in tasks} or set(
        research_by_task
    ) != set(tasks):
        raise UiExecutionError("Grounding traces are incomplete.")
    return {
        "backend_searches": [copy.deepcopy(backend_by_query[query]) for _, query in tasks],
        "research_calls": [copy.deepcopy(research_by_task[task]) for task in tasks],
        "decomposition": copy.deepcopy(model_stages[0]),
    }


def _search_evidence_from_grounding(result: dict, grounding: dict) -> list[dict]:
    tasks = (
        [(0, result["question"])]
        if result["question_complexity"] == "simple"
        else list(enumerate(result["subquestions"], start=1))
    )
    calls = []
    settings = sanitized_search_settings()
    for (index, query), backend, research in zip(
        tasks, grounding["backend_searches"], grounding["research_calls"], strict=True
    ):
        results = []
        for rank, item in enumerate(backend["results"], start=1):
            if "provider_score" not in item:
                raise UiExecutionError("Grounded source telemetry is incomplete.")
            results.append(
                {
                    "rank": rank,
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                    "provider_score": item["provider_score"],
                    "source_type": "web",
                }
            )
        calls.append(
            {
                "subquestion_index": index,
                "query": query,
                "requested_max_results": backend["requested_max_results"],
                "settings": copy.deepcopy(settings),
                "status": "success",
                "error_category": None,
                "backend_duration_seconds": backend["duration_seconds"],
                "research_duration_seconds": research["duration_seconds"],
                "results": results,
            }
        )
    return calls


def _emit(
    callback: Callable[[UiProgressEvent], None] | None, stage: str
) -> None:
    if callback is None:
        return
    if not callable(callback):
        raise UiConfigurationError("Progress callback is invalid.")
    try:
        callback({"stage": stage, "label": _PROGRESS[stage]})
    except Exception:
        raise UiConfigurationError("Progress reporting failed.") from None
