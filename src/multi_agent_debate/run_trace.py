"""Thread-safe timing and evidence tracing for expanded debate runs."""

import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from threading import Lock
from typing import Literal, TypeAlias, TypedDict

from multi_agent_debate.expanded_graph import QuestionDecomposer
from multi_agent_debate.expanded_state import DecompositionResult, ExpandedDebateResult
from multi_agent_debate.mcp_research import (
    MCPResearchConfigurationError,
    MCPResearchError,
    MCPResearchProtocolError,
    MCPResearchResponseError,
    MCPResearchToolError,
    MCPResearchTransportError,
)
from multi_agent_debate.model import DebateModel
from multi_agent_debate.ollama_decomposer import (
    OllamaDecompositionInvocationError,
    OllamaDecompositionResponseError,
)
from multi_agent_debate.research import ResearchFinding, ResearchProvider
from multi_agent_debate.search_backend import (
    BackendSearchResult,
    SearchBackend,
    SearchBackendConfigurationError,
    SearchBackendError,
    SearchBackendHTTPError,
    SearchBackendResponseError,
    SearchBackendTransportError,
    validate_backend_search_results,
)

MonotonicClock: TypeAlias = Callable[[], float]
TraceStatus: TypeAlias = Literal["success", "error"]
ModelStage: TypeAlias = Literal[
    "decomposition",
    "team_a",
    "team_b",
    "neutral_synthesis",
]


class BackendSearchTrace(TypedDict):
    query: str
    requested_max_results: int
    settings: dict[str, object]
    results: list[BackendSearchResult]
    duration_seconds: float
    status: TraceStatus
    error_category: str | None


class ResearchTrace(TypedDict):
    subquestion_index: int
    query: str
    duration_seconds: float
    status: TraceStatus
    error_category: str | None


class ModelStageTrace(TypedDict):
    stage: ModelStage
    duration_seconds: float
    status: TraceStatus
    error_category: str | None


class CanonicalRunTraces(TypedDict):
    backend_searches: list[BackendSearchTrace]
    research_calls: list[ResearchTrace]
    model_stages: list[ModelStageTrace]


class TraceValidationError(ValueError):
    """Raised when completed-run traces are missing or internally inconsistent."""


class RunTraceRecorder:
    """Own lock-protected trace collections and expose defensive snapshots."""

    def __init__(self) -> None:
        self._backend_searches: list[BackendSearchTrace] = []
        self._research_calls: list[ResearchTrace] = []
        self._model_stages: list[ModelStageTrace] = []
        self._lock = Lock()

    def record_backend_search(self, trace: BackendSearchTrace) -> None:
        with self._lock:
            self._backend_searches.append(deepcopy(trace))

    def record_research(self, trace: ResearchTrace) -> None:
        with self._lock:
            self._research_calls.append(deepcopy(trace))

    def record_model_stage(self, trace: ModelStageTrace) -> None:
        with self._lock:
            self._model_stages.append(deepcopy(trace))

    @property
    def backend_searches(self) -> list[BackendSearchTrace]:
        with self._lock:
            return deepcopy(self._backend_searches)

    @property
    def research_calls(self) -> list[ResearchTrace]:
        with self._lock:
            return deepcopy(self._research_calls)

    @property
    def model_stages(self) -> list[ModelStageTrace]:
        with self._lock:
            return deepcopy(self._model_stages)


class RecordingSearchBackend:
    """Record one sanitized trace around each delegated backend search."""

    def __init__(
        self,
        backend: SearchBackend,
        recorder: RunTraceRecorder,
        *,
        settings: Mapping[str, object],
        clock: MonotonicClock = time.perf_counter,
    ) -> None:
        self._backend = backend
        self._recorder = recorder
        resolved_settings = dict(settings)
        expected_settings = {
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if resolved_settings != expected_settings or any(
            type(resolved_settings[field]) is not type(expected_value)
            for field, expected_value in expected_settings.items()
        ):
            raise ValueError("recorded search settings must be sanitized")
        self._settings = deepcopy(resolved_settings)
        self._clock = clock

    def search(
        self,
        *,
        query: str,
        max_results: int,
    ) -> list[BackendSearchResult]:
        started = self._clock()
        try:
            raw_results = self._backend.search(
                query=query,
                max_results=max_results,
            )
            results = validate_backend_search_results(
                raw_results,
                max_results=max_results,
            )
        except Exception as error:
            self._recorder.record_backend_search(
                {
                    "query": query,
                    "requested_max_results": max_results,
                    "settings": deepcopy(self._settings),
                    "results": [],
                    "duration_seconds": _elapsed(self._clock, started),
                    "status": "error",
                    "error_category": bounded_error_category(error),
                }
            )
            raise

        recorded_results = [result.copy() for result in results]
        self._recorder.record_backend_search(
            {
                "query": query,
                "requested_max_results": max_results,
                "settings": deepcopy(self._settings),
                "results": recorded_results,
                "duration_seconds": _elapsed(self._clock, started),
                "status": "success",
                "error_category": None,
            }
        )
        return [result.copy() for result in results]


class TimedResearchProvider:
    """Record complete synchronous research-provider calls."""

    def __init__(
        self,
        provider: ResearchProvider,
        recorder: RunTraceRecorder,
        *,
        clock: MonotonicClock = time.perf_counter,
    ) -> None:
        self._provider = provider
        self._recorder = recorder
        self._clock = clock

    def research(
        self,
        *,
        subquestion_index: int,
        subquestion: str,
    ) -> Sequence[ResearchFinding]:
        started = self._clock()
        try:
            result = self._provider.research(
                subquestion_index=subquestion_index,
                subquestion=subquestion,
            )
        except Exception as error:
            self._record(
                subquestion_index=subquestion_index,
                query=subquestion,
                started=started,
                status="error",
                error_category=bounded_error_category(error),
            )
            raise
        self._record(
            subquestion_index=subquestion_index,
            query=subquestion,
            started=started,
            status="success",
            error_category=None,
        )
        return result

    def _record(
        self,
        *,
        subquestion_index: int,
        query: str,
        started: float,
        status: TraceStatus,
        error_category: str | None,
    ) -> None:
        self._recorder.record_research(
            {
                "subquestion_index": subquestion_index,
                "query": query,
                "duration_seconds": _elapsed(self._clock, started),
                "status": status,
                "error_category": error_category,
            }
        )


class TimedQuestionDecomposer:
    """Record one question-decomposition call."""

    def __init__(
        self,
        decomposer: QuestionDecomposer,
        recorder: RunTraceRecorder,
        *,
        clock: MonotonicClock = time.perf_counter,
    ) -> None:
        self._decomposer = decomposer
        self._recorder = recorder
        self._clock = clock

    def decompose(self, *, question: str) -> DecompositionResult:
        started = self._clock()
        try:
            result = self._decomposer.decompose(question=question)
        except Exception as error:
            self._record(started, "error", bounded_error_category(error))
            raise
        self._record(started, "success", None)
        return result

    def _record(
        self,
        started: float,
        status: TraceStatus,
        error_category: str | None,
    ) -> None:
        self._recorder.record_model_stage(
            {
                "stage": "decomposition",
                "duration_seconds": _elapsed(self._clock, started),
                "status": status,
                "error_category": error_category,
            }
        )


class TimedDebateModel:
    """Record Team A, Team B, and neutral-synthesis model calls."""

    def __init__(
        self,
        model: DebateModel,
        recorder: RunTraceRecorder,
        *,
        clock: MonotonicClock = time.perf_counter,
    ) -> None:
        self._model = model
        self._recorder = recorder
        self._clock = clock

    def research(self, *, question: str) -> list[str]:
        raise AssertionError("expanded graph must not call DebateModel.research")

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        stage: ModelStage = "team_a" if opponent_argument is None else "team_b"
        started = self._clock()
        try:
            result = self._model.write_argument(
                question=question,
                stance=stance,
                research_findings=research_findings,
                opponent_argument=opponent_argument,
            )
        except Exception as error:
            self._record(stage, started, "error", bounded_error_category(error))
            raise
        self._record(stage, started, "success", None)
        return result

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: list[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        started = self._clock()
        try:
            result = self._model.synthesize_neutrally(
                question=question,
                research_findings=research_findings,
                team_a_argument=team_a_argument,
                team_b_argument=team_b_argument,
            )
        except Exception as error:
            self._record(
                "neutral_synthesis",
                started,
                "error",
                bounded_error_category(error),
            )
            raise
        self._record("neutral_synthesis", started, "success", None)
        return result

    def _record(
        self,
        stage: ModelStage,
        started: float,
        status: TraceStatus,
        error_category: str | None,
    ) -> None:
        self._recorder.record_model_stage(
            {
                "stage": stage,
                "duration_seconds": _elapsed(self._clock, started),
                "status": status,
                "error_category": error_category,
            }
        )


def canonicalize_completed_traces(
    result: ExpandedDebateResult,
    recorder: RunTraceRecorder,
) -> CanonicalRunTraces:
    """Validate and order completed traces by the graph's decomposition."""

    expected_tasks = _expected_tasks(result)
    backend_by_query = _group_unique(
        recorder.backend_searches,
        key=lambda trace: trace["query"],
        expected=[query for _, query in expected_tasks],
        context="backend-search",
    )
    research_by_task = _group_unique(
        recorder.research_calls,
        key=lambda trace: (trace["subquestion_index"], trace["query"]),
        expected=expected_tasks,
        context="research-provider",
    )

    expected_stages: list[ModelStage] = [
        "decomposition",
        "team_a",
        "team_b",
        "neutral_synthesis",
    ]
    stages_by_name = _group_unique(
        recorder.model_stages,
        key=lambda trace: trace["stage"],
        expected=expected_stages,
        context="model-stage",
    )
    return {
        "backend_searches": [
            deepcopy(backend_by_query[query]) for _, query in expected_tasks
        ],
        "research_calls": [
            deepcopy(research_by_task[task]) for task in expected_tasks
        ],
        "model_stages": [
            deepcopy(stages_by_name[stage]) for stage in expected_stages
        ],
    }


def bounded_error_category(error: BaseException) -> str:
    """Map an exception type to a fixed category without inspecting its message."""

    if isinstance(error, SearchBackendConfigurationError):
        return "configuration_error"
    if isinstance(error, SearchBackendTransportError):
        return "transport_error"
    if isinstance(error, SearchBackendHTTPError):
        return "http_error"
    if isinstance(error, SearchBackendResponseError):
        return "response_error"
    if isinstance(error, SearchBackendError):
        return "search_backend_error"
    if isinstance(error, MCPResearchConfigurationError):
        return "configuration_error"
    if isinstance(error, MCPResearchTransportError):
        return "transport_error"
    if isinstance(error, MCPResearchProtocolError):
        return "mcp_protocol_error"
    if isinstance(error, MCPResearchToolError):
        return "mcp_tool_error"
    if isinstance(error, MCPResearchResponseError):
        return "mcp_response_error"
    if isinstance(error, MCPResearchError):
        return "mcp_error"
    if isinstance(error, OllamaDecompositionInvocationError):
        return "model_invocation_error"
    if isinstance(error, OllamaDecompositionResponseError):
        return "model_response_error"
    if isinstance(error, ValueError):
        return "response_validation_error"
    return "unexpected_error"


def _elapsed(clock: MonotonicClock, started: float) -> float:
    return float(clock() - started)


def _expected_tasks(
    result: ExpandedDebateResult,
) -> list[tuple[int, str]]:
    if result["question_complexity"] == "simple":
        if result["subquestions"]:
            raise TraceValidationError("simple result contains subquestions")
        return [(0, result["question"])]
    if result["question_complexity"] == "complex":
        subquestions = result["subquestions"]
        if len(subquestions) < 2 or len(set(subquestions)) != len(subquestions):
            raise TraceValidationError("complex result has invalid subquestions")
        return list(enumerate(subquestions, start=1))
    raise TraceValidationError("result has invalid question complexity")


def _group_unique(
    values: Sequence[object],
    *,
    key: Callable[[object], object],
    expected: Sequence[object],
    context: str,
) -> dict[object, object]:
    grouped: dict[object, object] = {}
    expected_set = set(expected)
    for value in values:
        identity = key(value)
        if identity not in expected_set:
            raise TraceValidationError(f"unexpected {context} trace")
        if identity in grouped:
            raise TraceValidationError(f"duplicate {context} trace")
        grouped[identity] = value
    if set(grouped) != expected_set:
        raise TraceValidationError(f"missing {context} trace")
    return grouped
