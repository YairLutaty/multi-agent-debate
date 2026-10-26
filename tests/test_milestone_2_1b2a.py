"""Offline tests for the Tavily-backed project MCP search path."""

import asyncio
import hmac
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import inf, nan
from threading import Lock
from typing import cast
from unittest.mock import patch

import httpx
import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.types import CallToolResult

import multi_agent_debate.mcp_backend_search_server as backend_server_module
import multi_agent_debate.mcp_search_server as deterministic_server_module
import multi_agent_debate.tavily_search as tavily_search
from multi_agent_debate.expanded_graph import run_expanded_debate
from multi_agent_debate.expanded_state import DecompositionResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.mcp_backend_search_server import (
    create_backend_search_server_factory,
)
from multi_agent_debate.mcp_research import (
    MCPResearchProvider,
    MCPResearchToolError,
)
from multi_agent_debate.mcp_search_contract import (
    SearchResult as SharedSearchResult,
    SearchWebResponse as SharedSearchWebResponse,
)
from multi_agent_debate.mcp_search_server import (
    SearchCallRecorder,
    create_search_server_factory,
)
from multi_agent_debate.research import validate_research_finding
from multi_agent_debate.search_backend import (
    BackendSearchResult,
    SearchBackendConfigurationError,
    SearchBackendHTTPError,
    SearchBackendResponseError,
    SearchBackendTransportError,
    validate_backend_search_results,
)
from multi_agent_debate.tavily_search import (
    TAVILY_API_KEY_ENV,
    TAVILY_SEARCH_URL,
    TavilySearchBackend,
)

FAKE_API_KEY = "offline-test-credential"
QUESTION = "Should a city expand public transport?"
SUBQUESTION_1 = "What access benefits could expansion provide?"
SUBQUESTION_2 = "What fiscal costs could expansion create?"
TEAM_A_ARGUMENT = "Expansion can improve access."
TEAM_B_ARGUMENT = "Expansion can impose substantial costs."
SYNTHESIS = "The evidence presents access benefits and fiscal costs."


def _tavily_result(
    title: str,
    *,
    content: str | None = None,
    score: object = 0.5,
    url: str | None = None,
) -> dict[str, object]:
    return {
        "title": title,
        "url": url or f"urn:tavily:{title.casefold().replace(' ', '-')}",
        "content": content or f"Evidence from {title}.",
        "score": score,
    }


def _response_payload(
    query: str,
    *,
    results: list[object] | None = None,
) -> dict[str, object]:
    return {
        "query": query,
        "results": results if results is not None else [_tavily_result("Source 1")],
    }


class ClientFactoryRecorder:
    """Create fresh MockTransport clients and count calls safely."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self._lock = Lock()
        self._count = 0
        self._clients: list[httpx.Client] = []

    def __call__(self) -> httpx.Client:
        client = httpx.Client(
            transport=httpx.MockTransport(self._handler),
            trust_env=False,
            follow_redirects=False,
        )
        with self._lock:
            self._count += 1
            self._clients.append(client)
        return client

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def client_ids(self) -> set[int]:
        with self._lock:
            return {id(client) for client in self._clients}


class RequestRecorder:
    """Record only sanitized request details behind a lock."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: list[dict[str, object]] = []

    def record(self, request: httpx.Request) -> dict[str, object]:
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {FAKE_API_KEY}"
        if not hmac.compare_digest(authorization, expected):
            raise AssertionError("Bearer authorization did not match")
        body = json.loads(request.content)
        sanitized = {
            "method": request.method,
            "url": str(request.url),
            "content_type": request.headers.get("content-type"),
            "authorization_valid": True,
            "body": body,
            "timeout": dict(request.extensions["timeout"]),
        }
        with self._lock:
            self._requests.append(deepcopy(sanitized))
        return body

    @property
    def requests(self) -> list[dict[str, object]]:
        with self._lock:
            return deepcopy(self._requests)


def _backend_for_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    timeout_seconds: float = 10.0,
) -> tuple[TavilySearchBackend, ClientFactoryRecorder]:
    factory = ClientFactoryRecorder(handler)
    return (
        TavilySearchBackend(
            api_key=FAKE_API_KEY,
            timeout_seconds=timeout_seconds,
            client_factory=factory,
        ),
        factory,
    )


async def _call_search_tool(
    server: MCPServer,
    *,
    query: str,
    max_results: int,
) -> CallToolResult:
    async with Client(server) as client:
        return await client.call_tool(
            "search_web",
            {"query": query, "max_results": max_results},
        )


@pytest.mark.parametrize("api_key", [None, "", "   ", 7, b"bytes"])
def test_invalid_api_keys_fail_before_client_creation(api_key: object) -> None:
    factory = ClientFactoryRecorder(
        lambda request: pytest.fail("HTTP handler must not run")
    )

    with pytest.raises(SearchBackendConfigurationError, match="API key"):
        TavilySearchBackend(
            api_key=cast(str, api_key),
            client_factory=factory,
        )

    assert factory.count == 0


def test_from_environment_uses_only_the_explicit_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenEnvironment(dict[str, str]):
        def get(self, key: str, default: str | None = None) -> str | None:
            raise AssertionError("real process environment was read")

    monkeypatch.setattr(tavily_search.os, "environ", ForbiddenEnvironment())
    backend = TavilySearchBackend.from_environment(
        environ={TAVILY_API_KEY_ENV: FAKE_API_KEY},
        client_factory=ClientFactoryRecorder(
            lambda request: httpx.Response(
                200,
                json=_response_payload(QUESTION, results=[]),
            )
        ),
    )

    assert FAKE_API_KEY not in repr(backend)


@pytest.mark.parametrize("environ", [{}, {TAVILY_API_KEY_ENV: "   "}])
def test_missing_environment_key_is_a_configuration_error(
    environ: Mapping[str, str],
) -> None:
    with pytest.raises(SearchBackendConfigurationError, match="API key"):
        TavilySearchBackend.from_environment(environ=environ)


def test_api_key_is_absent_from_repr_and_public_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"failure containing {FAKE_API_KEY}",
            request=request,
        )

    backend, _ = _backend_for_handler(handler)

    assert FAKE_API_KEY not in repr(backend)
    with pytest.raises(SearchBackendTransportError) as caught:
        backend.search(query=QUESTION, max_results=1)

    assert FAKE_API_KEY not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("query", [None, "", "   ", 7, b"bytes"])
def test_invalid_queries_fail_before_client_creation(query: object) -> None:
    factory = ClientFactoryRecorder(
        lambda request: pytest.fail("HTTP handler must not run")
    )
    backend = TavilySearchBackend(api_key=FAKE_API_KEY, client_factory=factory)

    with pytest.raises(SearchBackendConfigurationError, match="query"):
        backend.search(query=cast(str, query), max_results=1)

    assert factory.count == 0


@pytest.mark.parametrize("max_results", [True, False, 0, -1, 1.0, "1", None])
def test_invalid_maximums_fail_before_client_creation(max_results: object) -> None:
    factory = ClientFactoryRecorder(
        lambda request: pytest.fail("HTTP handler must not run")
    )
    backend = TavilySearchBackend(api_key=FAKE_API_KEY, client_factory=factory)

    with pytest.raises(SearchBackendConfigurationError, match="max_results"):
        backend.search(query=QUESTION, max_results=cast(int, max_results))

    assert factory.count == 0


@pytest.mark.parametrize(
    "timeout",
    [True, False, 0, -1, nan, inf, -inf, "10", None],
)
def test_invalid_timeouts_are_rejected(timeout: object) -> None:
    with pytest.raises(SearchBackendConfigurationError, match="timeout_seconds"):
        TavilySearchBackend(
            api_key=FAKE_API_KEY,
            timeout_seconds=cast(float, timeout),
        )


def test_invalid_client_factories_fail_deterministically() -> None:
    with pytest.raises(SearchBackendConfigurationError, match="callable"):
        TavilySearchBackend(
            api_key=FAKE_API_KEY,
            client_factory=cast(Callable[[], httpx.Client], 7),
        )

    raising_backend = TavilySearchBackend(
        api_key=FAKE_API_KEY,
        client_factory=lambda: (_ for _ in ()).throw(RuntimeError(FAKE_API_KEY)),
    )
    with pytest.raises(SearchBackendConfigurationError) as factory_error:
        raising_backend.search(query=QUESTION, max_results=1)
    assert FAKE_API_KEY not in str(factory_error.value)
    assert factory_error.value.__cause__ is None

    wrong_backend = TavilySearchBackend(
        api_key=FAKE_API_KEY,
        client_factory=cast(Callable[[], httpx.Client], lambda: object()),
    )
    with pytest.raises(SearchBackendConfigurationError, match="httpx.Client"):
        wrong_backend.search(query=QUESTION, max_results=1)


def test_default_client_factory_disables_environment_and_redirects() -> None:
    with patch("multi_agent_debate.tavily_search.httpx.Client") as constructor:
        created = tavily_search._default_client_factory()

    assert created is constructor.return_value
    constructor.assert_called_once_with(
        trust_env=False,
        follow_redirects=False,
    )


def test_exact_request_and_timeout_are_sent_once() -> None:
    recorder = RequestRecorder()

    def handler(request: httpx.Request) -> httpx.Response:
        body = recorder.record(request)
        return httpx.Response(
            200,
            json=_response_payload(cast(str, body["query"]), results=[]),
        )

    backend, factory = _backend_for_handler(handler, timeout_seconds=2.5)
    original_query = "  Preserve this query exactly?  "

    assert backend.search(query=original_query, max_results=3) == []

    assert factory.count == 1
    assert len(factory.client_ids) == 1
    assert recorder.requests == [
        {
            "method": "POST",
            "url": TAVILY_SEARCH_URL,
            "content_type": "application/json",
            "authorization_valid": True,
            "body": {
                "query": original_query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            "timeout": {
                "connect": 2.5,
                "read": 2.5,
                "write": 2.5,
                "pool": 2.5,
            },
        }
    ]


def test_results_preserve_provider_order_and_normalize_scores() -> None:
    raw_results = [
        _tavily_result("Lower score first", score=0.1),
        _tavily_result("Integer score second", score=1),
        _tavily_result("Middle score third", score=0.5),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **_response_payload(QUESTION, results=raw_results),
                "request_id": "ignored",
                "extra": "ignored",
            },
        )

    backend, _ = _backend_for_handler(handler)

    results = backend.search(query=QUESTION, max_results=3)

    assert [result["title"] for result in results] == [
        "Lower score first",
        "Integer score second",
        "Middle score third",
    ]
    assert [result["provider_score"] for result in results] == [0.1, 1.0, 0.5]
    assert all(type(result["provider_score"]) is float for result in results)


def test_extra_tavily_result_fields_are_ignored() -> None:
    raw = {**_tavily_result("Source"), "raw_content": "ignored", "extra": 3}
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(
            200,
            json=_response_payload(QUESTION, results=[raw]),
        )
    )

    assert backend.search(query=QUESTION, max_results=1) == [
        {
            "title": "Source",
            "url": "urn:tavily:source",
            "snippet": "Evidence from Source.",
            "provider_score": 0.5,
        }
    ]


def test_empty_tavily_results_are_valid() -> None:
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(
            200,
            json=_response_payload(QUESTION, results=[]),
        )
    )

    assert backend.search(query=QUESTION, max_results=2) == []


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"query": 7, "results": []},
        {"query": QUESTION},
        {"query": QUESTION, "results": {}},
        {"query": QUESTION, "results": ["not a dictionary"]},
    ],
    ids=[
        "top-level-list",
        "missing-query-and-results",
        "query-wrong-type",
        "missing-results",
        "results-wrong-type",
        "result-wrong-type",
    ],
)
def test_wrong_tavily_json_shapes_are_rejected(payload: object) -> None:
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(200, json=payload)
    )

    with pytest.raises(SearchBackendResponseError):
        backend.search(query=QUESTION, max_results=2)


def test_mismatched_tavily_query_is_rejected() -> None:
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(
            200,
            json=_response_payload("A different query"),
        )
    )

    with pytest.raises(SearchBackendResponseError, match="does not match"):
        backend.search(query=QUESTION, max_results=1)


def _malformed_tavily_results() -> list[object]:
    valid = _tavily_result("Source")
    malformed: list[object] = []
    for field in ("title", "url", "content", "score"):
        missing = valid.copy()
        del missing[field]
        malformed.append(missing)
    for field in ("title", "url", "content"):
        malformed.append({**valid, field: "   "})
        malformed.append({**valid, field: 7})
    for score in (True, False, "0.5", None, nan, inf, -inf):
        malformed.append({**valid, "score": score})
    return malformed


@pytest.mark.parametrize("raw_result", _malformed_tavily_results())
def test_malformed_tavily_result_fields_are_rejected(raw_result: object) -> None:
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(
            200,
            content=json.dumps(
                _response_payload(QUESTION, results=[raw_result])
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
    )

    with pytest.raises(SearchBackendResponseError):
        backend.search(query=QUESTION, max_results=1)


def test_excess_tavily_results_are_rejected() -> None:
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(
            200,
            json=_response_payload(
                QUESTION,
                results=[_tavily_result("One"), _tavily_result("Two")],
            ),
        )
    )

    with pytest.raises(SearchBackendResponseError, match="too many"):
        backend.search(query=QUESTION, max_results=1)


def test_redirect_with_location_header_is_not_followed() -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"Location": "https://redirect.invalid/next"},
            content=b"redirect body must not be parsed",
        )

    backend, _ = _backend_for_handler(handler)

    with pytest.raises(SearchBackendHTTPError) as caught:
        backend.search(query=QUESTION, max_results=1)

    assert caught.value.status_code == 302
    assert requested_urls == [TAVILY_SEARCH_URL]


@pytest.mark.parametrize("status_code", [400, 401, 403, 429, 500, 503])
def test_non_success_statuses_are_safe_and_never_retried(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status_code,
            content=f"unparsed body containing {FAKE_API_KEY}".encode(),
        )

    backend, _ = _backend_for_handler(handler)

    with pytest.raises(SearchBackendHTTPError) as caught:
        backend.search(query=QUESTION, max_results=1)

    assert calls == 1
    assert caught.value.status_code == status_code
    assert caught.value.__dict__ == {"status_code": status_code}
    assert FAKE_API_KEY not in str(caught.value)


def test_invalid_json_is_rejected_without_raw_content() -> None:
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(
            200,
            content=f"not JSON {FAKE_API_KEY}".encode(),
        )
    )

    with pytest.raises(SearchBackendResponseError) as caught:
        backend.search(query=QUESTION, max_results=1)

    assert FAKE_API_KEY not in str(caught.value)


@pytest.mark.parametrize("failure", ["connection", "timeout"])
def test_transport_failures_are_safe_and_never_retried(failure: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        error_type = httpx.ConnectError if failure == "connection" else httpx.ReadTimeout
        raise error_type(FAKE_API_KEY, request=request)

    backend, _ = _backend_for_handler(handler)

    with pytest.raises(SearchBackendTransportError) as caught:
        backend.search(query=QUESTION, max_results=1)

    assert calls == 1
    assert caught.value.__cause__ is None
    assert FAKE_API_KEY not in str(caught.value)


def _backend_result(
    title: str = "Backend source",
    *,
    score: float | int | None = 0.4,
) -> BackendSearchResult:
    result: BackendSearchResult = {
        "title": title,
        "url": f"urn:backend:{title.casefold().replace(' ', '-')}",
        "snippet": f"Backend evidence from {title}.",
    }
    if score is not None:
        result["provider_score"] = cast(float, score)
    return result


def test_backend_result_validator_copies_normalizes_and_preserves_order() -> None:
    values = [_backend_result("First", score=1), _backend_result("Second", score=None)]

    validated = validate_backend_search_results(values, max_results=2)

    assert validated == [
        {**values[0], "provider_score": 1.0},
        values[1],
    ]
    assert validated[0] is not values[0]
    assert validated[1] is not values[1]


def test_backend_result_validator_accepts_a_tuple_collection() -> None:
    original = _backend_result("Tuple result", score=1)

    validated = validate_backend_search_results((original,), max_results=1)

    assert validated == [{**original, "provider_score": 1.0}]
    assert validated[0] is not original


@pytest.mark.parametrize("values", ["text", b"bytes", {}, iter(())])
def test_backend_result_validator_requires_a_nonstring_sequence(
    values: object,
) -> None:
    with pytest.raises(SearchBackendResponseError, match="sequence"):
        validate_backend_search_results(values, max_results=1)


def _malformed_backend_results() -> list[object]:
    valid = _backend_result()
    missing = valid.copy()
    del missing["title"]
    unexpected = {**valid, "unexpected": "value"}
    return [
        ["not a dictionary"],
        [missing],
        [unexpected],
        [{**valid, "title": "   "}],
        [{**valid, "url": 7}],
        [{**valid, "provider_score": True}],
        [{**valid, "provider_score": "0.4"}],
        [{**valid, "provider_score": nan}],
        [{**valid, "provider_score": inf}],
    ]


@pytest.mark.parametrize("values", _malformed_backend_results())
def test_backend_result_validator_rejects_malformed_items(values: object) -> None:
    with pytest.raises(SearchBackendResponseError):
        validate_backend_search_results(values, max_results=1)


def test_backend_result_validator_rejects_excess_results() -> None:
    with pytest.raises(SearchBackendResponseError, match="too many"):
        validate_backend_search_results(
            [_backend_result("One"), _backend_result("Two")],
            max_results=1,
        )


class RecordingBackend:
    def __init__(
        self,
        responses: Mapping[str, Sequence[BackendSearchResult]],
    ) -> None:
        self._responses = responses
        self._lock = Lock()
        self._calls: list[tuple[str, int]] = []

    def search(
        self,
        *,
        query: str,
        max_results: int,
    ) -> Sequence[BackendSearchResult]:
        with self._lock:
            self._calls.append((query, max_results))
        return [result.copy() for result in self._responses.get(query, ())]

    @property
    def calls(self) -> list[tuple[str, int]]:
        with self._lock:
            return self._calls.copy()


def test_both_mcp_servers_use_the_same_shared_output_schema() -> None:
    deterministic_server = create_search_server_factory(
        {},
        SearchCallRecorder(),
    )()
    backend_server = create_backend_search_server_factory(
        RecordingBackend({})
    )()

    async def output_schemas() -> tuple[object, object]:
        deterministic_tools = await deterministic_server.list_tools()
        backend_tools = await backend_server.list_tools()
        return (
            deterministic_tools[0].output_schema,
            backend_tools[0].output_schema,
        )

    deterministic_schema, backend_schema = asyncio.run(output_schemas())

    assert deterministic_server_module.SearchResult is SharedSearchResult
    assert backend_server_module.SearchResult is SharedSearchResult
    assert (
        deterministic_server_module.SearchWebResponse
        is SharedSearchWebResponse
    )
    assert backend_server_module.SearchWebResponse is SharedSearchWebResponse
    assert deterministic_schema == backend_schema


def test_backend_mcp_server_calls_once_and_emits_only_frozen_contract() -> None:
    backend = RecordingBackend(
        {QUESTION: [_backend_result("Low", score=0.1), _backend_result("High", score=0.9)]}
    )
    server = create_backend_search_server_factory(backend)()

    result = asyncio.run(
        _call_search_tool(server, query=QUESTION, max_results=2)
    )

    assert backend.calls == [(QUESTION, 2)]
    assert result.is_error is False
    assert result.structured_content == {
        "query": QUESTION,
        "results": [
            {
                "rank": 1,
                "title": "Low",
                "url": "urn:backend:low",
                "snippet": "Backend evidence from Low.",
                "source_type": "web",
            },
            {
                "rank": 2,
                "title": "High",
                "url": "urn:backend:high",
                "snippet": "Backend evidence from High.",
                "source_type": "web",
            },
        ],
    }
    for item in cast(dict[str, list[dict[str, object]]], result.structured_content)[
        "results"
    ]:
        assert set(item) == {"rank", "title", "url", "snippet", "source_type"}


@pytest.mark.parametrize("known", [True, False])
def test_backend_failures_become_bounded_genuine_mcp_errors(known: bool) -> None:
    class FailingBackend:
        def search(
            self,
            *,
            query: str,
            max_results: int,
        ) -> Sequence[BackendSearchResult]:
            if known:
                raise SearchBackendResponseError("safe backend failure")
            raise RuntimeError(FAKE_API_KEY)

    result = asyncio.run(
        _call_search_tool(
            create_backend_search_server_factory(FailingBackend())(),
            query=QUESTION,
            max_results=1,
        )
    )

    assert result.is_error is True
    assert FAKE_API_KEY not in str(result.content)


def test_genuine_mcp_traversal_uses_mocked_tavily_and_maps_findings() -> None:
    recorder = RequestRecorder()

    def handler(request: httpx.Request) -> httpx.Response:
        body = recorder.record(request)
        query = cast(str, body["query"])
        return httpx.Response(
            200,
            json=_response_payload(
                query,
                results=[
                    _tavily_result(
                        "Grounded source",
                        content="A grounded snippet.",
                        score=0.8,
                    )
                ],
            ),
        )

    backend, factory = _backend_for_handler(handler)
    provider = MCPResearchProvider(
        create_backend_search_server_factory(backend),
        max_results=1,
    )

    with (
        patch("socket.create_connection", side_effect=AssertionError("network")),
        patch.object(
            subprocess,
            "Popen",
            side_effect=AssertionError("subprocess"),
        ),
    ):
        findings = provider.research(subquestion_index=4, subquestion=QUESTION)

    assert factory.count == 1
    assert len(recorder.requests) == 1
    assert findings == [
        {
            "subquestion_index": 4,
            "subquestion": QUESTION,
            "finding_index": 1,
            "claim": "A grounded snippet.",
            "source_title": "Grounded source",
            "source_url": "urn:tavily:grounded-source",
            "source_type": "web",
            "supporting_excerpt": "A grounded snippet.",
        }
    ]
    assert validate_research_finding(
        findings[0],
        expected_subquestion_index=4,
        expected_subquestion=QUESTION,
    ) == findings[0]


def test_tavily_http_error_reaches_frozen_mcp_tool_error_boundary() -> None:
    backend, _ = _backend_for_handler(
        lambda request: httpx.Response(401, content=b"denied")
    )
    provider = MCPResearchProvider(
        create_backend_search_server_factory(backend),
        max_results=1,
    )

    with pytest.raises(MCPResearchToolError, match="error result"):
        provider.research(subquestion_index=0, subquestion=QUESTION)


def test_concurrent_provider_calls_do_not_cross_contaminate() -> None:
    request_log: list[tuple[str, str]] = []
    request_lock = Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], json.loads(request.content))
        query = cast(str, body["query"])
        content = f"Evidence only for {query}"
        with request_lock:
            request_log.append((query, content))
        return httpx.Response(
            200,
            json=_response_payload(
                query,
                results=[_tavily_result(query, content=content)],
            ),
        )

    backend, factory = _backend_for_handler(handler)
    provider = MCPResearchProvider(
        create_backend_search_server_factory(backend),
        max_results=1,
    )
    calls = [(1, SUBQUESTION_1), (2, SUBQUESTION_2)]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            (index, query): executor.submit(
                provider.research,
                subquestion_index=index,
                subquestion=query,
            )
            for index, query in calls
        }
        results = {call: future.result() for call, future in futures.items()}

    assert sorted(request_log) == sorted(
        [
            (SUBQUESTION_1, f"Evidence only for {SUBQUESTION_1}"),
            (SUBQUESTION_2, f"Evidence only for {SUBQUESTION_2}"),
        ]
    )
    assert factory.count == 2
    assert len(factory.client_ids) == 2
    for (index, query), findings in results.items():
        assert findings[0]["subquestion_index"] == index
        assert findings[0]["subquestion"] == query
        assert findings[0]["claim"] == f"Evidence only for {query}"


class FakeDecomposer:
    def decompose(self, *, question: str) -> DecompositionResult:
        assert question == QUESTION
        return {
            "question_complexity": "complex",
            "subquestions": [SUBQUESTION_1, SUBQUESTION_2],
        }


class FakeDebateModel:
    def __init__(self) -> None:
        self.research_called = False
        self.argument_research: list[tuple[str, ...]] = []
        self.synthesis_research: tuple[str, ...] | None = None

    def research(self, *, question: str) -> list[str]:
        self.research_called = True
        raise AssertionError("expanded graph must not call model.research")

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        assert question == QUESTION
        self.argument_research.append(tuple(research_findings))
        if stance == TEAM_A_STANCE:
            assert opponent_argument is None
            return TEAM_A_ARGUMENT
        assert stance == TEAM_B_STANCE
        assert opponent_argument == TEAM_A_ARGUMENT
        return TEAM_B_ARGUMENT

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: list[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        assert question == QUESTION
        assert team_a_argument == TEAM_A_ARGUMENT
        assert team_b_argument == TEAM_B_ARGUMENT
        self.synthesis_research = tuple(research_findings)
        return SYNTHESIS


def test_complete_expanded_graph_makes_one_mocked_request_per_subquestion() -> None:
    request_queries: list[str] = []
    request_lock = Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], json.loads(request.content))
        query = cast(str, body["query"])
        with request_lock:
            request_queries.append(query)
        return httpx.Response(
            200,
            json=_response_payload(
                query,
                results=[
                    _tavily_result(
                        f"Source for {query}",
                        content=f"Grounded evidence for {query}",
                    )
                ],
            ),
        )

    backend, factory = _backend_for_handler(handler)
    provider = MCPResearchProvider(
        create_backend_search_server_factory(backend),
        max_results=1,
    )
    model = FakeDebateModel()

    with (
        patch("socket.create_connection", side_effect=AssertionError("network")),
        patch.object(
            subprocess,
            "Popen",
            side_effect=AssertionError("subprocess"),
        ),
    ):
        result = run_expanded_debate(
            QUESTION,
            FakeDecomposer(),
            provider,
            model,
        )

    assert sorted(request_queries) == sorted([SUBQUESTION_1, SUBQUESTION_2])
    assert factory.count == 2
    assert len(result["raw_research_findings"]) == 2
    assert len(result["consolidated_evidence"]) == 2
    shared_research = tuple(result["research_findings"])
    assert len(shared_research) == 2
    assert model.research_called is False
    assert model.argument_research == [shared_research, shared_research]
    assert model.synthesis_research == shared_research
    assert result["team_a_argument"] == TEAM_A_ARGUMENT
    assert result["team_b_argument"] == TEAM_B_ARGUMENT
    assert result["synthesis"] == SYNTHESIS
