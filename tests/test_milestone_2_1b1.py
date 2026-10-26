"""Offline tests for deterministic MCP research-provider integration."""

import asyncio
import builtins
import gc
import importlib
import subprocess
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from threading import Lock
from typing import Annotated, Any, cast
from unittest.mock import patch

import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.types import CallToolResult
from pydantic import Field

import multi_agent_debate.mcp_research as mcp_research
from multi_agent_debate.expanded_graph import run_expanded_debate
from multi_agent_debate.expanded_state import DecompositionResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.mcp_research import (
    MCPResearchConfigurationError,
    MCPResearchProtocolError,
    MCPResearchProvider,
    MCPResearchResponseError,
    MCPResearchToolError,
    MCPResearchTransportError,
    map_search_tool_result,
)
from multi_agent_debate.mcp_search_server import (
    SearchCallRecorder,
    SearchResult,
    SearchWebResponse,
    create_search_server_factory,
)
from multi_agent_debate.research import validate_research_finding

QUESTION = "Should a city expand its public transport network?"
SUBQUESTION_1 = "What benefits can transport expansion provide?"
SUBQUESTION_2 = "What costs can transport expansion impose?"
TEAM_A_ARGUMENT = "Expansion can improve mobility and access."
TEAM_B_ARGUMENT = "Expansion can impose material fiscal costs."
SYNTHESIS = "The arguments identify both access benefits and fiscal costs."

_StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]


def _result(
    rank: int,
    *,
    title: str | None = None,
    snippet: str | None = None,
) -> SearchResult:
    return {
        "rank": rank,
        "title": title or f"Fixture source {rank}",
        "url": f"urn:fixture:source:{rank}",
        "snippet": snippet or f"Fixture evidence {rank}.",
        "source_type": "deterministic fixture",
    }


CATALOG = {
    QUESTION: [_result(1), _result(2)],
    SUBQUESTION_1: [
        _result(1, title="Benefits source", snippet="Access can improve."),
        _result(2, title="Mobility source", snippet="Mobility can increase."),
    ],
    SUBQUESTION_2: [
        _result(1, title="Costs source", snippet="Capital costs can be high."),
        _result(2, title="Budget source", snippet="Budgets constrain scope."),
    ],
}


def _tool_result(
    structured_content: object | None,
    *,
    is_error: bool = False,
) -> CallToolResult:
    return CallToolResult(
        content=[],
        structuredContent=structured_content,
        isError=is_error,
    )


def _structured_response(
    *,
    query: str = QUESTION,
    results: list[object] | None = None,
) -> dict[str, object]:
    return {
        "query": query,
        "results": deepcopy(results if results is not None else [_result(1)]),
    }


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
        self.argument_calls: list[tuple[str, tuple[str, ...], str | None]] = []
        self.synthesis_research: tuple[str, ...] | None = None

    def research(self, *, question: str) -> list[str]:
        self.research_called = True
        raise AssertionError("the expanded graph must not call model.research")

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        assert question == QUESTION
        self.argument_calls.append(
            (stance, tuple(research_findings), opponent_argument)
        )
        return TEAM_A_ARGUMENT if opponent_argument is None else TEAM_B_ARGUMENT

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


def test_importing_adapter_does_not_import_concrete_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_name = "multi_agent_debate.mcp_research"
    concrete_names = (
        "multi_agent_debate.mcp_search_server",
        "multi_agent_debate.mcp_backend_search_server",
    )
    package = sys.modules["multi_agent_debate"]
    saved_adapter = sys.modules.pop(adapter_name)
    saved_concrete_modules = {
        name: sys.modules.pop(name, None) for name in concrete_names
    }
    saved_adapter_attribute = getattr(package, "mcp_research", None)
    missing = object()
    saved_concrete_attributes = {
        name: getattr(package, name.rsplit(".", 1)[1], missing)
        for name in concrete_names
    }
    if hasattr(package, "mcp_research"):
        delattr(package, "mcp_research")
    for name in concrete_names:
        attribute = name.rsplit(".", 1)[1]
        if hasattr(package, attribute):
            delattr(package, attribute)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if any(
            name == concrete or name.startswith(f"{concrete}.")
            for concrete in concrete_names
        ):
            raise AssertionError("adapter imported a concrete MCP server")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        imported = importlib.import_module(adapter_name)
        assert imported.MCPResearchProvider is not None
        assert all(name not in sys.modules for name in concrete_names)
    finally:
        sys.modules.pop(adapter_name, None)
        sys.modules[adapter_name] = saved_adapter
        setattr(package, "mcp_research", saved_adapter_attribute)
        for name in concrete_names:
            sys.modules.pop(name, None)
            saved_module = saved_concrete_modules[name]
            if saved_module is not None:
                sys.modules[name] = saved_module
            attribute = name.rsplit(".", 1)[1]
            saved_attribute = saved_concrete_attributes[name]
            if saved_attribute is missing:
                if hasattr(package, attribute):
                    delattr(package, attribute)
            else:
                setattr(package, attribute, saved_attribute)


def test_mcp_v2_server_and_client_complete_a_real_in_memory_tool_call() -> None:
    recorder = SearchCallRecorder()
    server = create_search_server_factory(CATALOG, recorder)()

    async def invoke() -> tuple[CallToolResult, list[object]]:
        tools = await server.list_tools()
        async with Client(server) as client:
            result = await client.call_tool(
                "search_web",
                {"query": QUESTION, "max_results": 1},
            )
        return result, tools

    with (
        patch("socket.create_connection", side_effect=AssertionError("network")),
        patch.object(
            subprocess,
            "Popen",
            side_effect=AssertionError("subprocess"),
        ),
    ):
        result, tools = asyncio.run(invoke())

    assert isinstance(server, MCPServer)
    assert [tool.name for tool in tools] == ["search_web"]
    assert tools[0].output_schema is not None
    assert set(tools[0].output_schema["required"]) == {"query", "results"}
    assert result.is_error is False
    assert result.structured_content == {
        "query": QUESTION,
        "results": [CATALOG[QUESTION][0]],
    }
    assert recorder.calls == [(QUESTION, 1)]


def test_provider_propagates_exact_arguments_and_maps_structured_results() -> None:
    recorder = SearchCallRecorder()
    provider = MCPResearchProvider(
        create_search_server_factory(CATALOG, recorder),
        max_results=2,
    )

    findings = provider.research(subquestion_index=7, subquestion=QUESTION)

    assert recorder.calls == [(QUESTION, 2)]
    assert findings == [
        {
            "subquestion_index": 7,
            "subquestion": QUESTION,
            "finding_index": 1,
            "claim": "Fixture evidence 1.",
            "source_title": "Fixture source 1",
            "source_url": "urn:fixture:source:1",
            "source_type": "deterministic fixture",
            "supporting_excerpt": "Fixture evidence 1.",
        },
        {
            "subquestion_index": 7,
            "subquestion": QUESTION,
            "finding_index": 2,
            "claim": "Fixture evidence 2.",
            "source_title": "Fixture source 2",
            "source_url": "urn:fixture:source:2",
            "source_type": "deterministic fixture",
            "supporting_excerpt": "Fixture evidence 2.",
        },
    ]
    for finding in findings:
        assert validate_research_finding(
            finding,
            expected_subquestion_index=7,
            expected_subquestion=QUESTION,
        ) == finding


def test_catalog_and_returned_records_are_defensively_copied() -> None:
    catalog = {QUESTION: [_result(1)]}
    original = deepcopy(catalog)
    recorder = SearchCallRecorder()
    factory = create_search_server_factory(catalog, recorder)
    catalog[QUESTION][0]["snippet"] = "Changed after construction."

    first = MCPResearchProvider(factory).research(
        subquestion_index=0,
        subquestion=QUESTION,
    )
    first[0]["claim"] = "Changed by caller."
    second = MCPResearchProvider(factory).research(
        subquestion_index=0,
        subquestion=QUESTION,
    )

    assert second[0]["claim"] == original[QUESTION][0]["snippet"]
    assert catalog[QUESTION][0]["snippet"] == "Changed after construction."


class _TrackingClient:
    """Delegate to the real client while recording independent lifecycles."""

    instances: list["_TrackingClient"] = []
    lock = Lock()

    def __init__(self, server: MCPServer, **kwargs: object) -> None:
        self.server = server
        self.delegate = Client(server, **kwargs)
        self.entered = 0
        self.exited = 0
        with self.lock:
            self.instances.append(self)

    async def __aenter__(self) -> "_TrackingClient":
        await self.delegate.__aenter__()
        self.entered += 1
        return self

    async def __aexit__(self, *args: object) -> object:
        try:
            return await self.delegate.__aexit__(*args)
        finally:
            self.exited += 1

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> CallToolResult:
        return await self.delegate.call_tool(name, arguments)


def test_separate_provider_calls_use_fresh_server_and_client_lifecycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = SearchCallRecorder()
    base_factory = create_search_server_factory(CATALOG, recorder)
    servers: list[MCPServer] = []

    def factory() -> MCPServer:
        server = base_factory()
        servers.append(server)
        return server

    _TrackingClient.instances = []
    monkeypatch.setattr(mcp_research, "Client", _TrackingClient)
    provider = MCPResearchProvider(factory, max_results=1)

    provider.research(subquestion_index=1, subquestion=SUBQUESTION_1)
    provider.research(subquestion_index=2, subquestion=SUBQUESTION_2)

    assert len(servers) == 2
    assert servers[0] is not servers[1]
    assert len(_TrackingClient.instances) == 2
    assert _TrackingClient.instances[0].delegate is not _TrackingClient.instances[1].delegate
    assert [instance.server for instance in _TrackingClient.instances] == servers
    assert [instance.entered for instance in _TrackingClient.instances] == [1, 1]
    assert [instance.exited for instance in _TrackingClient.instances] == [1, 1]


def test_concurrent_calls_are_isolated_without_query_cross_contamination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = SearchCallRecorder()
    base_factory = create_search_server_factory(CATALOG, recorder)
    servers: list[MCPServer] = []
    servers_lock = Lock()

    def factory() -> MCPServer:
        server = base_factory()
        with servers_lock:
            servers.append(server)
        return server

    _TrackingClient.instances = []
    monkeypatch.setattr(mcp_research, "Client", _TrackingClient)
    provider = MCPResearchProvider(factory, max_results=2)
    calls = [(1, SUBQUESTION_1), (2, SUBQUESTION_2)]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            (index, subquestion): executor.submit(
                provider.research,
                subquestion_index=index,
                subquestion=subquestion,
            )
            for index, subquestion in calls
        }
        findings_by_call = {
            call: future.result() for call, future in futures.items()
        }

    assert sorted(recorder.calls) == sorted(
        [(SUBQUESTION_1, 2), (SUBQUESTION_2, 2)]
    )
    assert len(servers) == 2
    assert servers[0] is not servers[1]
    assert len(_TrackingClient.instances) == 2
    assert len({id(instance.delegate) for instance in _TrackingClient.instances}) == 2
    for (index, subquestion), findings in findings_by_call.items():
        assert findings
        assert {finding["subquestion_index"] for finding in findings} == {index}
        assert {finding["subquestion"] for finding in findings} == {subquestion}
        expected_snippets = {
            result["snippet"] for result in CATALOG[subquestion]
        }
        assert {finding["claim"] for finding in findings} == expected_snippets


def test_unknown_query_returns_an_empty_finding_list() -> None:
    recorder = SearchCallRecorder()
    provider = MCPResearchProvider(
        create_search_server_factory(CATALOG, recorder),
        max_results=2,
    )

    assert provider.research(subquestion_index=0, subquestion="Unknown query") == []
    assert recorder.calls == [("Unknown query", 2)]


def test_real_mcp_tool_error_becomes_project_tool_error() -> None:
    def factory() -> MCPServer:
        server = MCPServer(name="failing-tool")

        @server.tool(name="search_web")
        def search_web(
            query: str,
            max_results: _StrictPositiveInt,
        ) -> SearchWebResponse:
            raise RuntimeError("deterministic tool failure")

        return server

    provider = MCPResearchProvider(factory)

    with pytest.raises(MCPResearchToolError, match="error result"):
        provider.research(subquestion_index=0, subquestion=QUESTION)


def test_boolean_output_rank_is_rejected_by_real_mcp_schema_and_provider() -> None:
    def factory() -> MCPServer:
        server = MCPServer(name="malformed-output")

        @server.tool(name="search_web")
        def search_web(
            query: str,
            max_results: _StrictPositiveInt,
        ) -> SearchWebResponse:
            return cast(
                SearchWebResponse,
                {
                    "query": query,
                    "results": [{**_result(1), "rank": True}],
                },
            )

        return server

    async def invoke_directly() -> CallToolResult:
        async with Client(factory()) as client:
            return await client.call_tool(
                "search_web",
                {"query": QUESTION, "max_results": 1},
            )

    result = asyncio.run(invoke_directly())

    assert result.is_error is True
    with pytest.raises(MCPResearchToolError, match="error result"):
        MCPResearchProvider(factory, max_results=1).research(
            subquestion_index=0,
            subquestion=QUESTION,
        )


def test_mcp_call_failure_becomes_project_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CallFailingClient:
        def __init__(self, server: MCPServer, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "CallFailingClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object],
        ) -> CallToolResult:
            raise ValueError("SDK schema failure")

    monkeypatch.setattr(mcp_research, "Client", CallFailingClient)
    provider = MCPResearchProvider(create_search_server_factory(CATALOG, SearchCallRecorder()))

    with pytest.raises(MCPResearchProtocolError, match="call failed") as caught:
        provider.research(subquestion_index=0, subquestion=QUESTION)

    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("failure_point", ["enter", "exit"])
def test_mcp_lifecycle_failure_becomes_project_transport_error(
    failure_point: str,
) -> None:
    @asynccontextmanager
    async def lifespan(server: MCPServer):
        if failure_point == "enter":
            raise RuntimeError("enter failed")
        yield {}
        raise RuntimeError("exit failed")

    def factory() -> MCPServer:
        server = MCPServer(name="broken-lifecycle", lifespan=lifespan)

        @server.tool(name="search_web")
        def search_web(
            query: str,
            max_results: _StrictPositiveInt,
        ) -> SearchWebResponse:
            return {"query": query, "results": []}

        return server

    with pytest.raises(MCPResearchTransportError, match="lifecycle failed"):
        MCPResearchProvider(factory).research(
            subquestion_index=0,
            subquestion=QUESTION,
        )


@pytest.mark.parametrize(
    "structured_content",
    [
        None,
        [],
        {"query": QUESTION},
        {"query": QUESTION, "results": [], "unexpected": "value"},
        {"query": 7, "results": []},
        {"query": QUESTION, "results": "not a list"},
        {"query": QUESTION, "results": ["not a dictionary"]},
        {
            "query": QUESTION,
            "results": [{"rank": 1, "title": "Only two fields"}],
        },
        {
            "query": QUESTION,
            "results": [{**_result(1), "unexpected": "value"}],
        },
        {
            "query": QUESTION,
            "results": [{**_result(1), "snippet": "  "}],
        },
    ],
    ids=[
        "missing",
        "non-dictionary",
        "missing-top-level-key",
        "unexpected-top-level-key",
        "non-string-query",
        "non-list-results",
        "non-dictionary-item",
        "missing-item-keys",
        "unexpected-item-key",
        "empty-item-text",
    ],
)
def test_missing_or_malformed_structured_content_is_rejected(
    structured_content: object | None,
) -> None:
    with pytest.raises(MCPResearchResponseError):
        map_search_tool_result(
            _tool_result(structured_content),
            requested_query=QUESTION,
            subquestion_index=0,
            max_results=2,
        )


def test_wrong_query_echo_is_rejected() -> None:
    with pytest.raises(MCPResearchResponseError, match="does not match"):
        map_search_tool_result(
            _tool_result(_structured_response(query="Different query")),
            requested_query=QUESTION,
            subquestion_index=0,
            max_results=2,
        )


def test_excess_results_are_rejected() -> None:
    with pytest.raises(MCPResearchResponseError, match="exceeds"):
        map_search_tool_result(
            _tool_result(_structured_response(results=[_result(1), _result(2)])),
            requested_query=QUESTION,
            subquestion_index=0,
            max_results=1,
        )


@pytest.mark.parametrize(
    "results",
    [
        [{**_result(1), "rank": True}],
        [{**_result(1), "rank": False}],
        [{**_result(1), "rank": 0}],
        [{**_result(1), "rank": -1}],
        [_result(1), {**_result(2), "rank": 1}],
        [_result(1), {**_result(2), "rank": 3}],
        [{**_result(1), "rank": 2}, {**_result(2), "rank": 1}],
    ],
    ids=[
        "boolean-true",
        "boolean-false",
        "zero",
        "negative",
        "duplicate",
        "skipped",
        "out-of-order",
    ],
)
def test_invalid_result_ranks_are_rejected(results: list[object]) -> None:
    with pytest.raises(MCPResearchResponseError, match="rank"):
        map_search_tool_result(
            _tool_result(_structured_response(results=results)),
            requested_query=QUESTION,
            subquestion_index=0,
            max_results=3,
        )


@pytest.mark.parametrize("value", [True, False, 0, -1])
def test_provider_rejects_invalid_boolean_or_nonpositive_max_results(
    value: int,
) -> None:
    with pytest.raises(MCPResearchConfigurationError, match="max_results"):
        MCPResearchProvider(
            create_search_server_factory(CATALOG, SearchCallRecorder()),
            max_results=value,
        )


@pytest.mark.parametrize("value", [True, False, -1])
def test_provider_rejects_invalid_subquestion_indexes(value: int) -> None:
    provider = MCPResearchProvider(
        create_search_server_factory(CATALOG, SearchCallRecorder())
    )
    with pytest.raises(MCPResearchConfigurationError, match="subquestion_index"):
        provider.research(subquestion_index=value, subquestion=QUESTION)


@pytest.mark.parametrize("max_results", [True, False, 0, -1])
def test_real_mcp_server_rejects_invalid_max_results(max_results: int) -> None:
    recorder = SearchCallRecorder()
    server = create_search_server_factory(CATALOG, recorder)()

    async def invoke() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool(
                "search_web",
                {"query": QUESTION, "max_results": max_results},
            )

    result = asyncio.run(invoke())

    assert result.is_error is True
    assert recorder.calls == []


@pytest.mark.parametrize("rank", [True, False])
def test_catalog_rejects_boolean_ranks(rank: int) -> None:
    with pytest.raises(ValueError, match="ranks"):
        create_search_server_factory(
            {QUESTION: [{**_result(1), "rank": rank}]},
            SearchCallRecorder(),
        )


def test_running_event_loop_is_rejected_before_server_or_coroutine_creation() -> None:
    factory_calls = 0

    def factory() -> MCPServer:
        nonlocal factory_calls
        factory_calls += 1
        return create_search_server_factory(CATALOG, SearchCallRecorder())()

    provider = MCPResearchProvider(factory)

    async def invoke() -> None:
        with pytest.raises(MCPResearchConfigurationError, match="active asyncio"):
            provider.research(subquestion_index=0, subquestion=QUESTION)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(invoke())
        gc.collect()

    assert factory_calls == 0
    assert not [
        warning
        for warning in caught
        if "was never awaited" in str(warning.message)
    ]


@pytest.mark.parametrize(
    "factory",
    [
        cast(Callable[[], MCPServer], 42),
        cast(Callable[[], MCPServer], lambda: object()),
        cast(
            Callable[[], MCPServer],
            lambda: (_ for _ in ()).throw(RuntimeError("factory failed")),
        ),
    ],
    ids=["not-callable", "wrong-return-type", "factory-raises"],
)
def test_invalid_server_factory_configuration_is_project_specific(
    factory: Callable[[], MCPServer],
) -> None:
    if callable(factory):
        provider = MCPResearchProvider(factory)
        operation: Callable[[], object] = lambda: provider.research(
            subquestion_index=0,
            subquestion=QUESTION,
        )
    else:
        operation = lambda: MCPResearchProvider(factory)

    with pytest.raises(MCPResearchConfigurationError):
        operation()


def test_expanded_graph_runs_one_real_mcp_tool_call_per_subquestion() -> None:
    recorder = SearchCallRecorder()
    provider = MCPResearchProvider(
        create_search_server_factory(CATALOG, recorder),
        max_results=2,
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

    assert sorted(recorder.calls) == sorted(
        [(SUBQUESTION_1, 2), (SUBQUESTION_2, 2)]
    )
    assert result["question_complexity"] == "complex"
    assert result["subquestions"] == [SUBQUESTION_1, SUBQUESTION_2]
    assert len(result["raw_research_findings"]) == 4
    assert len(result["consolidated_evidence"]) == 4
    assert [finding["subquestion_index"] for finding in result["consolidated_evidence"]] == [
        1,
        1,
        2,
        2,
    ]
    research = tuple(result["research_findings"])
    assert model.research_called is False
    assert model.argument_calls == [
        (TEAM_A_STANCE, research, None),
        (TEAM_B_STANCE, research, TEAM_A_ARGUMENT),
    ]
    assert model.synthesis_research == research
    assert result["team_a_argument"] == TEAM_A_ARGUMENT
    assert result["team_b_argument"] == TEAM_B_ARGUMENT
    assert result["synthesis"] == SYNTHESIS
