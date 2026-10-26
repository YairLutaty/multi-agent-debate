"""Synchronous ResearchProvider adapter over the MCP v2 high-level client."""

import asyncio
from collections.abc import Callable
from typing import cast

from mcp import Client
from mcp.server import MCPServer
from mcp.types import CallToolResult

from multi_agent_debate.mcp_search_contract import (
    SEARCH_RESPONSE_KEYS,
    SEARCH_RESULT_KEYS,
)
from multi_agent_debate.research import ResearchFinding

DEFAULT_MAX_RESULTS = 3
SEARCH_TOOL_NAME = "search_web"
_RESULT_TEXT_FIELDS = ("title", "url", "snippet", "source_type")

ServerFactory = Callable[[], MCPServer]


class MCPResearchError(RuntimeError):
    """Base exception exposed by the MCP research adapter."""


class MCPResearchConfigurationError(MCPResearchError):
    """Raised for invalid provider settings or invocation context."""


class MCPResearchTransportError(MCPResearchError):
    """Raised when the MCP client/server lifecycle cannot be completed."""


class MCPResearchProtocolError(MCPResearchError):
    """Raised when the MCP call or SDK result validation raises."""


class MCPResearchToolError(MCPResearchError):
    """Raised when search_web returns an MCP error result."""


class MCPResearchResponseError(MCPResearchError):
    """Raised when structured search output violates the domain contract."""


class MCPResearchProvider:
    """Obtain deterministic structured research through isolated MCP clients."""

    def __init__(
        self,
        server_factory: ServerFactory,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        if not callable(server_factory):
            raise MCPResearchConfigurationError("server_factory must be callable")
        if type(max_results) is not int or max_results <= 0:
            raise MCPResearchConfigurationError(
                "max_results must be a positive integer"
            )
        self._server_factory = server_factory
        self._max_results = max_results

    def research(
        self,
        *,
        subquestion_index: int,
        subquestion: str,
    ) -> list[ResearchFinding]:
        """Run one fresh in-memory MCP lifecycle for one research question."""

        if type(subquestion_index) is not int or subquestion_index < 0:
            raise MCPResearchConfigurationError(
                "subquestion_index must be a nonnegative integer"
            )
        if not isinstance(subquestion, str) or not subquestion.strip():
            raise MCPResearchConfigurationError(
                "subquestion must be a nonempty string"
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise MCPResearchConfigurationError(
                "research cannot run in a thread with an active asyncio event loop"
            )

        try:
            server = self._server_factory()
        except Exception as error:
            raise MCPResearchConfigurationError(
                "server_factory failed to create an MCP server"
            ) from error
        if not isinstance(server, MCPServer):
            raise MCPResearchConfigurationError(
                "server_factory must return a fresh MCPServer"
            )

        try:
            return asyncio.run(
                self._research_once(
                    server=server,
                    subquestion_index=subquestion_index,
                    subquestion=subquestion,
                )
            )
        except MCPResearchError:
            raise
        except Exception as error:
            raise MCPResearchProtocolError(
                "unexpected failure during MCP research"
            ) from error

    async def _research_once(
        self,
        *,
        server: MCPServer,
        subquestion_index: int,
        subquestion: str,
    ) -> list[ResearchFinding]:
        result = await _call_search_tool(
            server=server,
            query=subquestion,
            max_results=self._max_results,
        )
        return map_search_tool_result(
            result,
            requested_query=subquestion,
            subquestion_index=subquestion_index,
            max_results=self._max_results,
        )


async def _call_search_tool(
    *,
    server: MCPServer,
    query: str,
    max_results: int,
) -> CallToolResult:
    try:
        async with Client(server, raise_exceptions=False) as client:
            try:
                return await client.call_tool(
                    SEARCH_TOOL_NAME,
                    {
                        "query": query,
                        "max_results": max_results,
                    },
                )
            except Exception as error:
                raise MCPResearchProtocolError(
                    "MCP search_web call failed"
                ) from error
    except MCPResearchProtocolError:
        raise
    except Exception as error:
        raise MCPResearchTransportError(
            "MCP client/server lifecycle failed"
        ) from error


def map_search_tool_result(
    result: CallToolResult,
    *,
    requested_query: str,
    subquestion_index: int,
    max_results: int,
) -> list[ResearchFinding]:
    """Validate one MCP result and map it to fresh domain findings."""

    if result.is_error:
        raise MCPResearchToolError("search_web returned an MCP error result")
    structured = result.structured_content
    if structured is None:
        raise MCPResearchResponseError(
            "search_web result is missing structured content"
        )
    if not isinstance(structured, dict):
        raise MCPResearchResponseError(
            "search_web structured content must be a dictionary"
        )
    _require_exact_keys(
        structured,
        set(SEARCH_RESPONSE_KEYS),
        context="search response",
    )

    returned_query = structured["query"]
    if not isinstance(returned_query, str):
        raise MCPResearchResponseError("search response query must be a string")
    if returned_query != requested_query:
        raise MCPResearchResponseError("search response query does not match request")

    results = structured["results"]
    if not isinstance(results, list):
        raise MCPResearchResponseError("search response results must be a list")
    if len(results) > max_results:
        raise MCPResearchResponseError("search response exceeds max_results")

    findings: list[ResearchFinding] = []
    for expected_rank, raw_result in enumerate(results, start=1):
        if not isinstance(raw_result, dict):
            raise MCPResearchResponseError("search result must be a dictionary")
        _require_exact_keys(
            raw_result,
            set(SEARCH_RESULT_KEYS),
            context="search result",
        )

        rank = raw_result["rank"]
        # MCP's strict output schema is the primary Boolean-coercion boundary;
        # this exact-type check remains a defense-in-depth backstop.
        if type(rank) is not int:
            raise MCPResearchResponseError("search result rank must be an integer")
        if rank != expected_rank:
            raise MCPResearchResponseError(
                "search result ranks must be consecutive and one-based"
            )
        for field in _RESULT_TEXT_FIELDS:
            value = raw_result[field]
            if not isinstance(value, str) or not value.strip():
                raise MCPResearchResponseError(
                    f"search result {field} must be a nonempty string"
                )

        snippet = cast(str, raw_result["snippet"])
        findings.append(
            {
                "subquestion_index": subquestion_index,
                "subquestion": requested_query,
                "finding_index": rank,
                "claim": snippet,
                "source_title": cast(str, raw_result["title"]),
                "source_url": cast(str, raw_result["url"]),
                "source_type": cast(str, raw_result["source_type"]),
                "supporting_excerpt": snippet,
            }
        )
    return findings


def _require_exact_keys(
    value: dict[object, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(repr(key) for key in actual - expected)
        raise MCPResearchResponseError(
            f"{context} has invalid keys; "
            f"missing={missing}, unexpected={unexpected}"
        )
