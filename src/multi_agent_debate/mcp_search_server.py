"""Deterministic in-memory MCP search server for Milestone 2.1B1."""

from collections.abc import Callable, Mapping, Sequence
from threading import Lock
from types import MappingProxyType
from typing import Annotated, TypeAlias

from mcp.server import MCPServer
from pydantic import Field

from multi_agent_debate.mcp_search_contract import (
    SEARCH_RESULT_KEYS,
    SearchResult,
    SearchWebResponse,
)


SearchCatalog: TypeAlias = Mapping[str, Sequence[SearchResult]]
ServerFactory: TypeAlias = Callable[[], MCPServer]

_SEARCH_TEXT_FIELDS = ("title", "url", "snippet", "source_type")
_StrictPositiveInt: TypeAlias = Annotated[int, Field(strict=True, gt=0)]


class SearchCallRecorder:
    """Thread-safe record of deterministic MCP tool invocations."""

    def __init__(self) -> None:
        self._calls: list[tuple[str, int]] = []
        self._lock = Lock()

    def record(self, query: str, max_results: int) -> None:
        with self._lock:
            self._calls.append((query, max_results))

    @property
    def calls(self) -> list[tuple[str, int]]:
        with self._lock:
            return self._calls.copy()


def create_search_server_factory(
    catalog: SearchCatalog,
    recorder: SearchCallRecorder,
) -> ServerFactory:
    """Return a factory that creates fresh deterministic MCP server objects."""

    frozen_catalog = _copy_and_validate_catalog(catalog)

    def create_server() -> MCPServer:
        server = MCPServer(
            name="deterministic-search",
            description="Offline deterministic search fixture for MCP integration.",
        )

        @server.tool(name="search_web")
        def search_web(
            query: str,
            max_results: _StrictPositiveInt,
        ) -> SearchWebResponse:
            if not query.strip():
                raise ValueError("query must be nonempty")
            if type(max_results) is not int or max_results <= 0:
                raise ValueError("max_results must be a positive integer")

            recorder.record(query, max_results)
            results = frozen_catalog.get(query, ())[:max_results]
            return {
                "query": query,
                "results": [result.copy() for result in results],
            }

        return server

    return create_server


def _copy_and_validate_catalog(
    catalog: SearchCatalog,
) -> Mapping[str, tuple[SearchResult, ...]]:
    copied: dict[str, tuple[SearchResult, ...]] = {}
    for query, results in catalog.items():
        if not isinstance(query, str) or not query.strip():
            raise ValueError("catalog queries must be nonempty strings")
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            raise ValueError("catalog results must be a sequence")
        copied[query] = tuple(
            _copy_and_validate_search_result(result, expected_rank=index)
            for index, result in enumerate(results, start=1)
        )
    return MappingProxyType(copied)


def _copy_and_validate_search_result(
    value: object,
    *,
    expected_rank: int,
) -> SearchResult:
    if not isinstance(value, dict):
        raise ValueError("catalog search result must be a dictionary")
    if set(value) != SEARCH_RESULT_KEYS:
        raise ValueError("catalog search result must contain exactly the required keys")

    rank = value["rank"]
    if type(rank) is not int or rank != expected_rank:
        raise ValueError("catalog ranks must be consecutive and one-based")
    for field in _SEARCH_TEXT_FIELDS:
        field_value = value[field]
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"catalog {field} must be a nonempty string")

    return {
        "rank": rank,
        "title": value["title"],
        "url": value["url"],
        "snippet": value["snippet"],
        "source_type": value["source_type"],
    }
