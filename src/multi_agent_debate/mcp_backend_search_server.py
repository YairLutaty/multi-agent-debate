"""MCP search server factory backed by a provider-neutral search backend."""

from collections.abc import Callable
from typing import Annotated, TypeAlias

from mcp.server import MCPServer
from pydantic import Field

from multi_agent_debate.mcp_search_contract import SearchResult, SearchWebResponse
from multi_agent_debate.search_backend import (
    SearchBackend,
    SearchBackendConfigurationError,
    SearchBackendError,
    validate_backend_search_results,
)


_StrictPositiveInt: TypeAlias = Annotated[int, Field(strict=True, gt=0)]


def create_backend_search_server_factory(
    backend: SearchBackend,
) -> Callable[[], MCPServer]:
    """Return a factory for fresh MCP servers sharing a concurrency-safe backend."""

    if not callable(getattr(backend, "search", None)):
        raise SearchBackendConfigurationError(
            "backend must provide a callable search method"
        )

    def create_server() -> MCPServer:
        server = MCPServer(
            name="backend-search",
            description="Provider-neutral search backend exposed through MCP.",
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

            try:
                raw_results = backend.search(
                    query=query,
                    max_results=max_results,
                )
                results = validate_backend_search_results(
                    raw_results,
                    max_results=max_results,
                )
            except SearchBackendError:
                raise
            except Exception:
                raise RuntimeError(
                    "search backend failed unexpectedly"
                ) from None

            mcp_results: list[SearchResult] = [
                {
                    "rank": rank,
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "source_type": "web",
                }
                for rank, result in enumerate(results, start=1)
            ]
            return {
                "query": query,
                "results": mcp_results,
            }

        return server

    return create_server
