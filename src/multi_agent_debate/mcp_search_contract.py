"""Provider-neutral structured output contract for the MCP search_web tool."""

from typing import Annotated, TypedDict

from pydantic import Field


class SearchResult(TypedDict):
    rank: Annotated[int, Field(strict=True)]
    title: str
    url: str
    snippet: str
    source_type: str


class SearchWebResponse(TypedDict):
    query: str
    results: list[SearchResult]


SEARCH_RESULT_KEYS = frozenset(SearchResult.__required_keys__)
SEARCH_RESPONSE_KEYS = frozenset(SearchWebResponse.__required_keys__)
