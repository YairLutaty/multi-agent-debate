"""Synchronous Tavily Search HTTP backend with strict safe boundaries."""

import os
from collections.abc import Callable, Mapping
from math import isfinite
from typing import cast

import httpx

from multi_agent_debate.search_backend import (
    BackendSearchResult,
    SearchBackendConfigurationError,
    SearchBackendHTTPError,
    SearchBackendResponseError,
    SearchBackendTransportError,
    validate_backend_search_results,
)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 10.0

ClientFactory = Callable[[], httpx.Client]


def _default_client_factory() -> httpx.Client:
    return httpx.Client(
        trust_env=False,
        follow_redirects=False,
    )


class TavilySearchBackend:
    """Tavily implementation using one fresh synchronous client per search."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: ClientFactory | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise SearchBackendConfigurationError(
                "Tavily API key must be a nonempty string"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise SearchBackendConfigurationError(
                "timeout_seconds must be a finite positive number"
            )
        resolved_factory = (
            _default_client_factory if client_factory is None else client_factory
        )
        if not callable(resolved_factory):
            raise SearchBackendConfigurationError(
                "client_factory must be callable"
            )

        self._api_key = api_key
        self._timeout_seconds = float(timeout_seconds)
        self._client_factory = resolved_factory

    @classmethod
    def from_environment(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: ClientFactory | None = None,
    ) -> "TavilySearchBackend":
        """Construct explicitly from ``TAVILY_API_KEY`` without import-time reads."""

        source = os.environ if environ is None else environ
        return cls(
            api_key=source.get(TAVILY_API_KEY_ENV, ""),
            timeout_seconds=timeout_seconds,
            client_factory=client_factory,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"timeout_seconds={self._timeout_seconds!r})"
        )

    def search(
        self,
        *,
        query: str,
        max_results: int,
    ) -> list[BackendSearchResult]:
        """Make exactly one Tavily request and return validated ordered results."""

        if not isinstance(query, str) or not query.strip():
            raise SearchBackendConfigurationError(
                "query must be a nonempty string"
            )
        if type(max_results) is not int or max_results <= 0:
            raise SearchBackendConfigurationError(
                "max_results must be a positive integer"
            )

        try:
            client = self._client_factory()
        except Exception:
            raise SearchBackendConfigurationError(
                "client_factory failed to create an HTTP client"
            ) from None
        if not isinstance(client, httpx.Client):
            raise SearchBackendConfigurationError(
                "client_factory must return an httpx.Client"
            )

        try:
            with client:
                response = client.post(
                    TAVILY_SEARCH_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "search_depth": "basic",
                        "max_results": max_results,
                        "include_answer": False,
                        "include_raw_content": False,
                        "include_images": False,
                    },
                    timeout=self._timeout_seconds,
                )
        except httpx.TimeoutException:
            raise SearchBackendTransportError(
                "Tavily search request timed out"
            ) from None
        except httpx.RequestError:
            raise SearchBackendTransportError(
                "Tavily search request failed"
            ) from None
        except Exception:
            raise SearchBackendTransportError(
                "Tavily HTTP client lifecycle failed"
            ) from None

        if not 200 <= response.status_code < 300:
            raise SearchBackendHTTPError(response.status_code)

        try:
            payload = response.json()
        except (ValueError, UnicodeError):
            raise SearchBackendResponseError(
                "Tavily response did not contain valid JSON"
            ) from None

        return _parse_tavily_response(
            payload,
            requested_query=query,
            max_results=max_results,
        )


def _parse_tavily_response(
    payload: object,
    *,
    requested_query: str,
    max_results: int,
) -> list[BackendSearchResult]:
    if not isinstance(payload, dict):
        raise SearchBackendResponseError(
            "Tavily response must be a dictionary"
        )

    returned_query = payload.get("query")
    if not isinstance(returned_query, str):
        raise SearchBackendResponseError(
            "Tavily response query must be a string"
        )
    if returned_query != requested_query:
        raise SearchBackendResponseError(
            "Tavily response query does not match the request"
        )

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise SearchBackendResponseError(
            "Tavily response results must be a list"
        )
    if len(raw_results) > max_results:
        raise SearchBackendResponseError(
            "Tavily response contains too many results"
        )

    mapped: list[BackendSearchResult] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise SearchBackendResponseError(
                "each Tavily result must be a dictionary"
            )
        for field in ("title", "url", "content"):
            value = raw_result.get(field)
            if not isinstance(value, str) or not value.strip():
                raise SearchBackendResponseError(
                    f"Tavily result {field} must be a nonempty string"
                )

        score = raw_result.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(score)
        ):
            raise SearchBackendResponseError(
                "Tavily result score must be a finite number"
            )

        mapped.append(
            {
                "title": cast(str, raw_result["title"]),
                "url": cast(str, raw_result["url"]),
                "snippet": cast(str, raw_result["content"]),
                "provider_score": float(score),
            }
        )

    return validate_backend_search_results(mapped, max_results=max_results)
