"""Provider-neutral synchronous search contracts and validation."""

from collections.abc import Sequence
from math import isfinite
from typing import NotRequired, Protocol, TypedDict, cast


class BackendSearchResult(TypedDict):
    """One ordered search result before mapping to the MCP contract."""

    title: str
    url: str
    snippet: str
    provider_score: NotRequired[float]


class SearchBackend(Protocol):
    """Return ordered search results for one query.

    Synchronous LangGraph workers may cause ``search`` to be called concurrently
    from separate threads. Implementations must therefore be thread-safe or
    isolate mutable resources within each call.
    """

    def search(
        self,
        *,
        query: str,
        max_results: int,
    ) -> Sequence[BackendSearchResult]:
        """Return at most ``max_results`` results in provider order."""
        ...


class SearchBackendError(RuntimeError):
    """Base exception exposed by project search backends."""


class SearchBackendConfigurationError(SearchBackendError):
    """Raised when backend configuration or invocation input is invalid."""


class SearchBackendTransportError(SearchBackendError):
    """Raised when a search request cannot complete its HTTP lifecycle."""


class SearchBackendHTTPError(SearchBackendError):
    """A bounded non-success HTTP status without response data."""

    def __init__(self, status_code: int) -> None:
        if type(status_code) is not int or not 100 <= status_code <= 599:
            raise SearchBackendConfigurationError(
                "HTTP status code must be an integer from 100 through 599"
            )
        self.status_code = status_code
        super().__init__(f"search backend returned HTTP {status_code}")


class SearchBackendResponseError(SearchBackendError):
    """Raised when a successful provider response violates the contract."""


_REQUIRED_RESULT_KEYS = frozenset({"title", "url", "snippet"})
_ALLOWED_RESULT_KEYS = _REQUIRED_RESULT_KEYS | {"provider_score"}


def validate_backend_search_results(
    values: object,
    *,
    max_results: int,
) -> list[BackendSearchResult]:
    """Validate, normalize, and copy backend results without reordering them."""

    if type(max_results) is not int or max_results <= 0:
        raise SearchBackendConfigurationError(
            "max_results must be a positive integer"
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SearchBackendResponseError("backend results must be a sequence")
    if len(values) > max_results:
        raise SearchBackendResponseError("backend returned too many results")

    validated: list[BackendSearchResult] = []
    for value in values:
        if not isinstance(value, dict):
            raise SearchBackendResponseError(
                "each backend result must be a dictionary"
            )

        keys = set(value)
        if not _REQUIRED_RESULT_KEYS <= keys or not keys <= _ALLOWED_RESULT_KEYS:
            raise SearchBackendResponseError(
                "backend result must contain exactly the required fields and "
                "optional provider_score"
            )

        for field in ("title", "url", "snippet"):
            field_value = value[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise SearchBackendResponseError(
                    f"backend result {field} must be a nonempty string"
                )

        copied: BackendSearchResult = {
            "title": cast(str, value["title"]),
            "url": cast(str, value["url"]),
            "snippet": cast(str, value["snippet"]),
        }
        if "provider_score" in value:
            score = value["provider_score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(score)
            ):
                raise SearchBackendResponseError(
                    "backend provider_score must be a finite number"
                )
            copied["provider_score"] = float(score)
        validated.append(copied)

    return validated
