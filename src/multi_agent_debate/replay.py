"""Exact-hash loading and deterministic providers for frozen debate replay."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TypedDict

from multi_agent_debate.expanded_state import DecompositionResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.research import (
    ResearchFinding,
    consolidate_research_findings,
    format_research_finding,
    validate_research_finding,
)
from multi_agent_debate.run_record import (
    ARCHITECTURE_VERSION,
    RUN_RECORD_VERSION,
    RunRecord,
    validate_run_record,
)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReplayError(RuntimeError):
    """Base exception for deterministic replay boundaries."""


class ReplaySourceError(ReplayError):
    """The source artifact could not be read or decoded safely."""


class ReplayIntegrityError(ReplayError):
    """The source bytes do not match the expected immutable artifact."""


class ReplayValidationError(ReplayError):
    """Replay data violates the source or provider contract."""


class ReplayProviderError(ReplayError):
    """A replay provider received an input outside the frozen task set."""


class ReplayResearchTask(TypedDict):
    subquestion_index: int
    query: str
    raw_research_findings: list[ResearchFinding]


class ReplaySource(TypedDict):
    source_artifact_path: str
    source_sha256: str
    source_record: RunRecord
    research_tasks: list[ReplayResearchTask]


_REPLAY_SOURCE_KEYS = frozenset(ReplaySource.__required_keys__)
_REPLAY_TASK_KEYS = frozenset(ReplayResearchTask.__required_keys__)


def expected_research_tasks(record: RunRecord) -> list[tuple[int, str]]:
    """Reconstruct the exact research routing identities from a v1.0 record."""

    if record["decomposition"]["question_complexity"] == "simple":
        return [(0, record["input"]["question"])]
    return list(enumerate(record["decomposition"]["subquestions"], start=1))


def load_replay_source(
    path: Path,
    *,
    expected_sha256: str,
) -> ReplaySource:
    """Load, authenticate, validate, and independently reconstruct a replay."""

    if not isinstance(path, Path):
        raise ReplayValidationError("replay source path must be a Path")
    if (
        not isinstance(expected_sha256, str)
        or SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ReplayValidationError("expected replay SHA-256 is invalid")
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        raise ReplaySourceError("replay source could not be read") from None

    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ReplayIntegrityError("replay source SHA-256 does not match")
    try:
        decoded = raw_bytes.decode("utf-8")
        parsed = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ReplaySourceError("replay source is not valid UTF-8 JSON") from None

    try:
        record = validate_run_record(parsed)
    except Exception:
        raise ReplayValidationError("replay source record is invalid") from None
    if record["run_record_version"] != RUN_RECORD_VERSION:
        raise ReplayValidationError("replay source version is incompatible")
    if record["provenance"]["architecture_version"] != ARCHITECTURE_VERSION:
        raise ReplayValidationError("replay source architecture is incompatible")
    if record["input"]["team_a_stance"] != TEAM_A_STANCE:
        raise ReplayValidationError("replay source Team A stance is incompatible")
    if record["input"]["team_b_stance"] != TEAM_B_STANCE:
        raise ReplayValidationError("replay source Team B stance is incompatible")

    research_tasks = _group_and_validate_research(record)
    source: ReplaySource = {
        "source_artifact_path": str(path),
        "source_sha256": actual_sha256,
        "source_record": record,
        "research_tasks": research_tasks,
    }
    return validate_replay_source(source)


def validate_replay_source(value: object) -> ReplaySource:
    """Strictly cross-check and deeply copy an in-memory replay source."""

    if not isinstance(value, dict) or set(value) != _REPLAY_SOURCE_KEYS:
        raise ReplayValidationError("replay source has an invalid schema")
    path = value["source_artifact_path"]
    digest = value["source_sha256"]
    if not isinstance(path, str) or not path.strip():
        raise ReplayValidationError("replay source artifact path is invalid")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ReplayValidationError("replay source SHA-256 is invalid")
    try:
        record = validate_run_record(value["source_record"])
    except Exception:
        raise ReplayValidationError("replay source record is invalid") from None
    if record["run_record_version"] != RUN_RECORD_VERSION:
        raise ReplayValidationError("replay source version is incompatible")
    if record["provenance"]["architecture_version"] != ARCHITECTURE_VERSION:
        raise ReplayValidationError("replay source architecture is incompatible")

    expected = _group_and_validate_research(record)
    raw_tasks = value["research_tasks"]
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(expected):
        raise ReplayValidationError("replay research tasks are incomplete")
    copied_tasks: list[ReplayResearchTask] = []
    for raw_task, expected_task in zip(raw_tasks, expected, strict=True):
        if not isinstance(raw_task, dict) or set(raw_task) != _REPLAY_TASK_KEYS:
            raise ReplayValidationError("replay research task has an invalid schema")
        if (
            type(raw_task["subquestion_index"]) is not int
            or raw_task["subquestion_index"] != expected_task["subquestion_index"]
            or raw_task["query"] != expected_task["query"]
        ):
            raise ReplayValidationError(
                "replay research tasks are duplicated, reordered, or unexpected"
            )
        findings = _validate_task_findings(
            raw_task["raw_research_findings"],
            expected_index=expected_task["subquestion_index"],
            expected_query=expected_task["query"],
        )
        if findings != expected_task["raw_research_findings"]:
            raise ReplayValidationError("replay research task findings have drifted")
        copied_tasks.append(
            {
                "subquestion_index": expected_task["subquestion_index"],
                "query": expected_task["query"],
                "raw_research_findings": [item.copy() for item in findings],
            }
        )
    return {
        "source_artifact_path": path,
        "source_sha256": digest,
        "source_record": record,
        "research_tasks": copied_tasks,
    }


def _group_and_validate_research(record: RunRecord) -> list[ReplayResearchTask]:
    expected_tasks = expected_research_tasks(record)
    expected_set = set(expected_tasks)
    search_identities = [
        (item["subquestion_index"], item["query"])
        for item in record["search_evidence"]
    ]
    if search_identities != expected_tasks or len(set(search_identities)) != len(
        search_identities
    ):
        raise ReplayValidationError(
            "replay search tasks are missing, duplicated, reordered, or unexpected"
        )

    grouped: dict[tuple[int, str], list[ResearchFinding]] = {
        task: [] for task in expected_tasks
    }
    for raw_finding in record["graph_evidence"]["raw_research_findings"]:
        try:
            finding = validate_research_finding(raw_finding)
        except ValueError:
            raise ReplayValidationError(
                "replay source contains an invalid raw finding"
            ) from None
        identity = (finding["subquestion_index"], finding["subquestion"])
        if identity not in expected_set:
            raise ReplayValidationError("replay source contains an unexpected finding")
        grouped[identity].append(finding)
    if any(not grouped[task] for task in expected_tasks):
        raise ReplayValidationError("replay source is missing a research task")

    consolidated = consolidate_research_findings(
        question=record["input"]["question"],
        question_complexity=record["decomposition"]["question_complexity"],
        subquestions=record["decomposition"]["subquestions"],
        raw_findings=record["graph_evidence"]["raw_research_findings"],
    )
    if consolidated != record["graph_evidence"]["consolidated_evidence"]:
        raise ReplayValidationError("replay source consolidation has drifted")
    formatted = [format_research_finding(item) for item in consolidated]
    if formatted != record["graph_evidence"]["research_findings"]:
        raise ReplayValidationError("replay source formatting has drifted")

    return [
        {
            "subquestion_index": index,
            "query": query,
            "raw_research_findings": [item.copy() for item in grouped[(index, query)]],
        }
        for index, query in expected_tasks
    ]


def _validate_task_findings(
    value: object,
    *,
    expected_index: int,
    expected_query: str,
) -> list[ResearchFinding]:
    if not isinstance(value, list) or not value:
        raise ReplayValidationError("replay task findings must be a nonempty list")
    copied: list[ResearchFinding] = []
    for item in value:
        try:
            copied.append(
                validate_research_finding(
                    item,
                    expected_subquestion_index=expected_index,
                    expected_subquestion=expected_query,
                )
            )
        except ValueError:
            raise ReplayValidationError("replay task finding is invalid") from None
    return copied


class ReplayQuestionDecomposer:
    """Return only the frozen classification for the exact source question."""

    def __init__(self, replay_source: ReplaySource) -> None:
        source = validate_replay_source(replay_source)
        record = source["source_record"]
        self._question = record["input"]["question"]
        self._result: DecompositionResult = {
            "question_complexity": record["decomposition"]["question_complexity"],
            "subquestions": record["decomposition"]["subquestions"].copy(),
        }

    def decompose(self, *, question: str) -> DecompositionResult:
        if question != self._question:
            raise ReplayProviderError("replay decomposer received an unknown question")
        return {
            "question_complexity": self._result["question_complexity"],
            "subquestions": self._result["subquestions"].copy(),
        }


class ReplayResearchProvider:
    """Immutable lookup provider safe for concurrent LangGraph Send workers."""

    def __init__(self, replay_source: ReplaySource) -> None:
        source = validate_replay_source(replay_source)
        task_map = {
            (task["subquestion_index"], task["query"]): tuple(
                copy.deepcopy(task["raw_research_findings"])
            )
            for task in source["research_tasks"]
        }
        if len(task_map) != len(source["research_tasks"]):
            raise ReplayValidationError("replay research tasks contain duplicates")
        self._findings_by_task = MappingProxyType(task_map)

    def research(
        self,
        *,
        subquestion_index: int,
        subquestion: str,
    ) -> Sequence[ResearchFinding]:
        if type(subquestion_index) is not int or not isinstance(subquestion, str):
            raise ReplayProviderError("replay research task identity is invalid")
        key = (subquestion_index, subquestion)
        if key not in self._findings_by_task:
            raise ReplayProviderError("replay research provider received an unknown task")
        return [item.copy() for item in self._findings_by_task[key]]
