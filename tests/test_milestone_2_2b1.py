"""Offline tests for deterministic replay and comparison infrastructure."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.metrics import scorecard_to_json
from multi_agent_debate.replay import (
    ReplayIntegrityError,
    ReplayProviderError,
    ReplayQuestionDecomposer,
    ReplayResearchProvider,
    ReplaySource,
    ReplaySourceError,
    ReplayValidationError,
    expected_research_tasks,
    load_replay_source,
    validate_replay_source,
)
from multi_agent_debate.replay_run_record import (
    CONTROL_ARCHITECTURE_VERSION,
    HIERARCHICAL_ARCHITECTURE_VERSION,
    REPLAY_RUN_RECORD_VERSION,
    ReplayRunRecordValidationError,
    replay_run_record_to_json,
    validate_replay_run_record,
)
from multi_agent_debate.replay_runner import run_replay_variant
from multi_agent_debate.research import ResearchFinding, format_research_finding
from multi_agent_debate.run_record import score_expanded_result, validate_run_record
from multi_agent_debate.team_trace import TeamTraceRecorder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "experiments/runs/m2_1b2b_ubi_qwen3_8b.json"
SOURCE_SHA256 = "6d88274129f2ae6d1f708847d300a00f76f953bc1d42fbd493bef7f190c506c7"
IMPLEMENTATION_COMMIT = "a" * 40
MODEL_NAME = "qwen3:8b"
RUN_TIME = datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc)
CONTROL_TEAM_A = "Control Team A gives a focused evidence based supporting argument."
CONTROL_TEAM_B = "Control Team B gives a focused evidence based opposing argument."
HIERARCHICAL_TEAM_A_DRAFT = (
    "Hierarchical Team A drafts a focused evidence based supporting argument."
)
HIERARCHICAL_TEAM_A_FINAL = (
    "Hierarchical Team A delivers a revised evidence based supporting argument."
)
HIERARCHICAL_TEAM_B_DRAFT = (
    "Hierarchical Team B drafts a focused evidence based opposing response."
)
HIERARCHICAL_TEAM_B_FINAL = (
    "Hierarchical Team B delivers a revised evidence based opposing response."
)


class IncrementingClock:
    def __init__(self, step: float = 0.25) -> None:
        self._value = 0.0
        self._step = step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self._value += self._step
            return value


class DeterministicControlModel:
    def __init__(self) -> None:
        self.argument_calls: list[dict[str, object]] = []
        self.research_called = False
        self.synthesis_called = False

    def research(self, *, question: str) -> list[str]:
        self.research_called = True
        raise AssertionError("replay control must not use model research")

    def write_argument(self, **values: object) -> str:
        self.argument_calls.append(copy.deepcopy(values))
        return (
            CONTROL_TEAM_A
            if values["opponent_argument"] is None
            else CONTROL_TEAM_B
        )

    def synthesize_neutrally(self, **values: object) -> str:
        self.synthesis_called = True
        raise AssertionError("the injected shared synthesis model must be used")


class DeterministicTeamModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def analyze_evidence(self, **values: object) -> dict[str, object]:
        self.calls.append(("analyze_evidence", copy.deepcopy(values)))
        finding = cast(list[ResearchFinding], values["consolidated_evidence"])[0]
        return {
            "strongest_supporting_evidence": [
                {
                    "subquestion_index": finding["subquestion_index"],
                    "finding_index": finding["finding_index"],
                    "relevance": "This finding directly informs the assigned stance.",
                }
            ],
            "limitations_or_conflicting_evidence": [],
            "key_claims_to_make": ["Use a grounded and bounded central claim."],
            "unsupported_claims_to_avoid": ["Avoid claims beyond replay evidence."],
            "opponent_points_to_address": (
                []
                if values["team_name"] == "team_a"
                else ["Address Team A's exact final argument."]
            ),
        }

    def draft_argument(self, **values: object) -> str:
        self.calls.append(("draft_argument", copy.deepcopy(values)))
        return (
            HIERARCHICAL_TEAM_A_DRAFT
            if values["team_name"] == "team_a"
            else HIERARCHICAL_TEAM_B_DRAFT
        )

    def review_and_revise(self, **values: object) -> dict[str, object]:
        self.calls.append(("review_and_revise", copy.deepcopy(values)))
        revised = (
            HIERARCHICAL_TEAM_A_FINAL
            if values["team_name"] == "team_a"
            else HIERARCHICAL_TEAM_B_FINAL
        )
        return {
            "critique": {
                "stance_consistency": "The stance remains consistent.",
                "evidence_usage": "The draft uses supplied evidence.",
                "unsupported_factual_claims": "No unsupported facts remain.",
                "opponent_response": "The opponent is addressed appropriately.",
                "unnecessary_repetition": "Repetition is limited.",
                "clarity_and_concision": "The revision is clear and concise.",
            },
            "revised_argument": revised,
        }


class DeterministicSynthesisModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def synthesize_neutrally(self, **values: object) -> str:
        self.calls.append(copy.deepcopy(values))
        return (
            "Neutral synthesis compares the supporting argument with the opposing "
            "argument using the shared replay evidence."
        )


@pytest.fixture(scope="module")
def replay_source() -> ReplaySource:
    return load_replay_source(SOURCE_PATH, expected_sha256=SOURCE_SHA256)


@pytest.fixture()
def paired_runs(
    replay_source: ReplaySource,
) -> dict[str, object]:
    control_model = DeterministicControlModel()
    team_model = DeterministicTeamModel()
    synthesis_model = DeterministicSynthesisModel()
    control = run_replay_variant(
        variant="control",
        replay_source=replay_source,
        control_model=control_model,
        team_model=None,
        synthesis_model=synthesis_model,
        team_trace_recorder=None,
        clock=IncrementingClock(),
        utc_now=lambda: RUN_TIME,
        implementation_commit=IMPLEMENTATION_COMMIT,
        model_name=MODEL_NAME,
    )
    hierarchical = run_replay_variant(
        variant="hierarchical",
        replay_source=replay_source,
        control_model=None,
        team_model=team_model,
        synthesis_model=synthesis_model,
        team_trace_recorder=TeamTraceRecorder(),
        clock=IncrementingClock(),
        utc_now=lambda: RUN_TIME,
        implementation_commit=IMPLEMENTATION_COMMIT,
        model_name=MODEL_NAME,
    )
    return {
        "control": control,
        "hierarchical": hierarchical,
        "control_model": control_model,
        "team_model": team_model,
        "synthesis_model": synthesis_model,
    }


def _write_json_source(tmp_path: Path, value: object) -> tuple[Path, str]:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    path = tmp_path / "source.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_exact_source_sha_is_accepted_and_source_is_unchanged(
    replay_source: ReplaySource,
) -> None:
    before = SOURCE_PATH.read_bytes()
    assert hashlib.sha256(before).hexdigest() == SOURCE_SHA256
    assert replay_source["source_sha256"] == SOURCE_SHA256
    assert replay_source["source_record"]["run_record_version"] == "1.0"
    assert SOURCE_PATH.read_bytes() == before


def test_incorrect_source_sha_is_rejected() -> None:
    with pytest.raises(ReplayIntegrityError):
        load_replay_source(SOURCE_PATH, expected_sha256="0" * 64)


def test_malformed_json_is_rejected_without_parser_details(tmp_path: Path) -> None:
    raw = b'{"private": invalid}'
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)
    with pytest.raises(ReplaySourceError) as caught:
        load_replay_source(path, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_record_version", "2.0"),
        ("architecture_version", "milestone-other"),
    ],
)
def test_incompatible_source_version_or_architecture_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    raw = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if field == "run_record_version":
        raw[field] = value
    else:
        raw["provenance"][field] = value
    path, digest = _write_json_source(tmp_path, raw)
    with pytest.raises(ReplayValidationError):
        load_replay_source(path, expected_sha256=digest)


def test_loader_invokes_frozen_v1_validator() -> None:
    with patch(
        "multi_agent_debate.replay.validate_run_record",
        wraps=validate_run_record,
    ) as validator:
        load_replay_source(SOURCE_PATH, expected_sha256=SOURCE_SHA256)
    assert validator.call_count >= 1


def test_source_preserves_exact_input_decomposition_and_tasks(
    replay_source: ReplaySource,
) -> None:
    record = replay_source["source_record"]
    assert record["input"] == {
        "question": "Should governments provide universal basic income to all adult citizens?",
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
    }
    assert record["decomposition"]["question_complexity"] == "complex"
    expected = list(enumerate(record["decomposition"]["subquestions"], start=1))
    assert expected_research_tasks(record) == expected
    assert [
        (task["subquestion_index"], task["query"])
        for task in replay_source["research_tasks"]
    ] == expected
    assert [len(task["raw_research_findings"]) for task in replay_source["research_tasks"]] == [3, 3, 3]


def test_source_reproduces_consolidation_and_formatting(
    replay_source: ReplaySource,
) -> None:
    graph = replay_source["source_record"]["graph_evidence"]
    assert graph["consolidated_evidence"] == graph["raw_research_findings"]
    assert graph["research_findings"] == [
        format_research_finding(item) for item in graph["consolidated_evidence"]
    ]


@pytest.mark.parametrize("tamper", ["stance", "formatted"])
def test_source_content_tampering_is_rejected(
    tmp_path: Path,
    tamper: str,
) -> None:
    raw = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if tamper == "stance":
        raw["input"]["team_a_stance"] = "Different stance"
    else:
        raw["graph_evidence"]["research_findings"][0] = "Different formatting"
    path, digest = _write_json_source(tmp_path, raw)
    with pytest.raises(ReplayValidationError):
        load_replay_source(path, expected_sha256=digest)


def test_tampered_question_bytes_fail_the_frozen_expected_hash(
    tmp_path: Path,
) -> None:
    raw = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    raw["input"]["question"] = "Different question"
    path, _ = _write_json_source(tmp_path, raw)
    with pytest.raises(ReplayIntegrityError):
        load_replay_source(path, expected_sha256=SOURCE_SHA256)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "unexpected"])
def test_in_memory_replay_source_rejects_task_identity_drift(
    replay_source: ReplaySource,
    mutation: str,
) -> None:
    changed = copy.deepcopy(replay_source)
    tasks = changed["research_tasks"]
    if mutation == "missing":
        tasks.pop()
    elif mutation == "duplicate":
        tasks[1] = copy.deepcopy(tasks[0])
    elif mutation == "reordered":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    else:
        tasks[0]["query"] = "Unexpected query"
    with pytest.raises(ReplayValidationError):
        validate_replay_source(changed)


def test_replay_decomposer_is_exact_and_defensive(
    replay_source: ReplaySource,
) -> None:
    decomposer = ReplayQuestionDecomposer(replay_source)
    question = replay_source["source_record"]["input"]["question"]
    first = decomposer.decompose(question=question)
    first["subquestions"][0] = "mutated"
    second = decomposer.decompose(question=question)
    assert second == replay_source["source_record"]["decomposition"]
    with pytest.raises(ReplayProviderError):
        decomposer.decompose(question=question + " ")


def test_replay_provider_is_exact_and_returns_defensive_copies(
    replay_source: ReplaySource,
) -> None:
    provider = ReplayResearchProvider(replay_source)
    task = replay_source["research_tasks"][0]
    first = list(
        provider.research(
            subquestion_index=task["subquestion_index"],
            subquestion=task["query"],
        )
    )
    first[0]["claim"] = "mutated"
    second = list(
        provider.research(
            subquestion_index=task["subquestion_index"],
            subquestion=task["query"],
        )
    )
    assert second == task["raw_research_findings"]
    with pytest.raises(ReplayProviderError):
        provider.research(subquestion_index=99, subquestion=task["query"])
    with pytest.raises(ReplayProviderError):
        provider.research(
            subquestion_index=task["subquestion_index"],
            subquestion=task["query"] + " ",
        )


def test_replay_provider_is_concurrent_and_filesystem_free_after_construction(
    replay_source: ReplaySource,
) -> None:
    provider = ReplayResearchProvider(replay_source)
    tasks = replay_source["research_tasks"]
    with patch.object(Path, "read_bytes", side_effect=AssertionError("filesystem")):
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    provider.research,
                    subquestion_index=task["subquestion_index"],
                    subquestion=task["query"],
                )
                for task in reversed(tasks)
            ]
            results = [future.result() for future in futures]
    by_query = {
        findings[0]["subquestion"]: findings for findings in results
    }
    assert set(by_query) == {task["query"] for task in tasks}
    assert all(len(findings) == 3 for findings in results)


def test_both_variants_use_byte_equivalent_replay_evidence(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
) -> None:
    control = cast(dict[str, Any], paired_runs["control"])["run_record"]
    hierarchical = cast(dict[str, Any], paired_runs["hierarchical"])["run_record"]
    source_graph = replay_source["source_record"]["graph_evidence"]
    for record in (control, hierarchical):
        assert record["graph_evidence"]["raw_research_findings"] == source_graph[
            "raw_research_findings"
        ]
        assert record["graph_evidence"]["consolidated_evidence"] == source_graph[
            "consolidated_evidence"
        ]
        assert record["graph_evidence"]["research_findings"] == source_graph[
            "research_findings"
        ]
    canonical = lambda value: json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert canonical(control["graph_evidence"]["consolidated_evidence"]) == canonical(
        hierarchical["graph_evidence"]["consolidated_evidence"]
    )


def test_synthesis_is_regenerated_twice_by_the_same_implementation(
    paired_runs: dict[str, object],
) -> None:
    synthesis = cast(DeterministicSynthesisModel, paired_runs["synthesis_model"])
    assert len(synthesis.calls) == 2
    assert synthesis.calls[0]["research_findings"] == synthesis.calls[1][
        "research_findings"
    ]
    assert synthesis.calls[0]["team_a_argument"] == CONTROL_TEAM_A
    assert synthesis.calls[1]["team_a_argument"] == HIERARCHICAL_TEAM_A_FINAL


def test_control_call_counts_and_argument_routing_are_actual(
    paired_runs: dict[str, object],
) -> None:
    artifact = cast(dict[str, Any], paired_runs["control"])
    model = cast(DeterministicControlModel, paired_runs["control_model"])
    assert len(model.argument_calls) == 2
    assert model.argument_calls[0]["opponent_argument"] is None
    assert model.argument_calls[1]["opponent_argument"] == CONTROL_TEAM_A
    assert model.research_called is False
    assert model.synthesis_called is False
    assert artifact["run_record"]["model_call_counts"] == {
        "replay_decomposition": 0,
        "replay_research": 0,
        "team_a_argument": 1,
        "team_b_argument": 1,
        "team_a_analysis": 0,
        "team_a_draft": 0,
        "team_a_review": 0,
        "team_b_analysis": 0,
        "team_b_draft": 0,
        "team_b_review": 0,
        "synthesis": 1,
        "total_model_calls": 3,
    }


def test_hierarchical_call_counts_routing_and_work_products_are_actual(
    paired_runs: dict[str, object],
) -> None:
    artifact = cast(dict[str, Any], paired_runs["hierarchical"])
    model = cast(DeterministicTeamModel, paired_runs["team_model"])
    assert [stage for stage, _ in model.calls] == [
        "analyze_evidence",
        "draft_argument",
        "review_and_revise",
        "analyze_evidence",
        "draft_argument",
        "review_and_revise",
    ]
    analyses = [values for stage, values in model.calls if stage == "analyze_evidence"]
    assert analyses[0]["opponent_argument"] is None
    assert analyses[1]["opponent_argument"] == HIERARCHICAL_TEAM_A_FINAL
    record = artifact["run_record"]
    assert record["model_call_counts"]["total_model_calls"] == 7
    assert record["model_call_counts"]["replay_decomposition"] == 0
    assert record["model_call_counts"]["replay_research"] == 0
    assert record["team_execution"]["team_a"]["final_argument"] == record[
        "graph_evidence"
    ]["team_a_argument"]
    assert record["team_execution"]["team_b"]["final_argument"] == record[
        "graph_evidence"
    ]["team_b_argument"]


def test_control_and_hierarchical_v11_records_validate(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
) -> None:
    for variant, architecture in (
        ("control", CONTROL_ARCHITECTURE_VERSION),
        ("hierarchical", HIERARCHICAL_ARCHITECTURE_VERSION),
    ):
        record = cast(dict[str, Any], paired_runs[variant])["run_record"]
        validated = validate_replay_run_record(record, replay_source=replay_source)
        assert validated["run_record_version"] == REPLAY_RUN_RECORD_VERSION
        assert validated["provenance"]["architecture_version"] == architecture
        assert validated["replay_source"]["source_sha256"] == SOURCE_SHA256


@pytest.mark.parametrize(
    "field",
    [
        "provenance",
        "replay_source",
        "input",
        "decomposition",
        "source_search_evidence",
        "graph_evidence",
        "team_execution",
        "timings",
        "model_call_counts",
        "metrics",
    ],
)
def test_every_required_top_level_field_is_enforced(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
    field: str,
) -> None:
    record = copy.deepcopy(cast(dict[str, Any], paired_runs["control"])["run_record"])
    record.pop(field)
    with pytest.raises(ReplayRunRecordValidationError):
        validate_replay_run_record(record, replay_source=replay_source)


@pytest.mark.parametrize(
    "path",
    [
        ("provenance", "model", "name"),
        ("replay_source", "source_sha256"),
        ("graph_evidence", "synthesis"),
        ("team_execution", "team_a", "duration_seconds"),
        ("timings", "synthesis_duration_seconds"),
        ("model_call_counts", "synthesis"),
    ],
)
def test_required_nested_fields_are_enforced(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
    path: tuple[str, ...],
) -> None:
    record = copy.deepcopy(cast(dict[str, Any], paired_runs["control"])["run_record"])
    target: dict[str, Any] = record
    for component in path[:-1]:
        target = target[component]
    target.pop(path[-1])
    with pytest.raises(ReplayRunRecordValidationError):
        validate_replay_run_record(record, replay_source=replay_source)


@pytest.mark.parametrize(
    "tamper",
    ["question", "decomposition", "raw", "formatted", "argument", "opponent"],
)
def test_cross_field_tampering_is_rejected(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
    tamper: str,
) -> None:
    record = copy.deepcopy(
        cast(dict[str, Any], paired_runs["hierarchical"])["run_record"]
    )
    if tamper == "question":
        record["input"]["question"] = "different"
    elif tamper == "decomposition":
        record["decomposition"]["subquestions"].reverse()
    elif tamper == "raw":
        record["graph_evidence"]["raw_research_findings"].pop()
    elif tamper == "formatted":
        record["graph_evidence"]["research_findings"][0] = "different"
    elif tamper == "argument":
        record["graph_evidence"]["team_a_argument"] = "Different final argument."
    else:
        record["team_execution"]["team_b"]["opponent_argument"] = "different"
    with pytest.raises(ReplayRunRecordValidationError):
        validate_replay_run_record(record, replay_source=replay_source)


def test_invalid_hierarchical_evidence_reference_is_rejected(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
) -> None:
    record = copy.deepcopy(
        cast(dict[str, Any], paired_runs["hierarchical"])["run_record"]
    )
    record["team_execution"]["team_a"]["evidence_analysis"][
        "strongest_supporting_evidence"
    ][0]["finding_index"] = 999
    with pytest.raises(ReplayRunRecordValidationError):
        validate_replay_run_record(record, replay_source=replay_source)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_negative_and_nonfinite_durations_are_rejected(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
    value: float,
) -> None:
    record = copy.deepcopy(cast(dict[str, Any], paired_runs["control"])["run_record"])
    record["timings"]["total_replay_duration_seconds"] = value
    with pytest.raises(ReplayRunRecordValidationError):
        validate_replay_run_record(record, replay_source=replay_source)


def test_incorrect_model_call_counts_are_rejected(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
) -> None:
    record = copy.deepcopy(cast(dict[str, Any], paired_runs["control"])["run_record"])
    record["model_call_counts"]["total_model_calls"] = 4
    with pytest.raises(ReplayRunRecordValidationError):
        validate_replay_run_record(record, replay_source=replay_source)


def test_metrics_use_only_consolidated_claims_and_final_outputs(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
) -> None:
    artifact = cast(dict[str, Any], paired_runs["hierarchical"])
    record = artifact["run_record"]
    source_record = replay_source["source_record"]
    reconstructed = {
        "question": source_record["input"]["question"],
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
        "question_complexity": source_record["decomposition"]["question_complexity"],
        "subquestions": source_record["decomposition"]["subquestions"],
        **record["graph_evidence"],
    }
    expected = json.loads(scorecard_to_json(score_expanded_result(reconstructed)))
    assert record["metrics"] == expected
    serialized_metrics = json.dumps(record["metrics"], sort_keys=True)
    assert HIERARCHICAL_TEAM_A_DRAFT not in serialized_metrics
    assert "The stance remains consistent." not in serialized_metrics
    assert artifact["scorecard_json"] == scorecard_to_json(
        score_expanded_result(reconstructed)
    )


def test_replay_json_is_stable_sorted_and_defensive(
    replay_source: ReplaySource,
    paired_runs: dict[str, object],
) -> None:
    artifact = cast(dict[str, Any], paired_runs["hierarchical"])
    record = artifact["run_record"]
    first = replay_run_record_to_json(record, replay_source=replay_source)
    second = replay_run_record_to_json(record, replay_source=replay_source)
    assert first == second == artifact["run_record_json"]
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert json.loads(first) == record

    changed = copy.deepcopy(record)
    changed["graph_evidence"]["consolidated_evidence"][0]["claim"] = "mutated"
    assert artifact["run_record"]["graph_evidence"]["consolidated_evidence"][0][
        "claim"
    ] != "mutated"


def test_complete_pair_is_offline_and_publishes_no_artifacts(
    replay_source: ReplaySource,
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist.json"
    synthesis = DeterministicSynthesisModel()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(socket, "socket", side_effect=AssertionError("socket")),
        patch.object(
            socket,
            "create_connection",
            side_effect=AssertionError("network"),
        ),
        patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess")),
        patch.object(subprocess, "run", side_effect=AssertionError("subprocess")),
    ):
        run_replay_variant(
            variant="control",
            replay_source=replay_source,
            control_model=DeterministicControlModel(),
            team_model=None,
            synthesis_model=synthesis,
            team_trace_recorder=None,
            clock=IncrementingClock(),
            utc_now=lambda: RUN_TIME,
            implementation_commit=IMPLEMENTATION_COMMIT,
            model_name=MODEL_NAME,
        )
    assert not output.exists()
