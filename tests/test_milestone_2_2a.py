"""Offline tests for deterministic hierarchical team subgraphs."""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
import socket
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from langgraph.errors import GraphRecursionError

from multi_agent_debate.expanded_state import DecompositionResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.hierarchical_graph import (
    HIERARCHICAL_RECURSION_LIMIT,
    build_hierarchical_graph,
    run_hierarchical_debate,
)
from multi_agent_debate.research import ResearchFinding, format_research_finding
from multi_agent_debate.team_graph import build_team_subgraph
from multi_agent_debate.team_model import (
    EVIDENCE_ANALYSIS_KEYS,
    TEAM_CRITIQUE_KEYS,
    TEAM_REVISION_KEYS,
    HierarchicalTeamError,
    TeamGraphValidationError,
    TeamModelError,
    TeamModelInvocationError,
    TeamModelResponseError,
    validate_configured_stance,
    validate_evidence_analysis,
    validate_team_argument,
    validate_team_critique,
    validate_team_name,
    validate_team_revision,
    validate_team_work_product,
)
from multi_agent_debate.team_trace import (
    TEAM_STAGES,
    TeamTraceRecorder,
    canonicalize_successful_team_traces,
    validate_trace_duration,
)

QUESTION = "Should a city adopt a universal transit pass?"
SUBQUESTION_1 = "What access benefits could a universal pass provide?"
SUBQUESTION_2 = "What fiscal limitations could a universal pass create?"
TEAM_A_DRAFT = "Team A draft supports broad and affordable transit access."
TEAM_A_FINAL = "Team A final supports broad and affordable transit access."
TEAM_B_DRAFT = "Team B draft warns about substantial and recurring fiscal costs."
TEAM_B_FINAL = "Team B final warns about substantial and recurring fiscal costs."
SYNTHESIS = "The evidence supports access benefits while identifying fiscal tradeoffs."


def _finding(
    subquestion_index: int,
    subquestion: str,
    finding_index: int = 1,
) -> ResearchFinding:
    return {
        "subquestion_index": subquestion_index,
        "subquestion": subquestion,
        "finding_index": finding_index,
        "claim": f"Claim {subquestion_index}-{finding_index}",
        "source_title": f"Source {subquestion_index}-{finding_index}",
        "source_url": f"https://example.test/{subquestion_index}/{finding_index}",
        "source_type": "test",
        "supporting_excerpt": f"Excerpt {subquestion_index}-{finding_index}",
    }


def _analysis(
    *,
    team: str,
    finding: ResearchFinding,
) -> dict[str, object]:
    return {
        "strongest_supporting_evidence": [
            {
                "subquestion_index": finding["subquestion_index"],
                "finding_index": finding["finding_index"],
                "relevance": "Directly informs the assigned position.",
            }
        ],
        "limitations_or_conflicting_evidence": [],
        "key_claims_to_make": ["Use the strongest documented consideration."],
        "unsupported_claims_to_avoid": ["Avoid claims beyond the supplied evidence."],
        "opponent_points_to_address": (
            [] if team == "team_a" else ["Address Team A's access claim."]
        ),
    }


def _critique() -> dict[str, str]:
    return {
        "stance_consistency": "The draft consistently follows the stance.",
        "evidence_usage": "The cited evidence supports the central claim.",
        "unsupported_factual_claims": "No unsupported facts were introduced.",
        "opponent_response": "The opponent response is proportionate.",
        "unnecessary_repetition": "No unnecessary repetition remains.",
        "clarity_and_concision": "The final wording is clear and concise.",
    }


class FakeDecomposer:
    def __init__(self, result: DecompositionResult) -> None:
        self.result = copy.deepcopy(result)
        self.calls: list[str] = []

    def decompose(self, *, question: str) -> DecompositionResult:
        self.calls.append(question)
        return copy.deepcopy(self.result)


class FakeResearchProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self._lock = threading.Lock()

    def research(
        self,
        *,
        subquestion_index: int,
        subquestion: str,
    ) -> Sequence[ResearchFinding]:
        with self._lock:
            self.calls.append((subquestion_index, subquestion))
        return [_finding(subquestion_index, subquestion)]


class FakeTeamModel:
    def __init__(
        self,
        *,
        fail_stage: str | None = None,
        failure: Exception | None = None,
        malformed: dict[str, object] | None = None,
        mutate_inputs: bool = False,
        unchanged_revision: bool = False,
    ) -> None:
        self.fail_stage = fail_stage
        self.failure = failure or ValueError("private provider detail")
        self.malformed = malformed or {}
        self.mutate_inputs = mutate_inputs
        self.unchanged_revision = unchanged_revision
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.held_inputs: list[object] = []

    def _record(self, stage: str, values: dict[str, object]) -> None:
        self.held_inputs.extend(
            item
            for key, item in values.items()
            if key in {"consolidated_evidence", "research_findings", "evidence_analysis"}
        )
        self.calls.append((stage, copy.deepcopy(values)))
        if self.fail_stage == stage:
            raise self.failure

    def analyze_evidence(self, **values: object) -> object:
        self._record("analyze_evidence", values)
        evidence = cast(list[ResearchFinding], values["consolidated_evidence"])
        if self.mutate_inputs:
            evidence[0]["claim"] = "model attempted mutation"
        if "analyze_evidence" in self.malformed:
            return self.malformed["analyze_evidence"]
        return _analysis(team=cast(str, values["team_name"]), finding=evidence[0])

    def draft_argument(self, **values: object) -> object:
        self._record("draft_argument", values)
        if self.mutate_inputs:
            cast(list[str], values["research_findings"])[0] = "changed"
            cast(dict[str, object], values["evidence_analysis"])[
                "key_claims_to_make"
            ] = ["changed"]
        if "draft_argument" in self.malformed:
            return self.malformed["draft_argument"]
        return (
            TEAM_A_DRAFT
            if values["team_name"] == "team_a"
            else TEAM_B_DRAFT
        )

    def review_and_revise(self, **values: object) -> object:
        self._record("review_and_revise", values)
        if "review_and_revise" in self.malformed:
            return self.malformed["review_and_revise"]
        if self.unchanged_revision:
            revised = cast(str, values["draft_argument"])
        else:
            revised = (
                TEAM_A_FINAL
                if values["team_name"] == "team_a"
                else TEAM_B_FINAL
            )
        return {"critique": _critique(), "revised_argument": revised}


class FakeSynthesisModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def synthesize_neutrally(self, **values: object) -> str:
        self.calls.append(copy.deepcopy(values))
        return SYNTHESIS


class IncrementingClock:
    def __init__(self, step: float = 1.0) -> None:
        self.value = 0.0
        self.step = step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            current = self.value
            self.value += self.step
            return current


class PackageNamedProviderError(RuntimeError):
    """An unrelated error deliberately disguised as project-package code."""


PackageNamedProviderError.__module__ = "multi_agent_debate.fake_provider"


def _team_input(team: str = "team_a") -> dict[str, object]:
    finding = _finding(0, QUESTION)
    common: dict[str, object] = {
        "question": QUESTION,
        "consolidated_evidence": [finding],
        "research_findings": [format_research_finding(finding)],
    }
    if team == "team_a":
        common["team_a_stance"] = TEAM_A_STANCE
    else:
        common["team_b_stance"] = TEAM_B_STANCE
        common["team_a_argument"] = TEAM_A_FINAL
    return common


def _run(
    *,
    complex_question: bool = False,
    model: FakeTeamModel | None = None,
    recorder: TeamTraceRecorder | None = None,
) -> tuple[
    dict[str, object],
    FakeDecomposer,
    FakeResearchProvider,
    FakeTeamModel,
    FakeSynthesisModel,
]:
    decomposition: DecompositionResult = (
        {
            "question_complexity": "complex",
            "subquestions": [SUBQUESTION_1, SUBQUESTION_2],
        }
        if complex_question
        else {"question_complexity": "simple", "subquestions": []}
    )
    decomposer = FakeDecomposer(decomposition)
    provider = FakeResearchProvider()
    team_model = model or FakeTeamModel()
    synthesis = FakeSynthesisModel()
    result = run_hierarchical_debate(
        QUESTION,
        decomposer,
        provider,
        team_model,
        synthesis,
        team_trace_recorder=recorder,
        clock=IncrementingClock(),
    )
    return (
        cast(dict[str, object], result),
        decomposer,
        provider,
        team_model,
        synthesis,
    )


def test_error_hierarchy_separates_model_and_graph_validation() -> None:
    assert issubclass(TeamModelInvocationError, TeamModelError)
    assert issubclass(TeamModelResponseError, TeamModelError)
    assert issubclass(TeamModelError, HierarchicalTeamError)
    assert issubclass(TeamGraphValidationError, HierarchicalTeamError)
    assert not issubclass(TeamGraphValidationError, TeamModelError)


@pytest.mark.parametrize("value", ["team_c", "", None, True])
def test_team_name_validation_rejects_invalid_values(value: object) -> None:
    with pytest.raises(TeamGraphValidationError):
        validate_team_name(value)


@pytest.mark.parametrize("value", ["", "  ", None, 1])
def test_stance_validation_rejects_blank_or_wrong_types(value: object) -> None:
    with pytest.raises(TeamGraphValidationError):
        validate_configured_stance(value)
    with pytest.raises(TeamGraphValidationError):
        build_team_subgraph(
            team_name="team_a", stance=cast(str, value), model=FakeTeamModel()
        )


def test_evidence_analysis_exact_schema_and_defensive_copy() -> None:
    finding = _finding(0, QUESTION)
    original = _analysis(team="team_a", finding=finding)
    validated = validate_evidence_analysis(
        original,
        team_name="team_a",
        consolidated_evidence=[finding],
    )
    assert set(validated) == EVIDENCE_ANALYSIS_KEYS
    cast(list[dict[str, object]], original["strongest_supporting_evidence"])[0][
        "relevance"
    ] = "mutated"
    cast(list[str], original["key_claims_to_make"])[0] = "mutated"
    assert validated["strongest_supporting_evidence"][0]["relevance"] != "mutated"
    assert validated["key_claims_to_make"] != ["mutated"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("key_claims_to_make"),
        lambda value: value.update({"unexpected": []}),
        lambda value: value.update({"key_claims_to_make": ("not", "a list")}),
        lambda value: value.update({"key_claims_to_make": ["same", "same"]}),
    ],
)
def test_evidence_analysis_rejects_schema_list_and_duplicate_errors(
    mutation: Any,
) -> None:
    finding = _finding(0, QUESTION)
    value = _analysis(team="team_a", finding=finding)
    mutation(value)
    with pytest.raises(TeamModelResponseError):
        validate_evidence_analysis(
            value,
            team_name="team_a",
            consolidated_evidence=[finding],
        )


def test_evidence_references_must_resolve_and_cannot_repeat() -> None:
    finding = _finding(0, QUESTION)
    unresolved = _analysis(team="team_a", finding=finding)
    cast(list[dict[str, object]], unresolved["strongest_supporting_evidence"])[0][
        "finding_index"
    ] = 99
    with pytest.raises(TeamModelResponseError, match="does not resolve"):
        validate_evidence_analysis(
            unresolved,
            team_name="team_a",
            consolidated_evidence=[finding],
        )

    repeated = _analysis(team="team_a", finding=finding)
    repeated["limitations_or_conflicting_evidence"] = copy.deepcopy(
        repeated["strongest_supporting_evidence"]
    )
    with pytest.raises(TeamModelResponseError, match="repeat"):
        validate_evidence_analysis(
            repeated,
            team_name="team_a",
            consolidated_evidence=[finding],
        )


@pytest.mark.parametrize(
    ("team", "points"),
    [
        ("team_a", ["Team A cannot receive opponent points."]),
        ("team_b", []),
        ("team_b", ["one", "two", "three", "four"]),
    ],
)
def test_team_specific_opponent_point_counts_are_enforced(
    team: str,
    points: list[str],
) -> None:
    finding = _finding(0, QUESTION)
    value = _analysis(team=team, finding=finding)
    value["opponent_points_to_address"] = points
    with pytest.raises(TeamModelResponseError):
        validate_evidence_analysis(
            value,
            team_name=cast(Any, team),
            consolidated_evidence=[finding],
        )


def test_critique_and_revision_require_exact_nonblank_schema() -> None:
    critique = validate_team_critique(_critique())
    assert set(critique) == TEAM_CRITIQUE_KEYS
    revision = validate_team_revision(
        {"critique": critique, "revised_argument": TEAM_A_FINAL}
    )
    assert set(revision) == TEAM_REVISION_KEYS

    missing = _critique()
    missing.pop("evidence_usage")
    with pytest.raises(TeamModelResponseError):
        validate_team_critique(missing)
    blank = _critique()
    blank["evidence_usage"] = " "
    with pytest.raises(TeamModelResponseError):
        validate_team_critique(blank)
    with pytest.raises(TeamModelResponseError):
        validate_team_revision(
            {
                "critique": _critique(),
                "revised_argument": TEAM_A_FINAL,
                "extra": "no",
            }
        )


@pytest.mark.parametrize("value", ["", "TODO", "tbd", "N/A", "placeholder", "too short"])
def test_argument_validation_rejects_blank_placeholder_and_short_outputs(
    value: str,
) -> None:
    with pytest.raises(TeamModelResponseError):
        validate_team_argument(value)


def test_unchanged_draft_is_a_valid_revision() -> None:
    revision = validate_team_revision(
        {"critique": _critique(), "revised_argument": TEAM_A_DRAFT}
    )
    assert revision["revised_argument"] == TEAM_A_DRAFT

    graph = build_team_subgraph(
        team_name="team_a",
        stance=TEAM_A_STANCE,
        model=FakeTeamModel(unchanged_revision=True),
    )
    result = graph.invoke(_team_input())
    assert result["team_a_argument"] == TEAM_A_DRAFT


def test_work_product_validator_returns_nested_defensive_copies() -> None:
    finding = _finding(0, QUESTION)
    source: dict[str, object] = {
        "team": "team_a",
        "assigned_stance": TEAM_A_STANCE,
        "evidence_analysis": _analysis(team="team_a", finding=finding),
        "draft_argument": TEAM_A_DRAFT,
        "internal_critique": _critique(),
        "final_argument": TEAM_A_FINAL,
    }
    copied = validate_team_work_product(
        source,
        consolidated_evidence=[finding],
        expected_team="team_a",
        expected_stance=TEAM_A_STANCE,
    )
    cast(dict[str, object], source["evidence_analysis"])["key_claims_to_make"] = [
        "mutated"
    ]
    cast(dict[str, str], source["internal_critique"])["evidence_usage"] = "mutated"
    assert copied["evidence_analysis"]["key_claims_to_make"] != ["mutated"]
    assert copied["internal_critique"]["evidence_usage"] != "mutated"


def test_both_roles_use_identical_static_subgraph_topology() -> None:
    model = FakeTeamModel()
    team_a = build_team_subgraph(
        team_name="team_a", stance=TEAM_A_STANCE, model=model
    )
    team_b = build_team_subgraph(
        team_name="team_b", stance=TEAM_B_STANCE, model=model
    )
    a_graph = team_a.get_graph()
    b_graph = team_b.get_graph()
    assert set(a_graph.nodes) == set(b_graph.nodes) == {
        "__start__",
        "evidence_analyst",
        "argument_drafter",
        "internal_critic_reviser",
        "finalize",
        "__end__",
    }
    a_edges = {(edge.source, edge.target, edge.conditional) for edge in a_graph.edges}
    b_edges = {(edge.source, edge.target, edge.conditional) for edge in b_graph.edges}
    assert a_edges == b_edges == {
        ("__start__", "evidence_analyst", False),
        ("evidence_analyst", "argument_drafter", False),
        ("argument_drafter", "internal_critic_reviser", False),
        ("internal_critic_reviser", "finalize", False),
        ("finalize", "__end__", False),
    }


def test_team_subgraph_executes_each_model_stage_once_and_exports_only_result() -> None:
    model = FakeTeamModel()
    graph = build_team_subgraph(
        team_name="team_a", stance=TEAM_A_STANCE, model=model
    )
    result = graph.invoke(_team_input())
    assert [stage for stage, _ in model.calls] == [
        "analyze_evidence",
        "draft_argument",
        "review_and_revise",
    ]
    assert set(result) == {"team_a_argument", "team_a_work_product"}
    assert result["team_a_argument"] == TEAM_A_FINAL
    assert result["team_a_work_product"]["final_argument"] == TEAM_A_FINAL


def test_role_specific_stance_and_opponent_inputs_are_exact() -> None:
    model = FakeTeamModel()
    team_a = build_team_subgraph(
        team_name="team_a", stance=TEAM_A_STANCE, model=model
    )
    team_b = build_team_subgraph(
        team_name="team_b", stance=TEAM_B_STANCE, model=model
    )
    team_a.invoke(_team_input("team_a"))
    team_b.invoke(_team_input("team_b"))
    analyses = [values for stage, values in model.calls if stage == "analyze_evidence"]
    assert analyses[0]["stance"] == TEAM_A_STANCE
    assert analyses[0]["opponent_argument"] is None
    assert analyses[1]["stance"] == TEAM_B_STANCE
    assert analyses[1]["opponent_argument"] == TEAM_A_FINAL

    mismatched = _team_input("team_a")
    mismatched["team_a_stance"] = TEAM_B_STANCE
    with pytest.raises(TeamGraphValidationError, match="does not match"):
        team_a.invoke(mismatched)


def test_parent_directly_mounts_both_compiled_subgraphs() -> None:
    graph = build_hierarchical_graph(
        FakeDecomposer({"question_complexity": "simple", "subquestions": []}),
        FakeResearchProvider(),
        FakeTeamModel(),
        FakeSynthesisModel(),
    )
    subgraphs = dict(graph.get_subgraphs())
    assert set(subgraphs) == {"team_a", "team_b"}
    assert all(type(value).__name__ == "CompiledStateGraph" for value in subgraphs.values())
    source = inspect.getsource(build_hierarchical_graph)
    assert ".invoke(" not in source


@pytest.mark.parametrize("complex_question", [False, True])
def test_hierarchical_graph_runs_simple_and_complex_paths_end_to_end(
    complex_question: bool,
) -> None:
    result, decomposer, provider, model, synthesis = _run(
        complex_question=complex_question
    )
    expected_tasks = (
        [(1, SUBQUESTION_1), (2, SUBQUESTION_2)]
        if complex_question
        else [(0, QUESTION)]
    )
    assert decomposer.calls == [QUESTION]
    assert sorted(provider.calls) == expected_tasks
    assert result["question_complexity"] == (
        "complex" if complex_question else "simple"
    )
    assert result["subquestions"] == (
        [SUBQUESTION_1, SUBQUESTION_2] if complex_question else []
    )
    assert [stage for stage, _ in model.calls] == [
        "analyze_evidence",
        "draft_argument",
        "review_and_revise",
        "analyze_evidence",
        "draft_argument",
        "review_and_revise",
    ]
    assert result["team_a_argument"] == TEAM_A_FINAL
    assert result["team_b_argument"] == TEAM_B_FINAL
    assert result["synthesis"] == SYNTHESIS
    assert len(synthesis.calls) == 1


def test_parent_routes_only_final_argument_and_shared_evidence() -> None:
    result, _, _, model, synthesis = _run(complex_question=True)
    analysis_calls = [values for stage, values in model.calls if stage == "analyze_evidence"]
    assert analysis_calls[0]["opponent_argument"] is None
    assert analysis_calls[1]["opponent_argument"] == TEAM_A_FINAL
    assert "team_a_work_product" not in analysis_calls[1]
    assert analysis_calls[0]["consolidated_evidence"] == analysis_calls[1][
        "consolidated_evidence"
    ]
    assert analysis_calls[0]["consolidated_evidence"] is not analysis_calls[1][
        "consolidated_evidence"
    ]
    assert set(synthesis.calls[0]) == {
        "question",
        "research_findings",
        "team_a_argument",
        "team_b_argument",
    }
    assert synthesis.calls[0]["team_a_argument"] == TEAM_A_FINAL
    assert synthesis.calls[0]["team_b_argument"] == TEAM_B_FINAL
    assert synthesis.calls[0]["research_findings"] == result["research_findings"]
    assert not {
        "assigned_stance",
        "opponent_argument",
        "evidence_analysis",
        "draft_argument",
        "internal_critique",
        "final_argument",
        "team_started_at",
    } & set(result)


def test_model_mutation_cannot_change_parent_evidence_or_other_products() -> None:
    model = FakeTeamModel(mutate_inputs=True)
    result, _, _, _, _ = _run(complex_question=False, model=model)
    original_claim = _finding(0, QUESTION)["claim"]
    assert cast(list[ResearchFinding], result["consolidated_evidence"])[0][
        "claim"
    ] == original_claim
    team_b_before = copy.deepcopy(result["team_b_work_product"])
    model_call_before = copy.deepcopy(model.calls[0])
    cast(dict[str, Any], result["team_a_work_product"])["evidence_analysis"][
        "key_claims_to_make"
    ][0] = "post-run mutation"
    assert result["team_b_work_product"] == team_b_before
    assert model.calls[0] == model_call_before
    assert cast(list[ResearchFinding], result["consolidated_evidence"])[0][
        "claim"
    ] == original_claim


def test_successful_team_traces_are_complete_and_canonical() -> None:
    recorder = TeamTraceRecorder()
    _run(complex_question=True, recorder=recorder)
    canonical = canonicalize_successful_team_traces(recorder)
    assert [
        (trace["team"], trace["stage"]) for trace in canonical["stage_traces"]
    ] == [
        (team, stage)
        for team in ("team_a", "team_b")
        for stage in TEAM_STAGES
    ]
    assert all(trace["duration_seconds"] == 1.0 for trace in canonical["stage_traces"])
    assert [trace["team"] for trace in canonical["total_traces"]] == [
        "team_a",
        "team_b",
    ]
    assert all(trace["duration_seconds"] == 7.0 for trace in canonical["total_traces"])

    snapshot = recorder.stage_snapshot()
    snapshot[0]["error_category"] = "mutated"
    assert recorder.stage_snapshot()[0]["error_category"] is None


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), True, "1"])
def test_trace_duration_validation_rejects_invalid_values(duration: object) -> None:
    with pytest.raises(TeamGraphValidationError):
        validate_trace_duration(duration)


def test_trace_recorder_rejects_unbounded_error_categories() -> None:
    recorder = TeamTraceRecorder()
    with pytest.raises(TeamGraphValidationError, match="unknown"):
        recorder.record_stage(
            team="team_a",
            stage="evidence_analysis",
            duration_seconds=1.0,
            status="error",
            error_category="private exception details",
        )


def test_failed_stage_records_bounded_error_without_total_trace() -> None:
    recorder = TeamTraceRecorder()
    model = FakeTeamModel(fail_stage="draft_argument")
    graph = build_team_subgraph(
        team_name="team_a",
        stance=TEAM_A_STANCE,
        model=model,
        trace_recorder=recorder,
        clock=IncrementingClock(),
    )
    with pytest.raises(TeamModelInvocationError) as caught:
        graph.invoke(_team_input())
    assert caught.value.__cause__ is None
    assert str(caught.value) == "team model invocation failed"
    assert "private provider detail" not in str(caught.value)
    traces = recorder.stage_snapshot()
    assert [(trace["stage"], trace["status"]) for trace in traces] == [
        ("evidence_analysis", "success"),
        ("argument_draft", "error"),
    ]
    assert traces[-1]["error_category"] == "team_model_invocation_error"
    assert recorder.total_snapshot() == []


@pytest.mark.parametrize(
    "expected",
    [
        TeamModelResponseError("bounded response error"),
        TeamModelInvocationError("bounded invocation error"),
        TeamGraphValidationError("bounded graph error"),
    ],
)
def test_hierarchical_team_exceptions_propagate_unchanged(
    expected: HierarchicalTeamError,
) -> None:
    recorder = TeamTraceRecorder()
    graph = build_team_subgraph(
        team_name="team_a",
        stance=TEAM_A_STANCE,
        model=FakeTeamModel(fail_stage="analyze_evidence", failure=expected),
        trace_recorder=recorder,
        clock=IncrementingClock(),
    )
    with pytest.raises(type(expected)) as caught:
        graph.invoke(_team_input())
    assert caught.value is expected


def test_package_named_non_team_exception_is_wrapped_and_sanitized() -> None:
    private_detail = "private package-named provider detail"
    graph = build_team_subgraph(
        team_name="team_a",
        stance=TEAM_A_STANCE,
        model=FakeTeamModel(
            fail_stage="analyze_evidence",
            failure=PackageNamedProviderError(private_detail),
        ),
        clock=IncrementingClock(),
    )
    with pytest.raises(TeamModelInvocationError) as caught:
        graph.invoke(_team_input())
    assert caught.value.__cause__ is None
    assert str(caught.value) == "team model invocation failed"
    assert private_detail not in str(caught.value)


@pytest.mark.parametrize(
    ("stage", "malformed"),
    [
        ("analyze_evidence", {}),
        ("draft_argument", "TODO"),
        (
            "review_and_revise",
            {"critique": _critique(), "revised_argument": "N/A"},
        ),
    ],
)
def test_malformed_model_outputs_are_rejected_without_retry(
    stage: str,
    malformed: object,
) -> None:
    model = FakeTeamModel(malformed={stage: malformed})
    graph = build_team_subgraph(
        team_name="team_a", stance=TEAM_A_STANCE, model=model
    )
    with pytest.raises(TeamModelResponseError):
        graph.invoke(_team_input())
    stage_names = [name for name, _ in model.calls]
    assert stage_names.count(stage) == 1


def test_research_fanout_reducer_and_duplicate_consolidation_remain_correct() -> None:
    result, _, provider, _, _ = _run(complex_question=True)
    raw = cast(list[ResearchFinding], result["raw_research_findings"])
    consolidated = cast(list[ResearchFinding], result["consolidated_evidence"])
    assert sorted(provider.calls) == [(1, SUBQUESTION_1), (2, SUBQUESTION_2)]
    assert [(item["subquestion_index"], item["finding_index"]) for item in raw] == [
        (1, 1),
        (2, 1),
    ]
    assert consolidated == raw
    assert result["research_findings"] == [
        format_research_finding(item) for item in consolidated
    ]


def test_hierarchical_execution_uses_explicit_recursion_limit() -> None:
    assert HIERARCHICAL_RECURSION_LIMIT == 32
    source = inspect.getsource(run_hierarchical_debate)
    assert '"recursion_limit": HIERARCHICAL_RECURSION_LIMIT' in source

    graph = build_hierarchical_graph(
        FakeDecomposer({"question_complexity": "simple", "subquestions": []}),
        FakeResearchProvider(),
        FakeTeamModel(),
        FakeSynthesisModel(),
    )
    with pytest.raises(GraphRecursionError):
        graph.invoke(
            {
                "question": QUESTION,
                "team_a_stance": TEAM_A_STANCE,
                "team_b_stance": TEAM_B_STANCE,
            },
            config={"recursion_limit": 1},
        )

    result, _, _, _, _ = _run(complex_question=False)
    assert result["synthesis"] == SYNTHESIS


def test_frozen_m2_1_graph_files_match_their_pre_milestone_hashes() -> None:
    project_root = Path(__file__).resolve().parents[1]
    expected = {
        "src/multi_agent_debate/expanded_graph.py": (
            "af9fd9a7e1c92b66415c138f933204545d7614721d92201f5673f797787e5fc3"
        ),
        "src/multi_agent_debate/expanded_state.py": (
            "c145d2cf7f322f1f6ddbfdc090db6e6cfd516f9db507d2ff52e6bc43300ca7f8"
        ),
    }
    for relative_path, digest in expected.items():
        assert hashlib.sha256((project_root / relative_path).read_bytes()).hexdigest() == digest


def test_complete_path_uses_no_live_or_external_capabilities() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(socket, "socket", side_effect=AssertionError("socket forbidden")),
        patch.object(
            socket,
            "create_connection",
            side_effect=AssertionError("network forbidden"),
        ),
        patch.object(
            subprocess,
            "Popen",
            side_effect=AssertionError("subprocess forbidden"),
        ),
        patch.object(
            subprocess,
            "run",
            side_effect=AssertionError("subprocess forbidden"),
        ),
    ):
        result, _, _, _, _ = _run(complex_question=True)
    assert result["synthesis"] == SYNTHESIS
