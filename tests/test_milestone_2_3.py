from __future__ import annotations

import copy
import json
import math
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pytest
from langgraph.errors import GraphRecursionError

from multi_agent_debate.app import (
    DEFAULT_MAX_RESULTS,
    ProductLiveDependencies,
    build_live_dependencies,
    execute_product_run,
    main,
)
from multi_agent_debate.expanded_state import DecompositionResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.multi_round_graph import (
    MULTI_ROUND_RECURSION_LIMIT,
    build_multi_round_graph,
    run_multi_round_debate,
    validate_multi_round_result,
)
from multi_agent_debate.multi_round_run_record import (
    ARCHITECTURE_VERSION,
    RUN_RECORD_VERSION,
    MultiRoundRunRecordValidationError,
    multi_round_run_record_to_json,
    score_multi_round_result,
    validate_multi_round_run_record,
)
from multi_agent_debate.multi_round_state import (
    DebateRound,
    MultiRoundDebateInput,
    MultiRoundDebateResult,
)
from multi_agent_debate.multi_round_trace import (
    MultiRoundTraceError,
    MultiRoundTraceRecorder,
    canonicalize_successful_traces,
)
from multi_agent_debate.research import ResearchFinding
from multi_agent_debate.run_record import (
    ArtifactPublicationError,
    publish_serialized_artifacts,
    sanitized_search_settings,
)
from multi_agent_debate.run_trace import RunTraceRecorder
from multi_agent_debate.termination import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NOVELTY_THRESHOLD,
    MultiRoundError,
    MultiRoundModelInvocationError,
    MultiRoundModelResponseError,
    MultiRoundValidationError,
    calculate_same_team_novelty,
    decide_termination,
    validate_max_rounds,
    validate_novelty_threshold,
    validate_termination_decision,
)

QUESTION = "Should a city adopt a universal public transit pass?"
COMMIT = "a" * 40
UTC_NOW = lambda: datetime(2026, 8, 5, tzinfo=timezone.utc)


class StepClock:
    def __init__(self, step: float = 0.125) -> None:
        self._value = 0.0
        self._step = step
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self._value += self._step
            return value


class FakeDecomposer:
    def __init__(self, *, complex_question: bool = False) -> None:
        self.complex_question = complex_question
        self.calls: list[str] = []

    def decompose(self, *, question: str) -> DecompositionResult:
        self.calls.append(question)
        if self.complex_question:
            return {
                "question_complexity": "complex",
                "subquestions": ["What are the benefits?", "What are the costs?"],
            }
        return {"question_complexity": "simple", "subquestions": []}


class FakeResearchProvider:
    def __init__(self, *, recorder: RunTraceRecorder | None = None) -> None:
        self.calls: list[tuple[int, str]] = []
        self.received = []
        self._lock = Lock()
        self._recorder = recorder

    def research(
        self, *, subquestion_index: int, subquestion: str
    ) -> list[ResearchFinding]:
        with self._lock:
            self.calls.append((subquestion_index, subquestion))
        claim = f"Evidence for task {subquestion_index} supports careful policy design."
        if self._recorder is not None:
            self._recorder.record_backend_search(
                {
                    "query": subquestion,
                    "requested_max_results": DEFAULT_MAX_RESULTS,
                    "settings": sanitized_search_settings(),
                    "results": [
                        {
                            "title": f"Source {subquestion_index}",
                            "url": f"https://example.test/{subquestion_index}",
                            "snippet": claim,
                            "provider_score": 0.75,
                        }
                    ],
                    "duration_seconds": 0.25,
                    "status": "success",
                    "error_category": None,
                }
            )
        return [
            {
                "subquestion_index": subquestion_index,
                "subquestion": subquestion,
                "finding_index": 1,
                "claim": claim,
                "source_title": f"Source {subquestion_index}",
                "source_url": f"https://example.test/{subquestion_index}",
                "source_type": "web",
                "supporting_excerpt": claim,
            }
        ]


class FakeDebateModel:
    def __init__(
        self,
        *,
        low_novelty: bool = False,
        fail_team: str | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.low_novelty = low_novelty
        self.fail_team = fail_team
        self.failure = failure or RuntimeError("private transport token")
        self.argument_calls: list[dict[str, object]] = []
        self.synthesis_calls: list[dict[str, object]] = []
        self.research_calls = 0
        self._team_counts = {"team_a": 0, "team_b": 0}
        self._lock = Lock()

    def research(self, *, question: str) -> list[str]:
        self.research_calls += 1
        raise AssertionError("model research must not run")

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        team = "team_a" if stance == TEAM_A_STANCE else "team_b"
        with self._lock:
            self._team_counts[team] += 1
            number = self._team_counts[team]
            self.argument_calls.append(
                {
                    "team": team,
                    "round": number,
                    "question": question,
                    "stance": stance,
                    "research_findings": research_findings.copy(),
                    "research_list_id": id(research_findings),
                    "opponent_argument": opponent_argument,
                }
            )
        if self.fail_team == team:
            raise self.failure
        if self.low_novelty:
            return (
                "Team A consistently supports the policy using shared grounded evidence."
                if team == "team_a"
                else "Team B consistently opposes the policy using shared grounded evidence."
            )
        label = ("alpha fiscal access" if team == "team_a" else "beta budget limits")
        return (
            f"{team} round {number} presents {label} and distinct evidence point "
            f"number {number} for this contested policy."
        )

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: list[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        self.synthesis_calls.append(
            {
                "question": question,
                "research_findings": research_findings.copy(),
                "team_a_argument": team_a_argument,
                "team_b_argument": team_b_argument,
            }
        )
        return "The final positions emphasize access benefits and budget tradeoffs in context."


def run_graph(
    *,
    max_rounds: int = 2,
    low_novelty: bool = False,
    complex_question: bool = False,
    recorder: MultiRoundTraceRecorder | None = None,
    model: FakeDebateModel | None = None,
) -> tuple[MultiRoundDebateResult, FakeDecomposer, FakeResearchProvider, FakeDebateModel, MultiRoundTraceRecorder]:
    decomposer = FakeDecomposer(complex_question=complex_question)
    provider = FakeResearchProvider()
    debate_model = model or FakeDebateModel(low_novelty=low_novelty)
    traces = recorder or MultiRoundTraceRecorder()
    result = run_multi_round_debate(
        QUESTION,
        decomposer,
        provider,
        debate_model,
        max_rounds=max_rounds,
        trace_recorder=traces,
        clock=StepClock(),
    )
    return result, decomposer, provider, debate_model, traces


def make_execution(*, max_rounds: int = 2, low_novelty: bool = False):
    grounding = RunTraceRecorder()
    debate = MultiRoundTraceRecorder()
    provider = FakeResearchProvider(recorder=grounding)
    model = FakeDebateModel(low_novelty=low_novelty)
    execution = execute_product_run(
        question=QUESTION,
        mode="single-round" if max_rounds == 1 else "multi-round",
        model_name="fake-model",
        max_rounds=max_rounds,
        novelty_threshold=DEFAULT_NOVELTY_THRESHOLD,
        max_results=DEFAULT_MAX_RESULTS,
        implementation_commit=COMMIT,
        decomposer=FakeDecomposer(),
        research_provider=provider,
        debate_model=model,
        grounding_recorder=grounding,
        debate_recorder=debate,
        clock=StepClock(),
        utc_timestamp_provider=UTC_NOW,
    )
    return execution, model, provider


@pytest.mark.parametrize("value", [0, 4, -1, True, False, 1.0, "2"])
def test_maximum_round_validation_rejects_invalid_values(value: object) -> None:
    with pytest.raises(MultiRoundValidationError):
        validate_max_rounds(value)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, True, False, "0.1"])
def test_novelty_threshold_rejects_invalid_values(value: object) -> None:
    with pytest.raises(MultiRoundValidationError):
        validate_novelty_threshold(value)


def test_policy_defaults_and_valid_boundaries() -> None:
    assert DEFAULT_MAX_ROUNDS == 2
    assert DEFAULT_NOVELTY_THRESHOLD == 0.15
    assert [validate_max_rounds(value) for value in (1, 2, 3)] == [1, 2, 3]
    assert validate_novelty_threshold(0) == 0.0
    assert validate_novelty_threshold(1.0) == 1.0


def test_unicode_nfkc_casefold_and_short_text_fallback() -> None:
    assert calculate_same_team_novelty("ＡＬＰＨＡ beta", "alpha BETA") == 0.0
    assert calculate_same_team_novelty("alpha beta", "gamma delta") == 1.0


def test_three_token_shingle_novelty() -> None:
    novelty = calculate_same_team_novelty(
        "alpha beta gamma delta", "alpha beta gamma echo"
    )
    assert novelty == pytest.approx(2 / 3)


def test_identical_and_disjoint_arguments_have_extreme_novelty() -> None:
    assert calculate_same_team_novelty("alpha beta gamma", "alpha beta gamma") == 0
    assert calculate_same_team_novelty("alpha beta gamma", "delta echo foxtrot") == 1


def test_round_one_continues_or_stops_at_single_round_limit() -> None:
    histories = {"team_a_history": ["alpha evidence claim"], "team_b_history": ["beta evidence claim"]}
    assert decide_termination(**histories, max_rounds=2)["reason"] == "continue"
    decision = decide_termination(**histories, max_rounds=1)
    assert decision == {
        "stop": True,
        "reason": "max_rounds",
        "team_a_novelty": None,
        "team_b_novelty": None,
    }


def test_low_novelty_precedes_max_rounds_and_high_novelty_continues() -> None:
    same = decide_termination(
        team_a_history=["alpha beta gamma", "alpha beta gamma"],
        team_b_history=["delta echo foxtrot", "delta echo foxtrot"],
        max_rounds=2,
    )
    assert same["reason"] == "low_novelty"
    assert same["stop"] is True
    different = decide_termination(
        team_a_history=["alpha beta gamma", "one two three"],
        team_b_history=["delta echo foxtrot", "four five six"],
        max_rounds=3,
    )
    assert different["reason"] == "continue"


def test_histories_must_match_and_contain_completed_round() -> None:
    with pytest.raises(MultiRoundValidationError):
        decide_termination(team_a_history=[], team_b_history=[])
    with pytest.raises(MultiRoundValidationError):
        decide_termination(
            team_a_history=["alpha beta"], team_b_history=["beta gamma", "extra text"]
        )


def test_termination_decision_validation_is_strict_and_defensive() -> None:
    original = decide_termination(
        team_a_history=["alpha beta gamma"], team_b_history=["delta echo foxtrot"], max_rounds=1
    )
    copied = validate_termination_decision(original)
    assert copied == original and copied is not original
    broken = dict(original, extra=True)
    with pytest.raises(MultiRoundValidationError):
        validate_termination_decision(broken)


def test_state_schemas_include_required_policy_and_round_fields() -> None:
    assert set(MultiRoundDebateInput.__required_keys__) == {
        "question", "team_a_stance", "team_b_stance", "max_rounds", "novelty_threshold"
    }
    assert {
        "round_number", "team_a_argument", "team_b_argument",
        "team_a_opponent_argument", "team_b_opponent_argument",
        "team_a_novelty", "team_b_novelty", "termination_decision",
    } <= set(DebateRound.__required_keys__)


def test_two_round_graph_routes_opponents_and_reuses_research_once() -> None:
    result, decomposer, provider, model, _ = run_graph(max_rounds=2)
    assert decomposer.calls == [QUESTION]
    assert provider.calls == [(0, QUESTION)]
    assert model.research_calls == 0
    assert len(result["debate_rounds"]) == 2
    calls = model.argument_calls
    assert calls[0]["opponent_argument"] is None
    assert calls[1]["opponent_argument"] == result["debate_rounds"][0]["team_a_argument"]
    assert calls[2]["opponent_argument"] == result["debate_rounds"][0]["team_b_argument"]
    assert calls[3]["opponent_argument"] == result["debate_rounds"][1]["team_a_argument"]
    assert result["termination_decision"]["reason"] == "max_rounds"


def test_three_round_graph_is_bounded_and_synthesizes_once() -> None:
    result, _, provider, model, _ = run_graph(max_rounds=3)
    assert len(result["debate_rounds"]) == 3
    assert len(model.argument_calls) == 6
    assert len(model.synthesis_calls) == 1
    assert provider.calls == [(0, QUESTION)]
    assert result["current_round_number"] == 3


def test_low_novelty_stops_early_at_round_two() -> None:
    result, _, _, model, _ = run_graph(max_rounds=3, low_novelty=True)
    assert len(result["debate_rounds"]) == 2
    assert result["termination_decision"]["reason"] == "low_novelty"
    assert len(model.argument_calls) == 4


def test_single_round_uses_same_graph_and_synthesizes_final_arguments() -> None:
    result, _, _, model, _ = run_graph(max_rounds=1)
    assert len(result["debate_rounds"]) == 1
    assert len(model.argument_calls) == 2
    assert model.synthesis_calls[0]["team_a_argument"] == result["team_a_argument"]
    assert model.synthesis_calls[0]["team_b_argument"] == result["team_b_argument"]
    assert "debate_rounds" not in model.synthesis_calls[0]


def test_complex_research_fanout_remains_one_pass() -> None:
    result, _, provider, model, _ = run_graph(max_rounds=2, complex_question=True)
    assert sorted(provider.calls) == [
        (1, "What are the benefits?"), (2, "What are the costs?")
    ]
    assert len(model.argument_calls) == 4
    assert all(len(call["research_findings"]) == 2 for call in model.argument_calls)
    assert len(result["raw_research_findings"]) == 2


def test_teams_receive_equal_evidence_through_distinct_lists() -> None:
    _, _, _, model, _ = run_graph(max_rounds=2)
    evidence_values = [call["research_findings"] for call in model.argument_calls]
    evidence_ids = [call["research_list_id"] for call in model.argument_calls]
    assert all(value == evidence_values[0] for value in evidence_values)
    assert len(set(evidence_ids)) == len(evidence_ids)


def test_trace_canonicalization_has_round_stage_order() -> None:
    result, _, _, _, recorder = run_graph(max_rounds=2)
    traces = canonicalize_successful_traces(
        recorder=recorder, completed_rounds=len(result["debate_rounds"])
    )
    assert [(item["round_number"], item["stage"]) for item in traces] == [
        (1, "team_a_argument"), (1, "team_b_argument"), (1, "termination"),
        (2, "team_a_argument"), (2, "team_b_argument"), (2, "termination"),
        (2, "neutral_synthesis"),
    ]


def test_trace_snapshots_are_defensive_and_missing_trace_is_rejected() -> None:
    result, _, _, _, recorder = run_graph(max_rounds=1)
    snapshot = recorder.snapshot()
    snapshot[0]["duration_seconds"] = 999
    assert recorder.snapshot()[0]["duration_seconds"] != 999
    incomplete = MultiRoundTraceRecorder()
    for trace in recorder.snapshot()[:-1]:
        incomplete.record(trace)
    with pytest.raises(MultiRoundTraceError):
        canonicalize_successful_traces(recorder=incomplete, completed_rounds=1)


def test_trace_recorder_rejects_unbounded_error_categories() -> None:
    recorder = MultiRoundTraceRecorder()
    with pytest.raises(MultiRoundTraceError):
        recorder.record(
            {
                "round_number": 1,
                "team": "team_a",
                "stage": "team_a_argument",
                "duration_seconds": 0.1,
                "status": "error",
                "error_category": "private failure detail",
            }
        )


def test_failed_model_turn_records_bounded_trace_and_unknown_error_is_unchained() -> None:
    recorder = MultiRoundTraceRecorder()
    model = FakeDebateModel(fail_team="team_a")
    with pytest.raises(MultiRoundModelInvocationError) as caught:
        run_graph(max_rounds=2, recorder=recorder, model=model)
    assert caught.value.__cause__ is None
    assert "private transport token" not in str(caught.value)
    assert recorder.snapshot() == [
        {
            "round_number": 1,
            "team": "team_a",
            "stage": "team_a_argument",
            "duration_seconds": 0.125,
            "status": "error",
            "error_category": "model_invocation_error",
        }
    ]


def test_explicit_multi_round_model_error_propagates_unchanged() -> None:
    error = MultiRoundModelResponseError("bounded")
    model = FakeDebateModel(fail_team="team_a", failure=error)
    with pytest.raises(MultiRoundModelResponseError) as caught:
        run_graph(model=model)
    assert caught.value is error


def test_invalid_model_product_records_a_bounded_response_error() -> None:
    recorder = MultiRoundTraceRecorder()
    model = FakeDebateModel()
    model.write_argument = lambda **kwargs: "   "
    with pytest.raises(MultiRoundModelResponseError):
        run_graph(model=model, recorder=recorder)
    assert recorder.snapshot()[0]["error_category"] == "model_response_error"


def test_langgraph_recursion_limit_reaches_engine_and_normal_limit_succeeds() -> None:
    graph = build_multi_round_graph(
        FakeDecomposer(), FakeResearchProvider(), FakeDebateModel()
    )
    valid_input: MultiRoundDebateInput = {
        "question": QUESTION,
        "team_a_stance": TEAM_A_STANCE,
        "team_b_stance": TEAM_B_STANCE,
        "max_rounds": 2,
        "novelty_threshold": 0.15,
    }
    with pytest.raises(GraphRecursionError):
        graph.invoke(valid_input, config={"recursion_limit": 1})
    assert MULTI_ROUND_RECURSION_LIMIT == 64
    result = run_multi_round_debate(
        QUESTION, FakeDecomposer(), FakeResearchProvider(), FakeDebateModel()
    )
    assert len(result["debate_rounds"]) == 2


def test_result_validator_returns_nested_defensive_copy() -> None:
    result, *_ = run_graph(max_rounds=2)
    validated = validate_multi_round_result(result)
    validated["debate_rounds"][0]["termination_decision"]["reason"] = "max_rounds"
    validated["consolidated_evidence"][0]["claim"] = "changed"
    assert result["debate_rounds"][0]["termination_decision"]["reason"] == "continue"
    assert result["consolidated_evidence"][0]["claim"] != "changed"


def test_complete_v2_record_and_call_counts() -> None:
    execution, model, provider = make_execution(max_rounds=2)
    record = execution.run_record
    assert record["run_record_version"] == RUN_RECORD_VERSION
    assert record["provenance"]["architecture_version"] == ARCHITECTURE_VERSION
    assert record["model_call_counts"] == {
        "decomposition": 1,
        "team_a_argument": 2,
        "team_b_argument": 2,
        "synthesis": 1,
        "research_provider_calls": 1,
        "total_model_calls": 6,
    }
    assert len(model.argument_calls) == 4
    assert len(provider.calls) == 1
    assert validate_multi_round_run_record(record) == record


@pytest.mark.parametrize(
    "field",
    [
        "provenance", "input", "decomposition", "search_evidence",
        "graph_evidence", "debate_rounds", "termination", "timings",
        "model_call_counts", "metrics",
    ],
)
def test_every_required_top_level_record_field_is_enforced(field: str) -> None:
    execution, _, _ = make_execution()
    broken = copy.deepcopy(execution.run_record)
    del broken[field]
    with pytest.raises(MultiRoundRunRecordValidationError):
        validate_multi_round_run_record(broken)


@pytest.mark.parametrize("tamper", ["round", "opponent", "novelty", "termination", "extra_round", "counts", "metrics"])
def test_run_record_cross_field_tampering_is_rejected(tamper: str) -> None:
    execution, _, _ = make_execution()
    broken = copy.deepcopy(execution.run_record)
    if tamper == "round":
        broken["debate_rounds"][0]["round_number"] = 2
    elif tamper == "opponent":
        broken["debate_rounds"][1]["team_a_opponent_argument"] = "wrong opponent text"
    elif tamper == "novelty":
        broken["debate_rounds"][1]["team_a_novelty"] = 0.0
    elif tamper == "termination":
        broken["termination"]["reason"] = "low_novelty"
    elif tamper == "extra_round":
        broken["debate_rounds"].append(copy.deepcopy(broken["debate_rounds"][-1]))
    elif tamper == "counts":
        broken["model_call_counts"]["total_model_calls"] = 99
    else:
        broken["metrics"]["team_novelty"]["score"] = 0.123
    with pytest.raises(MultiRoundRunRecordValidationError):
        validate_multi_round_run_record(broken)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1])
def test_nonfinite_or_negative_timings_are_rejected(bad: float) -> None:
    execution, _, _ = make_execution()
    broken = copy.deepcopy(execution.run_record)
    broken["timings"]["total_duration_seconds"] = bad
    with pytest.raises(MultiRoundRunRecordValidationError):
        validate_multi_round_run_record(broken)


def test_metrics_use_only_final_arguments_and_frozen_serialization() -> None:
    execution, _, _ = make_execution()
    record = execution.run_record
    assert record["metrics"] == json.loads(execution.scorecard_json)
    assert record["metrics"] == json.loads(
        json.dumps(json.loads(execution.scorecard_json))
    )
    final_round = record["debate_rounds"][-1]
    assert record["graph_evidence"]["final_team_a_argument"] == final_round["team_a_argument"]
    assert record["graph_evidence"]["final_team_b_argument"] == final_round["team_b_argument"]
    prior = record["debate_rounds"][0]
    assert prior["team_a_argument"] != final_round["team_a_argument"]


def test_v2_serialization_is_stable_sorted_and_has_one_newline() -> None:
    execution, _, _ = make_execution()
    first = multi_round_run_record_to_json(execution.run_record)
    second = multi_round_run_record_to_json(copy.deepcopy(execution.run_record))
    assert first == second
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert json.loads(first)["run_record_version"] == "2.0"


def test_uppercase_implementation_commit_is_normalized() -> None:
    grounding = RunTraceRecorder()
    execution = execute_product_run(
        question=QUESTION,
        mode="single-round",
        model_name="fake-model",
        max_rounds=1,
        novelty_threshold=0.15,
        max_results=3,
        implementation_commit="A" * 40,
        decomposer=FakeDecomposer(),
        research_provider=FakeResearchProvider(recorder=grounding),
        debate_model=FakeDebateModel(),
        grounding_recorder=grounding,
        debate_recorder=MultiRoundTraceRecorder(),
        clock=StepClock(),
        utc_timestamp_provider=UTC_NOW,
    )
    assert execution.run_record["provenance"]["implementation_commit"] == "a" * 40


def test_missing_tavily_key_fails_before_live_construction(monkeypatch) -> None:
    constructed = False

    def forbidden(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError

    monkeypatch.setattr("multi_agent_debate.app.TavilySearchBackend", forbidden)
    with pytest.raises(Exception) as caught:
        build_live_dependencies(model_name="qwen3:8b", environ={})
    assert "Tavily" in str(caught.value)
    assert constructed is False


def _cli_args(tmp_path: Path, *, mode: str = "multi-round", rounds: int | None = None) -> list[str]:
    args = [
        "--question", QUESTION,
        "--mode", mode,
        "--implementation-commit", COMMIT,
        "--output", str(tmp_path / "run.json"),
        "--score-output", str(tmp_path / "score.json"),
        "--summary",
    ]
    if rounds is not None:
        args.extend(["--max-rounds", str(rounds)])
    return args


def test_cli_single_round_forces_one_and_multi_round_defaults_two(tmp_path: Path) -> None:
    observed = []
    execution, _, _ = make_execution(max_rounds=1)

    def dependencies(**kwargs):
        return ProductLiveDependencies(
            FakeDecomposer(), FakeResearchProvider(), FakeDebateModel(),
            RunTraceRecorder(), MultiRoundTraceRecorder()
        ), "fake-key"

    def executor(**kwargs):
        observed.append(kwargs["max_rounds"])
        return execution

    assert main(_cli_args(tmp_path, mode="single-round"), dependency_builder=dependencies, executor=executor, publisher=lambda **kwargs: None) == 0
    second = tmp_path / "second"
    assert main(_cli_args(second), dependency_builder=dependencies, executor=executor, publisher=lambda **kwargs: None) == 0
    assert observed == [1, 2]


def test_cli_rejects_incompatible_single_round_maximum(tmp_path: Path, capsys) -> None:
    called = False

    def dependencies(**kwargs):
        nonlocal called
        called = True
        raise AssertionError

    assert main(_cli_args(tmp_path, mode="single-round", rounds=2), dependency_builder=dependencies) == 2
    assert called is False
    assert capsys.readouterr().err == "error: product configuration is invalid\n"


def test_output_preflight_precedes_dependency_construction(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    output.write_text("existing", encoding="utf-8")
    called = False

    def dependencies(**kwargs):
        nonlocal called
        called = True
        raise AssertionError

    assert main(_cli_args(tmp_path), dependency_builder=dependencies) == 2
    assert called is False


def test_complete_offline_cli_run_writes_two_artifacts_without_live_services(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("socket")))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("subprocess")))

    def dependencies(**kwargs):
        grounding = RunTraceRecorder()
        return ProductLiveDependencies(
            FakeDecomposer(),
            FakeResearchProvider(recorder=grounding),
            FakeDebateModel(),
            grounding,
            MultiRoundTraceRecorder(),
        ), "obvious-fake-key"

    status = main(
        _cli_args(tmp_path),
        environ={},
        dependency_builder=dependencies,
        clock=StepClock(),
        utc_timestamp_provider=UTC_NOW,
    )
    assert status == 0
    run_text = (tmp_path / "run.json").read_text(encoding="utf-8")
    score_text = (tmp_path / "score.json").read_text(encoding="utf-8")
    assert json.loads(run_text)["run_record_version"] == "2.0"
    assert json.loads(run_text)["metrics"] == json.loads(score_text)
    output = capsys.readouterr()
    assert output.err == ""
    assert "https://" not in output.out
    assert "Evidence for task" not in output.out
    assert "obvious-fake-key" not in output.out + run_text + score_text


def test_no_artifacts_are_written_after_model_failure(tmp_path: Path, capsys) -> None:
    def dependencies(**kwargs):
        grounding = RunTraceRecorder()
        return ProductLiveDependencies(
            FakeDecomposer(), FakeResearchProvider(recorder=grounding),
            FakeDebateModel(fail_team="team_a"), grounding,
            MultiRoundTraceRecorder(),
        ), "fake-key"

    assert main(_cli_args(tmp_path), dependency_builder=dependencies, clock=StepClock(), utc_timestamp_provider=UTC_NOW) == 1
    assert not (tmp_path / "run.json").exists()
    assert not (tmp_path / "score.json").exists()
    assert "private transport token" not in capsys.readouterr().err


def test_hardened_two_file_publication_rolls_back_new_score_on_run_failure(
    tmp_path: Path, monkeypatch
) -> None:
    run_output = tmp_path / "run.json"
    score_output = tmp_path / "score.json"
    real_replace = os.replace

    def replace(source, destination):
        if Path(destination) == run_output:
            raise OSError("bounded failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(ArtifactPublicationError):
        publish_serialized_artifacts(
            run_output=run_output,
            run_json='{"run": true}\n',
            score_output=score_output,
            scorecard_json='{"score": true}\n',
            overwrite=False,
        )
    assert not run_output.exists() and not score_output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_overwrite_failure_restores_exact_original_score_and_run_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    run_output = tmp_path / "run.json"
    score_output = tmp_path / "score.json"
    original_run = b"original run bytes\n"
    original_score = b"original score bytes\n"
    run_output.write_bytes(original_run)
    score_output.write_bytes(original_score)
    real_replace = os.replace

    def replace(source, destination):
        if Path(destination) == run_output:
            raise OSError("bounded failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(ArtifactPublicationError):
        publish_serialized_artifacts(
            run_output=run_output,
            run_json='{"run": true}\n',
            score_output=score_output,
            scorecard_json='{"score": true}\n',
            overwrite=True,
        )
    assert run_output.read_bytes() == original_run
    assert score_output.read_bytes() == original_score
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_exposes_no_api_key_argument(tmp_path: Path, capsys) -> None:
    args = [*_cli_args(tmp_path), "--api-key", "must-not-be-accepted"]
    assert main(args) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "error: product configuration is invalid\n"


def test_summary_contains_only_bounded_operational_fields(tmp_path: Path, capsys) -> None:
    execution, _, _ = make_execution()

    def dependencies(**kwargs):
        return ProductLiveDependencies(
            FakeDecomposer(), FakeResearchProvider(), FakeDebateModel(),
            RunTraceRecorder(), MultiRoundTraceRecorder()
        ), "summary-secret"

    assert main(_cli_args(tmp_path), dependency_builder=dependencies, executor=lambda **kwargs: execution, publisher=lambda **kwargs: None) == 0
    text = capsys.readouterr().out
    for forbidden in ("https://", "Evidence for task", "team_a round", "summary-secret"):
        assert forbidden not in text
    for expected in ("Mode:", "Complexity:", "Completed rounds:", "Termination reason:", "Model calls:"):
        assert expected in text
