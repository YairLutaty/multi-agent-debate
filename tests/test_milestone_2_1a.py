"""Offline tests for deterministic decomposition and research fan-out."""

from collections.abc import Sequence
from copy import deepcopy
from itertools import permutations
from threading import Lock
from typing import TypeAlias, cast
from unittest.mock import patch

import pytest
from langgraph.types import Send

from multi_agent_debate.expanded_graph import (
    build_expanded_graph,
    route_research,
    run_expanded_debate,
    validate_decomposition_result,
)
from multi_agent_debate.expanded_state import DecompositionResult
from multi_agent_debate.graph import TEAM_A_STANCE, TEAM_B_STANCE
from multi_agent_debate.research import (
    ResearchFinding,
    consolidate_research_findings,
    format_research_finding,
    merge_raw_findings,
)

QUESTION = "Should a city change its transport policy?"
SUBQUESTION_1 = "What benefits could the policy provide?"
SUBQUESTION_2 = "What costs or limitations could it create?"
TEAM_A_ARGUMENT = "The policy would produce meaningful public benefits."
TEAM_B_ARGUMENT = "The policy's costs could exceed those benefits."
SYNTHESIS = "The exchange identifies both potential benefits and costs."

ResearchCall: TypeAlias = tuple[int, str]
ModelCall: TypeAlias = tuple[object, ...]


def _finding(
    subquestion_index: int,
    subquestion: str,
    finding_index: int,
    *,
    claim: str | None = None,
    source_url: str | None = None,
) -> ResearchFinding:
    return {
        "subquestion_index": subquestion_index,
        "subquestion": subquestion,
        "finding_index": finding_index,
        "claim": claim or f"Claim {subquestion_index}-{finding_index}.",
        "source_title": f"Source {subquestion_index}-{finding_index}",
        "source_url": source_url
        or f"https://example.test/{subquestion_index}/{finding_index}",
        "source_type": "test source",
        "supporting_excerpt": f"Excerpt {subquestion_index}-{finding_index}.",
    }


class FakeDecomposer:
    def __init__(self, result: DecompositionResult) -> None:
        self.result = result
        self.calls: list[str] = []

    def decompose(self, *, question: str) -> DecompositionResult:
        self.calls.append(question)
        return {
            "question_complexity": self.result["question_complexity"],
            "subquestions": self.result["subquestions"].copy(),
        }


class RecordingResearchProvider:
    """Deterministic provider with a concurrency-safe mutable call log."""

    def __init__(
        self,
        responses: dict[ResearchCall, Sequence[object]],
    ) -> None:
        self.responses = responses
        self.calls: list[ResearchCall] = []
        self._call_lock = Lock()

    def research(
        self,
        *,
        subquestion_index: int,
        subquestion: str,
    ) -> Sequence[ResearchFinding]:
        key = (subquestion_index, subquestion)
        with self._call_lock:
            self.calls.append(key)
        values = self.responses.get(key, ())
        copies = [
            value.copy() if isinstance(value, dict) else value
            for value in values
        ]
        return cast(Sequence[ResearchFinding], copies)


class FakeDebateModel:
    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self.research_called = False

    def research(self, *, question: str) -> list[str]:
        self.research_called = True
        raise AssertionError("expanded graph must not call DebateModel.research")

    def write_argument(
        self,
        *,
        question: str,
        stance: str,
        research_findings: list[str],
        opponent_argument: str | None,
    ) -> str:
        self.calls.append(
            (
                "write_argument",
                question,
                stance,
                tuple(research_findings),
                opponent_argument,
            )
        )
        return TEAM_A_ARGUMENT if opponent_argument is None else TEAM_B_ARGUMENT

    def synthesize_neutrally(
        self,
        *,
        question: str,
        research_findings: list[str],
        team_a_argument: str,
        team_b_argument: str,
    ) -> str:
        self.calls.append(
            (
                "synthesize_neutrally",
                question,
                tuple(research_findings),
                team_a_argument,
                team_b_argument,
            )
        )
        return SYNTHESIS


def _simple_decomposition() -> DecompositionResult:
    return {"question_complexity": "simple", "subquestions": []}


def _complex_decomposition() -> DecompositionResult:
    return {
        "question_complexity": "complex",
        "subquestions": [SUBQUESTION_1, SUBQUESTION_2],
    }


def _assert_model_routing(model: FakeDebateModel, research: list[str]) -> None:
    assert model.research_called is False
    assert model.calls == [
        (
            "write_argument",
            QUESTION,
            TEAM_A_STANCE,
            tuple(research),
            None,
        ),
        (
            "write_argument",
            QUESTION,
            TEAM_B_STANCE,
            tuple(research),
            TEAM_A_ARGUMENT,
        ),
        (
            "synthesize_neutrally",
            QUESTION,
            tuple(research),
            TEAM_A_ARGUMENT,
            TEAM_B_ARGUMENT,
        ),
    ]


def test_route_research_returns_exact_simple_and_complex_sends() -> None:
    simple_sends = route_research(
        {
            "question": QUESTION,
            "question_complexity": "simple",
            "subquestions": [],
        }
    )
    complex_sends = route_research(
        {
            "question": QUESTION,
            "question_complexity": "complex",
            "subquestions": [SUBQUESTION_1, SUBQUESTION_2],
        }
    )

    assert simple_sends == [
        Send(
            "research_worker",
            {"subquestion_index": 0, "subquestion": QUESTION},
        )
    ]
    assert complex_sends == [
        Send(
            "research_worker",
            {"subquestion_index": 1, "subquestion": SUBQUESTION_1},
        ),
        Send(
            "research_worker",
            {"subquestion_index": 2, "subquestion": SUBQUESTION_2},
        ),
    ]


@pytest.mark.parametrize(
    "value",
    [
        {"question_complexity": "simple", "subquestions": ["unwanted"]},
        {"question_complexity": "complex", "subquestions": [SUBQUESTION_1]},
        {
            "question_complexity": "complex",
            "subquestions": [SUBQUESTION_1, SUBQUESTION_1],
        },
        {
            "question_complexity": "complex",
            "subquestions": [SUBQUESTION_1, "  "],
        },
    ],
)
def test_decomposition_validation_enforces_routing_contract(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        validate_decomposition_result(value)


def test_simple_graph_researches_original_question_once() -> None:
    finding = _finding(0, QUESTION, 1)
    duplicate = finding.copy()
    decomposer = FakeDecomposer(_simple_decomposition())
    provider = RecordingResearchProvider({(0, QUESTION): [finding, duplicate]})
    model = FakeDebateModel()

    result = run_expanded_debate(QUESTION, decomposer, provider, model)

    assert decomposer.calls == [QUESTION]
    assert provider.calls == [(0, QUESTION)]
    assert result["question"] == QUESTION
    assert result["team_a_stance"] == TEAM_A_STANCE
    assert result["team_b_stance"] == TEAM_B_STANCE
    assert result["question_complexity"] == "simple"
    assert result["subquestions"] == []
    assert result["raw_research_findings"] == [finding, duplicate]
    assert result["consolidated_evidence"] == [finding]
    assert result["research_findings"] == [format_research_finding(finding)]
    assert result["team_a_argument"] == TEAM_A_ARGUMENT
    assert result["team_b_argument"] == TEAM_B_ARGUMENT
    assert result["synthesis"] == SYNTHESIS
    _assert_model_routing(model, result["research_findings"])


def test_complex_graph_fans_out_and_preserves_all_findings() -> None:
    first = _finding(1, SUBQUESTION_1, 1)
    second = _finding(1, SUBQUESTION_1, 2)
    third = _finding(2, SUBQUESTION_2, 1)
    decomposer = FakeDecomposer(_complex_decomposition())
    provider = RecordingResearchProvider(
        {
            (1, SUBQUESTION_1): [second, first],
            (2, SUBQUESTION_2): [third],
        }
    )
    model = FakeDebateModel()

    result = run_expanded_debate(QUESTION, decomposer, provider, model)

    assert sorted(provider.calls) == sorted(
        [(1, SUBQUESTION_1), (2, SUBQUESTION_2)]
    )
    assert result["question_complexity"] == "complex"
    assert result["subquestions"] == [SUBQUESTION_1, SUBQUESTION_2]
    assert result["raw_research_findings"] == [first, second, third]
    assert result["consolidated_evidence"] == [first, second, third]
    assert result["research_findings"] == [
        format_research_finding(first),
        format_research_finding(second),
        format_research_finding(third),
    ]
    _assert_model_routing(model, result["research_findings"])


def test_reducer_is_associative_immutable_and_permutation_independent() -> None:
    first = _finding(1, SUBQUESTION_1, 1)
    second = _finding(1, SUBQUESTION_1, 2)
    third = _finding(2, SUBQUESTION_2, 1)
    duplicate = first.copy()
    left = [third, first]
    right = [second, duplicate]
    left_before = deepcopy(left)
    right_before = deepcopy(right)

    result = merge_raw_findings(left, right)
    associated_left = merge_raw_findings(
        merge_raw_findings([first], [second]),
        [third],
    )
    associated_right = merge_raw_findings(
        [first],
        merge_raw_findings([second], [third]),
    )
    permutation_results = {
        tuple(
            tuple(sorted(finding.items()))
            for finding in merge_raw_findings(list(order[:2]), list(order[2:]))
        )
        for order in permutations([first, second, third])
    }

    assert left == left_before
    assert right == right_before
    assert all(
        merged is not original
        for merged in result
        for original in (*left, *right)
    )
    assert result.count(first) == 2
    assert associated_left == associated_right
    assert len(permutation_results) == 1


def test_consolidation_is_deterministic_and_only_removes_exact_duplicates() -> None:
    first = _finding(
        1,
        SUBQUESTION_1,
        1,
        claim="Shared URL, first claim.",
        source_url="https://example.test/shared",
    )
    exact_duplicate = first.copy()
    same_url_other_claim = _finding(
        1,
        SUBQUESTION_1,
        1,
        claim="Shared URL, different claim.",
        source_url="https://example.test/shared",
    )
    other_question = _finding(2, SUBQUESTION_2, 1)
    raw = [other_question, exact_duplicate, same_url_other_claim, first]

    outcomes = [
        consolidate_research_findings(
            question=QUESTION,
            question_complexity="complex",
            subquestions=[SUBQUESTION_1, SUBQUESTION_2],
            raw_findings=list(order),
        )
        for order in permutations(raw)
    ]

    assert all(outcome == outcomes[0] for outcome in outcomes)
    assert len(outcomes[0]) == 3
    assert outcomes[0].count(first) == 1
    assert same_url_other_claim in outcomes[0]
    assert other_question in outcomes[0]
    for finding in outcomes[0]:
        assert set(finding) == {
            "subquestion_index",
            "subquestion",
            "finding_index",
            "claim",
            "source_title",
            "source_url",
            "source_type",
            "supporting_excerpt",
        }


def test_formatter_uses_fixed_readable_field_order() -> None:
    finding = _finding(1, SUBQUESTION_1, 2)

    assert format_research_finding(finding) == "\n".join(
        (
            f"Claim: {finding['claim']}",
            f"Source: {finding['source_title']}",
            f"Source type: {finding['source_type']}",
            f"URL: {finding['source_url']}",
            f"Supporting excerpt: {finding['supporting_excerpt']}",
            f"Research question: {finding['subquestion']}",
            "Subquestion index: 1",
            "Finding index: 2",
        )
    )


def test_one_empty_worker_does_not_erase_successful_worker() -> None:
    finding = _finding(2, SUBQUESTION_2, 1)
    provider = RecordingResearchProvider(
        {
            (1, SUBQUESTION_1): [],
            (2, SUBQUESTION_2): [finding],
        }
    )
    model = FakeDebateModel()

    result = run_expanded_debate(
        QUESTION,
        FakeDecomposer(_complex_decomposition()),
        provider,
        model,
    )

    assert sorted(provider.calls) == sorted(
        [(1, SUBQUESTION_1), (2, SUBQUESTION_2)]
    )
    assert result["raw_research_findings"] == [finding]
    assert result["consolidated_evidence"] == [finding]
    _assert_model_routing(model, result["research_findings"])


def test_all_empty_workers_fail_before_team_a() -> None:
    provider = RecordingResearchProvider(
        {
            (1, SUBQUESTION_1): [],
            (2, SUBQUESTION_2): [],
        }
    )
    model = FakeDebateModel()

    with pytest.raises(ValueError, match="research produced no findings"):
        run_expanded_debate(
            QUESTION,
            FakeDecomposer(_complex_decomposition()),
            provider,
            model,
        )

    assert sorted(provider.calls) == sorted(
        [(1, SUBQUESTION_1), (2, SUBQUESTION_2)]
    )
    assert model.calls == []
    assert model.research_called is False


def _malformed_findings() -> list[object]:
    valid = _finding(0, QUESTION, 1)
    missing = valid.copy()
    del missing["claim"]
    unexpected = {**valid, "unexpected": "value"}
    wrong_subquestion_index = {**valid, "subquestion_index": "0"}
    wrong_finding_index = {**valid, "finding_index": "1"}
    boolean_subquestion_indexes = [
        {**valid, "subquestion_index": value} for value in (True, False)
    ]
    boolean_finding_indexes = [
        {**valid, "finding_index": value} for value in (True, False)
    ]
    wrong_text_fields = [
        {**valid, field: 3}
        for field in (
            "subquestion",
            "claim",
            "source_title",
            "source_url",
            "source_type",
            "supporting_excerpt",
        )
    ]
    empty_text_fields = [
        {**valid, field: "   "}
        for field in (
            "subquestion",
            "claim",
            "source_title",
            "source_url",
            "source_type",
            "supporting_excerpt",
        )
    ]
    nonpositive_index = {**valid, "finding_index": 0}
    mismatched_index = {**valid, "subquestion_index": 1}
    mismatched_text = {**valid, "subquestion": "A different question"}
    return [
        "not a dictionary",
        missing,
        unexpected,
        wrong_subquestion_index,
        wrong_finding_index,
        *boolean_subquestion_indexes,
        *boolean_finding_indexes,
        *wrong_text_fields,
        *empty_text_fields,
        nonpositive_index,
        mismatched_index,
        mismatched_text,
    ]


@pytest.mark.parametrize("malformed", _malformed_findings())
def test_worker_rejects_malformed_provider_findings_before_consolidation(
    malformed: object,
) -> None:
    provider = RecordingResearchProvider({(0, QUESTION): [malformed]})
    model = FakeDebateModel()

    with patch(
        "multi_agent_debate.expanded_graph.consolidate_research_findings"
    ) as consolidate:
        graph = build_expanded_graph(
            FakeDecomposer(_simple_decomposition()),
            provider,
            model,
        )
        with pytest.raises(ValueError):
            graph.invoke(
                {
                    "question": QUESTION,
                    "team_a_stance": TEAM_A_STANCE,
                    "team_b_stance": TEAM_B_STANCE,
                }
            )

    consolidate.assert_not_called()
    assert provider.calls == [(0, QUESTION)]
    assert model.calls == []
    assert model.research_called is False
