"""Offline regression tests for final cost and efficiency measurement."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from multi_agent_debate.app import (
    _parse_args as parse_product_args,
    main as product_main,
)
from multi_agent_debate.costs import (
    CostValidationError,
    Pricing,
    calculate_cost,
    validate_price,
)
from multi_agent_debate.efficiency_cli import (
    EfficiencyArtifactError,
    EfficiencyDependencies,
    _parse_args as parse_efficiency_args,
    main as efficiency_main,
    publish_efficiency_report,
)
from multi_agent_debate.efficiency_replay import execute_efficiency_replay
from multi_agent_debate.efficiency_report import (
    EfficiencyReportValidationError,
    calculate_efficiency,
    calculate_metric_token_efficiency,
    efficiency_report_to_json,
    validate_efficiency_report,
)
from multi_agent_debate.graph import TEAM_A_STANCE
from multi_agent_debate.metrics import tokenize_words
from multi_agent_debate.multi_round_run_record import (
    RUN_RECORD_VERSION,
    RUN_RECORD_VERSION_2_1,
    MultiRoundRunRecordValidationError,
    multi_round_run_record_to_json,
    validate_multi_round_run_record,
)
from multi_agent_debate.ollama_decomposer import OllamaQuestionDecomposer
from multi_agent_debate.ollama_model import OllamaDebateModel
from multi_agent_debate.ollama_team_model import OllamaTeamModel
from multi_agent_debate.replay import (
    ReplayIntegrityError,
    ReplaySource,
    load_replay_source,
)
from multi_agent_debate.research import ResearchFinding
from multi_agent_debate.usage import (
    USAGE_EXTRACTION_VERSION,
    UsageRecorder,
    UsageValidationError,
    aggregate_usage,
    extract_message_usage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "experiments/runs/m2_1b2b_ubi_qwen3_8b.json"
SOURCE_SHA256 = "6d88274129f2ae6d1f708847d300a00f76f953bc1d42fbd493bef7f190c506c7"
V20_PATH = PROJECT_ROOT / "experiments/runs/m2_3_final_multi_round_ubi_qwen3_8b.json"
MODEL = "qwen3:8b"
COMMIT = "a" * 40
NOW = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)


class StepClock:
    def __init__(self, step: float = 0.125) -> None:
        self._value = 0.0
        self._step = step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self._value += self._step
            return value


def usage_message(input_tokens: int = 10, output_tokens: int = 5) -> AIMessage:
    return AIMessage(
        content="synthetic output",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def record_usage(
    recorder: UsageRecorder,
    *,
    stage: str,
    variant: str,
    team: str,
    round_number: int | None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> None:
    recorder.record_message(
        stage=stage,
        model=MODEL,
        message=usage_message(input_tokens, output_tokens),
        duration_seconds=0.25,
        variant=variant,
        round_number=round_number,
        team=team,
    )


class UsageOneShotModel:
    def __init__(self, recorder: UsageRecorder, variant: str) -> None:
        self.recorder = recorder
        self.variant = variant
        self.argument_calls: list[dict[str, object]] = []
        self.synthesis_calls: list[dict[str, object]] = []
        self.research_calls = 0
        self._rounds = {"team_a": 0, "team_b": 0}

    def research(self, *, question: str) -> list[str]:
        self.research_calls += 1
        raise AssertionError("replay must not use model research")

    def write_argument(self, **values: object) -> str:
        team = "team_a" if values["stance"] == TEAM_A_STANCE else "team_b"
        self._rounds[team] += 1
        round_number = self._rounds[team]
        self.argument_calls.append(copy.deepcopy(values))
        record_usage(
            self.recorder,
            stage=f"{team}_argument",
            variant=self.variant,
            team=team,
            round_number=round_number,
            input_tokens=20 + round_number,
            output_tokens=8,
        )
        if team == "team_a":
            return (
                f"Team A round {round_number} supports the policy using shared "
                f"evidence and distinct safeguard alpha {round_number}."
            )
        return (
            f"Team B round {round_number} opposes the policy using shared evidence "
            f"and distinct fiscal constraint beta {round_number}."
        )

    def synthesize_neutrally(self, **values: object) -> str:
        self.synthesis_calls.append(copy.deepcopy(values))
        completed_rounds = max(self._rounds.values())
        record_usage(
            self.recorder,
            stage="neutral_synthesis",
            variant=self.variant,
            team="system",
            round_number=completed_rounds,
            input_tokens=30,
            output_tokens=12,
        )
        return (
            "Neutral synthesis represents the supporting safeguards and opposing "
            "fiscal constraints from the final arguments."
        )


class UsageTeamModel:
    def __init__(self, recorder: UsageRecorder) -> None:
        self.recorder = recorder
        self.calls: list[tuple[str, dict[str, object]]] = []

    def analyze_evidence(self, **values: object) -> dict[str, object]:
        self.calls.append(("analyze_evidence", copy.deepcopy(values)))
        finding = cast(list[ResearchFinding], values["consolidated_evidence"])[0]
        team = cast(str, values["team_name"])
        record_usage(
            self.recorder,
            stage="evidence_analysis",
            variant="hierarchical_single_round",
            team=team,
            round_number=1,
        )
        return {
            "strongest_supporting_evidence": [
                {
                    "subquestion_index": finding["subquestion_index"],
                    "finding_index": finding["finding_index"],
                    "relevance": "The finding directly informs this stance.",
                }
            ],
            "limitations_or_conflicting_evidence": [],
            "key_claims_to_make": ["Make one grounded and bounded claim."],
            "unsupported_claims_to_avoid": ["Avoid claims beyond the evidence."],
            "opponent_points_to_address": (
                [] if team == "team_a" else ["Address Team A's final argument."]
            ),
        }

    def draft_argument(self, **values: object) -> str:
        self.calls.append(("draft_argument", copy.deepcopy(values)))
        team = cast(str, values["team_name"])
        record_usage(
            self.recorder,
            stage="argument_draft",
            variant="hierarchical_single_round",
            team=team,
            round_number=1,
        )
        return f"{team} drafts a substantive grounded argument using replay evidence."

    def review_and_revise(self, **values: object) -> dict[str, object]:
        self.calls.append(("review_and_revise", copy.deepcopy(values)))
        team = cast(str, values["team_name"])
        record_usage(
            self.recorder,
            stage="review_revision",
            variant="hierarchical_single_round",
            team=team,
            round_number=1,
        )
        return {
            "critique": {
                "stance_consistency": "The stance is consistent.",
                "evidence_usage": "The evidence is used directly.",
                "unsupported_factual_claims": "Unsupported claims are avoided.",
                "opponent_response": "The opponent response is proportionate.",
                "unnecessary_repetition": "Repetition is limited.",
                "clarity_and_concision": "The argument is clear and concise.",
            },
            "revised_argument": (
                f"{team} presents a revised substantive argument grounded in the "
                "shared replay evidence."
            ),
        }


class UsageSynthesisModel:
    def __init__(self, recorder: UsageRecorder) -> None:
        self.recorder = recorder
        self.calls: list[dict[str, object]] = []

    def synthesize_neutrally(self, **values: object) -> str:
        self.calls.append(copy.deepcopy(values))
        record_usage(
            self.recorder,
            stage="neutral_synthesis",
            variant="hierarchical_single_round",
            team="system",
            round_number=1,
            input_tokens=30,
            output_tokens=12,
        )
        return (
            "Neutral synthesis represents both hierarchical final arguments and "
            "their shared grounded evidence."
        )


@pytest.fixture(scope="module")
def replay_source() -> ReplaySource:
    return load_replay_source(SOURCE_PATH, expected_sha256=SOURCE_SHA256)


@pytest.fixture()
def efficiency_execution(replay_source: ReplaySource) -> dict[str, object]:
    single_usage = UsageRecorder()
    two_usage = UsageRecorder()
    hierarchical_usage = UsageRecorder()
    single = UsageOneShotModel(single_usage, "single_round_one_shot")
    two = UsageOneShotModel(two_usage, "two_round_one_shot")
    team = UsageTeamModel(hierarchical_usage)
    synthesis = UsageSynthesisModel(hierarchical_usage)
    artifacts = execute_efficiency_replay(
        replay_source=replay_source,
        single_round_model=single,
        two_round_model=two,
        hierarchical_team_model=team,
        hierarchical_synthesis_model=synthesis,
        single_round_usage=single_usage,
        two_round_usage=two_usage,
        hierarchical_usage=hierarchical_usage,
        pricing=Pricing(Decimal("1.25"), Decimal("2.5")),
        clock=StepClock(),
        utc_now=lambda: NOW,
        implementation_commit=COMMIT,
        model_name=MODEL,
    )
    return {
        "artifacts": artifacts,
        "single": single,
        "two": two,
        "team": team,
        "synthesis": synthesis,
    }


def _v21_from_committed_v20() -> dict[str, object]:
    record = json.loads(V20_PATH.read_text(encoding="utf-8"))
    rounds = len(record["debate_rounds"])
    recorder = UsageRecorder()
    record_usage(
        recorder,
        stage="decomposition",
        variant="final_product",
        team="system",
        round_number=None,
    )
    for number in range(1, rounds + 1):
        for team in ("team_a", "team_b"):
            record_usage(
                recorder,
                stage=f"{team}_argument",
                variant="final_product",
                team=team,
                round_number=number,
            )
    record_usage(
        recorder,
        stage="neutral_synthesis",
        variant="final_product",
        team="system",
        round_number=rounds,
    )
    entries = recorder.snapshot()
    totals = aggregate_usage(entries)
    visible_words = sum(
        len(tokenize_words(record["graph_evidence"][field]))
        for field in (
            "final_team_a_argument",
            "final_team_b_argument",
            "neutral_synthesis",
        )
    )
    all_words = (
        len(tokenize_words(record["decomposition"]["question_complexity"]))
        + sum(
            len(tokenize_words(item))
            for item in record["decomposition"]["subquestions"]
        )
        + sum(
            len(tokenize_words(item["team_a_argument"]))
            + len(tokenize_words(item["team_b_argument"]))
            for item in record["debate_rounds"]
        )
        + len(tokenize_words(record["graph_evidence"]["neutral_synthesis"]))
    )
    pricing = Pricing(Decimal("1"), Decimal("2"))
    record["run_record_version"] = RUN_RECORD_VERSION_2_1
    record["usage"] = {
        "usage_extraction_version": USAGE_EXTRACTION_VERSION,
        "entries": entries,
        "totals": totals,
    }
    record["cost_estimate"] = calculate_cost(totals, pricing=pricing)
    record["efficiency"] = calculate_efficiency(
        usage_totals=totals,
        visible_output_words=visible_words,
        all_generated_words=all_words,
        latency_seconds=record["timings"]["total_duration_seconds"],
        completed_rounds=rounds,
    )
    record["metric_token_efficiency"] = calculate_metric_token_efficiency(
        record["metrics"], total_tokens=totals["total_tokens"]
    )
    return record


def test_standard_usage_metadata_is_extracted() -> None:
    assert extract_message_usage(usage_message(11, 7)) == {
        "usage_status": "available",
        "usage_source": "usage_metadata",
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "unavailable_reason": None,
    }


def test_ollama_response_metadata_is_extracted() -> None:
    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"prompt_eval_count": 9, "eval_count": 4},
    )
    usage = extract_message_usage(message)
    assert usage["usage_source"] == "ollama_response_metadata"
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (
        9,
        4,
        13,
    )


def test_standard_usage_takes_precedence_without_double_counting() -> None:
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        response_metadata={"prompt_eval_count": 5, "eval_count": 3},
    )
    assert extract_message_usage(message)["usage_source"] == "usage_metadata"


def test_disagreeing_metadata_is_rejected() -> None:
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        response_metadata={"prompt_eval_count": 6, "eval_count": 3},
    )
    with pytest.raises(UsageValidationError, match="disagree"):
        extract_message_usage(message)


def test_missing_usage_is_explicitly_unavailable() -> None:
    usage = extract_message_usage(AIMessage(content="no metadata"))
    assert usage["usage_status"] == "unavailable"
    assert usage["usage_source"] == "unavailable"
    assert usage["input_tokens"] is usage["output_tokens"] is usage["total_tokens"] is None
    assert usage["unavailable_reason"] == "metadata_unavailable"


@pytest.mark.parametrize(
    "metadata",
    [
        {"input_tokens": -1, "output_tokens": 1, "total_tokens": 0},
        {"input_tokens": True, "output_tokens": 1, "total_tokens": 2},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 3},
    ],
)
def test_invalid_standard_usage_is_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(UsageValidationError):
        extract_message_usage(SimpleNamespace(usage_metadata=metadata, response_metadata={}))


@pytest.mark.parametrize(
    "metadata",
    [
        {"prompt_eval_count": True, "eval_count": 1},
        {"prompt_eval_count": -1, "eval_count": 1},
        {"prompt_eval_count": 1},
    ],
)
def test_invalid_ollama_usage_is_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(UsageValidationError):
        extract_message_usage(SimpleNamespace(usage_metadata=None, response_metadata=metadata))


def test_usage_recorder_is_defensive_and_canonical() -> None:
    recorder = UsageRecorder()
    record_usage(
        recorder,
        stage="neutral_synthesis",
        variant="final_product",
        team="system",
        round_number=1,
    )
    record_usage(
        recorder,
        stage="decomposition",
        variant="final_product",
        team="system",
        round_number=None,
    )
    snapshot = recorder.snapshot()
    assert [item["stage"] for item in snapshot] == [
        "decomposition",
        "neutral_synthesis",
    ]
    snapshot[0]["model"] = "mutated"
    assert recorder.snapshot()[0]["model"] == MODEL
    assert "prompt" not in json.dumps(recorder.snapshot()).casefold()
    assert "synthetic output" not in json.dumps(recorder.snapshot())


def test_usage_recorder_is_thread_safe() -> None:
    recorder = UsageRecorder()

    def add(number: int) -> None:
        record_usage(
            recorder,
            stage="team_a_argument",
            variant=f"custom-{number}",
            team="team_a",
            round_number=1,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(40)))
    assert len(recorder.snapshot()) == 40
    assert {item["variant"] for item in recorder.snapshot()} == {
        f"custom-{number}" for number in range(40)
    }


@pytest.mark.parametrize("value", [True, False, -1, -0.5, math.nan, math.inf, -math.inf, "1"])
def test_price_validation_rejects_invalid_values(value: object) -> None:
    with pytest.raises(CostValidationError):
        validate_price(value)


def test_cost_formula_is_decimal_exact() -> None:
    totals = {
        "model_calls": 1,
        "available_calls": 1,
        "unavailable_calls": 0,
        "all_usage_available": True,
        "input_tokens": 250_000,
        "output_tokens": 100_000,
        "total_tokens": 350_000,
    }
    cost = calculate_cost(totals, pricing=Pricing(Decimal("2"), Decimal("6")))
    assert cost["estimated_cost"] == "1.1"
    assert cost["pricing_basis"] == "user_supplied_rates"


def test_zero_local_pricing_and_unavailable_usage() -> None:
    available = aggregate_usage(
        [
            {
                "stage": "decomposition",
                "model": MODEL,
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "duration_seconds": 0.1,
                "usage_source": "usage_metadata",
                "usage_status": "available",
                "unavailable_reason": None,
                "variant": "final_product",
                "round_number": None,
                "team": "system",
            }
        ]
    )
    assert calculate_cost(available, pricing=Pricing()) == {
        "status": "available",
        "estimated_cost": "0",
        "pricing_basis": "local_zero_default",
        "input_cost_per_million": "0",
        "output_cost_per_million": "0",
        "unavailable_reason": None,
    }
    recorder = UsageRecorder()
    recorder.record_message(
        stage="decomposition",
        model=MODEL,
        message=AIMessage(content="missing"),
        duration_seconds=0.1,
        variant="final_product",
        team="system",
    )
    totals = aggregate_usage(recorder.snapshot())
    assert totals["input_tokens"] == 0
    assert totals["unavailable_calls"] == 1
    unavailable = calculate_cost(totals, pricing=Pricing())
    assert unavailable["status"] == "unavailable"
    assert unavailable["estimated_cost"] is None


def test_efficiency_ratios_and_zero_denominators() -> None:
    totals = aggregate_usage(
        [
            {
                "stage": "neutral_synthesis",
                "model": MODEL,
                "input_tokens": 60,
                "output_tokens": 40,
                "total_tokens": 100,
                "duration_seconds": 0.2,
                "usage_source": "usage_metadata",
                "usage_status": "available",
                "unavailable_reason": None,
                "variant": "single_round_one_shot",
                "round_number": 1,
                "team": "system",
            }
        ]
    )
    efficiency = calculate_efficiency(
        usage_totals=totals,
        visible_output_words=25,
        all_generated_words=40,
        latency_seconds=1.5,
        completed_rounds=1,
    )
    assert efficiency["tokens_per_call"] == 100
    assert efficiency["tokens_per_visible_output_word"] == 4
    no_words = calculate_efficiency(
        usage_totals=totals,
        visible_output_words=0,
        all_generated_words=0,
        latency_seconds=0,
        completed_rounds=1,
    )
    assert no_words["tokens_per_visible_output_word"] is None


def test_per_metric_ratios_have_no_weighted_score() -> None:
    record = json.loads(V20_PATH.read_text(encoding="utf-8"))
    ratios = calculate_metric_token_efficiency(record["metrics"], total_tokens=1000)
    assert ratios["research_utilization_per_1k_tokens"] == record["metrics"][
        "research_utilization"
    ]["score"]
    assert ratios["summary_policy_compliance"] == 1
    assert "weighted" not in json.dumps(ratios).casefold()
    assert all(value is None for key, value in calculate_metric_token_efficiency(
        record["metrics"], total_tokens=0
    ).items() if key.endswith("per_1k_tokens"))


def test_committed_v20_artifact_remains_valid() -> None:
    value = json.loads(V20_PATH.read_text(encoding="utf-8"))
    validated = validate_multi_round_run_record(value)
    assert validated["run_record_version"] == RUN_RECORD_VERSION == "2.0"
    assert multi_round_run_record_to_json(value).endswith("\n")


def test_v21_record_validates_and_serializes() -> None:
    record = _v21_from_committed_v20()
    validated = validate_multi_round_run_record(record)
    assert validated["run_record_version"] == RUN_RECORD_VERSION_2_1
    assert validated["usage"]["totals"]["model_calls"] == 6
    assert validated["cost_estimate"]["estimated_cost"] == "0.00012"
    serialized = multi_round_run_record_to_json(record)
    assert json.loads(serialized)["run_record_version"] == "2.1"
    assert serialized.endswith("\n") and not serialized.endswith("\n\n")


def test_v21_record_preserves_explicitly_unavailable_usage() -> None:
    record = _v21_from_committed_v20()
    entry = record["usage"]["entries"][0]
    entry.update(
        usage_status="unavailable",
        usage_source="unavailable",
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        unavailable_reason="metadata_unavailable",
    )
    totals = aggregate_usage(record["usage"]["entries"])
    record["usage"]["totals"] = totals
    record["cost_estimate"] = calculate_cost(
        totals, pricing=Pricing(Decimal("1"), Decimal("2"))
    )
    existing_efficiency = record["efficiency"]
    record["efficiency"] = calculate_efficiency(
        usage_totals=totals,
        visible_output_words=existing_efficiency["visible_output_words"],
        all_generated_words=existing_efficiency["all_generated_words"],
        latency_seconds=existing_efficiency["latency_seconds"],
        completed_rounds=existing_efficiency["completed_rounds"],
    )
    record["metric_token_efficiency"] = calculate_metric_token_efficiency(
        record["metrics"], total_tokens=None
    )
    validated = validate_multi_round_run_record(record)
    assert validated["usage"]["totals"]["unavailable_calls"] == 1
    assert validated["usage"]["totals"]["input_tokens"] > 0
    assert validated["efficiency"]["total_tokens"] is None
    assert validated["cost_estimate"]["status"] == "unavailable"
    assert validated["cost_estimate"]["estimated_cost"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["usage"]["entries"][0].update(total_tokens=999),
        lambda value: value["usage"]["entries"].pop(),
        lambda value: value["cost_estimate"].update(estimated_cost="123"),
        lambda value: value["efficiency"].update(total_tokens=999),
        lambda value: value["metric_token_efficiency"].update(
            research_utilization_per_1k_tokens=999
        ),
        lambda value: value["cost_estimate"].update(input_cost_per_million="3"),
    ],
)
def test_v21_tampering_is_rejected(mutation: object) -> None:
    record = _v21_from_committed_v20()
    cast(object, mutation)(record)  # type: ignore[operator]
    with pytest.raises(MultiRoundRunRecordValidationError):
        validate_multi_round_run_record(record)


class FakeRunnable:
    def __init__(self, response: object) -> None:
        self.response = response

    def invoke(self, messages: object) -> object:
        return copy.deepcopy(self.response)


class FakeChat:
    instances: list[FakeChat] = []
    structured_response: object = None
    plain_response: AIMessage = usage_message()

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.structured_calls: list[tuple[object, dict[str, object]]] = []
        self.__class__.instances.append(self)

    def with_structured_output(self, schema: object, **kwargs: object) -> FakeRunnable:
        self.structured_calls.append((schema, kwargs.copy()))
        return FakeRunnable(self.__class__.structured_response)

    def invoke(self, messages: object) -> AIMessage:
        return copy.deepcopy(self.__class__.plain_response)


def test_ollama_debate_adapter_records_plain_usage() -> None:
    FakeChat.instances.clear()
    recorder = UsageRecorder()
    with patch("multi_agent_debate.ollama_model.ChatOllama", FakeChat):
        model = OllamaDebateModel(
            usage_recorder=recorder,
            usage_variant="single_round_one_shot",
            clock=StepClock(),
        )
        assert model.write_argument(
            question="Question?",
            stance=TEAM_A_STANCE,
            research_findings=["Evidence"],
            opponent_argument=None,
        ) == "synthetic output"
    entry = recorder.snapshot()[0]
    assert entry["stage"] == "team_a_argument"
    assert entry["total_tokens"] == 15
    assert FakeChat.instances[0].structured_calls[0][1]["include_raw"] is True


def test_ollama_decomposer_records_structured_usage() -> None:
    FakeChat.instances.clear()
    FakeChat.structured_response = {
        "raw": usage_message(4, 2),
        "parsed": {"question_complexity": "simple", "subquestions": []},
    }
    recorder = UsageRecorder()
    with patch("multi_agent_debate.ollama_decomposer.ChatOllama", FakeChat):
        result = OllamaQuestionDecomposer(
            usage_recorder=recorder,
            usage_variant="final_product",
            clock=StepClock(),
        ).decompose(question="Question?")
    assert result == {"question_complexity": "simple", "subquestions": []}
    assert recorder.snapshot()[0]["stage"] == "decomposition"
    assert recorder.snapshot()[0]["total_tokens"] == 6
    assert FakeChat.instances[0].structured_calls[0][1]["include_raw"] is True


def test_ollama_team_adapter_records_all_three_stages(
    replay_source: ReplaySource,
) -> None:
    FakeChat.instances.clear()
    FakeChat.plain_response = AIMessage(
        content="Synthetic output remains a substantive draft argument.",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    )
    finding = replay_source["source_record"]["graph_evidence"][
        "consolidated_evidence"
    ][0]
    FakeChat.structured_response = {
        "raw": usage_message(4, 2),
        "parsed": {
            "strongest_supporting_evidence": [
                {
                    "subquestion_index": finding["subquestion_index"],
                    "finding_index": finding["finding_index"],
                    "relevance": "Relevant evidence.",
                }
            ],
            "limitations_or_conflicting_evidence": [],
            "key_claims_to_make": ["Make one supported claim."],
            "unsupported_claims_to_avoid": [],
            "opponent_points_to_address": [],
        },
    }
    recorder = UsageRecorder()
    with patch("multi_agent_debate.ollama_team_model.ChatOllama", FakeChat):
        model = OllamaTeamModel(
            usage_recorder=recorder,
            usage_variant="hierarchical_single_round",
            clock=StepClock(),
        )
        analysis = model.analyze_evidence(
            question="Question?",
            team_name="team_a",
            stance="Support",
            consolidated_evidence=[finding],
            opponent_argument=None,
        )
        model.draft_argument(
            question="Question?",
            team_name="team_a",
            stance="Support",
            research_findings=["Formatted evidence"],
            evidence_analysis=analysis,
            opponent_argument=None,
        )
        FakeChat.instances[0].structured_calls[1][0]
        model._structured_revision = FakeRunnable(
            {
                "raw": usage_message(5, 3),
                "parsed": {
                    "critique": {
                        "stance_consistency": "Consistent stance.",
                        "evidence_usage": "Evidence is used.",
                        "unsupported_factual_claims": "None remain.",
                        "opponent_response": "No opponent was supplied.",
                        "unnecessary_repetition": "No repetition remains.",
                        "clarity_and_concision": "The draft is clear.",
                    },
                    "revised_argument": "Synthetic output remains a substantive final argument.",
                },
            }
        )
        model.review_and_revise(
            question="Question?",
            team_name="team_a",
            stance="Support",
            consolidated_evidence=[finding],
            evidence_analysis=analysis,
            draft_argument="Synthetic output remains a substantive draft argument.",
            opponent_argument=None,
        )
    assert [item["stage"] for item in recorder.snapshot()] == [
        "evidence_analysis",
        "argument_draft",
        "review_revision",
    ]
    assert all(
        call[1].get("include_raw") is True
        for call in FakeChat.instances[0].structured_calls
    )


def test_efficiency_replay_has_exact_calls_and_identical_evidence(
    efficiency_execution: dict[str, object], replay_source: ReplaySource
) -> None:
    report = cast(dict[str, object], efficiency_execution["artifacts"])["report"]
    variants = cast(dict[str, object], report)["variants"]
    assert [item["model_call_breakdown"]["total_model_calls"] for item in variants] == [
        3,
        5,
        7,
    ]
    assert [item["efficiency"]["model_calls"] for item in variants] == [3, 5, 7]
    assert cast(UsageOneShotModel, efficiency_execution["single"]).research_calls == 0
    assert cast(UsageOneShotModel, efficiency_execution["two"]).research_calls == 0
    assert len(cast(UsageTeamModel, efficiency_execution["team"]).calls) == 6
    source_evidence = replay_source["source_record"]["graph_evidence"][
        "research_findings"
    ]
    single_model = cast(UsageOneShotModel, efficiency_execution["single"])
    two_model = cast(UsageOneShotModel, efficiency_execution["two"])
    team_model = cast(UsageTeamModel, efficiency_execution["team"])
    evidence_values = [single_model.argument_calls[0]["research_findings"]]
    evidence_values.append(two_model.argument_calls[0]["research_findings"])
    evidence_values.append(team_model.calls[0][1]["consolidated_evidence"])
    assert evidence_values[0] == evidence_values[1] == source_evidence
    assert evidence_values[2] == replay_source["source_record"]["graph_evidence"][
        "consolidated_evidence"
    ]
    assert id(evidence_values[0]) != id(evidence_values[1])


def test_efficiency_report_schema_deltas_and_serialization(
    efficiency_execution: dict[str, object], replay_source: ReplaySource
) -> None:
    artifacts = cast(dict[str, object], efficiency_execution["artifacts"])
    report = validate_efficiency_report(
        artifacts["report"], replay_source=replay_source
    )
    assert set(report) == {
        "efficiency_report_version",
        "provenance",
        "replay_source",
        "pricing",
        "variants",
        "comparison",
        "limitations",
    }
    assert "winner" not in json.dumps(report).casefold()
    assert "weighted" not in json.dumps(report).casefold()
    assert report["comparison"][0]["model_calls"] == 2
    assert report["comparison"][1]["model_calls"] == 4
    serialized = efficiency_report_to_json(report, replay_source=replay_source)
    assert serialized == artifacts["report_json"]
    assert serialized.endswith("\n") and not serialized.endswith("\n\n")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["variants"][0]["usage_entries"].pop(),
        lambda value: value["variants"][0]["usage_entries"][0].update(total_tokens=999),
        lambda value: value["variants"][0]["cost_estimate"].update(estimated_cost="9"),
        lambda value: value["variants"][0]["metric_token_efficiency"].update(
            team_novelty_per_1k_tokens=9
        ),
        lambda value: value["comparison"][0].update(total_tokens=999),
    ],
)
def test_efficiency_report_tampering_is_rejected(
    mutation: object,
    efficiency_execution: dict[str, object],
    replay_source: ReplaySource,
) -> None:
    report = copy.deepcopy(cast(dict[str, object], efficiency_execution["artifacts"])["report"])
    cast(object, mutation)(report)  # type: ignore[operator]
    with pytest.raises(EfficiencyReportValidationError):
        validate_efficiency_report(report, replay_source=replay_source)


def test_source_sha_mismatch_is_rejected() -> None:
    with pytest.raises(ReplayIntegrityError):
        load_replay_source(SOURCE_PATH, expected_sha256="0" * 64)


def test_product_and_efficiency_cli_pricing_defaults() -> None:
    product = parse_product_args(
        [
            "--question", "Question?",
            "--implementation-commit", COMMIT,
            "--output", "result.json",
        ]
    )
    efficiency = parse_efficiency_args(
        [
            "--source", str(SOURCE_PATH),
            "--source-sha256", SOURCE_SHA256,
            "--implementation-commit", COMMIT,
            "--output", "efficiency.json",
        ]
    )
    assert product.input_cost_per_million == product.output_cost_per_million == Decimal("0")
    assert efficiency.input_cost_per_million == efficiency.output_cost_per_million == Decimal("0")


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "true", "false", "not-a-number"])
def test_cli_invalid_pricing_is_rejected(value: str) -> None:
    with pytest.raises(Exception):
        parse_efficiency_args(
            [
                "--source", str(SOURCE_PATH),
                "--source-sha256", SOURCE_SHA256,
                "--implementation-commit", COMMIT,
                "--input-cost-per-million", value,
                "--output", "efficiency.json",
            ]
        )


def test_efficiency_cli_validates_before_model_construction(tmp_path: Path) -> None:
    called = False

    def dependencies(**kwargs: object) -> EfficiencyDependencies:
        nonlocal called
        called = True
        raise AssertionError

    result = efficiency_main(
        [
            "--source", str(SOURCE_PATH),
            "--source-sha256", "0" * 64,
            "--implementation-commit", COMMIT,
            "--output", str(tmp_path / "result.json"),
        ],
        dependency_builder=dependencies,
    )
    assert result == 2
    assert called is False


def test_product_cli_rejects_pricing_before_live_construction(tmp_path: Path) -> None:
    called = False

    def dependencies(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    result = product_main(
        [
            "--question", "Question?",
            "--implementation-commit", COMMIT,
            "--input-cost-per-million", "nan",
            "--output", str(tmp_path / "result.json"),
        ],
        dependency_builder=dependencies,
        environ={},
    )
    assert result == 2
    assert called is False


def test_efficiency_cli_complete_offline_path(
    tmp_path: Path,
    replay_source: ReplaySource,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "efficiency.json"
    constructed = False

    def dependencies(**kwargs: object) -> EfficiencyDependencies:
        nonlocal constructed
        constructed = True
        single_usage = UsageRecorder()
        two_usage = UsageRecorder()
        hierarchical_usage = UsageRecorder()
        return EfficiencyDependencies(
            UsageOneShotModel(single_usage, "single_round_one_shot"),
            UsageOneShotModel(two_usage, "two_round_one_shot"),
            UsageTeamModel(hierarchical_usage),
            UsageSynthesisModel(hierarchical_usage),
            single_usage,
            two_usage,
            hierarchical_usage,
        )

    result = efficiency_main(
        [
            "--source", str(SOURCE_PATH),
            "--source-sha256", SOURCE_SHA256,
            "--implementation-commit", COMMIT,
            "--output", str(output),
            "--summary",
        ],
        dependency_builder=dependencies,
        clock=StepClock(),
        utc_now=lambda: NOW,
    )
    assert result == 0 and constructed and output.exists()
    report = validate_efficiency_report(
        json.loads(output.read_text(encoding="utf-8")), replay_source=replay_source
    )
    assert [item["efficiency"]["model_calls"] for item in report["variants"]] == [3, 5, 7]
    stdout = capsys.readouterr().out
    for forbidden in ("https://", "snippet", "draft", "critique", "prompt"):
        assert forbidden not in stdout.casefold()


def test_model_failure_publishes_no_efficiency_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "efficiency.json"
    empty = UsageRecorder()
    dependencies = EfficiencyDependencies(
        cast(object, object()),
        cast(object, object()),
        cast(object, object()),
        cast(object, object()),
        empty,
        UsageRecorder(),
        UsageRecorder(),
    )

    def fail(**kwargs: object) -> object:
        raise RuntimeError("private model response and prompt")

    result = efficiency_main(
        [
            "--source", str(SOURCE_PATH),
            "--source-sha256", SOURCE_SHA256,
            "--implementation-commit", COMMIT,
            "--output", str(output),
        ],
        dependency_builder=lambda **kwargs: dependencies,
        executor=fail,
    )
    assert result == 1
    assert not output.exists()


def test_atomic_publication_refuses_overwrite_and_replaces_explicitly(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    output.write_bytes(b"original")
    with pytest.raises(Exception):
        publish_efficiency_report(output=output, report_json="{}\n", overwrite=False)
    assert output.read_bytes() == b"original"
    publish_efficiency_report(output=output, report_json="{}\n", overwrite=True)
    assert output.read_bytes() == b"{}\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_overwrite_failure_restores_exact_bytes(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    original = b"recognizable original bytes\n"
    output.write_bytes(original)
    real_replace = os.replace
    replacements = 0

    def fail_after_replace(source: object, destination: object) -> None:
        nonlocal replacements
        replacements += 1
        real_replace(source, destination)
        if replacements == 1:
            raise OSError("private filesystem detail")

    with patch("multi_agent_debate.efficiency_cli.os.replace", fail_after_replace):
        with pytest.raises(EfficiencyArtifactError):
            publish_efficiency_report(
                output=output,
                report_json="{\"new\": true}\n",
                overwrite=True,
            )
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
