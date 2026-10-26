"""Pure sequential replay of the three frozen debate architectures."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from datetime import datetime
from typing import TypedDict

from multi_agent_debate.costs import Pricing
from multi_agent_debate.efficiency_report import (
    EfficiencyReport,
    EfficiencyVariant,
    FinalVisibleOutputs,
    GeneratedStageWordCount,
    VariantId,
    build_efficiency_report,
    build_efficiency_variant,
    count_generated_words,
    efficiency_report_to_json,
)
from multi_agent_debate.hierarchical_graph import run_hierarchical_debate
from multi_agent_debate.model import DebateModel
from multi_agent_debate.multi_round_graph import run_multi_round_debate
from multi_agent_debate.multi_round_run_record import score_multi_round_result
from multi_agent_debate.multi_round_trace import (
    MultiRoundTraceRecorder,
    canonicalize_successful_traces,
)
from multi_agent_debate.replay import (
    ReplayQuestionDecomposer,
    ReplayResearchProvider,
    ReplaySource,
    validate_replay_source,
)
from multi_agent_debate.run_record import format_utc_timestamp, score_expanded_result
from multi_agent_debate.team_model import NeutralSynthesisModel, TeamModel
from multi_agent_debate.team_trace import (
    TeamTraceRecorder,
    canonicalize_successful_team_traces,
)
from multi_agent_debate.usage import UsageRecorder


class EfficiencyReplayError(RuntimeError):
    """Base bounded error for controlled efficiency replay."""


class EfficiencyReplayValidationError(EfficiencyReplayError):
    """A replay dependency, trace, or result violates the comparison contract."""


class EfficiencyReplayArtifacts(TypedDict):
    report: EfficiencyReport
    report_json: str


def execute_efficiency_replay(
    *,
    replay_source: ReplaySource,
    single_round_model: DebateModel,
    two_round_model: DebateModel,
    hierarchical_team_model: TeamModel,
    hierarchical_synthesis_model: NeutralSynthesisModel,
    single_round_usage: UsageRecorder,
    two_round_usage: UsageRecorder,
    hierarchical_usage: UsageRecorder,
    pricing: Pricing,
    clock: Callable[[], float],
    utc_now: Callable[[], datetime],
    implementation_commit: str,
    model_name: str,
) -> EfficiencyReplayArtifacts:
    """Run single, two-round, then hierarchical variants on frozen evidence."""

    source = validate_replay_source(replay_source)
    if not callable(clock) or not callable(utc_now):
        raise EfficiencyReplayValidationError("efficiency replay clocks are invalid")
    if not isinstance(model_name, str) or not model_name.strip():
        raise EfficiencyReplayValidationError("efficiency replay model is invalid")
    for recorder in (single_round_usage, two_round_usage, hierarchical_usage):
        if not isinstance(recorder, UsageRecorder) or recorder.snapshot():
            raise EfficiencyReplayValidationError(
                "efficiency replay requires empty usage recorders"
            )
    try:
        timestamp = format_utc_timestamp(utc_now())
    except Exception:
        raise EfficiencyReplayValidationError("efficiency replay timestamp is invalid") from None

    variants: list[EfficiencyVariant] = []
    variants.append(
        _run_one_shot_variant(
            variant_id="single_round_one_shot",
            source=source,
            model=single_round_model,
            usage_recorder=single_round_usage,
            max_rounds=1,
            pricing=pricing,
            clock=clock,
        )
    )
    variants.append(
        _run_one_shot_variant(
            variant_id="two_round_one_shot",
            source=source,
            model=two_round_model,
            usage_recorder=two_round_usage,
            max_rounds=2,
            pricing=pricing,
            clock=clock,
        )
    )
    variants.append(
        _run_hierarchical_variant(
            source=source,
            team_model=hierarchical_team_model,
            synthesis_model=hierarchical_synthesis_model,
            usage_recorder=hierarchical_usage,
            pricing=pricing,
            clock=clock,
        )
    )
    report = build_efficiency_report(
        replay_source=source,
        variants=variants,
        pricing=pricing,
        run_timestamp_utc=timestamp,
        implementation_commit=implementation_commit,
        model_name=model_name,
    )
    return {
        "report": copy.deepcopy(report),
        "report_json": efficiency_report_to_json(report, replay_source=source),
    }


def _run_one_shot_variant(
    *,
    variant_id: VariantId,
    source: ReplaySource,
    model: DebateModel,
    usage_recorder: UsageRecorder,
    max_rounds: int,
    pricing: Pricing,
    clock: Callable[[], float],
) -> EfficiencyVariant:
    recorder = MultiRoundTraceRecorder()
    started = _clock_value(clock)
    result = run_multi_round_debate(
        source["source_record"]["input"]["question"],
        ReplayQuestionDecomposer(source),
        ReplayResearchProvider(source),
        model,
        max_rounds=max_rounds,
        novelty_threshold=0.15,
        trace_recorder=recorder,
        clock=clock,
    )
    latency = _elapsed(clock, started)
    traces = canonicalize_successful_traces(
        recorder=recorder,
        completed_rounds=len(result["debate_rounds"]),
    )
    _require_replayed_evidence(result, source)
    expected_calls = 2 * len(result["debate_rounds"]) + 1
    if sum(trace["stage"] != "termination" for trace in traces) != expected_calls:
        raise EfficiencyReplayValidationError(
            "one-shot model calls do not match graph traces"
        )
    words: list[GeneratedStageWordCount] = []
    for round_record in result["debate_rounds"]:
        number = round_record["round_number"]
        words.extend(
            [
                {
                    "stage": "team_a_argument",
                    "team": "team_a",
                    "round_number": number,
                    "word_count": count_generated_words(
                        round_record["team_a_argument"]
                    ),
                },
                {
                    "stage": "team_b_argument",
                    "team": "team_b",
                    "round_number": number,
                    "word_count": count_generated_words(
                        round_record["team_b_argument"]
                    ),
                },
            ]
        )
    words.append(
        {
            "stage": "neutral_synthesis",
            "team": "system",
            "round_number": len(result["debate_rounds"]),
            "word_count": count_generated_words(result["synthesis"]),
        }
    )
    outputs: FinalVisibleOutputs = {
        "team_a_argument": result["team_a_argument"],
        "team_b_argument": result["team_b_argument"],
        "neutral_synthesis": result["synthesis"],
    }
    call_breakdown = {
        "team_a_argument": len(result["debate_rounds"]),
        "team_b_argument": len(result["debate_rounds"]),
        "neutral_synthesis": 1,
        "replay_decomposition": 0,
        "replay_research": 0,
        "total_model_calls": expected_calls,
    }
    return build_efficiency_variant(
        variant_id=variant_id,
        completed_rounds=len(result["debate_rounds"]),
        termination_reason=result["termination_decision"]["reason"],
        model_call_breakdown=call_breakdown,
        usage_entries=usage_recorder.snapshot(),
        generated_stage_word_counts=words,
        final_visible_outputs=outputs,
        latency_seconds=latency,
        pricing=pricing,
        scorecard=score_multi_round_result(result),
    )


def _run_hierarchical_variant(
    *,
    source: ReplaySource,
    team_model: TeamModel,
    synthesis_model: NeutralSynthesisModel,
    usage_recorder: UsageRecorder,
    pricing: Pricing,
    clock: Callable[[], float],
) -> EfficiencyVariant:
    recorder = TeamTraceRecorder()
    started = _clock_value(clock)
    result = run_hierarchical_debate(
        source["source_record"]["input"]["question"],
        ReplayQuestionDecomposer(source),
        ReplayResearchProvider(source),
        team_model,
        synthesis_model,
        team_trace_recorder=recorder,
        clock=clock,
    )
    latency = _elapsed(clock, started)
    traces = canonicalize_successful_team_traces(recorder)
    _require_replayed_evidence(result, source)
    if len(traces["stage_traces"]) != 6:
        raise EfficiencyReplayValidationError("hierarchical stage traces are incomplete")
    words: list[GeneratedStageWordCount] = []
    for team, product in (
        ("team_a", result["team_a_work_product"]),
        ("team_b", result["team_b_work_product"]),
    ):
        words.extend(
            [
                {
                    "stage": "evidence_analysis",
                    "team": team,
                    "round_number": 1,
                    "word_count": _nested_text_words(product["evidence_analysis"]),
                },
                {
                    "stage": "argument_draft",
                    "team": team,
                    "round_number": 1,
                    "word_count": count_generated_words(product["draft_argument"]),
                },
                {
                    "stage": "review_revision",
                    "team": team,
                    "round_number": 1,
                    "word_count": (
                        _nested_text_words(product["internal_critique"])
                        + count_generated_words(product["final_argument"])
                    ),
                },
            ]
        )
    words.append(
        {
            "stage": "neutral_synthesis",
            "team": "system",
            "round_number": 1,
            "word_count": count_generated_words(result["synthesis"]),
        }
    )
    outputs: FinalVisibleOutputs = {
        "team_a_argument": result["team_a_argument"],
        "team_b_argument": result["team_b_argument"],
        "neutral_synthesis": result["synthesis"],
    }
    return build_efficiency_variant(
        variant_id="hierarchical_single_round",
        completed_rounds=1,
        termination_reason=None,
        model_call_breakdown={
            "team_a_analysis": 1,
            "team_a_draft": 1,
            "team_a_review": 1,
            "team_b_analysis": 1,
            "team_b_draft": 1,
            "team_b_review": 1,
            "neutral_synthesis": 1,
            "replay_decomposition": 0,
            "replay_research": 0,
            "total_model_calls": 7,
        },
        usage_entries=usage_recorder.snapshot(),
        generated_stage_word_counts=words,
        final_visible_outputs=outputs,
        latency_seconds=latency,
        pricing=pricing,
        scorecard=score_expanded_result(result),
    )


def _require_replayed_evidence(result: dict, source: ReplaySource) -> None:
    expected = source["source_record"]["graph_evidence"]
    for field in (
        "raw_research_findings",
        "consolidated_evidence",
        "research_findings",
    ):
        if _canonical_json(result[field]) != _canonical_json(expected[field]):
            raise EfficiencyReplayValidationError(
                "replay variants did not receive identical evidence"
            )


def _nested_text_words(value: object) -> int:
    if isinstance(value, str):
        return count_generated_words(value)
    if isinstance(value, dict):
        return sum(_nested_text_words(item) for item in value.values())
    if isinstance(value, list):
        return sum(_nested_text_words(item) for item in value)
    return 0


def _clock_value(clock: Callable[[], float]) -> float:
    try:
        value = clock()
    except Exception:
        raise EfficiencyReplayValidationError("efficiency replay clock failed") from None
    if type(value) not in (int, float):
        raise EfficiencyReplayValidationError("efficiency replay clock is invalid")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise EfficiencyReplayValidationError("efficiency replay clock is invalid")
    return resolved


def _elapsed(clock: Callable[[], float], started: float) -> float:
    elapsed = _clock_value(clock) - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise EfficiencyReplayValidationError("efficiency replay duration is invalid")
    return elapsed


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise EfficiencyReplayValidationError("replay evidence is invalid") from None
