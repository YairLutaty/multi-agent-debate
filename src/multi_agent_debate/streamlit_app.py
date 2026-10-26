"""Local Streamlit dashboard for grounded debate and advisory judgment."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from decimal import Decimal, InvalidOperation

from multi_agent_debate.costs import CostValidationError, Pricing
from multi_agent_debate.judge import RUBRIC_CRITERIA
from multi_agent_debate.ollama_model import DEFAULT_MODEL_NAME
from multi_agent_debate.ui_result import (
    debate_overview,
    debate_round_rows,
    display_metric_rows,
    evidence_rows,
    judgment_presentation,
    judgment_usage_and_cost,
    metrics_and_cost,
    ui_result_to_json,
    validate_ui_result,
)
from multi_agent_debate.ui_service import (
    DEFAULT_UI_MODE,
    DEFAULT_UI_ROUNDS,
    UiConfigurationError,
    UiDependencyError,
    UiExecutionError,
    resolve_tavily_api_key,
    run_ui_debate,
)

_RESULT_KEY = "latest_validated_ui_result"


def clear_latest_result(session_state: MutableMapping[str, object]) -> None:
    """Remove only the nonsecret validated result retained across reruns."""

    session_state.pop(_RESULT_KEY, None)


def should_start_debate(submitted: object) -> bool:
    """A Streamlit rerun starts work only for an explicit form submission."""

    return submitted is True


def resolve_streamlit_tavily_key(
    *, streamlit_module: object, environ: Mapping[str, str]
) -> str:
    """Resolve Tavily configuration even when Streamlit secrets are unavailable."""

    try:
        secrets = streamlit_module.secrets
    except Exception:
        secrets = {}
    return resolve_tavily_api_key(
        streamlit_secrets=secrets,
        environ=environ,
    )


def main() -> None:
    """Render the dashboard; Streamlit is imported only at the UI boundary."""

    import streamlit as st

    st.set_page_config(
        page_title="Grounded Multi-Agent Debate",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("Grounded Multi-Agent Debate")
    st.caption(
        "Research a contested question, run a bounded local-model debate, and "
        "receive an advisory structured judgment."
    )

    try:
        tavily_key = resolve_streamlit_tavily_key(
            streamlit_module=st,
            environ=os.environ,
        )
        tavily_ready = True
    except UiConfigurationError:
        tavily_key = None
        tavily_ready = False

    with st.sidebar:
        st.header("Debate configuration")
        if tavily_ready:
            st.success("Tavily is configured.", icon="✅")
        else:
            st.warning(
                "Tavily is not configured. Add TAVILY_API_KEY to Streamlit "
                "secrets or the process environment before starting.",
                icon="⚠️",
            )
        mode = st.selectbox(
            "Mode",
            options=("multi-round", "single-round", "hierarchical"),
            index=("multi-round", "single-round", "hierarchical").index(
                DEFAULT_UI_MODE
            ),
            format_func=lambda value: {
                "multi-round": "Multi-round one-shot",
                "single-round": "Single-round one-shot",
                "hierarchical": "Hierarchical specialist teams",
            }[value],
        )
        if mode == "multi-round":
            max_rounds = st.select_slider(
                "Maximum rounds", options=(1, 2, 3), value=DEFAULT_UI_ROUNDS
            )
        else:
            max_rounds = 1
            st.caption("This mode uses exactly one round.")
        if mode == "hierarchical":
            st.info(
                "Hierarchical mode supports new grounded questions as a "
                "single-round specialist-team experiment."
            )

        with st.form("debate_form", clear_on_submit=False):
            question = st.text_area(
                "Debate question",
                height=120,
                placeholder="Enter a genuinely contested question…",
            )
            model_name = st.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
            novelty_threshold = st.number_input(
                "Low-novelty threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                disabled=mode != "multi-round",
            )
            input_rate = st.number_input(
                "Input cost per million tokens",
                min_value=0.0,
                value=0.0,
                step=0.01,
            )
            output_rate = st.number_input(
                "Output cost per million tokens",
                min_value=0.0,
                value=0.0,
                step=0.01,
            )
            implementation_commit = st.text_input(
                "Implementation commit",
                value=os.environ.get("DEBATE_IMPLEMENTATION_COMMIT", ""),
                help="Use the 40-character SHA from: git rev-parse HEAD",
            )
            submitted = st.form_submit_button(
                "Start debate", type="primary", width="stretch"
            )

        if st.button("Clear result", width="stretch"):
            clear_latest_result(st.session_state)

    if should_start_debate(submitted):
        if not tavily_ready or tavily_key is None:
            st.error("Tavily is not configured.")
        else:
            try:
                pricing = Pricing(Decimal(str(input_rate)), Decimal(str(output_rate)))
            except (InvalidOperation, CostValidationError, ValueError):
                st.error("Pricing settings are invalid.")
            else:
                try:
                    with st.status("Starting grounded debate…", expanded=True) as status:
                        def on_progress(event: dict[str, str]) -> None:
                            status.write(event["label"])

                        output = run_ui_debate(
                            question=question,
                            mode=mode,
                            max_rounds=max_rounds,
                            novelty_threshold=(
                                float(novelty_threshold)
                                if mode == "multi-round"
                                else 0.15
                            ),
                            model_name=model_name,
                            implementation_commit=implementation_commit,
                            pricing=pricing,
                            tavily_api_key=tavily_key,
                            progress_callback=on_progress,
                        )
                        st.session_state[_RESULT_KEY] = output.result
                        judgment_available = (
                            output.result["judgment_status"]["status"] == "available"
                        )
                        status.update(
                            label=(
                                "Debate and advisory judgment complete"
                                if judgment_available
                                else "Debate complete; advisory judgment unavailable"
                            ),
                            state="complete",
                            expanded=False,
                        )
                except UiConfigurationError as error:
                    st.error(str(error))
                except UiDependencyError:
                    st.error(
                        "Ollama is unavailable or the selected model is not installed."
                    )
                except UiExecutionError:
                    st.error("The debate could not be completed.")
                except Exception:
                    st.error("The debate could not be completed.")

    latest = st.session_state.get(_RESULT_KEY)
    if latest is not None:
        try:
            result = validate_ui_result(latest)
        except Exception:
            clear_latest_result(st.session_state)
            st.error("The saved result is invalid and was cleared.")
        else:
            _render_result(st, result)


def _render_result(st: object, result: dict) -> None:
    overview_tab, rounds_tab, judge_tab, evidence_tab, metrics_tab, raw_tab = st.tabs(
        (
            "Overview",
            "Debate rounds",
            "Judge",
            "Evidence",
            "Metrics and cost",
            "Raw result",
        )
    )
    debate = result["debate_result"]
    judgment = result["judgment"]
    judgment_panel = judgment_presentation(result)

    with overview_tab:
        overview = debate_overview(result)
        if judgment is None:
            st.subheader("Debate complete")
            st.info(judgment_panel["notice"])
        else:
            st.subheader("LLM-generated judgment")
            st.caption("Advisory model assessment; it is not an objective truth.")
            columns = st.columns(4)
            columns[0].metric("Outcome", _winner_label(overview["winner"]))
            columns[1].metric("Confidence", f"{overview['confidence']} / 100")
            columns[2].metric("Team A", f"{overview['team_a_overall']} / 100")
            columns[3].metric("Team B", f"{overview['team_b_overall']} / 100")
        columns = st.columns(4)
        columns[0].metric("Rounds", overview["completed_rounds"])
        columns[1].metric("Termination", overview["termination_reason"])
        columns[2].metric("Sources", overview["source_count"])
        columns[3].metric("Total tokens", overview["total_tokens"])
        columns = st.columns(2)
        columns[0].metric("Estimated API cost", overview["estimated_cost"])
        columns[1].metric(
            "Total latency", f"{overview['total_latency_seconds']:.3f} s"
        )
        st.markdown("#### Neutral synthesis")
        st.write(debate["final_outputs"]["neutral_synthesis"])

    with rounds_tab:
        for item in debate_round_rows(debate):
            heading = f"Round {item['round_number']}"
            if item["is_final_round"]:
                heading += " · final"
            st.subheader(heading)
            left, right = st.columns(2)
            with left:
                st.markdown("**Team A — supports the proposition**")
                st.caption(
                    f"Opponent input: {item['team_a_opponent_argument'] or 'n/a'}"
                )
                st.caption(f"Same-team novelty: {item['team_a_novelty_display']}")
                with st.expander("Team A argument", expanded=True):
                    st.write(item["team_a_argument"])
            with right:
                st.markdown("**Team B — opposes the proposition**")
                st.caption("Opponent input: Team A's argument from this round")
                st.caption(f"Same-team novelty: {item['team_b_novelty_display']}")
                with st.expander("Team B argument", expanded=True):
                    st.write(item["team_b_argument"])

    with judge_tab:
        st.subheader("LLM-generated judgment")
        if judgment is None:
            st.info(judgment_panel["notice"])
        else:
            if judgment["stance_drift"]["team_a"] or judgment["stance_drift"]["team_b"]:
                teams = [
                    label
                    for key, label in (("team_a", "Team A"), ("team_b", "Team B"))
                    if judgment["stance_drift"][key]
                ]
                st.warning("Possible stance drift detected: " + ", ".join(teams))
            score_rows = [
                {
                    "Criterion": criterion.replace("_", " ").title(),
                    "Team A": judgment["scores"]["team_a"][criterion],
                    "Team B": judgment["scores"]["team_b"][criterion],
                }
                for criterion in (*RUBRIC_CRITERIA, "overall_score")
            ]
            st.dataframe(score_rows, hide_index=True, width="stretch")
            st.write(judgment["explanation"])
            _render_points(st, "Decisive points", judgment["decisive_points"])
            left, right = st.columns(2)
            with left:
                _render_points(st, "Team A strengths", judgment["strengths"]["team_a"])
                _render_points(st, "Team A weaknesses", judgment["weaknesses"]["team_a"])
            with right:
                _render_points(st, "Team B strengths", judgment["strengths"]["team_b"])
                _render_points(st, "Team B weaknesses", judgment["weaknesses"]["team_b"])
            _render_points(st, "Limitations", judgment["limitations"])

    with evidence_tab:
        rows = evidence_rows(debate)
        st.caption(f"{len(rows)} consolidated sources")
        for item in rows:
            with st.expander(
                f"{item['source_title']} · {item['domain']}", expanded=False
            ):
                st.write(item["claim"])
                st.caption(f"Research question: {item['subquestion']}")
                if item["source_url"] is not None:
                    st.link_button("Open source", item["source_url"])
                else:
                    st.caption(item["source_link_label"])

    with metrics_tab:
        st.markdown("#### Debate engine")
        values = metrics_and_cost(debate)
        st.dataframe(
            display_metric_rows(values),
            hide_index=True,
            width="stretch",
        )
        st.markdown("#### Advisory judge")
        if judgment is None:
            st.info(judgment_panel["notice"])
        else:
            judge_values = judgment_usage_and_cost(result)
            st.dataframe(
                display_metric_rows(judge_values),
                hide_index=True,
                width="stretch",
            )
        st.info(
            "Estimated API-style cost uses the configured rates. Local zero "
            "pricing excludes hardware, electricity, cooling, maintenance, "
            "and personnel."
        )

    with raw_tab:
        serialized = ui_result_to_json(result)
        with st.expander("Formatted JSON", expanded=False):
            st.json(result, expanded=False)
        safe_timestamp = result["display_metadata"]["created_utc"].replace(":", "-")
        st.download_button(
            "Download complete JSON",
            data=serialized,
            file_name=f"debate-result-{safe_timestamp}.json",
            mime="application/json",
            width="stretch",
        )


def _render_points(st: object, heading: str, points: list[str]) -> None:
    st.markdown(f"**{heading}**")
    for point in points:
        st.markdown(f"- {point}")


def _winner_label(value: object) -> str:
    return {"team_a": "Team A", "team_b": "Team B", "draw": "Draw"}.get(
        value, "Unavailable"
    )


if __name__ == "__main__":
    main()
