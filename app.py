import html
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from main import run_scout

st.set_page_config(
    page_title="Sydney Site Scout",
    page_icon="📍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .hero { padding: 1.5rem 0 1rem 0; }
    .hero h1 { font-size: 2.7rem; margin-bottom: 0.2rem; }
    .hero p { font-size: 1.1rem; color: #666; max-width: 760px; }
    .result-card {
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        background: white;
        margin-bottom: 0.8rem;
    }
    .score { font-size: 2rem; font-weight: 700; line-height: 1; }
    .muted { color: #6b7280; font-size: 0.9rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e6e6e6;
        padding: 0.8rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📍 Sydney Site Scout</h1>
        <p>
            AI-assisted location intelligence using real Sydney pedestrian
            count data. Explore how different trading periods and business
            requirements change the evidence for each location.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("scout_form"):
    business = st.text_input(
        "What business are you considering?",
        placeholder="e.g. museum, café, boutique, gallery",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        start_hour = st.slider(
            "Opening hour", min_value=0, max_value=23, value=7
        )

    with col2:
        end_hour = st.slider(
            "Closing hour", min_value=1, max_value=24, value=10
        )

    with col3:
        day_filter = st.selectbox(
            "Day filter",
            [
                "All", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday",
            ],
        )

    submitted = st.form_submit_button(
        "Analyse locations",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not business.strip():
        st.warning("Enter a business type to begin.")
        st.stop()

    if end_hour <= start_hour:
        st.warning("Closing hour must be later than opening hour.")
        st.stop()

    with st.spinner("Analysing Sydney pedestrian data..."):
        try:
            result = run_scout(
                business.strip(),
                start_hour,
                end_hour,
                day_filter,
            )
        except Exception as exc:
            st.error(
                "The analysis could not be completed. "
                "Check your Gemini API configuration and try again."
            )
            st.exception(exc)
            st.stop()

    locations = result["opportunity_scores"]

    if not locations:
        st.info(result["report"])
        st.stop()

    st.divider()

    top = locations[0]

    st.subheader("Location overview")
    st.caption(
        f"Analysis for {business.strip()} · "
        f"{start_hour:02d}:00–{end_hour:02d}:00 · {day_filter}"
    )

    top_col, metric1, metric2, metric3 = st.columns([1.2, 1, 1, 1])

    with top_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="muted">Highest comparative fit</div>
                <div class="score">{top['business_fit_score']:.1f}</div>
                <h3>{html.escape(top['Location_Name'])}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric1:
        st.metric("Average pedestrians", f"{top['avg_pedestrians']:,.0f}/hr")

    with metric2:
        st.metric("P95 pedestrian count", f"{top['p95_pedestrians']:,.0f}/hr")

    with metric3:
        st.metric("Recent change", f"{top['recent_change_pct']:+.1f}%")

    map_url = (
        "https://www.google.com/maps/search/?api=1&query="
        + quote_plus(f"{top['Location_Name']}, Sydney NSW")
    )

    st.link_button(
        f"Open {top['Location_Name']} in Google Maps",
        map_url,
    )

    st.caption(
        "Google Maps is provided as a navigation aid. "
        "The analysis does not use Google Maps data."
    )

    st.subheader("Compare locations")

    table = pd.DataFrame(locations)

    display_columns = {
        "Location_Name": "Location",
        "business_fit_score": "Business fit",
        "avg_pedestrians": "Avg / hr",
        "p95_pedestrians": "P95 / hr",
        "peak_intensity": "P95 / avg",
        "traffic_consistency": "Consistency",
        "weekday_avg": "Weekday avg",
        "weekend_avg": "Weekend avg",
        "recent_change_pct": "Recent change",
        "yoy_change_pct": "YoY change",
    }

    display = table[list(display_columns)].rename(columns=display_columns)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Business fit": st.column_config.NumberColumn(format="%.1f"),
            "Avg / hr": st.column_config.NumberColumn(format="%.0f"),
            "P95 / hr": st.column_config.NumberColumn(format="%.0f"),
            "P95 / avg": st.column_config.NumberColumn(format="%.2fx"),
            "Consistency": st.column_config.NumberColumn(format="%.2f"),
            "Weekday avg": st.column_config.NumberColumn(format="%.0f"),
            "Weekend avg": st.column_config.NumberColumn(format="%.0f"),
            "Recent change": st.column_config.NumberColumn(format="%+.1f%%"),
            "YoY change": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

    st.subheader("AI decision brief")
    st.markdown(result["report"])

    with st.expander("How the analysis works"):
        st.markdown(
            """
            **1. Data preparation**  
            The raw pedestrian dataset is parsed, numeric fields are
            converted, invalid required observations are removed, and the
            selected trading window is isolated.

            **2. Deterministic analysis**  
            Pandas calculates demand, P95 peak traffic, volatility,
            consistency, weekday/weekend behaviour, recent change,
            year-on-year change and historical performance.

            **3. Business profile**  
            Gemini interprets the business concept and produces explicit
            business-importance weights.

            **4. Business-fit scoring**  
            The application normalises those weights and combines them with
            deterministic 0–100 relative metrics. The score is a transparent
            decision-support index, not a prediction.

            **5. AI interpretation**  
            Gemini receives the calculated evidence and writes the final
            business brief. It is instructed not to invent information that
            is absent from the dataset.

            **6. Workflow orchestration**  
            LangGraph carries shared state through the analysis, quality,
            scoring, risk-analysis and reporting stages.
            """
        )

    st.caption(
        "This is decision-support, not a commercial forecast. "
        "Pedestrian volume does not establish customer intent, revenue, "
        "rent affordability or business viability."
    )

else:
    st.info(
        "Enter a business and trading period above to analyse the available "
        "Sydney pedestrian-count locations."
    )

    st.markdown(
        """
        ### What you get

        - Comparative pedestrian-demand analysis
        - Business-specific weighting
        - Weekday and weekend evidence
        - Trend and historical comparisons
        - Risk flags
        - An AI-generated decision brief
        - A Google Maps link for the leading location

        The underlying calculations remain deterministic and reproducible;
        the LLM is used for business interpretation rather than inventing
        the analytical results.
        """
    )
