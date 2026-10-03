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
    
    .stApp {
        background: #f7f9fc;
        color: #0f172a;
    }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }

    .hero {
        padding: 2.1rem 2.2rem;
        margin-bottom: 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #172554 58%, #1e3a8a 100%);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.13);
    }
    .eyebrow {
        color: #93c5fd; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .55rem;
    }
    .hero h1 { color: white; font-size: 2.65rem; letter-spacing: -.04em; margin: 0 0 .45rem 0; }
    .hero p { color: #cbd5e1; font-size: 1.05rem; line-height: 1.65; max-width: 760px; margin: 0; }
    .hero-badges { margin-top: 1.1rem; }
    .badge {
        display: inline-block; padding: .32rem .65rem; margin-right: .35rem;
        border: 1px solid rgba(255,255,255,.16); border-radius: 999px;
        color: #dbeafe; background: rgba(255,255,255,.07); font-size: .76rem;
    }

    div[data-testid="stForm"] {
        background: white; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 1.35rem 1.45rem .55rem; box-shadow: 0 8px 30px rgba(15,23,42,.05);
    }
    div[data-testid="stForm"] label { font-weight: 600; color: #334155; }
    div[data-testid="stTextInput"] input { border-radius: 10px; }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 3rem; border-radius: 10px; font-weight: 700;
    }

    .result-card {
        border: 1px solid #bfdbfe; border-radius: 16px; padding: 1.2rem 1.3rem;
        background: linear-gradient(145deg, #eff6ff 0%, #ffffff 78%);
        min-height: 148px; box-shadow: 0 8px 24px rgba(30,64,175,.06);
    }
    .score { font-size: 2.15rem; font-weight: 800; line-height: 1.05; color: #0f172a; margin: .25rem 0; }
    .muted { color: #64748b; font-size: .78rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
    .location-name { color: #1e3a8a; font-size: 1.05rem; font-weight: 700; margin-top: .35rem; }

    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; padding: .9rem 1rem;
        border-radius: 14px; min-height: 148px; box-shadow: 0 8px 24px rgba(15,23,42,.04);
    }
    div[data-testid="stMetricLabel"] { color: #64748b; }
    div[data-testid="stMetricValue"] { color: #0f172a; }
    div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; }

    .empty-state {
        background: white; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 1.5rem 1.7rem; margin-top: .3rem;
    }
    .empty-state h3 { margin-top: 0; color: #0f172a; }
    .empty-state p { color: #64748b; margin-bottom: 0; }

    h2, h3 { color: #0f172a; letter-spacing: -.02em; }
    hr { border-color: #e2e8f0 !important; }
    #MainMenu, footer { visibility: hidden; }

    /* Standard Streamlit text */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #334155;
    }

    /* Headings */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #0f172a;
    }

    /* Captions */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #64748b !important;
    }

    /* AI decision brief */
    .ai-brief {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    .ai-brief,
    .ai-brief p {
        color: #334155 !important;
    }

    .ai-brief p {
        margin: 0;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Location Intelligence · Sydney</div>
        <h1>Sydney Site Scout</h1>
        <p>
            Compare candidate locations using real pedestrian activity,
            business-specific scoring and an AI-generated decision brief.
        </p>
        <div class="hero-badges">
            <span class="badge">City of Sydney data</span>
            <span class="badge">Deterministic scoring</span>
            <span class="badge">Gemini interpretation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("### Build a site analysis")
st.caption("Describe the business, then set the trading window you want to evaluate.")

with st.form("scout_form"):
    business = st.text_input(
        "What business are you considering?",
        placeholder="e.g. museum, café, boutique, gallery",
    )

    def format_hour(hour: int) -> str:
        if hour == 0:
            return "12:00 AM"
        if hour == 12:
            return "12:00 PM"
        if hour == 24:
            return "12:00 AM"
        if hour < 12:
            return f"{hour}:00 AM"
        return f"{hour - 12}:00 PM"


    col1, col2, col3 = st.columns(3)

    with col1:
        start_hour = st.selectbox(
            "Opening time",
            options=list(range(24)),
            index=7,
            format_func=format_hour,
        )

    with col2:
        end_hour = st.selectbox(
            "Closing time",
            options=list(range(1, 25)),
            index=9,
            format_func=format_hour,
        )

    st.markdown("**Trading days**")

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    day_cols = st.columns(7)

    selected_days = []

    for col, day in zip(day_cols, days):
        with col:
            if st.checkbox(
                day[:3],
                value=True,
                key=f"day_{day}",
            ):
                selected_days.append(day)

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

    if not selected_days:
        st.warning("Select at least one trading day.")
        st.stop()

    with st.spinner("Analysing Sydney pedestrian data..."):
        try:
            result = run_scout(
                business.strip(),
                start_hour,
                end_hour,
                selected_days,
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

    st.write(
        f"Based on pedestrian activity during the selected trading period, "
        f"**{top['Location_Name']}** has the strongest overall business fit for "
        f"the proposed **{business.strip()}**. The score combines observed pedestrian "
        f"demand, traffic consistency, peak activity and historical trends using "
        f"business-specific priorities."
    )

    st.caption(
        f"Analysis for {business.strip()} · "
        f"{start_hour:02d}:00–{end_hour:02d}:00 · {selected_days}"
    )

    top_col, metric1, metric2, metric3 = st.columns([1.2, 1, 1, 1])

    with top_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="muted">Recommended location</div>
                <div class="score">{top['business_fit_score']:.1f}<span style="font-size:.9rem;color:#64748b;font-weight:600"> / 100</span></div>
                <div class="location-name">{html.escape(top['Location_Name'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric1:
        st.metric(
            "Average pedestrians",
            f"{top['avg_pedestrians']:,.0f}/hr",
        )

    with metric2:
        st.metric(
            "P95 pedestrian count",
            f"{top['p95_pedestrians']:,.0f}/hr",
        )

    with metric3:
        st.metric(
            "Recent change",
            f"{top['recent_change_pct']:+.1f}%",
        )

    map_url = (
        "https://www.google.com/maps/search/?api=1&query="
        + quote_plus(f"{top['Location_Name']}, Sydney NSW")
    )

    st.link_button(
        f"Open {top['Location_Name']} in Google Maps",
        map_url,
        use_container_width=False,
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
            "Business fit": st.column_config.NumberColumn(
                format="%.1f",
            ),
            "Avg / hr": st.column_config.NumberColumn(
                format="%.0f",
            ),
            "P95 / hr": st.column_config.NumberColumn(
                format="%.0f",
            ),
            "P95 / avg": st.column_config.NumberColumn(
                format="%.2fx",
            ),
            "Consistency": st.column_config.NumberColumn(
                format="%.2f",
            ),
            "Weekday avg": st.column_config.NumberColumn(
                format="%.0f",
            ),
            "Weekend avg": st.column_config.NumberColumn(
                format="%.0f",
            ),
            "Recent change": st.column_config.NumberColumn(
                format="%+.1f%%",
            ),
            "YoY change": st.column_config.NumberColumn(
                format="%+.1f%%",
            ),
        },
    )

    st.subheader("AI decision brief")

    brief = result.get("report", "").strip()

    if brief:
        st.markdown(
            f"""
            <div class="ai-brief">
                <p>{html.escape(brief)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No AI decision brief was generated.")


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
            """
        )

    st.caption(
        "This is decision-support, not a commercial forecast. "
        "Pedestrian volume does not establish customer intent, revenue, "
        "rent affordability or business viability."
    )


else:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Turn a business idea into location evidence</h3>
            <p>
                Enter a business and trading period above. Site Scout will compare
                the available locations, rank their business fit and generate a
                concise decision brief grounded in the calculated pedestrian data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**01 · Measure**")
        st.caption("Filter real pedestrian observations to your trading window.")
    with col_b:
        st.markdown("**02 · Compare**")
        st.caption("Score four Sydney locations against business-specific priorities.")
    with col_c:
        st.markdown("**03 · Explain**")
        st.caption("Generate a concise Gemini decision brief from the calculated evidence.")
