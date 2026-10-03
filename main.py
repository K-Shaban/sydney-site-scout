import json
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from analysis import load_data, analyse_locations


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    max_retries=3,
)

ALL_DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

WEEKDAYS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
}

WEEKENDS = {
    "Saturday",
    "Sunday",
}

# ---------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------

class ScoutState(TypedDict):
    question: str
    business_type: str
    start_hour: int
    end_hour: int
    selected_days: list[str]

    business_profile: dict
    location_analysis: list
    data_quality: dict
    opportunity_scores: list
    risk_flags: list
    report: str


# ---------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------

def invoke_llm(prompt, attempts=2):
    """Retry transient failures that can occur during API calls."""
    last_error = None

    for attempt in range(attempts):
        try:
            return llm.invoke(prompt)
        except Exception as error:
            last_error = error

            if attempt < attempts - 1:
                time.sleep(2 ** attempt)

    raise last_error


def response_text(response):
    """Extract text safely from a LangChain model response."""
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    parts.append(str(text))

        return "".join(parts)

    return str(content)


# ---------------------------------------------------------------------
# 1. Understand the proposed business
# ---------------------------------------------------------------------

def understand_business(state: ScoutState):
    """Preserve the business context for later AI interpretation."""

    return {
        "business_profile": {
            "business_type": state["business_type"],
        }
    }


# ---------------------------------------------------------------------
# 2. Analyse pedestrian data
# ---------------------------------------------------------------------

def analyse_data(state: ScoutState):
    """Run deterministic analysis for the selected trading period."""

    df = load_data()

    results = analyse_locations(
        df,
        start_hour=state["start_hour"],
        end_hour=state["end_hour"],
        selected_days=state["selected_days"],
    )

    return {
        "location_analysis": results,
    }


# ---------------------------------------------------------------------
# 3. Assess data quality
# ---------------------------------------------------------------------

def assess_data_quality(state: ScoutState):
    """Check whether enough observations exist for comparison."""

    results = state["location_analysis"]

    if not results:
        return {
            "data_quality": {
                "sufficient": False,
                "reason": "No observations matched the selected period.",
                "locations": 0,
                "observations": 0,
            }
        }

    total_observations = sum(
        row["observations"]
        for row in results
    )

    sufficient = (
        len(results) >= 2
        and total_observations >= 100
    )

    return {
        "data_quality": {
            "sufficient": sufficient,
            "locations": len(results),
            "observations": total_observations,
            "reason": (
                "Sufficient data for comparative analysis."
                if sufficient
                else "Insufficient data for reliable comparison."
            ),
        }
    }


# ---------------------------------------------------------------------
# 4. Score locations
# ---------------------------------------------------------------------

def score_locations(state: ScoutState):
    """
    Calculate a reproducible, business-neutral comparison score.

    Each available pedestrian metric contributes equally. Gemini receives
    the resulting evidence later and interprets it for the proposed business.
    """

    results = state["location_analysis"]
    selected_days = set(state["selected_days"])

    score_fields = [
        "demand_score",
        "consistency_score",
        "peak_score",
        "trend_score",
        "historical_score",
    ]

    if selected_days.intersection(WEEKDAYS):
        score_fields.append("weekday_score")

    if selected_days.intersection(WEEKENDS):
        score_fields.append("weekend_score")

    scored = []

    for row in results:
        score = sum(float(row[field]) for field in score_fields) / len(score_fields)

        updated = dict(row)
        updated["business_fit_score"] = round(score, 1)
        scored.append(updated)

    scored.sort(
        key=lambda row: row["business_fit_score"],
        reverse=True,
    )

    return {
        "opportunity_scores": scored,
    }


# ---------------------------------------------------------------------
# 5. Collect risk flags
# ---------------------------------------------------------------------

def analyse_risks(state: ScoutState):
    """Collect deterministic risk flags from the location analysis."""

    risks = []

    for location in state["opportunity_scores"]:
        for risk in location.get("risk_flags", []):
            risks.append(
                {
                    "location": location["Location_Name"],
                    "risk": risk,
                }
            )

    return {
        "risk_flags": risks,
    }


# ---------------------------------------------------------------------
# Insufficient-data branch
# ---------------------------------------------------------------------

def no_data(state: ScoutState):
    """Return a useful message when there is insufficient data."""

    return {
        "report": (
            "There is not enough pedestrian-count data for the selected "
            "trading period. Try a different time window or selection of "
            "trading days."
        ),
    }


# ---------------------------------------------------------------------
# 6. Generate concise AI decision brief
# ---------------------------------------------------------------------

def generate_report(state: ScoutState):
    """
    Ask Gemini to interpret the deterministic evidence.

    Gemini interprets the calculated evidence for the proposed business.
    """

    results = state["opportunity_scores"]
    profile = state["business_profile"]
    quality = state["data_quality"]

    selected_days = ", ".join(state["selected_days"])

    summary = []

    for row in results:
        summary.append(
            {
                "location": row["Location_Name"],
                "business_fit_score": row["business_fit_score"],
                "average_pedestrians": row["avg_pedestrians"],
                "p95_pedestrians": row["p95_pedestrians"],
                "peak_intensity": row["peak_intensity"],
                "consistency": row["traffic_consistency"],
                "weekday_avg": row["weekday_avg"],
                "weekend_avg": row["weekend_avg"],
                "recent_change_pct": row["recent_change_pct"],
                "yoy_change_pct": row["yoy_change_pct"],
                "historical_index": row["historical_index"],
                "risks": row["risk_flags"],
            }
        )

    prompt = f"""
You are a business location analyst.

Business:
{state["business_type"]}

Trading period:
{state["start_hour"]}:00-{state["end_hour"]}:00

Selected trading days:
{selected_days}

Business context:
{json.dumps(profile, indent=2)}

Data quality:
{json.dumps(quality, indent=2)}

Deterministic location analysis:
{json.dumps(summary, indent=2)}

Write ONE concise professional paragraph of approximately 70-100 words.

Explain:
- whether the highest-scoring location is a sensible leading option for this business,
- the most important evidence supporting the result,
- one important limitation,
- and what should be investigated before making a site decision.

Rules:
- Write exactly one paragraph.
- Do not use headings.
- Do not use bullet points.
- Do not return JSON.
- Do not repeat every metric.
- Focus on the most decision-relevant evidence.
- Pedestrian counts are observed evidence, not a forecast of business success.
- The business-fit score is a reproducible, business-neutral comparison index, not a probability.
- Use the business type to interpret the evidence, but do not invent new numerical scores.
- Do not invent rent, demographics, competition, zoning, customer intent or revenue.
- Do not assume pedestrians are customers or commuters.
- Use P95 pedestrian count rather than maximum traffic when discussing peak activity.
- Mention that only four locations are represented in the dataset.
- Return only the paragraph.
"""

    response = invoke_llm(prompt)

    return {
        "report": response_text(response).strip(),
    }


# ---------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------

def route_after_quality(state: ScoutState):
    if state["data_quality"]["sufficient"]:
        return "score_locations"

    return "no_data"


# ---------------------------------------------------------------------
# Build LangGraph workflow
# ---------------------------------------------------------------------

graph = StateGraph(ScoutState)

graph.add_node(
    "understand_business",
    understand_business,
)

graph.add_node(
    "analyse_data",
    analyse_data,
)

graph.add_node(
    "assess_data_quality",
    assess_data_quality,
)

graph.add_node(
    "score_locations",
    score_locations,
)

graph.add_node(
    "analyse_risks",
    analyse_risks,
)

graph.add_node(
    "generate_report",
    generate_report,
)

graph.add_node(
    "no_data",
    no_data,
)


graph.add_edge(
    START,
    "understand_business",
)

graph.add_edge(
    "understand_business",
    "analyse_data",
)

graph.add_edge(
    "analyse_data",
    "assess_data_quality",
)

graph.add_conditional_edges(
    "assess_data_quality",
    route_after_quality,
    {
        "score_locations": "score_locations",
        "no_data": "no_data",
    },
)

graph.add_edge(
    "score_locations",
    "analyse_risks",
)

graph.add_edge(
    "analyse_risks",
    "generate_report",
)

graph.add_edge(
    "generate_report",
    END,
)

graph.add_edge(
    "no_data",
    END,
)


app = graph.compile()


# ---------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------

def run_scout(
    business_type,
    start_hour=7,
    end_hour=10,
    selected_days=None,
):
    if selected_days is None:
        selected_days = ALL_DAYS.copy()

    return app.invoke(
        {
            "question": business_type,
            "business_type": business_type,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "selected_days": selected_days,
            "business_profile": {},
            "location_analysis": [],
            "data_quality": {},
            "opportunity_scores": [],
            "risk_flags": [],
            "report": "",
        }
    )


# ---------------------------------------------------------------------
# CLI testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    business = input(
        "What business are you considering opening in Sydney?\n> "
    )

    result = run_scout(business)

    print("\n" + "=" * 70)
    print("LOCATION ANALYSIS")
    print("=" * 70)

    for row in result["opportunity_scores"]:
        print(
            f"\n{row['Location_Name']}"
            f"\nBusiness fit: {row['business_fit_score']:.1f}"
            f"\nAverage traffic: {row['avg_pedestrians']:,.0f}/hour"
            f"\nP95 traffic: {row['p95_pedestrians']:,.0f}/hour"
            f"\nRecent trend: {row['recent_change_pct']:.1f}%"
        )

    print("\n" + "=" * 70)
    print("AI BUSINESS BRIEF")
    print("=" * 70)
    print(result["report"])