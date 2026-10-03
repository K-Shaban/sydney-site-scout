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
    """
    Use Gemini to translate the business concept into analytical priorities.

    Gemini provides business-specific importance weights.
    It does not directly score or rank locations.
    """

    selected_days = ", ".join(state["selected_days"])

    prompt = f"""
You are defining an analytical business profile for a Sydney location study.

Business concept:
{state["business_type"]}

Trading period:
{state["start_hour"]}:00-{state["end_hour"]}:00

Selected trading days:
{selected_days}

Return ONLY valid JSON using this structure:

{{
  "business_type": "...",
  "primary_customer_period": "...",
  "weekday_importance": 0.0,
  "weekend_importance": 0.0,
  "demand_importance": 0.0,
  "consistency_importance": 0.0,
  "peak_importance": 0.0,
  "trend_importance": 0.0,
  "historical_importance": 0.0,
  "risk_tolerance": "low|medium|high"
}}

Rules:
- All importance values must be between 0 and 1.
- Importance values do not need to sum to 1.
- The application will normalise the values before scoring.
- Consider the selected trading days when assigning weekday and weekend importance.
- Use reasonable generic commercial assumptions.
- Do not make claims about particular Sydney locations.
- Return JSON only.
"""

    response = invoke_llm(prompt)

    content = (
        response_text(response)
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        profile = json.loads(content)

    except (json.JSONDecodeError, TypeError):
        # Deterministic fallback if Gemini does not return valid JSON.
        profile = {
            "business_type": state["business_type"],
            "primary_customer_period": "General",
            "weekday_importance": 0.5,
            "weekend_importance": 0.5,
            "demand_importance": 0.30,
            "consistency_importance": 0.20,
            "peak_importance": 0.15,
            "trend_importance": 0.15,
            "historical_importance": 0.10,
            "risk_tolerance": "medium",
        }

    return {
        "business_profile": profile,
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
    Combine deterministic location metrics with business-specific priorities.

    Gemini supplies the importance weights.
    Python performs the actual scoring and ranking.
    """

    results = state["location_analysis"]
    profile = state["business_profile"]

    raw_weights = {
        "demand": profile.get("demand_importance", 0.30),
        "consistency": profile.get("consistency_importance", 0.20),
        "peak": profile.get("peak_importance", 0.15),
        "trend": profile.get("trend_importance", 0.15),
        "historical": profile.get("historical_importance", 0.10),
        "weekday": profile.get("weekday_importance", 0.05),
        "weekend": profile.get("weekend_importance", 0.05),
    }

    selected_days = set(state["selected_days"])

    # Do not let an unselected day category affect the score.
    if not selected_days.intersection(WEEKDAYS):
        raw_weights["weekday"] = 0

    if not selected_days.intersection(WEEKENDS):
        raw_weights["weekend"] = 0

    clean_weights = {}

    for key, value in raw_weights.items():
        try:
            clean_weights[key] = max(float(value), 0)
        except (TypeError, ValueError):
            clean_weights[key] = 0

    total_weight = sum(clean_weights.values())

    if total_weight == 0:
        weights = {
            key: 1 / len(clean_weights)
            for key in clean_weights
        }
    else:
        weights = {
            key: value / total_weight
            for key, value in clean_weights.items()
        }

    scored = []

    for row in results:
        score = (
            row["demand_score"] * weights["demand"]
            + row["consistency_score"] * weights["consistency"]
            + row["peak_score"] * weights["peak"]
            + row["trend_score"] * weights["trend"]
            + row["historical_score"] * weights["historical"]
            + row["weekday_score"] * weights["weekday"]
            + row["weekend_score"] * weights["weekend"]
        )

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

    Gemini explains the result but does not calculate the ranking.
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

Business profile:
{json.dumps(profile, indent=2)}

Data quality:
{json.dumps(quality, indent=2)}

Deterministic location analysis:
{json.dumps(summary, indent=2)}

Write ONE concise professional paragraph of approximately 70-100 words.

Explain:
- which location is the leading option,
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
- The business-fit score is a relative decision-support index, not a probability.
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