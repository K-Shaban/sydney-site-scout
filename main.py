import json
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from analysis import load_data, analyse_locations


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    max_retries=3,
)


class ScoutState(TypedDict):
    question: str
    business_type: str
    start_hour: int
    end_hour: int
    day_filter: str

    business_profile: dict
    location_analysis: list
    data_quality: dict
    opportunity_scores: list
    risk_flags: list
    report: str


def invoke_llm(prompt, attempts=2):
    """Retry only the transient failures that can occur during API calls."""
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
    content = response.content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return str(content)


def understand_business(state: ScoutState):
    prompt = f"""
You are defining an analytical business profile for a Sydney location study.

Business concept: {state["business_type"]}
Trading period: {state["start_hour"]}:00-{state["end_hour"]}:00

Return ONLY valid JSON:

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

Values must be between 0 and 1.
The importance values do not need to sum to 1 because the application
will normalise them before scoring.

Use reasonable generic commercial assumptions.
Do not make claims about particular Sydney locations.
"""

    response = invoke_llm(prompt)
    content = response_text(response).replace("```json", "").replace("```", "").strip()

    try:
        profile = json.loads(content)
    except json.JSONDecodeError:
        profile = {
            "business_type": state["business_type"],
            "primary_customer_period": "General",
            "weekday_importance": 0.5,
            "weekend_importance": 0.5,
            "demand_importance": 0.3,
            "consistency_importance": 0.2,
            "peak_importance": 0.15,
            "trend_importance": 0.15,
            "historical_importance": 0.2,
            "risk_tolerance": "medium",
        }

    return {"business_profile": profile}


def analyse_data(state: ScoutState):
    df = load_data()

    results = analyse_locations(
        df,
        start_hour=state["start_hour"],
        end_hour=state["end_hour"],
        day=state["day_filter"],
    )

    return {"location_analysis": results}


def assess_data_quality(state: ScoutState):
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

    total_observations = sum(row["observations"] for row in results)

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


def score_locations(state: ScoutState):
    results = state["location_analysis"]
    profile = state["business_profile"]

    # These are business preferences, not probabilities.
    raw_weights = {
        "demand": profile.get("demand_importance", 0.30),
        "consistency": profile.get("consistency_importance", 0.20),
        "peak": profile.get("peak_importance", 0.15),
        "trend": profile.get("trend_importance", 0.15),
        "historical": profile.get("historical_importance", 0.10),
        "weekday": profile.get("weekday_importance", 0.05),
        "weekend": profile.get("weekend_importance", 0.05),
    }

    total_weight = sum(max(float(v), 0) for v in raw_weights.values())

    if total_weight == 0:
        weights = {key: 1 / len(raw_weights) for key in raw_weights}
    else:
        weights = {
            key: max(float(value), 0) / total_weight
            for key, value in raw_weights.items()
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


def analyse_risks(state: ScoutState):
    risks = []

    for location in state["opportunity_scores"]:
        for risk in location.get("risk_flags", []):
            risks.append(
                {
                    "location": location["Location_Name"],
                    "risk": risk,
                }
            )

    return {"risk_flags": risks}


def no_data(state: ScoutState):
    return {
        "report": (
            "There is not enough pedestrian-count data for the selected "
            "trading period. Try a different time window or day filter."
        )
    }


def generate_report(state: ScoutState):
    results = state["opportunity_scores"]
    profile = state["business_profile"]
    quality = state["data_quality"]

    summary = []

    for row in results:
        summary.append(
            {
                "location": row["Location_Name"],
                "business_fit_score": row["business_fit_score"],
                "average_pedestrians": row["avg_pedestrians"],
                "p95_pedestrians": row["p95_pedestrians"],
                "max_pedestrians": row["max_pedestrians"],
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

Business profile:
{json.dumps(profile, indent=2)}

Data quality:
{json.dumps(quality, indent=2)}

Deterministic location analysis:
{json.dumps(summary, indent=2)}

Write a concise professional decision brief.

Rules:
- The pedestrian data is observed evidence, not a forecast of business success.
- The business-fit score is a transparent relative decision-support index.
- Do not invent rent, demographics, competition, zoning, customer intent,
  or other facts not present in the supplied data.
- Do not assume morning pedestrians are commuters. The dataset does not
  identify pedestrian purpose.
- Explain the leading site's score using the supplied metrics.
- Discuss meaningful differences and risks.
- Explicitly note that the dataset contains only four locations.
- Use "P95 pedestrian count" as the robust peak measure. The maximum is
  an observed extreme and should not be treated as a typical peak.
- Keep observed facts separate from interpretation.

Use these headings:

Executive Summary
Leading Opportunity
Key Evidence
Key Risks
Recommended Next Steps
Limitations
"""

    response = invoke_llm(prompt)

    return {"report": response_text(response)}


def route_after_quality(state: ScoutState):
    return (
        "score_locations"
        if state["data_quality"]["sufficient"]
        else "no_data"
    )


graph = StateGraph(ScoutState)

graph.add_node("understand_business", understand_business)
graph.add_node("analyse_data", analyse_data)
graph.add_node("assess_data_quality", assess_data_quality)
graph.add_node("score_locations", score_locations)
graph.add_node("analyse_risks", analyse_risks)
graph.add_node("generate_report", generate_report)
graph.add_node("no_data", no_data)

graph.add_edge(START, "understand_business")
graph.add_edge("understand_business", "analyse_data")
graph.add_edge("analyse_data", "assess_data_quality")

graph.add_conditional_edges(
    "assess_data_quality",
    route_after_quality,
    {
        "score_locations": "score_locations",
        "no_data": "no_data",
    },
)

graph.add_edge("score_locations", "analyse_risks")
graph.add_edge("analyse_risks", "generate_report")
graph.add_edge("generate_report", END)
graph.add_edge("no_data", END)

app = graph.compile()


def run_scout(
    business_type,
    start_hour=7,
    end_hour=10,
    day_filter="All",
):
    return app.invoke(
        {
            "question": business_type,
            "business_type": business_type,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "day_filter": day_filter,
            "business_profile": {},
            "location_analysis": [],
            "data_quality": {},
            "opportunity_scores": [],
            "risk_flags": [],
            "report": "",
        }
    )


if __name__ == "__main__":
    business = input("What business are you considering opening in Sydney?\n> ")

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
