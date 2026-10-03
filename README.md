# Sydney Site Scout

AI-assisted location intelligence application for comparing Sydney business locations using real pedestrian activity data.

## Overview

Sydney Site Scout analyses pedestrian activity across Sydney locations and ranks locations based on the requirements of a proposed business.

The application combines deterministic data analysis with a small generative-AI layer:

- Python and pandas process and analyse pedestrian count data.
- Gemini interprets the proposed business and generates business-specific priorities.
- Python calculates deterministic business-fit scores for each location.
- LangGraph manages the analysis workflow and data-quality checks.
- Gemini generates a concise business-facing decision brief.
- Streamlit provides an interactive user interface.

## Demo

The application allows a user to describe a proposed business, select its trading hours and days, and compare locations based on pedestrian activity and business-specific priorities.

### Business Input

The user enters a business type and selects the trading period to be analysed.

![Sydney Site Scout input](docs/demo/site-scout-input.png)

### Location Analysis

The application ranks the available locations and displays the recommended location, business-fit score, pedestrian metrics and an AI-generated decision brief.

![Sydney Site Scout result](docs/demo/site-scout-result.png)

The numerical ranking is calculated in Python, while Gemini interprets the results and provides a concise explanation.

## Data

The project uses City of Sydney pedestrian count data containing **188,720 observations** across four locations:

- Park Street
- Market Street
- Bridge Street
- Elizabeth Street

The data is filtered according to the user's selected trading hours and days.

Location analysis includes:

- **Average Pedestrian Activity** = Typical pedestrian volume during the selected period
- **P95 Pedestrian Activity** = High pedestrian activity without relying on extreme maximum values
- **Traffic Consistency** = Stability of pedestrian activity
- **Weekday / Weekend Activity** = Activity patterns across trading days
- **Recent Change** = Change in recent pedestrian activity
- **Year-on-Year Change** = Change compared with the corresponding historical period

## Scoring

Gemini converts the proposed business into business-specific priorities for factors including pedestrian demand, consistency, peak activity and historical trends.

Python then combines these priorities with deterministic location metrics to calculate a **0–100 business-fit score** and rank the available locations.

The AI does not directly calculate or modify the location scores.

## Workflow

```text
Business input
      ↓
AI business profile
      ↓
Pedestrian data analysis
      ↓
Data quality check
      ↓
Location scoring
      ↓
Risk analysis
      ↓
AI decision brief
      ↓
Streamlit
```

## Key considerations

- Pedestrian activity does not necessarily represent customers or purchasing intent.
- Business-fit scores compare the four available locations and are not probabilities of business success.
- The analysis does not include rent, competition, demographics, zoning or revenue.
- AI-generated insights explain the calculated results but do not determine the underlying location scores.

## Running locally

Install the project dependencies, add a Gemini API key to `.env`, then run:

```bash
streamlit run app.py
```

## Stack

Python · pandas · NumPy · LangChain · LangGraph · Google Gemini · Streamlit · Git