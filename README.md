# Sydney Site Scout

AI-powered location intelligence using real City of Sydney
pedestrian-count data.

## Overview

Sydney Site Scout compares pedestrian activity across four Sydney
locations for a user-defined business and trading period.

The application combines deterministic data analysis with a small
generative-AI layer:

-   **Pandas** prepares the data and calculates location metrics.
-   **Python** applies the business-fit scoring logic.
-   **Gemini** converts the business description into structured
    priorities and writes the final decision brief.
-   **LangGraph** manages the workflow and data-quality branch.
-   **LangChain** provides the Gemini integration.
-   **Streamlit** provides the interface.

The main design principle is:

> **The code calculates the evidence; the AI interprets it.**

## Data

The project uses real pedestrian-count observations from the City of
Sydney, covering:

-   Park Street
-   Market Street
-   Bridge Street
-   Elizabeth Street

The analysis filters observations to the selected trading hours and
optional day before calculating location-level features.

Key features include:

-   average pedestrian demand
-   P95 pedestrian count
-   peak intensity
-   traffic consistency
-   weekday/weekend activity
-   recent change
-   year-on-year change
-   historical performance

## Scoring

Locations receive relative 0--100 scores across the main metrics.

Gemini creates a structured business profile from the user's
description, including the relative importance of demand, consistency,
peak activity, trends and weekday/weekend behaviour.

Python then combines those preferences with the calculated metrics to
produce a **business-fit score**.

The score is deterministic and relative to the four locations in the
dataset. It is not a prediction of revenue or business success.

## LangGraph workflow

The application is organised as a small stateful workflow:

``` text
Business profile
      ↓
Data analysis
      ↓
Data quality check
      ↓
Location scoring
      ↓
Risk analysis
      ↓
AI decision brief
```

If the selected period does not contain enough usable data, the workflow
stops before generating a location report.

## Key considerations

-   Pedestrian counts are an activity signal, not customer counts.
-   Only four locations are available, so comparisons are relative
    rather than Sydney-wide.
-   The application does not account for rent, competition,
    demographics, zoning or revenue.
-   P95 is used as the main peak measure rather than relying on a single
    maximum observation.
-   The LLM is deliberately kept out of the numerical scoring layer.

## Running locally

Install the dependencies and add a Gemini API key to `.env`:

``` env
GEMINI_API_KEY=your_key_here
```

Then run:

``` powershell
streamlit run app.py
```

## Stack

Python · pandas · NumPy · LangChain · LangGraph · Gemini · Streamlit ·
Git
