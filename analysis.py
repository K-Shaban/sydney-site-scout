import numpy as np
import pandas as pd


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
WEEKENDS = ["Saturday", "Sunday"]


def load_data(path="data/pedestrian_counts.csv"):
    """Load and clean the City of Sydney pedestrian-count dataset."""
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    numeric_columns = [
        "TotalCount",
        "Hour",
        "LastWeek",
        "Previous4DayTimeAvg",
        "LastYear",
        "Previous52DayTimeAvg",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    required = ["Location_code", "Location_Name", "Date", "TotalCount", "Hour", "Day"]
    df = df.dropna(subset=required).copy()

    df["Day"] = df["Day"].astype(str)

    return df


def _percentile_score(series):
    """Convert a metric into a 0-100 relative score across locations."""
    series = pd.to_numeric(series, errors="coerce")

    if series.notna().sum() == 0:
        return pd.Series(50.0, index=series.index)

    filled = series.fillna(series.median())

    if filled.nunique() <= 1:
        return pd.Series(50.0, index=series.index, dtype=float)

    return filled.rank(pct=True) * 100


def _safe_change(current, baseline):
    """Return mean percentage change against a positive baseline."""
    valid = baseline > 0
    if valid.sum() == 0:
        return np.nan

    changes = ((current[valid] - baseline[valid]) / baseline[valid]) * 100
    return changes.mean()


def analyse_locations(df, start_hour=7, end_hour=10, selected_days=None):
    """
    Build location-level features for the selected trading window.

    The function deliberately keeps calculations deterministic:
    Pandas calculates the evidence; the LLM interprets it later.
    """
    filtered = df[
        (df["Hour"] >= start_hour)
        & (df["Hour"] < end_hour)
    ].copy()

    if selected_days:
        filtered = filtered[filtered["Day"].isin(selected_days)]
        
    if filtered.empty:
        return []

    rows = []

    for (location_code, location_name), group in filtered.groupby(
        ["Location_code", "Location_Name"],
        dropna=False,
    ):
        traffic = group["TotalCount"].dropna()

        if traffic.empty:
            continue

        average = traffic.mean()
        p95 = traffic.quantile(0.95)
        maximum = traffic.max()
        std = traffic.std()

        coefficient_variation = std / average if average else np.nan
        consistency = (
            1 / (1 + coefficient_variation)
            if pd.notna(coefficient_variation)
            else np.nan
        )

        peak_intensity = p95 / average if average else np.nan

        weekday_values = group[
            group["Day"].isin(WEEKDAYS)
        ]["TotalCount"]

        weekend_values = group[
            group["Day"].isin(WEEKENDS)
        ]["TotalCount"]

        weekday_avg = weekday_values.mean()
        weekend_avg = weekend_values.mean()

        weekday_weekend_ratio = (
            weekday_avg / weekend_avg
            if pd.notna(weekday_avg)
            and pd.notna(weekend_avg)
            and weekend_avg != 0
            else np.nan
        )

        recent_change = _safe_change(
            group["TotalCount"],
            group["LastWeek"],
        )

        yoy_change = _safe_change(
            group["TotalCount"],
            group["LastYear"],
        )

        baseline = group["Previous52DayTimeAvg"]
        valid_baseline = baseline > 0

        historical_index = (
            (group.loc[valid_baseline, "TotalCount"]
             / baseline[valid_baseline]).mean()
            if valid_baseline.sum() > 0
            else np.nan
        )

        rows.append(
            {
                "Location_code": location_code,
                "Location_Name": location_name,
                "avg_pedestrians": average,
                "p95_pedestrians": p95,
                "max_pedestrians": maximum,
                "peak_intensity": peak_intensity,
                "traffic_volatility": coefficient_variation,
                "traffic_consistency": consistency,
                "weekday_avg": weekday_avg,
                "weekend_avg": weekend_avg,
                "weekday_weekend_ratio": weekday_weekend_ratio,
                "recent_change_pct": recent_change,
                "yoy_change_pct": yoy_change,
                "historical_index": historical_index,
                "observations": len(group),
            }
        )

    results = pd.DataFrame(rows)

    if results.empty:
        return []

    # Relative scores are calculated only after all locations are available.
    results["demand_score"] = _percentile_score(results["avg_pedestrians"])
    results["peak_score"] = _percentile_score(results["p95_pedestrians"])
    results["consistency_score"] = _percentile_score(
        results["traffic_consistency"]
    )
    results["weekday_score"] = _percentile_score(results["weekday_avg"])
    results["weekend_score"] = _percentile_score(results["weekend_avg"])
    results["trend_score"] = _percentile_score(
        results["recent_change_pct"].fillna(0)
    )
    results["historical_score"] = _percentile_score(
        results["historical_index"].fillna(1)
    )

    def build_risk_flags(row):
        flags = []

        if pd.notna(row["traffic_volatility"]) and row["traffic_volatility"] > 0.75:
            flags.append("High traffic volatility")

        if pd.notna(row["peak_intensity"]) and row["peak_intensity"] > 2:
            flags.append("Traffic concentrated around peak periods")

        if pd.notna(row["recent_change_pct"]) and row["recent_change_pct"] < -10:
            flags.append("Recent traffic decline")

        if pd.notna(row["yoy_change_pct"]) and row["yoy_change_pct"] < -10:
            flags.append("Year-on-year traffic decline")

        if row["observations"] < 100:
            flags.append("Limited observations")

        return flags

    results["risk_flags"] = results.apply(build_risk_flags, axis=1)

    numeric_columns = [
        "avg_pedestrians",
        "p95_pedestrians",
        "max_pedestrians",
        "peak_intensity",
        "traffic_volatility",
        "traffic_consistency",
        "weekday_avg",
        "weekend_avg",
        "weekday_weekend_ratio",
        "recent_change_pct",
        "yoy_change_pct",
        "historical_index",
        "demand_score",
        "peak_score",
        "consistency_score",
        "weekday_score",
        "weekend_score",
        "trend_score",
        "historical_score",
    ]

    results[numeric_columns] = results[numeric_columns].round(2)

    return results.to_dict("records")
