"""
Statistics and Automated Narrative Generation for OxyPredict Batch Analysis.
"""

import pandas as pd


def calculate_population_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics across all patient predictions.

    Args:
        df: Enriched prediction results DataFrame.

    Returns:
        dict: Population statistical metrics.
    """
    total = len(df)
    if total == 0:
        return {}

    need_oxy = int((df["Prediction"] == "Yes").sum())
    no_oxy = int((df["Prediction"] == "No").sum())
    avg_prob = float(df["Probability"].mean())

    # Map risk levels to Low, Medium, High counts
    low_risk = int(df["Risk Level"].isin(["Low Risk", "Low-Moderate Risk"]).sum())
    med_risk = int(df["Risk Level"].isin(["Moderate Risk", "Medium Risk"]).sum())
    high_risk = int(df["Risk Level"].isin(["High Risk", "Very High Risk"]).sum())

    need_oxy_pct = (need_oxy / total) * 100.0 if total > 0 else 0.0

    return {
        "total_patients": total,
        "need_oxy": need_oxy,
        "no_oxy": no_oxy,
        "avg_probability": avg_prob,
        "low_risk": low_risk,
        "med_risk": med_risk,
        "high_risk": high_risk,
        "need_oxy_pct": need_oxy_pct,
    }


def generate_population_narrative(stats: dict) -> str:
    """
    Generate dynamic clinical narrative summary for the analyzed population.

    Args:
        stats: Dictionary computed by calculate_population_stats.

    Returns:
        str: Automatically generated narrative string.
    """
    total = stats.get("total_patients", 0)
    need_oxy = stats.get("need_oxy", 0)
    need_oxy_pct = stats.get("need_oxy_pct", 0.0)
    high_risk = stats.get("high_risk", 0)
    med_risk = stats.get("med_risk", 0)
    avg_prob = stats.get("avg_probability", 0.0)

    narrative = (
        f"Among {total} pediatric patients analyzed, the model predicts that {need_oxy} patients "
        f"({need_oxy_pct:.1f}%) are likely to require oxygen therapy. "
        f"{high_risk} patients are classified as High Risk while {med_risk} are Medium Risk. "
        f"The average predicted probability across all patients is {avg_prob:.1f}%. "
        "These findings suggest that a considerable proportion of the analyzed population "
        "presents clinical characteristics associated with respiratory compromise. "
        "This report may assist healthcare professionals in prioritizing patient assessment "
        "and oxygen resource allocation."
    )
    return narrative
