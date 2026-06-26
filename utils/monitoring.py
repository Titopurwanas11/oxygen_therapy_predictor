"""
Telemetry and Monitoring Data Managers for OxyPredict.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.prediction import get_confidence_level, get_risk_level


def generate_dummy_monitoring_data() -> pd.DataFrame:
    """Generate a realistic dataset of 120 patients for the monitoring dashboard preview."""
    np.random.seed(42)
    n = 120

    predictions = np.random.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
    probabilities = []
    conf_levels = []
    risk_levels = []

    for pred in predictions:
        if pred == "Yes":
            prob_pct = np.random.uniform(50.0, 97.0)
        else:
            prob_pct = np.random.uniform(3.0, 49.9)

        prob_dec = prob_pct / 100.0
        probabilities.append(round(prob_pct, 2))

        # Get confidence & risk
        _, conf, _, _ = get_confidence_level(prob_dec, pred)
        risk, _, _ = get_risk_level(prob_dec)

        conf_levels.append(conf)
        risk_levels.append(risk)

    dummy_df = pd.DataFrame({
        "Prediction": predictions,
        "Probability": probabilities,
        "Confidence Level": conf_levels,
        "Risk Level": risk_levels
    })

    dummy_df.insert(0, "Patient ID", [f"Sample Patient #{i+1}" for i in range(n)])
    return dummy_df


def get_monitoring_data() -> tuple:
    """
    Get prediction data for analytics. Checks session_state first.

    Returns:
        tuple: (pd.DataFrame, bool) -> (Dataframe, is_dummy_flag)
    """
    if st.session_state.get("batch_predicted", False) and "batch_results" in st.session_state:
        return st.session_state.batch_results, False
    else:
        return generate_dummy_monitoring_data(), True


def calculate_confidence_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate counts and percentages for all confidence groups.

    Args:
        df: Input enriched results DataFrame.

    Returns:
        pd.DataFrame: Table with Confidence Level, Count, and Percentage (%).
    """
    levels = ["Very High", "High", "Moderate", "Low", "Very Low"]
    counts = []
    pcts = []
    total = len(df)

    for lvl in levels:
        c = int((df["Confidence Level"] == lvl).sum())
        counts.append(c)
        pcts.append(round((c / total) * 100.0, 1) if total > 0 else 0.0)

    dist_df = pd.DataFrame({
        "Confidence Level": levels,
        "Count": counts,
        "Percentage (%)": pcts
    })
    return dist_df
