"""
Prediction utility functions for the Oxygen Therapy Predictor.
Handles model loading, single prediction, and batch prediction.
"""

import joblib
import pandas as pd
import streamlit as st
import numpy as np

from utils.config import MODEL_PATH, ALL_FEATURES


class ModelLoadError(Exception):
    """Custom exception raised when the model fails to load."""
    pass


@st.cache_resource
def load_model():
    """Load the trained Random Forest pipeline model."""
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        from utils.config import logger
        logger.error("Failed to load model from path %s: %s", MODEL_PATH, str(e), exc_info=True)
        raise ModelLoadError(f"Model loading failed: {e}")


def predict_single(input_data: dict) -> tuple:
    """
    Predict oxygen therapy need for a single patient.

    Args:
        input_data: Dictionary of feature name -> value.

    Returns:
        Tuple of (prediction_label, probability).
        prediction_label: 'Yes' or 'No'
        probability: float between 0 and 1 (probability of class 1 = Yes)
    """
    model = load_model()
    df = pd.DataFrame([input_data], columns=ALL_FEATURES)
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    # Class 1 = Yes (needs oxygen therapy)
    prob_yes = probabilities[1]
    label = "Yes" if prediction == 1 else "No"

    return label, prob_yes


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict oxygen therapy need for a batch of patients.

    Args:
        df: DataFrame with columns matching ALL_FEATURES.

    Returns:
        DataFrame with added 'Prediction' and 'Probability (%)' columns.
    """
    model = load_model()

    # Ensure column order matches expected features
    df_input = df[ALL_FEATURES].copy()

    predictions = model.predict(df_input)
    probabilities = model.predict_proba(df_input)[:, 1]

    result_df = df.copy()
    result_df["Prediction"] = np.where(predictions == 1, "Yes", "No")
    result_df["Probability"] = np.round(probabilities * 100, 2)

    return result_df


def validate_columns(df: pd.DataFrame) -> tuple:
    """
    Validate that uploaded DataFrame has all required columns.

    Args:
        df: Uploaded DataFrame.

    Returns:
        Tuple of (is_valid, missing_columns, extra_columns).
    """
    required = set(ALL_FEATURES)
    actual = set(df.columns)

    missing = required - actual
    extra = actual - required

    is_valid = len(missing) == 0

    return is_valid, list(missing), list(extra)


def get_confidence_level(probability: float, prediction: str) -> tuple:
    """
    Calculate the model's prediction confidence.

    Args:
        probability: float, predicted probability for class 1 (Yes).
        prediction: str, 'Yes' or 'No'.

    Returns:
        Tuple of (confidence_percentage, confidence_label, confidence_color, confidence_icon).
    """
    if prediction == "Yes" or prediction == 1:
        conf_pct = probability * 100
    else:
        conf_pct = (1.0 - probability) * 100

    if conf_pct >= 95.0:
        return conf_pct, "Sangat Tinggi", "#166534", "🟢"
    elif conf_pct >= 90.0:
        return conf_pct, "Tinggi", "#16a34a", "🟢"
    elif conf_pct >= 80.0:
        return conf_pct, "Sedang", "#ca8a04", "🟡"
    elif conf_pct >= 70.0:
        return conf_pct, "Rendah", "#ea580c", "🟠"
    else:
        return conf_pct, "Sangat Rendah", "#dc2626", "🔴"


def get_risk_level(probability: float) -> tuple:
    """
    Determine the clinical risk level based on the probability of needing oxygen therapy.

    Args:
        probability: float, predicted probability for class 1 (Yes).

    Returns:
        Tuple of (risk_label, risk_color, risk_icon).
    """
    prob_pct = probability * 100

    if prob_pct < 30.0:
        return "Risiko Rendah", "#16a34a", "🟢"
    elif prob_pct < 50.0:
        return "Risiko Rendah-Sedang", "#84cc16", "🟢"
    elif prob_pct < 70.0:
        return "Risiko Sedang", "#ca8a04", "🟡"
    elif prob_pct <= 90.0:
        return "Risiko Tinggi", "#ea580c", "🟠"
    else:
        return "Risiko Sangat Tinggi", "#dc2626", "🔴"
