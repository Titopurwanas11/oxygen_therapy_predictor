"""
Batch Prediction Logic for OxyPredict.
Handles file validation and model prediction batch enrichment.
"""

import pandas as pd
from utils.prediction import predict_batch, validate_columns, get_confidence_level, get_risk_level
from utils.recommendation import generate_recommendation


def validate_uploaded_file(df: pd.DataFrame) -> tuple:
    """
    Validate that the uploaded dataframe contains all required features.

    Returns:
        tuple: (is_valid, missing_columns, extra_columns)
    """
    return validate_columns(df)


def run_batch_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute batch prediction and enrich the dataframe with risk levels, confidence levels, and recommendations.

    Args:
        df: Input DataFrame containing the clinical features.

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    # Execute base batch prediction (returns Prediction and Probability in %)
    result_df = predict_batch(df)

    conf_levels = []
    risk_levels = []
    recommendations = []
    priorities = []

    for _, row in result_df.iterrows():
        # Get raw probability as a decimal (0.0 to 1.0)
        prob_pct = row["Probability"]
        prob_dec = prob_pct / 100.0
        pred = row["Prediction"]

        # Calculate confidence & risk classifications
        _, conf_lbl, _, _ = get_confidence_level(prob_dec, pred)
        risk_lbl, _, _ = get_risk_level(prob_dec)

        conf_levels.append(conf_lbl)
        risk_levels.append(risk_lbl)

        # Build patient dictionary from row features
        row_dict = row.to_dict()

        # Generate recommendation
        rec = generate_recommendation(
            prediction=pred,
            probability=prob_dec,
            risk_level=risk_lbl,
            confidence_level=conf_lbl,
            patient_data=row_dict
        )

        recommendations.append("; ".join(rec["clinical_action"]))
        priorities.append(rec["priority"])

    result_df["Confidence Level"] = conf_levels
    result_df["Risk Level"] = risk_levels
    result_df["Recommendation"] = recommendations
    result_df["Priority"] = priorities

    return result_df
