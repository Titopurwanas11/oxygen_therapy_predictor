"""
Telemetry and Monitoring Data Managers for OxyPredict.
"""

import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import ALL_FEATURES

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "prediction_history.csv")

def record_prediction(patient_data: dict, prediction: str, confidence: float, risk_level: str, type: str = "Single"):
    """
    Record a single prediction result with all 44 input features to prediction_history.csv.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_dict = {
        "Timestamp": timestamp,
        "Prediction": str(prediction),
        "Confidence": float(confidence),
        "Risk Level": str(risk_level),
        "Type": str(type)
    }
    
    # Extract age explicitly as "Age" for backward compatibility and quick statistics
    row_dict["Age"] = float(patient_data.get("Age (months)", 0))
    
    # Store all 44 features
    for feat in ALL_FEATURES:
        row_dict[feat] = patient_data.get(feat, np.nan)
        
    new_data = pd.DataFrame([row_dict])
    
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(HISTORY_FILE, index=False)
        except Exception as e:
            from utils.config import logger
            logger.error("Failed to append single prediction to history: %s", str(e), exc_info=True)
            new_data.to_csv(HISTORY_FILE, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, index=False)

def record_predictions_from_df(df_results: pd.DataFrame, type: str = "Batch"):
    """
    Record batch predictions with all 44 features to prediction_history.csv from an enriched result DataFrame.
    """
    if df_results.empty:
        return
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    
    for _, row in df_results.iterrows():
        # calculate confidence
        prob = row["Probability"]
        pred = row["Prediction"]
        conf = prob if pred == "Yes" else (100.0 - prob)
        
        row_dict = {
            "Timestamp": timestamp,
            "Prediction": str(pred),
            "Confidence": float(conf),
            "Risk Level": str(row.get("Risk Level", "Unknown")),
            "Type": str(type)
        }
        
        # Age extraction for statistics
        row_dict["Age"] = float(row.get("Age (months)", 0))
        
        # Store all 44 features
        for feat in ALL_FEATURES:
            row_dict[feat] = row.get(feat, np.nan)
            
        records.append(row_dict)
        
    new_data = pd.DataFrame(records)
    
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(HISTORY_FILE, index=False)
        except Exception as e:
            from utils.config import logger
            logger.error("Failed to append batch predictions to history: %s", str(e), exc_info=True)
            new_data.to_csv(HISTORY_FILE, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, index=False)

def get_prediction_history() -> pd.DataFrame:
    """
    Retrieve the complete prediction history from CSV file.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            return df
        except Exception as e:
            from utils.config import logger
            logger.error("Failed to read prediction history: %s", str(e), exc_info=True)
            return pd.DataFrame(columns=["Timestamp", "Age", "Prediction", "Confidence", "Risk Level", "Type"])
    return pd.DataFrame(columns=["Timestamp", "Age", "Prediction", "Confidence", "Risk Level", "Type"])

def calculate_confidence_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate counts and percentages for all confidence groups.
    """
    levels = ["Very High", "High", "Moderate", "Low", "Very Low"]
    counts = []
    pcts = []
    total = len(df)

    # Map confidence float to category
    conf_cats = []
    for _, row in df.iterrows():
        c = row["Confidence"]
        if c >= 95.0:
            conf_cats.append("Very High")
        elif c >= 90.0:
            conf_cats.append("High")
        elif c >= 80.0:
            conf_cats.append("Moderate")
        elif c >= 70.0:
            conf_cats.append("Low")
        else:
            conf_cats.append("Very Low")
            
    df_temp = df.copy()
    df_temp["Confidence Level"] = conf_cats

    for lvl in levels:
        c = int((df_temp["Confidence Level"] == lvl).sum())
        counts.append(c)
        pcts.append(round((c / total) * 100.0, 1) if total > 0 else 0.0)

    dist_df = pd.DataFrame({
        "Confidence Level": levels,
        "Count": counts,
        "Percentage (%)": pcts
    })
    return dist_df
