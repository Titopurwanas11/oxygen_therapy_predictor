"""
Plotly Chart Generation for OxyPredict Batch Prediction Dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_pie_chart(df: pd.DataFrame):
    """
    Create a Pie Chart showing Need Oxygen vs No Oxygen counts.
    """
    counts_dict = df["Prediction"].value_counts().to_dict()
    status_order = ["Yes", "No"]
    counts = pd.DataFrame([
        {"Status": "Need Oxygen" if status == "Yes" else "No Oxygen", "Count": counts_dict.get(status, 0)}
        for status in status_order
    ])

    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Need Oxygen": "#ef4444", "No Oxygen": "#10b981"},
        hole=0.4,
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_risk_distribution_chart(df: pd.DataFrame):
    """
    Create a Bar Chart showing the distribution of patients across Low, Medium, and High risk levels.
    """
    def map_risk(risk):
        if risk in ["Low Risk", "Low-Moderate Risk"]:
            return "Low"
        if risk in ["Moderate Risk", "Medium Risk"]:
            return "Medium"
        return "High"

    mapped_risks = df["Risk Level"].apply(map_risk)
    counts_dict = mapped_risks.value_counts().to_dict()
    
    risk_order = ["Low", "Medium", "High"]
    risks = pd.DataFrame([
        {"Risk Level": rsk, "Count": counts_dict.get(rsk, 0)}
        for rsk in risk_order
    ])

    fig = px.bar(
        risks,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
        category_orders={"Risk Level": ["Low", "Medium", "High"]}
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9")
    )
    return fig


def create_probability_histogram(df: pd.DataFrame):
    """
    Create a Histogram showing prediction probability frequency.
    """
    fig = px.histogram(
        df,
        x="Probability",
        nbins=20,
        color_discrete_sequence=["#3b82f6"],
        labels={"Probability": "Prediction Probability (%)"}
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9")
    )
    return fig


def create_avg_prob_per_risk_chart(df: pd.DataFrame):
    """
    Create a Bar Chart showing the average predicted probability per Risk Level.
    """
    def map_risk(risk):
        if risk in ["Low Risk", "Low-Moderate Risk"]:
            return "Low"
        if risk in ["Moderate Risk", "Medium Risk"]:
            return "Medium"
        return "High"

    temp_df = df.copy()
    temp_df["Risk Group"] = temp_df["Risk Level"].apply(map_risk)

    avg_probs_dict = temp_df.groupby("Risk Group")["Probability"].mean().to_dict()
    
    risk_order = ["Low", "Medium", "High"]
    avg_probs = pd.DataFrame([
        {"Risk Group": rsk, "Average Probability (%)": avg_probs_dict.get(rsk, 0.0)}
        for rsk in risk_order
    ])

    fig = px.bar(
        avg_probs,
        x="Risk Group",
        y="Average Probability (%)",
        color="Risk Group",
        color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
        category_orders={"Risk Group": ["Low", "Medium", "High"]}
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9")
    )
    return fig


def create_confidence_histogram(df: pd.DataFrame):
    """
    Create a Bar Chart showing the count of patients in each confidence category.
    """
    levels_order = ["Very High", "High", "Moderate", "Low", "Very Low"]
    counts_dict = df["Confidence Level"].value_counts().to_dict()
    
    counts = pd.DataFrame([
        {"Confidence Level": lvl, "Count": counts_dict.get(lvl, 0)}
        for lvl in levels_order
    ])

    fig = px.bar(
        counts,
        x="Confidence Level",
        y="Count",
        color="Confidence Level",
        color_discrete_map={
            "Very High": "#166534",
            "High": "#16a34a",
            "Moderate": "#ca8a04",
            "Low": "#ea580c",
            "Very Low": "#dc2626"
        },
        category_orders={"Confidence Level": levels_order}
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9")
    )
    return fig


def create_avg_confidence_gauge(avg_conf: float):
    """
    Create a Gauge Chart of the average confidence percentage.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_conf,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0a2e52"},
            'bar': {'color': "#2563eb"},
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 70], 'color': 'rgba(220, 38, 38, 0.08)'},
                {'range': [70, 85], 'color': 'rgba(245, 158, 11, 0.08)'},
                {'range': [85, 100], 'color': 'rgba(22, 163, 74, 0.08)'}
            ],
        }
    ))
    fig.update_layout(
        margin=dict(t=30, b=10, l=15, r=15),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#0a2e52", 'family': "Inter"}
    )
    return fig


def create_prediction_ratio_donut(df: pd.DataFrame):
    """
    Create a Donut Chart showing prediction ratio.
    """
    counts_dict = df["Prediction"].value_counts().to_dict()
    status_order = ["Yes", "No"]
    counts = pd.DataFrame([
        {"Status": "Need Oxygen" if status == "Yes" else "No Oxygen", "Count": counts_dict.get(status, 0)}
        for status in status_order
    ])

    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Need Oxygen": "#ef4444", "No Oxygen": "#10b981"},
        hole=0.6,
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
