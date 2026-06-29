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
        {"Status": "Butuh Oksigen" if status == "Yes" else "Tidak Butuh Oksigen", "Count": counts_dict.get(status, 0)}
        for status in status_order
    ])

    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Butuh Oksigen": "#EF4444", "Tidak Butuh Oksigen": "#22C55E"},
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
        risk_text = str(risk).lower()
        if "rendah" in risk_text:
            return "Rendah"
        if "sedang" in risk_text:
            return "Sedang"
        if "tinggi" in risk_text:
            return "Tinggi"
        return "Tinggi"

    mapped_risks = df["Risk Level"].apply(map_risk)
    counts_dict = mapped_risks.value_counts().to_dict()
    
    risk_order = ["Rendah", "Sedang", "Tinggi"]
    risks = pd.DataFrame([
        {"Tingkat Risiko": rsk, "Count": counts_dict.get(rsk, 0)}
        for rsk in risk_order
    ])

    fig = px.bar(
        risks,
        x="Tingkat Risiko",
        y="Count",
        color="Tingkat Risiko",
        color_discrete_map={"Rendah": "#22C55E", "Sedang": "#F59E0B", "Tinggi": "#EF4444"},
        category_orders={"Tingkat Risiko": ["Rendah", "Sedang", "Tinggi"]}
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
        color_discrete_sequence=["#3282B8"],
        labels={"Probability": "Probabilitas Prediksi (%)"}
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
        risk_text = str(risk).lower()
        if "rendah" in risk_text:
            return "Rendah"
        if "sedang" in risk_text:
            return "Sedang"
        if "tinggi" in risk_text:
            return "Tinggi"
        return "Tinggi"

    temp_df = df.copy()
    temp_df["Kelompok Risiko"] = temp_df["Risk Level"].apply(map_risk)

    avg_probs_dict = temp_df.groupby("Kelompok Risiko")["Probability"].mean().to_dict()
    
    risk_order = ["Rendah", "Sedang", "Tinggi"]
    avg_probs = pd.DataFrame([
        {"Kelompok Risiko": rsk, "Rata-rata Probabilitas (%)": avg_probs_dict.get(rsk, 0.0)}
        for rsk in risk_order
    ])

    fig = px.bar(
        avg_probs,
        x="Kelompok Risiko",
        y="Rata-rata Probabilitas (%)",
        color="Kelompok Risiko",
        color_discrete_map={"Rendah": "#22C55E", "Sedang": "#F59E0B", "Tinggi": "#EF4444"},
        category_orders={"Kelompok Risiko": ["Rendah", "Sedang", "Tinggi"]}
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
    levels_order = ["Sangat Tinggi", "Tinggi", "Sedang", "Rendah", "Sangat Rendah"]
    mapping = {
        "Very High": "Sangat Tinggi",
        "High": "Tinggi",
        "Moderate": "Sedang",
        "Low": "Rendah",
        "Very Low": "Sangat Rendah"
    }
    counts_dict = df["Confidence Level"].map(mapping).value_counts().to_dict()
    
    counts = pd.DataFrame([
        {"Tingkat Keyakinan": lvl, "Count": counts_dict.get(lvl, 0)}
        for lvl in levels_order
    ])

    fig = px.bar(
        counts,
        x="Tingkat Keyakinan",
        y="Count",
        color="Tingkat Keyakinan",
        color_discrete_map={
            "Sangat Tinggi": "#15803D",
            "Tinggi": "#22C55E",
            "Sedang": "#F59E0B",
            "Rendah": "#EA580C",
            "Sangat Rendah": "#EF4444"
        },
        category_orders={"Tingkat Keyakinan": levels_order}
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
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0F4C75"},
            'bar': {'color': "#3282B8"},
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#D6E4F0",
            'steps': [
                {'range': [0, 70], 'color': 'rgba(239, 68, 68, 0.08)'},
                {'range': [70, 85], 'color': 'rgba(245, 158, 11, 0.08)'},
                {'range': [85, 100], 'color': 'rgba(34, 197, 94, 0.08)'}
            ],
        }
    ))
    fig.update_layout(
        margin=dict(t=30, b=10, l=15, r=15),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#0F172A", 'family': "Inter"}
    )
    return fig


def create_prediction_ratio_donut(df: pd.DataFrame):
    """
    Create a Donut Chart showing prediction ratio.
    """
    counts_dict = df["Prediction"].value_counts().to_dict()
    status_order = ["Yes", "No"]
    counts = pd.DataFrame([
        {"Status": "Butuh Oksigen" if status == "Yes" else "Tidak Butuh Oksigen", "Count": counts_dict.get(status, 0)}
        for status in status_order
    ])

    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Butuh Oksigen": "#EF4444", "Tidak Butuh Oksigen": "#22C55E"},
        hole=0.6,
    )
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
