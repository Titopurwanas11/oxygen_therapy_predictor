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

    # Map risk levels to Low, Medium, High counts using Indonesian labels
    low_risk = int(df["Risk Level"].isin(["Risiko Rendah", "Risiko Rendah-Sedang"]).sum())
    med_risk = int(df["Risk Level"].isin(["Risiko Sedang"]).sum())
    high_risk = int(df["Risk Level"].isin(["Risiko Tinggi", "Risiko Sangat Tinggi"]).sum())

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
        f"Dari {total} pasien pediatrik yang dianalisis, model memprediksi bahwa {need_oxy} pasien "
        f"({need_oxy_pct:.1f}%) kemungkinan membutuhkan terapi oksigen. "
        f"Sebanyak {high_risk} pasien diklasifikasikan dalam Risiko Tinggi sementara {med_risk} pasien dalam Risiko Sedang. "
        f"Rata-rata probabilitas prediksi pada seluruh pasien adalah {avg_prob:.1f}%. "
        "Temuan ini menunjukkan bahwa proporsi pasien yang dianalisis memperlihatkan karakteristik klinis yang terkait dengan gangguan pernapasan. "
        "Laporan ini dapat membantu tenaga kesehatan dalam memprioritaskan penilaian pasien dan alokasi sumber daya oksigen."
    )
    return narrative
