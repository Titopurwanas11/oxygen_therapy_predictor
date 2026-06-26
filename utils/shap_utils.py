"""
SHAP Explanation utilities for the Oxygen Therapy Predictor.

Contains:
- Metadata and descriptions for pre-generated SHAP visualisation images.
- Real SHAP computation functions for individual patient predictions using
  TreeExplainer on the Random Forest model inside the sklearn Pipeline.
"""

import os
import numpy as np
import pandas as pd

try:
    import streamlit as st
    _has_st = True
except ImportError:
    _has_st = False

from utils.config import ASSETS_DIR, ALL_FEATURES


# =============================================================================
# 1) Plot metadata helpers (unchanged from the original)
# =============================================================================

def get_shap_plots_metadata() -> list:
    """
    Get metadata and descriptions for the SHAP plots.
    
    Returns:
        List of dictionaries containing plot title, filename, descriptions, and styling.
    """
    return [
        {
            "title": "📊 SHAP Bar Plot — Feature Importance",
            "filename": "shap_bar_plot.png",
            "description": """
            **SHAP Bar Plot** menampilkan rata-rata nilai absolut SHAP untuk setiap fitur,
            yang menunjukkan **tingkat kepentingan (importance)** masing-masing fitur dalam model.
            Fitur dengan bar lebih panjang memiliki pengaruh lebih besar terhadap keputusan prediksi model.

            **Cara membaca:**
            - Sumbu X menunjukkan rata-rata |SHAP value|
            - Sumbu Y menampilkan nama fitur, diurutkan dari yang paling penting
            - Fitur di posisi atas adalah fitur yang paling berpengaruh dalam model
            """,
            "color_start": "#1a73e8",
            "color_end": "#1557b0",
            "bg": "linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)",
            "border": "#bfdbfe",
        },
        {
            "title": "🎯 SHAP Summary Plot — Feature Impact",
            "filename": "shap_summary_plot.png",
            "description": """
            **SHAP Summary Plot (Beeswarm)** menampilkan distribusi SHAP values untuk setiap fitur
            di seluruh dataset. Plot ini menunjukkan **arah dan besaran pengaruh** setiap fitur.

            **Cara membaca:**
            - Setiap titik mewakili satu data pasien
            - Warna menunjukkan nilai fitur: <span style="color: #dc2626; font-weight: 700;">merah = tinggi</span>,
              <span style="color: #2563eb; font-weight: 700;">biru = rendah</span>
            - Posisi horizontal menunjukkan dampak terhadap prediksi
            - Titik di kanan garis nol → mendorong prediksi ke "Butuh Terapi Oksigen"
            - Titik di kiri garis nol → mendorong prediksi ke "Tidak Butuh Terapi Oksigen"
            """,
            "color_start": "#059669",
            "color_end": "#047857",
            "bg": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
            "border": "#bbf7d0",
        },
        {
            "title": "🔬 SHAP Waterfall Plot — Individual Explanation",
            "filename": "shap_waterfall.png",
            "description": """
            **SHAP Waterfall Plot** menjelaskan prediksi untuk **satu pasien tertentu** secara detail.
            Plot ini menampilkan bagaimana setiap fitur mendorong prediksi dari nilai dasar (base value)
            menuju nilai prediksi akhir.

            **Cara membaca:**
            - Base value (E[f(x)]) adalah rata-rata prediksi model
            - Bar <span style="color: #dc2626; font-weight: 700;">merah</span> menunjukkan fitur
              yang **meningkatkan** probabilitas terapi oksigen
            - Bar <span style="color: #2563eb; font-weight: 700;">biru</span> menunjukkan fitur
              yang **menurunkan** probabilitas terapi oksigen
            - f(x) di bagian atas adalah nilai prediksi akhir
            """,
            "color_start": "#7c3aed",
            "color_end": "#6d28d9",
            "bg": "linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%)",
            "border": "#d8b4fe",
        },
    ]


def get_plot_path(filename: str) -> str:
    """Get the absolute path to a SHAP plot image."""
    return os.path.join(ASSETS_DIR, filename)


def check_plot_exists(filename: str) -> bool:
    """Check if the SHAP plot image exists in the assets directory."""
    return os.path.exists(get_plot_path(filename))


# =============================================================================
# 2) Real SHAP computation for individual patient predictions
# =============================================================================

def _cache_resource(func):
    """Apply st.cache_resource when running inside Streamlit, otherwise passthrough."""
    if _has_st:
        return st.cache_resource(func)
    return func


@_cache_resource
def _get_shap_explainer():
    """
    Create and cache a SHAP TreeExplainer for the Random Forest model.

    The sklearn Pipeline is split into:
      - preprocessor  (ColumnTransformer: imputation + encoding + scaling)
      - estimator     (RandomForestClassifier)

    TreeExplainer is applied to the raw estimator so that SHAP values
    are computed natively on the tree structure (fast, exact).

    Returns:
        Tuple of (shap.TreeExplainer, preprocessor, feature_names_out).
    """
    import shap
    from utils.prediction import load_model

    pipeline = load_model()
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(estimator)

    # Get transformed feature names from the preprocessor
    # We need a dummy row to fit-transform; the preprocessor is already fitted.
    dummy = pd.DataFrame(
        {feat: ["dummy"] if feat not in _numerical_set() else [0.0] for feat in ALL_FEATURES},
        columns=ALL_FEATURES,
    )
    try:
        feature_names_out = list(preprocessor.get_feature_names_out(ALL_FEATURES))
    except Exception:
        # Fallback: transform and infer
        transformed = preprocessor.transform(dummy)
        feature_names_out = [f"feature_{i}" for i in range(transformed.shape[1])]

    return explainer, preprocessor, feature_names_out


def _numerical_set():
    """Return set of numerical feature names (from config)."""
    from utils.config import NUMERICAL_FEATURES
    return set(NUMERICAL_FEATURES)


def _map_transformed_to_original(feature_names_out: list) -> dict:
    """
    Build a mapping from each transformed feature name to its original feature.

    The ColumnTransformer produces names like:
      - num__Age (months)
      - cat__Gender_Male
      - cat__Wheezing_Yes

    For one-hot-encoded features the suffix after the last underscore is the
    category; we need to strip it to recover the original column name.

    Returns:
        Dict mapping transformed name -> original feature name.
    """
    mapping = {}
    for tname in feature_names_out:
        matched_original = None
        for feat in ALL_FEATURES:
            # Numerical: "num__<feat>"
            if tname == f"num__{feat}":
                matched_original = feat
                break
            # Categorical (binary dropped): "cat__<feat>"
            if tname == f"cat__{feat}":
                matched_original = feat
                break
            # Categorical (one-hot): "cat__<feat>_<category>"
            if tname.startswith(f"cat__{feat}_"):
                matched_original = feat
                break
        mapping[tname] = matched_original if matched_original else tname
    return mapping


def compute_shap_for_patient(input_data: dict) -> dict:
    """
    Compute real SHAP values for a single patient using TreeExplainer.

    Steps:
    1. Preprocess the input through the Pipeline's ColumnTransformer.
    2. Run TreeExplainer.shap_values() on the transformed data.
    3. Group the per-transformed-feature SHAP values back to the 44 original
       clinical features by summing contributions from one-hot columns.
    4. Return a structured result with per-feature SHAP values, the base value,
       and the predicted class.

    Args:
        input_data: Dict of {feature_name: value} for all 44 features.

    Returns:
        Dict with keys:
          - 'shap_values': list of dicts with 'feature', 'patient_value', 'shap_value'
                           sorted by |shap_value| descending.
          - 'base_value': float, the expected model output (base probability).
          - 'predicted_class': int (0 or 1).
          - 'predicted_prob': float, probability of class 1 (Yes).
    """
    explainer, preprocessor, feature_names_out = _get_shap_explainer()
    mapping = _map_transformed_to_original(feature_names_out)

    # Build a DataFrame from the input
    df = pd.DataFrame([input_data], columns=ALL_FEATURES)

    # Preprocess
    x_transformed = preprocessor.transform(df)

    # SHAP values — TreeExplainer returns a list of arrays (one per class) or
    # a numpy array depending on the version. We always want class-1 values.
    shap_raw = explainer.shap_values(x_transformed)

    if isinstance(shap_raw, list):
        # shap_raw[0] = class 0, shap_raw[1] = class 1
        shap_class1 = shap_raw[1][0] if len(shap_raw) > 1 else shap_raw[0][0]
    elif isinstance(shap_raw, np.ndarray):
        if shap_raw.ndim == 3:
            # Shape: (n_samples, n_features, n_classes)
            # We want sample 0, class 1
            shap_class1 = shap_raw[0, :, 1]
        elif shap_raw.ndim == 2:
            # Shape: (n_samples, n_features)
            shap_class1 = shap_raw[0]
        else:
            shap_class1 = shap_raw
    else:
        shap_class1 = shap_raw

    # Base value
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_value = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_value = float(base_val)

    # Group SHAP values back to original 44 features
    grouped = {}
    for idx, tname in enumerate(feature_names_out):
        orig = mapping.get(tname, tname)
        grouped[orig] = grouped.get(orig, 0.0) + float(shap_class1[idx])

    # Build result list
    num_set = _numerical_set()
    results = []
    for feat in ALL_FEATURES:
        raw_val = input_data.get(feat, "")

        # Format patient value with clinical context
        formatted_val = str(raw_val)
        feat_lower = feat.lower()
        if "saturation" in feat_lower or "sao2" in feat_lower:
            formatted_val = f"{raw_val}%"
        elif "temperature" in feat_lower:
            formatted_val = f"{raw_val}°C"
        elif feat == "Weight (Kg)":
            formatted_val = f"{raw_val} Kg"
        elif feat == "Height (cm)":
            formatted_val = f"{raw_val} cm"
        elif "respiratory rate" in feat_lower:
            formatted_val = f"{raw_val} bpm"
        elif "heart rate" in feat_lower:
            formatted_val = f"{raw_val} bpm"
        elif feat_lower.startswith("number of days"):
            formatted_val = f"{raw_val} days"
        elif feat == "Age (months)":
            formatted_val = f"{raw_val} mo"

        results.append({
            "feature": feat,
            "patient_value": formatted_val,
            "shap_value": grouped.get(feat, 0.0),
        })

    # Sort by absolute SHAP value descending
    results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Predicted probability (via the full pipeline)
    from utils.prediction import load_model
    pipeline = load_model()
    proba = pipeline.predict_proba(df)[0]
    pred_class = int(pipeline.predict(df)[0])

    return {
        "shap_values": results,
        "base_value": base_value,
        "predicted_class": pred_class,
        "predicted_prob": float(proba[1]),
    }


# =============================================================================
# 3) Clinical interpretation generator from SHAP results
# =============================================================================

def generate_shap_clinical_interpretation(shap_result: dict, label: str) -> str:
    """
    Generate a clinical interpretation paragraph based on SHAP values.

    The text is dynamically constructed from the top positive and negative
    contributing features so that it accurately reflects the model's reasoning.

    Args:
        shap_result: Output from compute_shap_for_patient().
        label: 'Yes' or 'No' — the prediction label.

    Returns:
        A multi-sentence clinical interpretation string (plain text with no markup).
    """
    all_sv = shap_result["shap_values"]
    top_positive = [s for s in all_sv if s["shap_value"] > 0.005][:5]
    top_negative = [s for s in all_sv if s["shap_value"] < -0.005][:5]

    def _humanise(feat_name: str) -> str:
        """Convert feature name to a more readable clinical term."""
        name = feat_name.lower()
        if "oxygen saturation" in name or "sao2" in name:
            return "oxygen saturation level"
        if "temperature" in name:
            return "body temperature"
        if "respiratory rate" in name:
            return "respiratory rate"
        if "heart rate" in name:
            return "heart rate"
        if "wheezing" in name:
            return "wheezing"
        if "nasal flaring" in name:
            return "nasal flaring"
        if "cyanosis" in name:
            return "cyanosis"
        if "crackles" in name:
            return "crackles"
        if "rhonchi" in name:
            return "rhonchi"
        if "stridor" in name:
            return "laryngeal stridor"
        if "sleepiness" in name:
            return "unusual sleepiness"
        if "consciousness" in name:
            return "disorders of consciousness"
        if "c-reactive" in name or "protein" in name:
            return "C-reactive protein"
        if "procalcitonin" in name:
            return "procalcitonin"
        if "hypoventilation" in name:
            return "hypoventilation"
        if "dehydration" in name:
            return "dehydration signs"
        if "restlessness" in name:
            return "restlessness"
        if "paleness" in name:
            return "paleness"
        if "weight" in name:
            return "body weight"
        if "height" in name:
            return "body height"
        if "age" in name:
            return "patient age"
        if "fever" in name and "days" in name:
            return "duration of fever"
        if "fever" in name:
            return "history of fever"
        if "vomiting" in name and "days" in name:
            return "duration of vomiting"
        if "vomiting" in name:
            return "history of vomiting"
        if "cough" in name:
            return "history of cough"
        if "diarrhea" in name:
            return "history of diarrhea"
        if "rhinorrhea" in name:
            return "rhinorrhea"
        if "breastfeeding" in name:
            return "breastfeeding status"
        if "nasopharyngeal" in name:
            return "nasopharyngeal aspiration"
        # Generic fallback
        clean = feat_name.replace("Health history : ", "").replace("history of ", "")
        return clean.strip().lower()

    def _list_str(items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + ", and " + items[-1]

    pos_terms = [_humanise(f["feature"]) for f in top_positive]
    neg_terms = [_humanise(f["feature"]) for f in top_negative]

    parts = []

    if label == "Yes":
        if pos_terms:
            parts.append(
                f"The prediction is mainly influenced by the presence of {_list_str(pos_terms)}. "
                f"These findings indicate clinical factors that strongly contribute to the prediction "
                f"that oxygen therapy is required."
            )
        if neg_terms:
            parts.append(
                f"Conversely, {_list_str(neg_terms)} partially reduced the prediction score, "
                f"but the overall evidence still supports the need for oxygen therapy."
            )
    else:
        if neg_terms:
            parts.append(
                f"The prediction is mainly supported by stable clinical indicators including "
                f"{_list_str(neg_terms)}. "
                f"These factors contribute to the prediction that oxygen therapy is not required."
            )
        if pos_terms:
            parts.append(
                f"However, {_list_str(pos_terms)} slightly increased the risk score, "
                f"suggesting continued monitoring may be warranted."
            )

    if not parts:
        parts.append(
            "The prediction is based on a balanced combination of clinical features "
            "without a single dominant factor."
        )

    return " ".join(parts)


def get_shap_feature_icon(feature_name: str) -> str:
    """Return an appropriate emoji icon for a clinical feature."""
    name = feature_name.lower()
    if "temperature" in name:
        return "🌡️"
    if "saturation" in name or "sao2" in name:
        return "💉"
    if "respiratory" in name:
        return "💨"
    if "heart" in name:
        return "❤️"
    if "wheezing" in name:
        return "🫁"
    if "nasal" in name:
        return "👃"
    if "cyanosis" in name:
        return "🔵"
    if "crackles" in name or "rhonchi" in name:
        return "🩻"
    if "weight" in name:
        return "⚖️"
    if "height" in name:
        return "📏"
    if "age" in name:
        return "👶"
    if "fever" in name:
        return "🤒"
    if "cough" in name:
        return "😷"
    if "protein" in name or "procalcitonin" in name:
        return "🧬"
    if "stridor" in name:
        return "🔊"
    if "sleepiness" in name or "consciousness" in name:
        return "😴"
    if "vomiting" in name or "diarrhea" in name:
        return "🤢"
    if "dehydration" in name:
        return "💧"
    return "📋"
