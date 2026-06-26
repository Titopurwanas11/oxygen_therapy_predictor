"""
Explainable AI (XAI) utility functions for the Oxygen Therapy Predictor.
Calculates local feature contributions (Saabas method) for a single patient's prediction
using decision paths in the Random Forest model.
"""

import pandas as pd
import numpy as np
from utils.prediction import load_model
from utils.config import ALL_FEATURES

def load_explainer():
    """
    Load the explainer component. Since we are using our custom scikit-learn tree interpreter
    method, this simply returns the pre-trained pipeline model.
    """
    return load_model()

def explain_prediction(input_data: dict) -> list:
    """
    Explain a single prediction by calculating feature contributions using decision paths.
    
    Args:
        input_data: Dictionary of feature name -> value.
        
    Returns:
        List of dictionaries containing:
        - 'feature': Original feature name
        - 'value': Formatted input value
        - 'impact': Numerical contribution to the prediction probability
    """
    model = load_model()
    df = pd.DataFrame([input_data], columns=ALL_FEATURES)
    
    preprocess = model.named_steps['preprocess']
    forest = model.named_steps['model']
    
    # 1. Preprocess the input sample
    x_trans = preprocess.transform(df)
    names = preprocess.get_feature_names_out(ALL_FEATURES)
    
    # 2. Trace decision paths for all estimators to calculate contributions
    # We focus on target class 1 ("Yes" - Needs Oxygen Therapy)
    contribs = np.zeros(x_trans.shape[1])
    
    for estimator in forest.estimators_:
        tree = estimator.tree_
        # value array has shape (n_nodes, 1, 2)
        value = tree.value[:, 0, :]
        value_probs = value / value.sum(axis=1, keepdims=True)
        
        # Get path indices for this tree
        indicator = estimator.decision_path(x_trans)
        visited = indicator.indices
        
        for i in range(len(visited) - 1):
            parent = visited[i]
            child = visited[i + 1]
            split_feat = tree.feature[parent]
            if split_feat >= 0:
                # Difference in probability of class 1
                diff = value_probs[child, 1] - value_probs[parent, 1]
                contribs[split_feat] += diff
                
    # Average contributions over all decision trees
    contribs /= len(forest.estimators_)
    
    # 3. Group the 74 preprocessed feature contributions back to the 44 original features
    grouped_contribs = {}
    for name, val in zip(names, contribs):
        orig_feat = None
        for feat in ALL_FEATURES:
            if name.startswith("num__" + feat) or name.startswith("cat__" + feat):
                orig_feat = feat
                break
        if orig_feat:
            grouped_contribs[orig_feat] = grouped_contribs.get(orig_feat, 0.0) + val
            
    # 4. Format and pack results
    explanations = []
    for feat in ALL_FEATURES:
        impact = grouped_contribs.get(feat, 0.0)
        raw_val = input_data.get(feat, "")
        
        # Format values with clinical units where applicable
        formatted_val = str(raw_val)
        if "saturation" in feat.lower() or "sao2" in feat.lower():
            formatted_val = f"{raw_val}%"
        elif "temperature" in feat.lower():
            formatted_val = f"{raw_val}\u00b0C"
        elif "weight" in feat.lower():
            formatted_val = f"{raw_val} Kg"
        elif "height" in feat.lower():
            formatted_val = f"{raw_val} cm"
            
        explanations.append({
            "feature": feat,
            "value": formatted_val,
            "impact": impact
        })
        
    # Sort by absolute impact descending
    explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)
    
    return explanations

def get_feature_icon(feature_name: str) -> str:
    """Get visual icon for a feature based on its category."""
    respiratory_keywords = [
        "cough", "rhinorrhea", "nasal flaring", "laryngeal stridor", 
        "rhonchi", "crackles", "wheezing", "hypoventilation", 
        "nasopharyngeal", "respiratory condition", "asthmatic"
    ]
    if "temperature" in feature_name.lower():
        return "🌡️"
    elif any(kw in feature_name.lower() for kw in ["saturation", "sao2", "respiratory rate", "heart rate", "weight", "height"]):
        return "❤️"
    elif any(kw in feature_name.lower() for kw in respiratory_keywords):
        return "🫁"
    else:
        return "📋"

def get_clinical_interpretation(label: str, top_factors: list) -> str:
    """Generate a dynamic clinical interpretation based on top features."""
    supporting_features = []
    for f in top_factors:
        name = f["feature"]
        impact = f["impact"]
        # High positive impact supports Yes, negative supports No
        if (label == "Yes" and impact > 0) or (label == "No" and impact < 0):
            supporting_features.append(name.lower())
            
    if not supporting_features:
        # Fallback to top absolute features
        supporting_features = [f["feature"].lower() for f in top_factors[:3]]
        
    # Map to clinical descriptions
    clean_terms = []
    for sf in supporting_features:
        if "oxygen saturation" in sf or "sao2" in sf:
            clean_terms.append("oxygen saturation levels")
        elif "temperature" in sf:
            clean_terms.append("elevated body temperature" if label == "Yes" else "stable body temperature")
        elif "respiratory rate" in sf:
            clean_terms.append("increased respiratory rate" if label == "Yes" else "normal respiratory rate")
        elif "wheezing" in sf:
            clean_terms.append("presence of wheezing" if label == "Yes" else "absence of wheezing")
        elif "nasal flaring" in sf:
            clean_terms.append("nasal flaring")
        elif "stridor" in sf:
            clean_terms.append("laryngeal stridor")
        elif "heart rate" in sf:
            clean_terms.append("heart rate changes")
        elif "sleepiness" in sf or "consciousness" in sf:
            clean_terms.append("disorders of consciousness")
        elif "crp" in sf or "protein" in sf or "procalcitonin" in sf:
            clean_terms.append("elevated inflammatory markers" if label == "Yes" else "normal inflammatory markers")
        else:
            clean_name = sf.replace("health history :", "").replace("history of", "").strip()
            clean_terms.append(clean_name)
            
    clean_terms = list(dict.fromkeys(clean_terms))[:3]
    
    if label == "Yes":
        features_str = ", ".join(clean_terms[:-1]) + ", and " + clean_terms[-1] if len(clean_terms) > 1 else clean_terms[0]
        return f"The prediction is mainly influenced by respiratory distress indicators including **{features_str}**."
    else:
        features_str = ", ".join(clean_terms[:-1]) + ", and " + clean_terms[-1] if len(clean_terms) > 1 else clean_terms[0]
        return f"The prediction is mainly supported by stable clinical indicators including **{features_str}**."
