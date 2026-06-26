"""
OxyPredict Rule-Based Clinical Recommendation Engine.
This module processes model outputs and key clinical features to generate recommendations.
"""

def get_base_rule_recommendation(prediction: str, prob_pct: float, risk_normalized: str) -> tuple:
    """Helper to evaluate the base priority, actions, and monitoring guidelines."""
    priority = "Low"
    actions = []
    monitoring = []
    
    # RULE 1: Prediction Yes, Risk High (or Very High), Prob >= 90%
    if (prediction == "Yes" or prediction == 1) and "high" in risk_normalized and prob_pct >= 90.0:
        priority = "Emergency"
        actions = [
            "Immediate oxygen therapy assessment is recommended.",
            "Evaluate respiratory distress immediately.",
            "Prepare oxygen support according to hospital protocol.",
            "Urgent pediatric evaluation is recommended."
        ]
        monitoring = ["Continuous pulse oximetry monitoring."]
        
    # RULE 2: Prediction Yes, Risk High (or Very High), Prob 80-89%
    elif (prediction == "Yes" or prediction == 1) and "high" in risk_normalized and prob_pct >= 80.0:
        priority = "High"
        actions = [
            "Consider oxygen therapy.",
            "Repeat SpO₂ measurement.",
            "Reassess within 30 minutes."
        ]
        monitoring = ["Monitor respiratory status closely."]
        
    # RULE 3: Prediction Yes, Risk Medium (or Moderate)
    elif (prediction == "Yes" or prediction == 1) and ("medium" in risk_normalized or "moderate" in risk_normalized):
        priority = "Medium"
        actions = [
            "Patient should be closely observed.",
            "Repeat clinical assessment if symptoms worsen."
        ]
        monitoring = ["Monitor oxygen saturation regularly."]
        
    # RULE 4: Prediction No, Risk Low (or Low-Moderate)
    elif (prediction == "No" or prediction == 0) and "low" in risk_normalized:
        priority = "Low"
        actions = [
            "Routine observation is recommended.",
            "Continue standard supportive treatment.",
            "Repeat assessment if new respiratory symptoms appear."
        ]
        monitoring = ["Routine vital sign checks."]
        
    # Fallbacks for other states (e.g. Risk is Low but prediction is Yes, or Risk is Medium but prediction is No)
    elif prediction == "Yes" or prediction == 1:
        priority = "Medium"
        actions = [
            "Patient should be closely observed.",
            "Reassess respiratory status if conditions change."
        ]
        monitoring = ["Monitor oxygen saturation regularly."]
    else:
        priority = "Low"
        actions = [
            "Routine observation is recommended.",
            "Re-evaluate if new respiratory symptoms develop."
        ]
        monitoring = ["Standard nursing monitoring guidelines."]
        
    return priority, actions, monitoring


def apply_feature_rules(patient_data: dict, actions: list, monitoring: list) -> tuple:
    """Helper to apply feature-specific clinical guidelines and append actions/monitoring."""
    # Normalize keys to lowercase to support variations in CSV headers or user inputs
    pat_lower = {k.lower(): v for k, v in patient_data.items()}
    
    # 1. SaO2 < 90
    sao2_keys = ["oxygen saturation (sao2) at admission", "oxygen saturation", "sao2", "sao2_admission"]
    sao2_val = None
    for k in sao2_keys:
        if k in pat_lower:
            sao2_val = pat_lower[k]
            break
            
    if sao2_val is not None:
        try:
            if float(sao2_val) < 90.0:
                actions.append("Severe oxygen desaturation detected. Immediate oxygen assessment is advised.")
        except (ValueError, TypeError):
            pass

    # 2. Respiratory rate (Tachypnea / High RR)
    rr_keys = ["respiratory rate", "rr"]
    rr_val = None
    for k in rr_keys:
        if k in pat_lower:
            rr_val = pat_lower[k]
            break
            
    if rr_val is not None:
        try:
            if float(rr_val) > 40.0:
                monitoring.append("Monitor respiratory effort closely.")
        except (ValueError, TypeError):
            pass

    # 3. Cyanosis
    cyanosis_val = pat_lower.get("cyanosis")
    if cyanosis_val == "Yes" or cyanosis_val == 1 or str(cyanosis_val).lower() == "yes":
        actions.append("Signs of severe hypoxemia may be present.")

    # 4. Nasal Flaring
    flaring_val = pat_lower.get("nasal flaring")
    if flaring_val == "Yes" or flaring_val == 1 or str(flaring_val).lower() == "yes":
        actions.append("Evidence of increased work of breathing.")

    # 5. Wheezing
    wheezing_val = pat_lower.get("wheezing")
    if wheezing_val == "Yes" or wheezing_val == 1 or str(wheezing_val).lower() == "yes":
        actions.append("Consider airway evaluation.")

    # 6. Crackles
    crackles_val = pat_lower.get("crackles")
    if crackles_val == "Yes" or crackles_val == 1 or str(crackles_val).lower() == "yes":
        actions.append("Consider lower respiratory tract involvement.")

    # 7. Unusual Sleepiness
    sleepiness_val = pat_lower.get("unusual sleepiness")
    if sleepiness_val == "Yes" or sleepiness_val == 1 or str(sleepiness_val).lower() == "yes":
        monitoring.append("Neurological status should be monitored.")

    # 8. Heart rate
    hr_keys = ["heart rate", "hr"]
    hr_val = None
    for k in hr_keys:
        if k in pat_lower:
            hr_val = pat_lower[k]
            break
            
    if hr_val is not None:
        try:
            if float(hr_val) > 140.0:
                monitoring.append("Continuous vital sign monitoring is recommended.")
        except (ValueError, TypeError):
            pass
            
    return actions, monitoring


def generate_recommendation(
    prediction: str,
    probability: float,
    risk_level: str,
    confidence_level: str,
    patient_data: dict,
    top_shap_features: list = None
) -> dict:
    """
    Generate structured rule-based clinical recommendations based on prediction outputs and patient features.
    
    Returns:
        dict: containing clinical_action (list), priority (str), monitoring (list), notes (list), recommendation_level (str)
    """
    # 1. Normalize probability to a percentage scale (0 - 100)
    prob_pct = probability if probability > 1.0 else probability * 100.0
    
    # 2. Normalize risk level string
    risk_normalized = risk_level.lower()
    
    # 3. Apply base rules (Rule 1-4)
    priority, actions, monitoring = get_base_rule_recommendation(prediction, prob_pct, risk_normalized)
    
    # 4. Apply feature-specific rules
    actions, monitoring = apply_feature_rules(patient_data, actions, monitoring)
    
    # Deduplicate actions and monitoring list while keeping order
    actions_uniq = list(dict.fromkeys(actions))
    monitoring_uniq = list(dict.fromkeys(monitoring))
    
    # Notes block as specified in instructions
    notes = [
        "Clinical recommendations are generated using predefined clinical decision rules based on the model prediction and selected patient characteristics.",
        "These recommendations are intended to support healthcare professionals and should not replace physician judgment, institutional protocols, or clinical examination."
    ]
    
    return {
        "priority": priority,
        "clinical_action": actions_uniq,
        "monitoring": monitoring_uniq,
        "notes": notes,
        "recommendation_level": "AI-assisted Rule-Based Recommendation"
    }
