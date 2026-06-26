"""
System Health Monitoring logic for OxyPredict.
"""

from utils.prediction import load_model

def get_system_health_status() -> dict:
    """
    Check availability of key sub-services.

    Returns:
        dict: Service Name -> 'Healthy' or 'Error'
    """
    # 1. Model Loaded Check
    try:
        model = load_model()
        model_ok = model is not None
    except Exception:
        model_ok = False

    # 2. SHAP Loaded Check
    try:
        from utils.shap_utils import compute_shap_for_patient
        shap_ok = True
    except Exception:
        shap_ok = False

    # 3. PDF Generator Check
    try:
        from utils.pdf_report import generate_pdf_report
        from utils.batch_report import generate_batch_pdf_report
        pdf_ok = True
    except Exception:
        pdf_ok = False

    # 4. Upload & Predict Services
    pred_ok = model_ok
    upload_ok = True # Standard pandas reader is built-in

    return {
        "Model Loaded": "Healthy" if model_ok else "Error",
        "SHAP Loaded": "Healthy" if shap_ok else "Error",
        "Prediction Service": "Healthy" if pred_ok else "Error",
        "Upload Service": "Healthy" if upload_ok else "Error",
        "PDF Generator": "Healthy" if pdf_ok else "Error"
    }
