"""
Clinical Decision Guide Page — Integrated Clinical and User Guide for OxyPredict.
"""

import streamlit as st
from utils.config import (
    setup_page,
    render_page_header,
    render_section_divider,
    render_footer,
)

st.set_page_config(page_title="Clinical Decision Guide — OxyPredict", page_icon="assets/favicon-64x64.png", layout="wide")
setup_page("Clinical Decision Guide — OxyPredict")

# Helper function to render HTML safely
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)

# ─── SECTION 1: HEADER ───────────────────────────────────────────────────────
st_html(render_page_header(
    "📘",
    "Clinical Decision Guide",
    "Comprehensive guide for clinical implementation, results interpretation, and operation of OxyPredict."
))

# ─── SECTION 1: Clinical Decision Support Overview ──────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 0.8rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">local_hospital</span>
        1. Clinical Decision Support Overview
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1rem;">
        <strong>OxyPredict</strong> is a Clinical Decision Support System (CDSS) designed to assist healthcare professionals in evaluating whether pediatric patients with Acute Respiratory Infections (ARI) and Pneumonia require supplementary oxygen therapy.
    </p>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.2rem;">
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 3px solid #3282B8;">
            <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #2563EB;">target</span>
                System Purpose
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Provides an evidence-based second opinion on the necessity of oxygen support, aiding in resource optimization.
            </span>
        </div>
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 3px solid #3282B8;">
            <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #2563EB;">group</span>
                Intended Users
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Physicians, nurses, pediatricians, and clinical triage staff handling pediatric acute respiratory cases.
            </span>
        </div>
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 3px solid #3282B8;">
            <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #2563EB;">schedule</span>
                When to Use
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Used during the initial clinical examination or triage in emergency departments and pediatric wards.
            </span>
        </div>
    </div>
    <div style="background: #F8FAFC; padding: 1.2rem; border-radius: 10px; text-align: center;">
        <strong style="font-size: 13px; color: #64748B; display: block; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">Clinical Decision Support Workflow</strong>
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.5rem; font-size: 14px; font-weight: 600;">
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">Patient Clinical Data</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">AI Prediction</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">Risk Assessment</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">SHAP Feature Contribution</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: #DBEAFE; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #93C5FD; color: #1D4ED8; font-weight: 700;">Clinical Decision</span>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 2: How to Use OxyPredict ───────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">settings</span>
        2. How to Use OxyPredict
    </h3>
    <div style="position: relative; padding-left: 1.5rem; border-left: 2px solid #D8E2EC;">
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">1</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Dashboard Navigation</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Check the real-time operational overview, patient distributions, average model confidence, risk trends, and recent records.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">2</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Single Prediction Submission</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Navigate to the 'Single Prediction' page, enter individual patient clinical parameters (age, vital signs, physical symptoms), and click predict.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">3</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Interpret Prediction & Risk Assessment</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Review the diagnostic prediction, the confidence probability percentage, the risk level label, and read the dynamic <strong>AI Clinical Summary</strong>.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">4</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Examine SHAP Explanations</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Look at the SHAP feature contribution charts to see exactly which physiological factors (like oxygen saturation, chest indrawing, etc.) drove the model's decision.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">5</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Download Diagnostic Reports</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Export a clinical PDF report containing the prediction results, risk evaluation, clinical recommendations, and SHAP visualizations for patient files.</span>
        </div>
        <div style="position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">6</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Perform Batch Predictions</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Upload structured files (Excel or CSV formats), execute model predictions for all patients concurrently, review results, and download consolidated reports.</span>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 3: Understanding Prediction Results ─────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">analytics</span>
        3. Understanding Prediction Results
    </h3>
    <p style="font-size: 14px; color: #64748B; margin-bottom: 1rem;">
        Each prediction returned by OxyPredict includes four main clinical variables:
    </p>
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 1px solid #D6E4F0; border-radius: 16px; overflow: hidden;">
        <thead>
            <tr style="background-color: #0F4C75; border-bottom: 2px solid #3282B8;">
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Variable</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Meaning</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Clinical Application</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #EAF2F8;">
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Prediction</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">The model's binary classification: "Yes" (Needs Oxygen) or "No" (No Oxygen Needed).</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">Acts as the primary clinical indicator for triage prioritization.</td>
            </tr>
            <tr style="border-bottom: 1px solid #EAF2F8; background-color: #F8FAFC;">
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Probability</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">The model's internal probability of class membership (0% to 100% chance of needing oxygen).</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">Helps determine patient severity (higher probability means a higher likelihood of acute distress).</td>
            </tr>
            <tr style="border-bottom: 1px solid #EAF2F8;">
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Confidence Level</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Model confidence based on distance to the decision threshold.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">High confidence indicates robust feature patterns, while low confidence suggests boundary cases.</td>
            </tr>
            <tr>
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Risk Level</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">The clinical risk categorization derived directly from prediction probability.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">Directs the recommended clinical action path (e.g., immediate evaluation vs. observation).</td>
            </tr>
        </tbody>
    </table>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 4: Risk Level Interpretation ───────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #EF4444;">warning</span>
        4. Risk Level Interpretation
    </h3>
    <p style="font-size: 14px; color: #64748B; margin-bottom: 1rem;">
        Risk levels prioritize patient monitoring and define action paths:
    </p>
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 1px solid #D6E4F0; border-radius: 16px; overflow: hidden;">
        <thead>
            <tr style="background-color: #0F4C75; border-bottom: 2px solid #3282B8;">
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Risk Category</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Probability Range</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Patient Status</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Clinical Recommendation</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #EAF2F8;">
                <td style="padding: 0.7rem 1rem; font-weight: 600;"><span class="cdss-badge cdss-badge-success">Low Risk</span></td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">&lt; 50%</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Physiologically stable; low indicators of severe respiratory failure.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B; font-weight: 600;">Routine observation recommended.</td>
            </tr>
            <tr style="border-bottom: 1px solid #EAF2F8; background-color: #F8FAFC;">
                <td style="padding: 0.7rem 1rem; font-weight: 600;"><span class="cdss-badge cdss-badge-warning">Moderate Risk</span></td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">50% – 70%</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Mild respiratory distress; presence of minor boundary indicators.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B; font-weight: 600;">Close telemetry monitoring and re-evaluation.</td>
            </tr>
            <tr>
                <td style="padding: 0.7rem 1rem; font-weight: 600;"><span class="cdss-badge cdss-badge-danger">High Risk</span></td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">&gt; 70%</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Severe respiratory distress; critical indicators of hypoxia.</td>
                <td style="padding: 0.7rem 1rem; color: #EF4444; font-weight: 600;">Immediate clinical evaluation and oxygen support assessment.</td>
            </tr>
        </tbody>
    </table>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 5: Confidence Level Interpretation ─────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">speed</span>
        5. Model Confidence Level
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1.2rem;">
        The confidence level indicates how strongly the features match patterns from past patient cases.
    </p>
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 0.3rem;">
            <strong style="color: #15803D;">Very High Confidence (e.g. 95%)</strong>
            <span style="color: #64748B;">Highly defined clinical parameters; clear diagnostic path.</span>
        </div>
        <div style="background: #E2F0F9; height: 8px; border-radius: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #15803D, #22C55E); width: 95%; height: 100%; border-radius: 6px;"></div>
        </div>
    </div>
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 0.3rem;">
            <strong style="color: #B45309;">Moderate Confidence (e.g. 75%)</strong>
            <span style="color: #64748B;">Some conflicting features; patient is near decision boundary.</span>
        </div>
        <div style="background: #E2F0F9; height: 8px; border-radius: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #B45309, #F59E0B); width: 75%; height: 100%; border-radius: 6px;"></div>
        </div>
    </div>
    <div>
        <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 0.3rem;">
            <strong style="color: #B91C1C;">Low Confidence (e.g. 55%)</strong>
            <span style="color: #64748B;">Substantial physiological variation; interpret result with high caution.</span>
        </div>
        <div style="background: #E2F0F9; height: 8px; border-radius: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #B91C1C, #EF4444); width: 55%; height: 100%; border-radius: 6px;"></div>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 6: Understanding SHAP Explanation ──────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">biotech</span>
        6. Understanding SHAP Explanations
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1rem;">
        <strong>SHAP (SHapley Additive exPlanations)</strong> is an explainable AI framework that breaks down the prediction by showing the contribution of each patient parameter:
    </p>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem;">
        <div style="background: #FEF2F2; padding: 1rem; border-radius: 10px; border-left: 3px solid #EF4444;">
            <strong style="color: #EF4444; font-size: 15px; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle;">trending_up</span>
                Positive Contributions (Increases Risk)
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Factors with red SHAP bars push the patient's likelihood <strong>toward</strong> requiring oxygen therapy (e.g., extremely low SpO2, severe chest indrawing, tachypnea).
            </span>
        </div>
        <div style="background: #ECFDF5; padding: 1rem; border-radius: 10px; border-left: 3px solid #22C55E;">
            <strong style="color: #15803D; font-size: 15px; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle;">trending_down</span>
                Negative Contributions (Decreases Risk)
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Factors with blue SHAP bars pull the patient's likelihood <strong>away</strong> from oxygen therapy, indicating stable parameters (e.g., normal SpO2, normal heart rate, no chest retractions).
            </span>
        </div>
    </div>
    <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px;">
        <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
            <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #3282B8;">description</span>
            AI Clinical Summary Narrative
        </strong>
        <span style="font-size: 14px; color: #64748B; line-height: 1.6; display: block;">
            The diagnostic interface generates a clinical summary based directly on the leading SHAP features. This allows medical practitioners to quickly review the primary physiological drivers behind the assessment.
        </span>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 7: Batch Prediction Guide ───────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">folder_open</span>
        7. Batch Prediction Guide
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1rem;">
        The Batch Prediction page allows processing multi-patient datasets simultaneously:
    </p>
    <ul style="font-size: 15px; color: #64748B; margin: 0 0 1rem 1.2rem; padding: 0; line-height: 1.7;">
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">File Upload:</strong> Upload an Excel (.xlsx) or CSV (.csv) file containing pediatric patient vital signs.</li>
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">Column Validation:</strong> The system automatically verifies that all 44 clinical features required by the Random Forest model are present.</li>
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">Batch Diagnostic Run:</strong> Click run to calculate the predictions, confidence percentages, and clinical risk levels for the entire cohort.</li>
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">Cohort Summary:</strong> View distribution statistics, average variables, and group-level risks.</li>
        <li><strong style="color: #1E293B;">Consolidated Export:</strong> Download the complete diagnostic cohort results as an Excel sheet or generate patient reports.</li>
    </ul>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 8: Clinical Recommendations ─────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #15803D;">lightbulb</span>
        8. Clinical Recommendations
    </h3>
    <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1.2rem;">
        <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 15px; color: #15803D; line-height: 1.7;">
            <li style="margin-bottom: 0.6rem;">✓ <strong>Clinical Support Only:</strong> Use OxyPredict prediction results as a supportive reference tool alongside standard pediatric guidelines.</li>
            <li style="margin-bottom: 0.6rem;">✓ <strong>Comprehensive Evaluation:</strong> Always combine model outputs with thorough bedside clinical examinations, physical assessments, and patient medical history.</li>
            <li style="margin-bottom: 0.6rem;">✓ <strong>Patient-centric View:</strong> Evaluate the patient holistically; do not base critical clinical decisions solely on numerical probability scores.</li>
            <li>✓ <strong>Risk Factor Prioritization:</strong> Closely review the leading positive (risk-increasing) physiological factors identified by the SHAP visual chart.</li>
        </ul>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 9: System Limitations ──────────────────────────────────────────
st_html("""
<div class="cdss-card" style="border-left: 5px solid #F59E0B;">
    <h3 style="font-size: 22px; color: #F59E0B; margin: 0 0 0.8rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #F59E0B;">gpp_maybe</span>
        9. System Limitations & Disclaimers
    </h3>
    <ul style="font-size: 15px; color: #B45309; line-height: 1.7; margin: 0 0 0 1.2rem; padding: 0;">
        <li style="margin-bottom: 0.5rem;"><strong>Model Development Scope:</strong> The predictive engine was developed and validated using a dataset of 801 pediatric acute respiratory cases.</li>
        <li style="margin-bottom: 0.5rem;"><strong>Population Limitation:</strong> The clinical validity is calibrated for the specific pediatric demographic parameters defined in the clinical study.</li>
        <li style="margin-bottom: 0.5rem;"><strong>CDSS Notice:</strong> OxyPredict does not replace, override, or supersede the professional clinical judgment of a licensed medical practitioner.</li>
        <li><strong>Bedside Priority:</strong> Final diagnostic decisions and clinical treatment paths must always prioritize bedside symptoms and established medical protocols.</li>
    </ul>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 10: Frequently Asked Questions ──────────────────────────────────
st_html("""
<h3 class="section-title-custom" style="display: flex; align-items: center; gap: 0.5rem;">
    <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">help</span>
    10. Frequently Asked Questions
</h3>
""")

with st.expander("What does prediction probability represent?"):
    st.write(
        "Prediction probability is the model's internal score representing the statistical chance that the "
        "patient belongs to the group requiring supplementary oxygen therapy. A higher score implies a higher probability."
    )

with st.expander("What is the difference between confidence level and risk level?"):
    st.write(
        "Prediction probability determines the Risk Level (how severe the distress is). Confidence Level indicates how "
        "strongly the patient's data features match the patterns learned during training (how confident the model is in its decision)."
    )

with st.expander("Why is SHAP explainability important for clinicians?"):
    st.write(
        "SHAP makes the machine learning model transparent. Instead of a black box prediction, it shows exactly "
        "how much weight each symptom (like breathing rate or chest retraction) had in driving the decision."
    )

with st.expander("Why do predictions differ for patients with similar SpO2 levels?"):
    st.write(
        "OxyPredict processes 44 physiological features simultaneously. Additional variables like chest indrawing, "
        "heart rate, age, and temperature interact to form a holistic picture, which may shift the output."
    )

with st.expander("Can the model generate an incorrect prediction?"):
    st.write(
        "Yes, like all statistical systems, predictive errors can occur. This is why the CDSS is classified as a "
        "second opinion tool and should always be paired with physician assessment."
    )

st.markdown("")

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
