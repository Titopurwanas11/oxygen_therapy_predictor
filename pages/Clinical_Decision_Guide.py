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

st.set_page_config(page_title="Clinical Decision Guide — OxyPredict", page_icon="📘", layout="wide")
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
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 0.8rem 0; font-weight: 700;">🏥 1. Clinical Decision Support Overview</h3>
    <p style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin-bottom: 0.8rem;">
        <strong>OxyPredict</strong> is a Clinical Decision Support System (CDSS) designed to assist healthcare professionals in evaluating whether pediatric patients with Acute Respiratory Infections (ARI) and Pneumonia require supplementary oxygen therapy.
    </p>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.2rem;">
        <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #3b82f6;">
            <strong style="font-size: 0.85rem; color: #0a2e52; display: block; margin-bottom: 0.3rem;">🎯 System Purpose</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.4; display: block;">
                Provides an evidence-based second opinion on the necessity of oxygen support, aiding in resource optimization.
            </span>
        </div>
        <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #3b82f6;">
            <strong style="font-size: 0.85rem; color: #0a2e52; display: block; margin-bottom: 0.3rem;">👥 Intended Users</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.4; display: block;">
                Physicians, nurses, pediatricians, and clinical triage staff handling pediatric acute respiratory cases.
            </span>
        </div>
        <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #3b82f6;">
            <strong style="font-size: 0.85rem; color: #0a2e52; display: block; margin-bottom: 0.3rem;">🕒 When to Use</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.4; display: block;">
                Used during the initial clinical examination or triage in emergency departments and pediatric wards.
            </span>
        </div>
    </div>
    <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; text-align: center;">
        <strong style="font-size: 0.85rem; color: #0a2e52; display: block; margin-bottom: 0.6rem;">🔄 Clinical Decision Support Workflow</strong>
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.5rem; font-size: 0.8rem; font-weight: 600;">
            <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 6px; border: 1px solid #e2e8f0; color: #0a2e52;">Patient Clinical Data</span>
            <span style="color: #64748b;">➔</span>
            <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 6px; border: 1px solid #e2e8f0; color: #0a2e52;">AI Prediction</span>
            <span style="color: #64748b;">➔</span>
            <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 6px; border: 1px solid #e2e8f0; color: #0a2e52;">Risk Assessment</span>
            <span style="color: #64748b;">➔</span>
            <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 6px; border: 1px solid #e2e8f0; color: #0a2e52;">SHAP Feature Contribution</span>
            <span style="color: #64748b;">➔</span>
            <span style="background: white; padding: 0.4rem 0.8rem; border-radius: 6px; border: 1px solid #e2e8f0; color: #2563eb;">Clinical Decision</span>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 2: How to Use OxyPredict ───────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;">⚙️ 2. How to Use OxyPredict</h3>
    <div style="position: relative; padding-left: 1.5rem; border-left: 2px solid #e2e8f0;">
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #2563eb; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700;">1</div>
            <strong style="font-size: 0.88rem; color: #0f172a; display: block;">Dashboard Navigation</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.5;">Check the real-time operational overview, patient distributions, average model confidence, risk trends, and recent records.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #2563eb; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700;">2</div>
            <strong style="font-size: 0.88rem; color: #0f172a; display: block;">Single Prediction Submission</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.5;">Navigate to the 'Single Prediction' page, enter individual patient clinical parameters (age, vital signs, physical symptoms), and click predict.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #2563eb; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700;">3</div>
            <strong style="font-size: 0.88rem; color: #0f172a; display: block;">Interpret Prediction & Risk Assessment</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.5;">Review the diagnostic prediction, the confidence probability percentage, the risk level label, and read the dynamic <strong>AI Clinical Summary</strong>.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #2563eb; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700;">4</div>
            <strong style="font-size: 0.88rem; color: #0f172a; display: block;">Examine SHAP Explanations</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.5;">Look at the SHAP feature contribution charts to see exactly which physiological factors (like oxygen saturation, chest indrawing, etc.) drove the model's decision.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #2563eb; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700;">5</div>
            <strong style="font-size: 0.88rem; color: #0f172a; display: block;">Download Diagnostic Reports</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.5;">Export a clinical PDF report containing the prediction results, risk evaluation, clinical recommendations, and SHAP visualizations for patient files.</span>
        </div>
        <div style="position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #2563eb; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700;">6</div>
            <strong style="font-size: 0.88rem; color: #0f172a; display: block;">Perform Batch Predictions</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.5;">Upload structured files (Excel or CSV formats), execute model predictions for all patients concurrently, review results, and download consolidated reports.</span>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 3: Understanding Prediction Results ─────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;">📊 3. Understanding Prediction Results</h3>
    <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 0.8rem;">
        Each prediction returned by OxyPredict includes four main clinical variables:
    </p>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem;">
        <thead>
            <tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0; text-align: left;">
                <th style="padding: 0.6rem; color: #0a2e52; font-weight: 700;">Variable</th>
                <th style="padding: 0.6rem; color: #0a2e52; font-weight: 700;">Meaning</th>
                <th style="padding: 0.6rem; color: #0a2e52; font-weight: 700;">Clinical Application</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.6rem; font-weight: 700; color: #0f172a;">Prediction</td>
                <td style="padding: 0.6rem; color: #334155;">The model's binary classification: "Yes" (Needs Oxygen) or "No" (No Oxygen Needed).</td>
                <td style="padding: 0.6rem; color: #475569;">Acts as the primary clinical indicator for triage prioritization.</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.6rem; font-weight: 700; color: #0f172a;">Probability</td>
                <td style="padding: 0.6rem; color: #334155;">The model's internal probability of class membership (0% to 100% chance of needing oxygen).</td>
                <td style="padding: 0.6rem; color: #475569;">Helps determine patient severity (higher probability means a higher likelihood of acute distress).</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.6rem; font-weight: 700; color: #0f172a;">Confidence Level</td>
                <td style="padding: 0.6rem; color: #334155;">Model confidence based on distance to the decision threshold.</td>
                <td style="padding: 0.6rem; color: #475569;">High confidence indicates robust feature patterns, while low confidence suggests boundary cases.</td>
            </tr>
            <tr>
                <td style="padding: 0.6rem; font-weight: 700; color: #0f172a;">Risk Level</td>
                <td style="padding: 0.6rem; color: #334155;">The clinical risk categorization derived directly from prediction probability.</td>
                <td style="padding: 0.6rem; color: #475569;">Directs the recommended clinical action path (e.g., immediate evaluation vs. observation).</td>
            </tr>
        </tbody>
    </table>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 4: Risk Level Interpretation ───────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;">🔴 4. Risk Level Interpretation</h3>
    <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 0.8rem;">
        Risk levels prioritize patient monitoring and define action paths:
    </p>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem;">
        <thead>
            <tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0; text-align: left;">
                <th style="padding: 0.6rem; color: #0a2e52; font-weight: 700;">Risk Category</th>
                <th style="padding: 0.6rem; color: #0a2e52; font-weight: 700;">Probability Range</th>
                <th style="padding: 0.6rem; color: #0a2e52; font-weight: 700;">Patient Status</th>
                <th style="padding: 0.6rem; color: #0a2e52; font-weight: 700;">Clinical Recommendation</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.6rem; font-weight: 700;"><span style="color: #16a34a; background: #f0fdf4; padding: 0.2rem 0.5rem; border-radius: 4px;">🟢 Low Risk</span></td>
                <td style="padding: 0.6rem; color: #334155;">&lt; 50%</td>
                <td style="padding: 0.6rem; color: #334155;">Physiologically stable; low indicators of severe respiratory failure.</td>
                <td style="padding: 0.6rem; color: #475569; font-weight: 600;">Routine observation recommended.</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.6rem; font-weight: 700;"><span style="color: #ca8a04; background: #fffbeb; padding: 0.2rem 0.5rem; border-radius: 4px;">🟡 Moderate Risk</span></td>
                <td style="padding: 0.6rem; color: #334155;">50% – 70%</td>
                <td style="padding: 0.6rem; color: #334155;">Mild respiratory distress; presence of minor boundary indicators.</td>
                <td style="padding: 0.6rem; color: #475569; font-weight: 600;">Close telemetry monitoring and re-evaluation.</td>
            </tr>
            <tr>
                <td style="padding: 0.6rem; font-weight: 700;"><span style="color: #dc2626; background: #fdf2f2; padding: 0.2rem 0.5rem; border-radius: 4px;">🔴 High Risk</span></td>
                <td style="padding: 0.6rem; color: #334155;">&gt; 70%</td>
                <td style="padding: 0.6rem; color: #334155;">Severe respiratory distress; critical indicators of hypoxia.</td>
                <td style="padding: 0.6rem; color: #475569; font-weight: 600; color: #dc2626;">Immediate clinical evaluation and oxygen support assessment.</td>
            </tr>
        </tbody>
    </table>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 5: Confidence Level Interpretation ─────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;">⚡ 5. Model Confidence Level</h3>
    <p style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin-bottom: 1rem;">
        The confidence level indicates how strongly the features match patterns from past patient cases.
    </p>
    <div style="margin-bottom: 0.8rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
            <strong style="color: #16a34a;">Very High Confidence (e.g. 95%)</strong>
            <span style="color: #64748b;">Highly defined clinical parameters; clear diagnostic path.</span>
        </div>
        <div style="background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: #16a34a; width: 95%; height: 100%;"></div>
        </div>
    </div>
    <div style="margin-bottom: 0.8rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
            <strong style="color: #ca8a04;">Moderate Confidence (e.g. 75%)</strong>
            <span style="color: #64748b;">Some conflicting features; patient is near decision boundary.</span>
        </div>
        <div style="background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: #ca8a04; width: 75%; height: 100%;"></div>
        </div>
    </div>
    <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
            <strong style="color: #dc2626;">Low Confidence (e.g. 55%)</strong>
            <span style="color: #64748b;">Substantial physiological variation; interpret result with high caution.</span>
        </div>
        <div style="background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: #dc2626; width: 55%; height: 100%;"></div>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 6: Understanding SHAP Explanation ──────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;">🔬 6. Understanding SHAP Explanations</h3>
    <p style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin-bottom: 0.8rem;">
        <strong>SHAP (SHapley Additive exPlanations)</strong> is an explainable AI framework that breaks down the prediction by showing the contribution of each patient parameter:
    </p>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem;">
        <div style="background: #fdf2f2; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #dc2626;">
            <strong style="color: #dc2626; font-size: 0.85rem; display: block; margin-bottom: 0.3rem;">🔴 Positive Contributions (Increases Risk)</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.4; display: block;">
                Factors with red SHAP bars push the patient's likelihood <strong>toward</strong> requiring oxygen therapy (e.g., extremely low SpO2, severe chest indrawing, tachypnea).
            </span>
        </div>
        <div style="background: #f0fdf4; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #16a34a;">
            <strong style="color: #16a34a; font-size: 0.85rem; display: block; margin-bottom: 0.3rem;">🟢 Negative Contributions (Decreases Risk)</strong>
            <span style="font-size: 0.8rem; color: #475569; line-height: 1.4; display: block;">
                Factors with blue SHAP bars pull the patient's likelihood <strong>away</strong> from oxygen therapy, indicating stable parameters (e.g., normal SpO2, normal heart rate, no chest retractions).
            </span>
        </div>
    </div>
    <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px;">
        <strong style="font-size: 0.85rem; color: #0a2e52; display: block; margin-bottom: 0.3rem;">📋 AI Clinical Summary Narrative</strong>
        <span style="font-size: 0.8rem; color: #475569; line-height: 1.5; display: block;">
            The diagnostic interface generates a clinical summary based directly on the leading SHAP features. This allows medical practitioners to quickly review the primary physiological drivers behind the assessment.
        </span>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 7: Batch Prediction Guide ───────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;">📂 7. Batch Prediction Guide</h3>
    <p style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin-bottom: 0.8rem;">
        The Batch Prediction page allows processing multi-patient datasets simultaneously:
    </p>
    <ul style="font-size: 0.8rem; color: #475569; margin: 0 0 1rem 1.2rem; padding: 0; line-height: 1.6;">
        <li style="margin-bottom: 0.4rem;"><strong>File Upload:</strong> Upload an Excel (.xlsx) or CSV (.csv) file containing pediatric patient vital signs.</li>
        <li style="margin-bottom: 0.4rem;"><strong>Column Validation:</strong> The system automatically verifies that all 44 clinical features required by the Random Forest model are present.</li>
        <li style="margin-bottom: 0.4rem;"><strong>Batch Diagnostic Run:</strong> Click run to calculate the predictions, confidence percentages, and clinical risk levels for the entire cohort.</li>
        <li style="margin-bottom: 0.4rem;"><strong>Cohort Summary:</strong> View distribution statistics, average variables, and group-level risks.</li>
        <li><strong>Consolidated Export:</strong> Download the complete diagnostic cohort results as an Excel sheet or generate patient reports.</li>
    </ul>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 8: Clinical Recommendations ─────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 1.05rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;">💡 8. Clinical Recommendations</h3>
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 1rem;">
        <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 0.82rem; color: #166534; line-height: 1.6;">
            <li style="margin-bottom: 0.6rem;">✔️ <strong>Clinical Support Only:</strong> Use OxyPredict prediction results as a supportive reference tool alongside standard pediatric guidelines.</li>
            <li style="margin-bottom: 0.6rem;">✔️ <strong>Comprehensive Evaluation:</strong> Always combine model outputs with thorough bedside clinical examinations, physical assessments, and patient medical history.</li>
            <li style="margin-bottom: 0.6rem;">✔️ <strong>Patient-centric View:</strong> Evaluate the patient holistically; do not base critical clinical decisions solely on numerical probability scores.</li>
            <li>✔️ <strong>Risk Factor Prioritization:</strong> Closely review the leading positive (risk-increasing) physiological factors identified by the SHAP visual chart.</li>
        </ul>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 9: System Limitations ──────────────────────────────────────────
st_html("""
<div class="cdss-card" style="border: 1px solid #fef3c7; background: #fffbeb;">
    <h3 style="font-size: 1.05rem; color: #b45309; margin: 0 0 0.8rem 0; font-weight: 700;">⚠️ 9. System Limitations & Disclaimers</h3>
    <ul style="font-size: 0.82rem; color: #78350f; line-height: 1.6; margin: 0 0 0 1.2rem; padding: 0;">
        <li style="margin-bottom: 0.4rem;"><strong>Model Development Scope:</strong> The predictive engine was developed and validated using a dataset of 801 pediatric acute respiratory cases.</li>
        <li style="margin-bottom: 0.4rem;"><strong>Population Limitation:</strong> The clinical validity is calibrated for the specific pediatric demographic parameters defined in the clinical study.</li>
        <li style="margin-bottom: 0.4rem;"><strong>CDSS Notice:</strong> OxyPredict does not replace, override, or supersede the professional clinical judgment of a licensed medical practitioner.</li>
        <li><strong>Bedside Priority:</strong> Final diagnostic decisions and clinical treatment paths must always prioritize bedside symptoms and established medical protocols.</li>
    </ul>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 10: Frequently Asked Questions ──────────────────────────────────
st_html("""
<h3 class="section-title-custom">❓ 10. Frequently Asked Questions</h3>
""")

with st.expander("❓ What does prediction probability represent?"):
    st.write(
        "Prediction probability is the model's internal score representing the statistical chance that the "
        "patient belongs to the group requiring supplementary oxygen therapy. A higher score implies a higher probability."
    )

with st.expander("❓ What is the difference between confidence level and risk level?"):
    st.write(
        "Prediction probability determines the Risk Level (how severe the distress is). Confidence Level indicates how "
        "strongly the patient's data features match the patterns learned during training (how confident the model is in its decision)."
    )

with st.expander("❓ Why is SHAP explainability important for clinicians?"):
    st.write(
        "SHAP makes the machine learning model transparent. Instead of a black box prediction, it shows exactly "
        "how much weight each symptom (like breathing rate or chest retraction) had in driving the decision."
    )

with st.expander("❓ Why do predictions differ for patients with similar SpO2 levels?"):
    st.write(
        "OxyPredict processes 44 physiological features simultaneously. Additional variables like chest indrawing, "
        "heart rate, age, and temperature interact to form a holistic picture, which may shift the output."
    )

with st.expander("❓ Can the model generate an incorrect prediction?"):
    st.write(
        "Yes, like all statistical systems, predictive errors can occur. This is why the CDSS is classified as a "
        "second opinion tool and should always be paired with physician assessment."
    )

st.markdown("")

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
