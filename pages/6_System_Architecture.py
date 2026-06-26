"""
System Architecture Page — Visual documentation of the OxyPredict CDSS architecture.
Illustrates the complete workflow from data input to clinical decision support.
"""

import streamlit as st
from utils.config import (
    MODEL_NAME,
    MODEL_N_ESTIMATORS,
    MODEL_ACCURACY,
    MODEL_F1_MACRO,
    MODEL_ROC_AUC,
    DATASET_TOTAL_SAMPLES,
    DATASET_SELECTED_FEATURES,
    setup_page,
)

st.set_page_config(page_title="System Architecture — OxyPredict", page_icon="🫁", layout="wide")
setup_page("System Architecture — OxyPredict")


# Helper — strip leading whitespace to prevent Streamlit markdown code-block rendering
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


# =============================================================================
# Custom CSS
# =============================================================================
st_html("""
<style>
/* ── Typography ────────────────────────────────────── */
.arch-page { font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

/* ── Hero Header ───────────────────────────────────── */
.arch-hero {
    background: linear-gradient(135deg, #0a2e52 0%, #153e75 40%, #1e40af 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem 2.5rem;
    text-align: center;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(10, 46, 82, 0.25);
}
.arch-hero::before {
    content: "";
    position: absolute;
    top: -60%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.arch-hero::after {
    content: "";
    position: absolute;
    bottom: -50%;
    left: -15%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(96,165,250,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.arch-hero-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
    animation: heroFloat 3s ease-in-out infinite;
}
@keyframes heroFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
.arch-hero h1 {
    margin: 0;
    font-size: 2.3rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.arch-hero .hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.2);
    border: 1px solid rgba(147,197,253,0.3);
    border-radius: 50px;
    padding: 0.35rem 1.2rem;
    font-size: 0.8rem;
    color: #93c5fd;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 0.75rem;
    position: relative;
    z-index: 1;
}
.arch-hero .hero-sub {
    margin: 1rem auto 0 auto;
    max-width: 680px;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}

/* ── Section Header ────────────────────────────────── */
.arch-section-hdr {
    color: #0a2e52;
    font-weight: 700;
    font-size: 1.3rem;
    margin-top: 2rem;
    margin-bottom: 1.25rem;
    border-left: 4px solid #2563eb;
    padding-left: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Overview Cards ────────────────────────────────── */
.overview-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
@media (max-width: 800px) {
    .overview-grid { grid-template-columns: repeat(2, 1fr); }
}
.ov-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}
.ov-card::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}
.ov-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px -4px rgba(10,46,82,0.1);
    border-color: #93c5fd;
}
.ov-card:hover::after { transform: scaleX(1); }
.ov-card-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-radius: 14px;
}
.ov-card-label {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.2rem;
}
.ov-card-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
}

/* ── Main Architecture Diagram ─────────────────────── */
.arch-diagram-wrap {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 2rem;
}
.arch-flow {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
}
.arch-node {
    width: 340px;
    max-width: 90%;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    transition: all 0.3s ease;
    position: relative;
}
.arch-node:hover {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-color: #93c5fd;
    transform: scale(1.03);
    box-shadow: 0 8px 24px rgba(37,99,235,0.08);
}
.arch-node-icon {
    font-size: 1.5rem;
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(37,99,235,0.2);
}
.arch-node-label {
    font-size: 0.92rem;
    font-weight: 600;
    color: #0f172a;
}
.arch-node-desc {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.15rem;
}
.arch-arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 36px;
    justify-content: center;
    position: relative;
}
.arch-arrow-line {
    width: 2px;
    height: 20px;
    background: linear-gradient(180deg, #93c5fd 0%, #2563eb 100%);
}
.arch-arrow-head {
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #2563eb;
}

/* ── Preprocessing Pipeline ────────────────────────── */
.preproc-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 2rem;
}
.preproc-steps {
    display: flex;
    flex-direction: column;
    gap: 0;
    align-items: center;
}
.preproc-step {
    width: 420px;
    max-width: 95%;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    transition: all 0.25s ease;
}
.preproc-step:hover {
    border-color: #93c5fd;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    transform: translateX(6px);
    box-shadow: 0 4px 16px rgba(37,99,235,0.06);
}
.preproc-step-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #0a2e52;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.preproc-step-desc {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
    line-height: 1.5;
}
.preproc-arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 28px;
    justify-content: center;
}
.preproc-arrow-line {
    width: 2px;
    height: 14px;
    background: linear-gradient(180deg, #cbd5e1, #94a3b8);
}
.preproc-arrow-head {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #94a3b8;
}

/* ── ML Timeline ───────────────────────────────────── */
.ml-timeline-wrap {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 2rem;
    overflow-x: auto;
}
.ml-timeline {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 0;
    min-width: 700px;
    padding: 1rem 0;
}
.ml-tl-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    width: 110px;
    flex-shrink: 0;
    position: relative;
}
.ml-tl-dot {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    position: relative;
    z-index: 2;
    box-shadow: 0 4px 12px rgba(37,99,235,0.15);
    transition: all 0.3s ease;
}
.ml-tl-step:hover .ml-tl-dot {
    transform: scale(1.15);
    box-shadow: 0 6px 20px rgba(37,99,235,0.25);
}
.ml-tl-dot-blue { background: linear-gradient(135deg, #2563eb, #1d4ed8); }
.ml-tl-dot-green { background: linear-gradient(135deg, #059669, #047857); }
.ml-tl-dot-amber { background: linear-gradient(135deg, #d97706, #b45309); }
.ml-tl-dot-purple { background: linear-gradient(135deg, #7c3aed, #6d28d9); }
.ml-tl-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0.7rem;
    line-height: 1.3;
}
.ml-tl-sublabel {
    font-size: 0.65rem;
    color: #94a3b8;
    margin-top: 0.15rem;
}
.ml-tl-connector {
    display: flex;
    align-items: center;
    flex-shrink: 0;
    height: 44px;
    position: relative;
}
.ml-tl-connector-line {
    width: 28px;
    height: 2px;
    background: linear-gradient(90deg, #93c5fd, #2563eb);
}
.ml-tl-connector-arr {
    width: 0; height: 0;
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
    border-left: 7px solid #2563eb;
}

/* ── Tech Stack Grid ───────────────────────────────── */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
@media (max-width: 800px) { .tech-grid { grid-template-columns: repeat(2, 1fr); } }
.tech-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    transition: all 0.3s ease;
}
.tech-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(10,46,82,0.08);
    border-color: #93c5fd;
}
.tech-card-icon {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
}
.tech-card-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.tech-card-desc {
    font-size: 0.7rem;
    color: #64748b;
    line-height: 1.4;
}

/* ── Clinical Workflow ─────────────────────────────── */
.clinical-flow-wrap {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 2rem;
}
.clinical-node {
    width: 360px;
    max-width: 90%;
    border-radius: 14px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    transition: all 0.25s ease;
}
.clinical-node:hover {
    transform: scale(1.03);
}
.cnode-primary {
    background: linear-gradient(135deg, #0a2e52 0%, #1e40af 100%);
    border: none;
    color: #ffffff;
    box-shadow: 0 6px 20px rgba(10,46,82,0.15);
}
.cnode-primary .clinical-node-label { color: #ffffff; }
.cnode-primary .clinical-node-sub { color: #93c5fd; }
.cnode-secondary {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1.5px solid #e2e8f0;
}
.cnode-secondary:hover {
    border-color: #93c5fd;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
}
.cnode-accent {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1.5px solid #93c5fd;
}
.cnode-accent:hover {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
}
.clinical-node-icon {
    font-size: 1.4rem;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.ci-white { background: rgba(255,255,255,0.2); }
.ci-blue { background: linear-gradient(135deg, #2563eb, #1d4ed8); box-shadow: 0 3px 10px rgba(37,99,235,0.2); }
.ci-light { background: rgba(37,99,235,0.1); }
.clinical-node-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #0f172a;
}
.clinical-node-sub {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.1rem;
}

/* ── System Components ─────────────────────────────── */
.comp-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
@media (max-width: 800px) { .comp-grid { grid-template-columns: repeat(2, 1fr); } }
.comp-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
    transition: all 0.3s ease;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.comp-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}
.comp-card-data::before { background: linear-gradient(90deg, #2563eb, #60a5fa); }
.comp-card-proc::before { background: linear-gradient(90deg, #059669, #34d399); }
.comp-card-pred::before { background: linear-gradient(90deg, #d97706, #fbbf24); }
.comp-card-pres::before { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.comp-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.06);
    border-color: #93c5fd;
}
.comp-card-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
}
.comp-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0a2e52;
    margin-bottom: 0.3rem;
}
.comp-card-desc {
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.5;
}

/* ── Advantages Grid ───────────────────────────────── */
.adv-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
@media (max-width: 800px) { .adv-grid { grid-template-columns: repeat(2, 1fr); } }
.adv-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.3rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    transition: all 0.3s ease;
}
.adv-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(10,46,82,0.08);
    border-color: #93c5fd;
}
.adv-card-icon {
    font-size: 1.4rem;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.adv-card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
}
.adv-card-desc {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.15rem;
    line-height: 1.4;
}

/* ── Disclaimer ────────────────────────────────────── */
.disclaimer-card {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #fcd34d;
    border-left: 5px solid #f59e0b;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(245,158,11,0.08);
}
.disclaimer-title {
    font-size: 1rem;
    font-weight: 700;
    color: #92400e;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.disclaimer-body {
    font-size: 0.82rem;
    color: #78350f;
    line-height: 1.7;
}
.disclaimer-body p { margin: 0.4rem 0; }

/* ── Footer ────────────────────────────────────────── */
.arch-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 1rem;
}
.arch-footer-brand {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0a2e52, #2563eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.arch-footer-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.2rem;
    line-height: 1.6;
}
.arch-footer-tags {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.75rem;
}
.arch-footer-tag {
    font-size: 0.65rem;
    color: #64748b;
    background: #f1f5f9;
    border-radius: 50px;
    padding: 0.25rem 0.75rem;
    font-weight: 500;
}
</style>
""")

# =============================================================================
# SECTION 1 — Hero Header
# =============================================================================
st_html("""
<div class="arch-hero">
    <div class="arch-hero-icon">🫁</div>
    <h1>System Architecture</h1>
    <div class="hero-badge">Clinical Decision Support System</div>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem; color: #e2e8f0; font-weight: 300; position: relative; z-index: 1;">
        for Pediatric Oxygen Therapy Prediction
    </p>
    <p class="hero-sub">
        This page illustrates the complete workflow of the OxyPredict system, including data preprocessing,
        machine learning prediction, explainability, and clinical decision support.
    </p>
</div>
""")


# =============================================================================
# SECTION 2 — System Overview
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">📋 System Overview</h3>
""")

st_html(f"""
<div class="overview-grid">
    <div class="ov-card">
        <div class="ov-card-icon">🗂️</div>
        <div class="ov-card-label">Dataset</div>
        <div class="ov-card-value">{DATASET_TOTAL_SAMPLES} Patients</div>
    </div>
    <div class="ov-card">
        <div class="ov-card-icon">🔬</div>
        <div class="ov-card-label">Clinical Features</div>
        <div class="ov-card-value">{DATASET_SELECTED_FEATURES}</div>
    </div>
    <div class="ov-card">
        <div class="ov-card-icon">🌲</div>
        <div class="ov-card-label">Machine Learning</div>
        <div class="ov-card-value">{MODEL_NAME}</div>
    </div>
    <div class="ov-card">
        <div class="ov-card-icon">🎯</div>
        <div class="ov-card-label">Prediction Output</div>
        <div class="ov-card-value">Yes / No</div>
    </div>
</div>
""")


# =============================================================================
# SECTION 3 — Main Architecture Diagram
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">🏗️ Main Architecture Diagram</h3>
""")

arch_nodes = [
    ("📋", "Patient Clinical Data", "44 clinical features collected from patient records"),
    ("✅", "Data Validation", "Verify completeness and format of input data"),
    ("⚙️", "Preprocessing Pipeline", "Imputation, encoding, and scaling transformations"),
    ("🌲", "Random Forest Model", "Ensemble of 800 decision trees for classification"),
    ("📈", "Probability Prediction", "Compute class probabilities and confidence scores"),
    ("🧠", "SHAP Explainability", "Feature importance analysis for model transparency"),
    ("🩺", "Clinical Decision Support", "Evidence-based recommendation with risk assessment"),
    ("📄", "Prediction Report", "Comprehensive patient report with clinical interpretation"),
]

arch_html = '<div class="arch-diagram-wrap"><div class="arch-flow">'
for i, (icon, label, desc) in enumerate(arch_nodes):
    arch_html += f"""
    <div class="arch-node">
        <div class="arch-node-icon">{icon}</div>
        <div>
            <div class="arch-node-label">{label}</div>
            <div class="arch-node-desc">{desc}</div>
        </div>
    </div>
    """
    if i < len(arch_nodes) - 1:
        arch_html += """
        <div class="arch-arrow">
            <div class="arch-arrow-line"></div>
            <div class="arch-arrow-head"></div>
        </div>
        """
arch_html += '</div></div>'
st_html(arch_html)


# =============================================================================
# SECTION 4 — Preprocessing Pipeline
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">🔄 Preprocessing Pipeline</h3>
""")

preproc_steps = [
    ("📥", "Raw Clinical Data", "Original patient data with 44 clinical features from the BD IRA dataset"),
    ("🔧", "Missing Value Imputation", "Handle incomplete records using statistical imputation strategies to preserve data integrity"),
    ("🏷️", "Categorical Encoding", "Transform categorical variables (e.g., Gender, Yes/No features) into numerical representations"),
    ("📏", "Feature Scaling", "Normalize numerical features to standardized ranges for optimal model performance"),
    ("✅", "Model Ready Data", "Clean, encoded, and scaled feature matrix ready for Random Forest prediction"),
]

preproc_html = '<div class="preproc-card"><div class="preproc-steps">'
for i, (icon, title, desc) in enumerate(preproc_steps):
    preproc_html += f"""
    <div class="preproc-step">
        <div class="preproc-step-title">{icon} {title}</div>
        <div class="preproc-step-desc">{desc}</div>
    </div>
    """
    if i < len(preproc_steps) - 1:
        preproc_html += """
        <div class="preproc-arrow">
            <div class="preproc-arrow-line"></div>
            <div class="preproc-arrow-head"></div>
        </div>
        """
preproc_html += '</div></div>'
st_html(preproc_html)


# =============================================================================
# SECTION 5 — Machine Learning Pipeline (Horizontal Timeline)
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">🤖 Machine Learning Pipeline</h3>
""")

tl_steps = [
    ("📥", "Input", "Patient data", "ml-tl-dot-blue"),
    ("⚙️", "Preprocessing", "Clean & encode", "ml-tl-dot-blue"),
    ("🌲", "Random Forest", "800 trees", "ml-tl-dot-green"),
    ("📊", "Probability", "Score output", "ml-tl-dot-amber"),
    ("🏷️", "Classification", "Yes / No", "ml-tl-dot-purple"),
    ("🩺", "Recommendation", "Clinical output", "ml-tl-dot-blue"),
]

tl_html = '<div class="ml-timeline-wrap"><div class="ml-timeline">'
for i, (icon, label, sub, color) in enumerate(tl_steps):
    tl_html += f"""
    <div class="ml-tl-step">
        <div class="ml-tl-dot {color}">{icon}</div>
        <div class="ml-tl-label">{label}</div>
        <div class="ml-tl-sublabel">{sub}</div>
    </div>
    """
    if i < len(tl_steps) - 1:
        tl_html += """
        <div class="ml-tl-connector">
            <div class="ml-tl-connector-line"></div>
            <div class="ml-tl-connector-arr"></div>
        </div>
        """
tl_html += '</div></div>'
st_html(tl_html)


# =============================================================================
# SECTION 6 — Tech Stack
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">🛠️ Technology Stack</h3>
""")

techs = [
    ("🐍", "Python", "Core programming language for data science and ML"),
    ("🌐", "Streamlit", "Interactive web application framework for dashboards"),
    ("🤖", "Scikit-learn", "Machine learning library for Random Forest classifier"),
    ("🧠", "SHAP", "Explainable AI framework for model interpretability"),
    ("🐼", "Pandas", "Data manipulation and analysis library"),
    ("🔢", "NumPy", "Numerical computing and array operations"),
    ("💾", "Joblib", "Model serialization and efficient persistence"),
    ("📊", "OpenPyXL", "Excel file processing for batch prediction exports"),
]

tech_html = '<div class="tech-grid">'
for icon, name, desc in techs:
    tech_html += f"""
    <div class="tech-card">
        <div class="tech-card-icon">{icon}</div>
        <div class="tech-card-name">{name}</div>
        <div class="tech-card-desc">{desc}</div>
    </div>
    """
tech_html += '</div>'
st_html(tech_html)


# =============================================================================
# SECTION 7 — Model Performance (st.metric)
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">📈 Model Performance</h3>
""")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Accuracy", f"{MODEL_ACCURACY}%")
with m2:
    st.metric("F1 Macro", f"{MODEL_F1_MACRO}%")
with m3:
    st.metric("ROC-AUC", f"{MODEL_ROC_AUC}%")

m4, m5, m6 = st.columns(3)
with m4:
    st.metric("Features", f"{DATASET_SELECTED_FEATURES}")
with m5:
    st.metric("Dataset", f"{DATASET_TOTAL_SAMPLES}")
with m6:
    st.metric("Algorithm", f"{MODEL_NAME}")

st.markdown("")


# =============================================================================
# SECTION 8 — Clinical Workflow
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">🏥 Clinical Workflow</h3>
""")

clinical_nodes = [
    ("👨‍⚕️", "Healthcare Professional", "Physician or clinical staff initiates assessment", "cnode-primary", "ci-white"),
    ("📝", "Input Patient Data", "Enter 44 clinical features into the system", "cnode-secondary", "ci-blue"),
    ("🫁", "OxyPredict CDSS", "System processes data through ML pipeline", "cnode-accent", "ci-light"),
    ("📊", "Prediction Result", "Probability score and classification output", "cnode-secondary", "ci-blue"),
    ("🔍", "Clinical Interpretation", "SHAP-based explanation of prediction factors", "cnode-secondary", "ci-blue"),
    ("✅", "Final Medical Decision", "Clinician makes informed treatment decision", "cnode-primary", "ci-white"),
]

cflow_html = '<div class="clinical-flow-wrap"><div class="arch-flow">'
for i, (icon, label, sub, style, icon_style) in enumerate(clinical_nodes):
    cflow_html += f"""
    <div class="clinical-node {style}">
        <div class="clinical-node-icon {icon_style}">{icon}</div>
        <div>
            <div class="clinical-node-label">{label}</div>
            <div class="clinical-node-sub">{sub}</div>
        </div>
    </div>
    """
    if i < len(clinical_nodes) - 1:
        cflow_html += """
        <div class="arch-arrow">
            <div class="arch-arrow-line"></div>
            <div class="arch-arrow-head"></div>
        </div>
        """
cflow_html += '</div></div>'
st_html(cflow_html)

st_html("""
<div style="
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 2rem;
    font-size: 0.82rem;
    color: #1e40af;
    line-height: 1.6;
">
    <strong>📌 Note:</strong> OxyPredict serves as a decision support tool. The system provides evidence-based
    recommendations to assist clinicians, but the final medical decision always remains with the qualified
    healthcare professional.
</div>
""")


# =============================================================================
# SECTION 9 — System Components
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">🧩 System Components</h3>
""")

st_html("""
<div class="comp-grid">
    <div class="comp-card comp-card-data">
        <div class="comp-card-icon">🗄️</div>
        <div class="comp-card-title">Data Layer</div>
        <div class="comp-card-desc">BD IRA dataset containing 801 patient records with 44 selected clinical features for model training and validation.</div>
    </div>
    <div class="comp-card comp-card-proc">
        <div class="comp-card-icon">⚙️</div>
        <div class="comp-card-title">Processing Layer</div>
        <div class="comp-card-desc">Preprocessing pipeline including missing value imputation, categorical encoding, and feature scaling.</div>
    </div>
    <div class="comp-card comp-card-pred">
        <div class="comp-card-icon">🧠</div>
        <div class="comp-card-title">Prediction Layer</div>
        <div class="comp-card-desc">Random Forest classifier with 800 estimators providing probability-based predictions with SHAP explanations.</div>
    </div>
    <div class="comp-card comp-card-pres">
        <div class="comp-card-icon">🖥️</div>
        <div class="comp-card-title">Presentation Layer</div>
        <div class="comp-card-desc">Streamlit-powered interactive dashboard with single & batch prediction, SHAP visualization, and report generation.</div>
    </div>
</div>
""")


# =============================================================================
# SECTION 10 — System Advantages
# =============================================================================
st_html("""
<h3 class="arch-section-hdr">✨ System Advantages</h3>
""")

st_html("""
<div class="adv-grid">
    <div class="adv-card">
        <div class="adv-card-icon">⚡</div>
        <div>
            <div class="adv-card-title">Fast Prediction</div>
            <div class="adv-card-desc">Real-time inference with sub-second response time for immediate clinical decision support.</div>
        </div>
    </div>
    <div class="adv-card">
        <div class="adv-card-icon">🧠</div>
        <div>
            <div class="adv-card-title">Explainable AI</div>
            <div class="adv-card-desc">SHAP-based feature importance analysis providing transparent and interpretable predictions.</div>
        </div>
    </div>
    <div class="adv-card">
        <div class="adv-card-icon">📊</div>
        <div>
            <div class="adv-card-title">Batch Prediction</div>
            <div class="adv-card-desc">Process multiple patient records simultaneously via CSV/XLSX file upload.</div>
        </div>
    </div>
    <div class="adv-card">
        <div class="adv-card-icon">🖥️</div>
        <div>
            <div class="adv-card-title">Interactive Dashboard</div>
            <div class="adv-card-desc">User-friendly web interface with dynamic visualizations and intuitive navigation.</div>
        </div>
    </div>
    <div class="adv-card">
        <div class="adv-card-icon">🎨</div>
        <div>
            <div class="adv-card-title">Modern UI</div>
            <div class="adv-card-desc">Professional healthcare-grade design with responsive layout and smooth animations.</div>
        </div>
    </div>
    <div class="adv-card">
        <div class="adv-card-icon">🩺</div>
        <div>
            <div class="adv-card-title">Clinical Decision Support</div>
            <div class="adv-card-desc">Evidence-based recommendations with risk classification to assist clinical judgment.</div>
        </div>
    </div>
</div>
""")


# =============================================================================
# SECTION 11 — Disclaimer
# =============================================================================
st_html("""
<div class="disclaimer-card">
    <div class="disclaimer-title">⚠️ Clinical Disclaimer</div>
    <div class="disclaimer-body">
        <p><strong>OxyPredict</strong> is designed as a <strong>Clinical Decision Support System (CDSS)</strong>.</p>
        <p>The prediction results generated by the machine learning model are intended to <strong>assist healthcare professionals</strong>
        and should <strong>not replace clinical judgment</strong>.</p>
        <p>Final medical decisions remain the <strong>responsibility of qualified healthcare providers</strong>.</p>
    </div>
</div>
""")


# =============================================================================
# SECTION 12 — Footer
# =============================================================================
st_html("""
<div class="arch-footer">
    <div class="arch-footer-brand">OxyPredict</div>
    <div class="arch-footer-sub">
        Clinical Decision Support System<br>
        Prototype for Undergraduate Thesis
    </div>
    <div class="arch-footer-tags">
        <span class="arch-footer-tag">Machine Learning</span>
        <span class="arch-footer-tag">Explainable AI</span>
        <span class="arch-footer-tag">Streamlit</span>
        <span class="arch-footer-tag">© 2026</span>
    </div>
</div>
""")
