"""
Configuration file for the Oxygen Therapy Predictor application.
Contains feature definitions, model information, application constants,
and the unified design system.
"""

import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_random_forest.pkl")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# =============================================================================
# Dataset Information
# =============================================================================
DATASET_NAME = "BD IRA (bd_raw.csv)"
DATASET_TOTAL_SAMPLES = 801
DATASET_ORIGINAL_COLUMNS = 91
DATASET_SELECTED_FEATURES = 44
TARGET_COLUMN = "Oxygen Therapy"

# =============================================================================
# Model Information
# =============================================================================
MODEL_NAME = "Random Forest"
MODEL_N_ESTIMATORS = 800
MODEL_ACCURACY = 89.44
MODEL_F1_MACRO = 86.20
MODEL_ROC_AUC = 90.93

# =============================================================================
# Application Metadata
# =============================================================================
APP_VERSION = "1.0"
APP_DEVELOPER = "Tito Purwana S."
APP_TITLE = "OxyPredict"
APP_SUBTITLE = "Prediksi Terapi Oksigen"

# =============================================================================
# Design System Tokens
# =============================================================================
COLORS = {
    "primary": "#0A2E52",
    "secondary": "#2563EB",
    "light_blue": "#DBEAFE",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "background": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#64748B",
    "text_light": "#94A3B8",
    "border": "#E2E8F0",
    "border_light": "#F1F5F9",
}

# =============================================================================
# Feature Definitions
# =============================================================================

# Numerical features (15 features)
NUMERICAL_FEATURES = [
    "Age (months)",
    "Number of persons living in house",
    "Number of siblings",
    "Number of rooms in house",
    "Duration of pain before consultation (days)",
    "Number of days with fever",
    "Number of days of vomiting",
    "Weight (Kg)",
    "Height (cm)",
    "Oxygen saturation (SaO2) at admission",
    "Axillary temperature (\u00b0C)",
    "Respiratory rate",
    "Heart rate",
    "C-reactive protein",
    "Procalcitonin",
]

# Categorical features with Yes/No options (28 features)
BINARY_CATEGORICAL_FEATURES = [
    "Gender",
    "Medical insurance",
    "Smokers at home",
    "Contact with a tuberculosis patient",
    "Health history : Prior admission because of respiratory condition",
    "Health history : Prematurity",
    "Breastfeeding",
    "Previous history of antibiotic use in the 2 weeks leading up to the admission",
    "Known asthmatic patient",
    "Patient with a diagnosed chronic condition",
    "History of fever",
    "History of vomiting",
    "History of diarrhea",
    "History of cough",
    "History of rhinorrhea",
    "Unusual sleepiness",
    "Paleness",
    "Disorders of consciousness",
    "Dehydration signs",
    "Restlessness",
    "Cyanosis",
    "Nasal flaring",
    "Laryngeal stridor",
    "Rhonchi",
    "Crackles",
    "Wheezing",
    "Hypoventilation",
    "Nasopharyngeal aspiration",
]

# Multi-class categorical features
MULTI_CATEGORICAL_FEATURES = {
    "Age-appropriate vaccinations": ["No", "Partially", "Yes"],
}

# Options for binary categorical features
BINARY_OPTIONS = {
    "Gender": ["Female", "Male"],
}
# All other binary categoricals use ["No", "Yes"]
for _feat in BINARY_CATEGORICAL_FEATURES:
    if _feat not in BINARY_OPTIONS:
        BINARY_OPTIONS[_feat] = ["No", "Yes"]

# Complete ordered list of all 44 features (as expected by the model)
ALL_FEATURES = [
    "Age (months)",
    "Gender",
    "Medical insurance",
    "Number of persons living in house",
    "Number of siblings",
    "Number of rooms in house",
    "Smokers at home",
    "Contact with a tuberculosis patient",
    "Health history : Prior admission because of respiratory condition",
    "Health history : Prematurity",
    "Breastfeeding",
    "Age-appropriate vaccinations",
    "Previous history of antibiotic use in the 2 weeks leading up to the admission",
    "Known asthmatic patient",
    "Patient with a diagnosed chronic condition",
    "Duration of pain before consultation (days)",
    "History of fever",
    "Number of days with fever",
    "History of vomiting",
    "Number of days of vomiting",
    "History of diarrhea",
    "History of cough",
    "History of rhinorrhea",
    "Weight (Kg)",
    "Height (cm)",
    "Unusual sleepiness",
    "Oxygen saturation (SaO2) at admission",
    "Axillary temperature (\u00b0C)",
    "Respiratory rate",
    "Heart rate",
    "Paleness",
    "Disorders of consciousness",
    "Dehydration signs",
    "Restlessness",
    "Cyanosis",
    "Nasal flaring",
    "Laryngeal stridor",
    "Rhonchi",
    "Crackles",
    "Wheezing",
    "Hypoventilation",
    "Nasopharyngeal aspiration",
    "C-reactive protein",
    "Procalcitonin",
]

# =============================================================================
# Numerical Feature Ranges (min, max, default, step)
# =============================================================================
NUMERICAL_RANGES = {
    "Age (months)": (0, 216, 24, 1),
    "Number of persons living in house": (1, 20, 5, 1),
    "Number of siblings": (0, 15, 2, 1),
    "Number of rooms in house": (1, 15, 3, 1),
    "Duration of pain before consultation (days)": (0, 60, 3, 1),
    "Number of days with fever": (0, 30, 2, 1),
    "Number of days of vomiting": (0, 30, 0, 1),
    "Weight (Kg)": (1.0, 80.0, 10.0, 0.1),
    "Height (cm)": (30.0, 180.0, 75.0, 0.1),
    "Oxygen saturation (SaO2) at admission": (50.0, 100.0, 96.0, 0.1),
    "Axillary temperature (\u00b0C)": (35.0, 42.0, 37.0, 0.1),
    "Respiratory rate": (10, 100, 30, 1),
    "Heart rate": (40, 220, 120, 1),
    "C-reactive protein": (0.0, 500.0, 10.0, 0.1),
    "Procalcitonin": (0.0, 200.0, 0.5, 0.01),
}

FEATURE_GROUPS = {
    "Demografi": [
        "Age (months)",
        "Gender",
        "Medical insurance",
    ],
    "Kondisi Rumah dan Riwayat": [
        "Number of persons living in house",
        "Number of siblings",
        "Number of rooms in house",
        "Smokers at home",
        "Contact with a tuberculosis patient",
        "Health history : Prior admission because of respiratory condition",
        "Health history : Prematurity",
        "Breastfeeding",
        "Age-appropriate vaccinations",
        "Previous history of antibiotic use in the 2 weeks leading up to the admission",
    ],
    "Riwayat Penyakit dan Gejala": [
        "Duration of pain before consultation (days)",
        "History of fever",
        "Number of days with fever",
        "History of vomiting",
        "Number of days of vomiting",
        "History of diarrhea",
        "History of cough",
        "History of rhinorrhea",
        "Known asthmatic patient",
        "Patient with a diagnosed chronic condition",
    ],
    "Tanda Vital": [
        "Weight (Kg)",
        "Height (cm)",
        "Oxygen saturation (SaO2) at admission",
        "Axillary temperature (\u00b0C)",
        "Respiratory rate",
        "Heart rate",
    ],
    "Pemeriksaan Klinis": [
        "Unusual sleepiness",
        "Paleness",
        "Disorders of consciousness",
        "Dehydration signs",
        "Restlessness",
        "Cyanosis",
        "Nasal flaring",
        "Laryngeal stridor",
        "Rhonchi",
        "Crackles",
        "Wheezing",
        "Hypoventilation",
        "Nasopharyngeal aspiration",
    ],
    "Pemeriksaan Laboratorium": [
        "C-reactive protein",
        "Procalcitonin",
    ],
}


# =============================================================================
# Unified Design System — Global CSS & Helper Functions
# =============================================================================

def _get_global_css():
    """Return the unified global CSS for the entire application."""
    return """
    <style>
        /* ================================================================
           IMPORT GOOGLE FONT
           ================================================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ================================================================
           GLOBAL RESET & FONT
           ================================================================ */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* ================================================================
           KEYFRAME ANIMATIONS
           ================================================================ */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50%      { transform: scale(1.08); }
        }
        @keyframes heroFloat {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-6px); }
        }
        @keyframes shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        /* ================================================================
           SIDEBAR STYLING
           ================================================================ */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1628 0%, #112240 50%, #0d2137 100%);
            color: #e0e8f0;
        }

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stMarkdown span {
            color: #c8d6e5 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(100, 180, 255, 0.15);
        }

        /* Sidebar Nav Container & Buttons styling */
        section[data-testid="stSidebar"] .stSidebarNavItems {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        /* Enforce custom sidebar ordering using flexbox */
        section[data-testid="stSidebar"] .stSidebarNavItems,
        section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] ul,
        section[data-testid="stSidebar"] ul[data-testid="stSidebarNavItems"] {
            display: flex !important;
            flex-direction: column !important;
        }

        /* Order 1: Dashboard */
        section[data-testid="stSidebar"] ul li:first-child,
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/"],
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:first-of-type {
            order: 1 !important;
        }

        /* Order 2: Single Prediction */
        section[data-testid="stSidebar"] ul li:has(a[href*="Single"]),
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Single"] {
            order: 2 !important;
        }

        /* Order 3: Batch Prediction */
        section[data-testid="stSidebar"] ul li:has(a[href*="Batch"]),
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Batch"] {
            order: 3 !important;
        }

        /* Order 4: Clinical Decision Guide */
        section[data-testid="stSidebar"] ul li:has(a[href*="Clinical"]),
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Clinical"] {
            order: 4 !important;
        }

        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
            display: flex !important;
            align-items: center !important;
            padding: 0.6rem 0.8rem !important;
            margin: 0.3rem 0 !important;
            border-radius: 10px !important;
            color: #a0b4c8 !important;
            background-color: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            text-decoration: none !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
        }

        /* Hide default Streamlit page icons */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] svg {
            display: none !important;
        }

        /* Hover Navigation Button */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
            background-color: rgba(59, 130, 246, 0.12) !important;
            color: #60a5fa !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
            transform: translateX(4px) !important;
        }

        /* Active Navigation Button */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
            background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
            color: #ffffff !important;
            border-color: #3b82f6 !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(29, 78, 216, 0.25) !important;
        }

        /* Custom Icons using pseudo-elements */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]::before {
            display: inline-block !important;
            font-size: 1.1rem !important;
            margin-right: 0.6rem !important;
            line-height: 1 !important;
        }

        /* ── Page-specific sidebar icons ────────────────────────── */
        /* Dashboard (Main Entrypoint) */
        section[data-testid="stSidebar"] ul li:first-child a[data-testid="stSidebarNavLink"]::before,
        section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] li:first-child a::before {
            content: "📊" !important;
        }
        section[data-testid="stSidebar"] ul li:first-child a[data-testid="stSidebarNavLink"] span,
        section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] li:first-child a span {
            display: none !important;
        }
        section[data-testid="stSidebar"] ul li:first-child a[data-testid="stSidebarNavLink"]::after,
        section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] li:first-child a::after {
            content: "Dashboard" !important;
            font-weight: inherit !important;
            display: inline-block !important;
        }

        /* Single Prediction */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Single"]::before {
            content: "🩺" !important;
        }

        /* Batch Prediction */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Batch"]::before {
            content: "📂" !important;
        }

        /* Clinical Decision Guide */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Clinical"]::before {
            content: "📘" !important;
        }

        /* ================================================================
           METRIC CARDS (Streamlit st.metric)
           ================================================================ */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
            border: 1px solid #bdd8f1;
            border-radius: 16px;
            padding: 16px 20px;
            box-shadow: 0 4px 12px rgba(10, 46, 82, 0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(10, 46, 82, 0.12);
        }

        div[data-testid="stMetric"] label {
            color: #3b6fa0 !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0a2e52 !important;
            font-weight: 700 !important;
        }

        /* ================================================================
           BUTTONS
           ================================================================ */
        .stButton > button {
            background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.3px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(26, 115, 232, 0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1557b0 0%, #0d47a1 100%);
            box-shadow: 0 6px 20px rgba(26, 115, 232, 0.35);
            transform: translateY(-2px);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #00897b 0%, #00695c 100%) !important;
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0, 137, 123, 0.25);
            transition: all 0.3s ease;
        }

        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #00695c 0%, #004d40 100%) !important;
            box-shadow: 0 6px 20px rgba(0, 137, 123, 0.35);
            transform: translateY(-2px);
        }

        /* ================================================================
           EXPANDER
           ================================================================ */
        .streamlit-expanderHeader {
            background-color: #f0f7ff;
            border-radius: 10px;
            font-weight: 600;
            color: #1a3a5c;
        }

        /* ================================================================
           FILE UPLOADER
           ================================================================ */
        [data-testid="stFileUploader"] {
            border: 2px dashed #90caf9;
            border-radius: 16px;
            padding: 1rem;
            background: #f8fbff;
        }

        /* ================================================================
           TABS
           ================================================================ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 8px 20px;
            font-weight: 500;
        }

        /* ================================================================
           DATAFRAME / TABLE
           ================================================================ */
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
        }

        /* ================================================================
           SELECTBOX & NUMBER INPUT
           ================================================================ */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            border-radius: 10px !important;
        }

        /* ================================================================
           HIDE DEFAULT STREAMLIT CHROME & PRESERVE SIDEBAR CONTROLS
           ================================================================ */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Make header background transparent to prevent a white stripe, but keep it visible for sidebar button */
        [data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99 !important;
        }

        /* Hide Deploy button and hamburger menu specifically, keep the sidebar collapse/expand controls */
        [data-testid="stHeader"] [data-testid="stActionButton"],
        [data-testid="stHeader"] button[aria-label="Deploy"],
        [data-testid="stHeader"] button[aria-label="MainMenu"],
        [data-testid="stHeader"] button[aria-label="developer options"],
        [data-testid="stHeader"] #MainMenu {
            display: none !important;
        }

        /* Smooth transition for opening/closing sidebar and main content adjustment */
        section[data-testid="stSidebar"] {
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        section[data-testid="stMain"] {
            transition: margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        /* Smooth transition for content inside sidebar */
        div[data-testid="stSidebarUserContent"] {
            transition: opacity 0.2s ease-in-out !important;
        }
        
        /* Keep sidebar scroll position container clean and stable */
        div[data-testid="stSidebarContent"] {
            overflow-y: auto !important;
            scrollbar-gutter: stable;
        }

        /* ================================================================
           CUSTOM SCROLLBAR
           ================================================================ */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        ::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #64748b;
        }

        /* ================================================================
           UNIFIED SECTION TITLE
           ================================================================ */
        .section-title-custom {
            color: #0a2e52;
            font-weight: 700;
            font-size: 1.2rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid #2563eb;
            padding-left: 0.6rem;
        }

        /* ================================================================
           UNIFIED CARD STYLE
           ================================================================ */
        .cdss-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(10, 46, 82, 0.06);
            margin-bottom: 1.25rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .cdss-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(10, 46, 82, 0.1);
        }

        /* ================================================================
           PAGE ANIMATION
           ================================================================ */
        .stMainBlockContainer {
            animation: fadeIn 0.4s ease-out;
        }
    </style>
    """


def _get_sidebar_html():
    """Return the sidebar branding HTML."""
    return f"""
    <div style="text-align: center; padding: 1.2rem 0 0.5rem 0;">
        <div style="
            font-size: 2.8rem;
            margin-bottom: 0.3rem;
            filter: drop-shadow(0 2px 8px rgba(96,165,250,0.4));
        ">🫁</div>
        <h2 style="
            margin: 0;
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 1px;
        ">{APP_TITLE}</h2>
        <p style="
            margin: 0.2rem 0 0 0;
            font-size: 0.72rem;
            color: #64748b !important;
            font-weight: 400;
            letter-spacing: 0.3px;
        ">{APP_SUBTITLE}</p>
        <div style="
            display: inline-block;
            margin-top: 0.5rem;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 50px;
            padding: 0.15rem 0.6rem;
            font-size: 0.65rem;
            color: #60a5fa !important;
            font-weight: 700;
            letter-spacing: 0.5px;
        ">v{APP_VERSION}</div>
    </div>
    """


def _get_sidebar_info_html():
    """Return the sidebar info card HTML."""
    return """
    <div style="
        padding: 0.8rem;
        background: rgba(59, 130, 246, 0.08);
        border-radius: 10px;
        border-left: 3px solid #3b82f6;
        margin: 0.5rem 0;
    ">
        <p style="
            margin: 0;
            font-size: 0.75rem;
            color: #94a3b8 !important;
            line-height: 1.5;
        ">
            Sistem Pendukung Keputusan Klinis untuk prediksi kebutuhan
            terapi oksigen pada pasien anak dengan <strong style="color: #60a5fa !important;">ISPA</strong>
            dan <strong style="color: #60a5fa !important;">Pneumonia</strong>.
        </p>
    </div>
    """


def _get_sidebar_nav_label():
    """Return sidebar navigation section label."""
    return """
    <div style="padding: 0 0.5rem;">
        <p style="font-size: 0.7rem; color: #475569 !important; margin-bottom: 0.4rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
            Navigasi
        </p>
    </div>
    """


def _get_sidebar_footer_html():
    """Return the sidebar footer HTML."""
    return f"""
    <div style="
        padding: 0.8rem 0.5rem;
        margin-top: 1.5rem;
        border-top: 1px solid #475569;
        text-align: center;
        font-family: 'Inter', sans-serif;
    ">
        <p style="
            margin: 0;
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748b !important;
        ">OxyPredict v{APP_VERSION}</p>
        <p style="
            margin: 0.2rem 0 0 0;
            font-size: 0.68rem;
            color: #64748b !important;
            line-height: 1.4;
        ">
            Clinical Decision Support System<br>
            Pediatric Acute Respiratory Infection
        </p>
        <p style="
            margin: 0.4rem 0 0 0;
            font-size: 0.65rem;
            color: #475569 !important;
        ">© 2026</p>
    </div>
    """


def setup_page(page_title: str):
    """Set up page with unified CSS, sidebar branding, and consistent design system."""
    import streamlit as st

    # 1. Global CSS
    st.markdown(_get_global_css(), unsafe_allow_html=True)

    # 2. Sidebar Branding
    with st.sidebar:
        st.markdown(_get_sidebar_html(), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(_get_sidebar_info_html(), unsafe_allow_html=True)
        st.markdown("")
        st.markdown(_get_sidebar_nav_label(), unsafe_allow_html=True)

        # Add spacer and footer at the bottom of sidebar
        st.markdown("")
        st.markdown("")
        st.markdown(_get_sidebar_footer_html(), unsafe_allow_html=True)


# =============================================================================
# Reusable UI Component Helpers
# =============================================================================

def render_page_header(icon: str, title: str, subtitle: str) -> str:
    """
    Return HTML for a standardized page hero header.
    Usage: st.markdown(render_page_header("📊", "Dashboard", "..."), unsafe_allow_html=True)
    """
    return f"""
<div style="
    background: linear-gradient(135deg, #0a2e52 0%, #1a4a7a 50%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(10, 46, 82, 0.18);
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute;
        top: -60%;
        right: -15%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
        border-radius: 50%;
    "></div>
    <div style="position: relative; z-index: 1;">
        <div style="font-size: 2.2rem; margin-bottom: 0.5rem; animation: heroFloat 3s ease-in-out infinite;">{icon}</div>
        <h1 style="
            margin: 0;
            color: white;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.3px;
        ">{title}</h1>
        <p style="
            margin: 0.4rem 0 0 0;
            color: #93c5fd;
            font-size: 0.95rem;
            max-width: 700px;
        ">{subtitle}</p>
    </div>
</div>
"""


def render_section_divider() -> str:
    """
    Return HTML for an elegant section divider.
    Usage: st.markdown(render_section_divider(), unsafe_allow_html=True)
    """
    return """
<div style="
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0;
">
    <div style="flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #cbd5e1, transparent);"></div>
    <div style="
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563eb, #60a5fa);
        box-shadow: 0 0 8px rgba(37,99,235,0.3);
    "></div>
    <div style="flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #cbd5e1, transparent);"></div>
</div>
"""


def render_footer() -> str:
    """
    Return HTML for the standardized page footer.
    Usage: st.markdown(render_footer(), unsafe_allow_html=True)
    """
    return f"""
<div style="
    text-align: center;
    padding: 2rem 0 1rem 0;
    margin-top: 2rem;
    border-top: 1px solid #e2e8f0;
">
    <div style="
        font-size: 1.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0a2e52, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    ">{APP_TITLE} v{APP_VERSION}</div>
    <p style="margin: 0; color: #64748b; font-size: 0.78rem; line-height: 1.6;">
        Clinical Decision Support System<br>
        Powered by Random Forest + SHAP
    </p>
    <p style="margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 0.7rem;">
        Developed for Undergraduate Thesis Research
    </p>
    <div style="display: flex; gap: 0.4rem; justify-content: center; flex-wrap: wrap; margin-top: 0.6rem;">
        <span style="font-size: 0.62rem; color: #64748b; background: #f1f5f9; border-radius: 50px; padding: 0.2rem 0.6rem; font-weight: 500;">Streamlit</span>
        <span style="font-size: 0.62rem; color: #64748b; background: #f1f5f9; border-radius: 50px; padding: 0.2rem 0.6rem; font-weight: 500;">scikit-learn</span>
        <span style="font-size: 0.62rem; color: #64748b; background: #f1f5f9; border-radius: 50px; padding: 0.2rem 0.6rem; font-weight: 500;">SHAP</span>
        <span style="font-size: 0.62rem; color: #64748b; background: #f1f5f9; border-radius: 50px; padding: 0.2rem 0.6rem; font-weight: 500;">© 2026</span>
    </div>
</div>
"""


def render_empty_state(icon: str, title: str, description: str) -> str:
    """
    Return HTML for a modern empty state placeholder.
    Usage: st.markdown(render_empty_state("📂", "No File Uploaded", "..."), unsafe_allow_html=True)
    """
    return f"""
<div style="
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
">
    <div style="
        font-size: 3rem;
        margin-bottom: 0.8rem;
        animation: pulse 2.5s ease-in-out infinite;
    ">{icon}</div>
    <h3 style="
        margin: 0 0 0.5rem 0;
        color: #0a2e52;
        font-size: 1.1rem;
        font-weight: 700;
    ">{title}</h3>
    <p style="
        margin: 0;
        color: #64748b;
        font-size: 0.88rem;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    ">{description}</p>
</div>
"""
