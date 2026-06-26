"""
Configuration file for the Oxygen Therapy Predictor application.
Contains feature definitions, model information, and application constants.
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


def setup_page(page_title: str):
    import streamlit as st
    
    # 1. Custom CSS
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar styling */
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

        /* Custom Icons & Labels using pseudo-elements */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]::before {
            display: inline-block !important;
            font-size: 1.1rem !important;
            margin-right: 0.6rem !important;
            line-height: 1 !important;
        }

        /* Home Page ("app") Customization */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/"]::before {
            content: "🏠" !important;
        }
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/"] span {
            display: none !important;
        }
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/"]::after {
            content: "Home" !important;
            font-weight: inherit !important;
        }

        /* Dashboard Page Customization */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/Dashboard"]::before {
            content: "📊" !important;
        }

        /* Single Prediction Page Customization */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/Single_Prediction"]::before {
            content: "🩺" !important;
        }

        /* Batch Prediction Page Customization */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/Batch_Prediction"]::before {
            content: "📁" !important;
        }

        /* SHAP Explanation Page Customization */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/SHAP_Explanation"]::before {
            content: "🔍" !important;
        }

        /* About Model Page Customization */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href="/About_Model"]::before {
            content: "ℹ️" !important;
        }

        /* Main metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
            border: 1px solid #bdd8f1;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(10, 60, 120, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(10, 60, 120, 0.12);
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

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.3px;
            transition: all 0.3s ease;
            box-shadow: 0 3px 12px rgba(26, 115, 232, 0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1557b0 0%, #0d47a1 100%);
            box-shadow: 0 6px 20px rgba(26, 115, 232, 0.35);
            transform: translateY(-1px);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #00897b 0%, #00695c 100%) !important;
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 3px 12px rgba(0, 137, 123, 0.25);
            transition: all 0.3s ease;
        }

        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #00695c 0%, #004d40 100%) !important;
            box-shadow: 0 6px 20px rgba(0, 137, 123, 0.35);
            transform: translateY(-1px);
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f0f7ff;
            border-radius: 10px;
            font-weight: 600;
            color: #1a3a5c;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #90caf9;
            border-radius: 12px;
            padding: 1rem;
            background: #f8fbff;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-weight: 500;
        }

        /* Dataframe */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Selectbox & number input */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            border-radius: 8px !important;
        }

        /* Hide default Streamlit footer & header */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom scrollbar */
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
    </style>
    """, unsafe_allow_html=True)

    # 2. Sidebar Branding
    with st.sidebar:
        st.markdown("""
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
            ">OxyPredict</h2>
            <p style="
                margin: 0.2rem 0 0 0;
                font-size: 0.72rem;
                color: #64748b !important;
                font-weight: 400;
                letter-spacing: 0.3px;
            ">Prediksi Terapi Oksigen</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
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
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("""
        <div style="padding: 0 0.5rem;">
            <p style="font-size: 0.7rem; color: #475569 !important; margin-bottom: 0.4rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                Navigasi
            </p>
        </div>
        """, unsafe_allow_html=True)


