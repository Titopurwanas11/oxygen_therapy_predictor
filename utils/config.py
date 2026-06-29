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
    "primary": "#0F4C75",
    "secondary": "#3282B8",
    "light_blue": "#E2F0F9",
    "accent": "#14B8A6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "background": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "text_primary": "#1E293B",
    "text_secondary": "#64748B",
    "text_muted": "#8892B0",
    "text_light": "#B0C4DE",
    "border": "#D6E4F0",
    "border_light": "#EAF2F8",
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

# Combined options for all categorical features
CATEGORICAL_OPTIONS = {**BINARY_OPTIONS, **MULTI_CATEGORICAL_FEATURES}


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
           IMPORT GOOGLE FONTS & MATERIAL SYMBOLS OUTLINED
           ================================================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

        /* ================================================================
           GLOBAL RESET & APP BACKGROUND
           ================================================================ */
        html, body, [data-testid="stApp"] {
            font-family: 'Inter', 'Manrope', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #F8FAFC !important;
            color: #1E293B !important;
        }
        
        .stApp {
            background-color: #F8FAFC !important;
        }

        /* Set page width content container padding */
        .stMainBlockContainer {
            padding: 3rem 4rem !important;
            max-width: 1200px !important;
            animation: fadeIn 0.2s ease-out;
        }

        /* ================================================================
           KEYFRAME ANIMATIONS
           ================================================================ */
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }

        /* ================================================================
           SIDEBAR STYLING (CLINICAL NAVY GRADIENT)
           ================================================================ */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F4C75, #154C79, #1F5D91) !important;
            border-right: 1px solid #D6E4F0;
        }

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stMarkdown span {
            color: #E2F0F9 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.08);
        }

        /* Sidebar Nav Container & Buttons styling */
        section[data-testid="stSidebar"] .stSidebarNavItems {
            padding: 0.5rem 0.75rem !important;
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

        /* Sidebar Link Styling */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
            display: flex !important;
            align-items: center !important;
            padding: 0.65rem 0.85rem !important;
            margin: 0.15rem 0 !important;
            border-radius: 12px !important;
            color: #E2F0F9 !important;
            background-color: transparent !important;
            border: 1px solid transparent !important;
            transition: all 0.15s ease-in-out !important;
            text-decoration: none !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            border-left: 3px solid transparent !important;
        }

        /* Hide default Streamlit page icons */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] svg {
            display: none !important;
        }

        /* Hover Navigation Button */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
            transform: none !important;
        }

        /* Active Navigation Button */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
            border-left: 3px solid #3282B8 !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }

        /* Custom Icons using pseudo-elements with Material Symbols Outlined */
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]::before {
            font-family: 'Material Symbols Outlined' !important;
            display: inline-block !important;
            font-size: 1.25rem !important;
            margin-right: 0.65rem !important;
            line-height: 1 !important;
            vertical-align: middle !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Page-specific sidebar icons */
        section[data-testid="stSidebar"] ul li:first-child a[data-testid="stSidebarNavLink"]::before,
        section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] li:first-child a::before {
            content: "grid_view" !important;
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

        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Single"]::before {
            content: "stethoscope" !important;
        }

        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Batch"]::before {
            content: "folder" !important;
        }

        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][href*="Clinical"]::before {
            content: "menu_book" !important;
        }

        /* Ensure sidebar toggle expand button is always visible, high-contrast, and clickable */
        button[data-testid="collapsedSidebarMenu"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            background-color: #0F4C75 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            padding: 4px !important;
            box-shadow: 0 2px 8px rgba(15, 76, 117, 0.2) !important;
            margin-left: 10px !important;
        }

        /* ================================================================
           METRIC CARDS (Professional White Border Cards)
           ================================================================ */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D6E4F0 !important;
            border-radius: 16px !important;
            padding: 18px 22px !important;
            box-shadow: 0 2px 8px rgba(15, 76, 117, 0.04) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 16px rgba(15, 76, 117, 0.08) !important;
        }

        div[data-testid="stMetric"] label {
            color: #64748B !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1E293B !important;
            font-weight: 700 !important;
            font-size: 2.2rem !important;
        }

        /* ================================================================
           BUTTONS (Epic / Cerner Medical Grade Style)
           ================================================================ */
        /* Secondary Button (Outline Style by default) */
        .stButton > button {
            background-color: #FFFFFF !important;
            background-image: none !important;
            color: #3282B8 !important;
            border: 1px solid #D6E4F0 !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.75rem !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: none !important;
        }
        .stButton > button:hover {
            background-color: #F8FAFC !important;
            border-color: #3282B8 !important;
            color: #0F4C75 !important;
            transform: translateY(-1px) !important;
        }

        /* Primary Button (CDSS Blue Gradient Style) */
        .stButton > button[data-testid="stBaseButton-primary"],
        .stButton > button:first-child[style*="background-color: rgb(255, 75, 75)"] {
            background: linear-gradient(135deg, #0F4C75 0%, #3282B8 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #0F4C75 !important;
            box-shadow: 0 2px 4px rgba(15, 76, 117, 0.15) !important;
        }
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, #145E90 0%, #3E91C9 100%) !important;
            border-color: #145E90 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(15, 76, 117, 0.25) !important;
        }

        /* Download Button (Clinical Teal Accent Style) */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #14B8A6 !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.75rem !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 2px 4px rgba(20, 184, 166, 0.15) !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
            border-color: #0F766E !important;
            color: #FFFFFF !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(20, 184, 166, 0.25) !important;
        }

        .stButton > button:disabled {
            background: #EAF2F8 !important;
            border-color: #EAF2F8 !important;
            color: #B0C4DE !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }

        /* ================================================================
           EXPANDER
           ================================================================ */
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            border-radius: 12px !important;
            font-weight: 500 !important;
            color: #1E293B !important;
            border: 1px solid #D6E4F0 !important;
        }

        /* ================================================================
           FILE UPLOADER
           ================================================================ */
        [data-testid="stFileUploader"] {
            border: 2px dashed #3282B8 !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            background-color: #FFFFFF !important;
            transition: all 0.2s ease-in-out !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #0F4C75 !important;
            background-color: #F8FAFC !important;
        }

        /* ================================================================
           TABS
           ================================================================ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px !important;
            border-bottom: 1px solid #D6E4F0 !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0 !important;
            padding: 8px 18px !important;
            font-weight: 500 !important;
            color: #64748B !important;
            background-color: transparent !important;
            border: none !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #0F4C75 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #0F4C75 !important;
        }

        /* ================================================================
           DATAFRAME / TABLE
           ================================================================ */
        .stDataFrame {
            border-radius: 16px !important;
            overflow: hidden !important;
            border: 1px solid #D6E4F0 !important;
        }

        /* Custom HTML Table Styling */
        .cdss-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #D6E4F0;
        }
        .cdss-table thead tr {
            background-color: #0F4C75;
        }
        .cdss-table th {
            padding: 0.85rem 1.2rem;
            text-align: left;
            color: #FFFFFF;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            border-bottom: 2px solid #3282B8;
        }
        .cdss-table td {
            padding: 0.8rem 1.2rem;
            color: #1E293B;
            font-size: 14px;
            border-bottom: 1px solid #EAF2F8;
        }
        .cdss-table tbody tr:nth-child(even) {
            background-color: #F8FAFC;
        }
        .cdss-table tbody tr:hover {
            background-color: #E2F0F9;
        }

        /* ================================================================
           SELECTBOX & NUMBER INPUT & DROPDOWNS
           ================================================================ */
        /* Inputs & Textareas */
        .stNumberInput div > div > input,
        .stTextInput div > div > input,
        .stTextArea div > div > textarea,
        .stDateInput div > div > input {
            border-radius: 12px !important;
            border: 1px solid #D6E4F0 !important;
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            padding: 0.5rem 0.75rem !important;
            font-size: 15px !important;
            height: 42px !important;
            transition: all 0.2s ease-in-out !important;
        }

        /* Selectboxes (Outer Container) */
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 12px !important;
            border: 1px solid #D6E4F0 !important;
            background-color: #FFFFFF !important;
            height: 42px !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        /* Selectbox inner elements text color */
        .stSelectbox div[data-baseweb="select"] div {
            color: #1E293B !important;
        }

        /* Guarantee dropdown options contrast and readability */
        ul[role="listbox"], [data-baseweb="menu"], [data-baseweb="popover"], .stSelectbox div[role="listbox"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }
        ul[role="listbox"] li, [data-baseweb="menu"] li, [role="option"] {
            color: #1E293B !important;
            background-color: #FFFFFF !important;
        }
        ul[role="listbox"] li:hover, [data-baseweb="menu"] li:hover, [role="option"]:hover {
            background-color: #E2F0F9 !important;
            color: #0F4C75 !important;
        }

        .stNumberInput div > div > input:focus,
        .stTextInput div > div > input:focus,
        .stTextArea div > div > textarea:focus,
        .stDateInput div > div > input:focus,
        .stSelectbox div[data-baseweb="select"]:focus-within {
            border-color: #3282B8 !important;
            box-shadow: 0 0 0 3px rgba(50, 130, 184, 0.12) !important;
            outline: none !important;
        }

        /* ================================================================
           PROGRESS BAR CUSTOMIZATION
           ================================================================ */
        div[data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #3282B8, #0F4C75) !important;
            border-radius: 10px !important;
        }
        div[data-testid="stProgress"] {
            background-color: #E2F0F9 !important;
            border-radius: 10px !important;
            height: 8px !important;
        }

        /* ================================================================
           HIDE DEFAULT STREAMLIT CHROME & PRESERVE SIDEBAR CONTROLS
           ================================================================ */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        [data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99 !important;
        }

        [data-testid="stHeader"] [data-testid="stActionButton"],
        [data-testid="stHeader"] button[aria-label="Deploy"],
        [data-testid="stHeader"] button[aria-label="MainMenu"],
        [data-testid="stHeader"] button[aria-label="developer options"],
        [data-testid="stHeader"] #MainMenu {
            display: none !important;
        }

        /* Smooth transitions */
        section[data-testid="stSidebar"] {
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), width 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        section[data-testid="stMain"] {
            transition: margin-left 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        /* ================================================================
           UNIFIED SECTION TITLE
           ================================================================ */
        .section-title-custom {
            color: #0F4C75;
            font-weight: 600;
            font-size: 22px;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-left: 4px solid #14B8A6;
            padding-left: 0.75rem;
            line-height: 1.2;
        }

        /* ================================================================
           UNIFIED CARD STYLE
           ================================================================ */
        .cdss-card {
            background-color: #FFFFFF;
            border: 1px solid #D6E4F0;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(15, 76, 117, 0.04);
            margin-bottom: 20px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .cdss-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(15, 76, 117, 0.08);
            border-color: #3282B8;
        }

        /* ================================================================
           CLINICAL BADGES (WCAG COMPLIANT HIGH CONTRAST)
           ================================================================ */
        .cdss-badge {
            display: inline-block !important;
            padding: 0.3rem 0.75rem !important;
            border-radius: 8px !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.3px !important;
            border: 1px solid transparent !important;
        }
        .cdss-badge-danger {
            background-color: #FEF2F2 !important;
            color: #EF4444 !important;
            border-color: #FCA5A5 !important;
        }
        .cdss-badge-warning {
            background-color: #FFFBEB !important;
            color: #F59E0B !important;
            border-color: #FCD34D !important;
        }
        .cdss-badge-success {
            background-color: #ECFDF5 !important;
            color: #22C55E !important;
            border-color: #A7F3D0 !important;
        }
        .cdss-badge-info {
            background-color: #EFF6FF !important;
            color: #3282B8 !important;
            border-color: #BFDBFE !important;
        }
    </style>
    """


def _get_sidebar_html():
    """Return the sidebar branding HTML."""
    return f"""
    <div style="text-align: left; padding: 1.5rem 1rem 0.5rem 1rem;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.25rem;">
            <div style="font-size: 1.8rem; line-height: 1;">🫁</div>
            <h2 style="
                margin: 0;
                font-size: 1.25rem;
                font-weight: 700;
                color: #ffffff !important;
                letter-spacing: -0.5px;
            ">{APP_TITLE}</h2>
        </div>
        <p style="
            margin: 0;
            font-size: 0.72rem;
            color: #E2F0F9 !important;
            line-height: 1.3;
        ">{APP_SUBTITLE}</p>
        <div style="
            display: inline-block;
            margin-top: 0.5rem;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 4px;
            padding: 0.1rem 0.4rem;
            font-size: 0.65rem;
            color: #E2F0F9 !important;
            font-weight: 600;
        ">v{APP_VERSION}</div>
    </div>
    """


def _get_sidebar_info_html():
    """Return the sidebar info card HTML."""
    return """
    <div style="
        padding: 0.75rem 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border-left: 3px solid #3282B8;
        margin: 0.5rem 1rem;
    ">
        <p style="
            margin: 0;
            font-size: 0.72rem;
            color: #E2F0F9 !important;
            line-height: 1.4;
        ">
            Clinical Decision Support System for Pediatric Oxygen Therapy.
        </p>
    </div>
    """


def _get_sidebar_nav_label():
    """Return sidebar navigation section label."""
    return """
    <div style="padding: 0 1rem; margin-top: 0.75rem;">
        <p style="font-size: 0.65rem; color: #E2F0F9 !important; opacity: 0.7; margin-bottom: 0.4rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
            Menu
        </p>
    </div>
    """


def _get_sidebar_footer_html():
    """Return the sidebar footer HTML."""
    return f"""
    <div style="
        padding: 1.25rem 1rem 1rem 1rem;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    ">
        <p style="
            margin: 0;
            font-size: 0.72rem;
            font-weight: 700;
            color: #E2F0F9 !important;
        ">OxyPredict CDSS</p>
        <p style="
            margin: 0.2rem 0 0 0;
            font-size: 0.62rem;
            color: #B0C4DE !important;
            line-height: 1.3;
        ">
            Undergraduate Research Project
        </p>
        <p style="
            margin: 0.4rem 0 0 0;
            font-size: 0.62rem;
            color: #8892B0 !important;
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
    Return HTML for a standardized clean, minimalist clinical page header.
    Usage: st.markdown(render_page_header("📊", "Dashboard", "..."), unsafe_allow_html=True)
    """
    return f"""
<div style="
    background: linear-gradient(135deg, #0F4C75 0%, #1F6FA7 45%, #3282B8 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2.2rem;
    box-shadow: 0 4px 16px rgba(15, 76, 117, 0.06);
    animation: fadeIn 0.2s ease-out;
    color: #FFFFFF;
">
    <div style="
        font-size: 13px;
        color: #E2F0F9;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.4rem;
        opacity: 0.9;
    ">
        OxyPredict &nbsp;&rsaquo;&nbsp; {title}
    </div>
    <h1 style="
        margin: 0;
        color: #FFFFFF;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    ">
        <span style="font-size: 28px; line-height: 1;">{icon}</span>
        {title}
    </h1>
    <p style="
        margin: 0.4rem 0 0 0;
        color: #E2F0F9;
        font-size: 15px;
        line-height: 1.5;
        font-weight: 400;
        opacity: 0.9;
    ">{subtitle}</p>
</div>
"""


def render_section_divider() -> str:
    """
    Return HTML for an elegant section divider.
    Usage: st.markdown(render_section_divider(), unsafe_allow_html=True)
    """
    return """
<div style="
    margin: 2rem 0;
    border-bottom: 1px solid #D6E4F0;
"></div>
"""


def render_footer() -> str:
    """
    Return HTML for the standardized page footer.
    Usage: st.markdown(render_footer(), unsafe_allow_html=True)
    """
    return f"""
<div style="
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    margin-top: 3.5rem;
    border-top: 1px solid #D6E4F0;
">
    <div style="
        font-size: 16px;
        font-weight: 700;
        color: #0F4C75;
        margin-bottom: 0.3rem;
    ">{APP_TITLE} v{APP_VERSION}</div>
    <p style="margin: 0; color: #64748B; font-size: 13px; line-height: 1.6;">
        Clinical Decision Support System for Oxygen Therapy Assessment
    </p>
    <p style="margin: 0.3rem 0 0 0; color: #8892B0; font-size: 13px;">
        Developed for Academic Research & Clinical Reference
    </p>
</div>
"""


def render_empty_state(icon: str, title: str, description: str) -> str:
    """
    Return HTML for a modern empty state placeholder.
    Usage: st.markdown(render_empty_state("📂", "No File Uploaded", "..."), unsafe_allow_html=True)
    """
    return f"""
<div style="
    background-color: #FFFFFF;
    border: 1px dashed #3282B8;
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
">
    <div style="
        font-size: 2.8rem;
        margin-bottom: 1rem;
        line-height: 1;
    ">{icon}</div>
    <h3 style="
        margin: 0 0 0.5rem 0;
        color: #1E293B;
        font-size: 1.1rem;
        font-weight: 600;
    ">{title}</h3>
    <p style="
        margin: 0;
        color: #64748B;
        font-size: 0.88rem;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    ">{description}</p>
</div>
"""


# =============================================================================
# Python Logging Setup & Status UI Cards
# =============================================================================
import logging
import streamlit as st

# Ensure logs folder exists
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)
logger = logging.getLogger("OxyPredict")


def render_status_card(icon: str, title: str, message: str, border_color: str, bg_color: str, text_color: str) -> str:
    """Helper to render a styled clinical status card."""
    return f"""
    <div style="
        border-left: 5px solid {border_color};
        background-color: {bg_color};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    ">
        <div style="font-size: 1.5rem; line-height: 1; margin-top: 0.1rem;">{icon}</div>
        <div style="flex: 1;">
            <h4 style="margin: 0 0 0.4rem 0; color: #1E293B; font-size: 0.95rem; font-weight: 700;">{title}</h4>
            <div style="margin: 0; color: {text_color}; font-size: 0.88rem; line-height: 1.6; font-weight: 500;">
                {message}
            </div>
        </div>
    </div>
    """


def show_error_card(title: str, message: str, retry_button: bool = False):
    """Display a professional error card to the user."""
    card_html = render_status_card("❌", title, message, "#EF4444", "#FEF2F2", "#B91C1C")
    st.markdown(card_html, unsafe_allow_html=True)
    if retry_button:
        if st.button("🔄 Coba Lagi (Retry)"):
            st.rerun()


def show_warning_card(title: str, message: str):
    """Display a professional warning card to the user."""
    card_html = render_status_card("⚠️", title, message, "#F59E0B", "#FFFBEB", "#B45309")
    st.markdown(card_html, unsafe_allow_html=True)


def show_success_card(title: str, message: str):
    """Display a professional success card to the user."""
    card_html = render_status_card("✅", title, message, "#22C55E", "#ECFDF5", "#15803D")
    st.markdown(card_html, unsafe_allow_html=True)


def show_info_card(title: str, message: str):
    """Display a professional info card to the user."""
    card_html = render_status_card("ℹ️", title, message, "#3282B8", "#EFF6FF", "#1D4ED8")
    st.markdown(card_html, unsafe_allow_html=True)
