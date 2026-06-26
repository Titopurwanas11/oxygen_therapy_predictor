"""
Session Analytics and Uptime Tracker for OxyPredict.
Handles application logs, runtime calculations, and metrics counters.
"""

import datetime
import streamlit as st


def init_analytics_state():
    """Initialize session state variables for telemetry tracking."""
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.datetime.now()
    if "single_predictions_count" not in st.session_state:
        st.session_state.single_predictions_count = 0
    if "batch_predictions_count" not in st.session_state:
        st.session_state.batch_predictions_count = 0
    if "pdf_reports_count" not in st.session_state:
        st.session_state.pdf_reports_count = 0
    if "excel_downloads_count" not in st.session_state:
        st.session_state.excel_downloads_count = 0
    if "csv_downloads_count" not in st.session_state:
        st.session_state.csv_downloads_count = 0
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []


def add_log_entry(action: str, status: str):
    """Add a timestamped entry to the session activity log."""
    init_analytics_state()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.insert(0, {
        "time": now_time,
        "action": action,
        "status": status
    })
    # Keep log within a reasonable size
    if len(st.session_state.activity_log) > 20:
        st.session_state.activity_log = st.session_state.activity_log[:20]


def track_single_prediction(status="Completed"):
    """Increment single prediction telemetry counter and log action."""
    init_analytics_state()
    st.session_state.single_predictions_count += 1
    add_log_entry("Single Prediction", status)


def track_batch_prediction(status="Completed"):
    """Increment batch prediction telemetry counter and log action."""
    init_analytics_state()
    st.session_state.batch_predictions_count += 1
    add_log_entry("Batch Prediction", status)


def track_pdf_report_generated(status="Success"):
    """Increment PDF download telemetry counter and log action."""
    init_analytics_state()
    st.session_state.pdf_reports_count += 1
    add_log_entry("PDF Generated", status)


def track_excel_download(status="Success"):
    """Increment Excel export telemetry counter and log action."""
    init_analytics_state()
    st.session_state.excel_downloads_count += 1
    add_log_entry("Excel Exported", status)


def track_csv_download(status="Success"):
    """Increment CSV export telemetry counter and log action."""
    init_analytics_state()
    st.session_state.csv_downloads_count += 1
    add_log_entry("CSV Exported", status)


def get_session_runtime_str() -> str:
    """Calculate session duration and format it cleanly."""
    init_analytics_state()
    delta = datetime.datetime.now() - st.session_state.session_start_time
    total_seconds = int(delta.total_seconds())
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
