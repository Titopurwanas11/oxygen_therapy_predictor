"""
Clinical Prediction Report PDF Generator for OxyPredict.
Uses ReportLab to generate a highly professional, clinical-grade CDSS prediction report.
"""

import os
import datetime
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from utils.config import ALL_FEATURES


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that tracks pages to draw dynamic 'Page X of Y' footers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(HexColor("#64748b"))

        # Footer separator line
        self.setStrokeColor(HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)  # Margins: 0.75in (54pt) from left/right

        # Footer text as requested
        footer_text_left = "Dihasilkan oleh: OxyPredict Clinical Decision Support System"
        footer_text_center = ""
        footer_text_right = f"© 2026 | Halaman {self._pageNumber} dari {page_count}"

        self.drawString(54, 30, footer_text_left)
        self.drawCentredString(306, 30, footer_text_center)
        self.drawRightString(558, 30, footer_text_right)
        self.restoreState()


def clean_feat_name(name):
    """Clean feature names to match hospital report standards."""
    n = name.lower()
    if "wheezing" in n: return "Wheezing"
    if "nasal flaring" in n: return "Nasal Flaring"
    if "cyanosis" in n: return "Cyanosis"
    if "oxygen saturation" in n or "sao2" in n: return "Oxygen Saturation (SaO2)"
    if "respiratory rate" in n: return "Respiratory Rate"
    if "temperature" in n: return "Axillary Temperature"
    if "heart rate" in n: return "Heart Rate"
    if "restlessness" in n: return "Restlessness"
    if "sleepiness" in n: return "Unusual Sleepiness"
    if "crackles" in n: return "Crackles"
    if "rhonchi" in n: return "Rhonchi"
    if "c-reactive" in n or "crp" in n: return "C-Reactive Protein (CRP)"
    if "procalcitonin" in n: return "Procalcitonin"
    return name


def generate_pdf_report(
    patient_data: dict,
    prediction: str,
    probability: float,
    confidence: str,
    risk_level: str,
    clinical_summary: str,
    top_shap_features: list,
    shap_values: list,
    feature_names: list,
    recommendation: dict = None,
) -> bytes:
    """
    Generate a professional clinical CDSS report PDF.

    Returns:
        bytes: Raw PDF bytes.
    """
    buffer = BytesIO()

    # Document setup (0.75-inch margins, leave 1.0-inch at bottom for footer)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=HexColor("#0a2e52"),
    )
    style_subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=HexColor("#64748b"),
        textTransform="uppercase",
    )
    style_section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=HexColor("#0a2e52"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )
    style_body = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=HexColor("#334155"),
    )
    style_body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=style_body,
        fontName="Helvetica-Bold",
    )
    style_table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=HexColor("#475569"),
        textTransform="uppercase",
    )
    style_badge = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
    )

    story = []

    # ─────────────────────────────────────────────────────────────────────────
    # HEADER SECTION
    # ─────────────────────────────────────────────────────────────────────────
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    header_left = [
        Paragraph("🫁 OxyPredict", style_title),
        Paragraph("Sistem Dukungan Keputusan Klinis", style_subtitle),
        Spacer(1, 2),
        Paragraph("Laporan Prediksi", ParagraphStyle("HeaderPR", parent=style_body_bold, textColor=HexColor("#1e40af"))),
    ]

    header_right = [
        Paragraph(f"<b>Tanggal:</b> {date_str}", style_body),
        Paragraph(f"<b>Waktu:</b> {time_str}", style_body),
    ]

    header_table = Table(
        [[header_left, header_right]],
        colWidths=[300, 204],
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))

    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cbd5e1"), spaceBefore=5, spaceAfter=10))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: PATIENT INFORMATION
    # ─────────────────────────────────────────────────────────────────────────
    age_val = patient_data.get("Age (months)", "N/A")
    gender_val = patient_data.get("Gender", "N/A")
    weight_val = patient_data.get("Weight (Kg)", "N/A")
    height_val = patient_data.get("Height (cm)", "N/A")
    temp_val = patient_data.get("Axillary temperature (°C)", "N/A")
    rr_val = patient_data.get("Respiratory rate", "N/A")
    hr_val = patient_data.get("Heart rate", "N/A")
    sao2_val = patient_data.get("Oxygen saturation (SaO2) at admission", "N/A")
    
    age_str = f"{age_val} months" if age_val != "N/A" else "N/A"
    weight_str = f"{weight_val} Kg" if weight_val != "N/A" else "N/A"
    height_str = f"{height_val} cm" if height_val != "N/A" else "N/A"
    temp_str = f"{temp_val}°C" if temp_val != "N/A" else "N/A"
    rr_str = f"{rr_val} bpm" if rr_val != "N/A" else "N/A"
    hr_str = f"{hr_val} bpm" if hr_val != "N/A" else "N/A"
    sao2_str = f"{sao2_val}%" if sao2_val != "N/A" else "N/A"

    patient_rows = [
        [Paragraph("<b>Fitur Klinis</b>", style_table_header), Paragraph("<b>Nilai Pasien</b>", style_table_header)],
        [Paragraph("Usia Pasien", style_body), Paragraph(age_str, style_body_bold)],
        [Paragraph("Jenis Kelamin", style_body), Paragraph(str(gender_val), style_body_bold)],
        [Paragraph("Berat Badan", style_body), Paragraph(weight_str, style_body_bold)],
        [Paragraph("Tinggi Badan", style_body), Paragraph(height_str, style_body_bold)],
        [Paragraph("Suhu Tubuh", style_body), Paragraph(temp_str, style_body_bold)],
        [Paragraph("Laju Pernapasan", style_body), Paragraph(rr_str, style_body_bold)],
        [Paragraph("Laju Denyut Jantung", style_body), Paragraph(hr_str, style_body_bold)],
        [Paragraph("Saturasi Oksigen (SaO2)", style_body), Paragraph(sao2_str, style_body_bold)],
        [Paragraph("Tanggal Diagnosa", style_body), Paragraph(date_str, style_body_bold)]
    ]

    patient_table = Table(patient_rows, colWidths=[252, 252])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f1f5f9")),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(Paragraph("📋 Informasi Pasien", style_section_heading))
    story.append(patient_table)
    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: MODEL PREDICTION (PROMINENT CARD)
    # ─────────────────────────────────────────────────────────────────────────
    is_yes = prediction == "Yes" or "Need" in prediction
    pred_outcome_text = "Butuh Terapi Oksigen" if is_yes else "Tidak Butuh Terapi Oksigen"
    
    # Determine color scheme based on clinical risk level
    risk_cleaned = risk_level.lower()
    if "high" in risk_cleaned or "very high" in risk_cleaned:
        card_bg = "#fdf2f2"
        card_border = "#fecaca"
        risk_color_hex = "#dc2626"
    elif "moderate" in risk_cleaned:
        card_bg = "#fef9c3"
        card_border = "#fef08a"
        risk_color_hex = "#ca8a04"
    else: # Low / Low-Moderate
        card_bg = "#f0fdf4"
        card_border = "#bbf7d0"
        risk_color_hex = "#16a34a"

    style_pred_outcome = ParagraphStyle(
        "PredOutcome",
        parent=style_badge,
        textColor=HexColor(risk_color_hex),
    )

    # Format probability nicely (make sure it's shown as percentage)
    prob_val = probability if probability > 1.0 else probability * 100

    prediction_card_content = [
        [
            Paragraph("<b>RANGKUMAN PREDIKSI MODEL</b>", style_subtitle),
            ""
        ],
        [
            Paragraph("Prediksi:", style_body),
            Paragraph(f"<b>{pred_outcome_text}</b>", style_pred_outcome)
        ],
        [
            Paragraph("Probabilitas Prediksi:", style_body),
            Paragraph(f"<b>{prob_val:.1f}%</b>", style_body_bold)
        ],
        [
            Paragraph("Tingkat Keyakinan:", style_body),
            Paragraph(f"<b>{confidence}</b>", style_body_bold)
        ],
        [
            Paragraph("Tingkat Risiko:", style_body),
            Paragraph(f"<font color='{risk_color_hex}'><b>{risk_level}</b></font>", style_body_bold)
        ]
    ]

    prediction_card = Table(prediction_card_content, colWidths=[180, 324])
    prediction_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(card_bg)),
        ("BOX", (0, 0), (-1, -1), 1.2, HexColor(card_border)),
        ("SPAN", (0, 0), (1, 0)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (1, 0), 1, HexColor(card_border)),
    ]))

    story.append(Paragraph("🔮 Prediksi Model", style_section_heading))
    story.append(prediction_card)
    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: RINGKASAN KLINIS
    # ─────────────────────────────────────────────────────────────────────────
    summary_box_content = [
        [Paragraph("🩺 Ringkasan Klinis", style_section_heading)],
        [Paragraph(clinical_summary, style_body)]
    ]
    summary_box = Table(summary_box_content, colWidths=[504])
    summary_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 1, HexColor("#e2e8f0")),
        ("LINELEFT", (0, 0), (-1, -1), 5, HexColor("#1e40af")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(summary_box)
    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: TOP CONTRIBUTING FEATURES (SHAP TABLE)
    # ─────────────────────────────────────────────────────────────────────────
    shap_rows = [[
        Paragraph("Fitur", style_table_header),
        Paragraph("Nilai Pasien", style_table_header),
        Paragraph("Kontribusi SHAP", style_table_header),
        Paragraph("Efek Klinis", style_table_header)
    ]]

    # Ensure we cleanly take top 10 from list
    sorted_shap = sorted(shap_values, key=lambda x: abs(x["shap_value"]), reverse=True)[:10]

    for sv in sorted_shap:
        name_clean = clean_feat_name(sv["feature"])
        patient_val = str(sv["patient_value"])
        val = sv["shap_value"]
        
        sign = "+" if val > 0 else ""
        shap_str = f"{sign}{val:.4f}"
        
        effect = "Increase Risk" if val > 0 else "Reduce Risk"
        effect_color = "#dc2626" if val > 0 else "#2563eb"

        shap_rows.append([
            Paragraph(name_clean, style_body),
            Paragraph(patient_val, style_body_bold),
            Paragraph(f"<b>{shap_str}</b>", style_body),
            Paragraph(f"<font color='{effect_color}'><b>{effect}</b></font>", style_body)
        ])

    shap_table = Table(shap_rows, colWidths=[164, 100, 110, 130])
    shap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f1f5f9")),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(Paragraph("📊 Fitur Kontributor Utama", style_section_heading))
    story.append(shap_table)
    story.append(Spacer(1, 8))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: CATATAN INTERPRETABILITAS
    # ─────────────────────────────────────────────────────────────────────────
    interpretability_text = (
        "<b>Catatan Interpretabilitas:</b><br/>"
        "• Nilai SHAP positif meningkatkan kemungkinan terapi oksigen.<br/>"
        "• Nilai SHAP negatif menurunkan kemungkinan terapi oksigen.<br/>"
        "• Pentingnya fitur dihitung dari kontribusi numerik fitur terhadap prediksi."
    )
    interpretability_box = Table([[Paragraph(interpretability_text, style_body)]], colWidths=[504])
    interpretability_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(interpretability_box)
    story.append(Spacer(1, 10))

    if recommendation:
        # Determine color of Priority and badge
        prio = recommendation.get("priority", "Low")
        if prio == "Emergency":
            prio_color = "#dc2626"
            prio_bg = "#fdf2f2"
            prio_border = "#fecaca"
        elif prio == "High":
            prio_color = "#ea580c"
            prio_bg = "#fffbeb"
            prio_border = "#fed7aa"
        elif prio == "Medium":
            prio_color = "#ca8a04"
            prio_bg = "#fef9c3"
            prio_border = "#fef08a"
        else:
            prio_color = "#16a34a"
            prio_bg = "#f0fdf4"
            prio_border = "#bbf7d0"

        style_prio = ParagraphStyle(
            "PrioStyle",
            parent=style_body_bold,
            textColor=HexColor(prio_color),
        )

        rec_table_content = [
            [
                Paragraph("<b>RINGKASAN REKOMENDASI KLINIS CDSS</b>", style_subtitle),
                ""
            ],
            [
                Paragraph("Tingkat Prioritas:", style_body),
                Paragraph(f"<b>{prio} Priority</b>", style_prio)
            ],
            [
                Paragraph("Sumber Rekomendasi:", style_body),
                Paragraph("Mesin Rekomendasi CDSS Berbasis Aturan", style_body)
            ]
        ]
        rec_info_table = Table(rec_table_content, colWidths=[180, 324])
        rec_info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(prio_bg)),
            ("BOX", (0, 0), (-1, -1), 1.2, HexColor(prio_border)),
            ("SPAN", (0, 0), (1, 0)),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (1, 0), 1, HexColor(prio_border)),
        ]))

        # Format Actions & Monitoring bullets
        actions_bullet = "<br/>".join([f"• {act}" for act in recommendation.get("clinical_action", [])])
        monitoring_bullet = "<br/>".join([f"• {mon}" for mon in recommendation.get("monitoring", [])])

        rec_bullets_content = [
            [Paragraph("<b>Tindakan Klinis:</b>", style_body_bold), Paragraph("<b>Panduan Monitoring:</b>", style_body_bold)],
            [Paragraph(actions_bullet, style_body), Paragraph(monitoring_bullet, style_body)]
        ]
        rec_bullets_table = Table(rec_bullets_content, colWidths=[252, 252])
        rec_bullets_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))

        # Yellow Disclaimer box for Recommendations
        rec_disclaimer_text = " ".join(recommendation.get("notes", []))
        rec_disclaimer_box = Table([[Paragraph(f"<b>Disclaimer & Panduan CDSS:</b><br/>{rec_disclaimer_text}", style_body)]], colWidths=[504])
        rec_disclaimer_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#fefbeb")),
            ("BOX", (0, 0), (-1, -1), 1, HexColor("#fef08a")),
            ("LINELEFT", (0, 0), (-1, -1), 4, HexColor("#eab308")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))

        story.append(Paragraph("🩺 Mesin Rekomendasi Klinis", style_section_heading))
        story.append(rec_info_table)
        story.append(Spacer(1, 6))
        story.append(rec_bullets_table)
        story.append(Spacer(1, 6))
        story.append(rec_disclaimer_box)
        story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: MODEL INFORMATION
    # ─────────────────────────────────────────────────────────────────────────
    model_info_rows = [
        [Paragraph("<b>Parameter / Metrik</b>", style_table_header), Paragraph("<b>Nilai / Spesifikasi</b>", style_table_header)],
        [Paragraph("Algoritma", style_body), Paragraph("Random Forest Classifier", style_body_bold)],
        [Paragraph("Jumlah Pohon", style_body), Paragraph("800", style_body_bold)],
        [Paragraph("Akurasi", style_body), Paragraph("89.44%", style_body_bold)],
        [Paragraph("F1 Makro", style_body), Paragraph("86.20%", style_body_bold)],
        [Paragraph("ROC-AUC", style_body), Paragraph("90.93%", style_body_bold)],
        [Paragraph("Data", style_body), Paragraph("801 Pasien Pediatrik", style_body_bold)],
        [Paragraph("Fitur", style_body), Paragraph("44 Variabel Klinis", style_body_bold)]
    ]

    model_table = Table(model_info_rows, colWidths=[252, 252])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f1f5f9")),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # Wrap model information & disclaimer in a KeepTogether to avoid awkward page splitting
    model_section = [
        Paragraph("⚙️ Informasi Model", style_section_heading),
        model_table,
        Spacer(1, 10)
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7: DISCLAIMER
    # ─────────────────────────────────────────────────────────────────────────
    disclaimer_text = (
        "<b>Disclaimer</b><br/>"
        "Laporan ini dihasilkan secara otomatis oleh Sistem Dukungan Keputusan Klinis OxyPredict. "
        "Prediksi dimaksudkan untuk membantu tenaga kesehatan dan tidak boleh menggantikan penilaian klinis, "
        "pemeriksaan pasien, temuan laboratorium, atau pedoman terapi institusi. "
        "Keputusan medis akhir tetap menjadi tanggung jawab tenaga kesehatan berlisensi."
    )
    disclaimer_box = Table([[Paragraph(disclaimer_text, style_body)]], colWidths=[504])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#fefbeb")),
        ("BOX", (0, 0), (-1, -1), 1, HexColor("#fef08a")),
        ("LINELEFT", (0, 0), (-1, -1), 4, HexColor("#eab308")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    
    model_section.append(disclaimer_box)
    story.append(KeepTogether(model_section))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
