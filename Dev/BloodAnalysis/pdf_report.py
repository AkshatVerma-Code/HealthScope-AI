"""
PDF Report Generator using reportlab (pure Python, no native dependencies).
"""

import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────
NAVY      = colors.HexColor('#0a0f1e')
TEAL      = colors.HexColor('#00d4aa')
PURPLE    = colors.HexColor('#7b6cf6')
DARK_CARD = colors.HexColor('#141929')
TEXT_MAIN = colors.HexColor('#e8eaf6')
TEXT_SUB  = colors.HexColor('#9ca3af')
RED       = colors.HexColor('#ef4444')
ORANGE    = colors.HexColor('#f59e0b')
GREEN     = colors.HexColor('#10b981')
WHITE     = colors.white


def _risk_color(status: str):
    if 'High' in status:
        return RED
    elif 'Moderate' in status:
        return ORANGE
    return GREEN


def generate_pdf(blood_report) -> bytes:
    """
    Generate a PDF report for a BloodReport instance.
    Returns raw bytes of the PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'Title', parent=styles['Normal'],
        fontSize=24, textColor=TEAL, fontName='Helvetica-Bold',
        alignment=TA_CENTER, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=11, textColor=TEXT_SUB, fontName='Helvetica',
        alignment=TA_CENTER, spaceAfter=2,
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Normal'],
        fontSize=13, textColor=TEAL, fontName='Helvetica-Bold',
        spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, textColor=colors.black, fontName='Helvetica',
        spaceAfter=3,
    )
    rec_style = ParagraphStyle(
        'Rec', parent=styles['Normal'],
        fontSize=10, textColor=colors.black, fontName='Helvetica',
        leftIndent=12, spaceAfter=4,
    )

    patient = blood_report.session
    story = []

    # ── Header ──────────────────────────────────────────
    story.append(Paragraph("🧬 LabSense AI", title_style))
    story.append(Paragraph("AI-Powered Health Report", subtitle_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL))
    story.append(Spacer(1, 0.4 * cm))

    # ── Patient Details ──────────────────────────────────
    story.append(Paragraph("Patient Information", section_style))
    patient_data = [
        ['Full Name', patient.full_name, 'Age', str(patient.age)],
        ['Gender', patient.gender, 'Contact', patient.contact or '—'],
        ['Report Type', blood_report.get_report_type_display(), 'Generated', datetime.now().strftime('%d %b %Y, %I:%M %p')],
    ]
    patient_table = Table(patient_data, colWidths=[3.5 * cm, 6 * cm, 3.5 * cm, 5.5 * cm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f4f8')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#374151')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Extracted Parameters ─────────────────────────────
    params = blood_report.extracted_params
    if params:
        story.append(Paragraph("Extracted Laboratory Parameters", section_style))
        param_rows = [['Parameter', 'Extracted Value']]
        for key, val in params.items():
            param_rows.append([str(key), str(val) if val is not None else '—'])

        param_table = Table(param_rows, colWidths=[9 * cm, 9.5 * cm])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TEAL),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#f0fdf4')]),
            ('PADDING', (0, 0), (-1, -1), 7),
        ]))
        story.append(param_table)
        story.append(Spacer(1, 0.4 * cm))

    # ── Disease Risk Predictions ─────────────────────────
    predictions = blood_report.predictions
    if predictions:
        story.append(Paragraph("AI Disease Risk Assessment", section_style))
        pred_rows = [['Disease', 'Risk Score', 'Status']]
        for disease, data in predictions.items():
            if not data.get('available'):
                pred_rows.append([disease, '—', data.get('status', 'Unavailable')])
            else:
                risk = data.get('risk', 0)
                status = data.get('status', '—')
                pred_rows.append([disease, f"{risk:.1f}%", status])

        pred_table = Table(pred_rows, colWidths=[7 * cm, 5 * cm, 6.5 * cm])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#faf5ff')]),
            ('PADDING', (0, 0), (-1, -1), 7),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 0.4 * cm))

    # ── Recommendations ──────────────────────────────────
    recommendations = blood_report.recommendations
    if recommendations:
        story.append(Paragraph("Personalized Health Recommendations", section_style))
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", rec_style))
        story.append(Spacer(1, 0.3 * cm))

    # ── Disclaimer ───────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=TEXT_SUB))
    story.append(Spacer(1, 0.2 * cm))
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=TEXT_SUB, fontName='Helvetica-Oblique',
        alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "⚠️ This report is generated by an AI system and is intended for informational purposes only. "
        "It does not constitute medical advice. Please consult a qualified healthcare professional "
        "for diagnosis and treatment decisions.",
        disclaimer_style
    ))

    doc.build(story)
    return buffer.getvalue()
