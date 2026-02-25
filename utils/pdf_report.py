"""
PDF Report Generator
Creates professional downloadable PDF reports using ReportLab.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Brand colors
PRIMARY = HexColor("#6366f1")    # Indigo
SECONDARY = HexColor("#8b5cf6")  # Violet
DARK = HexColor("#1e1b4b")
LIGHT_BG = HexColor("#f8fafc")
TEXT_DARK = HexColor("#1e293b")
TEXT_MUTED = HexColor("#64748b")


def generate_pdf_report(result: dict, output_path: str = None) -> str:
    """
    Generate a professional PDF report from the full pipeline output.
    
    Args:
        result: Complete PaperTrail pipeline output dict
        output_path: Optional path for the PDF. Auto-generated if None.
        
    Returns:
        Path to the generated PDF file
    """
    if output_path is None:
        os.makedirs("data/reports", exist_ok=True)
        safe_title = result["parsed_paper"]["parsed"]["title"][:50].replace(" ", "_").replace("/", "_")
        output_path = f"data/reports/{safe_title}_report.pdf"

    logger.info(f"Generating PDF report: {output_path}")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    # Custom styles
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'BrandTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=PRIMARY,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        borderPadding=(0, 0, 4, 0),
    ))

    styles.add(ParagraphStyle(
        'SubHeader',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        'BodyText2',
        parent=styles['BodyText'],
        fontSize=10,
        textColor=TEXT_DARK,
        leading=14,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        'Muted',
        parent=styles['BodyText'],
        fontSize=9,
        textColor=TEXT_MUTED,
    ))

    # Build content
    story = []
    parsed = result["parsed_paper"]["parsed"]
    metadata = result["parsed_paper"]["metadata"]

    # ─── Header ───
    story.append(Paragraph("🔬 PaperTrail Analysis Report", styles['BrandTitle']))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles['Muted']
    ))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    story.append(Spacer(1, 12))

    # ─── Paper Info ───
    story.append(Paragraph("📄 Paper Information", styles['SectionHeader']))
    story.append(Paragraph(f"<b>{parsed['title']}</b>", styles['BodyText2']))
    story.append(Paragraph(
        f"Authors: {', '.join(metadata['authors'][:8])}",
        styles['BodyText2']
    ))
    story.append(Paragraph(
        f"Published: {metadata['published'][:10]} | "
        f"Type: {parsed['paper_type']} | "
        f"Categories: {', '.join(metadata['categories'])}",
        styles['Muted']
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Problem:</b> {parsed['problem_statement']}", styles['BodyText2']))
    story.append(Paragraph(f"<b>Methodology:</b> {parsed['methodology_summary']}", styles['BodyText2']))

    # ─── Explanations ───
    if result.get("explanations"):
        exp = result["explanations"]
        story.append(Paragraph("📖 Layered Explanations", styles['SectionHeader']))

        story.append(Paragraph(f"<b>💡 One Sentence:</b> {exp['one_sentence']}", styles['BodyText2']))
        story.append(Paragraph(f"<b>🔑 Key Insight:</b> {exp['key_insight']}", styles['BodyText2']))
        story.append(Spacer(1, 6))

        for level, emoji, label in [
            ("eli5", "🧒", "ELI5 (Simple)"),
            ("undergrad", "🎓", "Undergrad Level"),
            ("expert", "🔬", "Expert Level"),
        ]:
            story.append(Paragraph(f"{emoji} {label}", styles['SubHeader']))
            story.append(Paragraph(exp[level], styles['BodyText2']))

    # ─── Novelty Analysis ───
    if result.get("novelty_analysis"):
        nov = result["novelty_analysis"]
        story.append(Paragraph("🆕 Novelty Analysis", styles['SectionHeader']))
        story.append(Paragraph(
            f"<b>Novelty Score: {nov['novelty_score']}/10</b>",
            styles['BodyText2']
        ))
        story.append(Paragraph(nov['novelty_summary'], styles['BodyText2']))

        story.append(Paragraph("✨ Novel Contributions:", styles['SubHeader']))
        for item in nov["novel_contributions"]:
            story.append(Paragraph(f"• {item}", styles['BodyText2']))

        story.append(Paragraph("📈 Incremental Improvements:", styles['SubHeader']))
        for item in nov["incremental_improvements"]:
            story.append(Paragraph(f"• {item}", styles['BodyText2']))

        story.append(Paragraph("📚 Builds Upon:", styles['SubHeader']))
        for item in nov["builds_upon"]:
            story.append(Paragraph(f"• {item}", styles['BodyText2']))

    # ─── Questions ───
    if result.get("questions"):
        q = result["questions"]
        story.append(Paragraph("❓ Research Questions", styles['SectionHeader']))

        story.append(Paragraph("✅ Questions This Paper Answers:", styles['SubHeader']))
        for item in q["questions_answered"]:
            story.append(Paragraph(f"• {item}", styles['BodyText2']))

        story.append(Paragraph("🔮 Questions Left Open:", styles['SubHeader']))
        for item in q["questions_left_open"]:
            story.append(Paragraph(f"• {item}", styles['BodyText2']))

        story.append(Paragraph("📖 Suggested Follow-Up Reading:", styles['SubHeader']))
        for item in q["follow_up_reading"]:
            story.append(Paragraph(f"• {item}", styles['BodyText2']))

        story.append(Paragraph("💬 Discussion Questions:", styles['SubHeader']))
        for item in q["discussion_questions"]:
            story.append(Paragraph(f"• {item}", styles['BodyText2']))

    # ─── Related Papers ───
    if result.get("related_papers"):
        story.append(Paragraph("🔗 Related Papers", styles['SectionHeader']))
        for p in result["related_papers"]:
            score = p.get("similarity_score", 0)
            story.append(Paragraph(
                f"• <b>{p['title']}</b> (similarity: {score:.2f})",
                styles['BodyText2']
            ))

    # ─── Footer ───
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=1, color=TEXT_MUTED))
    story.append(Paragraph(
        "Generated by PaperTrail — Research Paper Understanding Engine | papertrail.streamlit.app",
        styles['Muted']
    ))

    # Build PDF
    doc.build(story)
    logger.info(f"  ✅ PDF report saved to {output_path}")

    return output_path
