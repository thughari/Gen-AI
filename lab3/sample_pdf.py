from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os

# Ensure directory exists
os.makedirs("./lab3/sample_pdfs", exist_ok=True)

doc = SimpleDocTemplate("./lab3/sample_pdfs/health_policy.pdf", pagesize=letter)
styles = getSampleStyleSheet()

story = []

# Title
story.append(Paragraph("InsureSafe Health Policy - Gold Plan", styles["Title"]))
story.append(Paragraph("Policy Number: ISH-2024-GOLD-001", styles["Normal"]))
story.append(Spacer(1, 12))

# Coverage Section
story.append(Paragraph("Coverage Details", styles["Heading1"]))

coverage = [
    ["Benefit", "Coverage Amount", "Notes"],
    ["Sum Insured", "Rs. 10,00,000", "Per Policy Year"],
    ["Hospitalisation", "Up to SI", "Min 24 hours admission"],
    ["Day Care", "Up to SI", "540+ listed procedures"],
    ["Pre-hospitalisation", "60 days", "Related expenses covered"],
    ["Post-hospitalisation", "90 days", "Related expenses covered"],
]

table = Table(coverage)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("GRID", (0, 0), (-1, -1), 1, colors.black),
]))

story.append(table)
story.append(Spacer(1, 12))

# Exclusions
story.append(Paragraph("Exclusions", styles["Heading1"]))

exclusions = [
    "1. Pre-existing conditions: 3-year waiting period applies.",
    "2. Cosmetic or plastic surgery not medically necessary.",
    "3. Dental treatment unless arising from accident.",
    "4. Maternity expenses in first 2 policy years.",
]

for ex in exclusions:
    story.append(Paragraph(ex, styles["Normal"]))

# Build PDF
doc.build(story)

print("âœ… Sample PDF created at sample_pdfs/health_policy.pdf")