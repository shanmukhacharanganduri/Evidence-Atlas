"""Generate realistic university governance PDFs for the KMEC RAG demo."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/ARIALUNI.TTF"),
    Path("C:/Windows/Fonts/seguisym.ttf"),
]


def register_font() -> str:
    for path in FONT_CANDIDATES:
        if path.exists():
            pdfmetrics.registerFont(TTFont("GovernanceFont", str(path)))
            return "GovernanceFont"
    return "Helvetica"


FONT = register_font()
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="DocumentTitle", parent=styles["Title"], fontName=FONT,
    fontSize=19, leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#12355B"),
    spaceAfter=14,
))
styles.add(ParagraphStyle(
    name="DocHeading", parent=styles["Heading2"], fontName=FONT,
    fontSize=13, leading=17, textColor=colors.HexColor("#12355B"), spaceBefore=9,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="DocBody", parent=styles["BodyText"], fontName=FONT,
    fontSize=10.3, leading=15, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="Small", parent=styles["BodyText"], fontName=FONT,
    fontSize=8.7, leading=11, textColor=colors.HexColor("#435466"),
))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#BCCCDC"))
    canvas.line(0.7 * inch, 0.56 * inch, A4[0] - 0.7 * inch, 0.56 * inch)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#435466"))
    canvas.drawString(0.7 * inch, 0.38 * inch, "KMEC University | Official Governance Record")
    canvas.drawRightString(A4[0] - 0.7 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def heading(text):
    return Paragraph(text, styles["DocHeading"])


def body(text):
    return Paragraph(text, styles["DocBody"])


def table(rows, widths=None):
    result = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12355B")),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A9BED1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def build(name, story):
    DOCS.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(
        str(DOCS / name), pagesize=A4, rightMargin=0.7 * inch, leftMargin=0.7 * inch,
        topMargin=0.65 * inch, bottomMargin=0.75 * inch,
        title=name.replace("_", " ").replace(".pdf", ""), author="KMEC University",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def academic_regulations():
    s = [
        Paragraph("ACADEMIC REGULATIONS 2024", styles["DocumentTitle"]),
        body("<b>KMEC University</b> | Approved by the Academic Council | Effective from Academic Year 2024-25"),
        heading("1. Degree Progression and Credit System"),
        body("All undergraduate degree programmes follow a semester-based credit system. Students progress through eight semesters by earning prescribed course credits, completing mandatory internships, and satisfying academic and conduct requirements."),
        table([
            ["Programme stage", "Credit expectation", "Progression requirement"],
            ["End of Year I", "Minimum 36 earned credits", "No more than two backlog courses"],
            ["End of Year II", "Minimum 84 earned credits", "Completion of foundation and laboratory credits"],
            ["End of Year III", "Minimum 132 earned credits", "Eligibility for capstone project registration"],
            ["Graduation", "Minimum 160 earned credits", "All required courses and project successfully completed"],
        ], [1.3 * inch, 1.7 * inch, 3.55 * inch]),
        heading("2. Minimum CGPA Requirement"),
        body("A student must secure a cumulative grade point average (CGPA) of <b>6.50 or higher</b> on the 10-point scale to be eligible for the award of a degree. Students below 6.50 CGPA may improve eligible course grades under the University improvement examination provisions."),
        body("Credits earned through approved electives, industry immersion and community engagement shall count only when the corresponding assessment has been successfully completed."),
        PageBreak(),
        Paragraph("ACADEMIC REGULATIONS 2024", styles["DocumentTitle"]),
        heading("3. Attendance and Condonation"),
        body("Attendance is recorded for every scheduled lecture, tutorial, laboratory and studio session. A minimum attendance of <b>75%</b> in each registered course is mandatory for permission to appear in the end-semester examination."),
        table([
            ["Attendance band", "Examination eligibility", "Required action"],
            ["75% and above", "Eligible", "No additional action required"],
            ["65% to below 75%", "Condonable", "Submit medical proof and prescribed application to the Dean"],
            ["Below 65%", "Not eligible", "Repeat course attendance in a subsequent offering"],
        ], [1.45 * inch, 1.7 * inch, 3.4 * inch]),
        heading("3.1 Medical Condonation"),
        body("Attendance between <b>65% and 74.99%</b> may be condoned only for documented medical circumstances. The student must submit original medical proof, a treating physician's certificate and an application within seven working days of return to class. Condonation is discretionary and is limited to one course review per semester."),
        heading("3.2 Attendance Review"),
        body("The Department Attendance Review Committee shall publish provisional attendance before examination registration. Students may submit factual corrections within three working days; the Committee's recommendation is forwarded to the Dean for final decision."),
        PageBreak(),
        Paragraph("ACADEMIC REGULATIONS 2024", styles["DocumentTitle"]),
        heading("4. Examination Guidelines"),
        body("End-semester examinations are conducted according to the notified timetable. Students shall carry a valid identity card and admit card, occupy only the assigned seat, and follow invigilator instructions. Use of unauthorized notes, electronic devices or communication during an examination is prohibited."),
        heading("4.1 Assessment and Results"),
        body("Course grades combine continuous assessment and end-semester performance as specified in the approved syllabus. Requests for scrutiny or revaluation must be filed through the Controller of Examinations within the announced period after publication of results."),
        heading("5. Academic Integrity and Disciplinary Committees"),
        body("Suspected malpractice, plagiarism, impersonation or disruption is reported to the Examination Disciplinary Committee. The Committee provides written notice, records the student's explanation and recommends proportionate action. Serious or repeated violations may be referred to the University Academic Integrity Committee."),
        table([
            ["Committee", "Responsibility"],
            ["Examination Disciplinary Committee", "Reviews examination incidents and recommends penalties"],
            ["University Academic Integrity Committee", "Hears serious, repeated or appeal cases involving academic misconduct"],
            ["Controller of Examinations", "Issues final examination orders and maintains official records"],
        ], [2.4 * inch, 4.15 * inch]),
    ]
    build("Academic_Regulations_2024.pdf", s)


def senate_minutes():
    s = [
        Paragraph("SENATE MEETING MINUTES - NOVEMBER 2024", styles["DocumentTitle"]),
        body("<b>KMEC University Senate</b> | Meeting held on 22 November 2024, 10:30 AM | Senate Hall, Administrative Block"),
        heading("1. Attendance and Opening"),
        body("The Vice-Chancellor chaired the meeting. The required quorum was confirmed by the Registrar before the Senate proceeded with the listed business."),
        table([
            ["Name / office", "Role in meeting", "Status"],
            ["Prof. R. Meenakshi, Vice-Chancellor", "Chair", "Present"],
            ["Prof. A. Nandini, Dean - Research", "Senate Member", "Present"],
            ["Dr. S. Raghavan, Dean - Academics", "Senate Member", "Present"],
            ["Ms. P. Kavitha, Registrar", "Secretary to Senate", "Present"],
            ["Heads of Schools and elected faculty members", "Senate Members", "Present as recorded"],
        ], [2.5 * inch, 2.35 * inch, 1.1 * inch]),
        heading("2. Confirmation of Previous Minutes"),
        body("The minutes of the Senate meeting dated 30 August 2024 were circulated in advance. No amendments were proposed. The Senate <b>approved the previous minutes</b> unanimously and authorized the Registrar to place the signed record in the University archives."),
        PageBreak(),
        Paragraph("SENATE MEETING MINUTES - NOVEMBER 2024", styles["DocumentTitle"]),
        heading("Agenda 3.1 - Approval of Research Seed Grants"),
        body("The Dean - Research presented the Seed Grant Scheme to strengthen faculty-led research, support high-quality data collection and encourage publication in internationally recognized top journals. The proposal was reviewed against the Research Office's eligibility and monitoring framework."),
        body("<b>Resolution:</b> The Senate approved Research Seed Grants of <b>₹5,00,000</b> per selected faculty proposal, subject to peer review, institutional ethics clearance where applicable, and quarterly utilization reporting. The grants are intended to support faculty publishing in top journals."),
        table([
            ["Approval item", "Decision"],
            ["Maximum grant", "₹5,00,000 per approved faculty proposal"],
            ["Purpose", "Research activity leading to publications in top journals"],
            ["Oversight", "Research Office review and quarterly utilization reports"],
            ["Effective cycle", "Seed Grant call commencing January 2025"],
        ], [2.1 * inch, 4.45 * inch]),
        heading("Action"),
        body("The Registrar will issue the formal approval memorandum, and the Dean - Research will publish the call for proposals and evaluation calendar."),
        PageBreak(),
        Paragraph("SENATE MEETING MINUTES - NOVEMBER 2024", styles["DocumentTitle"]),
        heading("Agenda 4.2 - Student-to-Faculty Ratio, Computer Science"),
        body("The Dean - Academics presented an intake, teaching-load and laboratory-supervision analysis for the Computer Science Department. Members discussed mentoring capacity, accreditation expectations and the need for timely faculty recruitment."),
        body("<b>Resolution:</b> The Senate approved a Student-to-Faculty Ratio of <b>15:1</b> for the Computer Science Department. Departments shall use this norm for intake planning, faculty recruitment proposals and annual quality assurance reporting."),
        table([
            ["Measure", "Approved norm", "Implementation owner"],
            ["Student-to-Faculty Ratio", "15:1", "Computer Science Department"],
            ["Monitoring", "Each semester", "Dean - Academics"],
            ["Recruitment planning", "Aligned to approved norm", "Registrar and Human Resources"],
        ], [2.2 * inch, 1.7 * inch, 2.65 * inch]),
        heading("5. Closure"),
        body("There being no further business, the Chair thanked the members and declared the meeting closed at 1:15 PM. These minutes are issued subject to the Chair's confirmation."),
    ]
    build("Senate_Meeting_Minutes_Nov2024.pdf", s)


def safety_policy():
    s = [
        Paragraph("CAMPUS SAFETY AND ANTI-RAGGING POLICY", styles["DocumentTitle"]),
        body("<b>KMEC University</b> | Applicable to all students, residents, staff and visitors | Policy revision: July 2024"),
        heading("1. Zero Tolerance Commitment"),
        body("KMEC University maintains a safe, inclusive and dignified learning environment. Ragging, intimidation, harassment, humiliation, coercion or any conduct that compromises a student's well-being is strictly prohibited on campus, in residences, online spaces and during University activities."),
        heading("2. Anti-Ragging Committee Roster"),
        table([
            ["Member", "Designation", "Committee responsibility"],
            ["Dr. S. Raghavan", "Dean - Student Welfare (Chair)", "Overall oversight and escalation"],
            ["Ms. P. Kavitha", "Registrar", "Institutional coordination"],
            ["Dr. N. Fatima", "Counsellor", "Student support and referral"],
            ["Mr. V. Arjun", "Chief Security Officer", "Incident response and evidence preservation"],
            ["Student representatives", "Elected members", "Peer outreach and awareness"],
        ], [1.55 * inch, 2.0 * inch, 2.95 * inch]),
        heading("3. Anti-Ragging Squad Contacts"),
        body("The Anti-Ragging Squad conducts preventive rounds in academic blocks, hostels and common areas. Immediate concerns may be reported through the following contacts:"),
        table([
            ["Service", "Contact"],
            ["24-hour Campus Safety Desk", "+91 90000 11001"],
            ["Anti-Ragging Squad Duty Officer", "+91 90000 11002"],
            ["Student Counselling Centre", "+91 90000 11003"],
            ["Email", "safety@kmec.edu.in"],
        ], [2.6 * inch, 3.9 * inch]),
        PageBreak(),
        Paragraph("CAMPUS SAFETY AND ANTI-RAGGING POLICY", styles["DocumentTitle"]),
        heading("4. Grievance Redressal Procedure"),
        body("A complaint may be made in person, by telephone, by email or through the confidential online reporting channel. Reports may be submitted by an affected student, witness, parent, faculty member or staff member. Retaliation against a complainant or witness is prohibited."),
        table([
            ["Stage", "Timeline", "Process"],
            ["Receipt and safety assessment", "Within 24 hours", "Safety Desk records the matter, offers immediate protection and informs the Chair"],
            ["Preliminary review", "Within 3 days", "Committee gathers initial statements and identifies interim measures"],
            ["Resolution and communication", "Within 7 days", "Written outcome, support plan and any disciplinary referral are communicated"],
        ], [1.6 * inch, 1.25 * inch, 3.65 * inch]),
        heading("4.1 Strict Resolution Timeline"),
        body("The University commits to a <b>strict 7-day resolution timeline</b> from receipt of a grievance. Where a matter requires a statutory investigation or external authority, the complainant will receive a written status update within seven days and at regular intervals thereafter."),
        heading("5. Confidentiality, Support and Appeals"),
        body("Information is shared only with persons who need it to protect safety, conduct a fair review or meet legal obligations. Students may request counselling, academic accommodations and temporary housing measures. An appeal against a disciplinary decision may be submitted to the Vice-Chancellor through the Registrar within seven working days of the written outcome."),
    ]
    build("Campus_Safety_and_AntiRagging_Policy.pdf", s)


if __name__ == "__main__":
    academic_regulations()
    senate_minutes()
    safety_policy()
    print(f"Generated PDFs in {DOCS}")
