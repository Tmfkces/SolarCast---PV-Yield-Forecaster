"""Build the formatted Word version of the student interview guide.

Run from pv-yield-pk:
    python docs/build_student_interview_guide.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "student_interview_guide.md"
OUTPUT = DOCS / "student_interview_guide.docx"

NAVY = "18324B"
TEAL = "147D78"
PALE_BLUE = "EAF1F5"
PALE_TEAL = "E7F3F1"
MID_GREY = "59636E"


def shade_cell(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.add_run("Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def add_formatted_text(paragraph, text: str) -> None:
    """Render the guide's small inline Markdown subset as Word runs."""
    pattern = re.compile(r"(\*\*.+?\*\*|`.+?`|\*.+?\*)")
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            paragraph.add_run(text[cursor : match.start()])
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor.from_string(TEAL)
        else:
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        cursor = match.end()
    if cursor < len(text):
        paragraph.add_run(text[cursor:])


def add_table(document: Document, lines: list[str]) -> None:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return
    columns = max(len(row) for row in rows)
    table = document.add_table(rows=1, cols=columns)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = True
    for row_index, values in enumerate(rows):
        target = table.rows[0].cells if row_index == 0 else table.add_row().cells
        for index in range(columns):
            cell = target[index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            value = values[index] if index < len(values) else ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            add_formatted_text(paragraph, value)
            for run in paragraph.runs:
                run.font.size = Pt(8)
                if row_index == 0:
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            if row_index == 0:
                shade_cell(cell, NAVY)
            elif row_index % 2 == 0:
                shade_cell(cell, PALE_BLUE)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def add_code_block(document: Document, lines: list[str]) -> None:
    table = document.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade_cell(cell, PALE_BLUE)
    set_cell_margins(cell, top=130, start=160, bottom=130, end=160)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run("\n".join(lines))
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(NAVY)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def add_picture(document: Document, markdown: str) -> None:
    match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", markdown.strip())
    if not match:
        return
    caption, relative_path = match.groups()
    image_path = DOCS / relative_path
    if image_path.exists():
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(image_path), width=Inches(6.0))
        caption_paragraph = document.add_paragraph()
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_paragraph.paragraph_format.space_after = Pt(8)
        run = caption_paragraph.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor.from_string(MID_GREY)


def add_cover(document: Document, markdown: str) -> None:
    title = next((line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), "PV Yield Forecaster")
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(92)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(title)
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor.from_string(NAVY)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("A beginner-friendly project and interview guide")
    run.font.size = Pt(17)
    run.font.color.rgb = RGBColor.from_string(TEAL)

    topic = document.add_paragraph()
    topic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    topic.paragraph_format.space_before = Pt(20)
    run = topic.add_run("Solar energy • NASA POWER weather • Physics model • Machine learning")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor.from_string(MID_GREY)

    note = document.add_table(rows=1, cols=1)
    note.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = note.cell(0, 0)
    shade_cell(cell, PALE_TEAL)
    set_cell_margins(cell, top=220, start=240, bottom=220, end=240)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        "Read this to understand what the application does, how its calculations work, "
        "what the results mean, and what you should be careful not to overclaim."
    )
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor.from_string(NAVY)

    attribution = document.add_paragraph()
    attribution.alignment = WD_ALIGN_PARAGRAPH.CENTER
    attribution.paragraph_format.space_before = Pt(64)
    run = attribution.add_run("Study reference for research preparation")
    run.font.size = Pt(10)
    run.italic = True
    run.font.color.rgb = RGBColor.from_string(MID_GREY)
    document.add_page_break()


def add_contents(document: Document, markdown: str) -> None:
    heading = document.add_heading("Quick contents", level=1)
    heading.paragraph_format.keep_with_next = True
    for line in markdown.splitlines():
        if line.startswith("## "):
            text = line[3:].strip()
            if text == "How to use this guide":
                continue
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.space_after = Pt(1)
            add_formatted_text(paragraph, text)
    document.add_page_break()


def build() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.66)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)
    section.start_type = WD_SECTION.CONTINUOUS

    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor.from_string("263442")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    for style_name, size, color in [
        ("Title", 28, NAVY),
        ("Heading 1", 18, NAVY),
        ("Heading 2", 14, TEAL),
        ("Heading 3", 11, NAVY),
    ]:
        style = document.styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(12 if style_name != "Heading 1" else 4)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.text = "PV YIELD FORECASTER  |  STUDENT INTERVIEW GUIDE"
    header.style = document.styles["Caption"]
    header.runs[0].font.color.rgb = RGBColor.from_string(TEAL)
    header.runs[0].font.bold = True
    header.runs[0].font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    add_page_number(footer)
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(MID_GREY)

    document.core_properties.title = "PV Yield Forecaster: A Beginner's Guide for Research Interviews"
    document.core_properties.subject = "Project explanation, methods, metrics, limitations, and interview preparation"
    document.core_properties.keywords = "photovoltaic, PV, solar energy, NASA POWER, machine learning, research"

    add_cover(document, markdown)
    add_contents(document, markdown)

    lines = markdown.splitlines()
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []
    skip_first_title = True
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            if table_lines:
                add_table(document, table_lines)
                table_lines = []
            if in_code:
                add_code_block(document, code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if stripped.startswith("|"):
            table_lines.append(stripped)
            index += 1
            continue
        if table_lines:
            add_table(document, table_lines)
            table_lines = []
        if not stripped or stripped == "---":
            index += 1
            continue
        if stripped.startswith("# "):
            if skip_first_title:
                skip_first_title = False
            else:
                document.add_heading(stripped[2:].strip(), level=0)
            index += 1
            continue
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            if title.startswith("13. A short interview explanation") or title.startswith("16. Glossary"):
                document.add_page_break()
            document.add_heading(title, level=1)
            index += 1
            continue
        if stripped.startswith("### "):
            document.add_heading(stripped[4:].strip(), level=2)
            index += 1
            continue
        if stripped.startswith("!["):
            add_picture(document, stripped)
            index += 1
            continue
        if stripped.startswith("> "):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.2)
            paragraph.paragraph_format.right_indent = Inches(0.2)
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(8)
            add_formatted_text(paragraph, stripped[2:])
            for run in paragraph.runs:
                run.italic = True
                run.font.color.rgb = RGBColor.from_string(NAVY)
            index += 1
            continue
        if stripped.startswith("- "):
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.space_after = Pt(2)
            add_formatted_text(paragraph, stripped[2:])
            index += 1
            continue
        ordered = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered:
            paragraph = document.add_paragraph(style="List Number")
            paragraph.paragraph_format.space_after = Pt(2)
            add_formatted_text(paragraph, ordered.group(1))
            index += 1
            continue
        paragraph = document.add_paragraph()
        add_formatted_text(paragraph, stripped)
        index += 1

    if table_lines:
        add_table(document, table_lines)
    if in_code and code_lines:
        add_code_block(document, code_lines)
    document.save(OUTPUT)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
