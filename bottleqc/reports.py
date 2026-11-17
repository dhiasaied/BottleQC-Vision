"""Report export: CSV, Excel (xlsx) and PDF."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

from .storage import Storage

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "snapshots"
COLUMNS = ["id", "session_id", "time", "track_id", "status", "cap_present", "cap_color",
           "color_conf", "det_conf", "observations", "snapshot"]


def _fmt_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


def _rows(storage: Storage, session_id, since, until):
    rows = storage.inspections(session_id=session_id, since=since, until=until, limit=1_000_000)
    rows.reverse()  # chronological
    for r in rows:
        r["time"] = _fmt_ts(r["ts"])
    return rows


def _title(storage, session_id):
    if session_id:
        s = storage.session(session_id) or {}
        return f"Session #{session_id} - {s.get('source', '')}"
    return "All sessions"


def export_csv(storage: Storage, session_id=None, since=None, until=None) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in _rows(storage, session_id, since, until):
        w.writerow(r)
    return buf.getvalue().encode("utf-8-sig")


def export_xlsx(storage: Storage, session_id=None, since=None, until=None) -> bytes:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, PieChart, Reference
    from openpyxl.styles import Alignment, Font, PatternFill

    rows = _rows(storage, session_id, since, until)
    summ = storage.summary(session_id, since, until)
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    head = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="1F2937")
    ws["A1"] = "Bottling Line Quality Report"
    ws["A1"].font = Font(bold=True, size=16)
    ws["A2"] = _title(storage, session_id)
    ws["A3"] = f"Generated {datetime.now():%Y-%m-%d %H:%M}"
    kpis = [("Total inspected", summ["total"]), ("Passed", summ["ok"]), ("Defects", summ["defects"]),
            ("Missing cap", summ["no_cap"]), ("Wrong cap colour", summ["wrong_color"]),
            ("Yield %", summ["yield"] if summ["yield"] is not None else "-"),
            ("First item", _fmt_ts(summ["first_ts"])), ("Last item", _fmt_ts(summ["last_ts"]))]
    ws.append([])
    ws.append(["Metric", "Value"])
    for c in ws[5]:
        c.font, c.fill = head, fill
    for k in kpis:
        ws.append(list(k))
    start = ws.max_row + 2
    ws.cell(start, 1, "Status").font = Font(bold=True)
    ws.cell(start, 2, "Count").font = Font(bold=True)
    for i, (k, v) in enumerate((("OK", summ["ok"]), ("NO_CAP", summ["no_cap"]), ("WRONG_COLOR", summ["wrong_color"])), 1):
        ws.cell(start + i, 1, k)
        ws.cell(start + i, 2, v)
    if summ["total"]:
        pie = PieChart()
        pie.title = "Inspection results"
        pie.add_data(Reference(ws, min_col=2, min_row=start, max_row=start + 3), titles_from_data=True)
        pie.set_categories(Reference(ws, min_col=1, min_row=start + 1, max_row=start + 3))
        ws.add_chart(pie, "D5")
    cstart = start + 5
    ws.cell(cstart, 1, "Cap colour").font = Font(bold=True)
    ws.cell(cstart, 2, "Count").font = Font(bold=True)
    for i, (k, v) in enumerate(sorted(summ["by_color"].items(), key=lambda x: -x[1]), 1):
        ws.cell(cstart + i, 1, k)
        ws.cell(cstart + i, 2, v)
    if summ["by_color"]:
        bar = BarChart()
        bar.title = "Cap colours"
        bar.add_data(Reference(ws, min_col=2, min_row=cstart, max_row=cstart + len(summ["by_color"])), titles_from_data=True)
        bar.set_categories(Reference(ws, min_col=1, min_row=cstart + 1, max_row=cstart + len(summ["by_color"])))
        ws.add_chart(bar, "D21")
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 22

    wd = wb.create_sheet("Inspections")
    wd.append(COLUMNS)
    for c in wd[1]:
        c.font, c.fill = head, fill
    red = PatternFill("solid", fgColor="FDE2E2")
    amber = PatternFill("solid", fgColor="FEF3C7")
    for r in rows:
        wd.append([r.get(k) for k in COLUMNS])
        if r["status"] != "OK":
            for c in wd[wd.max_row]:
                c.fill = red if r["status"] == "NO_CAP" else amber
    for col, w in zip("ABCDEFGHIJK", (8, 10, 20, 9, 14, 11, 11, 11, 9, 12, 30)):
        wd.column_dimensions[col].width = w
    wd.freeze_panes = "A2"
    wd.auto_filter.ref = wd.dimensions
    for row in wd.iter_rows(min_row=2):
        row[2].alignment = Alignment(horizontal="left")
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def export_pdf(storage: Storage, session_id=None, since=None, until=None) -> bytes:
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    rows = _rows(storage, session_id, since, until)
    summ = storage.summary(session_id, since, until)
    ss = getSampleStyleSheet()
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Bottling Line Quality Report")
    story = [Paragraph("Bottling Line Quality Report", ss["Title"]),
             Paragraph(f"{_title(storage, session_id)} &nbsp;·&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}", ss["Normal"]),
             Spacer(1, 8 * mm)]

    kpi = [["Total", "Passed", "Defects", "Missing cap", "Wrong colour", "Yield"],
           [summ["total"], summ["ok"], summ["defects"], summ["no_cap"], summ["wrong_color"],
            f"{summ['yield']:.1f}%" if summ["yield"] is not None else "-"]]
    t = Table(kpi, colWidths=[29 * mm] * 6)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 1), (-1, 1), 16), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 1), (-1, 1), 8), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ]))
    story += [t, Spacer(1, 6 * mm)]
    story.append(Paragraph(f"Period: {_fmt_ts(summ['first_ts']) or '-'} → {_fmt_ts(summ['last_ts']) or '-'}", ss["Normal"]))

    if summ["total"]:
        d = Drawing(170 * mm, 60 * mm)
        pie = Pie()
        pie.x, pie.y, pie.width, pie.height = 10, 10, 140, 140
        vals = [summ["ok"], summ["no_cap"], summ["wrong_color"]]
        labels = ["OK", "Missing cap", "Wrong colour"]
        pie.data = [v if v else 0.0001 for v in vals]
        pie.labels = [f"{l} ({v})" for l, v in zip(labels, vals)]
        for i, c in enumerate(("#22c55e", "#ef4444", "#f59e0b")):
            pie.slices[i].fillColor = colors.HexColor(c)
        d.add(pie)
        story += [Spacer(1, 4 * mm), d]

    defects = [r for r in rows if r["status"] != "OK"][:12]
    if defects:
        story += [Paragraph("Defect gallery", ss["Heading2"])]
        cells, line = [], []
        for r in defects:
            p = SNAP_DIR / (r["snapshot"] or "")
            img = Image(str(p), width=24 * mm, height=40 * mm, kind="proportional") if r["snapshot"] and p.exists() else Paragraph("-", ss["Normal"])
            cap = Paragraph(f"<font size=7>#{r['id']} {r['status']}<br/>{r['time'][11:]}</font>", ss["Normal"])
            line.append([img, cap])
            if len(line) == 6:
                cells.append(line); line = []
        if line:
            cells.append(line + [""] * (6 - len(line)))
        g = Table(cells, colWidths=[29 * mm] * 6)
        g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(g)

    story += [PageBreak(), Paragraph("Inspection log", ss["Heading2"])]
    data = [["#", "Time", "Track", "Status", "Cap", "Colour", "Col.conf", "Det.conf"]]
    for r in rows[-1500:]:
        data.append([r["id"], r["time"], r["track_id"], r["status"], "yes" if r["cap_present"] else "no",
                     r["cap_color"] or "-", f"{r['color_conf'] or 0:.2f}", f"{r['det_conf'] or 0:.2f}"])
    lt = Table(data, repeatRows=1, colWidths=[14 * mm, 38 * mm, 16 * mm, 30 * mm, 12 * mm, 20 * mm, 18 * mm, 18 * mm])
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb"))]
    for i, r in enumerate(rows[-1500:], 1):
        if r["status"] != "OK":
            style.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#dc2626" if r["status"] == "NO_CAP" else "#d97706")))
    lt.setStyle(TableStyle(style))
    story.append(lt)
    doc.build(story)
    return out.getvalue()
