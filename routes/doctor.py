from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from db.supabase_client import supabase
from auth.jwt_handler import require_doctor
import uuid
from datetime import datetime, timedelta

router = APIRouter()

class WalkinCreate(BaseModel):
    patient_name: str

class ResultsOverride(BaseModel):
    doctor_notes: Optional[str] = None
    override_dosha: Optional[str] = None

@router.get("/patients")
async def get_patients(user: dict = Depends(require_doctor)):
    scans = supabase.table("scans")\
        .select("*, users!scans_user_id_fkey(full_name)")\
        .eq("shared_with", user["sub"])\
        .eq("shared", True)\
        .order("created_at", desc=True)\
        .execute()
    
    if not scans.data:
        return []
    
    # Fetch results separately for each scan
    for scan in scans.data:
        result = supabase.table("results")\
            .select("*")\
            .eq("scan_id", scan["id"])\
            .execute()
        scan["results"] = result.data or []
    
    return scans.data

@router.get("/patient/{scan_id}")
async def get_patient(scan_id: str, user: dict = Depends(require_doctor)):
    scan = supabase.table("scans")\
        .select("*, users(full_name)")\
        .eq("id", scan_id)\
        .eq("shared_with", user["sub"])\
        .single()\
        .execute()
    
    if not scan.data:
        raise HTTPException(404, "Scan not found")
    
    result = supabase.table("results")\
        .select("*")\
        .eq("scan_id", scan_id)\
        .execute()
    scan.data["results"] = result.data or []
    
    return scan.data



@router.post("/walkin")
async def create_walkin(payload: WalkinCreate, user: dict = Depends(require_doctor)):
    claim_token = str(uuid.uuid4())
    expires = datetime.utcnow() + timedelta(hours=48)
    result = supabase.table("guest_scans").insert({
        "doctor_id": user["sub"],
        "patient_name": payload.patient_name,
        "claim_token": claim_token,
        "token_expires_at": expires.isoformat()
    }).execute()
    return result.data[0]

@router.get("/walkin/{session_id}")
async def get_walkin(session_id: str, user: dict = Depends(require_doctor)):
    session = supabase.table("guest_scans")\
        .select("*")\
        .eq("id", session_id)\
        .eq("doctor_id", user["sub"])\
        .single()\
        .execute()
    
    if not session.data:
        raise HTTPException(404, "Session not found")
    
    result = supabase.table("results")\
        .select("*")\
        .eq("guest_scan_id", session_id)\
        .execute()
    session.data["results"] = result.data or []
    
    return session.data

@router.patch("/results/{result_id}")
async def override_results(result_id: str, payload: ResultsOverride, user: dict = Depends(require_doctor)):
    update = supabase.table("results")\
        .update({
            "doctor_notes": payload.doctor_notes,
            "override_dosha": payload.override_dosha
        })\
        .eq("id", result_id)\
        .execute()
    
    if not update.data:
        raise HTTPException(404, "Result not found")
    
    return update.data[0]

@router.post("/finalise/{result_id}")
async def finalise_report(result_id: str, user: dict = Depends(require_doctor)):
    result = supabase.table("results")\
        .update({"finalised": True})\
        .eq("id", result_id)\
        .execute()

    if not result.data:
        raise HTTPException(404, "Result not found")

    row = result.data[0]
    scan_id = row.get("scan_id")
    guest_scan_id = row.get("guest_scan_id")

    # Patient scan — notify the patient who owns the scan
    if scan_id:
        scan = supabase.table("scans")\
            .select("user_id")\
            .eq("id", scan_id)\
            .single()\
            .execute()
        if scan.data:
            supabase.table("notifications").insert({
                "user_id": scan.data["user_id"],
                "type": "report_finalised",
                "reference_id": scan_id,
                "seen": False
            }).execute()

    # Walk-in scan — notify the patient only if they've already claimed it
    elif guest_scan_id:
        gs = supabase.table("guest_scans")\
            .select("claimed_by")\
            .eq("id", guest_scan_id)\
            .single()\
            .execute()
        claimed_by = gs.data.get("claimed_by") if gs.data else None
        if claimed_by:
            supabase.table("notifications").insert({
                "user_id": claimed_by,
                "type": "report_finalised",
                "reference_id": guest_scan_id,
                "seen": False
            }).execute()
        # If not claimed yet, no notification — patient doesn't have an account

    return row  

@router.get("/analytics")
async def get_analytics(user: dict = Depends(require_doctor)):
    doctor_id = user["sub"]

    # ── 1. Shared patient scans ───────────────────────────────────────────────
    shared_scans = supabase.table("scans")\
        .select("id, created_at, results(vata_pct, pitta_pct, kapha_pct, severity, finalised)")\
        .eq("shared_with", doctor_id)\
        .eq("shared", True)\
        .execute()

    # ── 2. Walk-in scans ──────────────────────────────────────────────────────
    walkin_scans = supabase.table("guest_scans")\
        .select("id, created_at, results(vata_pct, pitta_pct, kapha_pct, severity, finalised)")\
        .eq("doctor_id", doctor_id)\
        .execute()

    all_scans = (shared_scans.data or []) + (walkin_scans.data or [])
    total_scans = len(all_scans)

    # ── 3. Aggregate results ──────────────────────────────────────────────────
    dosha_counts   = {"Vata": 0, "Pitta": 0, "Kapha": 0}
    severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
    finalised_count = 0
    monthly_counts  = {}

    for scan in all_scans:
        results = scan.get("results") or []
        if isinstance(results, dict):
            results = [results]

        for r in results:
            if not r:
                continue

            # Dominant dosha
            vata  = r.get("vata_pct",  0) or 0
            pitta = r.get("pitta_pct", 0) or 0
            kapha = r.get("kapha_pct", 0) or 0
            dominant = max(
                ("Vata",  vata),
                ("Pitta", pitta),
                ("Kapha", kapha),
                key=lambda x: x[1]
            )[0]
            dosha_counts[dominant] += 1

            # Severity
            sev = r.get("severity", "mild")
            if sev in severity_counts:
                severity_counts[sev] += 1

            # Finalised
            if r.get("finalised"):
                finalised_count += 1

        # Monthly trend — based on scan created_at
        try:
            dt    = datetime.fromisoformat(scan["created_at"].replace("Z", "+00:00"))
            month = dt.strftime("%b %Y")   # e.g. "Apr 2026"
            monthly_counts[month] = monthly_counts.get(month, 0) + 1
        except Exception:
            pass

    # ── 4. Sort monthly by date ───────────────────────────────────────────────
    def month_sort_key(m):
        try:
            return datetime.strptime(m, "%b %Y")
        except Exception:
            return datetime.min

    monthly_trend = [
        {"month": m, "scans": monthly_counts[m]}
        for m in sorted(monthly_counts, key=month_sort_key)
    ]

    # ── 5. Pending (shared but not finalised) ─────────────────────────────────
    pending_count = total_scans - finalised_count

    return {
        "total_scans":      total_scans,
        "finalised":        finalised_count,
        "pending":          pending_count,
        "dosha_distribution": dosha_counts,
        "severity_breakdown": severity_counts,
        "monthly_trend":    monthly_trend,
    }


@router.get("/report/{result_id}/pdf")
async def export_report_pdf(result_id: str, user: dict = Depends(require_doctor)):
    """Generate and return a PDF report for a finalised result."""

    # Fetch result
    result_res = supabase.table("results")\
        .select("*")\
        .eq("id", result_id)\
        .single()\
        .execute()

    if not result_res.data:
        raise HTTPException(404, "Result not found")

    r = result_res.data

    # Fetch scan
    scan_res = supabase.table("scans")\
        .select("*, users(full_name)")\
        .eq("id", r["scan_id"])\
        .single()\
        .execute()

    scan        = scan_res.data or {}
    patient_name = (scan.get("users") or {}).get("full_name", "Patient")
    symptoms    = scan.get("symptoms_text", "—")
    created_at  = r.get("created_at", "")

    try:
        dt_str = datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        ).strftime("%d %B %Y")
    except Exception:
        dt_str = created_at[:10] if created_at else "—"

    pdf_bytes = _build_report_pdf(
        patient_name=patient_name,
        symptoms=symptoms,
        date_str=dt_str,
        result=r,
        doctor_name=user.get("name", "Doctor"),
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="sushrutha_report_{result_id[:8]}.pdf"'
        }
    )


def _build_report_pdf(
    patient_name: str,
    symptoms: str,
    date_str: str,
    result: dict,
    doctor_name: str,
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import io

    TURMERIC   = colors.HexColor("#E8A020")
    SANDALWOOD = colors.HexColor("#C4845A")
    NEEM       = colors.HexColor("#4A7C59")
    KUMKUM     = colors.HexColor("#C0392B")
    MUTED      = colors.HexColor("#A89880")
    BORDER     = colors.HexColor("#2E2820")
    LIGHT_BG   = colors.HexColor("#FAF7F2")
    ALT_BG     = colors.HexColor("#F5EFE6")

    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    TITLE  = s("T",  fontName="Helvetica-Bold",   fontSize=20, textColor=TURMERIC,   alignment=TA_CENTER, spaceAfter=4)
    SUB    = s("S",  fontName="Helvetica",         fontSize=10, textColor=SANDALWOOD, alignment=TA_CENTER, spaceAfter=2)
    H1     = s("H1", fontName="Helvetica-Bold",    fontSize=13, textColor=TURMERIC,   spaceBefore=14, spaceAfter=5)
    BODY   = s("B",  fontName="Helvetica",         fontSize=10, textColor=colors.HexColor("#2B2B2B"), leading=15, spaceAfter=5, alignment=TA_JUSTIFY)
    FOOTER = s("F",  fontName="Helvetica-Oblique", fontSize=8,  textColor=MUTED,      alignment=TA_CENTER)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)
    story = []

    # Header
    story.append(Paragraph("🌿 Sushrutha AI", TITLE))
    story.append(Paragraph("Ayurvedic Diagnostic Report", SUB))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TURMERIC, spaceAfter=10))

    # Patient info table
    vata  = result.get("vata_pct",  0)
    pitta = result.get("pitta_pct", 0)
    kapha = result.get("kapha_pct", 0)
    dominant = max(
        ("Vata", vata), ("Pitta", pitta), ("Kapha", kapha),
        key=lambda x: x[1]
    )[0]
    severity = result.get("severity", "mild").capitalize()
    sev_color = {"Mild": NEEM, "Moderate": SANDALWOOD, "Severe": KUMKUM}.get(severity, MUTED)

    info = [
        ["Patient",  patient_name,  "Date",     date_str],
        ["Doctor",   doctor_name,   "Severity", severity],
        ["Symptoms", symptoms,      "Dominant", dominant],
    ]
    t = Table(info, colWidths=[3*cm, 6.5*cm, 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ('FONTNAME',   (0,0), (0,-1),  'Helvetica-Bold'),
        ('FONTNAME',   (2,0), (2,-1),  'Helvetica-Bold'),
        ('FONTNAME',   (1,0), (1,-1),  'Helvetica'),
        ('FONTNAME',   (3,0), (3,-1),  'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('TEXTCOLOR',  (0,0), (0,-1),  TURMERIC),
        ('TEXTCOLOR',  (2,0), (2,-1),  TURMERIC),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [LIGHT_BG, ALT_BG, LIGHT_BG]),
        ('GRID',       (0,0), (-1,-1), 0.4, BORDER),
        ('PADDING',    (0,0), (-1,-1), 7),
        ('TEXTCOLOR',  (3,1), (3,1),   sev_color),
        ('FONTNAME',   (3,1), (3,1),   'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Dosha scores
    story.append(Paragraph("Dosha Analysis", H1))
    dosha_data = [
        ["Dosha", "Score", "Interpretation"],
        ["Vata",  f"{vata}%",  "Movement, Air, Space — nervousness, creativity, dryness"],
        ["Pitta", f"{pitta}%", "Fire, Water — ambition, digestion, inflammation"],
        ["Kapha", f"{kapha}%", "Earth, Water — stability, immunity, lethargy"],
    ]
    dt = Table(dosha_data, colWidths=[3*cm, 2.5*cm, 10*cm])
    dt.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  TURMERIC),
        ('TEXTCOLOR',      (0,0), (-1,0),  colors.white),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 9),
        ('FONTNAME',       (0,1), (-1,-1), 'Helvetica'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT_BG, ALT_BG, LIGHT_BG]),
        ('GRID',           (0,0), (-1,-1), 0.4, BORDER),
        ('PADDING',        (0,0), (-1,-1), 7),
    ]))
    story.append(dt)
    story.append(Spacer(1, 0.4*cm))

    # Recipe
    recipe = result.get("recipe_text", "")
    if recipe:
        story.append(Paragraph("Herbal Recipe", H1))
        story.append(Paragraph(recipe, BODY))

    # Doctor notes
    notes = result.get("doctor_notes", "")
    if notes:
        story.append(Paragraph("Doctor Notes", H1))
        story.append(Paragraph(notes, BODY))

    # Override dosha
    override = result.get("override_dosha", "")
    if override:
        story.append(Paragraph(f"Doctor Override: Dominant Dosha corrected to <b>{override}</b>", BODY))

    # Pulse info
    if result.get("pulse_used"):
        story.append(Paragraph("Pulse data from ESP32 sensor was included in this diagnosis.", BODY))

    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated by Sushrutha AI  |  {date_str}  |  This report is for informational purposes only.",
        FOOTER
    ))

    doc.build(story)
    return buf.getvalue()