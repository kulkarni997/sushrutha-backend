from fastapi import APIRouter, Depends, HTTPException
from db.supabase_client import supabase
from auth.jwt_handler import require_patient

router = APIRouter()

@router.get("/forecast/{scan_id}")
async def get_forecast(scan_id: str, user: dict = Depends(require_patient)):
    user_id = user["sub"]

    # 1. Get current scan's result
    try:
        current_res = supabase.table("results") \
            .select("vata_pct, pitta_pct, kapha_pct, severity") \
            .eq("scan_id", scan_id) \
            .single() \
            .execute()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Result not found for scan: {e}")

    if not current_res.data:
        raise HTTPException(status_code=404, detail="No result found for this scan")

    current = current_res.data
    vata   = current["vata_pct"]
    pitta  = current["pitta_pct"]
    kapha  = current["kapha_pct"]

    # Block forecast for severe cases
    if current.get("severity") == "severe":
        return {
            "forecast":  [],
            "days":      0,
            "blocked":   True,
            "reason":    "Severity is too high for forecast. Please consult a BAMS doctor.",
            "severity":  "severe"
        }

    # 2. Fetch past results for this user (excluding current scan)
    try:
        past_scans = supabase.table("scans") \
            .select("id") \
            .eq("user_id", user_id) \
            .neq("id", scan_id) \
            .execute()

        past_scan_ids = [s["id"] for s in (past_scans.data or [])]

        past_results = []
        if past_scan_ids:
            past_res = supabase.table("results") \
                .select("vata_pct, pitta_pct, kapha_pct, created_at") \
                .in_("scan_id", past_scan_ids) \
                .order("created_at", desc=False) \
                .limit(30) \
                .execute()
            past_results = past_res.data or []

    except Exception as e:
        print(f"Past results fetch error: {e}")
        past_results = []

    # 3. Generate forecast
    from ml.prophet_model import generate_forecast
    forecast = generate_forecast(
        past_results=past_results,
        current_vata=vata,
        current_pitta=pitta,
        current_kapha=kapha,
        periods=14
    )

    # 4. Save forecast_json to results table
    try:
        supabase.table("results") \
            .update({"forecast_json": forecast}) \
            .eq("scan_id", scan_id) \
            .execute()
    except Exception as e:
        print(f"Forecast save error: {e}")

    return {
        "forecast":  forecast,
        "days":      14,
        "blocked":   False,
        "severity":  current.get("severity"),
        "current":   {"vata": vata, "pitta": pitta, "kapha": kapha}
    }