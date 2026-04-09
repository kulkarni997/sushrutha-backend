import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

def build_dosha_series(past_results: list, dosha: str) -> pd.DataFrame:
    """
    Build a pandas DataFrame for Prophet from past scan results.
    past_results: list of {vata_pct, pitta_pct, kapha_pct, created_at}
    dosha: "vata_pct" | "pitta_pct" | "kapha_pct"
    """
    rows = []
    for r in past_results:
        try:
            ds = pd.to_datetime(r["created_at"]).tz_localize(None)
            y  = float(r[dosha])
            rows.append({"ds": ds, "y": y})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    return df

def run_prophet_forecast(df: pd.DataFrame, periods: int = 14) -> list:
    """
    Fit Prophet on df and return `periods` future predictions.
    Returns list of floats (one per future day).
    """
    from prophet import Prophet

    if len(df) < 2:
        return []

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.3,
        interval_width=0.80
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)

    # Return only the future rows (not history)
    future_forecast = forecast.tail(periods)["yhat"].tolist()
    return future_forecast

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def compute_healing_score(vata: float, pitta: float, kapha: float, day: int) -> float:
    """
    Healing score: balance = low dominant, smooth distribution.
    Day factor adds progressive improvement assumption.
    """
    dominant = max(vata, pitta, kapha)
    balance_score = 100 - dominant          # 0 (all in one) to ~66 (perfect balance)
    day_bonus = day * 0.5                   # small progressive improvement
    score = clamp(balance_score + day_bonus, 0, 100)
    return round(score, 1)

def generate_forecast(
    past_results: list,
    current_vata: int,
    current_pitta: int,
    current_kapha: int,
    periods: int = 14
) -> list:
    """
    Main entry point called from /forecast route.

    Args:
        past_results: list of previous results rows from Supabase
                      each must have: vata_pct, pitta_pct, kapha_pct, created_at
        current_vata/pitta/kapha: today's diagnosis scores
        periods: forecast days (default 14)

    Returns:
        list of dicts: [{day, vata, pitta, kapha, healing_score}, ...]
    """

    # Always inject today's result so Prophet has at least one real point
    today = datetime.utcnow().isoformat()
    all_results = past_results + [{
        "vata_pct":  current_vata,
        "pitta_pct": current_pitta,
        "kapha_pct": current_kapha,
        "created_at": today
    }]

    # Build series per dosha
    df_vata  = build_dosha_series(all_results, "vata_pct")
    df_pitta = build_dosha_series(all_results, "pitta_pct")
    df_kapha = build_dosha_series(all_results, "kapha_pct")

    use_prophet = all(len(df) >= 2 for df in [df_vata, df_pitta, df_kapha])

    if use_prophet:
        try:
            vata_forecast  = run_prophet_forecast(df_vata,  periods)
            pitta_forecast = run_prophet_forecast(df_pitta, periods)
            kapha_forecast = run_prophet_forecast(df_kapha, periods)
        except Exception as e:
            print(f"Prophet failed, using rule-based fallback: {e}")
            use_prophet = False

    if not use_prophet:
        # Rule-based fallback: gradual drift toward balance (33/33/34)
        vata_forecast  = []
        pitta_forecast = []
        kapha_forecast = []
        for i in range(1, periods + 1):
            t = i / periods  # 0→1
            vata_forecast.append( current_vata  + t * (33 - current_vata)  * 0.6)
            pitta_forecast.append(current_pitta + t * (33 - current_pitta) * 0.6)
            kapha_forecast.append(current_kapha + t * (34 - current_kapha) * 0.6)

    # Build output list
    output = []
    for i in range(periods):
        raw_v = vata_forecast[i]  if i < len(vata_forecast)  else current_vata
        raw_p = pitta_forecast[i] if i < len(pitta_forecast) else current_pitta
        raw_k = kapha_forecast[i] if i < len(kapha_forecast) else current_kapha

        # Clamp to valid range
        v = clamp(raw_v, 5, 90)
        p = clamp(raw_p, 5, 90)
        k = clamp(raw_k, 5, 90)

        # Normalize to 100
        total = v + p + k
        v = round((v / total) * 100)
        p = round((p / total) * 100)
        k = 100 - v - p

        day_num = i + 1
        output.append({
            "day":           day_num,
            "date":          (datetime.utcnow() + timedelta(days=day_num)).strftime("%Y-%m-%d"),
            "vata":          int(v),
            "pitta":         int(p),
            "kapha":         int(k),
            "healing_score": compute_healing_score(v, p, k, day_num)
        })

    return output