"""
Internal evaluation harness for Sushrutha AI ML pipeline.
Runs 30 synthetic patient profiles end-to-end and reports consistency metrics.

Usage:  python training/eval_pipeline.py
Output: stdout + eval_report.txt in the same directory
"""
import os
import sys
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.svm_ensemble import run_svm_ensemble
from ml.prophet_model import generate_forecast

# Use a DIFFERENT seed than train_svm.py so profiles are unseen
RNG = np.random.default_rng(1337)

ARCHETYPES = ["Vata", "Pitta", "Kapha"]
N_PER_ARCHETYPE = 10
TOTAL = N_PER_ARCHETYPE * len(ARCHETYPES)


def sample_profile(archetype: str) -> dict:
    """Generate a synthetic patient profile matching an Ayurvedic archetype."""
    if archetype == "Vata":
        vision_dosha = str(RNG.choice(["Vata", "Pitta", "Kapha"], p=[0.55, 0.25, 0.20]))
        vein_score = float(np.clip(RNG.normal(0.70, 0.15), 0.0, 1.0))
        voice_dosha = str(RNG.choice(["Vata", "Pitta", "Kapha"], p=[0.55, 0.25, 0.20]))
        voice_conf = float(np.clip(RNG.normal(0.70, 0.15), 0.3, 0.99))
        bpm = float(np.clip(RNG.normal(90, 10), 55, 120))
        spo2 = float(np.clip(RNG.normal(95, 1.5), 90, 100))
    elif archetype == "Pitta":
        vision_dosha = str(RNG.choice(["Pitta", "Vata", "Kapha"], p=[0.55, 0.25, 0.20]))
        vein_score = float(np.clip(RNG.normal(0.50, 0.15), 0.0, 1.0))
        voice_dosha = str(RNG.choice(["Pitta", "Vata", "Kapha"], p=[0.55, 0.25, 0.20]))
        voice_conf = float(np.clip(RNG.normal(0.70, 0.15), 0.3, 0.99))
        bpm = float(np.clip(RNG.normal(78, 8), 55, 110))
        spo2 = float(np.clip(RNG.normal(96, 1.3), 90, 100))
    else:  # Kapha
        vision_dosha = str(RNG.choice(["Kapha", "Pitta", "Vata"], p=[0.55, 0.25, 0.20]))
        vein_score = float(np.clip(RNG.normal(0.30, 0.15), 0.0, 1.0))
        voice_dosha = str(RNG.choice(["Kapha", "Pitta", "Vata"], p=[0.55, 0.25, 0.20]))
        voice_conf = float(np.clip(RNG.normal(0.70, 0.15), 0.3, 0.99))
        bpm = float(np.clip(RNG.normal(65, 8), 50, 95))
        spo2 = float(np.clip(RNG.normal(97, 1.2), 92, 100))

    return {
        "archetype": archetype,
        "vision_result": {"dosha_signal": vision_dosha, "vein_score": vein_score},
        "voice_result": {"voice_dosha": voice_dosha, "confidence": voice_conf},
        "bpm": bpm,
        "spo2": spo2,
    }


def run_profile(profile: dict) -> dict:
    """Run one profile through SVM ensemble + forecast."""
    scores = run_svm_ensemble(
        vision_result=profile["vision_result"],
        voice_result=profile["voice_result"],
        pulse_used=True,
        bpm=profile["bpm"],
        spo2=profile["spo2"],
    )

    dominant_idx = max(
        [("Vata", scores["vata_pct"]), ("Pitta", scores["pitta_pct"]), ("Kapha", scores["kapha_pct"])],
        key=lambda t: t[1],
    )[0]

    forecast = generate_forecast(
        past_results=[],
        current_vata=scores["vata_pct"],
        current_pitta=scores["pitta_pct"],
        current_kapha=scores["kapha_pct"],
        periods=14,
    )

    healing_trajectory = [day["healing_score"] for day in forecast]
    pct_sum = scores["vata_pct"] + scores["pitta_pct"] + scores["kapha_pct"]
    is_monotonic_nondecreasing = all(
        healing_trajectory[i] <= healing_trajectory[i + 1] + 0.5  # tiny tolerance
        for i in range(len(healing_trajectory) - 1)
    )

    return {
        "archetype": profile["archetype"],
        "predicted": dominant_idx,
        "scores": scores,
        "pct_sum": pct_sum,
        "healing_start": healing_trajectory[0],
        "healing_end": healing_trajectory[-1],
        "forecast_monotonic": is_monotonic_nondecreasing,
        "method": scores.get("_method", "unknown"),
    }


def main():
    print("=" * 70)
    print("SUSHRUTHA AI — PIPELINE EVALUATION HARNESS")
    print("=" * 70)
    print(f"Running {TOTAL} synthetic profiles ({N_PER_ARCHETYPE} per archetype)...\n")

    results = []
    for archetype in ARCHETYPES:
        for _ in range(N_PER_ARCHETYPE):
            profile = sample_profile(archetype)
            results.append(run_profile(profile))

    # Metric 1: Dominant-dosha consistency
    correct = sum(1 for r in results if r["archetype"] == r["predicted"])
    consistency = correct / TOTAL * 100

    # Metric 2: Per-class breakdown
    per_class = {a: {"total": 0, "correct": 0} for a in ARCHETYPES}
    for r in results:
        per_class[r["archetype"]]["total"] += 1
        if r["archetype"] == r["predicted"]:
            per_class[r["archetype"]]["correct"] += 1

    # Metric 3: Percentage sanity (must sum to 100)
    sanity_failures = sum(1 for r in results if r["pct_sum"] != 100)

    # Metric 4: Forecast monotonicity
    monotonic_count = sum(1 for r in results if r["forecast_monotonic"])

    # Method breakdown
    method_counts = {}
    for r in results:
        method_counts[r["method"]] = method_counts.get(r["method"], 0) + 1

    # Confusion matrix
    confusion = {a: {b: 0 for b in ARCHETYPES} for a in ARCHETYPES}
    for r in results:
        confusion[r["archetype"]][r["predicted"]] += 1

    # ─── Report ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("HEADLINE METRICS")
    lines.append("=" * 70)
    lines.append(f"Overall dominant-dosha consistency:  {correct}/{TOTAL}  ({consistency:.1f}%)")
    lines.append(f"Percentage-sum sanity (should be 0): {sanity_failures} failures")
    lines.append(f"Forecast monotonicity (healing up):  {monotonic_count}/{TOTAL}  ({monotonic_count/TOTAL*100:.1f}%)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("PER-ARCHETYPE BREAKDOWN")
    lines.append("=" * 70)
    for a in ARCHETYPES:
        c = per_class[a]["correct"]
        t = per_class[a]["total"]
        lines.append(f"  {a:<6}  {c}/{t}  ({c/t*100:.0f}%)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("CONFUSION MATRIX  (rows = archetype, cols = predicted)")
    lines.append("=" * 70)
    header = "          " + "  ".join(f"{a:>6}" for a in ARCHETYPES)
    lines.append(header)
    for a in ARCHETYPES:
        row = f"  {a:<6}  " + "  ".join(f"{confusion[a][b]:>6}" for b in ARCHETYPES)
        lines.append(row)
    lines.append("")
    lines.append("=" * 70)
    lines.append("INFERENCE METHOD USAGE")
    lines.append("=" * 70)
    for method, count in method_counts.items():
        lines.append(f"  {method:<20}  {count}/{TOTAL}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("SAMPLE PROFILES (first 3)")
    lines.append("=" * 70)
    for i, r in enumerate(results[:3]):
        match = "✓" if r["archetype"] == r["predicted"] else "✗"
        lines.append(
            f"  [{i+1}] archetype={r['archetype']:<6} "
            f"predicted={r['predicted']:<6} {match}  "
            f"scores={r['scores']['vata_pct']}/{r['scores']['pitta_pct']}/{r['scores']['kapha_pct']}  "
            f"healing {r['healing_start']:.1f}→{r['healing_end']:.1f}"
        )
    lines.append("")
    lines.append("=" * 70)
    lines.append("NOTES FOR PITCH")
    lines.append("=" * 70)
    lines.append("- This is a PIPELINE CONSISTENCY test, not a clinical accuracy claim.")
    lines.append("- Synthetic profiles from the same distribution family as training,")
    lines.append("  generated with a different random seed (held out from training).")
    lines.append("- SVM + rule-based forecast tested end-to-end per profile.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    # Save to file
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "eval_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[eval] Report saved to: {out_path}")


if __name__ == "__main__":
    main()