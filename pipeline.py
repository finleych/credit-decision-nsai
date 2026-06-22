"""
pipeline.py

Full end-to-end Neuro-Symbolic AI credit decision demo.

Combines all three stages of the Neuro|Symbolic pipeline paradigm:

  Stage 1 — Neural Perception   : CreditMLP predicts a credit score from
                                   raw applicant features.
  Stage 2 — Symbolic Reasoning  : CreditRuleEngine applies hard business
                                   and regulatory rules to the predicted score.
  Stage 3 — Natural Language    : Template explainer converts rule output into
                                   a human-readable explanation (no API key).

Generates 10 fresh, randomly sampled applicants (not loaded from credit_data.csv),
runs each through the full pipeline, and prints a formatted summary table plus
aggregate statistics.

Prerequisites (run in order before this script):
  python generate_data.py
  python neural_model.py
"""

import numpy as np
import torch
import joblib
from pathlib import Path
from collections import Counter
import textwrap

from generate_data import FEATURE_COLUMNS, generate_single_applicant
from neural_model import CreditMLP
from rule_engine import CreditRuleEngine
from explainer import get_explanation

MODELS_DIR = Path(__file__).parent / "models"


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_model_and_scaler() -> tuple:
    """
    Load the trained CreditMLP and the scaler bundle from disk.

    Returns
    -------
    (model, scaler, median_msd)
      model      : CreditMLP in eval mode
      scaler     : fitted StandardScaler
      median_msd : training-set median for months_since_last_delinquency imputation
    """
    model_path  = MODELS_DIR / "credit_model.pt"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Model artefacts not found — run neural_model.py first.\n"
            f"  Expected: {model_path}\n"
            f"            {scaler_path}"
        )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = CreditMLP(n_features=len(FEATURE_COLUMNS))
    model.load_state_dict(state_dict)
    model.eval()

    bundle = joblib.load(scaler_path)
    scaler     = bundle["scaler"]
    median_msd = bundle["median_msd"]

    return model, scaler, median_msd


def predict_credit_score(
    model: CreditMLP,
    scaler,
    applicant: dict,
) -> float:
    """
    Scale applicant features and run the neural model to get a predicted score.

    Parameters
    ----------
    model     : trained CreditMLP in eval mode
    scaler    : fitted StandardScaler (from scaler bundle)
    applicant : feature dict after null imputation

    Returns
    -------
    Predicted credit score clipped to [300, 850].
    """
    x = np.array([[applicant[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    x_tensor = torch.FloatTensor(x_scaled)
    with torch.no_grad():
        raw = model(x_tensor).item()
    return float(np.clip(raw, 300, 850))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(n_applicants: int = 10, random_seed: int = 2024) -> list[dict]:
    """
    Generate n_applicants and run each through the full NSAI pipeline.

    Parameters
    ----------
    n_applicants : number of fresh applicants to process
    random_seed  : seed for reproducible applicant generation

    Returns
    -------
    List of result dicts with keys:
        applicant_id, features, predicted_score, decision, risk_level,
        triggered_rules, explanation
    """
    model, scaler, median_msd = load_model_and_scaler()
    engine = CreditRuleEngine()

    rng = np.random.default_rng(random_seed)
    results = []

    for i in range(n_applicants):
        # Stage 0 — Generate a fresh applicant (not from CSV)
        raw = generate_single_applicant(rng)

        # Stage 0b — Null imputation (same median used during training)
        app = dict(raw)
        if app.get("months_since_last_delinquency") is None:
            app["months_since_last_delinquency"] = median_msd

        # Stage 1 — Neural prediction
        score = predict_credit_score(model, scaler, app)

        # Stage 2 — Symbolic rule evaluation
        rule_result = engine.evaluate(app, score)

        # Stage 3 — Template explanation
        explanation = get_explanation(
            applicant=app,
            predicted_score=score,
            decision=rule_result["decision"],
            risk_level=rule_result["risk_level"],
            triggered_rules=rule_result["triggered_rules"],
        )

        results.append({
            "applicant_id":   i + 1,
            "features":       app,
            "predicted_score": score,
            "decision":       rule_result["decision"],
            "risk_level":     rule_result["risk_level"],
            "triggered_rules": rule_result["triggered_rules"],
            "explanation":    explanation,
        })

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results_table(results: list[dict]) -> None:
    """Print a formatted per-applicant summary table."""
    width = 110
    print("\n" + "=" * width)
    print("NEURO-SYMBOLIC AI CREDIT DECISION PIPELINE — RESULTS")
    print("=" * width)

    for r in results:
        score    = r["predicted_score"]
        decision = r["decision"]
        risk     = r["risk_level"]
        app_id   = r["applicant_id"]

        print(
            f"\n  Applicant {app_id:>2d}  |  Score: {score:6.1f}  |  "
            f"Decision: {decision:<6}  |  Risk: {risk:<6}"
        )
        # Wrap the explanation at 100 chars with indentation
        for line in textwrap.wrap(r["explanation"], width=100):
            print(f"    {line}")

        if r["triggered_rules"]:
            print("    Rules fired:")
            for rule in r["triggered_rules"]:
                print(f"      • {rule}")


def print_summary_stats(results: list[dict]) -> None:
    """Print aggregate statistics over all processed applicants."""
    n = len(results)
    approved = [r for r in results if r["decision"] == "APPROVE"]
    denied   = [r for r in results if r["decision"] == "DENY"]
    scores   = [r["predicted_score"] for r in results]
    all_rules = [rule for r in results for rule in r["triggered_rules"]]

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Total applicants    : {n}")
    print(f"  Approved            : {len(approved)} ({len(approved)/n*100:.0f}%)")
    print(f"  Denied              : {len(denied)}   ({len(denied)/n*100:.0f}%)")
    print(f"  Average pred. score : {np.mean(scores):.1f}")
    print(f"  Min / Max score     : {min(scores):.1f} / {max(scores):.1f}")

    if all_rules:
        most_common, count = Counter(all_rules).most_common(1)[0]
        print(f"  Most triggered rule : \"{most_common}\" (×{count})")
    else:
        print("  Most triggered rule : None — all applicants approved")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the end-to-end pipeline on 10 freshly generated applicants."""
    print("Loading model artefacts…")
    results = run_pipeline(n_applicants=10, random_seed=2024)
    print_results_table(results)
    print_summary_stats(results)


if __name__ == "__main__":
    main()
