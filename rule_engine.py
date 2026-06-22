"""
rule_engine.py

Stage 2 (Symbolic Reasoning) of the Neuro-Symbolic AI credit decision pipeline.

Accepts the neural model's predicted credit score together with the applicant's
raw features and applies a set of hard, interpretable business and regulatory
rules to reach an APPROVE or DENY decision.

Why symbolic here?
  - Hard limits (DTI ≤ 43%, no active bankruptcy) encode regulatory constraints
    that a neural net alone cannot guarantee — a neural net may occasionally
    predict "approve" for a bankrupt applicant if the training data was noisy.
  - Every triggered rule is a human-readable string, giving auditors and
    applicants a clear, traceable reason for each decision.
  - The rule set is easy to update when regulations change without retraining.

Neuro-Symbolic role: symbolic reasoner in the Neuro|Symbolic pipeline paradigm.

v2 changes (2026-06-23):
  - Rules 7–12 added: collections, payment history score, public records,
    thin credit history, excessive total debt, no payment streak.
  - Risk tiers expanded: PREMIUM LOW (exceptional), LOW, MEDIUM, MEDIUM-WATCH
    (approved but has minor derogatory marks), HIGH (denied).
"""


class CreditRuleEngine:
    """
    Symbolic rule engine for credit decisioning.

    Applies twelve hard-deny rules and computes a risk tier for approved
    applicants:
      PREMIUM LOW   — exceptional payment history (score ≥ 750, PHS ≥ 90)
      LOW           — strong profile (score ≥ 740)
      MEDIUM        — adequate profile (score < 740, no derogatory marks)
      MEDIUM-WATCH  — approved but has collections or public records on file
      HIGH          — denied (any rule triggered)
    """

    def evaluate(self, applicant_features: dict, predicted_score: float) -> dict:
        """
        Apply symbolic rules to reach a credit decision.

        Parameters
        ----------
        applicant_features : dict
            Raw feature values for the applicant (after null imputation).
            New v2 features are read with .get() so the engine stays backward-
            compatible with v1 feature dicts (missing keys default to safe values).
        predicted_score : float
            Credit score predicted by the neural model (300–850).

        Returns
        -------
        dict with keys:
            "decision"       : "APPROVE" or "DENY"
            "risk_level"     : "PREMIUM LOW", "LOW", "MEDIUM", "MEDIUM-WATCH",
                               or "HIGH"
            "triggered_rules": list of human-readable rule strings that fired
        """
        triggered_rules: list[str] = []
        decision = "APPROVE"

        score    = predicted_score
        dti      = applicant_features["debt_to_income_ratio"]
        bkrpt    = applicant_features["has_bankruptcy"]
        msd      = applicant_features.get("months_since_last_delinquency", 999)
        late     = applicant_features["num_late_payments"]
        util     = applicant_features["credit_utilization_ratio"]
        inq      = applicant_features["num_hard_inquiries"]

        # v2 features — default to safe values so engine works with v1 dicts
        collections = applicant_features.get("num_collections", 0)
        pay_hist    = applicant_features.get("payment_history_score", 100)
        pub_rec     = applicant_features.get("public_records", 0)
        moa         = applicant_features.get("months_oldest_account", 999)
        total_debt  = applicant_features.get("total_debt_outstanding", 0)
        streak      = applicant_features.get("on_time_payment_streak", 99)

        # ----------------------------------------------------------------
        # Original 6 rules
        # ----------------------------------------------------------------

        # Rule 1 — minimum score threshold
        if score < 580:
            triggered_rules.append(
                f"Predicted credit score {score:.0f} is below the minimum threshold of 580"
            )
            decision = "DENY"

        # Rule 2 — regulatory DTI cap (CFPB qualified mortgage rule)
        if dti > 0.43:
            triggered_rules.append(
                f"Debt-to-income ratio {dti:.0%} exceeds the regulatory limit of 43%"
            )
            decision = "DENY"

        # Rule 3 — recent bankruptcy
        if bkrpt == 1 and msd < 24:
            triggered_rules.append(
                "Active bankruptcy record within the past 24 months"
            )
            decision = "DENY"

        # Rule 4 — late payments combined with low score
        if late >= 3 and score < 670:
            triggered_rules.append(
                "3 or more late payments combined with a score below 670"
            )
            decision = "DENY"

        # Rule 5 — dangerously high credit utilisation
        if util > 0.85:
            triggered_rules.append(
                f"Credit utilization {util:.0%} exceeds the safe threshold of 85%"
            )
            decision = "DENY"

        # Rule 6 — excessive recent hard inquiries
        if inq >= 5 and score < 700:
            triggered_rules.append(
                "5 or more hard inquiries combined with a score below 700"
            )
            decision = "DENY"

        # ----------------------------------------------------------------
        # New v2 rules
        # ----------------------------------------------------------------

        # Rule 7 — accounts in collections
        if collections >= 2:
            triggered_rules.append(
                "2 or more accounts currently in collections"
            )
            decision = "DENY"

        # Rule 8 — severely delinquent payment history
        if pay_hist < 40:
            triggered_rules.append(
                f"Payment history score {pay_hist:.0f}/100 indicates severely "
                "delinquent payment behavior"
            )
            decision = "DENY"

        # Rule 9 — multiple public records
        if pub_rec >= 2:
            triggered_rules.append(
                "2 or more public records (bankruptcies, liens, or judgments) on file"
            )
            decision = "DENY"

        # Rule 10 — thin credit history + low score
        if moa < 12 and score < 650:
            triggered_rules.append(
                "Credit history under 12 months combined with score below 650"
            )
            decision = "DENY"

        # Rule 11 — excessive total debt load
        if total_debt > 100_000 and dti > 0.35:
            triggered_rules.append(
                f"Total outstanding debt ${total_debt:,.0f} combined with "
                f"debt-to-income ratio {dti:.0%} indicates unsustainable debt load"
            )
            decision = "DENY"

        # Rule 12 — no payment streak + high utilisation
        if streak < 6 and util > 0.70:
            triggered_rules.append(
                f"Less than 6 consecutive on-time payments combined with "
                f"credit utilization of {util:.0%}"
            )
            decision = "DENY"

        # ----------------------------------------------------------------
        # Risk tier (APPROVE only; all DENY → HIGH)
        # ----------------------------------------------------------------
        if decision == "APPROVE":
            if pay_hist >= 90 and score >= 750:
                risk_level = "PREMIUM LOW"
            elif score >= 740:
                risk_level = "LOW"
            elif collections > 0 or pub_rec > 0:
                # Approved but has minor derogatory marks — flag for review
                risk_level = "MEDIUM-WATCH"
            else:
                risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "decision": decision,
            "risk_level": risk_level,
            "triggered_rules": triggered_rules,
        }


# ---------------------------------------------------------------------------
# Demo — 8 sample applicants covering all rules and risk tiers
# ---------------------------------------------------------------------------

SAMPLE_APPLICANTS = [
    # 1 — strong profile → APPROVE LOW
    {
        "age": 45.0, "annual_income": 120_000.0, "employment_length_years": 15.0,
        "debt_to_income_ratio": 0.18, "num_credit_accounts": 8.0,
        "num_late_payments": 0.0, "num_hard_inquiries": 1.0,
        "credit_utilization_ratio": 0.12, "loan_amount_requested": 15_000.0,
        "has_bankruptcy": 0.0, "months_since_last_delinquency": 84.0,
        "payment_history_score": 82.0, "total_debt_outstanding": 28_000.0,
        "num_collections": 0.0, "months_oldest_account": 144.0,
        "credit_mix_score": 6.0, "on_time_payment_streak": 48.0,
        "public_records": 0.0,
    },
    # 2 — decent profile → APPROVE MEDIUM
    {
        "age": 32.0, "annual_income": 55_000.0, "employment_length_years": 5.0,
        "debt_to_income_ratio": 0.32, "num_credit_accounts": 4.0,
        "num_late_payments": 1.0, "num_hard_inquiries": 2.0,
        "credit_utilization_ratio": 0.48, "loan_amount_requested": 8_000.0,
        "has_bankruptcy": 0.0, "months_since_last_delinquency": 36.0,
        "payment_history_score": 67.0, "total_debt_outstanding": 18_000.0,
        "num_collections": 0.0, "months_oldest_account": 60.0,
        "credit_mix_score": 3.0, "on_time_payment_streak": 18.0,
        "public_records": 0.0,
    },
    # 3 — low score + late payments → DENY (rules 1 & 4)
    {
        "age": 22.0, "annual_income": 28_000.0, "employment_length_years": 1.0,
        "debt_to_income_ratio": 0.35, "num_credit_accounts": 2.0,
        "num_late_payments": 4.0, "num_hard_inquiries": 3.0,
        "credit_utilization_ratio": 0.72, "loan_amount_requested": 5_000.0,
        "has_bankruptcy": 0.0, "months_since_last_delinquency": 6.0,
        "payment_history_score": 45.0, "total_debt_outstanding": 9_000.0,
        "num_collections": 0.0, "months_oldest_account": 18.0,
        "credit_mix_score": 2.0, "on_time_payment_streak": 3.0,
        "public_records": 0.0,
    },
    # 4 — high DTI → DENY (rule 2)
    {
        "age": 38.0, "annual_income": 42_000.0, "employment_length_years": 8.0,
        "debt_to_income_ratio": 0.55, "num_credit_accounts": 6.0,
        "num_late_payments": 0.0, "num_hard_inquiries": 1.0,
        "credit_utilization_ratio": 0.30, "loan_amount_requested": 12_000.0,
        "has_bankruptcy": 0.0, "months_since_last_delinquency": 60.0,
        "payment_history_score": 74.0, "total_debt_outstanding": 31_000.0,
        "num_collections": 0.0, "months_oldest_account": 96.0,
        "credit_mix_score": 4.0, "on_time_payment_streak": 24.0,
        "public_records": 0.0,
    },
    # 5 — recent bankruptcy + high utilisation → DENY (rules 3 & 5)
    {
        "age": 29.0, "annual_income": 35_000.0, "employment_length_years": 3.0,
        "debt_to_income_ratio": 0.28, "num_credit_accounts": 3.0,
        "num_late_payments": 2.0, "num_hard_inquiries": 4.0,
        "credit_utilization_ratio": 0.91, "loan_amount_requested": 7_500.0,
        "has_bankruptcy": 1.0, "months_since_last_delinquency": 10.0,
        "payment_history_score": 51.0, "total_debt_outstanding": 14_000.0,
        "num_collections": 0.0, "months_oldest_account": 36.0,
        "credit_mix_score": 2.0, "on_time_payment_streak": 2.0,
        "public_records": 0.0,
    },
    # 6 — 3 collections accounts → DENY (rule 7)
    {
        "age": 34.0, "annual_income": 48_000.0, "employment_length_years": 4.0,
        "debt_to_income_ratio": 0.30, "num_credit_accounts": 5.0,
        "num_late_payments": 2.0, "num_hard_inquiries": 2.0,
        "credit_utilization_ratio": 0.55, "loan_amount_requested": 6_000.0,
        "has_bankruptcy": 0.0, "months_since_last_delinquency": 18.0,
        "payment_history_score": 52.0, "total_debt_outstanding": 22_000.0,
        "num_collections": 3.0, "months_oldest_account": 48.0,
        "credit_mix_score": 3.0, "on_time_payment_streak": 4.0,
        "public_records": 0.0,
    },
    # 7 — excessive debt + high DTI → DENY (rule 11)
    {
        "age": 52.0, "annual_income": 85_000.0, "employment_length_years": 18.0,
        "debt_to_income_ratio": 0.48, "num_credit_accounts": 10.0,
        "num_late_payments": 0.0, "num_hard_inquiries": 1.0,
        "credit_utilization_ratio": 0.40, "loan_amount_requested": 25_000.0,
        "has_bankruptcy": 0.0, "months_since_last_delinquency": 96.0,
        "payment_history_score": 78.0, "total_debt_outstanding": 125_000.0,
        "num_collections": 0.0, "months_oldest_account": 180.0,
        "credit_mix_score": 6.0, "on_time_payment_streak": 42.0,
        "public_records": 0.0,
    },
    # 8 — exceptional profile → APPROVE PREMIUM LOW
    {
        "age": 58.0, "annual_income": 180_000.0, "employment_length_years": 25.0,
        "debt_to_income_ratio": 0.12, "num_credit_accounts": 15.0,
        "num_late_payments": 0.0, "num_hard_inquiries": 1.0,
        "credit_utilization_ratio": 0.08, "loan_amount_requested": 20_000.0,
        "has_bankruptcy": 0.0, "months_since_last_delinquency": 120.0,
        "payment_history_score": 97.0, "total_debt_outstanding": 18_000.0,
        "num_collections": 0.0, "months_oldest_account": 300.0,
        "credit_mix_score": 9.0, "on_time_payment_streak": 96.0,
        "public_records": 0.0,
    },
]

# Paired predicted scores (simulating neural model output)
SAMPLE_SCORES = [762.0, 698.0, 542.0, 695.0, 605.0, 620.0, 720.0, 790.0]


def main():
    """Run the rule engine on eight sample applicants and print a summary table."""
    engine = CreditRuleEngine()

    header = f"{'#':<4} {'Score':>7}  {'Decision':<8}  {'Risk':<14}  Triggered Rules"
    print("=" * 110)
    print("CREDIT RULE ENGINE — SAMPLE DECISIONS")
    print("=" * 110)
    print(header)
    print("-" * 110)

    for i, (app, score) in enumerate(zip(SAMPLE_APPLICANTS, SAMPLE_SCORES), start=1):
        result = engine.evaluate(app, score)
        rules_str = "; ".join(result["triggered_rules"]) if result["triggered_rules"] else "None"
        print(
            f"{i:<4} {score:>7.1f}  {result['decision']:<8}  "
            f"{result['risk_level']:<14}  {rules_str}"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()
