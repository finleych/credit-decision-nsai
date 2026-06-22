"""
explainer.py

Stage 3 (Template Explanation Layer) of the Neuro-Symbolic AI credit decision
pipeline.

Converts the symbolic rule engine's structured output into a natural-language
explanation that reads as if written by a human loan officer.

No API key, no LLM, no internet connection required. Explanations are generated
deterministically from the triggered rule strings, making them fully auditable
and reproducible — a critical property for regulated lending.

Neuro-Symbolic role: the "glass box" communication layer. It translates the
symbolic reasoning from Stage 2 into language applicants and compliance teams
can understand without exposing internal score thresholds or rule indices.

v2 changes (2026-06-23):
  - 6 new rule templates for rules 7–12 (collections, payment history score,
    public records, thin credit history, excessive debt, no payment streak).
  - PREMIUM LOW and MEDIUM-WATCH approval branches in get_explanation().
"""


# ---------------------------------------------------------------------------
# Rule-string → plain-English sentence mapping
# ---------------------------------------------------------------------------

# Each entry is (substring_to_match, template_fn).
# The substring is tested case-insensitively against each triggered rule string.
# The template_fn receives the applicant dict and predicted score, returning one
# plain-English sentence about that specific risk factor.
_RULE_TEMPLATES = [
    # Original 6 templates
    (
        "below the minimum threshold",
        lambda app, score: (
            f"Your predicted credit score of {score:.0f} does not meet our "
            "minimum requirement of 580."
        ),
    ),
    (
        "regulatory limit of 43%",
        lambda app, score: (
            f"Your debt-to-income ratio of {app['debt_to_income_ratio']:.0%} "
            "exceeds the maximum allowable limit of 43%, which is a standard "
            "regulatory cap for qualified mortgages and personal loans."
        ),
    ),
    (
        "bankruptcy",
        lambda app, score: (
            "Your credit file contains an active bankruptcy record that was "
            "filed within the past 24 months, which falls outside our "
            "eligibility window."
        ),
    ),
    (
        "late payments",
        lambda app, score: (
            f"Your record of {int(app['num_late_payments'])} late payment(s), "
            "combined with your current score, represents a level of payment "
            "risk we are unable to accept at this time."
        ),
    ),
    (
        "safe threshold of 85%",
        lambda app, score: (
            f"Your credit utilization rate of {app['credit_utilization_ratio']:.0%} "
            "is significantly above our acceptable threshold of 85%, indicating "
            "that your existing credit lines are nearly fully drawn."
        ),
    ),
    (
        "hard inquiries",
        lambda app, score: (
            f"The {int(app['num_hard_inquiries'])} recent hard inquiries on "
            "your credit report, combined with your current score, suggest "
            "elevated near-term credit demand that increases lending risk."
        ),
    ),
    # New v2 templates
    (
        "accounts currently in collections",
        lambda app, score: (
            f"You have {int(app.get('num_collections', 0))} account(s) currently "
            "in collections, which signals unresolved debt obligations and poses "
            "significant credit risk."
        ),
    ),
    (
        "payment history score",
        lambda app, score: (
            f"Your payment history score of {app.get('payment_history_score', 0):.0f}/100 "
            "reflects a pattern of severely delinquent payments, which is the most "
            "heavily weighted factor in credit assessment."
        ),
    ),
    (
        "public records",
        lambda app, score: (
            f"Your credit file contains {int(app.get('public_records', 0))} public "
            "record(s) (bankruptcies, liens, or civil judgments) that disqualify this "
            "application under our underwriting guidelines."
        ),
    ),
    (
        "credit history under 12 months",
        lambda app, score: (
            f"Your credit history of {int(app.get('months_oldest_account', 0))} months "
            "is insufficient to establish reliable creditworthiness at your current "
            "score level."
        ),
    ),
    (
        "unsustainable debt load",
        lambda app, score: (
            f"Your total outstanding debt of ${app.get('total_debt_outstanding', 0):,.0f} "
            f"combined with a debt-to-income ratio of {app['debt_to_income_ratio']:.0%} "
            "indicates your current debt burden is unsustainable relative to your income."
        ),
    ),
    (
        "consecutive on-time payments",
        lambda app, score: (
            f"Your streak of {int(app.get('on_time_payment_streak', 0))} consecutive "
            f"on-time payments combined with credit utilization of "
            f"{app['credit_utilization_ratio']:.0%} suggests recent payment instability "
            "and high credit dependency."
        ),
    ),
]


def _explain_rule(rule: str, applicant: dict, predicted_score: float) -> str:
    """
    Map a single triggered-rule string to a plain-English sentence.

    Parameters
    ----------
    rule            : human-readable rule string from CreditRuleEngine
    applicant       : applicant feature dict (after imputation)
    predicted_score : neural model's credit score prediction

    Returns
    -------
    Plain-English sentence, or the original rule string as a fallback.
    """
    rule_lower = rule.lower()
    for substring, template_fn in _RULE_TEMPLATES:
        if substring in rule_lower:
            return template_fn(applicant, predicted_score)
    return rule  # fallback: echo the rule verbatim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_explanation(
    applicant: dict,
    predicted_score: float,
    decision: str,
    risk_level: str,
    triggered_rules: list,
) -> str:
    """
    Generate a natural-language explanation for a credit decision.

    Parameters
    ----------
    applicant       : applicant feature dict (after null imputation)
    predicted_score : credit score predicted by the neural model
    decision        : "APPROVE" or "DENY"
    risk_level      : "PREMIUM LOW", "LOW", "MEDIUM", "MEDIUM-WATCH", or "HIGH"
    triggered_rules : list of human-readable rule strings from CreditRuleEngine

    Returns
    -------
    str — 2–4 sentence explanation suitable for delivery to the applicant.
    """
    if decision == "DENY":
        sentences = [
            "After reviewing your application, we are unable to approve your "
            "loan request at this time."
        ]
        for rule in triggered_rules:
            sentences.append(_explain_rule(rule, applicant, predicted_score))
        sentences.append(
            "We encourage you to address these factors and reapply in the future."
        )
        return " ".join(sentences)

    if risk_level == "PREMIUM LOW":
        return (
            "Congratulations — your application has been approved with our premium tier. "
            f"Your exceptional payment history score of "
            f"{applicant.get('payment_history_score', 0):.0f}/100 and predicted credit score "
            f"of {predicted_score:.0f} place you among our most creditworthy borrowers. "
            "You qualify for our most competitive rates and terms."
        )

    if risk_level == "LOW":
        return (
            "Congratulations — your application has been approved. "
            f"Your credit profile demonstrates strong financial health with a "
            f"predicted score of {predicted_score:.0f}, well above our approval threshold."
        )

    if risk_level == "MEDIUM-WATCH":
        return (
            f"Your application has been approved. Your predicted credit score of "
            f"{predicted_score:.0f} meets our requirements; however, minor derogatory "
            "marks on your credit file mean your account will be subject to periodic "
            "review. We recommend addressing any open collections or public records "
            "to improve your standing."
        )

    # APPROVE MEDIUM
    return (
        f"Your application has been approved. Your predicted credit score of "
        f"{predicted_score:.0f} meets our requirements. We recommend continuing "
        "to reduce outstanding balances to strengthen your credit profile further."
    )


# ---------------------------------------------------------------------------
# Demo — 8 sample applicants covering all 5 risk tiers
# ---------------------------------------------------------------------------

_DEMO_CASES = [
    # 1 — APPROVE LOW
    {
        "applicant": {
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
        "predicted_score": 762.0,
        "decision": "APPROVE",
        "risk_level": "LOW",
        "triggered_rules": [],
    },
    # 2 — APPROVE MEDIUM
    {
        "applicant": {
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
        "predicted_score": 698.0,
        "decision": "APPROVE",
        "risk_level": "MEDIUM",
        "triggered_rules": [],
    },
    # 3 — DENY (low score + late payments + no streak)
    {
        "applicant": {
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
        "predicted_score": 542.0,
        "decision": "DENY",
        "risk_level": "HIGH",
        "triggered_rules": [
            "Predicted credit score 542 is below the minimum threshold of 580",
            "3 or more late payments combined with a score below 670",
            "Less than 6 consecutive on-time payments combined with credit utilization of 72%",
        ],
    },
    # 4 — DENY (high DTI)
    {
        "applicant": {
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
        "predicted_score": 695.0,
        "decision": "DENY",
        "risk_level": "HIGH",
        "triggered_rules": [
            "Debt-to-income ratio 55% exceeds the regulatory limit of 43%",
        ],
    },
    # 5 — DENY (bankruptcy + high utilisation + no payment streak)
    {
        "applicant": {
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
        "predicted_score": 605.0,
        "decision": "DENY",
        "risk_level": "HIGH",
        "triggered_rules": [
            "Active bankruptcy record within the past 24 months",
            "Credit utilization 91% exceeds the safe threshold of 85%",
            "Less than 6 consecutive on-time payments combined with credit utilization of 91%",
        ],
    },
    # 6 — DENY (collections — rule 7)
    {
        "applicant": {
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
        "predicted_score": 620.0,
        "decision": "DENY",
        "risk_level": "HIGH",
        "triggered_rules": [
            "2 or more accounts currently in collections",
        ],
    },
    # 7 — DENY (excessive debt + high DTI — rule 11)
    {
        "applicant": {
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
        "predicted_score": 720.0,
        "decision": "DENY",
        "risk_level": "HIGH",
        "triggered_rules": [
            "Debt-to-income ratio 48% exceeds the regulatory limit of 43%",
            "Total outstanding debt $125,000 combined with debt-to-income ratio 48% "
            "indicates unsustainable debt load",
        ],
    },
    # 8 — APPROVE PREMIUM LOW
    {
        "applicant": {
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
        "predicted_score": 790.0,
        "decision": "APPROVE",
        "risk_level": "PREMIUM LOW",
        "triggered_rules": [],
    },
]


def main():
    """Generate and print explanations for eight sample credit decisions."""
    print("=" * 70)
    print("TEMPLATE EXPLAINER — SAMPLE EXPLANATIONS")
    print("=" * 70)

    for i, case in enumerate(_DEMO_CASES, start=1):
        explanation = get_explanation(
            applicant=case["applicant"],
            predicted_score=case["predicted_score"],
            decision=case["decision"],
            risk_level=case["risk_level"],
            triggered_rules=case["triggered_rules"],
        )
        outcome = f"{case['decision']} ({case['risk_level']})"
        print(f"\nApplicant {i} — {outcome}")
        print(f"  {explanation}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
