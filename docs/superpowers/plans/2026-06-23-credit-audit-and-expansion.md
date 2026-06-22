# Credit Decision Audit & Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 distribution/realism issues in generate_data.py, add 7 new credit features, expand rule_engine.py from 6 to 12 rules with 2 new approval tiers, update explainer.py templates, retrain the neural model, and update CLAUDE.md.

**Architecture:** All changes flow through generate_data.py (data + features) → neural_model.py (retrain only) → rule_engine.py (new rules) → explainer.py (new templates). pipeline.py needs no code changes — it uses FEATURE_COLUMNS dynamically.

**Tech Stack:** Python 3.11+, NumPy, pandas, PyTorch, scikit-learn, joblib; `.venv/bin/python3` from project root.

## Global Constraints

- Run all scripts from `/Users/finleychen/credit_decision_nsai` via `.venv/bin/python3 <script>.py`
- FEATURE_COLUMNS in generate_data.py is the single source of truth for column order — neural_model.py and pipeline.py import it
- scaler.pkl is a dict `{"scaler": StandardScaler, "median_msd": float}` — do not change this structure
- No LLM, no API key, no internet — all generation is deterministic or seeded NumPy
- No new nullable fields beyond `months_since_last_delinquency` — pipeline.py imputation logic stays unchanged
- R² must remain ≥ 0.70 after retraining on expanded data

---

## Audit Findings (pre-implementation)

| Check | Current | Target | Status |
|-------|---------|--------|--------|
| `has_bankruptcy` rate | 10.1% | 5–8% | **FAIL** |
| `annual_income` skewness | −0.006 (uniform) | >1.5 right-skew | **FAIL** |
| `employment_length_years` skewness | −0.016 (uniform) | right-skewed | **FAIL** |
| `credit_utilization_ratio` skewness | −0.015 (uniform) | not uniform | **FAIL** |
| `num_late_payments` skewness | 0.85 (want >1.0) | >1.0 | **FAIL** |
| Max `|corr|` with credit_score | 0.59 (`has_bankruptcy`) | <0.70 | **PASS** |
| Noise (residual std) | 117.9 pts | >60 pts | **PASS** |

---

## File Map

| File | Change type | What changes |
|------|-------------|--------------|
| `generate_data.py` | Modify | Fix 5 distributions, add 7 features to FEATURE_COLUMNS + generate_raw_features + generate_single_applicant + compute_credit_score |
| `rule_engine.py` | Modify | Add rules 7–12, PREMIUM LOW + MEDIUM-WATCH risk logic, 3 new + 5 updated sample applicants |
| `explainer.py` | Modify | Add 6 new _RULE_TEMPLATES entries, handle PREMIUM LOW + MEDIUM-WATCH in get_explanation, update _DEMO_CASES |
| `pipeline.py` | No change | Uses FEATURE_COLUMNS dynamically; retrain artifacts replace old ones |
| `neural_model.py` | No change | Retrain only — imports FEATURE_COLUMNS dynamically |
| `CLAUDE.md` | Modify | Update feature list, rule count, risk levels, distributions section |

---

## Task 1: Fix generate_data.py — Distributions + 7 New Features

**Files:**
- Modify: `generate_data.py`

**What to fix:**
1. `has_bankruptcy`: `p=[0.93, 0.07]` (was `[0.90, 0.10]`)
2. `annual_income`: log-normal `np.clip(rng.lognormal(mean=10.9, sigma=0.65, size=n), 20_000, 400_000)` — gives realistic right-skew with median ~$55k
3. `employment_length_years`: gamma `np.clip(rng.gamma(shape=1.5, scale=4.5, size=n), 0, 35)` — right-skewed, most people <10 yrs
4. `credit_utilization_ratio`: beta `rng.beta(a=2, b=3, size=n)` — peaks around 0.3, not uniform
5. `num_late_payments`: Poisson `lam=0.8` (was 1.5) — more zeroes, higher skew
6. Noise: `rng.normal(0, 0.40, size=n)` (was 0.25) — more realistic unpredictability

**New features to add to `generate_raw_features()`:**

```python
# payment_history_score (0-100): heavily right-skewed to high values — most people pay on time
payment_history = np.clip(rng.beta(a=8, b=2, size=n) * 100, 0, 100)

# total_debt_outstanding (5k-150k): right-skewed via lognormal
total_debt = np.clip(rng.lognormal(mean=10.4, sigma=0.8, size=n), 5_000, 150_000)

# num_collections (0-5): very right-skewed — most at 0
num_collections = np.clip(rng.poisson(lam=0.15, size=n), 0, 5).astype(float)

# months_oldest_account (6-360): gamma skewed, mean ~8 years (96 months)
months_oldest = np.clip(rng.gamma(shape=3, scale=32, size=n), 6, 360)

# credit_mix_score (0-10): discrete, Poisson(4) clipped
credit_mix = np.clip(rng.poisson(lam=4, size=n), 0, 10).astype(float)

# on_time_payment_streak (0-120): exponential, most people 0-36 months
payment_streak = np.clip(rng.exponential(scale=24, size=n), 0, 120)

# public_records (0-3): very rare, Poisson(0.08)
public_records = np.clip(rng.poisson(lam=0.08, size=n), 0, 3).astype(float)
```

**Update `compute_credit_score()` — add to q after existing terms:**

```python
# New positive contributors
q += (df["payment_history_score"].values / 100) * 1.40   # ~35% FICO weight
q += df["months_oldest_account"].values / 360 * 0.45      # ~15% FICO weight
q += df["credit_mix_score"].values / 10 * 0.30
q += df["on_time_payment_streak"].values / 120 * 0.40

# New negative contributors
q -= (df["total_debt_outstanding"].values - 5_000) / (150_000 - 5_000) * 0.50
q -= df["num_collections"].values / 5 * 0.80
q -= df["public_records"].values / 3 * 0.90
```

**Update `FEATURE_COLUMNS`** to append 7 new names (keep existing 11 first for backward reference during development):

```python
FEATURE_COLUMNS = [
    "age", "annual_income", "employment_length_years", "debt_to_income_ratio",
    "num_credit_accounts", "num_late_payments", "num_hard_inquiries",
    "credit_utilization_ratio", "loan_amount_requested", "has_bankruptcy",
    "months_since_last_delinquency",
    # New in v2
    "payment_history_score", "total_debt_outstanding", "num_collections",
    "months_oldest_account", "credit_mix_score", "on_time_payment_streak",
    "public_records",
]
```

**Update `generate_single_applicant()`** — add the same 7 features using identical distribution logic (but scalar draws).

- [ ] **Step 1: Replace generate_data.py** with the full updated version (distributions + new features)
- [ ] **Step 2: Run `generate_data.py` and verify output**
  - Run: `.venv/bin/python3 generate_data.py`
  - Verify: `data/credit_data.csv` has 18+1 columns (18 features + credit_score)
- [ ] **Step 3: Verify audit results pass**
  - Run the inline audit checks and confirm has_bankruptcy ≤8%, income skewness >1.5, no corr >0.65

---

## Task 2: Update rule_engine.py — 12 Rules + 2 New Risk Levels

**Files:**
- Modify: `rule_engine.py`

**New feature reads in `evaluate()`:**

```python
collections = applicant_features.get("num_collections", 0)
pay_hist    = applicant_features.get("payment_history_score", 100)
pub_rec     = applicant_features.get("public_records", 0)
moa         = applicant_features.get("months_oldest_account", 999)
total_debt  = applicant_features.get("total_debt_outstanding", 0)
streak      = applicant_features.get("on_time_payment_streak", 99)
```

**Rules 7–12:**

```python
# Rule 7 — collections
if collections >= 2:
    triggered_rules.append("2 or more accounts currently in collections")
    decision = "DENY"

# Rule 8 — payment history score
if pay_hist < 40:
    triggered_rules.append(
        f"Payment history score {pay_hist:.0f}/100 indicates severely delinquent payment behavior"
    )
    decision = "DENY"

# Rule 9 — public records
if pub_rec >= 2:
    triggered_rules.append("2 or more public records (bankruptcies, liens, or judgments) on file")
    decision = "DENY"

# Rule 10 — thin credit history
if moa < 12 and score < 650:
    triggered_rules.append("Credit history under 12 months combined with score below 650")
    decision = "DENY"

# Rule 11 — excessive total debt
if total_debt > 100_000 and dti > 0.35:
    triggered_rules.append(
        f"Total outstanding debt ${total_debt:,.0f} combined with debt-to-income ratio {dti:.0%} "
        "indicates unsustainable debt load"
    )
    decision = "DENY"

# Rule 12 — no payment streak + high utilization
if streak < 6 and util > 0.70:
    triggered_rules.append(
        f"Less than 6 consecutive on-time payments combined with credit utilization of {util:.0%}"
    )
    decision = "DENY"
```

**Updated risk tier logic:**

```python
if decision == "APPROVE":
    if pay_hist >= 90 and score >= 750:
        risk_level = "PREMIUM LOW"
    elif score >= 740:
        risk_level = "LOW"
    elif collections > 0 or pub_rec > 0:
        risk_level = "MEDIUM-WATCH"
    else:
        risk_level = "MEDIUM"
else:
    risk_level = "HIGH"
```

**3 new sample applicants** (add after existing 5):

```python
# 6 — triggers Rule 7 (2+ collections) → DENY HIGH
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

# 7 — triggers Rule 11 (excessive debt + high DTI) → DENY HIGH
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
```

**Update existing 5 sample applicants** — add new feature keys with realistic defaults (no nulls).

- [ ] **Step 1: Rewrite rule_engine.py** with 12 rules, updated risk logic, 8 sample applicants
- [ ] **Step 2: Run rule_engine.py and verify output**
  - Run: `.venv/bin/python3 rule_engine.py`
  - Verify: applicant 6 → DENY (collections), applicant 7 → DENY (excessive debt), applicant 8 → APPROVE PREMIUM LOW

---

## Task 3: Update explainer.py — 6 New Templates + New Risk Levels

**Files:**
- Modify: `explainer.py`

**6 new entries for `_RULE_TEMPLATES`** (add after existing 6):

```python
(
    "accounts currently in collections",
    lambda app, score: (
        f"You have {int(app['num_collections'])} account(s) currently in collections, "
        "which signals unresolved debt obligations and poses significant credit risk."
    ),
),
(
    "payment history score",
    lambda app, score: (
        f"Your payment history score of {app['payment_history_score']:.0f}/100 reflects a "
        "pattern of severely delinquent payments, which is the most heavily weighted "
        "factor in credit assessment."
    ),
),
(
    "public records",
    lambda app, score: (
        f"Your credit file contains {int(app['public_records'])} public record(s) "
        "(bankruptcies, liens, or civil judgments) that disqualify this application "
        "under our underwriting guidelines."
    ),
),
(
    "credit history under 12 months",
    lambda app, score: (
        f"Your credit history of {int(app['months_oldest_account'])} months is insufficient "
        "to establish reliable creditworthiness at your current score level."
    ),
),
(
    "unsustainable debt load",
    lambda app, score: (
        f"Your total outstanding debt of ${app['total_debt_outstanding']:,.0f} combined with "
        f"a debt-to-income ratio of {app['debt_to_income_ratio']:.0%} indicates your current "
        "debt burden is unsustainable relative to your income."
    ),
),
(
    "consecutive on-time payments",
    lambda app, score: (
        f"Your streak of {int(app['on_time_payment_streak'])} consecutive on-time payments "
        f"combined with credit utilization of {app['credit_utilization_ratio']:.0%} suggests "
        "recent payment instability and high credit dependency."
    ),
),
```

**Update `get_explanation()`** — add PREMIUM LOW and MEDIUM-WATCH branches:

```python
if risk_level == "PREMIUM LOW":
    return (
        "Congratulations — your application has been approved with our premium tier. "
        f"Your exceptional payment history score and predicted credit score of {predicted_score:.0f} "
        "place you in the top tier of creditworthy borrowers."
    )

# ... existing LOW branch ...

if risk_level == "MEDIUM-WATCH":
    return (
        f"Your application has been approved. Your predicted credit score of {predicted_score:.0f} "
        "meets our requirements; however, minor derogatory marks on your credit file mean your "
        "account will be subject to periodic review. We recommend addressing any open collections "
        "or public records to improve your standing."
    )
```

**Update `_DEMO_CASES`** — add new feature keys to all 5 existing cases (with realistic defaults), and add 3 new demo cases mirroring the new sample applicants.

- [ ] **Step 1: Rewrite explainer.py** with 12 rule templates and new risk level branches
- [ ] **Step 2: Run explainer.py and verify output**
  - Run: `.venv/bin/python3 explainer.py`
  - Verify: 8 demo cases print, PREMIUM LOW and MEDIUM-WATCH wording appears

---

## Task 4: Retrain + Full Pipeline Run

- [ ] **Step 1: Regenerate data**
  - Run: `.venv/bin/python3 generate_data.py`
- [ ] **Step 2: Run descriptive stats**
  - Run: `.venv/bin/python3 descriptive_stats.py`
  - Verify: credit_tiers.png recreated
- [ ] **Step 3: Retrain neural model**
  - Run: `.venv/bin/python3 neural_model.py`
  - Verify: R² ≥ 0.70 printed in output
- [ ] **Step 4: Run rule engine demo**
  - Run: `.venv/bin/python3 rule_engine.py`
- [ ] **Step 5: Run explainer demo**
  - Run: `.venv/bin/python3 explainer.py`
- [ ] **Step 6: Run end-to-end pipeline**
  - Run: `.venv/bin/python3 pipeline.py`
  - Verify: no KeyError, all 10 applicants process, risk levels include PREMIUM LOW or MEDIUM-WATCH in some runs

---

## Task 5: Update CLAUDE.md

**Sections to update:**
- `FEATURE_COLUMNS` reference: 11 → 18 features, list all new names
- Credit score formula weights: add the 7 new feature weights
- Rule engine: 6 → 12 rules, list new rule descriptions
- Risk levels: LOW/MEDIUM/HIGH → PREMIUM LOW / LOW / MEDIUM / MEDIUM-WATCH / HIGH
- Feature distributions section: add 7 new feature distribution descriptions; update the 5 corrected distributions

---

## Post-implementation verification checklist

- [ ] `data/credit_data.csv` has 19 columns (18 features + credit_score)
- [ ] `has_bankruptcy` rate ≤ 8%
- [ ] `annual_income` skewness > 1.5
- [ ] No feature correlation with credit_score > 0.65
- [ ] Rule engine output shows all 3 new risk levels across 8 sample applicants
- [ ] `pipeline.py` runs without KeyError for new features
- [ ] R² ≥ 0.70 after retraining
