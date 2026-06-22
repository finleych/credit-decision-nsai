# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running scripts

All scripts must be run from the project root using the local virtualenv:

```bash
cd /Users/finleychen/credit_decision_nsai
.venv/bin/python3 <script>.py
```

Scripts must be run in this order — each depends on the output of the previous:

```bash
.venv/bin/python3 generate_data.py      # writes data/credit_data.csv
.venv/bin/python3 descriptive_stats.py  # writes plots/ (3 PNGs)
.venv/bin/python3 neural_model.py       # writes models/ and plots/ (2 more PNGs)
.venv/bin/python3 rule_engine.py        # standalone demo, no file deps
.venv/bin/python3 explainer.py          # standalone demo, no file deps
.venv/bin/python3 pipeline.py           # requires models/ from neural_model.py
```

If you change `generate_data.py`, you must re-run everything from the top to keep the trained model consistent with the data.

## Architecture

This is a **Neuro|Symbolic pipeline** — three sequential stages with no LLM or API key:

```
generate_data.py  →  neural_model.py  →  rule_engine.py  →  explainer.py
  (CSV target)        (score pred.)       (APPROVE/DENY)     (NL reason)
       ↑                    ↑
  FEATURE_COLUMNS     CreditMLP class
  generate_single_applicant()
  (both imported by pipeline.py)
```

### Cross-module imports

`pipeline.py` imports from three other project files:
- `generate_data.FEATURE_COLUMNS` — the canonical ordered list of 18 feature names; used to build the numpy array in the correct column order for the scaler and model
- `generate_data.generate_single_applicant()` — generates new applicants without loading the CSV
- `neural_model.CreditMLP` — the model class needed to reconstruct the architecture before loading weights

`neural_model.py` imports `FEATURE_COLUMNS` from `generate_data` to ensure training column order matches inference.

### Saved artefacts

`models/scaler.pkl` is a **dict**, not a bare StandardScaler:
```python
{"scaler": StandardScaler, "median_msd": float}
```
The `median_msd` value is the training-set median of `months_since_last_delinquency`, used to impute nulls in `pipeline.py` with exactly the same value seen during training. Load it with `joblib.load`.

`models/credit_model.pt` contains only the model state dict (no architecture). Reconstruct with `CreditMLP(n_features=len(FEATURE_COLUMNS))` then call `load_state_dict`.

### Credit score formula (generate_data.py)

The target distribution matches real US FICO tiers:
- Poor (<580) 16% | Fair (580–669) 18% | Good (670–739) 21% | Very Good (740–799) 25% | Exceptional (800+) 20%

`compute_credit_score()` achieves this via a **two-step rank→quantile transform**:
1. Computes a weighted quality index `q` with weights roughly proportional to real FICO importance (see table below). Gaussian noise σ=0.25 is added.
2. Rank-sorts `q`, converts to percentile, then maps through a piecewise-linear inverse CDF calibrated to the tier breakpoints above.

This guarantees the exact tier distribution while preserving all correlation directions (the rank→score mapping is monotone). **Do not replace this with a plain linear formula + clip** — that causes ~79% of applicants to land in the Poor tier.

**Quality index weights (v2 — 18 features):**

| Feature | Direction | Weight | FICO factor |
|---------|-----------|--------|-------------|
| `payment_history_score` | positive | 1.40 | Payment history ~35% |
| `annual_income` | positive | 1.00 | Capacity (proxy) |
| `has_bankruptcy` | negative | 1.00 | Derogatory marks |
| `credit_utilization_ratio` | negative | 0.70 | Amounts owed ~30% |
| `debt_to_income_ratio` | negative | 0.60 | Capacity |
| `num_collections` | negative | 0.80 | Derogatory marks |
| `public_records` | negative | 0.90 | Derogatory marks |
| `months_oldest_account` | positive | 0.45 | Credit history ~15% |
| `on_time_payment_streak` | positive | 0.40 | Payment history |
| `months_since_last_delinquency` | positive | 0.40 | Payment history |
| `total_debt_outstanding` | negative | 0.50 | Amounts owed |
| `employment_length_years` | positive | 0.50 | Capacity |
| `num_late_payments` | negative | 0.50 | Payment history |
| `num_credit_accounts` | positive | 0.30 | Amounts owed |
| `credit_mix_score` | positive | 0.30 | Credit mix ~10% |
| `num_hard_inquiries` | negative | 0.30 | New credit ~10% |
| `age` | positive | 0.25 | Credit history |

`loan_amount_requested` carries no weight in the formula (independent of creditworthiness).

**Noise note:** σ=0.40 was tested but the rank→quantile transform amplifies noise into score variance, pushing R² below 0.70. σ=0.25 achieves R²=0.74 while the 18 realistic feature distributions provide the added realism.

### Rule engine (rule_engine.py)

`CreditRuleEngine.evaluate()` applies **12 hard DENY rules** in order (all rules are checked even after the first fires — multiple rules can trigger). Rules 7–12 read v2 features via `.get()` with safe defaults so the engine stays backward-compatible with v1 feature dicts.

| # | Trigger | Condition |
|---|---------|-----------|
| 1 | Min score | predicted_score < 580 |
| 2 | DTI cap (CFPB qualified mortgage) | dti > 43% |
| 3 | Recent bankruptcy | has_bankruptcy=1 AND months_since_last_delinquency < 24 |
| 4 | Late payments + low score | num_late_payments ≥ 3 AND score < 670 |
| 5 | High utilization | credit_utilization_ratio > 85% |
| 6 | Hard inquiries + low score | num_hard_inquiries ≥ 5 AND score < 700 |
| 7 | Collections | num_collections ≥ 2 |
| 8 | Delinquent payment history | payment_history_score < 40 |
| 9 | Public records | public_records ≥ 2 |
| 10 | Thin credit history | months_oldest_account < 12 AND score < 650 |
| 11 | Excessive total debt | total_debt_outstanding > $100k AND dti > 35% |
| 12 | No payment streak + high util | on_time_payment_streak < 6 AND util > 70% |

**Risk tiers (APPROVE only; any DENY → HIGH):**

| Tier | Condition |
|------|-----------|
| `PREMIUM LOW` | payment_history_score ≥ 90 AND score ≥ 750 |
| `LOW` | score ≥ 740 |
| `MEDIUM-WATCH` | score < 740 AND (num_collections > 0 OR public_records > 0) |
| `MEDIUM` | score < 740, no derogatory marks |
| `HIGH` | any rule fired → DENY |

### Explainer (explainer.py)

`explainer.get_explanation()` matches triggered rule strings by **substring** (case-insensitive) against `_RULE_TEMPLATES`. The substrings are designed to be unique across all 12 rules — do not use substrings that appear in more than one rule string:

| Rule | Matching substring |
|------|--------------------|
| 1 | `"below the minimum threshold"` |
| 2 | `"regulatory limit of 43%"` |
| 3 | `"bankruptcy"` |
| 4 | `"late payments"` |
| 5 | `"safe threshold of 85%"` |
| 6 | `"hard inquiries"` |
| 7 | `"accounts currently in collections"` |
| 8 | `"payment history score"` |
| 9 | `"public records"` |
| 10 | `"credit history under 12 months"` |
| 11 | `"unsustainable debt load"` |
| 12 | `"consecutive on-time payments"` |

If you rename rule strings in `rule_engine.py`, update the matching substrings in `explainer.py`. Keep substrings unique — rule 11 contains the phrase "debt-to-income ratio" (same as rule 2), so rule 2 uses `"regulatory limit of 43%"` and rule 11 uses `"unsustainable debt load"` to avoid collision.

## Feature distributions

All 18 features and their sampling distributions:

| Feature | Distribution | Notes |
|---------|-------------|-------|
| `age` | `integers(18, 76)` | Uniform |
| `annual_income` | `lognormal(10.9, 0.65)` clipped [20k, 400k] | Right-skewed, median ~$55k |
| `employment_length_years` | `gamma(1.5, 4.5)` clipped [0, 35] | Right-skewed |
| `debt_to_income_ratio` | `uniform(0.05, 0.65)` | Uniform |
| `num_credit_accounts` | `integers(1, 21)` | Uniform |
| `num_late_payments` | `Poisson(λ=0.8)` clipped [0, 10] | Right-skewed; ~45% have 0 |
| `num_hard_inquiries` | `Poisson(λ=2.0)` clipped [0, 8] | Right-skewed |
| `credit_utilization_ratio` | `beta(2, 3)` | Peaks ~0.28, not uniform |
| `loan_amount_requested` | `uniform(1k, 50k)` | Uniform, not in score formula |
| `has_bankruptcy` | `Bernoulli(p=0.07)` | ~7% rate; real US rate 5–8% |
| `months_since_last_delinquency` | `uniform(0, 120)` + 10% null | Nulls imputed with training median |
| `payment_history_score` | `beta(8, 2) × 100` clipped [0, 100] | Skewed high; most people pay on time |
| `total_debt_outstanding` | `lognormal(10.4, 0.8)` clipped [5k, 150k] | Right-skewed, median ~$33k |
| `num_collections` | `Poisson(λ=0.15)` clipped [0, 5] | ~86% have 0 |
| `months_oldest_account` | `gamma(3, 32)` clipped [6, 360] | Mean ~96 months |
| `credit_mix_score` | `Poisson(λ=4)` clipped [0, 10] | Discrete |
| `on_time_payment_streak` | `exponential(scale=24)` clipped [0, 120] | Right-skewed |
| `public_records` | `Poisson(λ=0.08)` clipped [0, 3] | ~92% have 0 |

`num_late_payments` and `num_hard_inquiries` use Poisson distributions so most applicants have low values — matching real credit report patterns. Both `generate_raw_features()` and `generate_single_applicant()` must use the same distributions for every feature.

### neural_model.py training settings (v2)

Epochs increased from 50 → 150 to handle the larger 18-feature input. R² on test set: **0.742**.
