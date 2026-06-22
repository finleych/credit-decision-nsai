# Neuro-Symbolic AI Credit Decision System

A modular Python project demonstrating the **Neuro|Symbolic pipeline paradigm**
applied to consumer credit scoring. Built for learning and presentation purposes.
Requires **no API key** and runs fully offline.

---

## What is Neuro-Symbolic AI?

Neuro-Symbolic AI (NSAI) combines the pattern-recognition power of neural
networks with the interpretability and logical rigor of symbolic AI (rules,
ontologies, formal logic).

- **Neural networks** excel at finding subtle patterns in high-dimensional, noisy
  data — but they operate as opaque "black boxes."
- **Symbolic AI** is transparent, auditable, and enforceable — but struggles with
  raw, messy real-world data on its own.

NSAI seeks the best of both: let the neural component handle *perception and
prediction*, while the symbolic component handles *structured reasoning and
decision-making*.

---

## The Five Kautz Paradigms

Henry Kautz identified five distinct paradigms of Neuro-Symbolic integration in
his 2020 AAAI Robert S. Engelmore Memorial Lecture:

| Paradigm | Description |
|---|---|
| **Symbolic Neuro Symbolic** | Input and output are symbolic; a neural net processes the middle representation |
| **Symbolic[Neuro]** | A symbolic AI system calls a neural network as a sub-process (e.g., MCTS + value network in AlphaGo) |
| **Neuro\|Symbolic** *(this project)* | Neural net output feeds sequentially into a symbolic reasoner — a pipeline where each stage has a distinct role |
| **Neuro:Symbolic→Neuro** | Symbolic knowledge constrains or regularises neural training (e.g., physics-informed neural networks) |
| **Neuro_{Symbolic}** | A neural architecture with symbolic operations embedded within it (e.g., differentiable logic, neural theorem provers) |

---

## Why Neuro|Symbolic Pipeline for Credit Scoring?

The **Neuro|Symbolic pipeline** is the right paradigm for regulated lending
because:

1. **Explainability** — Applicants and regulators are legally entitled to a
   reason for credit decisions (Equal Credit Opportunity Act, GDPR Article 22).
   The symbolic rule engine produces a traceable, human-readable audit trail.

2. **Regulatory compliance** — Hard limits such as DTI ≤ 43% (CFPB qualified
   mortgage rule) encode legal constraints that a neural net alone *cannot
   guarantee* — a model trained on noisy data may occasionally violate them.

3. **Auditability** — Every decision can be replayed deterministically: the
   predicted score follows from the model weights, and the rules are fixed and
   inspectable. There is no randomness in production.

4. **Trust and override** — Loan officers can review and override the symbolic
   layer without touching the neural model, giving human experts meaningful
   control.

---

## Architecture

```
                     Raw Applicant Features (18)
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Stage 1: Neural Network   │
                │   MLP (PyTorch)             │
                │   18 features → score       │
                │   R² = 0.742 on test set    │
                └─────────────────────────────┘
                              │
                     Predicted Credit Score (300–850)
                              │
                              ▼
                ┌─────────────────────────────┐
                │  Stage 2: Symbolic Rule     │
                │  Engine                     │
                │  12 hard DENY rules         │
                │  5 risk tiers               │
                │  → APPROVE / DENY + Risk    │
                └─────────────────────────────┘
                              │
                    Decision + Triggered Rules
                              │
                              ▼
                ┌─────────────────────────────┐
                │  Stage 3: Template          │
                │  Explanation Layer          │
                │  (no LLM, no API key)       │
                └─────────────────────────────┘
                              │
                              ▼
              Final Decision + Natural Language Reason
```

---

## No API Key Required

All natural-language explanations are generated programmatically by a
template-based system in `explainer.py`. There is no call to OpenAI, Anthropic,
or any other external service. The project runs entirely on your local machine.

---

## Features (18 total)

| Feature | Distribution | FICO Factor |
|---|---|---|
| `age` | Uniform [18, 75] | Credit history length |
| `annual_income` | Log-normal, median ~$55k | Repayment capacity |
| `employment_length_years` | Gamma, right-skewed | Capacity stability |
| `debt_to_income_ratio` | Uniform [0.05, 0.65] | Amounts owed |
| `num_credit_accounts` | Uniform integers [1, 20] | Credit mix |
| `num_late_payments` | Poisson (λ=0.8), ~45% at 0 | Payment history |
| `num_hard_inquiries` | Poisson (λ=2.0) | New credit |
| `credit_utilization_ratio` | Beta(2,3), peaks ~0.28 | Amounts owed |
| `loan_amount_requested` | Uniform [1k, 50k] | Not in score formula |
| `has_bankruptcy` | Bernoulli (p=0.07), ~7% rate | Derogatory marks |
| `months_since_last_delinquency` | Uniform [0,120], 10% null | Payment history |
| `payment_history_score` | Beta(8,2)×100, skewed high | Payment history ~35% |
| `total_debt_outstanding` | Log-normal, median ~$33k | Amounts owed ~30% |
| `num_collections` | Poisson (λ=0.15), ~86% at 0 | Derogatory marks |
| `months_oldest_account` | Gamma, mean ~96 months | Credit history ~15% |
| `credit_mix_score` | Poisson (λ=4), clipped [0,10] | Credit mix ~10% |
| `on_time_payment_streak` | Exponential (scale=24) | Payment history |
| `public_records` | Poisson (λ=0.08), ~92% at 0 | Derogatory marks |

---

## Symbolic Rule Engine (12 rules)

All 12 rules are checked for every applicant — multiple rules can trigger
simultaneously. Any triggered rule results in a DENY.

| # | Rule | Trigger condition |
|---|---|---|
| 1 | Minimum score | score < 580 |
| 2 | DTI cap (CFPB qualified mortgage) | DTI > 43% |
| 3 | Recent bankruptcy | bankruptcy filed AND months since delinquency < 24 |
| 4 | Late payments + low score | ≥ 3 late payments AND score < 670 |
| 5 | High credit utilization | utilization > 85% |
| 6 | Hard inquiries + low score | ≥ 5 inquiries AND score < 700 |
| 7 | Accounts in collections | num_collections ≥ 2 |
| 8 | Severely delinquent payment history | payment_history_score < 40 |
| 9 | Multiple public records | public_records ≥ 2 |
| 10 | Thin credit history + low score | oldest account < 12 months AND score < 650 |
| 11 | Excessive total debt | debt > $100k AND DTI > 35% |
| 12 | No payment streak + high utilization | streak < 6 months AND utilization > 70% |

### Risk Tiers

| Tier | Condition |
|---|---|
| **PREMIUM LOW** | Approved — payment_history_score ≥ 90 AND score ≥ 750 |
| **LOW** | Approved — score ≥ 740 |
| **MEDIUM** | Approved — score < 740, no derogatory marks |
| **MEDIUM-WATCH** | Approved — score < 740, but has collections or public records on file |
| **HIGH** | Denied — any rule triggered |

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch is included in `requirements.txt`. For GPU support, install the
appropriate CUDA build from [pytorch.org](https://pytorch.org) separately.

---

## Running the Pipeline

Run each script in order:

```bash
# 1. Generate 10,000 synthetic applicants → data/credit_data.csv
python generate_data.py

# 2. Exploratory analysis + plots → plots/
python descriptive_stats.py

# 3. Train MLP (150 epochs) + save model → models/
python neural_model.py

# 4. Demo the symbolic rule engine on 8 sample applicants
python rule_engine.py

# 5. Demo the template explainer on 8 sample outcomes
python explainer.py

# 6. Full end-to-end pipeline on 10 fresh random applicants
python pipeline.py
```

---

## File Descriptions

| File | Description |
|---|---|
| `generate_data.py` | Generates 10,000 synthetic applicants with 18 realistic credit features and a formula-derived credit score target using a rank→quantile transform |
| `descriptive_stats.py` | Loads the dataset and produces printed statistics plus three exploratory plots saved to `plots/` |
| `neural_model.py` | Defines and trains a PyTorch MLP regressor (150 epochs), evaluates it on a held-out test set (R²=0.742), and saves the model and scaler to `models/` |
| `rule_engine.py` | Symbolic rule engine with 12 hard DENY rules and 5 risk tiers that converts a predicted score and raw features into an APPROVE/DENY decision with traceable triggered rules |
| `explainer.py` | Template-based natural-language explainer that converts rule engine output into a human-readable explanation — no API key required |
| `pipeline.py` | End-to-end demo that loads trained artefacts, generates 10 new applicants, runs the full three-stage pipeline, and prints a formatted summary |

---

## Generated Artefacts

| Path | Contents |
|---|---|
| `data/credit_data.csv` | Synthetic dataset (10,000 rows × 19 columns) |
| `models/credit_model.pt` | Trained MLP state dict |
| `models/scaler.pkl` | Fitted StandardScaler + training median for imputation |
| `plots/score_distribution.png` | Histogram of credit score distribution |
| `plots/correlation_heatmap.png` | Feature correlation matrix (18 features) |
| `plots/credit_tiers.png` | Applicant counts by FICO credit tier |
| `plots/training_loss.png` | MSE loss curve over 150 training epochs |
| `plots/pred_vs_actual.png` | Predicted vs actual scatter on the test set |
