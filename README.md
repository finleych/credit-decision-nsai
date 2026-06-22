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
                     Raw Applicant Features
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Stage 1: Neural Network   │
                │   MLP (PyTorch)             │
                │   11 features → score       │
                └─────────────────────────────┘
                              │
                     Predicted Credit Score (300–850)
                              │
                              ▼
                ┌─────────────────────────────┐
                │  Stage 2: Symbolic Rule      │
                │  Engine                      │
                │  6 hard DENY rules           │
                │  → APPROVE / DENY + Risk     │
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

# 3. Train MLP + save model → models/
python neural_model.py

# 4. Demo the symbolic rule engine on 5 hardcoded applicants
python rule_engine.py

# 5. Demo the template explainer on 5 sample outcomes
python explainer.py

# 6. Full end-to-end pipeline on 10 fresh random applicants
python pipeline.py
```

---

## File Descriptions

| File | Description |
|---|---|
| `generate_data.py` | Generates 10,000 synthetic applicants with realistic credit features and a formula-derived credit score target |
| `descriptive_stats.py` | Loads the dataset and produces printed statistics plus three exploratory plots saved to `plots/` |
| `neural_model.py` | Defines and trains a PyTorch MLP regressor, evaluates it on a held-out test set, and saves the model and scaler to `models/` |
| `rule_engine.py` | Symbolic rule engine that converts a predicted score and raw features into an APPROVE/DENY decision with traceable triggered rules |
| `explainer.py` | Template-based natural-language explainer that converts rule engine output into a human-readable explanation — no API key required |
| `pipeline.py` | End-to-end demo that loads trained artefacts, generates 10 new applicants, runs the full three-stage pipeline, and prints a formatted summary |

---

## Generated Artefacts

| Path | Contents |
|---|---|
| `data/credit_data.csv` | Synthetic dataset (10,000 rows × 12 columns) |
| `models/credit_model.pt` | Trained MLP state dict |
| `models/scaler.pkl` | Fitted StandardScaler + training median for imputation |
| `plots/score_distribution.png` | Histogram of credit score distribution |
| `plots/correlation_heatmap.png` | Feature correlation matrix |
| `plots/credit_tiers.png` | Applicant counts by FICO credit tier |
| `plots/training_loss.png` | MSE loss curve over 50 training epochs |
| `plots/pred_vs_actual.png` | Predicted vs actual scatter on the test set |
