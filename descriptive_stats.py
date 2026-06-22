"""
descriptive_stats.py

Exploratory Data Analysis (EDA) for the Neuro-Symbolic AI credit decision
pipeline.

Loads the synthetic dataset produced by generate_data.py and produces:
  - Printed summary statistics (describe, score quartiles, key risk flags)
  - plots/score_distribution.png  — histogram of credit score
  - plots/correlation_heatmap.png — full feature correlation matrix
  - plots/credit_tiers.png        — applicant counts by FICO tier

Neuro-Symbolic role: validation step that helps confirm the synthetic data has
realistic feature relationships before training the neural model.
"""

import matplotlib
matplotlib.use("Agg")  # headless rendering — must precede pyplot import

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"


def load_data() -> pd.DataFrame:
    """Load credit_data.csv and return the DataFrame."""
    path = DATA_DIR / "credit_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run generate_data.py first."
        )
    return pd.read_csv(path)


def print_summary(df: pd.DataFrame) -> None:
    """Print df.describe() and key credit-score statistics."""
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(df.describe().to_string())

    cs = df["credit_score"]
    print("\nCredit Score Summary")
    print(f"  Average : {cs.mean():.2f}")
    print(f"  Median  : {cs.median():.2f}")
    print(f"  Min     : {cs.min():.2f}")
    print(f"  Max     : {cs.max():.2f}")

    n = len(df)
    pct_bankruptcy   = df["has_bankruptcy"].mean() * 100
    pct_late         = (df["num_late_payments"] >= 3).mean() * 100
    pct_high_util    = (df["credit_utilization_ratio"] > 0.85).mean() * 100

    print("\nRisk Flag Rates")
    print(f"  Applicants with bankruptcy           : {pct_bankruptcy:.1f}%")
    print(f"  Applicants with 3+ late payments     : {pct_late:.1f}%")
    print(f"  Applicants with utilization > 85%    : {pct_high_util:.1f}%")


def plot_score_distribution(df: pd.DataFrame) -> None:
    """Save a histogram of credit score distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["credit_score"], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(580, color="#e74c3c", linestyle="--", linewidth=1.2, label="Poor/Fair (580)")
    ax.axvline(670, color="#f39c12", linestyle="--", linewidth=1.2, label="Fair/Good (670)")
    ax.axvline(740, color="#27ae60", linestyle="--", linewidth=1.2, label="Good/Very Good (740)")
    ax.set_xlabel("Credit Score", fontsize=12)
    ax.set_ylabel("Number of Applicants", fontsize=12)
    ax.set_title("Credit Score Distribution (n=10,000)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = PLOTS_DIR / "score_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a full feature correlation heatmap."""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "correlation_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_credit_tiers(df: pd.DataFrame) -> None:
    """Save a bar chart of applicant counts per FICO credit tier."""
    bins   = [299, 579, 669, 739, 799, 851]
    labels = ["Poor\n(<580)", "Fair\n(580–669)", "Good\n(670–739)",
              "Very Good\n(740–799)", "Exceptional\n(800+)"]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#27ae60", "#8e44ad"]

    df = df.copy()
    df["tier"] = pd.cut(df["credit_score"], bins=bins, labels=labels)
    counts = df["tier"].value_counts().reindex(labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            f"{val:,}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("Credit Tier", fontsize=12)
    ax.set_ylabel("Number of Applicants", fontsize=12)
    ax.set_title("Applicants by Credit Tier", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "credit_tiers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def main():
    """Run the full EDA pipeline."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print_summary(df)

    print("\nGenerating plots…")
    plot_score_distribution(df)
    plot_correlation_heatmap(df)
    plot_credit_tiers(df)
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
