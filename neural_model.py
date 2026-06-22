"""
neural_model.py

Stage 1 (Neural Perception) of the Neuro-Symbolic AI credit decision pipeline.

Trains a multi-layer perceptron (MLP) in PyTorch to predict an applicant's
credit score (300–850) from their 11 financial features. The model learns
non-linear relationships that would be hard to capture with hand-written rules.

Its output — a continuous predicted score — feeds directly into Stage 2 (the
symbolic rule engine), embodying the Neuro|Symbolic pipeline paradigm: neural
handles perception, symbolic handles reasoning.

Artifacts saved:
  models/credit_model.pt  — model state dict (PyTorch)
  models/scaler.pkl       — dict with StandardScaler + training median MSD
  plots/training_loss.png — MSE loss curve over 50 epochs
  plots/pred_vs_actual.png — scatter of predictions vs ground-truth on test set
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

from generate_data import FEATURE_COLUMNS

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
PLOTS_DIR  = Path(__file__).parent / "plots"


# ---------------------------------------------------------------------------
# Model definition (also imported by pipeline.py)
# ---------------------------------------------------------------------------

class CreditMLP(nn.Module):
    """
    Multi-layer perceptron for credit score regression.

    Architecture:
        Input(n_features) → Linear(128) → ReLU → Dropout(0.2)
                          → Linear(64)  → ReLU → Dropout(0.2)
                          → Linear(32)  → ReLU
                          → Linear(1)
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar score prediction per sample (shape: [batch])."""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(
    model: CreditMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> list[float]:
    """
    Train the MLP with MSE loss and Adam optimiser.

    Parameters
    ----------
    model      : uninitialised CreditMLP
    X_train    : scaled training features (numpy)
    y_train    : training targets (numpy)
    epochs     : number of full passes over the training set
    batch_size : mini-batch size
    lr         : Adam learning rate

    Returns
    -------
    List of average MSE loss values per epoch.
    """
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_batch)

        avg_loss = running_loss / len(X_train)
        epoch_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return epoch_losses


def evaluate_model(
    model: CreditMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """
    Run inference on the test set and return predictions plus metrics.

    Returns
    -------
    (y_pred, mae, rmse, r2)
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).numpy()

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)
    return y_pred, mae, rmse, r2


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_loss_curve(losses: list[float]) -> None:
    """Save the training loss curve to plots/training_loss.png."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, color="#4C72B0", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = PLOTS_DIR / "training_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_pred_vs_actual(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a predicted-vs-actual scatter plot to plots/pred_vs_actual.png."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=8, color="#4C72B0", label="Applicants")
    lo, hi = 300, 850
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Credit Score", fontsize=12)
    ax.set_ylabel("Predicted Credit Score", fontsize=12)
    ax.set_title("Predicted vs Actual Credit Score (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = PLOTS_DIR / "pred_vs_actual.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Train the MLP, evaluate it, and save all artefacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    csv_path = DATA_DIR / "credit_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run generate_data.py first.")

    df = pd.read_csv(csv_path)

    # Impute nulls with training-set median
    median_msd = float(df["months_since_last_delinquency"].median())
    df["months_since_last_delinquency"] = df["months_since_last_delinquency"].fillna(median_msd)

    X = df[FEATURE_COLUMNS].values
    y = df["credit_score"].values.astype(np.float32)

    # --- Split ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Scale (fit on train only) ------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # --- Train ---------------------------------------------------------------
    print("Training CreditMLP…")
    model = CreditMLP(n_features=len(FEATURE_COLUMNS))
    losses = train_model(model, X_train_s, y_train, epochs=150)

    # --- Plot loss -----------------------------------------------------------
    plot_loss_curve(losses)

    # --- Evaluate ------------------------------------------------------------
    y_pred, mae, rmse, r2 = evaluate_model(model, X_test_s, y_test)

    print(f"\nTest-set metrics")
    print(f"  MAE  : {mae:.2f} points")
    print(f"  RMSE : {rmse:.2f} points")
    print(f"  R²   : {r2:.4f}")

    # --- Plot predictions ----------------------------------------------------
    plot_pred_vs_actual(y_test, y_pred)

    # --- Save model and scaler -----------------------------------------------
    torch.save(model.state_dict(), MODELS_DIR / "credit_model.pt")
    # Bundle the scaler together with the training median so pipeline.py
    # can impute new applicants with the same value used during training.
    joblib.dump(
        {"scaler": scaler, "median_msd": median_msd},
        MODELS_DIR / "scaler.pkl",
    )

    print(f"\n  Model  saved → {MODELS_DIR / 'credit_model.pt'}")
    print(f"  Scaler saved → {MODELS_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
