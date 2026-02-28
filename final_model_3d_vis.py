#!/usr/bin/env python3
"""
Final Model: Quantum Reservoir Computing for Swaption Pricing
=============================================================
Train on historical data, predict future swaption prices.

Pipeline:
  Raw prices → StandardScaler → PCA(5) → sliding window(5) →
  5× Perceval QRC (8m/3ph, UNBUNCHED, LexGrouping(10), fixed) + raw →
  Ridge → PCA inverse → Scaler inverse → prices
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

import perceval as pcvl
from merlin import QuantumLayer, ComputationSpace, LexGrouping

# ── Config ───────────────────────────────────────────────────────────────
N_PCA       = 5
WINDOW      = 5
N_MODES     = 8
N_PHOTONS   = 3
N_ENCODE    = 5
N_LAYERS    = 5
N_RESERVOIRS = 5
RIDGE_ALPHA = 10.0
LEX_OUT     = 10     # LexGrouping output dimension per reservoir

TOTAL_ENC   = N_ENCODE * N_LAYERS  # 25
INPUT_STATE = [1] * N_PHOTONS + [0] * (N_MODES - N_PHOTONS)

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA = BASE / "CHALLENGE RESOURCES" / "DATASETS"


# =====================================================================
# BUILDING BLOCKS
# =====================================================================

def build_circuit(n_modes, n_encode, n_layers, seed):
    """Build one hand-crafted Perceval photonic circuit."""
    circuit = pcvl.Circuit(n_modes)
    c = 0
    for li in range(n_layers):
        circuit.add(0, pcvl.GenericInterferometer(
            n_modes,
            lambda i, _l=li: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_l}_i{i}"))
                              // pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_l}_o{i}"))),
            shape=pcvl.InterferometerShape.RECTANGLE,
        ))
        for m in range(n_encode):
            circuit.add(m, pcvl.PS(pcvl.P(f"input{c}")))
            c += 1
    circuit.add(0, pcvl.GenericInterferometer(
        n_modes,
        lambda i: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_f_i{i}"))
                   // pcvl.BS() // pcvl.PS(pcvl.P(f"t_f_o{i}"))),
        shape=pcvl.InterferometerShape.RECTANGLE,
    ))
    return circuit, c


def build_reservoirs():
    """Build ensemble of fixed-random quantum reservoirs with LexGrouping."""
    reservoirs = []
    for r in range(N_RESERVOIRS):
        torch.manual_seed(42 + r * 1000)
        circ, n_enc = build_circuit(N_MODES, N_ENCODE, N_LAYERS, seed=42 + r * 1000)
        core = QuantumLayer(
            input_size=n_enc, circuit=circ, input_state=INPUT_STATE,
            input_parameters=["input"], trainable_parameters=["t"],
            computation_space=ComputationSpace.UNBUNCHED,
            dtype=torch.float32,
        )
        layer = nn.Sequential(core, LexGrouping(core.output_size, LEX_OUT))
        layer.eval()
        reservoirs.append(layer)
    return reservoirs


def quantum_features(x, reservoirs):
    """Pass a single 25-dim vector through all reservoirs → concatenated output."""
    if len(x) < TOTAL_ENC:
        x = np.pad(x, (0, TOTAL_ENC - len(x)))
    elif len(x) > TOTAL_ENC:
        x = x[:TOTAL_ENC]
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return np.concatenate([r(xt).squeeze(0).numpy() for r in reservoirs])


def make_windows(pc_data):
    """Create (X, y) pairs from PCA time series."""
    X, y = [], []
    for t in range(WINDOW, len(pc_data)):
        X.append(pc_data[t - WINDOW: t].flatten())
        y.append(pc_data[t])
    return np.array(X), np.array(y)


# =====================================================================
# TRAIN
# =====================================================================

def train(prices_train):
    """
    Train the full pipeline on raw price data.

    Args:
        prices_train: np.array of shape (N_days, 224)

    Returns:
        model dict with all fitted components + training diagnostics
    """
    # 1. Fit scaler & PCA
    scaler = StandardScaler().fit(prices_train)
    scaled = scaler.transform(prices_train)
    pca = PCA(n_components=N_PCA).fit(scaled)
    pc = pca.transform(scaled)

    # 2. Build reservoirs
    reservoirs = build_reservoirs()

    # 3. Create windows
    X, y = make_windows(pc)

    # 4. Extract features (quantum + raw)
    Q = np.array([quantum_features(x, reservoirs) for x in X])
    features = np.hstack([Q, X])

    # 5. Fit Ridge
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(features, y)

    # 6. Training predictions (for diagnostics)
    y_train_pred_pca = ridge.predict(features)
    y_train_pred_prices = scaler.inverse_transform(pca.inverse_transform(y_train_pred_pca))
    y_train_true_prices = scaler.inverse_transform(pca.inverse_transform(y))

    return {
        "scaler": scaler,
        "pca": pca,
        "reservoirs": reservoirs,
        "ridge": ridge,
        "last_pc": pc[-WINDOW:],  # last window for inference
        # diagnostics
        "train_true_pca": y,
        "train_pred_pca": y_train_pred_pca,
        "train_true_prices": y_train_true_prices,
        "train_pred_prices": y_train_pred_prices,
        "pca_explained_var": pca.explained_variance_ratio_,
        "ridge_coef": ridge.coef_,
    }


# =====================================================================
# PREDICT
# =====================================================================

def predict(model, n_days):
    """
    Predict n_days into the future autoregressively.

    Args:
        model: dict from train()
        n_days: how many future days to predict

    Returns:
        np.array of shape (n_days, 224) — predicted prices
    """
    buffer = list(model["last_pc"])
    reservoirs = model["reservoirs"]
    ridge = model["ridge"]
    preds_pca = []

    for _ in range(n_days):
        window = np.array(buffer[-WINDOW:]).flatten()
        q = quantum_features(window, reservoirs)
        feat = np.concatenate([q, window]).reshape(1, -1)
        pred = ridge.predict(feat)[0]
        preds_pca.append(pred)
        buffer.append(pred)

    preds_pca = np.array(preds_pca)
    scaled = model["pca"].inverse_transform(preds_pca)
    prices = model["scaler"].inverse_transform(scaled)
    return prices


# =====================================================================
# MAIN
# =====================================================================

def generate_plots(model, actual, predicted, features, train_days, pred_days, out):
    """Generate all diagnostic and inference plots."""

    # ── Plot 1: PCA Variance Explained ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    var = model["pca_explained_var"]
    cumvar = np.cumsum(var)
    x = np.arange(1, len(var) + 1)
    ax.bar(x, var * 100, color="#3498db", alpha=0.7, label="Individual")
    ax.plot(x, cumvar * 100, "o-", color="#e74c3c", linewidth=2, label="Cumulative")
    for i, cv in enumerate(cumvar):
        ax.annotate(f"{cv*100:.1f}%", (x[i], cv*100 + 1), ha="center", fontsize=9)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Variance Explained")
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out / "pca_variance.png", dpi=150)
    plt.close(fig)
    print("  [1/7] PCA variance plot saved")

    # ── Plot 2: Training Fit — PCA space (per component) ─────────────────
    y_true_pca = model["train_true_pca"]
    y_pred_pca = model["train_pred_pca"]
    n_comp = y_true_pca.shape[1]

    fig, axes = plt.subplots(n_comp, 1, figsize=(14, 3 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]
    for c in range(n_comp):
        ax = axes[c]
        ax.plot(y_true_pca[:, c], color="#2c3e50", linewidth=0.8, label="Actual", alpha=0.8)
        ax.plot(y_pred_pca[:, c], color="#e74c3c", linewidth=0.8, label="Predicted", alpha=0.7)
        r2 = r2_score(y_true_pca[:, c], y_pred_pca[:, c])
        ax.set_ylabel(f"PC{c+1}")
        ax.set_title(f"PC{c+1}  (train R² = {r2:.6f})", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Training sample index")
    fig.suptitle("Training Fit in PCA Space", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "training_fit_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/7] Training fit (PCA) plot saved")

    # ── Plot 3: Training Fit — Price space (sample features) ─────────────
    sample_idx = np.linspace(0, len(features) - 1, 6, dtype=int)
    y_true_px = model["train_true_prices"]
    y_pred_px = model["train_pred_prices"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for k, (ax, si) in enumerate(zip(axes.flat, sample_idx)):
        ax.plot(y_true_px[:, si], color="#2c3e50", linewidth=0.8, label="Actual")
        ax.plot(y_pred_px[:, si], color="#e74c3c", linewidth=0.8, alpha=0.7, label="Predicted")
        r2 = r2_score(y_true_px[:, si], y_pred_px[:, si])
        ax.set_title(f"{features[si]}  (R²={r2:.4f})", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Training Fit — Sample Swaption Prices", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "training_fit_prices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [3/7] Training fit (prices) plot saved")

    # ── Plot 4: Training Residual Distribution ───────────────────────────
    residuals = (y_true_px - y_pred_px).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(residuals, bins=80, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (actual - predicted)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Training Residuals  (mean={np.mean(residuals):.2e}, std={np.std(residuals):.2e})")
    axes[1].hist(np.abs(residuals), bins=80, color="#e67e22", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("|Residual|")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Absolute Residuals  (median={np.median(np.abs(residuals)):.2e})")
    fig.tight_layout()
    fig.savefig(out / "training_residuals.png", dpi=150)
    plt.close(fig)
    print("  [4/7] Training residuals plot saved")

    # ── Plot 5: Inference — Predicted vs Actual Scatter ──────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual.flatten(), predicted.flatten(), s=3, alpha=0.4, color="#3498db")
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    r2 = r2_score(actual.flatten(), predicted.flatten())
    mse = mean_squared_error(actual.flatten(), predicted.flatten())
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"Inference: Predicted vs Actual (Days {train_days+1}-{train_days+pred_days})\n"
                 f"R² = {r2:.6f}   MSE = {mse:.2e}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "inference_scatter.png", dpi=150)
    plt.close(fig)
    print("  [5/7] Inference scatter plot saved")

    # ── Plot 6: Inference — Error Heatmap ────────────────────────────────
    errors = predicted - actual  # (PRED_DAYS, 224)
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(errors, aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(errors)), vmax=np.max(np.abs(errors)))
    ax.set_yticks(range(pred_days))
    ax.set_yticklabels([f"Day {train_days+d+1}" for d in range(pred_days)])
    ax.set_xlabel("Feature index (224 swaption columns)")
    ax.set_title("Prediction Error Heatmap (Predicted − Actual)")
    plt.colorbar(im, ax=ax, label="Error", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "inference_error_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [6/7] Inference error heatmap saved")

    # ── Plot 7: Inference — Per-day MSE bar chart ────────────────────────
    day_mse = [mean_squared_error(actual[d], predicted[d]) for d in range(pred_days)]
    day_mae = [mean_absolute_error(actual[d], predicted[d]) for d in range(pred_days)]
    day_labels = [f"Day {train_days+d+1}" for d in range(pred_days)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, pred_days))
    axes[0].bar(day_labels, day_mse, color=colors, edgecolor="white")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Per-Day MSE")
    for i, v in enumerate(day_mse):
        axes[0].text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(day_labels, day_mae, color=colors, edgecolor="white")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Per-Day MAE")
    for i, v in enumerate(day_mae):
        axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Autoregressive Inference Error (Days {train_days+1}-{train_days+pred_days})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "inference_per_day_error.png", dpi=150)
    plt.close(fig)
    print("  [7/7] Per-day error bar chart saved")


if __name__ == "__main__":
    # Load data
    df = pd.read_excel(DATA / "train.xlsx")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    features = [c for c in df.columns if c != "Date"]
    prices = df[features].values.astype(np.float64)

    # --- Config ---
    TRAIN_DAYS = 450
    PRED_DAYS  = 6

    print(f"Training on days 1-{TRAIN_DAYS}...")
    model = train(prices[:TRAIN_DAYS])
    print("Done.\n")

    print(f"Predicting days {TRAIN_DAYS+1}-{TRAIN_DAYS+PRED_DAYS}...")
    predicted = predict(model, PRED_DAYS)
    actual = prices[TRAIN_DAYS:TRAIN_DAYS+PRED_DAYS]

    # Metrics table
    print(f"\n{'Day':<6} {'MSE':>14} {'MAE':>10} {'RelErr%':>10}")
    print("─" * 42)
    for d in range(PRED_DAYS):
        mse = mean_squared_error(actual[d], predicted[d])
        mae = mean_absolute_error(actual[d], predicted[d])
        rel = 100 * np.mean(np.abs(actual[d] - predicted[d]) / (np.abs(actual[d]) + 1e-10))
        print(f"{TRAIN_DAYS+d+1:<6} {mse:>14.2e} {mae:>10.6f} {rel:>9.4f}%")

    overall_mse = mean_squared_error(actual.flatten(), predicted.flatten())
    overall_r2 = r2_score(actual.flatten(), predicted.flatten())
    print(f"\nOverall MSE: {overall_mse:.2e}")
    print(f"Overall R² : {overall_r2:.6f}")

    # Save CSVs
    out = BASE / "quantum_results" / "final"
    out.mkdir(parents=True, exist_ok=True)

    for label, data in [("actual", actual), ("predicted", predicted)]:
        pd.DataFrame(data, columns=features,
                     index=[f"Day {TRAIN_DAYS+d+1}" for d in range(PRED_DAYS)]
                     ).to_csv(out / f"{label}.csv")

    # Generate all plots
    print("\nGenerating plots...")
    generate_plots(model, actual, predicted, features, TRAIN_DAYS, PRED_DAYS, out)

    # Export Data for Website 3D Plotly Interaction
    web_out = BASE / "qedi_website" / "assets" / "results_data.json"
    
    # Parse tenors and maturities from feature names for 3D axis
    tenors = []
    maturities = []
    
    # "Tenor : 0.5; Maturity : 1.0"
    for f in features:
        parts = f.split(";")
        t_str = parts[0].replace("Tenor :", "").strip()
        m_str = parts[1].replace("Maturity :", "").strip()
        tenors.append(float(t_str))
        maturities.append(float(m_str))
        
    export_dict = {
        "features": features,
        "tenors": tenors,
        "maturities": maturities,
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
        "pred_days": PRED_DAYS,
        "train_days": TRAIN_DAYS
    }
    
    with open(web_out, "w") as f:
        json.dump(export_dict, f)
        
    # Export PCA 1 Time Series for Web Animation
    anim_out = BASE / "qedi_website" / "assets" / "pca_anim_data.json"
    pc_slice_len = 200 # Animate the last 200 days of training
    anim_dict = {
        "true_pc1": model["train_true_pca"][-pc_slice_len:, 0].tolist(),
        "pred_pc1": model["train_pred_pca"][-pc_slice_len:, 0].tolist()
    }
    with open(anim_out, "w") as f:
        json.dump(anim_dict, f)

    print(f"\nAll saved to {out}/, {web_out}, and {anim_out}")
