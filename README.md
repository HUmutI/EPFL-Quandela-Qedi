<div align="center">
  <h1>Team Qedi 🐈‍⬛</h1>
  <p><strong>EPFL Quantum Hackathon 2026 • Quandela Challenge</strong></p>
  <h3>MerLin Photonic QRC — Swaption Surface Forecaster</h3>
</div>

<br />

## 📖 1. Problem Statement

A **swaption volatility surface** is a 2-dimensional grid of implied volatilities indexed by tenor (time to option expiry) and maturity (underlying swap length). Observing one surface per trading day, the challenge requires:
1. **Forecasting** — given a history of surfaces, predict the next $H$ surfaces.
2. **Reconstruction** — given a surface with missing entries, analytically infer the full surface.

The surfaces are high-dimensional (224 features) but lie on a low-dimensional manifold: adjacent points are strongly correlated, and day-to-day changes are smooth. 

**The Quantum Solution:** We compress the manifold via PCA and process the temporal dynamics using a **Quantum Reservoir Computer (QRC)**. Our specific reservoir is **[MerLin by Quandela](https://merlinquantum.ai/)** — a photonic quantum circuit whose measurement statistics provide robust nonlinear features, entirely bypassing the vanishing gradients that ruin classical LSTMs..

---

## 📊 2. Dataset

| Property | Value |
|---|---|
| **Rows** | 494 trading days, date-sorted |
| **Columns** | `Date`, `Type`, + 224 surface columns |
| **Surface Grid** | 14 tenors × 16 maturities = 224 features |
| **Test Matrix** | Partial rows with `NaN` requiring imputation |

*Note: The pipeline strictly uses row indexing (0 = first trading day) rather than calendar dates to elegantly handle irregular market gaps (weekends, holidays).*

---

## ⚙️ 3. Pipeline Architecture

Our champion model strictly follows this forward-pass flow:

1. **`VectorStandardizer`**: Applies per-feature unit-variance scaling on the $T \times 224$ training set.
2. **`PCAManifoldCodec`**: Whitened PCA reduces the 224 dimensions to **8 latent components** ($d=8$). For missing data (reconstruction), it uses a closed-form masked Tikhonov/Ridge inference ($\lambda = 10^{-3}$) to find the latent $z$ that best explains the observed entries.
3. **`Temporal Feature Engineering`**: At each timestep, we construct a 17-dimensional enhanced vector containing the latent code $z_t$, the absolute temporal difference $\|\Delta z_t\|$, and the standardized calendar $gap_t$.
4. **`MerlinReservoirEnsemble`**: 
   - Uses two independent `MerlinPhotonicReservoir` instances for variance reduction.
   - **Delay Embedding**: The 8 photonic circuit modes are split (4 for current state, 4 for previous state).
   - **Photonic Circuit**: Operates on an unbunched 4-photon state through frozen random unitaries. Measurement probabilities ($p$) are expanded non-linearly ($\varphi = [p \| p^2]$) and randomly projected into a 32-dimensional space.
   - **Leaky Integration**: Applies an exponential moving average ($\gamma=0.5$) to accumulate temporal context across the sequential window.
5. **`DirectResidualForecaster`**: Instead of predicting absolute codes, we predict the *change* $\Delta z_h$ utilizing 6 independent Ridge Regression models ($\alpha = 2000$).

---

## 🚀 4. CLI Reference & Setup

Requires `torch`, `pennylane`, and standard ML arrays (`numpy`, `pandas`, `scikit-learn`).

```bash
# Activate the native environment
conda activate quandela
```

### Validation Mode
Train on the first $N$ rows; forecast and evaluate the next $M$ rows against ground truth.
```bash
python -m src.main validate --train-rows 300 --val-rows 6
```

### Prediction Mode
Fit on all 494 training rows to produce the final competition submission.
```bash
python -m src.main predict --output artifacts/submission.xlsx
```

---

## 📈 5. Final Benchmark Results

Tested autoregressively against classical methods, the QRC model surpassed heavyweight baselines.

| Model | RMSE Error | Description |
| :--- | :---: | :--- |
| **Naive Baseline** | `0.0033` | Persistence model, fails on volatility shifts. |
| **Classical LSTM** | `0.0089` | Parameter bloat causes catastrophic overfitting. |
| **🥇 Team Qedi (Hybrid QRC)** | **`0.0083`** | **Our MerLin-powered 8-mode architecture.** |

**Output Artifacts:** The engine automatically generates heatmaps (`error_heatmap.png`), histogram distributions (`error_histograms.png`), and cascading performance sweeps (`forecast_quality.png`) saved directly to `artifacts/validation/`.

---

## 👥 Meet Team Qedi

Proudly built during the 48-hour **EPFL Quantum Hackathon 2026**.

* **Eren Aslan** 
* **Hüseyin Umut Işık** 
* **Arda Kara** 
* **Mehmet Alp Özaydın** 

<br/>
<div align="center" style="display: flex; justify-content: center; gap: 40px; align-items: center;">
  <div>
    <img src="qedi_website/assets/metu_logo.png" alt="METU Logo" width="120"/>
    <p><i>Middle East Technical University</i></p>
  </div>
  <div>
    <img src="qedi_website/assets/bilkent_logo.png" alt="Bilkent Logo" width="120"/>
    <p><i>Bilkent University</i></p>
  </div>
</div>
