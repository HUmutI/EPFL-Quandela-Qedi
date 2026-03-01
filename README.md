<div align="center">
  <h1>Team Qedi 🐈‍⬛</h1>
  <p><strong>EPFL Quantum Hackathon 2026 • Quandela Challenge</strong></p>
  <h3>Hybrid Photonic Temporal QRC (HPT-QRC) — Swaption Volatility Surface Forecaster</h3>
  <p><i>Teaching photons to predict the market so we can finally sleep.</i></p>
</div>

<br />

## 📖 1. The Challenge

A **swaption volatility surface** is a 2D grid indexed by tenor and maturity. Providing 224 high-dimensional features per day, the challenge demands two primary tasks:
1. **Predictive Forecasting** — Forecasting the complex non-linear surface shifts for the next $H$ days smoothly.
2. **Data Imputation Reconstruction** — Real markets suffer from outages, resulting in `NaN` entries across the option grid. The model must analytically infer missing surface areas by relying on deeply correlated temporal topologies.

**The Quantum Solution:** We compress the manifold via PCA and process the temporal dynamics using a entirely newly adapted **Hybrid Photonic Temporal Quantum Reservoir Computer (HPT-QRC)** architecture. Inspired by *Li et al. (2024)*, we took their qubit-based framework and adapted it to a purely Photonic Quantum Reservoir powered by **[MerLin by Quandela](https://merlinquantum.ai/)**.

---

## ⚙️ 2. Hybrid Photonic Temporal QRC Architecture

Instead of a standard QRC, we implemented a state-of-the-art **HPT-QRC pipeline**:

1. **Preprocessing**: Raw 224D Market Surface $\rightarrow$ `StandardScaler` + `PCA (5D)` $\rightarrow$ Rolling 5-Day Window (1×25)
2. **Dedicated Memory Modes**: Instead of mapping data to all spatial modes simultaneously, our temporal array uses **5 input modes** (phase encoded) and **3 dedicated memory modes** (unencoded loop). The memory modes continuously accumulate historical state contexts across 5 time steps through serial phase mixing.
3. **Virtual Nodes**: We sample the evolving physical system at multiple structural post-processing depths (Depths 1, 2, 3). These **Virtual Nodes** emulate capturing chronological measurement sub-intervals, massively expanding our temporal feature dimensionality without adding physical photon bounds.
4. **Ensemble LexGrouping Compression**: Instead of measuring impossibly vast raw Fock states, we group the probability vectors via **LexGrouping** across 3 random seeds × 3 virtual depths (= 9 Circuits). 
5. **Direct Target Ridge Forecaster**: By augmenting 90 Quantum Features with the 25 Classic Features, extracting the most prominent Non-linear Mutual Information quantum channels, an $L2$ regularized Ridge Regression projection ($\alpha=10.0$) predicts the consecutive future states.

---

## 🚀 3. CLI Reference & Setup

Requires `torch`, `pcvl`, and standard ML arrays (`numpy`, `pandas`, `scikit-learn`).

```bash
# First create the environment and install dependencies
conda create -n quandela python=3.10
conda activate quandela
pip install -r requirements.txt
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

## 📈 4. Final Benchmark Results

By executing the novel Hybrid Photonic Temporal QRC pipeline, we successfully beat our underlying standard QRC framework and left classical baselines absolutely obsolete.

| Model | RMSE Error |
| :--- | :---: |
| **QSVR** | `0.0233` | 
| **Hybrid QNN** | `0.0083` | 
| **LSTM** | `0.0073` | 
| **Photonic Linear QRC** | `0.0065` | 
| **Hybrid Photonic Linear QRC** | `0.0028` | 
| **🥇 Hybrid Photonic Temporal QRC (HPT-QRC)** | **`0.0027`** | 

*Maintains incredibly reliable sub-10% projection accuracy out to a 6-Day cascading prediction envelope.*

---

## 👥 Meet Team Qedi

Proudly built during the 24-hour **EPFL Quantum Hackathon 2026**.

* [**Eren Aslan**](https://www.linkedin.com/in/eren-aslan-421b66191/)   
* [**Hüseyin Umut Işık**](https://www.linkedin.com/in/h%C3%BCseyin-umut-i%C5%9F%C4%B1k-7b3ba4255/)      
* [**Arda Kara**](https://www.linkedin.com/in/arda-kara0/) 
* [**Mehmet Alp Özaydın**](https://www.linkedin.com/in/mehmet-alp-%C3%B6zayd%C4%B1n-8455bb246/) 

<br/>
<div align="center">
  <table>
    <tr align="center" valign="middle">
      <td width="300" style="border: none; background: transparent;">
        <a href="https://www.metu.edu.tr/tr" target="_blank"><img src="qedi_website/assets/metu_logo.png" alt="METU Logo" width="120"/></a>
      </td>
      <td width="300" style="border: none; background: transparent;">
        <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank"><img src="qedi_website/assets/bilkent_logo.png" alt="Bilkent Logo" width="120"/></a>
      </td>
    </tr>
    <tr align="center" valign="top">
      <td width="300" style="border: none; background: transparent;">
        <i>Middle East Technical University</i>
      </td>
      <td width="300" style="border: none; background: transparent;">
        <i>Bilkent University</i>
      </td>
    </tr>
  </table>
</div>
