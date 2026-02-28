# Quandela Hackathon - Option Pricing with Quantum Machine Learning

## Overview
This repository/folder contains a complete pipeline for predicting the prices of 224 combinations of Tenors and Maturities from historical data, using Quantum Machine Learning techniques. The pipeline processes time-series option data, applies dimensionality reduction, and trains several models to forecast future option distributions and impute missing data.

## Key Technical Decisions
1. **Data Preprocessing**:
   - The dataset contains 494 historical data points (Date) across 224 pricing columns (combinations of `Tenor` and `Maturity`).
   - We normalized the data utilizing `StandardScaler` and applied `PCA` (Principal Component Analysis) to reduce the dimensionality to the top **4 Principal Components**. These components explain ~99.9% of the variance while rendering quantum simulation computationally tractable.
   - We utilized a time-series sliding window approach (`WINDOW_SIZE=5`). This structures the problem such that historical trends dictating option pricing explicitly feed into the models.

2. **Models Implemented**:
   - **Classical Benchmark (MLP)**: A Classical Multi-Layer Perceptron to baseline performance.
   - **Hybrid QNN (Quantum Neural Network)**: Combines classical linear feature reduction (extracting robust representation) into a 4-qubit `qml.AngleEmbedding` followed by `qml.BasicEntanglerLayers`.
   - **VQC / VQR (Variational Quantum Regressor)**: Designed heavily relying on the quantum layer. Uses `tanh` bound inputs explicitly fed into the quantum sequence, capturing non-linear distributions.
   - **QSVR (Quantum Support Vector Regressor)**: Utilizes an explicit, computationally exact Quantum Kernel derived from overlap measurements built via PennyLane. It constructs the similarity mapping directly in Hilbert space!

3. **Predictions & Missing Data**:
   - The Hybrid QNN is wrapped in a generator function `forecast_future` which predicts 30 future steps iteratively over its own historical windows.
   - Predictions are then inversely passed through PCA and the Scaler, returning absolute Option Prices conforming precisely to the original dataset schema (`Future_Forecast_30Days.csv`).

## Output Artifacts (Available in `./results`)
- `*_loss.png`: Validation vs Training MSE loss per epoch showcasing convergence properties outperforming flat predictions.
- `*_preds.png`: Qualitative alignment between True values and Model predictions on unseen Test arrays. 
- `Future_Forecast_30Days.csv`: Auto-regressively generated forecast of the next 30 days for every Tenor/Maturity combination.

## Going Forward (Scaling up via qBraid)
To boost model accuracy and performance before submission:
1. Increase parameter layers: In `train_models.py`, increase `n_layers=3` to `n_layers=8` for the `Hybrid QNN`.
2. Move to qBraid: Use the qBraid SDK's `QbraidProvider` alongside Amazon Braket backend definitions if execution locally becomes too slow. Simply replace `dev = qml.device("default.qubit", wires=n_qubits)` with `dev = qml.device("braket.aws.qubit", device_arn="...", wires=n_qubits)`.
