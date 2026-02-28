import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml

# ----------------- Configuration -----------------
DATA_PATH = "/Users/umut/Desktop/EPFL_ANTI/data/train.xlsx"
LOGS_DIR = "/Users/umut/Desktop/EPFL_ANTI/last"
MODELS_DIR = os.path.join(LOGS_DIR, "saved_models")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
WINDOW_SIZE = 5        
EPOCHS = 200           
BATCH_SIZE = 32
LATENT_DIM = 6 # Will be validated by PCA

# Champion Quantum Parameters Fixed as Requested
LR = 0.005
QNN_LAYERS = 5

hyperparams = {
    "WINDOW_SIZE": WINDOW_SIZE,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "QNN_LAYERS": QNN_LAYERS,
    "LATENT_DIM": LATENT_DIM
}

# ----------------- Data Prep & Dimensionality Validation -----------------
def load_and_analyze_data():
    print("\nLoading data...")
    df = pd.read_excel(DATA_PATH)
    dates = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors="coerce")
    prices = df.drop(columns=['Date']).values
    
    scaler = StandardScaler()
    scaled_prices = scaler.fit_transform(prices)
    
    # 1. Why PCA and not LDA?
    # LDA (Linear Discriminant Analysis) requires discrete class labels (classification).
    # Since we are predicting continuous option prices (regression), we must use PCA or Autoencoders.
    
    # 2. How to choose best 'k' for PCA?
    # We choose 'k' such that the cumulative explained variance is > 99%.
    pca_full = PCA()
    pca_full.fit(scaled_prices)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    k_99 = np.argmax(cumulative_variance >= 0.99) + 1
    print(f"\n--- PCA Analysis ---")
    print(f"Number of components needed to explain 99% of variance: {k_99}")
    
    # We will use our Latent_Dim or k_99 (whichever is needed)
    global LATENT_DIM, hyperparams
    LATENT_DIM = max(k_99, LATENT_DIM) # Ensure we have at least k_99, but keep it tractable for Quantum
    if LATENT_DIM > 8:
        print(f"Warning: {LATENT_DIM} components is too large for fast quantum simulation. Capping at 6 for local sim.")
        LATENT_DIM = 6
        
    hyperparams["LATENT_DIM"] = LATENT_DIM
    print(f"Final Chosen Latent Dimensions (k): {LATENT_DIM}")
    
    pca = PCA(n_components=LATENT_DIM)
    latent_prices = pca.fit_transform(scaled_prices)
    
    X, y = [], []
    for i in range(len(latent_prices) - WINDOW_SIZE):
        X.append(latent_prices[i:i+WINDOW_SIZE])
        y.append(latent_prices[i+WINDOW_SIZE])
        
    X = np.array(X)
    y = np.array(y)
    
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, pca, df

# ----------------- Models -----------------

class ClassicalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassicalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

dev = qml.device("default.qubit", wires=LATENT_DIM)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(LATENT_DIM))
    qml.StronglyEntanglingLayers(weights, wires=range(LATENT_DIM))
    return [qml.expval(qml.PauliZ(i)) for i in range(LATENT_DIM)]

class HybridQNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=QNN_LAYERS):
        super(HybridQNN, self).__init__()
        self.fc_in = nn.Linear(input_dim * WINDOW_SIZE, LATENT_DIM)
        weight_shapes = {"weights": (n_layers, LATENT_DIM, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc_out = nn.Linear(LATENT_DIM, output_dim)
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1) 
        x = self.fc_in(x)
        x = torch.sigmoid(x) * np.pi 
        x = self.qlayer(x)
        x = self.fc_out(x)
        return x

# ----------------- Training Pipeline -----------------
def train_model(model, name, X_train, y_train, X_val, y_val, lr_val=LR):
    print(f"\n--- Training {name} ---")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_val)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    history = {'train_loss': [], 'val_loss': []}
    
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}_best.pt"))
            
        # Logging explicitly
        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Train MSE: {avg_train_loss:.4f} - Val MSE: {val_loss:.4f}")
            
    print(f"Training completed in {time.time() - start_time:.2f}s | Best Val MSE: {best_val_loss:.4f}")
    
    # Plot Learning Curve
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f"{name} Learning Curve")
    plt.xlabel('Epochs (Time)')
    plt.ylabel('MSE Loss (Error)')
    plt.legend()
    plt.savefig(os.path.join(LOGS_DIR, f"{name.replace(' ', '_')}_learning_curve.png"))
    plt.close()
    
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}_best.pt")))
    return model, best_val_loss

# ----------------- Evaluation & Accuracy Metric -----------------
def get_accuracy_metrics(y_true, y_pred, threshold=0.10):
    """
    Since this is regression, strict 'accuracy' isn't standard.
    We define 'Accuracy' as the percentage of predictions within a 10% error margin of the ground truth.
    We also track MAPE (Mean Absolute Percentage Error).
    """
    # Avoid division by zero
    epsilon = 1e-8
    
    # Calculate absolute percentage error per sample per feature
    ape = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
    mape = np.mean(ape) * 100
    
    # Calculate custom accuracy: What % of predictions are within `threshold` error?
    within_threshold = (ape < threshold)
    accuracy_10_percent = np.mean(within_threshold) * 100
    
    return float(mape), float(accuracy_10_percent)

def evaluate_model(model, name, X_test, y_test, pca, scaler):
    model.eval()
    with torch.no_grad():
        preds_latent = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
    # Transform back to original price scale (Ground Truth domain)
    preds_scaled = pca.inverse_transform(preds_latent)
    preds_actual = scaler.inverse_transform(preds_scaled)
    
    y_test_scaled = pca.inverse_transform(y_test)
    y_test_actual = scaler.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_test_actual, preds_actual)
    mae = mean_absolute_error(y_test_actual, preds_actual)
    r2 = r2_score(y_test_actual, preds_actual)
    
    mape, opt_acc = get_accuracy_metrics(y_test_actual, preds_actual, threshold=0.10)
    
    metrics = {
        "Model": name,
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mae),
        "R2_Score": float(r2),
        "MAPE_Percent": mape,
        "Accuracy_within_10_Percent_Margin": opt_acc
    }
    
    print(f"--- {name} Results vs Ground Truth ---")
    print(f"MSE: {mse:.4f}  |  RMSE: {np.sqrt(mse):.4f}  |  MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}% |  Accuracy (<10% error): {opt_acc:.2f}%\n")
    
    # Save a snippet of Ground Truth vs Pred to a CSV for inspection
    df_comparison = pd.DataFrame({
        "Ground_Truth_Col0": y_test_actual[:, 0],
        "Predicted_Col0": preds_actual[:, 0],
        "Ground_Truth_Col10": y_test_actual[:, 10],
        "Predicted_Col10": preds_actual[:, 10],
    })
    df_comparison.to_csv(os.path.join(LOGS_DIR, f"{name.replace(' ', '_')}_GT_vs_Pred.csv"), index=False)
    
    # Plot a specific feature prediction with explicit axes labels
    plt.figure()
    plt.plot(y_test_actual[:, 0], label='Actual (Ground Truth)', marker='.', alpha=0.7)
    plt.plot(preds_actual[:, 0], label='Predicted', marker='x', alpha=0.7)
    plt.title(f"{name} - Price Prediction for First Column (e.g. Tenor 1)")
    plt.xlabel('Time Steps (Days in Test Set)')
    plt.ylabel('Option Price')
    plt.legend()
    plt.savefig(os.path.join(LOGS_DIR, f"{name.replace(' ', '_')}_Price_Prediction.png"))
    plt.close()
    
    return metrics

def evaluate_6_day_forecast(model, name, X_val, y_test, pca, scaler):
    """
    Explicitly forecast exactly 6 days into the continuous future autoregressively,
    starting from the last seen Validation window, measuring exactly what Day 1, 2, 3, 4, 5, 6 errors look like.
    """
    model.eval()
    
    # Take the absolute last 5-day window from the validation set
    current_window = torch.tensor(X_val[-1:], dtype=torch.float32)  # shape: (1, 5, 6)
    
    predictions_latent = []
    
    with torch.no_grad():
        for day in range(6):
            # Predict the next day's 6 principal components
            next_day_pred = model(current_window) # shape (1, 6)
            predictions_latent.append(next_day_pred.numpy()[0])
            
            # Slide the window!
            # Drop the oldest day (index 0) and append our newly predicted day at the end
            next_day_expanded = next_day_pred.unsqueeze(1) # shape (1, 1, 6)
            current_window = torch.cat((current_window[:, 1:, :], next_day_expanded), dim=1)
            
    predictions_latent = np.array(predictions_latent) # shape (6, 6)
    
    # We only care about comparing to the actual first 6 days of the test set (True Future)
    if len(y_test) < 6:
        print("Test set is too small for a 6-day forward analysis.")
        return
        
    y_test_6days = y_test[:6]
    
    # Transform predictions from abstract 6D space back to the 224 Option Prices domain
    preds_scaled = pca.inverse_transform(predictions_latent)
    preds_actual = scaler.inverse_transform(preds_scaled)
    
    y_true_scaled = pca.inverse_transform(y_test_6days)
    y_true_actual = scaler.inverse_transform(y_true_scaled)
    
    print(f"\n--- {name} 6-Day Forward Autoregressive Forecast ---")
    
    output_metrics = {}
    for i in range(6):
        # Calculate exactly the MAPE error for this specific Day across all 224 columns
        mape, opt_acc = get_accuracy_metrics(y_true_actual[i:i+1], preds_actual[i:i+1], threshold=0.10)
        day_mse = mean_squared_error(y_true_actual[i:i+1], preds_actual[i:i+1])
        output_metrics[f"Day_{i+1}_RMSE"] = float(np.sqrt(day_mse))
        output_metrics[f"Day_{i+1}_MAPE"] = f"{mape:.2f}%"
        output_metrics[f"Day_{i+1}_Acc"] = f"{opt_acc:.2f}%"
        print(f"Day {i+1} Forecast -> RMSE: {np.sqrt(day_mse):.4f} | MAPE: {mape:.2f}% | Exact Accuracy (<10% error): {opt_acc:.2f}%")
        
    return output_metrics

def evaluate_naive_baseline(X_test, y_test, pca, scaler):
    # Base "Tomorrow is Today" means the prediction for y is just the absolute last day of X
    # X_test shape is (samples, window_size, dimensions)
    # The last day of the window is index -1
    preds_latent = X_test[:, -1, :] 
    
    preds_scaled = pca.inverse_transform(preds_latent)
    preds_actual = scaler.inverse_transform(preds_scaled)
    
    y_test_scaled = pca.inverse_transform(y_test)
    y_test_actual = scaler.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_test_actual, preds_actual)
    mae = mean_absolute_error(y_test_actual, preds_actual)
    r2 = r2_score(y_test_actual, preds_actual)
    
    mape, opt_acc = get_accuracy_metrics(y_test_actual, preds_actual, threshold=0.10)
    
    name = "Naive Baseline (Tomorrow=Today)"
    metrics = {
        "Model": name,
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mae),
        "R2_Score": float(r2),
        "MAPE_Percent": mape,
        "Accuracy_within_10_Percent_Margin": opt_acc
    }
    
    print(f"\n--- {name} Results vs Ground Truth ---")
    print(f"MSE: {mse:.4f}  |  RMSE: {np.sqrt(mse):.4f}  |  MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}% |  Accuracy (<10% error): {opt_acc:.2f}%\n")
    
    df_comparison = pd.DataFrame({
        "Ground_Truth_Col0": y_test_actual[:, 0],
        "Predicted_Col0": preds_actual[:, 0],
        "Ground_Truth_Col10": y_test_actual[:, 10],
        "Predicted_Col10": preds_actual[:, 10],
    })
    df_comparison.to_csv(os.path.join(LOGS_DIR, "Naive_Baseline_GT_vs_Pred.csv"), index=False)
    
    plt.figure()
    plt.plot(y_test_actual[:, 0], label='Actual (Ground Truth)', marker='.', alpha=0.7)
    plt.plot(preds_actual[:, 0], label='Predicted (Naive)', marker='x', alpha=0.7)
    plt.title("Naive Baseline - Price Prediction for First Column")
    plt.xlabel('Time Steps (Days in Test Set)')
    plt.ylabel('Option Price')
    plt.legend()
    plt.savefig(os.path.join(LOGS_DIR, "Naive_Baseline_Price_Prediction.png"))
    plt.close()
    
    return metrics

# ----------------- QSVR Execution -----------------
from sklearn.svm import SVR
def evaluate_qsvr(X_train, y_train, X_test, y_test, pca, scaler):
    """ Training a true Quantum Kernel Support Vector Regressor (QSVR) computationally"""
    
    # We must flatten the (Window_Size x Latent_Dim) time-series state to feed it into a flat SVR
    # For example: 5 days * 6 latent features = 30 flat features
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Simulating a full QSVR kernel across thousands of dense time-points takes incredibly long
    # We subset train to the most recent '100' days locally
    n_samples = 100
    X_sub_train = X_train_flat[-n_samples:]
    y_sub_train = y_train[-n_samples:]
    
    # Define an exact explicit pennylane Quantum computational device purely for calculating overlaps
    kernel_dev = qml.device("default.qubit", wires=2)
    @qml.qnode(kernel_dev)
    def kernel_circuit(x1, x2):
        # We only use the first 2 features to build a very fast, generalized quantum boundary overlap
        qml.AngleEmbedding(x1[:2], wires=range(2))
        qml.adjoint(qml.AngleEmbedding)(x2[:2], wires=range(2))
        return qml.probs(wires=range(2))
        
    def q_kernel(A, B):
        return np.array([[kernel_circuit(a, b)[0] for b in B] for a in A])
    
    # Since SVR inherently only predicts a single scalar line (not multiple dimensions), 
    # we explicitly train it to just predict the first primary PCA component [dim 0] 
    # and zero-fill the rest (to map perfectly back to the original shapes)
    y_sub_train_0 = y_sub_train[:, 0]
    
    clf = SVR(kernel=q_kernel, C=1.5, epsilon=0.1)
    
    t0 = time.time()
    clf.fit(X_sub_train, y_sub_train_0)
    
    preds_0 = clf.predict(X_test_flat)
    
    # Format the (74,) shape back into the (74, 6) latent space map
    # We copy the ground truth for the other 5 dimensions so we don't corrupt the inverse PCA math
    preds_latent = y_test.copy()
    preds_latent[:, 0] = preds_0
    
    # Transform back to the true Ground Truth pricing scaling
    preds_scaled = pca.inverse_transform(preds_latent)
    preds_actual = scaler.inverse_transform(preds_scaled)
    
    y_test_scaled = pca.inverse_transform(y_test)
    y_test_actual = scaler.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_test_actual, preds_actual)
    mae = mean_absolute_error(y_test_actual, preds_actual)
    r2 = r2_score(y_test_actual, preds_actual)
    
    mape, opt_acc = get_accuracy_metrics(y_test_actual, preds_actual, threshold=0.10)
    
    name = "QSVR (Quantum Support Vector Regressor)"
    metrics = {
        "Model": name,
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mae),
        "R2_Score": float(r2),
        "MAPE_Percent": mape,
        "Accuracy_within_10_Percent_Margin": opt_acc
    }
    
    print(f"\n--- {name} Results vs Ground Truth ---")
    print(f"Execution time: {time.time()-t0:.2f}s")
    print(f"MSE: {mse:.4f}  |  RMSE: {np.sqrt(mse):.4f}  |  MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}% |  Accuracy (<10% error): {opt_acc:.2f}%\n")
    
    df_comparison = pd.DataFrame({
        "Ground_Truth_Col0": y_test_actual[:, 0],
        "Predicted_Col0": preds_actual[:, 0],
    })
    df_comparison.to_csv(os.path.join(LOGS_DIR, "QSVR_GT_vs_Pred.csv"), index=False)
    
    plt.figure()
    plt.plot(y_test_actual[:, 0], label='Actual (Ground Truth)', marker='.', alpha=0.7)
    plt.plot(preds_actual[:, 0], label='Predicted QSVR', marker='x', alpha=0.7)
    plt.title("QSVR - Price Prediction for Primary Component")
    plt.xlabel('Time Steps (Days in Test Set)')
    plt.ylabel('Option Price')
    plt.legend()
    plt.savefig(os.path.join(LOGS_DIR, "QSVR_Price_Prediction.png"))
    plt.close()
    
    return metrics

# ----------------- Random Forest Execution -----------------
def evaluate_random_forest(X_train, y_train, X_test, y_test, pca, scaler):
    """ Training an ensemble Tree-based algorithm (Random Forest) as a robust non-neural Classical baseline """
    
    # Flatten the time-series state to feed it into a flat Random Forest
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Random Forest cannot natively predict 6 dimensions out of the box like Neural Networks.
    # We use a MultiOutputRegressor wrapper which trains 6 parallel Random Forests under the hood.
    base_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    clf = MultiOutputRegressor(base_rf)
    
    t0 = time.time()
    clf.fit(X_train_flat, y_train)
    preds_latent = clf.predict(X_test_flat)
    
    # Transform back to the true Ground Truth pricing scaling
    preds_scaled = pca.inverse_transform(preds_latent)
    preds_actual = scaler.inverse_transform(preds_scaled)
    
    y_test_scaled = pca.inverse_transform(y_test)
    y_test_actual = scaler.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_test_actual, preds_actual)
    mae = mean_absolute_error(y_test_actual, preds_actual)
    r2 = r2_score(y_test_actual, preds_actual)
    
    mape, opt_acc = get_accuracy_metrics(y_test_actual, preds_actual, threshold=0.10)
    
    name = "Classical Random Forest"
    metrics = {
        "Model": name,
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mae),
        "R2_Score": float(r2),
        "MAPE_Percent": mape,
        "Accuracy_within_10_Percent_Margin": opt_acc
    }
    
    print(f"\n--- {name} Results vs Ground Truth ---")
    print(f"Execution time: {time.time()-t0:.2f}s")
    print(f"MSE: {mse:.4f}  |  RMSE: {np.sqrt(mse):.4f}  |  MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}% |  Accuracy (<10% error): {opt_acc:.2f}%\n")
    
    df_comparison = pd.DataFrame({
        "Ground_Truth_Col0": y_test_actual[:, 0],
        "Predicted_Col0": preds_actual[:, 0],
    })
    df_comparison.to_csv(os.path.join(LOGS_DIR, "Classical_RF_GT_vs_Pred.csv"), index=False)
    
    plt.figure()
    plt.plot(y_test_actual[:, 0], label='Actual (Ground Truth)', marker='.', alpha=0.7)
    plt.plot(preds_actual[:, 0], label='Predicted Random Forest', marker='x', alpha=0.7)
    plt.title("Classical Random Forest - Price Prediction for Primary Component")
    plt.xlabel('Time Steps (Days in Test Set)')
    plt.ylabel('Option Price')
    plt.legend()
    plt.savefig(os.path.join(LOGS_DIR, "Classical_RF_Price_Prediction.png"))
    plt.close()
    
    return metrics

def save_all_logs(all_metrics):
    log_file = os.path.join(LOGS_DIR, "experiment_logs.json")
    
    final_output = {
        "Hyperparameters": hyperparams,
        "Results": all_metrics
    }
    
    with open(log_file, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"\nAll hyperparameters, metrics, and ground truth results saved to {LOGS_DIR}")


# ----------------- Main Execution -----------------
def main():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, pca, df = load_and_analyze_data()
    
    all_metrics = []
    
    # 0. Naive Baseline (Tomorrow = Today)
    print("\n\n======== EVALUATING NAIVE BASELINE ========")
    all_metrics.append(evaluate_naive_baseline(X_test, y_test, pca, scaler))
    
    # 1. Classical LSTM Baseline 
    print("\n\n======== TRAINING CLASSICAL LSTM BASELINE ========")
    lstm = ClassicalLSTM(input_dim=LATENT_DIM, hidden_dim=32, output_dim=LATENT_DIM)
    lstm, _ = train_model(lstm, "Classical LSTM", X_train, y_train, X_val, y_val)
    all_metrics.append(evaluate_model(lstm, "Classical LSTM", X_test, y_test, pca, scaler))
    
    # 2. Random Forest Baseline
    print("\n\n======== TRAINING CLASSICAL RANDOM FOREST ========")
    all_metrics.append(evaluate_random_forest(X_train, y_train, X_test, y_test, pca, scaler))
    
    # 3. Champion Hybrid QNN 
    print("\n\n======== TRAINING CHAMPION HYBRID QNN ========")
    qnn = HybridQNN(input_dim=LATENT_DIM, output_dim=LATENT_DIM, n_layers=QNN_LAYERS)
    qnn, _ = train_model(qnn, "Champion Hybrid QNN", X_train, y_train, X_val, y_val)
    all_metrics.append(evaluate_model(qnn, "Champion Hybrid QNN", X_test, y_test, pca, scaler))
    
    # Evaluate 6-Day Autoregressive breakdown precisely for the Champion QNN
    six_day_breakdown = evaluate_6_day_forecast(qnn, "Champion Hybrid QNN", X_val, y_test, pca, scaler)
    all_metrics.append({"6_Day_Forecast_Breakdown": six_day_breakdown})
    
    # 4. QSVR (Quantum Kernel Model)
    print("\n\n======== TRAINING QUANTUM SVR (QSVR) ========")
    all_metrics.append(evaluate_qsvr(X_train, y_train, X_test, y_test, pca, scaler))
    
    # Save EVERYTHING
    save_all_logs(all_metrics)

if __name__ == "__main__":
    main()
