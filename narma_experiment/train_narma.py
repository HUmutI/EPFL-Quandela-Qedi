import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_loader import load_narma10, load_sp500
from classical_baselines import ARModel, ARMAXModel, HARModel, HARXModel, LSTMWrapper, RCModel
from multi_qrc import HPT_QRC_Multi

matplotlib.use("Agg")

def qlike_loss(y_true, y_pred):
    """
    QLIKE Loss used in volatility forecasting:
    L(y, y_hat) = y / y_hat - log(y / y_hat) - 1
    Uses absolute values and epsilons for stability.
    """
    eps = 1e-8
    y_t = np.abs(y_true) + eps
    y_p = np.abs(y_pred) + eps
    ratio = y_t / y_p
    loss = ratio - np.log(ratio) - 1
    return np.mean(loss)

def pmcs_score(loss_values):
    """
    Placeholder for Model Confidence Set PMCS score computation.
    Computing a true PMCS requires block bootstrapping over the entire loss matrix. 
    Here we simply normalize the mean loss across models to mimic the tabular values presented.
    In real benchmarks, the 'target' PMCS is an iterative p-value elimination.
    We will just display the mean loss as standard, to replicate the metrics.
    """
    return np.mean(loss_values)

def benchmark_dataset(name, y_train, y_test, X_train=None, X_test=None):
    print("=" * 60)
    print(f"      Running Extended Benchmarks on {name}")
    print("=" * 60)
    
    results = []
    
    # 1. AR(1)
    print("-> Training AR(1)...")
    ar1 = ARModel(lags=1).fit(y_train)
    p_ar1 = ar1.predict(y_test)
    results.append({"Model": "AR1", "MSE": mean_squared_error(y_test, p_ar1), "QLIKE": qlike_loss(y_test, p_ar1)})
    
    # 2. AR(3)
    print("-> Training AR(3)...")
    ar3 = ARModel(lags=3).fit(y_train)
    p_ar3 = ar3.predict(y_test)
    results.append({"Model": "AR3", "MSE": mean_squared_error(y_test, p_ar3), "QLIKE": qlike_loss(y_test, p_ar3)})
    
    # 3. HAR
    print("-> Training HAR...")
    har = HARModel().fit(y_train)
    p_har = har.predict(y_test)
    results.append({"Model": "HAR", "MSE": mean_squared_error(y_test, p_har), "QLIKE": qlike_loss(y_test, p_har)})
    
    # 4. LSTM
    print("-> Training LSTM...")
    lstm = LSTMWrapper(input_dim=1).fit(y_train, y_train)
    p_lstm = lstm.predict(y_test, y_test)
    results.append({"Model": "LSTM", "MSE": mean_squared_error(y_test, p_lstm), "QLIKE": qlike_loss(y_test, p_lstm)})
    
    # 5. RC (Echo State Network)
    print("-> Training RC (Classic ESN)...")
    rc = RCModel(in_size=1).fit(y_train)
    p_rc = rc.predict(y_test)
    results.append({"Model": "RC", "MSE": mean_squared_error(y_test, p_rc), "QLIKE": qlike_loss(y_test, p_rc)})
    
    # 6. HPT-QRC (Quantum Reservoir)
    print("-> Training HPT-QRC (Quantum Reservoir)...")
    qrc = HPT_QRC_Multi(in_size=1, window=5).fit(y_train)
    p_qrc = qrc.predict(y_test)
    results.append({"Model": "HPT-QRC", "MSE": mean_squared_error(y_test, p_qrc), "QLIKE": qlike_loss(y_test, p_qrc)})

    # Exogenous Models (if X is provided)
    if X_train is not None and X_test is not None:
        print("--- Running Exogenous Variants ---")
        
        # 7. ARMAX
        print("-> Training ARMAX(1,0,0)...")
        armax = ARMAXModel(lags=1).fit(y_train, X_train)
        p_armax = armax.predict(y_test, X_test)
        results.append({"Model": "ARMAX", "MSE": mean_squared_error(y_test, p_armax), "QLIKE": qlike_loss(y_test, p_armax)})
        
        # 8. HARX
        print("-> Training HARX...")
        harx = HARXModel().fit(y_train, X_train)
        p_harx = harx.predict(y_test, X_test)
        results.append({"Model": "HARX", "MSE": mean_squared_error(y_test, p_harx), "QLIKE": qlike_loss(y_test, p_harx)})
        
        # 9. LSTMX
        print("-> Training LSTMX...")
        lstm_in_dim = 1 + X_train.shape[1]
        concat_train = np.hstack([y_train, X_train])
        concat_test = np.hstack([y_test, X_test])
        lstmx = LSTMWrapper(input_dim=lstm_in_dim).fit(concat_train, y_train)
        p_lstmx = lstmx.predict(concat_test, y_test)
        results.append({"Model": "LSTMX", "MSE": mean_squared_error(y_test, p_lstmx), "QLIKE": qlike_loss(y_test, p_lstmx)})
        
        # 10. RCX
        print("-> Training RCX (Exogenous ESN)...")
        rc_in_dim = 1 + X_train.shape[1]
        rcx = RCModel(in_size=rc_in_dim).fit(y_train, X_train)
        p_rcx = rcx.predict(y_test, X_test)
        results.append({"Model": "RCX", "MSE": mean_squared_error(y_test, p_rcx), "QLIKE": qlike_loss(y_test, p_rcx)})
        
        # 11. HPT-QRC-X (Exogenous Quantum Reservoir)
        print("-> Training HPT-QRC-X (Exogenous Quantum)...")
        qrc_in_dim = 1 + X_train.shape[1]
        qrc_x = HPT_QRC_Multi(in_size=qrc_in_dim, window=5).fit(y_train, X_train)
        p_qrc_x = qrc_x.predict(y_test, X_test)
        results.append({"Model": "HPT-QRC-X", "MSE": mean_squared_error(y_test, p_qrc_x), "QLIKE": qlike_loss(y_test, p_qrc_x)})

    df_res = pd.DataFrame(results)
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    df_res.to_csv(f"results/{name}_benchmark.csv", index=False)
    
    # Generate Plot
    plt.figure(figsize=(14, 7))
    plt.plot(y_test[:100], label='Actual Target', color='black', linewidth=2)
    plt.plot(p_rc[:100], label='RC (Classic)', linestyle='--', alpha=0.7)
    plt.plot(p_qrc[:100], label='HPT-QRC (Quantum)', linestyle='-.', alpha=0.9, linewidth=2)
    if X_train is not None:
        plt.plot(p_qrc_x[:100], label='HPT-QRC-X (Exogenous)', linestyle=':', alpha=0.9, linewidth=2)
        
    plt.title(f"{name} Benchmark (First 100 Test Steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{name}_overlay.png", dpi=150)
    plt.close()
    
    print(f"\n[DONE] {name} Benchmark Results:")
    print(df_res.to_string(index=False))
    print("-" * 60 + "\n")
    
    return df_res

def main():
    # 1. NARMA10 (Univariate)
    print("Loading NARMA10...")
    n_X_train, n_y_train, n_X_test, n_y_test = load_narma10()
    # NARMA10 is generated as X (input), y (target). We treat X as the exogenous 'input' sequence. 
    # The generation naturally creates a target trajectory based on X trailing sequence.
    benchmark_dataset("NARMA10", n_y_train, n_y_test, X_train=n_X_train, X_test=n_X_test)
    
    # 2. S&P 500 (Multivariate with Exogenous Fama-French markers)
    print("Loading S&P 500...")
    # 'load_sp500' returns y (RV targets), and X (Exogenous markers)
    s_y_train, s_y_test, s_X_train, s_X_test = load_sp500()
    benchmark_dataset("SP500_Realized_Volatility", s_y_train, s_y_test, X_train=s_X_train, X_test=s_X_test)

if __name__ == "__main__":
    main()
