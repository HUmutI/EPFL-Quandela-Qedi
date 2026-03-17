import os
import numpy as np
import pandas as pd

def load_narma10(n_train=800, n_test=200, warmup=1000, seed=42):
    """
    Generate the classic NARMA10 dataset.
    """
    np.random.seed(seed)
    total_length = warmup + n_train + n_test
    
    u = 0.5 * np.random.rand(total_length)
    y = np.zeros(total_length)
    
    for t in range(9, total_length - 1):
        y[t+1] = (0.3 * y[t] 
                  + 0.05 * y[t] * np.sum(y[t-9:t+1]) 
                  + 1.5 * u[t] * u[t-9] 
                  + 0.1)
        
    u_final = u[warmup:]
    y_final = y[warmup:]
    
    X_train = u_final[:n_train].reshape(-1, 1)
    y_train = y_final[:n_train].reshape(-1, 1)
    X_test = u_final[n_train:].reshape(-1, 1)
    y_test = y_final[n_train:].reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test

def load_sp500(filepath="/Users/umut/Desktop/EPFL_ANTI/narma_experiment/literature/qrc_repo/Data.CSV"):
    """
    Loads the Realized Volatility data from the QRC repository.
    The file 'Data.CSV' contains 'RV' as the target, and 'MKT', 'SMB', 'HML' etc as exogenous regressors.
    
    For exact replicability, we'll mimic the standard volatility forecasting train/test split:
    usually ~70-80% train, ~20-30% test, but here we do: 1950-1999 train, 2000+ test as standard,
    or just use a raw ratio if dates aren't easily partitioned.
    Actually, let's use an 80/20 train/test split.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"S&P 500 Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    # 'RV' is the target realized volatility
    y_target = df['RV'].values.reshape(-1, 1)
    
    # Exogenous variables (X) for HARX, ARMAX, RCX etc.
    # Standard choice in empirical finance is to use returns, risk-free rate, or other Fama-French markers.
    exog_cols = ['MKT', 'SMB', 'HML', 'TB', 'DEF', 'IP', 'INF']
    X_exog = df[exog_cols].values
    
    n_samples = len(df)
    n_train = int(n_samples * 0.8)
    
    # Core target history (univariate modeling without X)
    y_train = y_target[:n_train]
    y_test = y_target[n_train:]
    
    # Exogenous variable inputs
    X_train = X_exog[:n_train]
    X_test = X_exog[n_train:]
    
    return y_train, y_test, X_train, X_test
