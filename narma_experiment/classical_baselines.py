import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from esn_baseline import EchoStateNetwork

# --- Autoregressive & HAR Models ---

class ARModel:
    def __init__(self, lags=1):
        self.lags = lags
        self.model = None
        
    def fit(self, y_train):
        # sm.tsa.AutoReg requires 1D Series
        self.model = sm.tsa.AutoReg(y_train.flatten(), lags=self.lags).fit()
        return self
        
    def predict(self, y_test):
        # We need a rolling prediction since AR uses past y to predict future y.
        # For simplicity in evaluation, we will predict one-step ahead dynamically.
        y_hist = self.model.model.endog.tolist()
        preds = []
        for t in range(len(y_test)):
            pred = self.model.predict(start=len(y_hist), end=len(y_hist), dynamic=False)[0]
            preds.append(pred)
            y_hist.append(y_test[t][0])
            self.model = sm.tsa.AutoReg(y_hist, lags=self.lags).fit()
        return np.array(preds).reshape(-1, 1)

class ARMAXModel:
    def __init__(self, lags=1):
        self.lags = lags
        self.model = None
        
    def fit(self, y_train, X_train):
        self.model = sm.tsa.ARIMA(y_train.flatten(), exog=X_train, order=(self.lags, 0, 0)).fit()
        return self
        
    def predict(self, y_test, X_test):
        # In a real rolling forecast, we'd iteratively append y_test and refit.
        # For execution speed in our benchmark, we'll forecast the test set directly using all X_test.
        preds = self.model.predict(start=self.model.nobs, end=self.model.nobs + len(y_test) - 1, exog=X_test)
        return np.array(preds).reshape(-1, 1)

class HARModel:
    def __init__(self, ridge_alpha=1e-4):
        self.ridge = Ridge(alpha=ridge_alpha)
        
    def _create_har_features(self, y):
        # HAR features: 1-day lag, 5-day moving average, 22-day moving average
        features = []
        y_flat = y.flatten()
        for t in range(22, len(y_flat)):
            day_lag = y_flat[t-1]
            week_ma = np.mean(y_flat[t-5:t])
            month_ma = np.mean(y_flat[t-22:t])
            features.append([day_lag, week_ma, month_ma])
        return np.array(features)
        
    def fit(self, y_train):
        X_har = self._create_har_features(y_train)
        y_target = y_train[22:]
        self.ridge.fit(X_har, y_target)
        self.last_y = y_train.copy()
        return self
        
    def predict(self, y_test):
        # Rolling forecast
        y_concat = np.vstack([self.last_y[-22:], y_test])
        X_har = self._create_har_features(y_concat)
        return self.ridge.predict(X_har)

class HARXModel(HARModel):
    def _create_harx_features(self, y, X):
        features = []
        y_flat = y.flatten()
        for t in range(22, len(y_flat)):
            day_lag = y_flat[t-1]
            week_ma = np.mean(y_flat[t-5:t])
            month_ma = np.mean(y_flat[t-22:t])
            # Append exogenous X at time t-1
            exog_vars = X[t-1]
            feat = [day_lag, week_ma, month_ma] + list(exog_vars)
            features.append(feat)
        return np.array(features)
        
    def fit(self, y_train, X_train):
        X_harx = self._create_harx_features(y_train, X_train)
        y_target = y_train[22:]
        self.ridge.fit(X_harx, y_target)
        self.last_y = y_train[-22:].copy()
        self.last_X = X_train[-22:].copy()
        return self
        
    def predict(self, y_test, X_test):
        y_concat = np.vstack([self.last_y, y_test])
        X_concat = np.vstack([self.last_X, X_test])
        X_harx = self._create_harx_features(y_concat, X_concat)
        return self.ridge.predict(X_harx)


# --- Deep Learning (LSTM) ---

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

class LSTMWrapper:
    def __init__(self, input_dim=1, seq_length=5, epochs=50, lr=0.01):
        self.seq_length = seq_length
        self.epochs = epochs
        self.model = SimpleLSTM(input_dim=input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def _create_sequences(self, data, targets):
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            x = data[i:(i + self.seq_length)]
            y = targets[i + self.seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
        
    def fit(self, train_data, y_train):
        # train_data is just y_train for pure LSTM, or [y_train, X_train] for LSTMX
        X_seq, y_seq = self._create_sequences(train_data, y_train)
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)
        
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(X_t)
            loss = self.criterion(out, y_t)
            loss.backward()
            self.optimizer.step()
            
        self.last_data = train_data[-self.seq_length:]
        return self
        
    def predict(self, test_data, y_test):
        concat_data = np.vstack((self.last_data, test_data))
        X_seq, _ = self._create_sequences(concat_data, np.vstack([np.zeros((self.seq_length,1)), y_test]))
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).numpy()
        return preds

# --- Reservoir Computing ---
# The generic RC class relies on esn_baseline.EchoStateNetwork, 
# appending X inputs alongside y lags.

class RCModel:
    def __init__(self, in_size=1, ridge_alpha=1e-4):
        self.esn = EchoStateNetwork(in_size=in_size, ridge_alpha=ridge_alpha)
        
    def fit(self, y_train, X_train=None):
        if X_train is not None:
            features = np.hstack([y_train, X_train])
        else:
            features = y_train
        
        # Shift targets by 1 for forecasting
        feat_in = features[:-1]
        targ_out = y_train[1:]
        
        self.esn.fit(feat_in, targ_out, discard_steps=100)
        self.last_feat = features[-1:]
        return self
        
    def predict(self, y_test, X_test=None):
        if X_test is not None:
            features = np.hstack([y_test, X_test])
        else:
            features = y_test
            
        # Predict 1 step ahead continuously
        feat_in = np.vstack([self.last_feat, features[:-1]])
        return self.esn.predict(feat_in)
