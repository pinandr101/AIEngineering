"""Модели для прогнозирования: baseline и GRU."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def evaluate(y_true, y_pred, model_name=""):
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }

# ---------- Baseline модели ----------
def naive_last_predict(series, horizon=1):
    """Прогноз = последнее известное значение."""
    last_val = series[-1]
    return np.full(horizon, last_val)

def moving_average_predict(series, window=7, horizon=1):
    """Прогноз = среднее за последние window дней."""
    if len(series) < window:
        return np.full(horizon, np.mean(series))
    return np.full(horizon, np.mean(series[-window:]))

class RidgeForecaster:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = None  # будет StandardScaler
        self.feature_cols = None

    def fit(self, X_train, y_train):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.feature_cols = X_train.columns
        self.model.fit(X_scaled, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class GRUForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)

def create_sequences(data_2d, window_size):
    X, y = [], []
    for i in range(len(data_2d) - window_size):
        X.append(data_2d[i:i+window_size])
        y.append(data_2d[i+window_size, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
