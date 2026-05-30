"""Генерация признаков для временного ряда."""
import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out

def add_lag_features(df: pd.DataFrame, target_col: str,
                     lags: list = [1, 7, 14],
                     rolling_windows: list = [7]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for w in rolling_windows:
        out[f"rolling_mean_{w}"] = out[target_col].shift(1).rolling(window=w).mean()
        out[f"rolling_std_{w}"] = out[target_col].shift(1).rolling(window=w).std()
    return out

def build_features(df: pd.DataFrame, target_col: str = "sales",
                   lags: list = [1,7,14], rolling_windows: list = [7]):
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col, lags, rolling_windows)
    return df.dropna().reset_index(drop=True)
