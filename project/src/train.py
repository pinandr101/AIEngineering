"""Обучение моделей для каждой категории товаров (сохранение Ridge и GRU)."""
import random
import yaml
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from src.data import load_raw_data, aggregate_sales_by_category
from src.features import build_features
from src.models import (
    naive_last_predict, moving_average_predict,
    RidgeForecaster, evaluate,
    GRUForecaster, create_sequences
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def temporal_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    tr_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:tr_end], df.iloc[tr_end:val_end], df.iloc[val_end:]

def train_gru_for_series(series_df, cfg, device):
    """Обучение GRU на одном временном ряде. Возвращает model, scaler, metrics."""
    lags = cfg["features"]["lags"]
    rolling = cfg["features"]["rolling_windows"]
    feat_df = build_features(series_df, "sales", lags, rolling)
    feature_cols = [c for c in feat_df.columns if c not in ("date", "sales")]

    train_df, val_df, test_df = temporal_split(
        feat_df,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"]
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[["sales"]])
    val_scaled = scaler.transform(val_df[["sales"]])
    test_scaled = scaler.transform(test_df[["sales"]])

    window = cfg["model"]["gru"]["window_size"]
    X_tr, y_tr = create_sequences(train_scaled, window)
    X_val, y_val = create_sequences(val_scaled, window)
    X_te, y_te = create_sequences(test_scaled, window)

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=cfg["model"]["gru"]["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["model"]["gru"]["batch_size"])

    model = GRUForecaster(
        hidden_size=cfg["model"]["gru"]["hidden_size"],
        num_layers=cfg["model"]["gru"]["num_layers"]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["model"]["gru"]["lr"])
    crit = nn.MSELoss()
    best_mae = float("inf")
    best_state = None

    for epoch in range(1, cfg["model"]["gru"]["num_epochs"] + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                p = model(Xb).cpu().numpy()
                preds.append(p)
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        trues_orig = scaler.inverse_transform(trues.reshape(-1, 1)).ravel()
        mae = np.mean(np.abs(preds_orig - trues_orig))
        if mae < best_mae:
            best_mae = mae
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)

    # Тестовая оценка
    test_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    test_loader = DataLoader(test_ds, batch_size=cfg["model"]["gru"]["batch_size"])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            p = model(Xb).cpu().numpy()
            preds.append(p)
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    trues_orig = scaler.inverse_transform(trues.reshape(-1, 1)).ravel()
    gru_metrics = evaluate(trues_orig, preds_orig, "GRU")

    return model, scaler, gru_metrics

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw = load_raw_data(cfg["data"]["raw_path"])
    cat_series = aggregate_sales_by_category(raw)

    Path("data/processed").mkdir(exist_ok=True)
    for cat, s in cat_series.items():
        s.to_csv(f"data/processed/{cat}_series.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_all = []
    feature_cols_global = None  # сохраним для сервиса

    Path("artifacts").mkdir(exist_ok=True)

    # Обучаем модели для каждой категории
    for cat, series_df in cat_series.items():
        print(f"\n===== Training models for category: {cat} =====")
        lags = cfg["features"]["lags"]
        rolling = cfg["features"]["rolling_windows"]
        feat_df = build_features(series_df, "sales", lags, rolling)
        feature_cols = [c for c in feat_df.columns if c not in ("date", "sales")]
        if feature_cols_global is None:
            feature_cols_global = feature_cols

        train_df, val_df, test_df = temporal_split(
            feat_df,
            train_frac=cfg["split"]["train_frac"],
            val_frac=cfg["split"]["val_frac"]
        )
        y_train = train_df["sales"].values
        y_val = val_df["sales"].values

        # Baseline 1 naive
        naive_pred = naive_last_predict(y_train, len(y_val))
        naive_metrics = evaluate(y_val, naive_pred, "B1_naive_last")

        # Baseline 2 moving average
        ma_pred = moving_average_predict(y_train, window=7, horizon=len(y_val))
        ma_metrics = evaluate(y_val, ma_pred, "B2_moving_avg")

        # Ridge-модель
        X_tr = train_df[feature_cols]
        X_val = val_df[feature_cols]
        ridge = RidgeForecaster(alpha=1.0)
        ridge.fit(X_tr, y_train)
        ridge_pred = ridge.predict(X_val)
        ridge_metrics = evaluate(y_val, ridge_pred, "B3_ridge")

        # Сохраняем Ridge-модель и её scaler признаков (он внутри)
        joblib.dump(ridge, f"artifacts/best_ridge_{cat}.pkl")

        # GRU обучаем и сохраняем для истории
        gru_model, scaler, gru_metrics = train_gru_for_series(series_df, cfg, device)
        torch.save(gru_model.state_dict(), f"artifacts/best_gru_{cat}.pt")
        joblib.dump(scaler, f"artifacts/scaler_{cat}.pkl")

        results_all.append({
            "category": cat,
            "naive_mae": naive_metrics["MAE"],
            "naive_rmse": naive_metrics["RMSE"],
            "ma_mae": ma_metrics["MAE"],
            "ma_rmse": ma_metrics["RMSE"],
            "ridge_mae": ridge_metrics["MAE"],
            "ridge_rmse": ridge_metrics["RMSE"],
            "gru_mae": gru_metrics["MAE"],
            "gru_rmse": gru_metrics["RMSE"],
        })

    df_res = pd.DataFrame(results_all)
    df_res.to_csv("artifacts/runs.csv", index=False)

    # Сохраняем конфигурацию для сервиса (теперь нужны фичи)
    service_cfg = {
        "window_size": cfg["model"]["gru"]["window_size"],  # для совместимости, не используется Ridge
        "categories": list(cat_series.keys()),
        "feature_cols": feature_cols_global,
        "lags": cfg["features"]["lags"],
        "rolling_windows": cfg["features"]["rolling_windows"],
    }
    with open("artifacts/service_config.yaml", "w") as f:
        yaml.dump(service_cfg, f)

    print("\n===== Training complete =====")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
