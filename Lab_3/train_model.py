"""
Навчання MLP-регресора для прогнозування цін на житло (без сайту).

Запуск:
    .venv\\Scripts\\python.exe train_model.py

Результати:
    models/housing_mlp.joblib      — модель + scaler-и
    models/housing_mlp_meta.json   — метрики та історія
    models/training_plot.png       — графіки навчання
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "california_housing.npz"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "housing_mlp.joblib"
META_PATH = MODELS_DIR / "housing_mlp_meta.json"
PRICE = 100_000

HIDDEN_LAYERS = (64, 32)
EPOCHS = 30
SUBSET = 15000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
ALPHA = 0.0001
SEED = 42


def load_data():
    data = np.load(DATA_PATH)
    X = data["X"].astype("float32")
    y = data["y"].astype("float32")
    names = [str(n) for n in data["feature_names"]]
    return X, y, names


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, names = load_data()

    # Фіксований тест
    x_pool, x_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Випадкова підвибірка для навчання
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(x_pool), size=min(SUBSET, len(x_pool)), replace=False)
    x_pool, y_pool = x_pool[idx], y_pool[idx]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_pool, y_pool, test_size=0.15, random_state=SEED
    )

    x_scaler = StandardScaler().fit(x_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
    x_tr_s = x_scaler.transform(x_tr)
    x_val_s = x_scaler.transform(x_val)
    x_test_s = x_scaler.transform(x_test)
    y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=SEED,
        tol=0.0,
    )

    history = {"epoch": [], "train_rmse": [], "val_rmse": [], "val_r2": []}
    print(f"Навчання: {len(x_tr)} зразків, {EPOCHS} епох, шари {HIDDEN_LAYERS}")
    for epoch in range(1, EPOCHS + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_tr_s, y_tr_s)

        train_pred = y_scaler.inverse_transform(
            model.predict(x_tr_s).reshape(-1, 1)
        ).ravel()
        val_pred = y_scaler.inverse_transform(
            model.predict(x_val_s).reshape(-1, 1)
        ).ravel()
        train_rmse = float(np.sqrt(mean_squared_error(y_tr, train_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        val_r2 = float(r2_score(y_val, val_pred))

        history["epoch"].append(epoch)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)
        print(
            f"Епоха {epoch:2d}/{EPOCHS}: train RMSE=${train_rmse*PRICE:,.0f}, "
            f"val RMSE=${val_rmse*PRICE:,.0f}, val R²={val_r2:.3f}"
        )

    # Оцінка на тесті
    test_pred = y_scaler.inverse_transform(
        model.predict(x_test_s).reshape(-1, 1)
    ).ravel()
    test = metrics(y_test, test_pred)
    print("\nТестова якість:")
    print(f"  MAE  = ${test['mae']*PRICE:,.0f}")
    print(f"  RMSE = ${test['rmse']*PRICE:,.0f}")
    print(f"  R²   = {test['r2']:.4f}")

    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": {
            "hidden_layers": list(HIDDEN_LAYERS),
            "activation": "relu",
            "solver": "adam",
            "alpha": ALPHA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "subset": SUBSET,
            "seed": SEED,
        },
        "test": test,
        "history": history,
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "features": names,
    }

    joblib.dump(
        {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}, MODEL_PATH
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Графіки
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(
        history["epoch"],
        np.array(history["train_rmse"]) * PRICE,
        marker="o",
        label="Навчання",
        color="#0891b2",
    )
    ax[0, 0].plot(
        history["epoch"],
        np.array(history["val_rmse"]) * PRICE,
        marker="o",
        label="Валідація",
        color="#f59e0b",
    )
    ax[0, 0].set_title("RMSE")
    ax[0, 0].set_xlabel("Епоха")
    ax[0, 0].set_ylabel("Помилка, $")
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(
        history["epoch"], history["val_r2"], marker="o", color="#4f46e5"
    )
    ax[0, 1].set_title("R² на валідації")
    ax[0, 1].set_xlabel("Епоха")
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].scatter(y_test * PRICE, test_pred * PRICE, s=6, alpha=0.35, color="#0891b2")
    ax[1, 0].plot([0, 500000], [0, 500000], "--", color="#ef4444")
    ax[1, 0].set_title("Прогноз vs факт")
    ax[1, 0].set_xlabel("Фактична ціна, $")
    ax[1, 0].set_ylabel("Прогноз, $")
    ax[1, 0].grid(alpha=0.3)

    residuals = (test_pred - y_test) * PRICE
    ax[1, 1].hist(residuals, bins=50, color="#7c3aed", alpha=0.8)
    ax[1, 1].axvline(0, color="#ef4444", linestyle="--")
    ax[1, 1].set_title("Розподіл залишків")
    ax[1, 1].set_xlabel("Помилка, $")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(MODELS_DIR / "training_plot.png", dpi=150)

    print(f"\nМодель: {MODEL_PATH}")
    print(f"Метадані: {META_PATH}")
    print(f"Графік: {MODELS_DIR / 'training_plot.png'}")


if __name__ == "__main__":
    main()
