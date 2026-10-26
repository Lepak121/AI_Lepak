"""
Лабораторна робота 3. Нейронні мережі.
Прогнозування цін на житло в Каліфорнії за допомогою MLPRegressor (scikit-learn).

Запуск:
    .venv\\Scripts\\python.exe -m streamlit run app.py
або двічі клікнути start.bat
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Шляхи, константи, опис ознак
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "california_housing.npz"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "housing_mlp.joblib"
META_PATH = MODELS_DIR / "housing_mlp_meta.json"
PRICE = 100_000  # ціль у датасеті вимірюється у $100 000

# (назва, підпис, одиниця, min, max, step, default)
FEATURES = [
    ("MedInc", "Медіанний дохід", "× $10 000", 0.5, 15.0, 0.1, 3.9),
    ("HouseAge", "Вік будинку", "років", 1.0, 52.0, 1.0, 29.0),
    ("AveRooms", "Кімнат на житло", "шт.", 1.0, 15.0, 0.1, 5.3),
    ("AveBedrms", "Спалень на житло", "шт.", 0.5, 5.0, 0.1, 1.05),
    ("Population", "Населення району", "осіб", 3.0, 20000.0, 50.0, 1166.0),
    ("AveOccup", "Людей на житло", "осіб", 0.7, 10.0, 0.1, 2.8),
    ("Latitude", "Широта", "°", 32.5, 42.0, 0.01, 34.26),
    ("Longitude", "Довгота", "°", -124.4, -114.3, 0.01, -118.49),
]
FEATURE_LABELS = {f[0]: f[1] for f in FEATURES}

st.set_page_config(
    page_title="MLP · Ціни на житло",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Оформлення
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #070b14;
        --panel: rgba(15, 23, 42, 0.72);
        --line: rgba(148, 163, 184, 0.16);
        --muted: #94a3b8;
        --cyan: #22d3ee;
        --violet: #818cf8;
        --amber: #fbbf24;
    }
    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
    .stApp {
        background:
            radial-gradient(1100px 600px at 12% -10%, rgba(34, 211, 238, 0.14), transparent 60%),
            radial-gradient(1000px 520px at 92% 0%, rgba(251, 191, 36, 0.10), transparent 55%),
            radial-gradient(900px 600px at 70% 60%, rgba(129, 140, 248, 0.08), transparent 60%),
            var(--bg);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #070b14 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #dbeafe; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1440px; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.02em; }

    .hero {
        position: relative; overflow: hidden;
        padding: 2.2rem 2.4rem; border: 1px solid var(--line); border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(34, 211, 238, 0.10), rgba(251, 191, 36, 0.08) 40%, rgba(129, 140, 248, 0.12) 75%, rgba(15, 23, 42, 0.6)),
            rgba(10, 15, 28, 0.85);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.4rem;
    }
    .hero::after {
        content: ""; position: absolute; width: 440px; height: 440px; right: -130px; top: -190px;
        background: radial-gradient(circle, rgba(251, 191, 36, 0.16), transparent 65%);
        pointer-events: none;
    }
    .eyebrow { font-size: 0.73rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--cyan); font-weight: 700; margin-bottom: 0.7rem; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.05; margin: 0 0 0.85rem; color: #f8fafc; font-weight: 700; }
    .hero h1 span { background: linear-gradient(90deg, #67e8f9, #a5b4fc 55%, #fbbf24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p { color: #a8b6cc; max-width: 820px; font-size: 1.02rem; line-height: 1.65; margin: 0; }
    .badges { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.25rem; }
    .badge { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.38rem 0.72rem; border-radius: 999px; background: rgba(148, 163, 184, 0.09); border: 1px solid rgba(148, 163, 184, 0.18); color: #cbd5e1; font-size: 0.78rem; font-weight: 600; }
    .badge b { color: #fcd34d; }

    .card { border: 1px solid var(--line); border-radius: 18px; background: var(--panel); padding: 1.1rem 1.2rem; height: 100%; }
    .card .k { color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
    .card .v { color: #f1f5f9; font-size: 1.55rem; font-weight: 700; margin-top: 0.35rem; font-family: 'Space Grotesk', sans-serif; }
    .card .s { color: #64748b; font-size: 0.76rem; margin-top: 0.2rem; }

    .arch { display: flex; align-items: stretch; justify-content: center; gap: 0.5rem; flex-wrap: wrap; padding: 0.6rem 0 0.2rem; }
    .layer { flex: 1 1 110px; min-width: 100px; text-align: center; border: 1px solid rgba(34, 211, 238, 0.22); border-radius: 14px; padding: 0.75rem 0.6rem; background: linear-gradient(180deg, rgba(34, 211, 238, 0.07), rgba(15, 23, 42, 0.45)); }
    .layer b { display: block; color: #f8fafc; font-size: 1.15rem; font-family: 'Space Grotesk', sans-serif; }
    .layer span { color: #7dd3fc; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
    .arrow { display: flex; align-items: center; color: #334155; font-size: 1.15rem; font-weight: 700; }

    .price-card {
        border: 1px solid rgba(251, 191, 36, 0.28);
        border-radius: 22px; padding: 1.4rem 1.5rem;
        background:
            radial-gradient(600px 250px at 100% 0%, rgba(251, 191, 36, 0.14), transparent 60%),
            linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(8, 12, 24, 0.95));
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    }
    .price-card .label { color: #94a3b8; text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.7rem; font-weight: 700; }
    .price-card .price { color: #fcd34d; font-size: clamp(2rem, 4vw, 3.1rem); font-weight: 800; font-family: 'Space Grotesk', sans-serif; margin: 0.2rem 0 0.1rem; }
    .price-card .range { color: #cbd5e1; font-size: 0.9rem; }
    .price-card .hint { color: #64748b; font-size: 0.78rem; margin-top: 0.55rem; }

    .note { border-left: 3px solid var(--cyan); background: rgba(34, 211, 238, 0.06); border-radius: 0 12px 12px 0; padding: 0.8rem 1rem; color: #bae6fd; font-size: 0.9rem; }
    .warn { border-left: 3px solid var(--amber); background: rgba(251, 191, 36, 0.07); border-radius: 0 12px 12px 0; padding: 0.8rem 1rem; color: #fde68a; font-size: 0.88rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 0.45rem 0.9rem; background: rgba(148, 163, 184, 0.06); }
    .stTabs [aria-selected="true"] { background: rgba(34, 211, 238, 0.12) !important; color: #67e8f9 !important; }
    div.stButton > button { border-radius: 12px; font-weight: 600; border: 1px solid rgba(34, 211, 238, 0.35); }
    div.stButton > button[kind="primary"] { background: linear-gradient(90deg, #0891b2, #4f46e5); border: none; color: white; }
    .footer { color: #475569; text-align: center; font-size: 0.78rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Дані та модель
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    data = np.load(DATA_PATH)
    X = data["X"].astype("float32")
    y = data["y"].astype("float32")
    names = [str(n) for n in data["feature_names"]]
    return X, y, names


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_model(hidden_layers, activation, solver, alpha, lr, batch_size, seed):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=seed,
        tol=0.0,
    )


def predict_bundle(bundle, X):
    z = bundle["model"].predict(bundle["x_scaler"].transform(X))
    return bundle["y_scaler"].inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()


def save_bundle(bundle, meta):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_saved_model():
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), json.loads(
                META_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            return None, {}
    return None, {}


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    parts = [p for p in re.split(r"[\s,;]+", text.strip()) if p]
    if not parts:
        raise ValueError("Вкажіть хоча б один прихований шар.")
    layers = tuple(int(p) for p in parts)
    if any(n < 4 or n > 512 for n in layers):
        raise ValueError("Кількість нейронів у шарі має бути від 4 до 512.")
    if len(layers) > 4:
        raise ValueError("Максимум 4 приховані шари.")
    return layers


def render_history(history: dict):
    if not history:
        return
    df = pd.DataFrame(history).rename(
        columns={
            "epoch": "Епоха",
            "train_rmse": "RMSE (навчання)",
            "val_rmse": "RMSE (валідація)",
            "val_r2": "R² (валідація)",
        }
    )
    df = df.set_index("Епоха")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Помилка RMSE, у $100 000 (менше — краще)")
        st.line_chart(
            df[["RMSE (навчання)", "RMSE (валідація)"]],
            color=["#22d3ee", "#fbbf24"],
            height=260,
        )
    with c2:
        st.caption("Коефіцієнт детермінації R² (більше — краще)")
        st.line_chart(df[["R² (валідація)"]], color=["#818cf8"], height=260)


# ---------------------------------------------------------------------------
# Стан сесії
# ---------------------------------------------------------------------------
if "bundle" not in st.session_state:
    saved_bundle, saved_meta = load_saved_model()
    st.session_state.bundle = saved_bundle
    st.session_state.meta = saved_meta
    st.session_state.model_source = "Збережена модель" if saved_bundle else None
    st.session_state.history = saved_meta.get("history", [])
    st.session_state.test_metrics = saved_meta.get("test", {})

with st.spinner("Завантаження даних..."):
    X_all, y_all, feature_names = load_data()

X_df = pd.DataFrame(X_all, columns=feature_names)
y_series = pd.Series(y_all, name="MedHouseVal")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))


# ---------------------------------------------------------------------------
# Бокова панель
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Параметри")
    st.caption("Налаштуйте мережу та запустіть навчання.")

    layers_text = st.text_input(
        "Приховані шари",
        value="64, 32",
        help="Нейрони через кому. Наприклад: 64, 32 або 128, 64.",
    )
    activation = st.selectbox("Функція активації", ["relu", "tanh", "logistic"])
    solver = st.selectbox("Оптимізатор", ["adam", "sgd"], help="Adam зазвичай швидший.")
    alpha = st.selectbox("L2-регуляризація (alpha)", [0.0001, 0.001, 0.01], index=0)
    lr = st.selectbox("Швидкість навчання", [0.001, 0.0005, 0.0001], index=0)
    batch_size = st.selectbox("Розмір пакунка", [32, 64, 128, 256], index=1)
    epochs = st.slider("Кількість епох", 1, 60, 30)
    subset = st.slider("Навчальна вибірка", 2000, 17000, 15000, step=1000)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    st.markdown("---")
    train_clicked = st.button(
        "🚀 Навчити модель", type="primary", use_container_width=True
    )

    if st.session_state.bundle is not None:
        if st.button("↺ Скинути до збереженої", use_container_width=True):
            m, meta = load_saved_model()
            st.session_state.bundle = m
            st.session_state.meta = meta
            st.session_state.history = meta.get("history", [])
            st.session_state.test_metrics = meta.get("test", {})
            st.session_state.model_source = "Збережена модель" if m else None
            st.rerun()

    st.markdown("---")
    st.caption("Датасет: California Housing (20 640 районів)")
    st.caption("Бібліотека: scikit-learn · MLPRegressor")


# ---------------------------------------------------------------------------
# Заголовок
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Лабораторна робота 03 · Нейронні мережі</div>
        <h1>Прогнозування <span>цін на житло</span> в Каліфорнії</h1>
        <p>
            Багатошаровий перцептрон (MLP) навчається передбачати медіанну вартість
            житла за характеристиками району: доходом мешканців, віком будинків,
            кількістю кімнат, населенням і координатами. Змінюйте архітектуру,
            навчайте модель і одразу отримуйте оцінку вартості у браузері.
        </p>
        <div class="badges">
            <span class="badge">⚡ <b>MLPRegressor</b></span>
            <span class="badge">🏘️ 20 640 районів</span>
            <span class="badge">📐 8 ознак</span>
            <span class="badge">🌐 Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Навчання
# ---------------------------------------------------------------------------
if train_clicked:
    try:
        hidden_layers = parse_hidden_layers(layers_text)
    except ValueError as exc:
        st.error(f"Помилка в архітектурі: {exc}")
        st.stop()

    np.random.seed(int(seed))

    # Фіксований тестовий набір — щоб результати були порівнюваними
    x_pool, x_test_fixed, y_pool, y_test_fixed = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42
    )

    rng = np.random.RandomState(int(seed))
    take = min(int(subset), len(x_pool))
    idx = rng.choice(len(x_pool), size=take, replace=False)
    x_pool, y_pool = x_pool[idx], y_pool[idx]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_pool, y_pool, test_size=0.15, random_state=int(seed)
    )

    x_scaler = StandardScaler().fit(x_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
    x_tr_s = x_scaler.transform(x_tr)
    x_val_s = x_scaler.transform(x_val)
    x_test_s = x_scaler.transform(x_test_fixed)
    y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()

    model = make_model(
        hidden_layers,
        activation,
        solver,
        alpha,
        lr,
        min(int(batch_size), len(x_tr)),
        int(seed),
    )

    progress = st.progress(0.0, text="Підготовка...")
    status_box = st.empty()
    history = {"epoch": [], "train_rmse": [], "val_rmse": [], "val_r2": []}

    for epoch in range(1, int(epochs) + 1):
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

        progress.progress(
            epoch / int(epochs),
            text=(
                f"Епоха {epoch}/{epochs} · val RMSE ${val_rmse*PRICE:,.0f} · "
                f"R² {val_r2:.3f}"
            ),
        )
        status_box.markdown(
            f"**Епоха {epoch}/{epochs}** · "
            f"RMSE навчання `${train_rmse*PRICE:,.0f}` · "
            f"RMSE валідації `${val_rmse*PRICE:,.0f}` · "
            f"R² `{val_r2:.3f}`"
        )

    progress.progress(1.0, text="Оцінювання на тестовій вибірці...")
    test_pred = y_scaler.inverse_transform(
        model.predict(x_test_s).reshape(-1, 1)
    ).ravel()
    test = regression_metrics(y_test_fixed, test_pred)

    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": {
            "hidden_layers": list(hidden_layers),
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "learning_rate": lr,
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "subset": int(subset),
            "seed": int(seed),
        },
        "test": test,
        "history": history,
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test_fixed)),
    }
    bundle = {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}
    save_bundle(bundle, meta)

    st.session_state.bundle = bundle
    st.session_state.meta = meta
    st.session_state.model_source = "Навчено у цій сесії"
    st.session_state.history = history
    st.session_state.test_metrics = test
    st.session_state.pop("importance", None)

    progress.empty()
    status_box.empty()
    st.session_state.flash = (
        f"Навчання завершено! R² = {test['r2']:.3f}, "
        f"MAE = ${test['mae']*PRICE:,.0f}, RMSE = ${test['rmse']*PRICE:,.0f}."
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------
tab_train, tab_predict, tab_analysis, tab_about = st.tabs(
    ["🏋️ Навчання", "🏠 Прогноз", "🗺️ Карта та аналіз", "📘 Про проєкт"]
)

bundle = st.session_state.bundle
meta = st.session_state.meta or {}
test_metrics = st.session_state.test_metrics or {}
history = st.session_state.history or []

# ============================== НАВЧАННЯ ===================================
with tab_train:
    if bundle is not None:
        cfg = meta.get("config", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="card"><div class="k">R² моделі</div>'
                f'<div class="v">{test_metrics.get("r2", 0):.3f}</div>'
                f'<div class="s">частка поясненої дисперсії</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="card"><div class="k">MAE</div>'
                f'<div class="v">${test_metrics.get("mae", 0)*PRICE:,.0f}</div>'
                f'<div class="s">середня абсолютна помилка</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="card"><div class="k">RMSE</div>'
                f'<div class="v">${test_metrics.get("rmse", 0)*PRICE:,.0f}</div>'
                f'<div class="s">корінь із середньоквадратичної</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="card"><div class="k">Джерело</div>'
                f'<div class="v" style="font-size:1.0rem">{st.session_state.model_source or "—"}</div>'
                f'<div class="s">{meta.get("trained_at", "—")}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("### 🧬 Архітектура мережі")
        layers = [8] + list(cfg.get("hidden_layers", [])) + [1]
        labels = ["Вхід"] + [f"Шар {i}" for i in range(1, len(layers) - 1)] + ["Вихід"]
        blocks = []
        for i, (n, label) in enumerate(zip(layers, labels)):
            if i:
                blocks.append('<div class="arrow">→</div>')
            sub = (
                "8 ознак"
                if i == 0
                else ("ціна, $100k" if i == len(layers) - 1 else "ReLU")
            )
            blocks.append(
                f'<div class="layer"><span>{label}</span><b>{n}</b><span>{sub}</span></div>'
            )
        st.markdown(f'<div class="arch">{"".join(blocks)}</div>', unsafe_allow_html=True)

        if history:
            st.markdown("### 📈 Історія навчання")
            render_history(history)

        with st.expander("📋 Деталі навчання"):
            st.markdown(
                f"""
                - **Приховані шари:** `{cfg.get("hidden_layers")}`
                - **Активація:** `{cfg.get("activation")}`
                - **Оптимізатор:** `{cfg.get("solver")}`, learning rate `{cfg.get("learning_rate")}`
                - **L2 (alpha):** `{cfg.get("alpha")}`, batch size `{cfg.get("batch_size")}`
                - **Епохи:** `{cfg.get("epochs")}`, навчальна вибірка: `{cfg.get("subset")}`
                - **Розміри:** train `{meta.get("n_train", "—")}`, val `{meta.get("n_val", "—")}`, test `{meta.get("n_test", "—")}`
                """
            )
    else:
        st.info(
            "Модель ще не навчена. Натисніть **«Навчити модель»** у лівій панелі. "
            "Типові параметри (64, 32; 15 000 зразків; 30 епох) дають R² ≈ 0.78 "
            "і навчаються за кілька секунд."
        )

    st.markdown("---")
    st.markdown("### 📊 Датасет California Housing")
    st.caption(
        "Кожен рядок — census block group (район перепису). "
        "Цільова змінна — медіанна вартість житла, у $100 000."
    )
    preview = pd.DataFrame(
        X_all[:8], columns=[FEATURE_LABELS.get(n, n) for n in feature_names]
    )
    preview["Медіанна ціна"] = (y_all[:8] * PRICE).round(0).astype(int)
    st.dataframe(preview, hide_index=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Районів у датасеті", f"{len(X_all):,}")
    c2.metric("Медіанна ціна", f"${np.median(y_all)*PRICE:,.0f}")
    c3.metric("Діапазон цін", f"${y_all.min()*PRICE/1000:.0f}k – ${y_all.max()*PRICE/1000:.0f}k")

    st.markdown(
        '<div class="warn"><b>Важливо:</b> у датасеті ціни обрізані зверху на позначці '
        "$500 001. Тому модель не може передбачити дорожчі об'єкти — це обмеження даних, "
        "а не помилка навчання.</div>",
        unsafe_allow_html=True,
    )

# ================================ ПРОГНОЗ ==================================
with tab_predict:
    st.markdown("### 🏠 Оцініть вартість житла")
    st.caption(
        "Оберіть характеристики району — мережа одразу розрахує прогноз. "
        "Значення за замовчуванням близькі до медіан датасету."
    )

    if bundle is None:
        st.warning("Спочатку навчіть модель на вкладці «Навчання».")
    else:
        values = {}
        row1 = st.columns(4)
        row2 = st.columns(4)
        widget_cols = list(row1) + list(row2)
        for col, feat in zip(widget_cols, FEATURES):
            name, label, unit, lo, hi, step, default = feat
            with col:
                values[name] = st.slider(
                    f"{label}, {unit}",
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    key=f"pred_{name}",
                )

        x_input = np.array([[values[f[0]] for f in FEATURES]], dtype="float32")
        pred = float(predict_bundle(bundle, x_input)[0])
        pred = float(np.clip(pred, 0.15, 5.00001))
        price = pred * PRICE
        rmse = float(test_metrics.get("rmse", meta.get("test", {}).get("rmse", 0.5)))
        low = max(0.0, price - 1.96 * rmse * PRICE)
        high = price + 1.96 * rmse * PRICE

        st.write("")
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="label">Оцінка нейромережі</div>
                    <div class="price">${price:,.0f}</div>
                    <div class="range">Орієнтовний діапазон: ${low:,.0f} – ${high:,.0f}</div>
                    <div class="hint">Діапазон побудовано за типовою помилкою моделі RMSE
                    (95% інтервал для залишків). Це оцінка, а не юридична оцінка нерухомості.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.metric("Медіанна ціна в датасеті", f"${np.median(y_all)*PRICE:,.0f}")
            diff = price - np.median(y_all) * PRICE
            st.metric(
                "Різниця з медіаною",
                f"{'+' if diff >= 0 else '−'}${abs(diff):,.0f}",
                delta=f"{diff / (np.median(y_all)*PRICE)*100:+.1f}%",
            )
            st.metric("R² моделі", f"{test_metrics.get('r2', 0):.3f}")
            st.metric("Типова помилка (RMSE)", f"${rmse*PRICE:,.0f}")

        st.markdown("---")
        st.markdown("#### Що впливає на прогноз")
        summary = pd.DataFrame(
            [
                [FEATURE_LABELS[f[0]], f"{values[f[0]]:g}", f[2]]
                for f in FEATURES
            ],
            columns=["Ознака", "Значення", "Одиниця"],
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown(
            '<div class="note">Найсильніше на ціну зазвичай впливають медіанний дохід '
            "району, координати (близькість до узбережжя та великих міст) і кількість "
            "кімнат. Нейромережа враховує всі ознаки разом, тому прогноз не є простою "
            "лінійною формулою.</div>",
            unsafe_allow_html=True,
        )

# ============================ КАРТА ТА АНАЛІЗ ==============================
with tab_analysis:
    if bundle is None:
        st.warning("Спочатку навчіть модель на вкладці «Навчання».")
    else:
        # Фіксований тестовий набір для аналізу
        _, x_test_fixed, _, y_test_fixed = train_test_split(
            X_all, y_all, test_size=0.15, random_state=42
        )
        pred_test = predict_bundle(bundle, x_test_fixed)

        lon_idx = feature_names.index("Longitude")
        lat_idx = feature_names.index("Latitude")
        analysis = pd.DataFrame(
            {
                "Довгота": x_test_fixed[:, lon_idx],
                "Широта": x_test_fixed[:, lat_idx],
                "Фактична ціна": y_test_fixed * PRICE,
                "Прогноз": pred_test * PRICE,
            }
        )
        analysis["Помилка"] = analysis["Прогноз"] - analysis["Фактична ціна"]
        analysis["Абсолютна помилка"] = analysis["Помилка"].abs()

        st.markdown("### 🗺️ Географія цін")
        mode = st.radio(
            "Що показати на карті",
            ["Фактична ціна", "Прогноз моделі", "Помилка прогнозу"],
            horizontal=True,
        )
        map_df = analysis.sample(n=min(3000, len(analysis)), random_state=1)
        if mode == "Помилка прогнозу":
            color = alt.Color(
                "Помилка:Q",
                scale=alt.Scale(scheme="redblue", domain=[-2.5, 2.5]),
                title="Помилка, $100k",
            )
        else:
            color = alt.Color(
                f"{mode}:Q",
                scale=alt.Scale(scheme="viridis"),
                title=f"{mode}, $",
            )
        chart = (
            alt.Chart(map_df)
            .mark_circle(size=16, opacity=0.68)
            .encode(
                x=alt.X(
                    "Довгота:Q",
                    scale=alt.Scale(zero=False),
                    title="Довгота",
                ),
                y=alt.Y(
                    "Широта:Q",
                    scale=alt.Scale(zero=False),
                    title="Широта",
                ),
                color=color,
                tooltip=[
                    alt.Tooltip("Широта:Q", format=".2f"),
                    alt.Tooltip("Довгота:Q", format=".2f"),
                    alt.Tooltip("Фактична ціна:Q", format="$,.0f"),
                    alt.Tooltip("Прогноз:Q", format="$,.0f"),
                    alt.Tooltip("Помилка:Q", format="$,.0f"),
                ],
            )
            .properties(height=520, title="Тестова вибірка: кожна точка — район")
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Точки утворюють контур Каліфорнії. Видно, що дорожчі райони "
            "концентруються біля узбережжя та великих міст."
        )

        st.markdown("---")
        st.markdown("### 🎯 Наскільки точний прогноз")
        c1, c2 = st.columns(2)
        with c1:
            scatter_df = analysis.sample(n=min(2500, len(analysis)), random_state=2)
            points = (
                alt.Chart(scatter_df)
                .mark_circle(size=18, opacity=0.45, color="#22d3ee")
                .encode(
                    x=alt.X("Фактична ціна:Q", title="Фактична ціна, $"),
                    y=alt.Y("Прогноз:Q", title="Прогноз, $"),
                    tooltip=[
                        alt.Tooltip("Фактична ціна:Q", format="$,.0f"),
                        alt.Tooltip("Прогноз:Q", format="$,.0f"),
                    ],
                )
            )
            line_df = pd.DataFrame({"x": [0, 500000], "y": [0, 500000]})
            line = (
                alt.Chart(line_df)
                .mark_line(color="#fbbf24", strokeDash=[5, 4], size=2)
                .encode(x="x:Q", y="y:Q")
            )
            st.altair_chart(
                (points + line).properties(
                    height=380, title="Прогноз vs фактична ціна"
                ),
                use_container_width=True,
            )
        with c2:
            resid = (
                alt.Chart(analysis)
                .mark_bar(color="#818cf8", opacity=0.8)
                .encode(
                    x=alt.X(
                        "Помилка:Q",
                        bin=alt.Bin(maxbins=45),
                        title="Помилка (прогноз − факт), $100k",
                    ),
                    y=alt.Y("count():Q", title="Кількість районів"),
                )
                .properties(height=380, title="Розподіл залишків")
            )
            st.altair_chart(resid, use_container_width=True)
        st.caption(
            f"MAE = ${test_metrics.get('mae', 0)*PRICE:,.0f} · "
            f"RMSE = ${test_metrics.get('rmse', 0)*PRICE:,.0f} · "
            f"R² = {test_metrics.get('r2', 0):.3f}"
        )

        st.markdown("---")
        st.markdown("### 🔎 Важливість ознак")
        st.caption(
            "Permutation importance: наскільки падає R², якщо випадково перемішати "
            "значення однієї ознаки. Обчислення на 2 000 тестових зразках."
        )
        if st.button("Обчислити важливість ознак"):
            rng = np.random.RandomState(7)
            n = min(2000, len(x_test_fixed))
            sample_idx = rng.choice(len(x_test_fixed), size=n, replace=False)
            x_sample = x_test_fixed[sample_idx]
            y_sample = y_test_fixed[sample_idx]
            x_sample_s = bundle["x_scaler"].transform(x_sample)
            base_pred = predict_bundle(bundle, x_sample)
            base_r2 = r2_score(y_sample, base_pred)

            rows = []
            for i, name in enumerate(feature_names):
                x_perm = x_sample_s.copy()
                x_perm[:, i] = rng.permutation(x_perm[:, i])
                z = bundle["model"].predict(x_perm)
                pred = bundle["y_scaler"].inverse_transform(
                    z.reshape(-1, 1)
                ).ravel()
                rows.append(
                    {
                        "Ознака": FEATURE_LABELS.get(name, name),
                        "Падіння R²": max(0.0, base_r2 - r2_score(y_sample, pred)),
                    }
                )
            importance_df = (
                pd.DataFrame(rows)
                .sort_values("Падіння R²", ascending=False)
                .reset_index(drop=True)
            )
            st.session_state.importance = importance_df

        if "importance" in st.session_state:
            imp = st.session_state.importance
            chart = (
                alt.Chart(imp)
                .mark_bar(
                    color="#fbbf24",
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                )
                .encode(
                    x=alt.X("Падіння R²:Q", title="Падіння R²"),
                    y=alt.Y("Ознака:N", sort="-x", title=""),
                    tooltip=[
                        alt.Tooltip("Ознака:N"),
                        alt.Tooltip("Падіння R²:Q", format=".4f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Натисніть кнопку вище, щоб розрахувати важливість ознак.")

# ============================== ПРО ПРОЄКТ =================================
with tab_about:
    st.markdown("### 📘 Про проєкт")
    st.markdown(
        """
        Це лабораторний проєкт із нейронних мереж. Замість розпізнавання цифр
        тут розв'язується задача **регресії**: за 8 характеристиками району
        потрібно передбачити медіанну вартість житла.

        Використана бібліотека — **scikit-learn**, клас `MLPRegressor`
        (багатошаровий перцептрон для регресії). Навчання виконується методом
        зворотного поширення помилки. **FANN** згадано лише як приклад
        альтернативної бібліотеки: це C-бібліотека з Python-обгортками, тоді як
        scikit-learn є сучасним стандартом для класичного машинного навчання.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚙️ Як відбувається навчання")
        st.markdown(
            """
            1. **Ознаки:** 8 числових характеристик району.
            2. **Масштабування:** `StandardScaler` приводить ознаки до нульового
               середнього та одиничної дисперсії; цільова змінна масштабується окремо.
            3. **Forward pass:** вектор проходить приховані шари з ReLU-активацією.
            4. **Вихід:** один нейрон без активації дає прогноз ціни.
            5. **Loss:** середньоквадратична помилка (MSE) між прогнозом і фактом.
            6. **Backprop + Adam:** ваги коригуються, поки помилка не зменшиться.
            7. **Епоха:** один повний прохід по навчальній вибірці.
            """
        )
    with c2:
        st.markdown("#### 🧮 Метрики якості")
        st.latex(r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|")
        st.latex(
            r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}"
        )
        st.latex(
            r"R^2 = 1 - \frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}"
        )
        st.caption(
            "MAE і RMSE — у доларах (чим менше, тим краще). "
            "R² = 1 — ідеальний прогноз; 0 — рівень простого середнього."
        )

    st.markdown("---")
    st.markdown("### 🗂️ Структура проєкту")
    st.code(
        """Lab_3/
├── app.py                    # сайт: навчання, прогноз, карта, аналіз
├── train_model.py            # навчання та збереження моделі без сайту
├── mlp_numpy.py              # MLP «з нуля» на NumPy (додатковий приклад)
├── start.bat                 # запуск сайту подвійним кліком
├── requirements.txt
├── data/
│   └── california_housing.npz
└── models/
    ├── housing_mlp.joblib    # модель + scaler-и
    └── housing_mlp_meta.json # метрики та історія навчання""",
        language="text",
    )

    st.markdown("### ▶️ Як запустити")
    st.markdown(
        """
        **Варіант 1 — подвійний клік:** запустіть `start.bat`, потім відкрийте
        [http://localhost:8501](http://localhost:8501).

        **Варіант 2 — термінал:**
        """
    )
    st.code(
        """.venv\\Scripts\\activate
python -m streamlit run app.py""",
        language="powershell",
    )

    st.markdown("### 📚 Використані технології")
    tech = pd.DataFrame(
        [
            ["scikit-learn", "MLPRegressor — навчання нейронної мережі"],
            ["Streamlit", "веб-інтерфейс, графіки, інтерактивність"],
            ["NumPy", "обробка числових масивів"],
            ["pandas", "таблиці та аналіз даних"],
            ["Matplotlib", "візуалізація"],
            ["Altair", "інтерактивна карта та графіки"],
        ],
        columns=["Бібліотека", "Роль у проєкті"],
    )
    st.dataframe(tech, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="note">Дані: California Housing (StatLib, Pace &amp; Barry, 1997). '
        "Ціль обрізана на $500 001, тому максимальний прогноз також обмежений.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">Лабораторна робота 3 · Нейронні мережі · '
    "scikit-learn + Streamlit</div>",
    unsafe_allow_html=True,
)
