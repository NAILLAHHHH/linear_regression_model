# training helpers used by the notebook and the API

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
TARGET = "life_expectancy"
FEATURES = [
    "status",
    "adult_mortality",
    "alcohol",
    "bmi",
    "polio",
    "hiv_aids",
    "gdp",
    "income_composition",
    "schooling",
]

# rename messy WHO column names
RENAME = {
    "Life expectancy ": "life_expectancy",
    "Adult Mortality": "adult_mortality",
    "Alcohol": "alcohol",
    " BMI ": "bmi",
    "Polio": "polio",
    " HIV/AIDS": "hiv_aids",
    "GDP": "gdp",
    "Income composition of resources": "income_composition",
    "Schooling": "schooling",
    "Status": "status",
}


def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns=RENAME)
    if "status" in df.columns and not pd.api.types.is_numeric_dtype(df["status"]):
        df["status"] = df["status"].astype(str).str.strip().map(
            {"Developing": 0, "Developed": 1}
        )
    cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    return df[cols].copy()


def clean_data(df):
    needed = FEATURES + [TARGET]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    out = df[needed].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=[TARGET])
    if len(out) < 100:
        raise ValueError("need at least 100 rows")
    return out


def make_pipeline(model):
    # impute + standardize then the model
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def score(model, X, y):
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
    }


def fit_models(df):
    df = clean_data(df)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # tune SGD a bit
    sgd = GridSearchCV(
        make_pipeline(
            SGDRegressor(
                max_iter=3000,
                tol=1e-4,
                random_state=RANDOM_STATE,
                early_stopping=True,
            )
        ),
        {
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__eta0": [0.001, 0.01],
            "model__penalty": ["l2", "elasticnet"],
        },
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )
    sgd.fit(X_train, y_train)

    models = {
        "SGD Linear Regression": sgd.best_estimator_,
        "Ordinary Linear Regression": make_pipeline(LinearRegression()),
        "Decision Tree": make_pipeline(
            DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE)
        ),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }

    metrics = {}
    for name, model in models.items():
        if name != "SGD Linear Regression":
            model.fit(X_train, y_train)
        train_s = score(model, X_train, y_train)
        test_s = score(model, X_test, y_test)
        metrics[name] = {
            "train_rmse": train_s["rmse"],
            "test_rmse": test_s["rmse"],
            "test_mae": test_s["mae"],
            "test_r2": test_s["r2"],
        }

    return models, metrics, (X_train, X_test, y_train, y_test)


def sgd_loss_curve(X_train, X_test, y_train, y_test, epochs=100):
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )
    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_test)
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        eta0=0.01,
        random_state=RANDOM_STATE,
        warm_start=True,
    )
    rows = []
    for epoch in range(1, epochs + 1):
        model.partial_fit(Xt, y_train)
        rows.append(
            {
                "epoch": epoch,
                "train_rmse": mean_squared_error(y_train, model.predict(Xt)) ** 0.5,
                "test_rmse": mean_squared_error(y_test, model.predict(Xv)) ** 0.5,
            }
        )
    return pd.DataFrame(rows)


def train_and_save(data_path, model_path, metrics_path=None, plots_dir=None):
    df = load_data(data_path)
    models, metrics, split = fit_models(df)
    best_name = min(metrics, key=lambda n: metrics[n]["test_rmse"])
    best = models[best_name]

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best,
            "model_name": best_name,
            "features": FEATURES,
            "metrics": metrics[best_name],
        },
        model_path,
        compress=3,
    )

    if metrics_path:
        Path(metrics_path).write_text(
            json.dumps({"best_model": best_name, "models": metrics}, indent=2)
        )

    if plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        clean = clean_data(df)
        X_train, X_test, y_train, y_test = split
        history = sgd_loss_curve(X_train, X_test, y_train, y_test)

        # correlation heatmap
        plt.figure(figsize=(9, 7))
        cols = [
            "life_expectancy",
            "adult_mortality",
            "bmi",
            "polio",
            "hiv_aids",
            "gdp",
            "income_composition",
            "schooling",
        ]
        sns.heatmap(clean[cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Correlation heatmap")
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_heatmap.png", dpi=140)
        plt.close()

        # distributions
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(clean["life_expectancy"], bins=30, ax=axes[0])
        axes[0].set_title("Life expectancy distribution")
        sns.scatterplot(
            data=clean.sample(min(800, len(clean)), random_state=42),
            x="schooling",
            y="life_expectancy",
            ax=axes[1],
            alpha=0.4,
        )
        axes[1].set_title("Schooling vs life expectancy")
        fig.tight_layout()
        fig.savefig(plots_dir / "distributions.png", dpi=140)
        plt.close(fig)

        # loss curve
        plt.figure(figsize=(8, 4))
        plt.plot(history["epoch"], history["train_rmse"], label="train")
        plt.plot(history["epoch"], history["test_rmse"], label="test")
        plt.xlabel("epoch")
        plt.ylabel("RMSE")
        plt.title("SGD loss curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "sgd_loss_curve.png", dpi=140)
        plt.close()

        # before / after line on schooling
        sample = clean.dropna(subset=["schooling"]).sample(min(1000, len(clean)), random_state=42)
        x = sample[["schooling"]].to_numpy()
        y = sample["life_expectancy"].to_numpy()
        line = LinearRegression().fit(x, y)
        xs = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(x, y, alpha=0.25, s=12)
        axes[0].set_title("Before")
        axes[1].scatter(x, y, alpha=0.25, s=12)
        axes[1].plot(xs, line.predict(xs), color="red", linewidth=2)
        axes[1].set_title("After (linear fit)")
        for ax in axes:
            ax.set_xlabel("schooling")
            ax.set_ylabel("life expectancy")
        fig.tight_layout()
        fig.savefig(plots_dir / "before_after_regression_line.png", dpi=140)
        plt.close(fig)

    return best, metrics


# old name used in API
RAW_FEATURES = FEATURES
