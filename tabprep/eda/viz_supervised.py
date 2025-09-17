"""
Supervised Learning Visualization Helpers
-----------------------------------------
Functions take X and y as input (plus optional model) and produce
useful, fast visualizations for regression and classification.

Dependencies:
    - numpy, pandas
    - matplotlib
    - seaborn
    - scikit-learn (>= 1.1 recommended)

Author: You :)
"""

from __future__ import annotations
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import is_classifier, clone
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    r2_score,
    mean_squared_error,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    # available in sklearn >= 0.24
    from sklearn.inspection import PartialDependenceDisplay
    _HAS_PDP = True
except Exception:
    _HAS_PDP = False


# -----------------------------
# Utilities
# -----------------------------
def _to_dataframe(X) -> pd.DataFrame:
    """Ensure X is a pandas DataFrame with reasonable column names."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    cols = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


def _to_series(y) -> pd.Series:
    """Ensure y is a pandas Series."""
    if isinstance(y, pd.Series):
        return y.copy()
    return pd.Series(np.asarray(y), name="target")


def infer_task_type(y: Union[pd.Series, np.ndarray]) -> str:
    """
    Infer 'classification' or 'regression' from y.
    Heuristics:
      - if dtype is object/bool/category -> classification
      - if number of unique values <= 20 and integer-like -> classification
      - else regression
    """
    ys = _to_series(y)
    if pd.api.types.is_object_dtype(ys) or pd.api.types.is_bool_dtype(ys) or pd.api.types.is_categorical_dtype(ys):
        return "classification"
    unique_count = ys.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(ys) and unique_count <= 20:
        return "classification"
    return "regression"


def get_default_model(task: str):
    """Return a reasonable default model for the task."""
    if task == "classification":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


def _build_basic_pipeline(X: pd.DataFrame, base_model):
    """Numerical+categorical preprocessing with imputation, then model."""
    X = _to_dataframe(X)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return Pipeline([("pre", pre), ("model", base_model)])


def _encode_if_needed(y: pd.Series) -> Tuple[pd.Series, Optional[LabelEncoder]]:
    """For classification, encode string labels for metrics that require numeric arrays."""
    if infer_task_type(y) == "classification" and (pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y)):
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y), index=y.index, name=y.name), le
    return y, None


# -----------------------------
# Target-focused plots
# -----------------------------
def plot_target_distribution(y):
    """Histogram for regression targets or class balance for classification."""
    ys = _to_series(y)
    task = infer_task_type(ys)
    plt.figure(figsize=(7, 4))
    if task == "regression":
        sns.histplot(ys.dropna(), bins=30, edgecolor="black")
        plt.title("Target Distribution (Regression)")
        plt.xlabel(ys.name or "target")
    else:
        vc = ys.value_counts(dropna=False).sort_index()
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.title("Class Balance")
        plt.xlabel("Class")
        plt.ylabel("Count")
        for i, v in enumerate(vc.values):
            plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()


# -----------------------------
# Feature ↔ target relationships
# -----------------------------
def plot_top_feature_relationships(X, y, top_k: int = 6):
    """
    For regression: pick features with strongest |corr| with y (numeric only) and scatter with trend.
    For classification: for numeric features, violin/box per class for top_k by ANOVA F-like variance ratio.
    """
    X = _to_dataframe(X)
    ys = _to_series(y)
    task = infer_task_type(ys)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # choose top features
    if task == "regression":
        # correlation with numeric features
        corr = {c: np.corrcoef(X[c].dropna().align(ys, join="inner")[0], ys.dropna().align(X[c], join="inner")[0])[0, 1]
                for c in num_cols if X[c].nunique() > 1}
        corr = {k: v for k, v in corr.items() if np.isfinite(v)}
        top = sorted(corr, key=lambda c: abs(corr[c]), reverse=True)[:top_k]
        n = len(top)
        ncols = min(3, n or 1)
        nrows = int(np.ceil((n or 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), squeeze=False)
        for i, col in enumerate(top):
            ax = axes[i // ncols, i % ncols]
            sns.scatterplot(x=X[col], y=ys, alpha=0.6, s=12, ax=ax, edgecolor=None)
            sns.regplot(x=X[col], y=ys, scatter=False, ax=ax, ci=None, truncate=True, line_kws={"linewidth": 2})
            ax.set_title(f"{col} vs target (r={corr[col]:.2f})")
            ax.set_xlabel(col); ax.set_ylabel(ys.name or "target")
        # hide empty axes
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        fig.suptitle("Top Feature Relationships (Regression)", y=1.02)
        plt.tight_layout()

    else:
        # classification: numeric features — variance ratio approximation
        # compute between-class std / within-class std
        scores = {}
        for c in num_cols:
            x = X[c]
            if x.nunique() <= 1:
                continue
            grouped = x.groupby(ys).agg(["mean", "var", "count"])
            if grouped["var"].mean() == 0 or grouped["count"].min() <= 1:
                continue
            between = grouped["mean"].var()
            within = grouped["var"].mean()
            if within > 0 and np.isfinite(between / within):
                scores[c] = between / within
        top = sorted(scores, key=scores.get, reverse=True)[:top_k]

        n = len(top)
        ncols = min(3, n or 1)
        nrows = int(np.ceil((n or 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), squeeze=False)
        for i, col in enumerate(top):
            ax = axes[i // ncols, i % ncols]
            sns.violinplot(x=ys.astype(str), y=X[col], ax=ax, cut=0, inner="quartile")
            ax.set_title(f"{col} by Class")
            ax.set_xlabel("Class"); ax.set_ylabel(col)
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        fig.suptitle("Top Numeric Feature Distributions by Class", y=1.02)
        plt.tight_layout()

        # Optional: categorical features — target rate
        if len(cat_cols) > 0:
            plt.figure(figsize=(6, 0.6*len(cat_cols) + 2))
            rates = []
            for c in cat_cols:
                tmp = pd.crosstab(X[c], ys, normalize="index")
                # use max class rate as separability proxy
                rates.append((c, tmp.max(axis=1).mean()))
            if rates:
                cats_sorted = [c for c, _ in sorted(rates, key=lambda t: t[1], reverse=True)[:top_k]]
                for c in cats_sorted:
                    tmp = pd.crosstab(X[c], ys, normalize="index")
                    tmp.plot(kind="bar", stacked=True)
                    plt.title(f"Class Proportions by {c}")
                    plt.xlabel(c); plt.ylabel("Proportion")
                    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.tight_layout()


# -----------------------------
# Model-based diagnostics
# -----------------------------
def plot_permutation_importance_bars(
    X, y, model=None, n_repeats: int = 8, random_state: int = 42, max_features: int = 20, test_size: float = 0.25
):
    """
    Fit (or use provided) model, compute permutation importance on holdout, and plot bar chart.
    Works for both regression & classification.
    """
    X = _to_dataframe(X)
    ys = _to_series(y)
    task = infer_task_type(ys)

    base = model or get_default_model(task)
    pipe = _build_basic_pipeline(X, base)

    Xtr, Xte, ytr, yte = train_test_split(X, ys, test_size=test_size, random_state=random_state, stratify=ys if task == "classification" else None)
    pipe.fit(Xtr, ytr)
    result = permutation_importance(pipe, Xte, yte, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    # Map back feature names from ColumnTransformer
    # After the preprocessor, features are in the order of num+cat cols; we can approximate with original names
    feature_names = []
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    feature_names.extend(num_cols)
    feature_names.extend(cat_cols)

    importances = pd.Series(result.importances_mean[: len(feature_names)], index=feature_names).sort_values(ascending=True).tail(max_features)

    plt.figure(figsize=(8, 0.35*len(importances) + 2))
    plt.barh(importances.index, importances.values)
    plt.title("Permutation Importance (holdout)")
    plt.xlabel("Mean Importance (decrease in score)")
    plt.tight_layout()
    return pipe  # return fitted pipeline for downstream plots if desired


def plot_learning_curve(
    X, y, model=None, cv: int = 5, train_sizes: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0), random_state: int = 42
):
    """
    Plot learning curve (train & cross-val score vs. training size).
    """
    X = _to_dataframe(X)
    ys = _to_series(y)
    task = infer_task_type(ys)
    base = model or (LogisticRegression(max_iter=200) if task == "classification" else Ridge())

    cv_split = StratifiedKFold(cv, shuffle=True, random_state=random_state) if task == "classification" else KFold(cv, shuffle=True, random_state=random_state)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        base, X, ys, train_sizes=train_sizes, cv=cv_split, n_jobs=-1
    )
    plt.figure(figsize=(7, 4))
    plt.plot(train_sizes_abs, train_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(train_sizes_abs, test_scores.mean(axis=1), marker="o", label="CV")
    plt.fill_between(train_sizes_abs, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    plt.fill_between(train_sizes_abs, test_scores.mean(axis=1) - test_scores.std(axis=1),
                     test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.15)
    plt.title("Learning Curve")
    plt.xlabel("Training Samples"); plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()


def plot_confusion_matrix_cv(X, y, model=None, cv: int = 5, random_state: int = 42):
    """
    For classification: cross-val predicted labels → confusion matrix on full data.
    """
    X = _to_dataframe(X); ys = _to_series(y)
    if infer_task_type(ys) != "classification":
        warnings.warn("plot_confusion_matrix_cv is for classification only.")
        return
    base = model or LogisticRegression(max_iter=200)
    cv_split = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(base, X, ys, cv=cv_split, n_jobs=-1)
    disp = ConfusionMatrixDisplay.from_predictions(ys, y_pred, cmap="Blues")
    disp.ax_.set_title("Confusion Matrix (CV predictions)")
    plt.tight_layout()


def plot_roc_pr_curves(X, y, model=None, cv: int = 5, random_state: int = 42):
    """
    For binary classification: ROC & PR curves using cross-val probabilities.
    """
    X = _to_dataframe(X); ys = _to_series(y)
    if infer_task_type(ys) != "classification":
        warnings.warn("plot_roc_pr_curves is for classification only.")
        return

    # Encode if labels are not 0/1
    ys_enc, le = _encode_if_needed(ys)
    base = model or LogisticRegression(max_iter=200)
    cv_split = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(base, X, ys_enc, cv=cv_split, method="predict_proba", n_jobs=-1)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(ys_enc, y_proba, ax=axes[0])
    axes[0].set_title("ROC Curve (CV probabilities)")
    PrecisionRecallDisplay.from_predictions(ys_enc, y_proba, ax=axes[1])
    axes[1].set_title("Precision–Recall Curve (CV probabilities)")
    plt.tight_layout()


def plot_calibration_curve(X, y, model=None, cv: int = 5, bins: int = 10, random_state: int = 42):
    """
    For binary classification: reliability (calibration) curve with cross-val probabilities.
    """
    from sklearn.calibration import calibration_curve

    X = _to_dataframe(X); ys = _to_series(y)
    if infer_task_type(ys) != "classification":
        warnings.warn("plot_calibration_curve is for classification only.")
        return
    ys_enc, _ = _encode_if_needed(ys)
    base = model or LogisticRegression(max_iter=200)
    cv_split = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(base, X, ys_enc, cv=cv_split, method="predict_proba", n_jobs=-1)[:, 1]

    prob_true, prob_pred = calibration_curve(ys_enc, y_proba, n_bins=bins, strategy="quantile")
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.title("Calibration Curve")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.tight_layout()


def plot_regression_diagnostics(X, y, model=None, test_size: float = 0.25, random_state: int = 42):
    """
    Regression diagnostics: y vs y_pred and residuals vs y_pred on a holdout set.
    """
    X = _to_dataframe(X); ys = _to_series(y)
    if infer_task_type(ys) != "regression":
        warnings.warn("plot_regression_diagnostics is for regression only.")
        return
    base = model or Ridge()
    Xtr, Xte, ytr, yte = train_test_split(X, ys, test_size=test_size, random_state=random_state)
    base.fit(Xtr, ytr)
    y_pred = base.predict(Xte)
    r2 = r2_score(yte, y_pred)
    rmse = mean_squared_error(yte, y_pred, squared=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(yte, y_pred, s=16, alpha=0.7, rasterized=True)
    axes[0].plot([yte.min(), yte.max()], [yte.min(), yte.max()], "--")
    axes[0].set_title(f"y vs y_pred (R²={r2:.3f}, RMSE={rmse:.3f})")
    axes[0].set_xlabel("True"); axes[0].set_ylabel("Predicted")

    residuals = yte - y_pred
    axes[1].scatter(y_pred, residuals, s=16, alpha=0.7, rasterized=True)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")

    plt.tight_layout()


def plot_partial_dependence_topk(X, y, model=None, top_k: int = 4, random_state: int = 42):
    """
    Partial dependence on top_k features by permutation importance.
    Falls back silently if sklearn PDP is unavailable.
    """
    if not _HAS_PDP:
        warnings.warn("sklearn PartialDependenceDisplay not available; skipping PDP.")
        return

    X = _to_dataframe(X); ys = _to_series(y)
    task = infer_task_type(ys)

    # fit model via permutation importance helper to pick top features
    pipe = plot_permutation_importance_bars(X, ys, model=model, random_state=random_state)
    plt.show()  # ensure the importance figure renders before PDP

    # pick top_k features from numeric + categorical names list
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    feature_names = num_cols + cat_cols

    # Simple heuristic: use permutation importances if available from prior plot
    try:
        result = permutation_importance(pipe, X, ys, n_repeats=5, random_state=random_state, n_jobs=-1)
        importances = pd.Series(result.importances_mean[: len(feature_names)], index=feature_names)
        top_feats = importances.sort_values(ascending=False).head(top_k).index.tolist()
    except Exception:
        # fallback: pick first top_k numeric
        top_feats = (num_cols or feature_names)[:top_k]

    fig = plt.figure(figsize=(5 * min(top_k, 3), 4 * int(np.ceil(top_k / 3))))
    PartialDependenceDisplay.from_estimator(pipe, X, features=top_feats, kind="average", fig=fig)
    plt.suptitle("Partial Dependence (top features)", y=1.02)
    plt.tight_layout()


# -----------------------------
# Quick demo (optional)
# -----------------------------
if __name__ == "__main__":
    # Example with a classification dataset
    from sklearn.datasets import load_breast_cancer, fetch_california_housing

    # Classification demo
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    plot_target_distribution(y)
    plt.show()
    plot_top_feature_relationships(X, y, top_k=6)
    plt.show()
    plot_learning_curve(X, y)
    plt.show()
    plot_confusion_matrix_cv(X, y)
    plt.show()
    plot_roc_pr_curves(X, y)
    plt.show()
    _ = plot_permutation_importance_bars(X, y)
    plt.show()
    plot_partial_dependence_topk(X, y, top_k=4)
    plt.show()

    # Regression demo
    h = fetch_california_housing(as_frame=True)
    X, y = h.data, h.target

    plot_target_distribution(y)
    plt.show()
    plot_top_feature_relationships(X, y, top_k=6)
    plt.show()
    plot_learning_curve(X, y)
    plt.show()
    plot_regression_diagnostics(X, y)
    plt.show()
    _ = plot_permutation_importance_bars(X, y)
    plt.show()
    plot_partial_dependence_topk(X, y, top_k=4)
    plt.show()
