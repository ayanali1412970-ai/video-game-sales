"""Model definitions, walk-forward tuning and metrics."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import CV_VAL_YEARS, RANDOM_STATE
from .features import FEATURE_COLS, make_preprocessor


# ------------------------------------------------------------------------------------------
# walk-forward validation: validate on year v, train on every year before v
# ------------------------------------------------------------------------------------------
def year_folds(years: pd.Series, val_years=CV_VAL_YEARS):
    years = years.to_numpy()
    return [(np.where(years < v)[0], np.where(years == v)[0]) for v in val_years]


def _pipe(estimator):
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


# ------------------------------------------------------------------------------------------
# regression
# ------------------------------------------------------------------------------------------
def regression_candidates():
    """name -> (estimator, grid). The linear family is the focus; boosting is a non-linear reference."""
    return {
        "Baseline (mean)":      (DummyRegressor(strategy="mean"), {}),
        "Linear Regression":    (LinearRegression(), {}),
        "Ridge":                (Ridge(), {"model__alpha": np.logspace(-1, 3, 25)}),
        "Lasso":                (Lasso(max_iter=20000), {"model__alpha": np.logspace(-4, -0.5, 20)}),
        "Elastic Net":          (ElasticNet(max_iter=20000),
                                 {"model__alpha": np.logspace(-4, -0.5, 10), "model__l1_ratio": [0.2, 0.5, 0.8]}),
        "Gradient Boosting (reference)": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=300),
            {"model__learning_rate": [0.03, 0.08], "model__max_leaf_nodes": [8, 15],
             "model__min_samples_leaf": [30, 80]}),
    }


def tune_regressors(X, y, folds, candidates=None, n_jobs=-1):
    """Grid-search every candidate with walk-forward CV. Returns {name: fitted GridSearchCV}."""
    out = {}
    for name, (est, grid) in (candidates or regression_candidates()).items():
        gs = GridSearchCV(_pipe(est), grid or {}, cv=folds, scoring="neg_root_mean_squared_error",
                          n_jobs=n_jobs, refit=False)
        gs.fit(X, y)
        out[name] = gs
    return out


def regression_metrics(y_true_log, y_pred_log) -> dict:
    return {"RMSE": root_mean_squared_error(y_true_log, y_pred_log),
            "MAE": mean_absolute_error(y_true_log, y_pred_log),
            "R2": r2_score(y_true_log, y_pred_log),
            "Spearman": float(spearmanr(y_true_log, y_pred_log)[0]) if np.std(y_pred_log) > 0 else 0.0}


# ------------------------------------------------------------------------------------------
# classification
# ------------------------------------------------------------------------------------------
def logistic_grid():
    return {"model__C": np.logspace(-2.5, 3, 12), "model__l1_ratio": [0.0, 1.0],
            "model__class_weight": [None, "balanced"]}


def tune_logistic(X, y, folds, n_jobs=-1):
    est = LogisticRegression(solver="liblinear", max_iter=2000, random_state=RANDOM_STATE)
    gs = GridSearchCV(_pipe(est), logistic_grid(), cv=folds, scoring="average_precision",
                      n_jobs=n_jobs, refit=False)
    return gs.fit(X, y)


def oof_probabilities(pipe, X, y, folds):
    """Out-of-fold predicted probabilities over the walk-forward validation years."""
    idx, prob = [], []
    for tr, va in folds:
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        idx.append(va); prob.append(m.predict_proba(X.iloc[va])[:, 1])
    idx = np.concatenate(idx)
    return idx, np.concatenate(prob)


def best_f1_threshold(y_true, prob):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = np.array([f1_score(y_true, prob >= t, zero_division=0) for t in grid])
    return float(grid[f1s.argmax()]), grid, f1s


def classification_metrics(y_true, prob, threshold) -> dict:
    pred = prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    top = np.argsort(-prob)[: max(1, int(0.1 * len(prob)))]
    return {"ROC_AUC": roc_auc_score(y_true, prob), "PR_AUC": average_precision_score(y_true, prob),
            "precision": precision_score(y_true, pred, zero_division=0), "recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred), "threshold": threshold, "prevalence": float(np.mean(y_true)),
            "precision_at_top10pct": float(np.mean(np.asarray(y_true)[top])),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def baseline_classifier(X, y):
    return _pipe(DummyClassifier(strategy="prior")).fit(X, y)


def build_final(estimator_template, params):
    """Unfitted pipeline = template + chosen hyper-parameters."""
    return clone(_pipe(estimator_template)).set_params(**params)


def feature_names(fitted_pipe):
    return fitted_pipe.named_steps["prep"].get_feature_names_out()
