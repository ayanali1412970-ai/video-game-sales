"""End-to-end steps shared by analysis.py (cell by cell) and main.py (headless)."""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from . import data, features, models
from .config import HIT_THRESHOLD, OUT_DIR, TEST_YEARS


def prepare():
    """Load -> features -> modelling frame -> (train_pool, test, X/y views)."""
    full = features.add_features(data.load_clean())
    frame = data.modelling_frame(full)
    train, test = data.time_split(frame)
    return full, frame, train, test


def views(df):
    return df[features.FEATURE_COLS], df["y_log"], (df["global_sales"] >= HIT_THRESHOLD).astype(int)


# ------------------------------------------------------------------------------------------
def run_regression(train, test):
    X, y, _ = views(train); Xt, yt, _ = views(test)
    folds = models.year_folds(train["release_year"])
    cands = models.regression_candidates()
    searches = models.tune_regressors(X, y, folds, cands)

    rows, fitted = [], {}
    for name, gs in searches.items():
        res = pd.DataFrame(gs.cv_results_)
        best = res.loc[res["rank_test_score"].idxmin()]
        params = best["params"]
        pipe = models.build_final(cands[name][0], params).fit(X, y)           # refit on ALL train years
        pred = pipe.predict(Xt)
        fitted[name] = pipe
        rows.append({"model": name, "cv_RMSE": -best["mean_test_score"], "cv_RMSE_std": best["std_test_score"],
                     **{f"test_{k}": v for k, v in models.regression_metrics(yt, pred).items()},
                     "best_params": {k.replace("model__", ""): (int(v) if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()) else round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                                     for k, v in params.items()}})
    table = pd.DataFrame(rows)
    # model selection uses ONLY cross-validation, never the test years
    linear = table[table["model"].isin(["Linear Regression", "Ridge", "Lasso", "Elastic Net"])]
    best_name = linear.loc[linear["cv_RMSE"].idxmin(), "model"]
    return {"table": table, "fitted": fitted, "best_name": best_name,
            "pred_test": fitted[best_name].predict(Xt), "folds": folds}


# ------------------------------------------------------------------------------------------
def run_classification(train, test, reg):
    X, _, y = views(train); Xt, _, yt = views(test)
    folds = models.year_folds(train["release_year"])
    from sklearn.linear_model import LogisticRegression
    gs = models.tune_logistic(X, y, folds)
    res = pd.DataFrame(gs.cv_results_); best = res.loc[res["rank_test_score"].idxmin()]
    params = best["params"]
    template = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
    pipe_cv = models.build_final(template, params)

    # decision threshold from out-of-fold predictions (walk-forward), not from the test set
    idx, oof = models.oof_probabilities(pipe_cv, X, y, folds)
    thr, grid, f1s = models.best_f1_threshold(y.iloc[idx].to_numpy(), oof)

    final = clone(pipe_cv).fit(X, y)
    prob = final.predict_proba(Xt)[:, 1]
    base = models.baseline_classifier(X, y).predict_proba(Xt)[:, 1]

    metrics = models.classification_metrics(yt.to_numpy(), prob, thr)
    metrics_default = models.classification_metrics(yt.to_numpy(), prob, 0.5)
    # cross-check: can the LINEAR REGRESSION score rank hits too?
    reg_auc = roc_auc_score(yt, reg["pred_test"])
    return {"final": final, "prob": prob, "baseline_prob": base, "threshold": thr, "metrics": metrics,
            "metrics_at_0.5": metrics_default, "cv_PR_AUC": float(best["mean_test_score"]),
            "cv_PR_AUC_std": float(best["std_test_score"]), "params": {k.replace("model__", ""): (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in params.items()},
            "oof": (idx, oof), "thr_curve": (grid, f1s), "reg_score_auc": float(reg_auc)}


# ------------------------------------------------------------------------------------------
def coefficient_table(fitted_pipe, top=12):
    names = models.feature_names(fitted_pipe)
    coef = fitted_pipe.named_steps["model"].coef_.ravel()
    t = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coef": coef})
    return t.reindex(t["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def decile_lift(y_true_sales, score, n=10):
    d = pd.DataFrame({"sales": np.asarray(y_true_sales), "score": np.asarray(score)})
    d["decile"] = pd.qcut(d["score"].rank(method="first"), n, labels=False) + 1
    return d.groupby("decile")["sales"].mean().reset_index()


def to_jsonable(o):
    if isinstance(o, dict): return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, float): return o
    return o


def save_metrics(reg, clf, path=OUT_DIR / "metrics.json"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"test_years": list(TEST_YEARS), "hit_threshold_millions": HIT_THRESHOLD,
               "regression": {"selected_model": reg["best_name"],
                              "table": reg["table"].to_dict(orient="records")},
               "classification": {"metrics_tuned_threshold": clf["metrics"], "metrics_at_0.5": clf["metrics_at_0.5"],
                                  "cv_PR_AUC": clf["cv_PR_AUC"], "best_params": clf["params"],
                                  "regression_score_ROC_AUC": clf["reg_score_auc"]}}
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str))
    reg["table"].drop(columns=["best_params"]).round(4).to_csv(OUT_DIR / "regression_results.csv", index=False)
    return payload
