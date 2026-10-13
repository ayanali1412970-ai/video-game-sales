"""
Headless run of the whole project (no notebook needed):

    python3 main.py

Writes charts to outputs/figures/, metrics to outputs/metrics.json and
outputs/regression_results.csv, and test-set predictions to outputs/test_predictions.csv.
For the step-by-step, notebook-style version open analysis.py in VS Code.
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import numpy as np

from src import pipeline, plots
from src.config import OUT_DIR, TEST_YEARS

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("[1/5] Loading data + building features ...")
    full, frame, train, test = pipeline.prepare()
    print(f"      train {len(train):,} games (<= {train.release_year.max()}) | test {len(test):,} games ({TEST_YEARS[0]}-{TEST_YEARS[1]})")

    print("[2/5] EDA charts ...")
    plots.fig_sales_by_year(frame); plots.fig_region_by_genre(frame)
    plots.fig_platform_lifecycle(frame); plots.fig_target_distribution(frame)

    print("[3/5] Linear regression family (walk-forward CV tuning) ...")
    reg = pipeline.run_regression(train, test)
    print(reg["table"].drop(columns="best_params").round(3).to_string(index=False))
    Xt, yt, _ = pipeline.views(test)
    rho = reg["table"].loc[reg["table"].model == reg["best_name"], "test_Spearman"].iloc[0]
    plots.fig_model_comparison(reg["table"]); plots.fig_pred_vs_actual(yt, reg["pred_test"], reg["best_name"], rho)
    plots.fig_decile_lift(pipeline.decile_lift(test["global_sales"], reg["pred_test"]))
    plots.fig_coefficients(pipeline.coefficient_table(reg["fitted"][reg["best_name"]]),
                           f"What the {reg['best_name']} model rewards and penalises")

    print("[4/5] Logistic regression (hit = >= 1M units) ...")
    clf = pipeline.run_classification(train, test, reg)
    m = clf["metrics"]
    print(f"      ROC-AUC {m['ROC_AUC']:.3f} | PR-AUC {m['PR_AUC']:.3f} (chance {m['prevalence']:.3f}) | "
          f"precision {m['precision']:.3f} | recall {m['recall']:.3f} | F1 {m['F1']:.3f}")
    _, _, yc = pipeline.views(test); _, _, ytr = pipeline.views(train)
    idx, oof = clf["oof"]
    plots.fig_roc_pr(yc.to_numpy(), clf["prob"], reg["pred_test"])
    plots.fig_threshold(*clf["thr_curve"], ytr.iloc[idx].to_numpy(), oof, clf["threshold"])
    plots.fig_confusion(m); plots.fig_odds_ratios(pipeline.coefficient_table(clf["final"]))

    print("[5/5] Saving outputs ...")
    out = test[["name", "platform_code", "release_year", "genre", "publisher", "global_sales"]].copy()
    out["predicted_sales_M"] = np.expm1(reg["pred_test"]).clip(min=0).round(3)
    out["hit_probability"] = clf["prob"].round(3); out["is_hit"] = yc.to_numpy()
    out.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    pipeline.save_metrics(reg, clf)
    print(f"\nDone. See {OUT_DIR}")


if __name__ == "__main__":
    main()
