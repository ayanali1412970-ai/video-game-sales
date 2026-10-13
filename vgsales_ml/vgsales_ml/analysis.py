# %% [markdown]
# # Video Game Sales: predicting how much a game sells
# **Linear regression** (how many units?) and **logistic regression** (will it be a ≥1M hit?)
# on a cleaned VGChartz dataset, validated the way a real forecast would be: **by time, never randomly**.
#
# *Run in VS Code:* open this file and click **Run Cell** above any `# %%`, or **Run All**
# (needs the Python + Jupyter extensions). Each cell is one step of the story.

# %% 1. Imports and setup
import sys, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

from src import data, features, models, pipeline, plots
from src.config import HIT_THRESHOLD, TEST_YEARS

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

# %% 2. Load the cleaned dataset (see data/processed and reports/data_quality_report.md)
full, frame, train, test = pipeline.prepare()
print(f"all years loaded : {len(full):,} games (used to build 'past sales' features)")
print(f"modelling frame  : {len(frame):,} games, {frame.release_year.min()}-{frame.release_year.max()}")
print(f"train pool       : {len(train):,} games, years <= {train.release_year.max()}")
print(f"test (held out)  : {len(test):,} games, years {TEST_YEARS[0]}-{TEST_YEARS[1]}")
frame[["name", "platform_code", "release_year", "genre", "publisher", "global_sales"]].head()

# %% 3. Exploratory analysis: four charts that motivate the features
plots.fig_sales_by_year(frame); plt.show()
plots.fig_region_by_genre(frame); plt.show()
plots.fig_platform_lifecycle(frame); plt.show()
plots.fig_target_distribution(frame); plt.show()

# %% 4. Features (every one uses only information available BEFORE the release year)
print("Inputs:", features.FEATURE_COLS)
frame[["name", "release_year"] + features.NUM_COLS].sample(8, random_state=1)

# %% [markdown]
# **Why no random split?** A random split lets the model peek at the same publisher/platform in the same
# year as the game it is predicting. Here we use *walk-forward* validation: for each year 2009…2013 the
# model trains on all earlier years and is scored on that year. 2014-2015 stay untouched until the end.

# %% 5. Linear regression family: tune with walk-forward CV, then score on the held-out years
reg = pipeline.run_regression(train, test)
results = reg["table"].drop(columns="best_params").round(3)
print(results.to_string(index=False))
print(f"\nSelected (lowest CV error among linear models): {reg['best_name']}")
print(reg["table"][["model", "best_params"]].to_string(index=False))

# %% 6. Regression charts
Xt, yt, _ = pipeline.views(test)
pred_log = reg["pred_test"]
rho = results.loc[results.model == reg["best_name"], "test_Spearman"].iloc[0]
plots.fig_model_comparison(reg["table"]); plt.show()
plots.fig_pred_vs_actual(yt, pred_log, reg["best_name"], rho); plt.show()
lift = pipeline.decile_lift(test["global_sales"], pred_log)
plots.fig_decile_lift(lift); plt.show()
coef = pipeline.coefficient_table(reg["fitted"][reg["best_name"]])
plots.fig_coefficients(coef, f"What the {reg['best_name']} model rewards and penalises"); plt.show()

# %% 7. Logistic regression: "will this game sell at least 1M units?"
clf = pipeline.run_classification(train, test, reg)
m = clf["metrics"]
print(f"best params          : {clf['params']}")
print(f"CV PR-AUC            : {clf['cv_PR_AUC']:.3f} ± {clf['cv_PR_AUC_std']:.3f}")
print(f"tuned threshold      : {clf['threshold']:.2f}  (from out-of-fold predictions)")
print(f"TEST  ROC-AUC {m['ROC_AUC']:.3f} | PR-AUC {m['PR_AUC']:.3f} (chance {m['prevalence']:.3f})")
print(f"TEST  precision {m['precision']:.3f} | recall {m['recall']:.3f} | F1 {m['F1']:.3f}")
print(f"TEST  precision in top 10% of predictions: {m['precision_at_top10pct']:.3f}")
d = clf["metrics_at_0.5"]; print(f"(default 0.50 threshold would give F1 {d['F1']:.3f}, recall {d['recall']:.3f})")

# %% 8. Classification charts
_, _, yc = pipeline.views(test)
idx, oof = clf["oof"]; _, _, ytr = pipeline.views(train)
plots.fig_roc_pr(yc.to_numpy(), clf["prob"], reg["pred_test"]); plt.show()
plots.fig_threshold(*clf["thr_curve"], ytr.iloc[idx].to_numpy(), oof, clf["threshold"]); plt.show()
plots.fig_confusion(m); plt.show()
logit_coef = pipeline.coefficient_table(clf["final"])
plots.fig_odds_ratios(logit_coef); plt.show()

# %% 9. Head to head, then save everything
print(f"Can the linear-regression score rank hits? ROC-AUC {clf['reg_score_auc']:.3f} "
      f"vs logistic regression {m['ROC_AUC']:.3f}")
out = test[["name", "platform_code", "release_year", "genre", "publisher", "global_sales"]].copy()
out["predicted_sales_M"] = np.expm1(pred_log).clip(min=0).round(3)
out["hit_probability"] = clf["prob"].round(3); out["is_hit"] = yc.to_numpy()
out.to_csv(ROOT / "outputs" / "test_predictions.csv", index=False)
pipeline.save_metrics(reg, clf)
out.sort_values("hit_probability", ascending=False).head(10)
