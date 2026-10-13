"""All charts. Each function draws one figure, saves it to outputs/figures and returns it.

Design: one idea per chart, the title states the takeaway, thin marks, recessive grid,
palette = validated categorical order (blue, orange, aqua, yellow) + neutral gray.
"""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

from .config import C, FIG_DIR, HIT_THRESHOLD

SEQ = LinearSegmentedColormap.from_list("seq_blue", [C["surface"], "#bcd5f2", C["blue"], "#143f73"])
PALETTE = [C["blue"], C["orange"], C["aqua"], C["yellow"], "#e87ba4"]


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "text.color": C["ink"], "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "lines.linewidth": 2,
    })


setup_style()


def _title(ax, text, sub=None, width=78):
    ax.set_title("\n".join(textwrap.wrap(text, width)), pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=C["ink2"], va="bottom")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    return fig


def pretty(n):
    fixed = {"pub_prior_mean": "Publisher's past avg sales", "pub_prior_n_log": "Publisher's past # of games",
             "genre_prior_mean": "Genre's past avg sales", "platform_age": "Platform age (yrs)",
             "platform_age_sq": "Platform age squared", "n_platforms_same_year": "# platforms at launch",
             "title_has_number": "Title has number / sequel", "title_n_words": "Title length (words)",
             "publisher_infrequent_sklearn": "Publisher: small / other"}
    if n in fixed:
        return fixed[n]
    for pre, lab in [("platform_code_", "Platform: "), ("genre_", "Genre: "), ("manufacturer_", "Maker: "),
                     ("publisher_", "Publisher: ")]:
        if n.startswith(pre):
            return lab + n[len(pre):]
    return n


# ======================================================= EDA =========================================
def fig_sales_by_year(frame):
    s = frame.groupby("release_year")["global_sales"].sum()
    peak, last = s.idxmax(), s.index.max()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(s.index, s.values, color=C["blue"], width=0.72)
    ax.bar(peak, s[peak], color=C["blue"], width=0.72, edgecolor=C["ink"], linewidth=1.5)
    ax.annotate(f"peak {peak}: {s[peak]:,.0f}M", (peak, s[peak]), xytext=(0, 6), textcoords="offset points",
                ha="center", color=C["ink2"], fontsize=9)
    ax.set_xticks(range(s.index.min(), last + 1, 2)); ax.set_ylabel("Units sold (millions)"); ax.set_ylim(0, s.max() * 1.12)
    ax.grid(axis="x", visible=False)
    _title(ax, f"Industry unit sales peaked in {peak} and fell {1 - s[last] / s[peak]:.0%} by {last}",
           "Lifetime sales by release year (VGChartz estimates, mostly physical retail, so part of the late drop is the shift to digital)")
    return _save(fig, "01_sales_by_year")


def fig_region_by_genre(frame):
    cols = ["north_america_sales", "europe_sales", "japan_sales", "other_sales"]
    names = ["North America", "Europe", "Japan", "Other"]
    g = frame.groupby("genre")[cols].sum()
    g = g.loc[g.sum(axis=1).sort_values().index]
    na_share = g[cols[0]].sum() / g.to_numpy().sum()
    jp_top = (g[cols[2]] / g.sum(axis=1)).idxmax()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    left = np.zeros(len(g))
    for col, nm, colr in zip(cols, names, [C["blue"], C["orange"], C["aqua"], C["gray"]]):
        ax.barh(g.index, g[col], left=left, color=colr, label=nm, height=0.7, edgecolor=C["surface"], linewidth=2)
        left += g[col].to_numpy()
    ax.set_xlabel("Units sold (millions)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9.5)
    _title(ax, f"North America buys {na_share:.0%} of all units; Japan's share is highest in {jp_top}",
           None)
    return _save(fig, "02_region_by_genre")


def fig_platform_lifecycle(frame, top=5):
    tops = frame.groupby("platform_code")["global_sales"].sum().nlargest(top).index
    d = frame[frame["platform_code"].isin(tops)].groupby(["platform_code", "platform_age"])["global_sales"].sum().reset_index()
    peaks = d.loc[d.groupby("platform_code")["global_sales"].idxmax()]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for plat, colr in zip(tops, PALETTE):
        s = d[d["platform_code"] == plat]
        ax.plot(s["platform_age"], s["global_sales"], marker="o", ms=4, color=colr, label=plat)
    ax.set_xlabel("Years since platform launch"); ax.set_ylabel("Units sold (millions)")
    ax.set_xlim(-0.3, d["platform_age"].max() + 0.5); ax.legend(loc="upper right")
    _title(ax, f"Consoles follow one arc: sales peak about {peaks['platform_age'].median():.0f} years after launch",
           f"Top {top} platforms by total sales; this is why 'platform age' is a model feature")
    return _save(fig, "03_platform_lifecycle")


def fig_target_distribution(frame):
    s = frame["global_sales"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 45)
    ax.hist(s, bins=bins, color=C["blue"], edgecolor=C["surface"], linewidth=1)
    ax.set_xscale("log"); ax.axvline(HIT_THRESHOLD, color=C["orange"], lw=2, ls="--")
    ax.text(HIT_THRESHOLD * 1.08, ax.get_ylim()[1] * 0.9, f"'hit' = at least {HIT_THRESHOLD:g}M units\n"
            f"{(s >= HIT_THRESHOLD).mean():.0%} of titles", color=C["ink"], fontsize=9, va="top")
    ax.set_xlabel("Lifetime global sales per title (millions, log scale)"); ax.set_ylabel("Number of titles")
    ax.grid(axis="x", visible=False)
    _title(ax, f"Sales are extremely skewed: median title sells {s.median():.2f}M, a few sell 50M+",
           "Why the regression target is log(1 + sales) and the classifier uses a hit threshold")
    return _save(fig, "04_target_distribution")


# ======================================================= regression ===================================
def fig_model_comparison(table):
    t = table.copy()
    base = t.loc[t["model"] == "Baseline (mean)", "test_RMSE"].iloc[0]
    best = t[t["model"] != "Baseline (mean)"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(t))[::-1]; h = 0.36
    ax.barh(y + h / 2, t["cv_RMSE"], height=h, color=C["blue"], label="Walk-forward CV (2009-2013)")
    ax.barh(y - h / 2, t["test_RMSE"], height=h, color=C["orange"], label="Held-out test (2014-2015)")
    for yy, v in zip(y - h / 2, t["test_RMSE"]):
        ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=8.5, color=C["ink2"])
    ax.set_yticks(y); ax.set_yticklabels(t["model"]); ax.set_xlabel("RMSE on log(1 + sales)  (lower is better)")
    ax.set_xlim(0, t[["cv_RMSE", "test_RMSE"]].to_numpy().max() * 1.12); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    lin = t[t["model"].isin(["Linear Regression", "Ridge", "Lasso", "Elastic Net"])]
    _title(ax, f"Linear models cut test error {1 - lin['test_RMSE'].min() / base:.0%} vs the baseline, "
               f"within {abs(lin['test_RMSE'].min() - t[t['model'].str.startswith('Gradient')]['test_RMSE'].iloc[0]):.3f} of gradient boosting")
    return _save(fig, "05_model_comparison")


def fig_pred_vs_actual(y_true_log, pred_log, model_name, rho):
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.scatter(pred_log, y_true_log, s=14, color=C["blue"], alpha=0.35, linewidths=0)
    lim = [0, max(np.max(y_true_log), np.max(pred_log)) * 1.03]
    ax.plot(lim, lim, color=C["orange"], ls="--", lw=1.6, label="perfect prediction")
    ax.set_xlabel(f"Predicted log(1 + sales)  [{model_name}]"); ax.set_ylabel("Actual log(1 + sales)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.legend(loc="upper left")
    _title(ax, f"Ranking is good (Spearman {rho:.2f}); blockbusters are still underestimated",
           "Each dot is one game from 2014-2015", width=60)
    return _save(fig, "06_pred_vs_actual")


def fig_decile_lift(lift):
    overall = lift["sales"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(lift["decile"], lift["sales"], color=C["blue"], width=0.72)
    ax.axhline(overall, color=C["orange"], ls="--", lw=1.6)
    ax.text(0.55, overall, f" overall mean {overall:.2f}M", color=C["ink2"], fontsize=9, va="bottom")
    ax.set_xticks(lift["decile"]); ax.set_xlabel("Predicted-sales decile (1 = lowest, 10 = highest), test years")
    ax.set_ylabel("Actual mean sales (millions)"); ax.grid(axis="x", visible=False)
    top, bot = lift["sales"].iloc[-1], lift["sales"].iloc[0]
    _title(ax, f"Top predicted decile sells {top:.2f}M on average, {top / bot:.0f}x the bottom decile",
           "Out-of-sample: games from 2014-2015 that the model never saw")
    return _save(fig, "07_decile_lift")


def fig_coefficients(coef, title, n=9):
    c = coef.assign(label=coef["feature"].map(pretty))
    top = pd.concat([c.nlargest(n, "coef"), c.nsmallest(n, "coef")]).sort_values("coef")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(top["label"], top["coef"], color=np.where(top["coef"] > 0, C["blue"], C["orange"]), height=0.7)
    ax.axvline(0, color=C["ink2"], lw=1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Coefficient on log(1 + sales)  (numeric features standardised)")
    _title(ax, title, "Blue raises predicted sales, orange lowers them")
    return _save(fig, "08_linear_coefficients")


# ======================================================= classification ===============================
def fig_roc_pr(y, p_logit, p_reg):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for p, lab, col in [(p_logit, "Logistic regression", C["blue"]), (p_reg, "Linear regression score", C["orange"])]:
        fpr, tpr, _ = roc_curve(y, p); a1.plot(fpr, tpr, color=col, label=f"{lab} (AUC {roc_auc_score(y, p):.2f})")
        pr, rc, _ = precision_recall_curve(y, p); a2.plot(rc, pr, color=col, label=f"{lab} (AP {average_precision_score(y, p):.2f})")
    a1.plot([0, 1], [0, 1], color=C["gray"], ls="--", lw=1.2, label="chance")
    a2.axhline(np.mean(y), color=C["gray"], ls="--", lw=1.2, label=f"chance ({np.mean(y):.2f})")
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.legend(loc="lower right", fontsize=8.5)
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_ylim(0, 1.02); a2.legend(loc="upper right", fontsize=8.5)
    a1.set_title(f"ROC: hits are separated well (AUC {roc_auc_score(y, p_logit):.2f})", loc="left", fontsize=11)
    a2.set_title(f"Precision-recall: {average_precision_score(y, p_logit) / np.mean(y):.1f}x better than chance", loc="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, "09_roc_pr")


def fig_threshold(grid, f1s, y_oof, p_oof, thr):
    prec = [((p_oof >= t) & (y_oof == 1)).sum() / max((p_oof >= t).sum(), 1) for t in grid]
    rec = [((p_oof >= t) & (y_oof == 1)).sum() / max((y_oof == 1).sum(), 1) for t in grid]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(grid, f1s, color=C["blue"], label="F1"); ax.plot(grid, prec, color=C["orange"], label="Precision")
    ax.plot(grid, rec, color=C["aqua"], label="Recall"); ax.axvline(thr, color=C["ink2"], ls=":", lw=1.5)
    ax.text(thr + 0.01, 0.97, f"chosen: {thr:.2f}", fontsize=9, color=C["ink2"], va="top")
    ax.set_xlabel("Probability threshold for calling a game a 'hit'"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.02)
    ax.legend(loc="center right")
    _title(ax, f"A {thr:.2f} threshold (not the default 0.50) maximises F1 on out-of-fold data",
           "Chosen on walk-forward validation years only, never on the test set")
    return _save(fig, "10_threshold_tuning")


def fig_confusion(m):
    c = m["confusion"]; cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.0%} of row)", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else C["ink"], fontsize=10.5)
    ax.set_xticks([0, 1], ["Predicted: not hit", "Predicted: hit"]); ax.set_yticks([0, 1], ["Actual: not hit", "Actual: hit"])
    _title(ax, f"Catches {m['recall']:.0%} of real hits, at {m['precision']:.0%} precision", None, width=50)
    return _save(fig, "11_confusion_matrix")


def fig_odds_ratios(coef, n=8):
    c = coef.assign(label=coef["feature"].map(pretty), odds=np.exp(coef["coef"]))
    top = pd.concat([c.nlargest(n, "coef"), c.nsmallest(n, "coef")]).drop_duplicates("feature").sort_values("odds")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    col = np.where(top["odds"] > 1, C["blue"], C["orange"])
    ax.hlines(top["label"], 1, top["odds"], color=col, lw=2); ax.scatter(top["odds"], top["label"], color=col, s=42, zorder=3)
    ax.axvline(1, color=C["ink2"], lw=1); ax.set_xscale("log"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio for a hit (log scale; 1 = no effect; numeric features per +1 SD)")
    _title(ax, "What raises or lowers the odds of selling at least 1M units", "Logistic regression coefficients, exponentiated")
    return _save(fig, "12_odds_ratios")
