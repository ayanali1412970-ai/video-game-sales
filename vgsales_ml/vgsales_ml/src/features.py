"""Leakage-free feature engineering.

Rule: a feature for a game released in year Y may only use information available BEFORE Y
(platform launch dates, the title itself, and sales of games from years < Y).
Regional sales columns are never used as inputs: they sum to the target.
"""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["platform_code", "genre", "manufacturer"]
PUB_COL = ["publisher"]
NUM_COLS = ["platform_age", "platform_age_sq", "n_platforms_same_year", "title_has_number",
            "title_n_words", "pub_prior_mean", "pub_prior_n_log", "genre_prior_mean"]
FEATURE_COLS = CAT_COLS + PUB_COL + NUM_COLS

_SEQUEL = re.compile(r"\d|\b[IVX]{2,}\b")


def _prior_mean(df, group_col, y_col, smoothing):
    """Smoothed mean of y over rows of the same group in STRICTLY EARLIER release years."""
    g = df.groupby([group_col, "release_year"])[y_col].agg(["sum", "count"]).reset_index()
    g = g.sort_values([group_col, "release_year"])
    g["s_before"] = g.groupby(group_col)["sum"].cumsum() - g["sum"]
    g["n_before"] = g.groupby(group_col)["count"].cumsum() - g["count"]
    yr = df.groupby("release_year")[y_col].agg(["sum", "count"]).sort_index()
    yr["gm_before"] = (yr["sum"].cumsum() - yr["sum"]) / (yr["count"].cumsum() - yr["count"]).replace(0, np.nan)
    g = g.merge(yr[["gm_before"]], left_on="release_year", right_index=True)
    g["gm_before"] = g["gm_before"].fillna(0.0)
    g["prior_mean"] = (g["s_before"] + smoothing * g["gm_before"]) / (g["n_before"] + smoothing)
    out = df[[group_col, "release_year"]].merge(g[[group_col, "release_year", "prior_mean", "n_before"]],
                                                on=[group_col, "release_year"], how="left")
    return out["prior_mean"].to_numpy(), out["n_before"].to_numpy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with targets and engineered columns. Call on the FULL table (all years), then filter."""
    d = df.copy()
    d["y_log"] = np.log1p(d["global_sales"])                       # regression target (skewed -> log scale)

    # platform life-cycle position (PC has no console life-cycle -> age 0)
    age = (d["release_year"] - d["launch_year"]).clip(lower=0)
    d["platform_age"] = np.where(d["platform_code"] == "PC", 0, age)
    d["platform_age_sq"] = d["platform_age"] ** 2

    # simultaneous multi-platform launch (known at release time)
    d["n_platforms_same_year"] = d.groupby(["name", "release_year"])["platform_code"].transform("nunique")

    # franchise / sequel proxies from the title
    d["title_has_number"] = d["name"].str.contains(_SEQUEL).astype(int)
    d["title_n_words"] = d["name"].str.split().str.len()

    # publisher track record and genre momentum, from past years only
    d["pub_prior_mean"], n_before = _prior_mean(d, "publisher", "y_log", smoothing=5)
    d["pub_prior_n_log"] = np.log1p(n_before)
    d["genre_prior_mean"], _ = _prior_mean(d, "genre", "y_log", smoothing=20)
    return d


def make_preprocessor() -> ColumnTransformer:
    """One-hot + scaling, fitted INSIDE each CV fold via sklearn Pipeline (no leakage)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("pub", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=40, sparse_output=False), PUB_COL),
        ("num", StandardScaler(), NUM_COLS),
    ])
