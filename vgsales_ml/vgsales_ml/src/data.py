"""Loading the cleaned table and splitting it by time (never randomly)."""
import pandas as pd
from .config import DATA_PATH, FIRST_YEAR, LAST_YEAR, TEST_YEARS


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    """Cleaned, normalised vgsales table (see data/processed). Keeps only the years the scrape fully covers."""
    df = pd.read_csv(path)
    df = df[df["in_coverage"] & (df["release_year"] <= LAST_YEAR)].copy()
    df = df[df["global_sales"] > 0]                      # a title with zero recorded sales carries no signal
    return df.reset_index(drop=True)


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for modelling (modern era). Earlier rows were only needed for 'past sales' features."""
    return df[df["release_year"] >= FIRST_YEAR].reset_index(drop=True)


def time_split(df: pd.DataFrame):
    """(train_pool, test): train_pool = FIRST_YEAR..TEST_YEARS[0]-1, test = TEST_YEARS."""
    test_mask = df["release_year"].between(*TEST_YEARS)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
