"""Central settings: paths, time windows, thresholds and the chart palette."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "vgsales_model_table.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

RANDOM_STATE = 42

# --- time windows (the scrape is complete through 2015) --------------------------------
FIRST_YEAR = 1996            # modern era; earlier years are used only to build "past sales" features
LAST_YEAR = 2015
TEST_YEARS = (2014, 2015)    # never touched until the final evaluation
CV_VAL_YEARS = (2009, 2010, 2011, 2012, 2013)   # walk-forward folds: train on years < v, validate on v

# --- classification target --------------------------------------------------------------
HIT_THRESHOLD = 1.0          # a "hit" = at least 1 million units sold worldwide

# --- chart palette (validated categorical order: blue, orange, aqua, yellow) -------------
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         gray="#8a8984", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", surface="#fcfcfb")
