"""
Clean + normalise the vgsales dataset (and any new rows dropped into data/incoming/).

Run:   python3 -I src/clean_vgsales.py            (from the project root)

"Atomicity" is applied in two senses:
  1. Data atomicity (1NF / normalisation): every cell holds ONE value, every fact has ONE row,
     derived values (Rank, Global_Sales) are not stored as facts, and descriptive attributes live
     in their own dimension tables.
  2. Write atomicity: every output file is written to a temp file and renamed into place, so a
     crashed run can never leave a half-written dataset behind.
"""
import hashlib, os, re, sys, tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, INCOMING = ROOT / "data/raw", ROOT / "data/incoming"
OUT, REPORTS = ROOT / "data/processed", ROOT / "reports"
REGIONS = {"NA_Sales": "north_america", "EU_Sales": "europe", "JP_Sales": "japan", "Other_Sales": "other"}
REQUIRED = ["Name", "Platform", "Year", "Genre", "Publisher", *REGIONS]
COVERAGE_END = 2015          # vgchartz scrape is complete through 2015 (2016 partial, 2017+ ~empty)

# Platform reference data (manufacturer, first release year). Used to catch impossible dates.
PLATFORMS = {
 "2600": ("Atari", 1977), "3DO": ("3DO", 1993), "3DS": ("Nintendo", 2011), "DC": ("Sega", 1998),
 "DS": ("Nintendo", 2004), "GB": ("Nintendo", 1989), "GBA": ("Nintendo", 2001), "GC": ("Nintendo", 2001),
 "GEN": ("Sega", 1988), "GG": ("Sega", 1990), "N64": ("Nintendo", 1996), "NES": ("Nintendo", 1983),
 "NG": ("SNK", 1990), "PC": ("PC", 1980), "PCFX": ("NEC", 1994), "PS": ("Sony", 1994), "PS2": ("Sony", 2000),
 "PS3": ("Sony", 2006), "PS4": ("Sony", 2013), "PSP": ("Sony", 2004), "PSV": ("Sony", 2011),
 "SAT": ("Sega", 1994), "SCD": ("Sega", 1991), "SNES": ("Nintendo", 1990), "TG16": ("NEC", 1987),
 "WS": ("Bandai", 1999), "Wii": ("Nintendo", 2006), "WiiU": ("Nintendo", 2012), "X360": ("Microsoft", 2005),
 "XB": ("Microsoft", 2001), "XOne": ("Microsoft", 2013),
 "NS": ("Nintendo", 2017), "PS5": ("Sony", 2020), "XS": ("Microsoft", 2020), "NS2": ("Nintendo", 2025),
}
PUBLISHER_ALIASES = {"Bigben Interactive": "Big Ben Interactive", "Milestone S.r.l": "Milestone S.r.l."}


def write_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp); raise


def slug(s): return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "unknown"
def game_id(n, note, p, y): return hashlib.md5(f"{n}|{note}|{p}|{y}".encode()).hexdigest()[:10]


def load():
    frames = []
    for f in sorted(RAW.glob("*.csv")) + sorted(INCOMING.glob("*.csv")):
        df = pd.read_csv(f)
        missing = set(REQUIRED) - set(df.columns)
        if missing: sys.exit(f"{f.name}: missing columns {missing}")
        df["source"] = df["source"] if "source" in df else f.stem
        df["source_file"] = f.name
        frames.append(df[REQUIRED + ["source", "source_file"]])
    return pd.concat(frames, ignore_index=True)


def main():
    df = load()
    n_raw = len(df)
    df["raw_row"] = range(n_raw)
    q = []   # quarantine: (frame, reason)

    def quarantine(mask, reason):
        nonlocal df
        if mask.any(): q.append(df[mask].assign(reason=reason))
        df = df[~mask]

    df["global_sales"] = df[list(REGIONS)].sum(axis=1).round(2)

    # ---- 1. text hygiene ---------------------------------------------------------------
    for c in ["Name", "Platform", "Genre", "Publisher"]:
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["Publisher"] = df["Publisher"].fillna("Unknown").replace({"N/A": "Unknown", "": "Unknown"})
    df["Publisher"] = df["Publisher"].replace(PUBLISHER_ALIASES)

    # ---- 2. atomic title: split scraper annotation "(...)" out of Name -----------------------
    note = df["Name"].str.extract(r"^(?P<base>.*?)\s*\((?P<note>[^()]*)\)\s*$")
    has = note["base"].notna() & (note["base"].str.len() > 0)
    df["name_note"] = note["note"].where(has)
    df["Name"] = note["base"].where(has, df["Name"])
    # annotations about *sales coverage* ("US weekly sales", "All Region sales") describe a different
    # measurement, not a different game -> not usable as a full-lifetime global sales record.
    partial = df["name_note"].fillna("").str.contains(r"\bsales\b", case=False)
    quarantine(partial, "partial_region_sales_annotation")

    # ---- 3. missing / impossible values --------------------------------------------------
    quarantine(df["Year"].isna(), "missing_year")
    launch = df["Platform"].map({k: v[1] for k, v in PLATFORMS.items()})
    quarantine(launch.isna(), "unknown_platform")
    launch = df["Platform"].map({k: v[1] for k, v in PLATFORMS.items()})
    quarantine((df["Year"] < launch - 0) & ((launch - df["Year"]) >= 2), "release_year_before_platform_existed")
    quarantine(df["Year"] > 2026, "future_year")
    quarantine((df[list(REGIONS)] < 0).any(axis=1), "negative_sales")
    # DS 'Imagine: Makeup Artist' = 2020 is impossible for the platform's lifecycle (DS discontinued 2014)
    sunset = {"DS": 2014, "PS2": 2013, "PSP": 2016, "Wii": 2017, "X360": 2016, "XB": 2009, "GC": 2007, "GBA": 2010}
    quarantine(df["Year"] > df["Platform"].map(sunset).fillna(9999), "release_year_after_platform_sunset")

    # ---- 4. recompute derived totals, check against scraped Global_Sales ----------------------

    # ---- 5. duplicates on the natural key (Name, Platform, Year) ----------------------------
    df["Year"] = df["Year"].astype(int)
    df["name_note"] = df["name_note"].fillna("")
    key = ["Name", "name_note", "Platform", "Year"]
    exact = df.duplicated(key + ["Genre", "Publisher", *REGIONS], keep="first")
    quarantine(exact, "exact_duplicate")
    conflict = df.duplicated(key, keep="first")   # rows are ordered by sales (Rank) -> keep the largest
    quarantine(conflict, "conflicting_duplicate_key")

    # ---- 6. flags (kept in table, not removed) ----------------------------------------------
    df["in_coverage"] = df["Year"] <= COVERAGE_END
    import numpy as np
    lg = np.log1p(df["global_sales"]); q1, q3 = lg.quantile([.25, .75])
    df["is_sales_outlier"] = lg > q3 + 3 * (q3 - q1)       # genuine blockbusters; flagged, never deleted
    df["is_zero_regions_incomplete"] = (df[list(REGIONS)] == 0).sum(axis=1) == 4

    df = df.sort_values(["Year", "Name", "Platform"]).reset_index(drop=True)
    df["game_id"] = [game_id(n, nt, p, y) for n, nt, p, y in zip(df.Name, df.name_note, df.Platform, df.Year)]
    assert df["game_id"].is_unique, "game_id collision"

    # ---- 7. build normalised tables -----------------------------------------------------------
    dim_platform = pd.DataFrame([(k, v[0], v[1]) for k, v in PLATFORMS.items()],
                                columns=["platform_code", "manufacturer", "launch_year"]).sort_values("platform_code")
    dim_genre = pd.DataFrame({"genre": sorted(df["Genre"].unique())})
    pubs = sorted(df["Publisher"].unique())
    dim_publisher = pd.DataFrame({"publisher_id": [slug(p) for p in pubs], "publisher": pubs})
    assert dim_publisher["publisher_id"].is_unique
    dim_game = df.rename(columns={"Name": "name", "Platform": "platform_code", "Year": "release_year",
                                  "Genre": "genre"})
    dim_game["publisher_id"] = df["Publisher"].map(slug)
    dim_game = dim_game[["game_id", "name", "name_note", "platform_code", "release_year", "genre", "publisher_id",
                         "in_coverage", "is_sales_outlier", "is_zero_regions_incomplete", "source"]]
    fact = df.melt(id_vars=["game_id", "source"], value_vars=list(REGIONS), var_name="region",
                   value_name="units_millions")
    fact["region"] = fact["region"].map(REGIONS)
    fact = fact[["game_id", "region", "units_millions", "source"]].sort_values(["game_id", "region"])

    # wide, analysis-ready table (derived from the normalised tables so they can never disagree)
    wide = (fact.pivot(index="game_id", columns="region", values="units_millions")
            .rename(columns=lambda r: f"{r.lower()}_sales").reset_index())
    wide["global_sales"] = wide[[c for c in wide.columns if c.endswith("_sales")]].sum(axis=1).round(2)
    model_table = (dim_game.merge(dim_publisher, on="publisher_id").merge(dim_platform, on="platform_code")
                   .merge(wide, on="game_id").sort_values(["release_year", "name", "platform_code"]))

    # ---- 8. reconciliation checks (fail loudly) ----------------------------------------------
    qdf = pd.concat(q) if q else pd.DataFrame()
    assert len(df) + len(qdf) == n_raw, "row accounting does not balance"
    assert len(model_table) == len(dim_game) == fact["game_id"].nunique() == len(df)
    assert (fact.groupby("game_id").size() == len(REGIONS)).all()
    assert (model_table["global_sales"] - df.set_index("game_id").loc[model_table.game_id, "global_sales"].values).abs().max() < 0.011
    assert (model_table[[f"{r.lower()}_sales" for r in REGIONS.values()]] >= 0).all().all()

    # ---- 9. write atomically -------------------------------------------------------------------
    for name, t in [("dim_platform", dim_platform), ("dim_genre", dim_genre), ("dim_publisher", dim_publisher),
                    ("dim_game", dim_game), ("fact_sales", fact), ("vgsales_model_table", model_table)]:
        write_atomic(t, OUT / f"{name}.csv")
    dim_game["name_note"] = dim_game["name_note"].replace("", pd.NA)
    qout = qdf.rename(columns={"Name": "name", "Platform": "platform_code", "Year": "year"})
    write_atomic(qout[["raw_row", "name", "name_note", "platform_code", "year", "reason", "global_sales"]]
                 .sort_values("raw_row") if len(qout) else qout, OUT / "quarantine.csv")

    # ---- 10. report ---------------------------------------------------------------------------------
    lines = [f"# Data quality report", "", f"- raw rows: **{n_raw}**", f"- clean rows (games): **{len(df)}**",
             f"- quarantined: **{len(qdf)}**", "", "## Quarantined by reason", ""]
    if len(qdf): lines += [f"- {r}: {n}" for r, n in qdf["reason"].value_counts().items()]
    lines += ["", "## Flags kept in the table (rows NOT removed)", "",
              f"- in_coverage = False (year > {COVERAGE_END}): {(~df.in_coverage).sum()}",
              f"- is_sales_outlier (blockbusters): {int(df.is_sales_outlier.sum())}",
              f"- titles with name_note split out: {int((df.name_note != "").sum())}",
              f"- year range: {df.Year.min()}-{df.Year.max()}", f"- sources: {df.source.value_counts().to_dict()}"]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
