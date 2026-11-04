"""Load and clean the UK used-car dataset (adityadesai13/used-car-dataset-ford-and-mercedes)."""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

DATASET = "adityadesai13/used-car-dataset-ford-and-mercedes"

# One file per brand, all sharing the same schema. The dataset also ships
# cclass.csv / focus.csv (mostly duplicates of merc / ford rows, without tax
# and mpg) and their raw "unclean" scrapes; those are deliberately skipped.
BRAND_FILES = {
    "audi": "Audi",
    "bmw": "BMW",
    "ford": "Ford",
    "hyundi": "Hyundai",
    "merc": "Mercedes",
    "skoda": "Skoda",
    "toyota": "Toyota",
    "vauxhall": "Vauxhall",
    "vw": "Volkswagen",
}


def download() -> Path:
    """Download (or reuse the cached copy of) the dataset and return its directory."""
    return Path(kagglehub.dataset_download(DATASET))


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Concatenate all brand files into one frame with a `brand` column."""
    path = path or download()
    frames = []
    for stem, brand in BRAND_FILES.items():
        df = pd.read_csv(path / f"{stem}.csv").rename(columns={"tax(£)": "tax"})
        df["brand"] = brand
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the basic fixes found during the first profiling pass."""
    df = df.copy()
    df["model"] = df["model"].str.strip()
    df = df.drop_duplicates()
    # One Ford Fiesta is listed with year 2060; the data was scraped in 2020.
    df = df[df["year"] <= 2020]
    # engineSize == 0 on non-electric cars means "unknown", not zero.
    df.loc[(df["engineSize"] == 0) & (df["fuelType"] != "Electric"), "engineSize"] = pd.NA
    df["age"] = 2020 - df["year"]
    for col in ["brand", "transmission", "fuelType"]:
        df[col] = df[col].astype("category")
    return df.reset_index(drop=True)


def load(path: Path | None = None) -> pd.DataFrame:
    """Download if needed, then return the combined, cleaned dataset."""
    return clean(load_raw(path))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the modelling rules agreed in PROGRESS.md and add model-ready columns."""
    df = df[
        (df["age"] <= 15)
        & df["engineSize"].notna()
        & ~df["fuelType"].isin(["Other", "Electric"])
        & (df["transmission"] != "Other")
    ].copy()
    # Drop the now-empty levels, otherwise Bambi creates coefficients for them.
    for col in ["brand", "transmission", "fuelType"]:
        df[col] = df[col].cat.remove_unused_categories()
    df["log_price"] = np.log(df["price"])
    df["mileage_10k"] = df["mileage"] / 10_000
    return df.reset_index(drop=True)


def train_test(n: int = 10_000, seed: int = 2026, path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two disjoint random samples of `n` rows: a training set and a held-out test set.

    The fixed seed means every group member gets exactly the same rows.
    """
    df = prepare(load(path))
    sample = df.sample(n=2 * n, random_state=seed)
    return sample.iloc[:n].reset_index(drop=True), sample.iloc[n:].reset_index(drop=True)
