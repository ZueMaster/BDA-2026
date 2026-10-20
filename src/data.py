"""Load and clean the UK used-car dataset (adityadesai13/used-car-dataset-ford-and-mercedes)."""

from pathlib import Path

import kagglehub
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
