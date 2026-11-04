# BDA-2026

Analysis of UK used-car listings for nine brands (Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, VW),
from the Kaggle dataset [`adityadesai13/used-car-dataset-ford-and-mercedes`](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The data is downloaded on first use via `kagglehub` (cached under `~/.cache/kagglehub`), so it is not committed.

## Usage

```python
from src.data import load
df = load()  # ~97.7k rows: model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize, brand, age
```

`src/data.py` combines the nine per-brand CSVs and applies basic cleaning: it strips model names, drops exact
duplicates and an impossible year (2060), and treats `engineSize == 0` as missing.

For modelling, use the fixed training and test samples:

```python
from src.data import train_test
train, test = train_test()  # two disjoint 10k-row samples, seed 2026
```

`prepare()` applies the modelling rules from `PROGRESS.md` (age ≤ 15, known engine size, no Other/Electric fuel or Other
transmission) and adds `log_price` and `mileage_10k`. The dataset's `cclass.csv`,
`focus.csv` and `unclean *.csv` files are skipped because they mostly duplicate the Mercedes and Ford files.

## Notebooks

- `notebooks/01_eda.ipynb`: first look at prices by brand, age, mileage, fuel type and transmission.
