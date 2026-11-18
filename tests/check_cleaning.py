"""Checks for clean_transactions. Run from the repo root: python tests/check_cleaning.py

a) Cleaning the raw files gives the same data as notebook 01 saved in data/processed/.
b) Cleaning each test row alone gives the same result as cleaning all of test together,
   so a row never depends on other test rows (needed for the client interface).
c) One new purchase with all six categories missing goes through without an error.
d) load_clean reads the processed files with categories as text and purchase_date as a date.
e) add_candidate_features keeps every row in order, adds exactly the three kept features,
   and gives the same result for each test row alone as for all of test.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so "src" can be imported
from src.cleaning import clean_transactions
from src.data import CATEGORICALS, add_candidate_features, load_clean


def as_read(frame):
    """The table as it looks after to_csv / read_csv, so both sides have the same types."""
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)))


def report(name, ours, reference):
    """Print whether two tables are equal and, if not, how many cells differ per column."""
    ours, reference = as_read(ours), as_read(reference)
    if ours.shape != reference.shape or list(ours.columns) != list(reference.columns):
        print(f"{name}: DIFFERENT SHAPE OR COLUMNS {ours.shape} vs {reference.shape}")
        return
    different = {}
    for col in ours.columns:
        a, b = ours[col], reference[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a, b, rtol=0, atol=1e-6, equal_nan=True)   # numbers: rounding is not a difference
        else:
            same = a.astype(str) == b.astype(str)                          # text: exact comparison
        if (~same).sum() > 0:
            different[col] = int((~same).sum())
    print(f"{name}: {ours.shape} ->", "identical" if not different else f"cells that differ: {different}")


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_clean = clean_transactions(train, history=train, is_train=True)
test_clean = clean_transactions(test, history=train, is_train=False)

print("a) clean_transactions vs data/processed/ from notebook 01")
report("   train", train_clean, pd.read_csv("data/processed/train_clean.csv"))
report("   test ", test_clean, pd.read_csv("data/processed/test_clean.csv"))

print("b) each test row cleaned alone vs all of test together")
one_by_one = pd.concat([clean_transactions(test.iloc[[i]], history=train, is_train=False) for i in range(len(test))])
report("   test ", one_by_one, test_clean)

print("c) one new purchase with all six categories missing")
new_purchase = test.iloc[[0]].copy()
new_purchase[CATEGORICALS] = np.nan                                 # these columns become numeric NaN
cleaned = clean_transactions(new_purchase, history=train, is_train=False)
print("   no error | categories after cleaning:", cleaned[CATEGORICALS].iloc[0].tolist())

print("d) load_clean reads the processed files with the right types")
for name in ["train_clean", "test_clean"]:
    loaded = load_clean(f"data/processed/{name}.csv")
    categories_are_text = all(pd.api.types.is_string_dtype(loaded[col]) for col in CATEGORICALS)
    date_is_date = pd.api.types.is_datetime64_any_dtype(loaded["purchase_date"])
    print(f"   {name}: categories as text: {categories_are_text} | purchase_date as date: {date_is_date}")

print("e) add_candidate_features adds exactly gap_ratio, tax_rate and island_missing")
clientes = pd.read_csv("data/clientes.csv")
for name, data in [("train", train_clean), ("test ", test_clean)]:
    features = add_candidate_features(data, clientes)
    added = [col for col in features.columns if col not in data.columns]
    same_order = (features["ID"].to_numpy() == data["ID"].to_numpy()).all()
    island = features["island_missing"]
    print(f"   {name}: rows {len(data)} -> {len(features)} | same order: {same_order} | added: {added} | "
          f"island_missing NaN: {island.isna().sum()}, values: {sorted(island.unique().tolist())}")
test_features = add_candidate_features(test_clean, clientes)
one_by_one = pd.concat([add_candidate_features(test_clean.iloc[[i]], clientes) for i in range(len(test_clean))])
report("   each test row alone vs all of test", one_by_one, test_features)
