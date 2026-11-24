"""Checks for the preprocessor. Run from the repo root: python tests/check_preprocessor.py"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so "src" can be imported
from src.cleaning import clean_transactions
from src.data import add_candidate_features
from src.preprocessing import NEEDS_PAST_PURCHASES, build_preprocessor, fill_no_history, load_config

config = load_config("config.yaml")["preprocessing"]
no_history_days = config["no_history_days"]

# Same data the pipeline will use: raw files -> clean_transactions -> add_candidate_features
train_raw, test_raw = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
clientes = pd.read_csv("data/clientes.csv")
train = add_candidate_features(clean_transactions(train_raw, history=train_raw, is_train=True), clientes)
test = add_candidate_features(clean_transactions(test_raw, history=train_raw, is_train=False), clientes)
train = train.drop(columns=["repeat_purchase_90d", "spend_next_90d"])   # the preprocessor never sees the targets

print("1) no_history_days is larger than any real gap in train")
largest_recency, largest_gap = train["days_since_previous_purchase"].max(), train["prior_mean_gap_days"].max()
print(f"   largest days_since_previous_purchase: {largest_recency:.0f} | largest prior_mean_gap_days: {largest_gap:.0f}"
      f" | no_history_days: {no_history_days} -> larger: {no_history_days > max(largest_recency, largest_gap)}")

print("2) fill_no_history changes only the structural gaps")
no_history_ratio = config["no_history_ratio"]
filled = fill_no_history(train, no_history_days, no_history_ratio)
first = train["prior_purchase_count"] == 0
print(f"   first purchases: {first.sum()} | prior_mean_purchase_value = 0: "
      f"{filled.loc[first, 'prior_mean_purchase_value'].eq(0).all()} | days_since_previous_purchase = "
      f"{no_history_days}: {filled.loc[first, 'days_since_previous_purchase'].eq(no_history_days).all()}")
under_two = train["prior_purchase_count"] < 2
print(f"   fewer than 2 past purchases: {under_two.sum()} | gap_ratio = {no_history_ratio}: "
      f"{filled.loc[under_two, 'gap_ratio'].eq(no_history_ratio).all()}")
changed = (filled != train) & ~(filled.isna() & train.isna())        # cells whose value changed
structural = pd.DataFrame(False, index=train.index, columns=train.columns)
for col, needed in NEEDS_PAST_PURCHASES.items():
    structural[col] = train[col].isna() & (train["prior_purchase_count"] < needed)
print(f"   cells changed: {changed.sum().sum()} | all of them structural gaps: {(changed == structural).all().all()}"
      f" | rows with 3+ past purchases unchanged: {not changed[train['prior_purchase_count'] >= 3].any().any()}")

print("3) fit on train, transform test")
preprocessor = build_preprocessor(config).fit(train)
test_out = preprocessor.transform(test)
print(f"   output columns: {test_out.shape[1]} | test rows: {len(test_out)} | NaN left: {test_out.isna().sum().sum()}")

print("4) test categories never seen in train")
unseen = pd.Series(False, index=test.index)
for col in config["categorical_features"]:
    unseen |= test[col].notna() & ~test[col].isin(train[col].dropna().unique())
print(f"   test rows with an unseen category: {unseen.sum()} -> transformed without error")

print("5) one test row alone")
one_row = preprocessor.transform(test.iloc[[0]])
print(f"   shape: {one_row.shape} | same as that row in the full transform: "
      f"{np.allclose(one_row.to_numpy(), test_out.iloc[[0]].to_numpy())}")

print("6) every numeric_strategy, and capping / log1p on and off (fit on train, transform test)")
for strategy in ["median", "mean", "knn", "iterative", "random_forest"]:
    options = copy.deepcopy(config)
    options["imputation"].update(numeric_strategy=strategy, max_iter=3, n_estimators=10)   # small, so the check is quick
    out = build_preprocessor(options).fit(train).transform(test)
    print(f"   {strategy:14s} columns: {out.shape[1]} | NaN left: {out.isna().sum().sum()}")
for cap in [None, [0.01, 0.99]]:
    for log1p in [None, config["log1p_columns"]]:
        options = dict(config, cap_quantiles=cap, log1p_columns=log1p)
        out = build_preprocessor(options).fit(train).transform(test)
        print(f"   capping {'on ' if cap else 'off'} | log1p {'on ' if log1p else 'off'} -> columns: {out.shape[1]}"
              f" | NaN left: {out.isna().sum().sum()}")

print("7) capping touches only cap_columns (fit on train)")
fitted = build_preprocessor(config).fit(train)
numeric_in = fitted["columns"].transformers_[0][2]                  # the columns the numeric branch receives
before = fitted[:-1].transform(train)[numeric_in]                  # after drop_correlated and fill_no_history
after = fitted["columns"].named_transformers_["numeric"]["cap"].transform(before)
for col in ["customer_tenure_days", "Total a Pagar"]:
    changed = int(((after[col] != before[col]) & before[col].notna()).sum())
    print(f"   {col:22s} in cap_columns: {col in config['cap_columns']} | values changed by capping: {changed} | "
          f"max before: {before[col].max():.0f} -> after: {after[col].max():.0f}")
