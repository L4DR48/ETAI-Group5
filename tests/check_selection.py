"""Checks for the correlation filter and the VoteSelector. Run from the repo root:
python tests/check_selection.py   (everything is fitted on train only)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so "src" can be imported
from src.cleaning import clean_transactions
from src.data import add_candidate_features
from src.model import VoteSelector
from src.preprocessing import build_preprocessor, load_config

TARGET, SPEND = "repeat_purchase_90d", "spend_next_90d"
settings = load_config("config.yaml")
config, selection = settings["preprocessing"], settings["feature_selection"]

# Same data the pipeline will use: raw files -> clean_transactions -> add_candidate_features
train_raw, test_raw = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
clientes = pd.read_csv("data/clientes.csv")
train = add_candidate_features(clean_transactions(train_raw, history=train_raw, is_train=True), clientes)
test = add_candidate_features(clean_transactions(test_raw, history=train_raw, is_train=False), clientes)
X_train, y_train = train.drop(columns=[TARGET, SPEND]), train[TARGET]


def select(X, y, preprocessing_config, task):
    """Fits the preprocessor and a VoteSelector on (X, y); returns both and the selector's votes."""
    prep = build_preprocessor(preprocessing_config).fit(X)
    selector = VoteSelector(min_votes=selection["min_votes"], random_state=selection["random_state"],
                            task=task, lasso_alpha=selection["lasso_alpha"]).fit(prep.transform(X), y)
    votes = pd.Series(selector.votes_, index=prep.get_feature_names_out(), name="votes")
    return prep, selector, votes


def show_votes(votes):
    """Features grouped by number of votes (class 08, Section 4), and the ones kept."""
    for n in [3, 2, 1, 0]:
        names = [name.split("__", 1)[1] for name in votes.index[votes == n]]
        print(f"   {n} votes ({len(names)}): {', '.join(names) if n > 0 else '...'}")
    kept = votes[votes >= selection["min_votes"]]
    print(f"   kept with min_votes={selection['min_votes']}: {len(kept)} of {len(votes)}")
    return set(kept.index)


print("0) pairs with |Spearman| > 0.95 in train, and the column dropped from each")
corr = X_train[config["numeric_features"]].corr(method="spearman")
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
for (a, b), rho in pairs[pairs.abs() > 0.95].sort_values(ascending=False).items():
    dropped = [col for col in (a, b) if col in config["drop_correlated"]]
    print(f"   {a} - {b}: {rho:.3f} -> dropped: {', '.join(dropped) if dropped else 'NONE'}")

print("a) classification (repeat_purchase_90d)")
prep, selector, votes = select(X_train, y_train, config, "classification")
kept_classification = show_votes(votes)
print(f"   random_noise kept: {'numeric__random_noise' in kept_classification} | commercial_zone columns kept: "
      f"{sorted(n for n in kept_classification if 'commercial_zone' in n) or 'none'}")

# The L1 voter must really use an L1 penalty. In older scikit-learn, l1_ratio=1.0 is ignored without
# penalty="elasticnet", so the model falls back to L2 and no coefficient is ever exactly 0.
l1_voter = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(prep.transform(X_train), y_train)  # as in VoteSelector
zero_coefficients = int((l1_voter.coef_ == 0).sum())
print(f"   L1 voter: {zero_coefficients} of {l1_voter.coef_.size} coefficients exactly 0")
if zero_coefficients == 0:
    sys.exit("scikit-learn is older than the pinned version: the L1 voter is not using an L1 penalty")


def add_noise_columns(X, k, seed=0):
    """Adds k columns of random numbers (no information about the target). From class 08, Section 3."""
    rng = np.random.default_rng(seed)
    noise = pd.DataFrame(rng.normal(size=(len(X), k)), index=X.index, columns=[f"noise_{i:03d}" for i in range(k)])
    return pd.concat([X, noise], axis=1)


print("b) noise check, 5 seeds: 20 columns of random numbers added to train (class 08, Section 3)")
print("   (a diagnostic of how strict the selector is, not a way to pick settings)")
noise_config = {**config, "numeric_features": config["numeric_features"] + [f"noise_{i:03d}" for i in range(20)]}
counts = []
for seed in range(5):
    _, _, noise_votes = select(add_noise_columns(X_train, 20, seed=seed), y_train, noise_config, "classification")
    is_noise = noise_votes.index.str.contains("noise_")              # the 20 added columns
    for min_votes in [2, 3]:
        kept = noise_votes >= min_votes
        counts.append({"min_votes": min_votes, "noise kept (of 20)": (kept & is_noise).sum(),
                       "real features kept": (kept & ~is_noise).sum()})
averages = pd.DataFrame(counts).groupby("min_votes").mean()
for min_votes, row in averages.iterrows():
    print(f"   min_votes={min_votes}: noise kept {row['noise kept (of 20)']:.1f} of 20 | "
          f"real features kept {row['real features kept']:.1f} (average of 5 seeds)")

print("c) regression (log1p(spend_next_90d), returning customers with a known spend)")
returners = train[TARGET].eq(1) & train[SPEND].notna()
_, _, reg_votes = select(X_train[returners], np.log1p(train.loc[returners, SPEND]), config, "regression")
print(f"   rows: {returners.sum()}")
kept_regression = show_votes(reg_votes)
short = lambda names: sorted(n.split("__", 1)[1] for n in names)   # noqa: E731
print(f"   kept in both: {short(kept_classification & kept_regression)}")
print(f"   only classification: {short(kept_classification - kept_regression)}")
print(f"   only regression: {short(kept_regression - kept_classification)}")

print("d) the fitted classification selector on test and on one new purchase")
train_selected = selector.transform(prep.transform(X_train))
test_selected = selector.transform(prep.transform(test))
one_selected = selector.transform(prep.transform(test.iloc[[0]]))
# Kept names = the preprocessor's column names where the selector says "keep" (as in class 08's main.py)
kept_names = prep.get_feature_names_out()[selector.get_support()]
same_input_columns = list(prep.transform(test).columns) == list(prep.transform(X_train).columns)
print(f"   train: {train_selected.shape} | test: {test_selected.shape} | one purchase: {one_selected.shape} | "
      f"same columns before selection: {same_input_columns} | kept: {len(kept_names)}")
