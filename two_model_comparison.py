"""import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results_adult/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]

# Filter only the XGBoost runs
xgb_runs = runs[runs["model"] == "xgboost"].reset_index(drop=True)

res = corrected_paired_test(
    xgb_runs.loc[0, folds], xgb_runs.loc[1, folds], test_train_ratio=1 / 4
)
print(res)"""

from itertools import combinations
import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]

imputers = ["median", "knn", "iterative", "random_forest"]

# Keep the latest XGBoost run for each imputer
latest = (
    runs[(runs["model"] == "xgboost") & (runs["imputer"].isin(imputers))]
    .drop_duplicates(subset="imputer", keep="last")
    .set_index("imputer")
)

# Rank imputers from best to worst by mean CV score
summary = pd.DataFrame({
    "mean": latest[folds].mean(axis=1),
    "std": latest[folds].std(axis=1),
}).sort_values("mean", ascending=False)

print("=== Imputer Ranking ===")
print(summary.round(4))
print(f"\nBest imputer by mean CV score: {summary.index[0]}\n")

# Run corrected paired test on all pairwise combinations (ordered best to worst)
print("=== Pairwise Corrected Paired Tests ===")
for a, b in combinations(summary.index, 2):
    res = corrected_paired_test(
        latest.loc[a, folds], latest.loc[b, folds], test_train_ratio=1 / 4
    )
    print(f"{a} vs {b}: {res}")