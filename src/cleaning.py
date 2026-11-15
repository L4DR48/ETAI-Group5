"""Rule-based cleaning of the raw transactions: the steps of notebook 01 in one function."""

import pandas as pd

from src.data import CATEGORICALS, flag_prior_maximum, mark_prior_maximum_missing, mark_rolling_spend_missing
from src.preprocessing import (
    rebuild_max, rebuild_rolling_spend, recover_spending_targets,
    repair_missing_mean_gap, repair_missing_recency, repair_negative_days,
)

TARGET = "repeat_purchase_90d"
SPEND = "spend_next_90d"


def clean_transactions(data, history, is_train):
    """Clean raw transactions. `history` is the raw train table, used to check each row
    against the customer's earlier purchases. Rows are dropped only when is_train is True."""
    out = data.copy()
    out["purchase_date"] = pd.to_datetime(out["purchase_date"])

    if is_train:
        # 01, Section 2: drop the empty rows (only ID, customer and target filled)
        predictors = out.columns.drop(["ID", "customer_id", TARGET, SPEND])
        out = out.loc[~out[predictors].isna().all(axis=1)].copy()

    # 01, Section 3.2: mean of past purchases = total spend / number of purchases
    has_history = out["prior_purchase_count"] > 0
    out["prior_mean_purchase_value"] = (out["prior_total_spend"] / out["prior_purchase_count"]).where(has_history)

    # 01, Section 3.3: flag impossible maxima, set them to NaN, rebuild the ones that can be verified
    flagged = flag_prior_maximum(out)
    out, max_audit = mark_prior_maximum_missing(out, history=history, flagged_ids=flagged["ID"])
    out = rebuild_max(out, max_audit)

    # 01, Section 3.4: invalid rolling spend set to NaN, then rebuilt where it can be verified
    out, rolling_audit = mark_rolling_spend_missing(out, history=history)
    out = rebuild_rolling_spend(out, rolling_audit)

    # 01, Section 3.5: negative day counts (-4) repaired from the purchase dates
    out, _ = repair_negative_days(out, history=history)

    # 01, Section 3.6: missing days since the previous purchase, rebuilt from the dates
    out, _ = repair_missing_recency(out, history=history)

    # 01, Section 3.7: missing mean gap, rebuilt from the row's own tenure and recency
    out, _ = repair_missing_mean_gap(out)

    if is_train:
        # 01, Section 5: recover missing spend targets from later purchases (target only)
        out, _ = recover_spending_targets(out)

    # 01, Section 7: keep only the number of each anonymised category label, as text
    # (astype("string") first, so a column that is all NaN, e.g. one new purchase, doesn't crash)
    for col in CATEGORICALS:
        out[col] = out[col].astype("string").str.strip().str.extract(r"(\d+)$", expand=False)

    return out
