def rebuild_max(data, audit):
    """Restore verified maxima; leave unresolved cases as NaN."""
    result = data.copy()

    replacements = audit.set_index("ID")["verified_maximum"]
    flagged = result["ID"].isin(replacements.index)

    result.loc[flagged, "prior_max_purchase_value"] = (
        result.loc[flagged, "ID"].map(replacements)
    )

    return result

def rebuild_rolling_spend(data, audit):
    """Restore verified spending; leave unresolved cells as NaN."""
    result = data.copy()

    for column, group in audit.groupby("spend_column"):
        replacements = group.set_index("ID")["verified_spend"]
        flagged = result["ID"].isin(replacements.index)

        result.loc[flagged, column] = (
            result.loc[flagged, "ID"].map(replacements)
        )

    return result
import pandas as pd
import numpy as np
def repair_negative_days(data, history, tolerance=0.01):
    days = "days_since_previous_purchase"
    mean = "prior_mean_gap_days"
    result = data.copy()
    base = data.reset_index(drop=True)

    n = base["prior_purchase_count"]
    n = n.where(np.isfinite(n) & n.ge(1) & n.eq(n.round()))

    tenure = base["customer_tenure_days"]
    tenure = tenure.where(np.isfinite(tenure) & tenure.ge(0))

    recency = base[days].where(base[days].between(0, tenure))
    avg_gap = base[mean].where(base[mean].between(0, tenure))

    # Candidates from valid summaries in the same row.
    candidates = pd.DataFrame({
        days: (tenure - (n - 1) * avg_gap).where(n.gt(1)),
        mean: (tenure - recency) / (n - 1).where(n.gt(1)),
    })
    candidates.loc[n.eq(1), days] = tenure.loc[n.eq(1)]
    candidates = candidates.where(
        np.isfinite(candidates)
        & candidates.ge(0)
        & candidates.le(tenure, axis=0)
    )

    reference = history.assign(
        purchase_date=pd.to_datetime(
            history["purchase_date"], errors="coerce"
        )
    )

    records = []
    negative_rows = base[[days, mean]].lt(0).any(axis=1)

    for index, row in base.loc[negative_rows].iterrows():
        date = pd.to_datetime(row["purchase_date"], errors="coerce")
        past = reference.loc[
            reference["customer_id"].eq(row["customer_id"])
            & reference["purchase_date"].lt(date)
        ]
        amounts = past["Total a Pagar"]

        # Date-based reconstruction requires complete, usable history.
        complete = (
            len(past) == n.loc[index]
            and amounts.notna().all()
            and np.isfinite(amounts).all()
            and amounts.ge(0).all()
            and np.isclose(
                amounts.sum(), row["prior_total_spend"],
                atol=tolerance, rtol=0,
            )
        )

        for column in [days, mean]:
            if not row[column] < 0:
                continue

            value = candidates.at[index, column]
            status = (
                "verified_from_row"
                if pd.notna(value) else "unresolved"
            )

            if pd.isna(value) and complete:
                if column == days:
                    value = (date - past["purchase_date"].max()).days
                elif n.loc[index] > 1:
                    span = (
                        past["purchase_date"].max()
                        - past["purchase_date"].min()
                    ).days
                    value = span / (n.loc[index] - 1)

                if pd.notna(value):
                    status = "complete_history"

            result.loc[result["ID"].eq(row["ID"]), column] = value
            records.append(
                (row["ID"], column, row[column], value, status)
            )

    audit = pd.DataFrame(records, columns=[
        "ID", "column", "original_value",
        "verified_value", "recovery_status",
    ])
    return result, audit

def repair_missing_recency(data, history, tolerance=0.01):
    column = "days_since_previous_purchase"
    result = data.copy()

    reference = history.assign(
        purchase_date=pd.to_datetime(
            history["purchase_date"], errors="coerce"
        )
    )

    eligible = (
        data[column].isna()
        & data["prior_purchase_count"].gt(0)
    )
    records = []

    for _, row in data.loc[eligible].iterrows():
        n = row["prior_purchase_count"]
        tenure = row["customer_tenure_days"]
        avg = row["prior_mean_gap_days"]
        valid_count = np.isfinite(n) and n == int(n)
        value = np.nan

        # First try the summaries in this row.
        if (
            valid_count and pd.notna(tenure)
            and np.isfinite(tenure) and tenure >= 0
        ):
            if n == 1:
                value = tenure
            elif pd.notna(avg) and np.isfinite(avg) and avg >= 0:
                value = tenure - (n - 1) * avg

        valid_value = (
            np.isfinite(value)
            and -1e-8 <= value <= tenure + 1e-8
            and np.isclose(value, np.rint(value), atol=1e-8, rtol=0)
        )
        value = float(np.rint(value)) if valid_value else np.nan
        status = (
            "verified_from_row" if pd.notna(value) else "unresolved"
        )

        # Otherwise use dates only with verified complete history.
        if pd.isna(value):
            date = pd.to_datetime(row["purchase_date"], errors="coerce")
            past = reference.loc[
                reference["customer_id"].eq(row["customer_id"])
                & reference["purchase_date"].lt(date)
            ]
            amounts = past["Total a Pagar"]

            complete = (
                valid_count and len(past) == n
                and amounts.notna().all()
                and np.isfinite(amounts).all()
                and amounts.ge(0).all()
                and np.isclose(
                    amounts.sum(), row["prior_total_spend"],
                    atol=tolerance, rtol=0,
                )
            )

            if complete:
                value = (date - past["purchase_date"].max()).days
                status = "complete_history"

        result.loc[result["ID"].eq(row["ID"]), column] = value
        records.append((row["ID"], column, np.nan, value, status))

    audit = pd.DataFrame(records, columns=[
        "ID", "column", "original_value",
        "verified_value", "recovery_status",
    ])
    return result, audit


def repair_missing_mean_gap(data):
    result = data.copy()
    column = "prior_mean_gap_days"

    n = data["prior_purchase_count"]
    tenure = data["customer_tenure_days"]
    recency = data["days_since_previous_purchase"]

    eligible = data[column].isna() & n.ge(2)
    valid_count = np.isfinite(n) & n.ge(2) & n.eq(n.round())
    valid_timing = (
        np.isfinite(tenure) & tenure.ge(0)
        & np.isfinite(recency) & recency.between(0, tenure)
    )

    candidate = (tenure - recency) / (n - 1).where(valid_count)
    candidate = candidate.where(
        valid_timing & np.isfinite(candidate)
    )

    audit = data.loc[
        eligible, ["ID", "prior_purchase_count", column]
    ].rename(columns={column: "original_value"}).copy()

    audit["verified_value"] = candidate.loc[eligible].to_numpy()
    audit["recovery_status"] = np.where(
        audit["verified_value"].notna(),
        "verified_from_row",
        "unresolved",
    )

    result.loc[eligible, column] = candidate.loc[eligible].to_numpy()
    return result, audit


TARGET = "repeat_purchase_90d"
SPEND = "spend_next_90d"

def recover_spending_targets(data, tolerance=0.01):
    result = data.copy()

    missing = result[TARGET].eq(1) & result[SPEND].isna()

    # Keep a separate record of each attempted repair.
    repairs = (
        result.loc[missing, ["ID", "customer_id"]]
        .set_index("ID")
        .copy()
    )
    repairs["verified_spend"] = np.nan
    repairs["method"] = "unresolved"

    dated = result.dropna(subset=["customer_id", "purchase_date"])

    for _, group in (
        dated.sort_values(["customer_id", "purchase_date"])
        .groupby("customer_id")
    ):
        # Avoid ambiguous ordering of purchases on the same date.
        if group["purchase_date"].duplicated().any():
            continue

        dates = group["purchase_date"].to_numpy()
        paid = group["Total a Pagar"].to_numpy(
            dtype=float, na_value=np.nan
        )
        counts = group["prior_purchase_count"].to_numpy(
            dtype=float, na_value=np.nan
        )
        totals = group["prior_total_spend"].to_numpy(
            dtype=float, na_value=np.nan
        )
        repeat = group[TARGET].to_numpy()
        ids = group["ID"].to_numpy()

        def complete_between(i, j):
            """Check purchases from i up to, but excluding, j."""
            endpoint_counts = counts[[i, j]]
            endpoint_totals = totals[[i, j]]
            amounts = paid[i:j + 1]

            return (
                np.isfinite(endpoint_counts).all()
                and (endpoint_counts >= 0).all()
                and (endpoint_counts == np.floor(endpoint_counts)).all()
                and counts[j] - counts[i] == j - i
                and np.isfinite(endpoint_totals).all()
                and (endpoint_totals >= 0).all()
                and np.isfinite(amounts).all()
                and (amounts >= 0).all()
                and np.isclose(
                    totals[j] - totals[i],
                    paid[i:j].sum(),
                    atol=tolerance,
                    rtol=0
                )
            )

        for i, row_id in enumerate(ids):
            if row_id not in repairs.index:
                continue

            end = dates[i] + np.timedelta64(90, "D")

            inside = np.flatnonzero(
                (dates > dates[i]) & (dates <= end)
            )

            if not len(inside):
                continue

            closing = np.flatnonzero(dates >= end)
            method = None

            # A purchase at or beyond the boundary lets us check
            # that no purchases are missing before the window ends.
            if (
                len(closing)
                and complete_between(i, int(closing[0]))
            ):
                method = "verified_with_later_purchase"

            else:
                last = int(inside[-1])

                # repeat = 0 confirms no further purchase for 90 days
                # after this last purchase, covering the remaining window.
                if repeat[last] == 0 and complete_between(i, last):
                    method = "verified_with_no_further_purchase"

            if method is not None:
                repairs.at[row_id, "verified_spend"] = round(
                    float(paid[inside].sum()), 2
                )
                repairs.at[row_id, "method"] = method

    # Fill only verified missing targets; unresolved cases stay NaN.
    result.loc[missing, SPEND] = (
        result.loc[missing, "ID"]
        .map(repairs["verified_spend"])
        .to_numpy()
    )

    return result, repairs.reset_index()
