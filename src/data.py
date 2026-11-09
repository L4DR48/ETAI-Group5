"""Data loading."""
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(path)


"""Inspect whether reported earlier purchases exist in a reference table.

This module never modifies its input tables. When checking validation or test
rows for use in a prediction pipeline, pass only the corresponding training
transactions as ``history``. A matching count does not verify spend values.
"""

import numpy as np


def check_prior_transactions(data, history):
    """Compare each prior_purchase_count with strictly earlier reference rows.

    count_difference = reported count - number of earlier rows found.
    Positive differences indicate unavailable transactions in the reference.
    Negative differences indicate more reference rows than the reported count.
    Undated reference rows cannot be assigned to a historical time window.
    """
    reference = history[["customer_id", "purchase_date"]].copy()
    reference["purchase_date"] = pd.to_datetime(
        reference["purchase_date"], errors="coerce"
    )
    reference = reference.dropna(subset=["customer_id", "purchase_date"])
    dates_by_customer = {
        customer: group["purchase_date"].sort_values().to_numpy()
        for customer, group in reference.groupby("customer_id")
    }

    result = data[
        ["ID", "customer_id", "purchase_date", "prior_purchase_count"]
    ].copy()
    result["purchase_date"] = pd.to_datetime(
        result["purchase_date"], errors="coerce"
    )

    counts, statuses = [], []
    columns = ["customer_id", "purchase_date", "prior_purchase_count"]
    for customer, date, expected in result[columns].itertuples(
        index=False, name=None
    ):
        if (
            pd.isna(customer)
            or pd.isna(date)
            or pd.isna(expected)
            or not np.isfinite(expected)
            or expected < 0
            or expected != int(expected)
        ):
            counts.append(pd.NA)
            statuses.append("cannot_check")
            continue

        dates = dates_by_customer.get(customer)
        found = 0 if dates is None else int(
            np.searchsorted(dates, date.to_datetime64(), side="left")
        )
        counts.append(found)
        if found == expected:
            statuses.append("count_matches")
        elif found < expected:
            statuses.append("fewer_found")
        else:
            statuses.append("more_found")

    result["prior_transactions_found"] = pd.array(counts, dtype="Int64")
    result["count_difference"] = (
        result["prior_purchase_count"] - result["prior_transactions_found"]
    )
    result["status"] = statuses
    return result


def flag_prior_maximum(data, tolerance=0.01):
    """Inspect inconsistent maxima without changing the input data."""
    n = data["prior_purchase_count"]
    total = data["prior_total_spend"]
    minimum = data["prior_min_purchase_value"]
    maximum = data["prior_max_purchase_value"]
    std = data["prior_std_purchase_value"]

    mean = total / n.where(n > 0)

    upper_from_min = total - (n - 1) * minimum
    upper_from_std = mean + std * np.sqrt((n - 1).clip(lower=0))

    upper_limit = pd.concat(
        [total, upper_from_min, upper_from_std], axis=1
    ).min(axis=1)

    invalid = (n > 0) & (
        (maximum < mean - tolerance)
        | (maximum > upper_limit + tolerance)
    )

    columns = [
        "ID",
        "customer_id",
        "prior_purchase_count",
        "prior_total_spend",
        "prior_min_purchase_value",
        "prior_max_purchase_value",
        "prior_std_purchase_value",
    ]

    flagged = data.loc[invalid, columns].copy()
    flagged["recomputed_mean"] = mean.loc[invalid]
    flagged["upper_from_min"] = upper_from_min.loc[invalid]
    flagged["upper_from_std"] = upper_from_std.loc[invalid]
    flagged["upper_limit"] = upper_limit.loc[invalid]

    return flagged


"""Check provided historical spend against available earlier transactions.

Inputs are never modified. Use the appropriate training split as ``history``
when auditing validation or test rows for a prediction pipeline. Other test
rows must not be used as reference transactions.
"""

import numpy as np
import pandas as pd


"""Set flagged maxima to missing and retain a recovery audit for later.

Only the flagged ``prior_max_purchase_value`` cells are changed in the
returned copy. Verified replacement values are recorded, never applied.

Recovery classifications assume historical counts/totals are reliable and
purchase amounts are non-negative. Use only strictly earlier training
transactions as the reference. During validation, pass that fold's training
rows as ``history``; never pass other validation/test rows.
"""

import numpy as np
import pandas as pd


"""Audit rolling spend and mask invalid cells without applying reconstruction.

Use mark_rolling_spend_missing for configurable windows. The original
mark_spend_365d_missing name remains available for existing notebooks.
Historical counts/totals are assumed reliable and purchases non-negative.
Reference history must contain only the appropriate training rows, never
other validation/test rows. Current and future purchases are always excluded.
"""

import re
from numbers import Integral

import numpy as np
import pandas as pd


def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _valid_count(values):
    return np.isfinite(values) & values.ge(0) & values.eq(values.round())


def _prepare_history(history):
    if "ID" in history and (
        history["ID"].isna().any() or not history["ID"].is_unique
    ):
        raise ValueError("history must have non-missing, unique transaction IDs.")
    reference = history[
        ["customer_id", "purchase_date", "Total a Pagar"]
    ].copy()
    reference["purchase_date"] = pd.to_datetime(
        reference["purchase_date"], errors="coerce"
    )
    reference["Total a Pagar"] = _numeric(reference, "Total a Pagar")
    reference = reference.dropna(subset=["customer_id", "purchase_date"])
    prepared = {}
    for customer, group in reference.groupby("customer_id"):
        group = group.sort_values("purchase_date")
        dates = group["purchase_date"].to_numpy(dtype="datetime64[ns]")
        amounts = group["Total a Pagar"].to_numpy(dtype=float)
        usable = np.isfinite(amounts) & (amounts >= 0)
        prepared[customer] = (
            dates,
            np.r_[0.0, np.cumsum(np.where(usable, amounts, 0.0))],
            np.r_[0, np.cumsum(~usable)],
        )
    return prepared


def _window_history(base, prepared, days):
    counts, sums, bad_amounts = [], [], []
    for customer, date in base[
        ["customer_id", "purchase_date"]
    ].itertuples(index=False, name=None):
        if pd.isna(customer) or pd.isna(date):
            counts.append(np.nan)
            sums.append(np.nan)
            bad_amounts.append(np.nan)
            continue
        customer_history = prepared.get(customer)
        if customer_history is None:
            counts.append(0)
            sums.append(0.0)
            bad_amounts.append(0)
            continue
        dates, prefix_spend, prefix_bad = customer_history
        end = date.to_datetime64().astype("datetime64[ns]")
        start = end - np.timedelta64(days, "D")
        left = np.searchsorted(dates, start, side="left")
        right = np.searchsorted(dates, end, side="left")
        counts.append(int(right - left))
        sums.append(float(prefix_spend[right] - prefix_spend[left]))
        bad_amounts.append(int(prefix_bad[right] - prefix_bad[left]))
    return pd.DataFrame({
        "transactions_found": counts,
        "observed_spend": sums,
        "unusable_amounts": bad_amounts,
    }, index=base.index)


def _consistent_candidate(candidates, info, historical_total, tolerance):
    values = pd.DataFrame(candidates, index=info.index)
    values = values.where(np.isfinite(values))
    consistent = (
        values.notna().any(axis=1)
        & values.max(axis=1).sub(values.min(axis=1)).le(tolerance)
    )
    verified = values.bfill(axis=1).iloc[:, 0].where(consistent)
    valid = (
        verified.ge(0)
        & (~np.isfinite(historical_total) | verified.le(historical_total + tolerance))
        & (
            ~info["usable_history"]
            | verified.ge(info["observed_spend"] - tolerance)
        )
        & (
            ~info["complete_window_history"]
            | verified.sub(info["observed_spend"]).abs().le(tolerance)
        )
        & (
            info["transactions_found"].isna()
            | info["transactions_found"].le(info["expected_count"])
        )
    )
    return verified.where(valid)

"""Set flagged maxima to missing and retain a recovery audit for later.

Only the flagged ``prior_max_purchase_value`` cells are changed in the
returned copy. Verified replacement values are recorded, never applied.

Recovery classifications assume historical counts/totals are reliable and
purchase amounts are non-negative. Use only strictly earlier training
transactions as the reference. During validation, pass that fold's training
rows as ``history``; never pass other validation/test rows.
"""

import numpy as np
import pandas as pd


def mark_prior_maximum_missing(data, history, flagged_ids, tolerance=0.01):
    """Return (data_with_flagged_maxima_missing, recovery_audit).

    Parameters
    ----------
    data : pandas.DataFrame
        Rows whose flagged maxima should become NaN. Input is not modified.
    history : pandas.DataFrame
        Reference training purchases with customer_id, purchase_date and
        Total a Pagar. Input is not modified.
    flagged_ids : iterable
        Transaction IDs already identified by flag_prior_maximum.
    tolerance : float
        Absolute monetary tolerance in EUR, configurable by the caller.

    Recovery statuses
    -----------------
    complete_history
        Found count and spend match the provided history.
    incomplete_but_max_verified
        Some purchases are unavailable, but their combined spend cannot
        exceed the largest observed earlier purchase.
    unresolved
        These checks cannot establish the true maximum.

    The audit's verified_maximum is for later use. All flagged cells are
    NaN in the returned data, including those with a verified replacement.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    if data["ID"].isna().any() or not data["ID"].is_unique:
        raise ValueError("data must contain non-missing, unique transaction IDs.")

    flagged_ids = set(flagged_ids)
    unknown_ids = flagged_ids.difference(data["ID"])
    if unknown_ids:
        raise ValueError("Some flagged IDs are not present in data.")

    result = data.copy()
    flagged_mask = result["ID"].isin(flagged_ids)
    selected = result.loc[flagged_mask, [
        "ID", "customer_id", "purchase_date", "prior_purchase_count",
        "prior_total_spend", "prior_max_purchase_value",
    ]].copy()
    selected["purchase_date"] = pd.to_datetime(
        selected["purchase_date"], errors="coerce"
    )
    for column in ["prior_purchase_count", "prior_total_spend"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    reference = history[
        ["customer_id", "purchase_date", "Total a Pagar"]
    ].copy()
    reference["purchase_date"] = pd.to_datetime(
        reference["purchase_date"], errors="coerce"
    )
    reference["Total a Pagar"] = pd.to_numeric(
        reference["Total a Pagar"], errors="coerce"
    )
    reference = reference.dropna(subset=["customer_id", "purchase_date"])
    histories = {
        customer: group for customer, group in reference.groupby("customer_id")
    }
    empty_history = reference.iloc[:0]

    audit_columns = [
        "ID", "customer_id", "purchase_date", "original_maximum",
        "prior_purchase_count", "prior_transactions_found", "count_difference",
        "prior_total_spend", "observed_prior_spend", "spend_difference",
        "observed_maximum", "verified_maximum", "recovery_status", "reason",
    ]
    records = []
    for row in selected.itertuples(index=False):
        record = {
            "ID": row.ID,
            "customer_id": row.customer_id,
            "purchase_date": row.purchase_date,
            "original_maximum": row.prior_max_purchase_value,
            "prior_purchase_count": row.prior_purchase_count,
            "prior_transactions_found": np.nan,
            "count_difference": np.nan,
            "prior_total_spend": row.prior_total_spend,
            "observed_prior_spend": np.nan,
            "spend_difference": np.nan,
            "observed_maximum": np.nan,
            "verified_maximum": np.nan,
            "recovery_status": "unresolved",
            "reason": "Missing customer ID or purchase date.",
        }
        if pd.isna(row.customer_id) or pd.isna(row.purchase_date):
            records.append(record)
            continue

        customer_history = histories.get(row.customer_id, empty_history)
        previous = customer_history.loc[
            customer_history["purchase_date"] < row.purchase_date
        ]
        amounts = previous["Total a Pagar"]
        usable = (np.isfinite(amounts) & amounts.ge(0)).fillna(False)
        observed_spend = float(amounts.loc[usable].sum())
        observed_maximum = float(amounts.loc[usable].max())
        found = len(previous)
        count_difference = row.prior_purchase_count - found
        spend_difference = row.prior_total_spend - observed_spend
        record.update({
            "prior_transactions_found": found,
            "count_difference": count_difference,
            "observed_prior_spend": observed_spend,
            "spend_difference": spend_difference,
            "observed_maximum": observed_maximum,
        })

        valid_summaries = (
            pd.notna(row.prior_purchase_count)
            and np.isfinite(row.prior_purchase_count)
            and row.prior_purchase_count > 0
            and row.prior_purchase_count == int(row.prior_purchase_count)
            and pd.notna(row.prior_total_spend)
            and np.isfinite(row.prior_total_spend)
            and row.prior_total_spend >= 0
        )
        if not valid_summaries:
            record["reason"] = "Historical count or total is missing/invalid."
        elif not usable.all():
            record["reason"] = "An earlier dated row has an unusable purchase amount."
        elif count_difference < 0:
            record["reason"] = "Found more earlier rows than the reported count."
        elif spend_difference < -tolerance:
            record["reason"] = "Observed spend exceeds the reported historical total."
        elif found == 0:
            record["reason"] = "No usable earlier training transactions are available."
        elif count_difference == 0 and abs(spend_difference) <= tolerance:
            record.update({
                "verified_maximum": observed_maximum,
                "recovery_status": "complete_history",
                "reason": "Earlier transaction count and total spend both match.",
            })
        elif (
            count_difference > 0
            and spend_difference <= observed_maximum + tolerance
        ):
            record.update({
                "verified_maximum": observed_maximum,
                "recovery_status": "incomplete_but_max_verified",
                "reason": "Unobserved spend cannot exceed the largest observed purchase.",
            })
        else:
            record["reason"] = "Available history and totals do not establish the maximum."
        records.append(record)

    # Defer reconstruction: even recoverable flagged values become missing.
    result.loc[flagged_mask, "prior_max_purchase_value"] = np.nan
    audit = pd.DataFrame(records, columns=audit_columns)
    return result, audit

def mark_rolling_spend_missing(
    data, history, windows=(30, 90, 180, 365), tolerance=0.01, sentinel=1e9
):
    """Return (cleaned_copy, audit), checking the requested rolling windows.

    Each integer N uses spend_Nd, purchase_count_Nd and reference purchases
    in [purchase_date - N days, purchase_date). Existing missing values stay
    missing and are not labelled as newly detected errors.

    Only flagged cells in requested spend columns are set to NaN. Counts and
    all other fields are preserved. Each audit row identifies one flagged
    cell using ID, window_days and spend_column. Its original_spend and any
    verified_spend are retained for later treatment; no candidate is applied.

    Checks include sentinels, negatives/non-finite amounts, total-spend
    bounds, zero-count identities and reference-history comparisons. Other
    windows provide comparisons through verified amounts, so an unverified
    corrupt neighbour cannot contaminate the checks.

    windows, tolerance and sentinel can be supplied from config.yaml.
    """
    windows = tuple(windows)
    if not windows or any(
        isinstance(days, bool) or not isinstance(days, Integral) or days <= 0
        for days in windows
    ):
        raise ValueError("windows must contain positive integer day counts.")
    windows = tuple(sorted(set(int(days) for days in windows)))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    if data["ID"].isna().any() or not data["ID"].is_unique:
        raise ValueError("data must have non-missing, unique transaction IDs.")
    for days in windows:
        needed = {f"spend_{days}d", f"purchase_count_{days}d"}
        if not needed.issubset(data.columns):
            raise ValueError(f"Missing spend/count columns for {days} days.")

    available = sorted({
        int(match.group(1))
        for column in data.columns
        if (match := re.fullmatch(r"spend_(\d+)d", column))
        and int(match.group(1)) > 0
        and f"purchase_count_{match.group(1)}d" in data.columns
    })
    result = data.copy()
    base = data.reset_index(drop=True).copy()
    base["purchase_date"] = pd.to_datetime(base["purchase_date"], errors="coerce")
    total = _numeric(base, "prior_total_spend")
    lifetime_count = _numeric(base, "prior_purchase_count")
    prepared = _prepare_history(history)
    information, seed_values = {}, {}

    for days in available:
        expected = _numeric(base, f"purchase_count_{days}d")
        info = _window_history(base, prepared, days)
        info["expected_count"] = expected
        info["count_valid"] = _valid_count(expected)
        info["usable_history"] = (
            info["count_valid"] & info["transactions_found"].notna()
            & info["transactions_found"].le(expected)
            & info["unusable_amounts"].eq(0)
        )
        info["complete_window_history"] = (
            info["usable_history"] & info["transactions_found"].eq(expected)
        )
        all_history = (
            info["count_valid"] & _valid_count(lifetime_count)
            & expected.eq(lifetime_count) & total.ge(0)
        )
        seeds = {
            "complete_history": info["observed_spend"].where(
                info["complete_window_history"]
            ),
            "zero_count": pd.Series(0.0, index=base.index).where(
                info["count_valid"] & expected.eq(0)
            ),
            "all_history": total.where(all_history),
        }
        seed_values[days] = _consistent_candidate(seeds, info, total, tolerance)
        information[days] = info

    # Equal counts in nested windows imply identical purchase sets. Use only
    # independently verified neighbours when establishing that identity.
    verified_values = {}
    for days in available:
        info = information[days]
        candidates = {"own_window": seed_values[days]}
        for other in available:
            if other == days:
                continue
            same_count = (
                info["count_valid"] & information[other]["count_valid"]
                & info["expected_count"].eq(information[other]["expected_count"])
            )
            candidates[f"same_count_{other}d"] = seed_values[other].where(same_count)
        verified_values[days] = _consistent_candidate(
            candidates, info, total, tolerance
        )

    audits, empty_template = [], None
    for days in windows:
        column = f"spend_{days}d"
        original = base[column]
        spend = _numeric(base, column)
        info = information[days]
        expected = info["expected_count"]
        verified = verified_values[days]
        all_history = (
            info["count_valid"] & _valid_count(lifetime_count)
            & expected.eq(lifetime_count)
        )
        reasons = pd.DataFrame({
            "sentinel": spend.eq(sentinel),
            "negative": spend.lt(0),
            "non_numeric": original.notna() & spend.isna(),
            "non_finite": spend.notna() & ~np.isfinite(spend),
            "above_historical_total": spend.gt(total + tolerance),
            "nonzero_without_purchases": (
                info["count_valid"] & expected.eq(0) & spend.abs().gt(tolerance)
            ),
            "all_history_count_but_different_spend": (
                all_history & spend.sub(total).abs().gt(tolerance)
            ),
            "below_observed_spend": (
                info["usable_history"]
                & spend.lt(info["observed_spend"] - tolerance)
            ),
            "complete_history_spend_mismatch": (
                info["complete_window_history"]
                & spend.sub(info["observed_spend"]).abs().gt(tolerance)
            ),
            "different_from_verified_spend": (
                verified.notna() & spend.sub(verified).abs().gt(tolerance)
            ),
        }, index=base.index)
        for other in available:
            anchor = verified_values[other]
            if other < days:
                reasons[f"below_verified_{other}d_spend"] = spend.lt(anchor - tolerance)
            elif other > days:
                reasons[f"above_verified_{other}d_spend"] = spend.gt(anchor + tolerance)
        reasons = reasons.fillna(False)
        flagged = original.notna() & reasons.any(axis=1)

        audit = base[["ID", "customer_id", "purchase_date"]].copy()
        audit["window_days"] = days
        audit["spend_column"] = column
        audit["original_spend"] = original
        audit["expected_count"] = expected
        audit["transactions_found"] = info["transactions_found"]
        audit["count_difference"] = expected - info["transactions_found"]
        audit["observed_spend"] = info["observed_spend"]
        audit["unusable_amounts"] = info["unusable_amounts"]
        audit["prior_purchase_count"] = lifetime_count
        audit["prior_total_spend"] = total
        audit["complete_window_history"] = info["complete_window_history"]
        audit["verified_spend"] = verified
        audit["recovery_status"] = "unresolved"
        audit.loc[verified.notna(), "recovery_status"] = "verified_from_row"
        audit.loc[
            verified.notna() & info["complete_window_history"], "recovery_status"
        ] = "complete_history"
        audit["sign_flip_verified"] = (
            spend.lt(0) & spend.abs().sub(verified).abs().le(tolerance)
        )
        reason = pd.Series("", index=base.index, dtype=object)
        for label in reasons.columns:
            reason += np.where(reasons[label], label + "; ", "")
        audit["reason"] = reason.str.rstrip("; ")
        empty_template = audit.iloc[:0].copy()
        if flagged.any():
            audits.append(audit.loc[flagged].copy())
            result.loc[flagged.to_numpy(), column] = np.nan

    combined = pd.concat(audits, ignore_index=True) if audits else empty_template
    return result, combined




def future_spend(group, days=90):
    dates = group["purchase_date"].to_numpy()
    amounts = group["Total a Pagar"].to_numpy()

    totals = []
    for date in dates:
        in_window = (
            (dates > date)
            & (dates <= date + np.timedelta64(days, "D"))
        )
        totals.append(amounts[in_window].sum())

    return pd.Series(totals, index=group.index, dtype=float)


