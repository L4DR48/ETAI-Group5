"""The preprocessor that goes inside the model Pipeline.

1. fill_no_history: structural gaps (the customer has no history yet) get a fixed value.
2. Numeric columns: optional capping -> optional log1p -> scaler -> imputer for the random gaps.
3. Categorical columns: missing becomes its own category -> one-hot encoding.

Structure from class 10 (10_missing_values.ipynb): _NUMERIC_IMPUTERS chosen with numeric_strategy,
and numeric columns scaled first, then imputed. Everything learned (quantiles, scaler, imputer,
encoder) is fitted on the training rows only, and never sees the target.
"""
import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer   # noqa: F401  (makes IterativeImputer importable, as in class 10)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

# Number of past purchases a customer needs before each column can exist
NEEDS_PAST_PURCHASES = {
    "days_since_previous_purchase": 1,
    "prior_min_purchase_value": 1,
    "prior_mean_purchase_value": 1,
    "prior_max_purchase_value": 1,
    "prior_std_purchase_value": 2,
    "prior_mean_gap_days": 2,
    "prior_std_gap_days": 3,
    "gap_ratio": 2,
}
DAY_COLUMNS = ["days_since_previous_purchase", "prior_mean_gap_days"]

_SCALERS = {"none": "passthrough", "standard": StandardScaler, "robust": RobustScaler}

# Numeric imputers, copied from class 10. "iterative" is MICE with one completed dataset;
# "random_forest" is the missForest idea. They learn only from the feature columns, never the target.
_NUMERIC_IMPUTERS = {
    "median": lambda cfg, seed: SimpleImputer(strategy="median"),
    "mean": lambda cfg, seed: SimpleImputer(strategy="mean"),
    "knn": lambda cfg, seed: KNNImputer(n_neighbors=cfg.get("n_neighbors", 5)),
    "iterative": lambda cfg, seed: IterativeImputer(max_iter=cfg.get("max_iter", 10), random_state=seed),
    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=cfg.get("n_estimators", 50), min_samples_leaf=5,
                                        random_state=seed, n_jobs=-1),
        max_iter=cfg.get("max_iter", 10), random_state=seed),
}


def load_config(path="config.yaml"):
    """Read the project settings."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fill_no_history(X, no_history_days, no_history_ratio):
    """Structural gaps: fill a value only where it is missing because the customer has too few
    past purchases. Value columns become 0 (no past purchase, so no past value or variation);
    day columns become no_history_days ("never bought" is closest to "bought very long ago");
    gap_ratio becomes no_history_ratio (these customers repeat like the "very late" ones).
    Row-wise, learns nothing. Any other NaN is left for the imputer."""
    X = X.copy()
    for col, needed in NEEDS_PAST_PURCHASES.items():
        no_history = X[col].isna() & (X["prior_purchase_count"] < needed)
        if col in DAY_COLUMNS:
            X.loc[no_history, col] = no_history_days
        elif col == "gap_ratio":
            X.loc[no_history, col] = no_history_ratio
        else:
            X.loc[no_history, col] = 0
    return X


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Caps each column at a lower and upper quantile learned on the training rows (NaN ignored).
    Either side can be None, e.g. lower=None caps only the high values."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        self.low_ = None if self.lower is None else X.quantile(self.lower)     # pandas quantile skips NaN
        self.high_ = None if self.upper is None else X.quantile(self.upper)
        return self

    def transform(self, X):
        return X.clip(lower=self.low_, upper=self.high_, axis=1)   # NaN stays NaN

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def to_object(X):
    """Categories as plain Python objects with np.nan, which SimpleImputer handles reliably."""
    return X.astype(object).where(X.notna(), np.nan)


def build_preprocessor(config):
    """Unfitted preprocessor described by the "preprocessing" section of config.yaml."""
    imputation = config.get("imputation", {})
    numeric_strategy = imputation.get("numeric_strategy", "median")
    if numeric_strategy not in _NUMERIC_IMPUTERS:
        raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}. Options: {list(_NUMERIC_IMPUTERS)}")

    # Numeric branch: optional capping -> optional log1p -> scaler -> imputer (class 10: scale first, then impute)
    numeric_steps = []
    if config.get("cap_quantiles"):
        lower, upper = config["cap_quantiles"]
        numeric_steps.append(("cap", QuantileCapper(lower, upper)))
    if config.get("log1p_columns"):
        log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        numeric_steps.append(("log1p", ColumnTransformer([("log", log1p, config["log1p_columns"])],
                                                         remainder="passthrough", verbose_feature_names_out=False)))
    scaler = _SCALERS[config.get("scaler", "robust")]
    numeric_steps.append(("scale", scaler() if callable(scaler) else scaler))
    imputer = _NUMERIC_IMPUTERS[numeric_strategy](imputation, imputation.get("random_state"))
    numeric_steps.append(("impute", imputer.set_params(add_indicator=config.get("add_indicator", False))))

    # Categorical branch: missing is its own category; rare and unseen levels go to "infrequent"
    categorical_steps = [
        ("to_object", FunctionTransformer(to_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=config.get("min_frequency", 20), sparse_output=False)),
    ]

    columns = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), config["numeric_features"]),
        ("categorical", Pipeline(categorical_steps), config["categorical_features"]),
    ])
    no_history = FunctionTransformer(fill_no_history, feature_names_out="one-to-one",
                                     kw_args={"no_history_days": config["no_history_days"],
                                              "no_history_ratio": config["no_history_ratio"]})
    preprocessor = Pipeline([("no_history", no_history), ("columns", columns)])
    return preprocessor.set_output(transform="pandas")   # keep column names at every step
