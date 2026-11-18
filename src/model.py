"""Feature selection for the pipeline: VoteSelector (class 08, Section 4)."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import Lasso, LogisticRegression


class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Keeps the features chosen by at least `min_votes` of three models, fitted on the training
    rows only: a random forest and gradient boosting vote for features with above-average
    importance, an L1-penalised linear model for features it doesn't shrink to zero.

    task="classification" is the class 08 version, unchanged. task="regression" uses the
    regressor version of each voter, with Lasso(alpha=lasso_alpha) as the L1 model.
    """

    def __init__(self, min_votes: int = 2, random_state=None, task: str = "classification",
                 lasso_alpha: float = 0.01):
        self.min_votes = min_votes
        self.random_state = random_state
        self.task = task
        self.lasso_alpha = lasso_alpha

    def fit(self, X, y):
        if self.task == "classification":      # class 08, Section 4
            forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
            booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
            lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(X, y)
        elif self.task == "regression":        # same three kinds of voter, for a numeric target
            forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
            booster = GradientBoostingRegressor(random_state=self.random_state).fit(X, y)
            lasso = Lasso(alpha=self.lasso_alpha).fit(X, y)
        else:
            raise ValueError(f"Unknown task: {self.task}. Options: classification, regression")
        # np.ravel: LogisticRegression gives coef_ as 1 row, Lasso as a flat array
        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(np.ravel(lasso.coef_)) > 1e-8))
        self.support_ = self.votes_ >= self.min_votes
        return self

    def _get_support_mask(self):
        return self.support_
