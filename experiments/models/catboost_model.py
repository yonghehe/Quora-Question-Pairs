"""
models/catboost_model.py — CatBoost classifier.

Feature set: all embedding features + all lexical features (22 total).
No scaling required (tree-based model).
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import all_features, build_matrix


# Default hyper-parameters — override by subclassing or passing kwargs to __init__
_DEFAULTS = dict(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=42,
    verbose=100,
)


class CatBoostModel:
    """
    Plug-and-play wrapper around CatBoostClassifier.

    Interface required by run_experiment.py
    ----------------------------------------
    build_features(records)  → (X, y, feature_names)
    fit(X_train, y_train)
    predict_proba(X_test)    → 1-D array of positive-class probabilities
    feature_importances()    → dict[feature_name, importance]  (optional)
    """

    name = "CatBoost"

    def __init__(self, **kwargs):
        params = {**_DEFAULTS, **kwargs}
        self._model = CatBoostClassifier(**params)
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_fn(r: PairRecord) -> dict[str, float]:
        """All embedding + lexical features (same as original script)."""
        return all_features(r)

    def build_features(
        self, records: list[PairRecord]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build the feature matrix from a list of PairRecords.

        Returns
        -------
        X            : float32 array  (N, F)
        y            : int32  array   (N,)
        feature_names: list[str]
        """
        X, feature_names = build_matrix(records, self._feature_fn)
        y = np.array([r.label for r in records], dtype=np.int32)
        self._feature_names = feature_names
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Sklearn-style interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Optional extras consumed by report.py
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """Returns a name → importance mapping (only valid after fit)."""
        importances = self._model.get_feature_importance()
        return dict(zip(self._feature_names, importances.tolist()))
