"""
models/logreg_model.py — Logistic Regression classifier.

Feature set: all embedding features + all lexical features.
Applies StandardScaler internally (LR is sensitive to feature scale).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import all_features, build_matrix


class LogRegModel:
    """
    Logistic Regression with internal StandardScaler.

    Interface required by run_experiment.py
    ----------------------------------------
    build_features(records)  → (X, y, feature_names)
    fit(X_train, y_train)
    predict_proba(X_test)    → 1-D array of positive-class probabilities
    """

    name = "LogisticRegression"

    def __init__(self, max_iter: int = 1000, class_weight: str = "balanced", random_state: int = 42, **kwargs):
        self._lr = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs,
        )
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_fn(r: PairRecord) -> dict[str, float]:
        return all_features(r)

    def build_features(
        self, records: list[PairRecord]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        X, feature_names = build_matrix(records, self._feature_fn)
        y = np.array([r.label for r in records], dtype=np.int32)
        self._feature_names = feature_names
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Sklearn-style interface (scaler is baked in)
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_train)
        self._lr.fit(X_scaled, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X_test)
        return self._lr.predict_proba(X_scaled)[:, 1].astype(np.float32)
