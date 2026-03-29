"""
models/xgboost_model.py — XGBoost classifier with matryoshka-sliced features.

Feature set:
  - Matryoshka prefix-slice embedding statistics
  - Lexical overlap / length features
"""

from __future__ import annotations

import os
import sys

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import build_matrix, matryoshka_all_features


_DEFAULTS = dict(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)


class XGBoostModel:
    """Plug-and-play wrapper around xgboost.XGBClassifier."""

    name = "XGBoost"

    def __init__(
        self,
        matryoshka_dims: tuple[int, ...] | None = None,
        **kwargs,
    ):
        params = {**_DEFAULTS, **kwargs}
        self._model = XGBClassifier(**params)
        self._dims = matryoshka_dims
        self._feature_names: list[str] = []

    @property
    def matryoshka_dims(self) -> tuple[int, ...] | None:
        return self._dims

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _feature_fn(self, r: PairRecord) -> dict[str, float]:
        return matryoshka_all_features(r, dims=self._dims)

    def build_features(
        self,
        records: list[PairRecord],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))
