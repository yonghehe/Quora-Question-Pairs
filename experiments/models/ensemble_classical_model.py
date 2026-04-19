"""
models/ensemble_classical_model.py — Tuned three-way ensemble:
    XGBoostClassicalModel  + GRUModelV3 (tuned)  + CatBoostModel

Best hyperparameters are loaded automatically from the tuning artefacts in
experiments/tuning/ at instantiation time:

  • experiments/tuning/xgboost_best_params.json  → XGBoostClassicalModel
      Format: {"best_params": {...}, "best_score": ..., ...}
      Loaded via model.apply_tuned_params().

  • experiments/tuning/gru3params.json           → GRUModelV3
      Format: {"model_config": {"hidden_size": ..., "lr": ..., ...}, ...}
      Key fields extracted: hidden_size, num_layers, dropout, mlp_hidden,
      lr, weight_decay, threshold.

  • CatBoostModel uses its built-in defaults (no tuning file produced yet).

Design notes — stub-matrix trick
---------------------------------
Each member owns a completely different feature space:
  - XGBoostClassicalModel : matryoshka stats + TF-IDF + char-ngrams + topics
  - GRUModelV3            : raw embeddings (2×2560) + 6 scalar bridge features
  - CatBoostModel         : matryoshka stats + lexical features

A shared feature matrix is therefore impossible.  The same stub-matrix
approach used in EnsembleModel is applied here:

  build_features(records, train_idx=None)
    Calls each member's build_features() independently (forwarding train_idx
    to members that declare it in their signature, such as
    XGBoostClassicalModel whose featurizers must be fit on training data only
    to prevent leakage).  Returns a thin (N, n_members) stub whose column 0
    holds the original row index.  run_experiment.py slices this stub with
    train_idx / test_idx; fit() and predict_proba() recover the correct rows
    from the stored per-member full arrays via those indices.

  fit(X_train_stub, y_train)
    Reads train_idx from stub[:,0] and calls each member's fit() on its
    stored training slice.

  predict_proba(X_test_stub)
    Reads test_idx from stub[:,0], collects each member's probabilities, and
    combines them with the chosen strategy.

Strategies
----------
  "mean"     : (optionally weighted) average of base-model probabilities.
  "stacking" : trains a LogReg meta-learner on out-of-fold probability vectors
               produced by the base models.

Adding to run_experiment.py
---------------------------
  from models import EnsembleClassicalModel

  MODEL_REGISTRY = {
      ...
      "ensemble_classical_mean":  EnsembleClassicalModel(strategy="mean"),
      "ensemble_classical_weighted": EnsembleClassicalModel(
                                         strategy="mean",
                                         weights=[2.0, 1.0, 2.0],
                                     ),
      "ensemble_classical_stack": EnsembleClassicalModel(strategy="stacking"),
  }

Then run as normal:
  python run_experiment.py --model ensemble_classical_mean --name my_run
"""

from __future__ import annotations

import inspect
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord


# ---------------------------------------------------------------------------
# Helpers: load best hyperparameters from tuning artefacts
# ---------------------------------------------------------------------------

_TUNING_DIR = os.path.join(os.path.dirname(__file__), "..", "tuning")


def _load_xgboost_classical_params() -> dict:
    """
    Load XGBoostClassicalModel best params from tuning/xgboost_best_params.json.

    Expected schema (written by experiments/tune.py):
        { "best_params": { "max_depth": ..., ... }, "best_score": ..., ... }

    Returns an empty dict if the file is absent so the model falls back to
    its built-in defaults.
    """
    path = os.path.join(_TUNING_DIR, "xgboost_best_params.json")
    if not os.path.exists(path):
        print(
            f"[EnsembleClassical] WARNING: {path} not found — "
            "XGBoostClassicalModel will use built-in defaults.",
            flush=True,
        )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("best_params", {})
    if not isinstance(params, dict) or not params:
        print(
            f"[EnsembleClassical] WARNING: {path} has no usable 'best_params' "
            "— XGBoostClassicalModel will use built-in defaults.",
            flush=True,
        )
        return {}
    print(
        f"[EnsembleClassical] Loaded XGBoostClassical params from {path}: {params}",
        flush=True,
    )
    return {"best_params": params, "source": path, "cv_score": data.get("best_score")}


def _load_gru3_params() -> dict:
    """
    Load GRUModelV3 best hyperparameters from tuning/gru3params.json.

    This file is a result report (not a tune.py best_params.json), so the
    relevant fields live under the "model_config" key.  We extract only the
    constructor-level knobs that GRUModelV3.__init__(**overrides) accepts.

    Returns an empty dict if the file is absent.
    """
    path = os.path.join(_TUNING_DIR, "gru3params.json")
    if not os.path.exists(path):
        print(
            f"[EnsembleClassical] WARNING: {path} not found — "
            "GRUModelV3 will use built-in defaults.",
            flush=True,
        )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("model_config", {})
    # Extract the subset of keys that GRUModelV3 __init__ accepts as **overrides
    _WANTED = {
        "hidden_size", "num_layers", "dropout", "mlp_hidden",
        "lr", "weight_decay", "threshold",
    }
    params = {k: v for k, v in cfg.items() if k in _WANTED}
    if not params:
        print(
            f"[EnsembleClassical] WARNING: {path} has no extractable params "
            "— GRUModelV3 will use built-in defaults.",
            flush=True,
        )
        return {}
    print(
        f"[EnsembleClassical] Loaded GRUModelV3 params from {path}: {params}",
        flush=True,
    )
    return params


# ---------------------------------------------------------------------------
# Factory: build the three member models with their best hyperparameters
# ---------------------------------------------------------------------------

def _make_members() -> list:
    """
    Instantiate the three ensemble members with their best hyperparameters.

    Imports are deferred and the models directory is added to sys.path at
    call-time to avoid circular-import issues and to ensure the model
    modules are always resolvable regardless of from where Python is invoked.
    """
    _models_dir = os.path.dirname(os.path.abspath(__file__))
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)

    from xgboost_classical import XGBoostClassicalModel   # noqa: E402
    from gru_model_v3 import GRUModelV3                   # noqa: E402
    from catboost_model import CatBoostModel               # noqa: E402

    # ── XGBoostClassicalModel ──────────────────────────────────────────────
    xgb = XGBoostClassicalModel()
    xgb_info = _load_xgboost_classical_params()
    if xgb_info:
        xgb.apply_tuned_params(
            xgb_info["best_params"],
            source=xgb_info.get("source"),
            cv_score=xgb_info.get("cv_score"),
            method="tuning_file",
        )

    # ── GRUModelV3 (tuned) ─────────────────────────────────────────────────
    gru3_overrides = _load_gru3_params()
    gru = GRUModelV3(**gru3_overrides)

    # ── CatBoostModel (defaults; no tuning file yet) ───────────────────────
    catboost = CatBoostModel()

    return [xgb, gru, catboost]


# ---------------------------------------------------------------------------
# EnsembleClassicalModel
# ---------------------------------------------------------------------------

class EnsembleClassicalModel:
    """
    Three-way ensemble of XGBoostClassicalModel + GRUModelV3 + CatBoostModel,
    each initialised with the best hyperparameters stored in experiments/tuning/.

    Parameters
    ----------
    strategy   : "mean" or "stacking"
    weights    : optional list of 3 floats for weighted mean (ignored for
                 stacking).  E.g. [2, 1, 2] up-weights the two tree models.
    meta_folds : number of CV folds for the OOF stacking phase (default 5).
    threshold  : decision threshold exposed to run_experiment.py (default 0.5).
    """

    name = "EnsembleClassical"

    def __init__(
        self,
        strategy: str = "mean",
        weights: list[float] | None = None,
        meta_folds: int = 5,
        threshold: float = 0.5,
    ):
        if strategy not in ("mean", "stacking"):
            raise ValueError(
                f"strategy must be 'mean' or 'stacking', got {strategy!r}"
            )
        if weights is not None and len(weights) != 3:
            raise ValueError("weights must be a list of 3 floats (one per member)")

        self.members    = _make_members()
        self.strategy   = strategy
        self.weights    = weights
        self.meta_folds = meta_folds
        self.threshold  = threshold

        # Full per-member feature arrays — set in build_features, indexed via
        # the row indices carried in column 0 of the stub matrix.
        self._member_X_all: list[np.ndarray] = []
        self._train_idx: np.ndarray = np.empty(0, dtype=np.intp)

        # Stacking components
        self._meta_scaler = StandardScaler()
        self._meta_clf    = LogisticRegression(max_iter=1000, random_state=42)

        # Feature names reported to run_experiment.py / report.py
        self._feature_names: list[str] = [
            f"base_{getattr(m, 'name', type(m).__name__)}_proba"
            for m in self.members
        ]

    # ------------------------------------------------------------------ #
    # build_features                                                       #
    # ------------------------------------------------------------------ #

    def build_features(
        self,
        records: list[PairRecord],
        train_idx: "np.ndarray | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray, list[str]]":
        """
        Call each member's build_features() independently.

        train_idx is forwarded to any member whose build_features() signature
        declares it (currently XGBoostClassicalModel).  This is critical for
        preventing data leakage from the TF-IDF / char-ngram / topic
        featurizers that must be fit on training data only.

        Returns a thin stub matrix (N, n_members) where column 0 holds the
        original row index, so that fit() and predict_proba() can index the
        stored per-member arrays without any contiguous-block assumption.
        """
        self._member_X_all = []
        ys: list[np.ndarray] = []

        for i, model in enumerate(self.members):
            mname = getattr(model, "name", type(model).__name__)
            print(
                f"  [EnsembleClassical] build_features: "
                f"member {i + 1}/{len(self.members)} — {mname}",
                flush=True,
            )
            # Forward train_idx only to models that declare the parameter
            bf_sig = inspect.signature(model.build_features)
            if "train_idx" in bf_sig.parameters and train_idx is not None:
                print(
                    f"  [EnsembleClassical]   → passing train_idx "
                    f"({len(train_idx):,} rows) to {mname}",
                    flush=True,
                )
                X_m, y_m, _ = model.build_features(records, train_idx=train_idx)
            else:
                X_m, y_m, _ = model.build_features(records)

            self._member_X_all.append(X_m)
            ys.append(y_m)

        # Sanity-check: all members must produce identical labels
        for j, y_m in enumerate(ys[1:], start=1):
            if not np.array_equal(ys[0], y_m):
                raise RuntimeError(
                    f"Member {j} produced different labels than member 0. "
                    "All members must operate on the same records."
                )

        y = ys[0]
        n = len(records)

        # Stub: column 0 carries the original row index so fit() / predict_proba()
        # can slice the stored arrays correctly after run_experiment.py shuffles.
        stub = np.zeros((n, max(len(self.members), 1)), dtype=np.float32)
        stub[:, 0] = np.arange(n, dtype=np.float32)
        return stub, y, self._feature_names

    # ------------------------------------------------------------------ #
    # fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X_train_stub: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit every base model on its own feature slice.

        Column 0 of X_train_stub carries the original row indices placed by
        build_features().  We use those to index each member's stored full-
        dataset X, regardless of how run_experiment.py shuffled the split.
        """
        self._train_idx   = X_train_stub[:, 0].astype(np.intp)
        member_X_trains   = [X_m[self._train_idx] for X_m in self._member_X_all]

        if self.strategy == "stacking":
            self._fit_stacking(member_X_trains, y_train)
        else:
            self._fit_mean(member_X_trains, y_train)

    def _fit_mean(
        self,
        member_X_trains: list[np.ndarray],
        y_train: np.ndarray,
    ) -> None:
        """Fit each base model independently on its training slice."""
        for i, (model, X_m) in enumerate(zip(self.members, member_X_trains)):
            mname = getattr(model, "name", type(model).__name__)
            print(
                f"  [EnsembleClassical] Fitting member "
                f"{i + 1}/{len(self.members)} — {mname}",
                flush=True,
            )
            model.fit(X_m, y_train)

    def _fit_stacking(
        self,
        member_X_trains: list[np.ndarray],
        y_train: np.ndarray,
    ) -> None:
        """
        1. Collect OOF predictions from each base model via k-fold CV.
        2. Train a LogReg meta-learner on the OOF probability matrix.
        3. Re-fit every base model on the full training set for inference.
        """
        n   = len(y_train)
        oof = np.zeros((n, len(self.members)), dtype=np.float32)

        skf = StratifiedKFold(
            n_splits=self.meta_folds, shuffle=True, random_state=42
        )

        for fold_i, (tr_idx, val_idx) in enumerate(
            skf.split(np.arange(n), y_train)
        ):
            print(
                f"  [EnsembleClassical] Stacking fold "
                f"{fold_i + 1}/{self.meta_folds} "
                f"(train={len(tr_idx)}, val={len(val_idx)})",
                flush=True,
            )
            for j, (model, X_m) in enumerate(zip(self.members, member_X_trains)):
                mname = getattr(model, "name", type(model).__name__)
                print(
                    f"    [EnsembleClassical]   member "
                    f"{j + 1}/{len(self.members)} — {mname}",
                    flush=True,
                )
                model.fit(X_m[tr_idx], y_train[tr_idx])
                oof[val_idx, j] = model.predict_proba(X_m[val_idx])

        # Train meta-learner
        print(
            "  [EnsembleClassical] Training meta-learner on OOF predictions …",
            flush=True,
        )
        oof_scaled = self._meta_scaler.fit_transform(oof)
        self._meta_clf.fit(oof_scaled, y_train)
        print(
            f"  [EnsembleClassical] Meta-learner coefficients: "
            f"{self._meta_clf.coef_.round(3)}",
            flush=True,
        )

        # Re-fit every base model on the *full* training set
        print(
            "  [EnsembleClassical] Re-fitting base models on full training set …",
            flush=True,
        )
        for i, (model, X_m) in enumerate(zip(self.members, member_X_trains)):
            mname = getattr(model, "name", type(model).__name__)
            print(
                f"  [EnsembleClassical] Final fit — member "
                f"{i + 1}/{len(self.members)} — {mname}",
                flush=True,
            )
            model.fit(X_m, y_train)

    # ------------------------------------------------------------------ #
    # predict_proba                                                        #
    # ------------------------------------------------------------------ #

    def predict_proba(self, X_test_stub: np.ndarray) -> np.ndarray:
        """
        Recover each member's test rows from the stored full arrays, collect
        probability vectors, and combine via the chosen strategy.

        Column 0 of X_test_stub carries the original row indices placed by
        build_features() and preserved by run_experiment.py's numpy slice.
        """
        test_idx = X_test_stub[:, 0].astype(np.intp)
        member_X_tests = [X_m[test_idx] for X_m in self._member_X_all]

        probas = np.stack(
            [
                model.predict_proba(X_t)
                for model, X_t in zip(self.members, member_X_tests)
            ],
            axis=1,
        )  # (N_test, n_members)

        if self.strategy == "stacking":
            return (
                self._meta_clf.predict_proba(
                    self._meta_scaler.transform(probas)
                )[:, 1]
                .astype(np.float32)
            )

        # "mean" — optionally weighted
        w = (
            np.array(self.weights, dtype=np.float32)
            if self.weights is not None
            else np.ones(len(self.members), dtype=np.float32)
        )
        w = w / w.sum()
        return (probas * w).sum(axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Optional extras consumed by report.py                               #
    # ------------------------------------------------------------------ #

    def get_config(self) -> dict:
        member_configs = []
        for m in self.members:
            if hasattr(m, "get_config"):
                try:
                    member_configs.append(m.get_config())
                except Exception as exc:
                    member_configs.append({"error": str(exc)})
            else:
                member_configs.append(
                    {"model_class": getattr(m, "name", type(m).__name__)}
                )

        return {
            "model_class":    "EnsembleClassicalModel",
            "strategy":       self.strategy,
            "weights":        self.weights,
            "meta_folds":     self.meta_folds if self.strategy == "stacking" else None,
            "threshold":      self.threshold,
            "n_members":      len(self.members),
            "members":        [getattr(m, "name", type(m).__name__) for m in self.members],
            "member_configs": member_configs,
        }
