"""
run_experiment.py — Fixed experiment pipeline.

HOW TO USE
----------
Change the two lines under "# === EXPERIMENT CONFIG ===" to swap in a
different model, then run:

    cd experiments
    python run_experiment.py

The split indices are saved to splits/default_split.npz on the first run
and reused identically on every subsequent run, guaranteeing that all
experiments are evaluated on exactly the same test rows.

ADDING A NEW MODEL
------------------
1. Create experiments/models/my_model.py  (follow any existing model as template)
2. Import it here and set MODEL = MyModel()
3. Set EXPERIMENT_NAME to something descriptive
4. Run.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

# Make sure local modules are importable when running from inside experiments/
sys.path.insert(0, os.path.dirname(__file__))

from data import load_pairs
from report import generate_report

# ---------------------------------------------------------------------------
# === EXPERIMENT CONFIG — change these lines per experiment ===
# ---------------------------------------------------------------------------

from models.xgboost_model import XGBoostModel  # ← swap model import
MODEL = XGBoostModel()                         # ← instantiate model
EXPERIMENT_NAME = "xgboost_matryoshka_all_features"  # ← unique name for this run

# ---------------------------------------------------------------------------
# Pipeline config — usually leave these alone
# ---------------------------------------------------------------------------

ZARR_FILE     = "embeddings.zarr"    # relative to project root (one level up)
MAX_ROWS      = None                 # set e.g. 50_000 for fast smoke-tests
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
SPLITS_DIR    = os.path.join(os.path.dirname(__file__), "splits")
SPLIT_FILE    = os.path.join(SPLITS_DIR, "default_split.npz")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")

# CosineBaseline exposes a custom threshold; other models default to 0.5
THRESHOLD = getattr(MODEL, "threshold", 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_create_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (train_idx, test_idx) for a dataset of size n.

    On the first call the split is computed and saved to SPLIT_FILE.
    On every subsequent call the saved indices are loaded — even if n changes,
    we verify the indices are in range and raise clearly if not.
    """
    if os.path.exists(SPLIT_FILE):
        data = np.load(SPLIT_FILE)
        train_idx = data["train_idx"]
        test_idx  = data["test_idx"]
        print(
            f"[split] Loaded existing split from {SPLIT_FILE} "
            f"(train={len(train_idx)}, test={len(test_idx)})",
            flush=True,
        )
        if train_idx.max() >= n or test_idx.max() >= n:
            raise RuntimeError(
                f"Saved split has indices up to {max(train_idx.max(), test_idx.max())} "
                f"but dataset only has {n} rows. "
                f"Delete {SPLIT_FILE} to regenerate."
            )
        return train_idx, test_idx

    # First run — create and save
    indices = np.arange(n)
    # We need labels for stratified split; load them cheaply
    # (we don't have y yet, so we pass a dummy — caller must pass y)
    raise RuntimeError(
        "_load_or_create_split called before labels are available; "
        "use _get_split(n, y) instead."
    )


def _get_split(n: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx), creating and saving the split if needed."""
    if os.path.exists(SPLIT_FILE):
        return _load_or_create_split(n)

    print(f"[split] No saved split found — creating and saving to {SPLIT_FILE}", flush=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    np.savez(SPLIT_FILE, train_idx=train_idx, test_idx=test_idx)
    print(
        f"[split] Saved split to {SPLIT_FILE} "
        f"(train={len(train_idx)}, test={len(test_idx)})",
        flush=True,
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"[run] Experiment  : {EXPERIMENT_NAME}", flush=True)
    print(f"[run] Model       : {getattr(MODEL, 'name', type(MODEL).__name__)}", flush=True)
    print(f"[run] Threshold   : {THRESHOLD}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    # Zarr lives one directory up relative to this script
    zarr_path = os.path.join(os.path.dirname(__file__), "..", ZARR_FILE)
    records = load_pairs(
        zarr_file=zarr_path,
        max_rows=MAX_ROWS,
    )

    # ------------------------------------------------------------------
    # 2. Build features (model owns this step)
    # ------------------------------------------------------------------
    print(f"\n[run] Building features with {getattr(MODEL, 'name', type(MODEL).__name__)}...", flush=True)
    X, y, feature_names = MODEL.build_features(records)
    print(f"[run] Feature matrix: {X.shape}  labels: {y.shape}", flush=True)

    # ------------------------------------------------------------------
    # 3. Fixed train/test split (saved on first run, reused thereafter)
    # ------------------------------------------------------------------
    train_idx, test_idx = _get_split(len(records), y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_records = [records[i] for i in test_idx]

    print(f"[run] Train: {len(train_idx)}  Test: {len(test_idx)}", flush=True)

    # ------------------------------------------------------------------
    # 4. Fit
    # ------------------------------------------------------------------
    print(f"\n[run] Fitting model...", flush=True)
    t_fit = time.time()
    MODEL.fit(X_train, y_train)
    print(f"[run] Fit complete in {time.time() - t_fit:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    proba = MODEL.predict_proba(X_test)

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print(f"\n[run] Generating report...", flush=True)
    generate_report(
        experiment_name=EXPERIMENT_NAME,
        y_true=y_test,
        proba=proba,
        test_records=test_records,
        feature_names=feature_names,
        model=MODEL,
        threshold=THRESHOLD,
        results_dir=RESULTS_DIR,
    )

    print(f"\n[run] Total wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    run()
