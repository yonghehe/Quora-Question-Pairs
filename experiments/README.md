# experiments/

Plug-and-play ML experiment harness for the Quora Question-Pairs project.

## Structure

```
experiments/
├── data.py              Loads zarr + CSV → list[PairRecord]  (shared, no model logic)
├── features.py          Primitive feature functions (embedding, lexical, all)
├── report.py            Metrics printer + results writer
├── run_experiment.py    ← ENTRY POINT — run this
│
├── models/
│   ├── catboost_model.py
│   ├── logreg_model.py
│   ├── cosine_baseline.py
│   └── xgboost_model.py
│
├── splits/
│   └── default_split.npz   (auto-created on first run, reused forever)
│
└── results/
    ├── all_experiments.csv          one row per completed run
    ├── catboost_all_features/
    │   ├── metrics.txt
    │   ├── errors.csv
    │   └── feature_importance.txt
    └── …
```

## Running an experiment

```bash
cd experiments
python run_experiment.py
```

The first run saves `splits/default_split.npz`.  
Every subsequent run loads those exact indices so all models are
compared on the **same test rows**.

## Switching models / features

Open `run_experiment.py` and change the three lines in the
`EXPERIMENT CONFIG` block:

```python
from models.xgboost_model import XGBoostModel      # ← model import
MODEL = XGBoostModel()                             # ← model instance
EXPERIMENT_NAME = "xgboost_matryoshka_all_features"  # ← unique folder name
```

For matryoshka-aware experiments, pass explicit prefix dims if desired:

```python
MODEL = XGBoostModel(matryoshka_dims=(128, 256, 512, 1024, 2560))
```

## Adding a new model

1. Copy any existing file in `models/` as a starting point.
2. Implement:
   - `build_features(records) → (X, y, feature_names)`
   - `fit(X_train, y_train)`
   - `predict_proba(X_test) → 1-D float32 array`
3. Optionally add `feature_importances() → dict[str, float]` for
   automatic importance reporting.
4. Import and set `MODEL = YourModel()` in `run_experiment.py`.

## Adding a new feature set

Add a function to `features.py` that takes a `PairRecord` and returns
`dict[str, float]`.  Reference it from your model's `_feature_fn`.

This repo also includes:
- `matryoshka_embedding_features(...)`
- `matryoshka_all_features(...)`

which compute embedding statistics over prefix slices (e.g. 128→2560)
for models that want multi-scale matryoshka signals.

## Results comparison

After multiple runs, open `results/all_experiments.csv` to compare all
experiments side-by-side (accuracy, precision, recall, F1, TP/FP/TN/FN).
