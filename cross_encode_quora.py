import time
import os
import csv

from dotenv import load_dotenv
load_dotenv()  # loads KAGGLE_USERNAME and KAGGLE_KEY from .env

import kagglehub
import numpy as np
import zarr
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_FILE = "cross_encoder_scores.zarr"
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# ---------------------------------------------------------------------------
# Load dataset — collect all pairs (index, qid1, qid2, q1_text, q2_text)
# ---------------------------------------------------------------------------
print("[INFO] Loading dataset...", flush=True)
path = kagglehub.dataset_download("quora/question-pairs-dataset")
print("Path to dataset files:", path)

# Find the CSV file
csv_file = os.path.join(path, "questions.csv")
if not os.path.exists(csv_file):
    for fname in os.listdir(path):
        if fname.endswith(".csv"):
            csv_file = os.path.join(path, fname)
            break

REQUIRED_HEADERS = {"id", "qid1", "qid2", "question1", "question2", "is_duplicate"}

# Check headers of the found file; fall back to any CSV with required headers
with open(csv_file, newline="", encoding="utf-8") as f:
    headers = set(csv.DictReader(f).fieldnames or [])
if not REQUIRED_HEADERS.issubset(headers):
    found = False
    for fname in os.listdir(path):
        if fname.endswith(".csv"):
            candidate = os.path.join(path, fname)
            with open(candidate, newline="", encoding="utf-8") as f:
                h = set(csv.DictReader(f).fieldnames or [])
            if REQUIRED_HEADERS.issubset(h):
                csv_file = candidate
                found = True
                break
    if not found:
        raise FileNotFoundError(
            f"No CSV with headers {sorted(REQUIRED_HEADERS)} found in {path}"
        )

indices: list[int] = []
qid1s:   list[int] = []
qid2s:   list[int] = []
pairs:   list[tuple[str, str]] = []  # (question1, question2)

with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            idx  = int(row["id"])
            qid1 = int(row["qid1"])
            qid2 = int(row["qid2"])
            q1   = row["question1"] or ""
            q2   = row["question2"] or ""
        except (KeyError, ValueError, TypeError):
            continue
        indices.append(idx)
        qid1s.append(qid1)
        qid2s.append(qid2)
        pairs.append((q1, q2))

N = len(pairs)
print(f"[INFO] Total question pairs to encode: {N}", flush=True)

# ---------------------------------------------------------------------------
# Load cross-encoder model
# ---------------------------------------------------------------------------
print(f"[INFO] Loading cross-encoder model: {MODEL_NAME}", flush=True)
model = CrossEncoder(MODEL_NAME)

# ---------------------------------------------------------------------------
# Open zarr store
# ---------------------------------------------------------------------------
store = zarr.open(OUTPUT_FILE, mode="w")

# index array: row index from the original CSV
index_arr = store.zeros(
    name="index",
    shape=(N,),
    dtype="int64",
    chunks=(BATCH_SIZE,),
)
index_arr[:] = np.array(indices, dtype=np.int64)

# qid1 / qid2 arrays
qid1_arr = store.zeros(name="qid1", shape=(N,), dtype="int64", chunks=(BATCH_SIZE,))
qid1_arr[:] = np.array(qid1s, dtype=np.int64)

qid2_arr = store.zeros(name="qid2", shape=(N,), dtype="int64", chunks=(BATCH_SIZE,))
qid2_arr[:] = np.array(qid2s, dtype=np.int64)

# cross_encoder_score array — will be filled in batches
score_arr = store.full(
    name="cross_encoder_score",
    shape=(N,),
    fill_value=float("nan"),
    dtype="float32",
    chunks=(BATCH_SIZE,),
)

# ---------------------------------------------------------------------------
# Score in batches
# ---------------------------------------------------------------------------
print(
    f"[INFO] Starting cross-encoder scoring of {N} pairs "
    f"in batches of {BATCH_SIZE}...",
    flush=True,
)
score_start = time.time()
last_log_time = score_start

for i in range(0, N, BATCH_SIZE):
    batch = pairs[i : i + BATCH_SIZE]
    scores = model.predict(batch, show_progress_bar=False)
    # scores is a numpy array of shape (batch_size,)
    score_arr[i : i + len(batch)] = scores.astype(np.float32)

    now = time.time()
    elapsed = now - score_start
    done = i + len(batch)
    pct  = done / N * 100
    rate = done / elapsed if elapsed > 0 else 0
    eta  = (N - done) / rate if rate > 0 else 0

    if (now - last_log_time) >= 30 or i == 0:
        print(
            f"[PROGRESS] {done}/{N} pairs ({pct:.1f}%) | "
            f"Elapsed: {format_duration(elapsed)} | "
            f"ETA: {format_duration(eta)} | "
            f"Speed: {rate:.1f} pairs/s",
            flush=True,
        )
        last_log_time = now

total_time = time.time() - score_start
print(
    f"[DONE] Scoring complete in {format_duration(total_time)} "
    f"({N / total_time:.1f} pairs/s avg)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nSaved {N} cross-encoder scores to {OUTPUT_FILE}")
print(f"  store['index']               shape: {store['index'].shape}")
print(f"  store['qid1']                shape: {store['qid1'].shape}")
print(f"  store['qid2']                shape: {store['qid2'].shape}")
print(f"  store['cross_encoder_score'] shape: {store['cross_encoder_score'].shape}")
print()
print("Example lookup — first pair:")
print(f"  index              : {store['index'][0]}")
print(f"  qid1               : {store['qid1'][0]}")
print(f"  qid2               : {store['qid2'][0]}")
print(f"  cross_encoder_score: {store['cross_encoder_score'][0]:.6f}")
