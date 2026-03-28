import os
import csv
import time
import kagglehub
import numpy as np
import zarr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# Config
# =========================
ZARR_FILE = "embeddings.zarr"
DATASET_HANDLE = "quora/question-pairs-dataset"

MAX_ROWS = None          # e.g. 100000 for quick test
RANDOM_STATE = 42
COSINE_BASELINE_THRESHOLD = 0.76   # optional baseline only


# =========================
# Utilities
# =========================
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


def find_pairs_csv(dataset_path: str) -> str:
    preferred = os.path.join(dataset_path, "questions.csv")
    candidate_files = []

    if os.path.exists(preferred):
        candidate_files.append(preferred)

    for fname in os.listdir(dataset_path):
        if fname.endswith(".csv"):
            full = os.path.join(dataset_path, fname)
            if full not in candidate_files:
                candidate_files.append(full)

    required = {"qid1", "qid2", "question1", "question2", "is_duplicate"}

    for csv_file in candidate_files:
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = set(reader.fieldnames or [])
                if required.issubset(headers):
                    return csv_file
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not find a CSV in {dataset_path} with headers {sorted(required)}"
    )


def tokenize(text: str) -> list[str]:
    return (text or "").lower().strip().split()


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def evaluate_predictions(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 score  : {f1:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
    }


# =========================
# Load zarr
# =========================
print("[INFO] Loading zarr store...", flush=True)
store = zarr.open(ZARR_FILE, mode="r")

ids_arr = store["ids"][:]
texts_arr = store["texts"][:]
emb_arr = store["embeddings"][:].astype(np.float32)

print(f"[INFO] ids shape        : {ids_arr.shape}", flush=True)
print(f"[INFO] texts shape      : {texts_arr.shape}", flush=True)
print(f"[INFO] embeddings shape : {emb_arr.shape}", flush=True)

qid_to_pos = {int(qid): i for i, qid in enumerate(ids_arr)}

raw_emb = emb_arr
raw_norms = np.linalg.norm(raw_emb, axis=1)
norm_emb = raw_emb / np.clip(raw_norms[:, None], 1e-12, None)

print(f"[INFO] Built qid lookup for {len(qid_to_pos)} questions", flush=True)


# =========================
# Download dataset
# =========================
print(f"[INFO] Downloading dataset from kagglehub: {DATASET_HANDLE}", flush=True)
dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
print(f"[INFO] Dataset path: {dataset_path}", flush=True)

pairs_csv = find_pairs_csv(dataset_path)
print(f"[INFO] Using pairs CSV: {pairs_csv}", flush=True)


# =========================
# Build pairwise features
# =========================
print("[INFO] Building pairwise features...", flush=True)

X = []
y = []
cos_scores = []

missing_qids = 0
bad_rows = 0
used_rows = 0

start = time.time()
last_log = start

with open(pairs_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if MAX_ROWS is not None and used_rows >= MAX_ROWS:
            break

        try:
            qid1 = int(row["qid1"])
            qid2 = int(row["qid2"])
            q1 = row["question1"]
            q2 = row["question2"]
            label = int(row["is_duplicate"])
        except (KeyError, ValueError, TypeError):
            bad_rows += 1
            continue

        pos1 = qid_to_pos.get(qid1)
        pos2 = qid_to_pos.get(qid2)

        if pos1 is None or pos2 is None:
            missing_qids += 1
            continue

        u_raw = raw_emb[pos1]
        v_raw = raw_emb[pos2]

        u = norm_emb[pos1]
        v = norm_emb[pos2]

        cos_sim = float(np.dot(u, v))
        dot_raw = float(np.dot(u_raw, v_raw))
        euclidean = float(np.linalg.norm(u_raw - v_raw))
        manhattan = float(np.abs(u_raw - v_raw).sum())

        abs_diff = np.abs(u_raw - v_raw)
        prod = u_raw * v_raw

        abs_diff_mean = float(abs_diff.mean())
        abs_diff_max = float(abs_diff.max())
        abs_diff_std = float(abs_diff.std())

        prod_mean = float(prod.mean())
        prod_std = float(prod.std())

        norm1 = float(raw_norms[pos1])
        norm2 = float(raw_norms[pos2])
        norm_diff = abs(norm1 - norm2)

        q1_tokens = tokenize(q1)
        q2_tokens = tokenize(q2)

        set1 = set(q1_tokens)
        set2 = set(q2_tokens)

        inter = len(set1 & set2)
        union = len(set1 | set2)

        jaccard = safe_div(inter, union)
        overlap_min = safe_div(inter, min(len(set1), len(set2))) if min(len(set1), len(set2)) > 0 else 0.0

        len_q1_chars = len(q1 or "")
        len_q2_chars = len(q2 or "")
        len_q1_words = len(q1_tokens)
        len_q2_words = len(q2_tokens)

        char_len_diff = abs(len_q1_chars - len_q2_chars)
        word_len_diff = abs(len_q1_words - len_q2_words)

        feats = [
            cos_sim,
            dot_raw,
            euclidean,
            manhattan,
            abs_diff_mean,
            abs_diff_max,
            abs_diff_std,
            prod_mean,
            prod_std,
            norm1,
            norm2,
            norm_diff,
            len_q1_chars,
            len_q2_chars,
            char_len_diff,
            len_q1_words,
            len_q2_words,
            word_len_diff,
            inter,
            union,
            jaccard,
            overlap_min,
        ]

        X.append(feats)
        y.append(label)
        cos_scores.append(cos_sim)
        used_rows += 1

        now = time.time()
        if used_rows % 50000 == 0 or (now - last_log) >= 30:
            elapsed = now - start
            rate = used_rows / elapsed if elapsed > 0 else 0
            print(
                f"[PROGRESS] {used_rows} rows | "
                f"Elapsed: {format_duration(elapsed)} | "
                f"Speed: {rate:.1f} rows/s",
                flush=True,
            )
            last_log = now

print(f"[INFO] Used rows        : {used_rows}", flush=True)
print(f"[INFO] Missing qids    : {missing_qids}", flush=True)
print(f"[INFO] Bad rows skipped: {bad_rows}", flush=True)

if used_rows == 0:
    raise RuntimeError("No usable rows found.")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
cos_scores = np.array(cos_scores, dtype=np.float32)

print(f"[INFO] Feature matrix shape: {X.shape}", flush=True)


# =========================
# Split data
# =========================
X_train, X_temp, y_train, y_temp, cos_train, cos_temp = train_test_split(
    X, y, cos_scores,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test, cos_val, cos_test = train_test_split(
    X_temp, y_temp, cos_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print("[INFO] Split sizes:")
print(f"  train: {len(y_train)}")
print(f"  val  : {len(y_val)}")
print(f"  test : {len(y_test)}")


# =========================
# Optional cosine baseline
# =========================
cos_test_pred = (cos_test >= COSINE_BASELINE_THRESHOLD).astype(np.int32)
evaluate_predictions(
    f"Cosine baseline on TEST (threshold={COSINE_BASELINE_THRESHOLD})",
    y_test,
    cos_test_pred,
)


# =========================
# Logistic regression
# =========================
print("\n[INFO] Scaling features and training logistic regression...", flush=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

clf.fit(X_train_scaled, y_train)

test_probs = clf.predict_proba(X_test_scaled)[:, 1]
test_pred = (test_probs >= 0.5).astype(np.int32)

evaluate_predictions("Logistic regression on TEST", y_test, test_pred)


# =========================
# Show strongest coefficients
# =========================
feature_names = [
    "cos_sim",
    "dot_raw",
    "euclidean",
    "manhattan",
    "abs_diff_mean",
    "abs_diff_max",
    "abs_diff_std",
    "prod_mean",
    "prod_std",
    "norm1",
    "norm2",
    "norm_diff",
    "len_q1_chars",
    "len_q2_chars",
    "char_len_diff",
    "len_q1_words",
    "len_q2_words",
    "word_len_diff",
    "token_intersection",
    "token_union",
    "jaccard",
    "overlap_min",
]

coef = clf.coef_[0]
order = np.argsort(np.abs(coef))[::-1]

print("\n=== Strongest logistic regression coefficients ===")
for idx in order[:15]:
    print(f"{feature_names[idx]:20s} coef={coef[idx]: .4f}")