"""
embed_quora_test.py — Embed all unique questions from the Kaggle test set.

Downloads the Quora Question Pairs dataset (quora/question-pairs-dataset),
locates test.csv, extracts every unique question text, embeds them with
Qwen3-Embedding-4B, and writes the result to test_embeddings.zarr.

The output zarr store is keyed by sorted question text (alphabetical order):
    store["texts"]       shape (N,)         dtype: str      — unique question texts
    store["embeddings"]  shape (N, 2560)    dtype: float32  — embedding vectors

At query time, build a dict from store["texts"] → position for O(1) lookup.

No Kaggle authentication is required — the dataset is publicly available.

Usage:
    uv run embed_quora_test.py
    uv run embed_quora_test.py --batch-size 64
    uv run embed_quora_test.py --output my_test_embeddings.zarr
    uv run embed_quora_test.py --local-test-csv /path/to/test.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import kagglehub
import numpy as np
import zarr
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME    = "Qwen/Qwen3-Embedding-4B"
DEFAULT_OUT   = "test_embeddings.zarr"
BATCH_SIZE    = 128
DATASET_HANDLE = "quora/question-pairs-dataset"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _find_test_csv(competition_path: str) -> str:
    """Return path to the test CSV inside the competition download.

    Explicitly ignores ``test.csv.zip`` (and any other .zip files), which
    kagglehub may place alongside the real CSV but which cannot be read as
    plain text.
    """
    # Preferred name — must be a plain file, never a zip archive
    preferred = os.path.join(competition_path, "test.csv")
    if os.path.exists(preferred) and not preferred.endswith(".zip"):
        return preferred

    # Warn if we see the zip — it is not the file we want
    zip_path = os.path.join(competition_path, "test.csv.zip")
    if os.path.exists(zip_path):
        print(
            f"[WARN] Found {zip_path!r} — this is a zip archive and will be "
            "ignored.  Looking for a plain test.csv instead.",
            flush=True,
        )

    # Fall back: first plain CSV (not a zip) with the right headers
    required = {"test_id", "question1", "question2"}
    for fname in sorted(os.listdir(competition_path)):
        # Skip anything that is not a plain .csv file
        if not fname.endswith(".csv") or fname.endswith(".csv.zip"):
            continue
        full = os.path.join(competition_path, fname)
        try:
            with open(full, newline="", encoding="utf-8") as f:
                headers = set(csv.DictReader(f).fieldnames or [])
            if required.issubset(headers):
                print(f"[INFO] Found matching CSV via header scan: {full}", flush=True)
                return full
        except Exception:
            pass

    raise FileNotFoundError(
        f"No plain CSV with headers {sorted(required)} found in {competition_path!r}.\n"
        "Tip: copy your local test.csv into that directory, or pass "
        "--local-test-csv /path/to/test.csv to skip the download entirely."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed all unique questions from the Kaggle QQP test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output",     default=DEFAULT_OUT,   help="Output zarr store path.")
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int, help="Encoding batch size.")
    parser.add_argument("--model",      default=MODEL_NAME,    help="SentenceTransformer model name.")
    parser.add_argument(
        "--local-test-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to a locally available test.csv.  When provided the dataset "
            "is NOT downloaded from Kaggle — useful when kagglehub returns only "
            "a test.csv.zip or when you already have the file on disk."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Locate test.csv — local copy takes priority over a fresh download
    # ------------------------------------------------------------------
    if args.local_test_csv:
        local = os.path.abspath(args.local_test_csv)
        if not os.path.isfile(local):
            raise FileNotFoundError(
                f"--local-test-csv path does not exist or is not a file: {local!r}"
            )
        if local.endswith(".zip"):
            raise ValueError(
                f"--local-test-csv must be a plain CSV, not a zip archive: {local!r}"
            )
        test_csv = local
        print(f"[INFO] Using local test CSV (skipping download): {test_csv}", flush=True)
    else:
        print(f"[INFO] Downloading dataset ({DATASET_HANDLE})...", flush=True)
        comp_path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"[INFO] Dataset path : {comp_path}", flush=True)
        print(f"[INFO] Files        : {sorted(os.listdir(comp_path))}", flush=True)

        test_csv = _find_test_csv(comp_path)
        print(f"[INFO] Using test CSV   : {test_csv}", flush=True)

    # ------------------------------------------------------------------
    # 2. Collect unique question texts
    # ------------------------------------------------------------------
    print("[INFO] Reading test.csv and collecting unique question texts...", flush=True)
    unique_texts: dict[str, None] = {}          # ordered set (Python 3.7+ dicts preserve insertion)

    with open(test_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ("question1", "question2"):
                text = (row.get(col) or "").strip()
                if text not in unique_texts:
                    unique_texts[text] = None

    # Sort for deterministic zarr positions (enables np.searchsorted lookup later)
    sorted_texts = sorted(unique_texts.keys())
    N = len(sorted_texts)
    print(f"[INFO] Unique questions in test set : {N:,}", flush=True)

    # ------------------------------------------------------------------
    # 3. Load embedding model
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading model: {args.model} (SDPA attention)", flush=True)
    model = SentenceTransformer(
        args.model,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {dim}", flush=True)

    # ------------------------------------------------------------------
    # 4. Create zarr store
    # ------------------------------------------------------------------
    print(f"\n[INFO] Opening zarr store: {args.output}", flush=True)
    store = zarr.open(args.output, mode="w")

    # texts — sorted unique question strings
    texts_arr = store.create_array(
        name   = "texts",
        shape  = (N,),
        dtype  = "str",
        chunks = (args.batch_size,),
    )
    texts_arr[:] = sorted_texts
    print(f"[INFO] Wrote texts array  : {texts_arr.shape}", flush=True)

    # embeddings — float32, (N, dim)
    emb_arr = store.zeros(
        name   = "embeddings",
        shape  = (N, dim),
        dtype  = "float32",
        chunks = (args.batch_size, dim),
    )

    # ------------------------------------------------------------------
    # 5. Embed in batches
    # ------------------------------------------------------------------
    print(
        f"\n[INFO] Embedding {N:,} questions "
        f"in batches of {args.batch_size}...",
        flush=True,
    )
    start     = time.time()
    last_log  = start

    for i in range(0, N, args.batch_size):
        batch = sorted_texts[i : i + args.batch_size]
        embs  = model.encode(
            batch,
            convert_to_numpy  = True,
            show_progress_bar = False,
            prompt_name       = "query",
        )
        emb_arr[i : i + len(batch)] = embs

        now     = time.time()
        elapsed = now - start
        done    = i + len(batch)
        pct     = done / N * 100
        rate    = done / elapsed if elapsed > 0 else 0.0
        eta     = (N - done) / rate if rate > 0 else 0.0

        if (now - last_log) >= 30 or i == 0:
            print(
                f"[PROGRESS] {done:>8,}/{N:,} ({pct:5.1f}%) | "
                f"Elapsed: {_fmt(elapsed)} | "
                f"ETA: {_fmt(eta)} | "
                f"{rate:,.0f} q/s",
                flush=True,
            )
            last_log = now

    total = time.time() - start
    print(
        f"\n[DONE] Embedding complete in {_fmt(total)} "
        f"({N / total:,.0f} q/s avg)",
        flush=True,
    )
    print(f"Saved {N:,} embeddings to {args.output}")
    print(f"  store['texts']      shape : {store['texts'].shape}")
    print(f"  store['embeddings'] shape : {store['embeddings'].shape}")
    print()
    print("Example lookup:")
    ex_text = sorted_texts[0]
    print(f"  text        : {ex_text!r}")
    print(f"  embedding[:5]: {store['embeddings'][0, :5]}")


if __name__ == "__main__":
    main()
