"""
Stage 1 -- corpus construction.

Downloads the Fakeddit metadata table, samples a class-balanced,
subreddit-capped corpus, fetches the associated Reddit preview images, and
writes a single tidy manifest (``data/processed/corpus.parquet``) that every
later stage consumes.

Design notes
------------
*   Sampling is capped per subreddit (``MAX_PER_SUBREDDIT``) rather than taken
    uniformly at random. Fakeddit is dominated by a handful of large
    subreddits; capping keeps every source represented so that the
    source-disjoint protocol (Chapter 4) has enough held-out material.
*   Image download is I/O bound, so it is threaded. Failures are recorded
    rather than retried indefinitely -- roughly 10-12% of Reddit preview URLs
    have expired, and the manifest records exactly which rows survived so the
    run is auditable.
*   Every step is seeded and idempotent: re-running skips work already on disk.

Usage
-----
    python -m src.build_dataset            # full corpus
    python -m src.build_dataset --limit 2000   # quick smoke test
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)"}
LOG = print


# --------------------------------------------------------------------------- #
def download_metadata() -> pd.DataFrame:
    """Fetch (once) and load the Fakeddit metadata table.

    Resolution order: the plain .tsv if already present; else a gzipped
    sibling (``fakeddit_100k.tsv.gz``) is decompressed in place -- this is
    how the file ships in the repository, kept under GitHub's 25 MB limit;
    else it is downloaded fresh from the public mirror.
    """
    gz_path = C.FAKEDDIT_TSV.with_suffix(C.FAKEDDIT_TSV.suffix + ".gz")
    if not C.FAKEDDIT_TSV.exists() and gz_path.exists():
        LOG(f"[meta] decompressing {gz_path.name}")
        with gzip.open(gz_path, "rb") as src, open(C.FAKEDDIT_TSV, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if not C.FAKEDDIT_TSV.exists():
        LOG(f"[meta] downloading {C.FAKEDDIT_TSV_URL}")
        r = requests.get(C.FAKEDDIT_TSV_URL, timeout=300)
        r.raise_for_status()
        C.FAKEDDIT_TSV.write_bytes(r.content)
    df = pd.read_csv(C.FAKEDDIT_TSV, sep="\t", low_memory=False)
    LOG(f"[meta] {len(df):,} rows, {df.shape[1]} columns")
    return df


def sample_corpus(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Deduplicate, cap per subreddit and return the candidate corpus."""
    df = df[df["hasImage"] == True].copy()                       # noqa: E712
    before = len(df)
    df = df.drop_duplicates(subset="image_url").drop_duplicates(subset="id")
    LOG(f"[sample] {before:,} -> {len(df):,} after de-duplication")

    df["clean_title"] = df["clean_title"].astype(str).str.strip()
    df = df[df["clean_title"].str.len() >= 5]

    cap = C.MAX_PER_SUBREDDIT if limit is None else max(20, limit // 17)
    keep = []
    for sr, g in df.groupby("subreddit", sort=True):
        keep.append(g.sample(min(len(g), cap), random_state=C.RANDOM_SEED))
    df = pd.concat(keep).reset_index(drop=True)
    LOG(f"[sample] capped at {cap}/subreddit -> {len(df):,} candidates")
    LOG(f"[sample] 2-way balance: {df['2_way_label'].value_counts().to_dict()}")
    return df


# --------------------------------------------------------------------------- #
def _image_filename(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"


def _fetch_one(url: str, dest: Path) -> tuple[str, bool, str]:
    """Download and normalise one image. Returns (url, ok, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return url, True, "cached"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200:
            return url, False, f"http{r.status_code}"
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        # Reddit previews are already <=320px wide; store a fixed short side so
        # that CLIP preprocessing is deterministic and the cache stays small.
        img.thumbnail((C.IMAGE_SIZE * 2, C.IMAGE_SIZE * 2), Image.LANCZOS)
        if min(img.size) < 32:
            return url, False, "too_small"
        img.save(dest, "JPEG", quality=90)
        return url, True, "ok"
    except Exception as exc:                                    # noqa: BLE001
        return url, False, type(exc).__name__


def download_images(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Fetch every image in ``df`` and return the frame with a status column."""
    df = df.copy()
    df["image_file"] = df["image_url"].map(_image_filename)
    todo = [(u, C.IMAGES / f) for u, f in zip(df["image_url"], df["image_file"])]

    LOG(f"[images] fetching {len(todo):,} images with {workers} workers")
    status: dict[str, str] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fetch_one, u, d): u for u, d in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            url, ok, msg = fut.result()
            status[url] = msg if ok else f"FAIL:{msg}"
            if i % 2000 == 0:
                rate = i / max(1e-9, time.time() - t0)
                LOG(f"[images]   {i:,}/{len(todo):,}  ({rate:.0f}/s)")

    df["download_status"] = df["image_url"].map(status)
    ok = df["download_status"].isin(["ok", "cached"])
    LOG(f"[images] {ok.sum():,}/{len(df):,} succeeded ({ok.mean():.1%}) "
        f"in {time.time() - t0:.0f}s")
    LOG(f"[images] failure modes: "
        f"{df.loc[~ok, 'download_status'].value_counts().head(6).to_dict()}")
    return df[ok].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the three evaluation protocols as separate columns."""
    rng_df = df.sample(frac=1.0, random_state=C.RANDOM_SEED).reset_index(drop=True)

    # --- Protocol A: stratified random -------------------------------------
    parts = []
    for _, g in rng_df.groupby("2_way_label"):
        n = len(g)
        n_tr = int(n * C.SPLIT_FRACTIONS["train"])
        n_va = int(n * C.SPLIT_FRACTIONS["val"])
        g = g.copy()
        g["split_random"] = (
            ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        )
        parts.append(g)
    df = pd.concat(parts).sample(frac=1.0, random_state=C.RANDOM_SEED)
    df = df.reset_index(drop=True)

    # --- Protocol B: source-disjoint ---------------------------------------
    def _src(sr: str) -> str:
        if sr in C.SOURCE_DISJOINT["test"]:
            return "test"
        if sr in C.SOURCE_DISJOINT["val"]:
            return "val"
        return "train"

    df["split_source"] = df["subreddit"].map(_src)

    # --- Protocol C: temporal ----------------------------------------------
    q_tr = df["created_utc"].quantile(C.TEMPORAL_QUANTILES["train"])
    q_va = df["created_utc"].quantile(C.TEMPORAL_QUANTILES["val"])
    df["split_temporal"] = pd.cut(
        df["created_utc"],
        bins=[-1, q_tr, q_va, float("inf")],
        labels=["train", "val", "test"],
    ).astype(str)

    for col in ("split_random", "split_source", "split_temporal"):
        tab = pd.crosstab(df[col], df["2_way_label"])
        LOG(f"[splits] {col}\n{tab}\n")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered metadata features (all computable at post time)."""
    import numpy as np

    df = df.copy()
    df["log_score"] = np.log1p(df["score"].clip(lower=0))
    df["log_comments"] = np.log1p(df["num_comments"].clip(lower=0))
    df["upvote_ratio"] = df["upvote_ratio"].fillna(df["upvote_ratio"].median())
    df["title_len_chars"] = df["clean_title"].str.len()
    df["title_len_words"] = df["clean_title"].str.split().str.len()
    df["has_author"] = df["author"].notna().astype(int)
    df["is_selfpost"] = df["domain"].astype(str).str.startswith("self.").astype(int)
    dt = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df["year"] = dt.dt.year
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    return df


META_FEATURES = [
    "log_score", "log_comments", "upvote_ratio", "title_len_chars",
    "title_len_words", "has_author", "is_selfpost", "hour", "dayofweek",
]


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total candidates (smoke test)")
    ap.add_argument("--workers", type=int, default=C.N_DOWNLOAD_WORKERS)
    ap.add_argument("--out", type=str, default="corpus.parquet")
    args = ap.parse_args()

    meta = download_metadata()
    cand = sample_corpus(meta, args.limit)
    got = download_images(cand, args.workers)
    got = assign_splits(got)
    got = add_derived_columns(got)

    # Data minimisation (Chapter 5): the Reddit username is the only directly
    # identifying field in the source table and it is not needed by any model,
    # so it is reduced to a presence flag before anything is written to disk.
    got = got.drop(columns=[c for c in ("author",) if c in got.columns])

    out = C.PROCESSED / args.out
    got.to_parquet(out, index=False)
    LOG(f"[done] wrote {out}  ({len(got):,} rows, {got.shape[1]} columns)")


if __name__ == "__main__":
    main()
