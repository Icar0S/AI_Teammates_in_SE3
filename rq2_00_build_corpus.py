#!/usr/bin/env python3
"""
RQ2 — Script 00: Build Test Corpus
====================================
Identifica todos os pares (pr_id, commit_sha) que tocam em arquivos de teste,
enriquece com metadados (agente, linguagem, tipo de tarefa) e prepara o corpus
para detecção de smells.

Fontes de dados
---------------
  OFFICIALDATASET.zip (ou HuggingFace) — 6 tabelas parquet:
    pr_commit_details, pr_commits, pull_request, repository,
    human_pull_request, pr_task_type, human_pr_task_type

Outputs
-------
  AIDev/rq2_corpus.parquet — (pr_id, commit_sha, commit_order, created_at,
                               agent, is_human, task_type, repo_language,
                               n_test_files_in_commit, has_temporal_data)
  AIDev/rq2_corpus_summary.csv — contagens por agente e linguagem

Usage
-----
  python rq2_00_build_corpus.py
  python rq2_00_build_corpus.py --offline
  python rq2_00_build_corpus.py --offline --limit 5000
  python rq2_00_build_corpus.py --zip CAMINHO/OUTRO.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    CSV_DIR,
    FIG_DIR,
    NAME_MAPPING,
    TEST_FILE_PATTERNS,
    ZipLoader,
    load_parquet,
)

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
CORPUS_PARQUET = OUT_DIR / "rq2_corpus.parquet"
SUMMARY_CSV = OUT_DIR / "rq2_corpus_summary.csv"

PRIMARY_LANGS = {"JavaScript", "TypeScript", "Python"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_test_file(filename: str) -> bool:
    return bool(TEST_FILE_PATTERNS.search(filename))


def _infer_language(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "py": "Python",
        "js": "JavaScript",
        "jsx": "JavaScript",
        "ts": "TypeScript",
        "tsx": "TypeScript",
        "java": "Java",
    }.get(ext, "Other")


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

def build_test_commit_index(
    offline: bool,
    loader: ZipLoader | None,
    limit: int | None,
) -> pd.DataFrame:
    """Return DataFrame of (pr_id, commit_sha, n_test_files) from diffs."""
    print("Loading pr_commit_details (filename column only)...")
    details = load_parquet(
        "pr_commit_details.parquet",
        offline,
        loader,
        columns=["pr_id", "commit_sha", "filename"],
    )
    if limit:
        details = details.head(limit)
    print(f"  Loaded {len(details):,} rows.")

    mask = details["filename"].apply(_is_test_file)
    test_rows = details[mask].copy()
    print(f"  Test file rows: {len(test_rows):,}")

    grouped = (
        test_rows.groupby(["pr_id", "commit_sha"])
        .size()
        .reset_index(name="n_test_files_in_commit")
    )
    print(f"  Unique (pr_id, commit_sha) pairs with test files: {len(grouped):,}")
    return grouped


def attach_commit_order(
    index: pd.DataFrame,
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    """Join with pr_commits to attach commit_order and created_at."""
    print("Loading pr_commits...")
    commits = load_parquet(
        "pr_commits.parquet",
        offline,
        loader,
        columns=["pr_id", "commit_sha", "commit_order", "created_at"],
    )
    merged = index.merge(commits, on=["pr_id", "commit_sha"], how="left")
    missing = merged["commit_order"].isna().sum()
    if missing:
        print(f"  Warning: {missing:,} pairs without commit_order (dropped).")
    merged = merged.dropna(subset=["commit_order"])
    merged["commit_order"] = merged["commit_order"].astype(int)
    return merged


def attach_agentic_metadata(
    df: pd.DataFrame,
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    """Join pull_request + repository to attach agent and language."""
    print("Loading pull_request...")
    pr = load_parquet(
        "pull_request.parquet",
        offline,
        loader,
        columns=["id", "agent", "repo_id"],
    ).rename(columns={"id": "pr_id"})

    print("Loading repository...")
    repo = load_parquet(
        "repository.parquet",
        offline,
        loader,
        columns=["id", "language"],
    ).rename(columns={"id": "repo_id"})

    print("Loading pr_task_type...")
    task = load_parquet(
        "pr_task_type.parquet",
        offline,
        loader,
        columns=["pr_id", "task_type"],
    )

    pr_repo = pr.merge(repo, on="repo_id", how="left")
    pr_repo = pr_repo.merge(task, on="pr_id", how="left")
    pr_repo["agent"] = pr_repo["agent"].map(NAME_MAPPING).fillna(pr_repo["agent"])
    pr_repo["is_human"] = False

    result = df.merge(pr_repo[["pr_id", "agent", "language", "task_type", "is_human"]],
                      on="pr_id", how="inner")
    result = result.rename(columns={"language": "repo_language"})
    print(f"  Agentic rows after join: {len(result):,}")
    return result


def attach_human_metadata(
    df: pd.DataFrame,
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    """Build same structure for human PRs and append."""
    print("Loading human_pull_request...")
    hpr = load_parquet(
        "human_pull_request.parquet",
        offline,
        loader,
        columns=["id", "repo_id"],
    ).rename(columns={"id": "pr_id"})

    print("Loading repository (for human PRs)...")
    repo = load_parquet(
        "repository.parquet",
        offline,
        loader,
        columns=["id", "language"],
    ).rename(columns={"id": "repo_id"})

    print("Loading human_pr_task_type...")
    htask = load_parquet(
        "human_pr_task_type.parquet",
        offline,
        loader,
        columns=["pr_id", "task_type"],
    )

    human_pr_ids = set(hpr["pr_id"].unique())

    human_df = df[df["pr_id"].isin(human_pr_ids)].copy()
    if human_df.empty:
        print("  No human PRs found in commit index (separate dataset — building from scratch).")
        return pd.DataFrame(columns=df.columns)

    hpr_meta = hpr.merge(repo, on="repo_id", how="left")
    hpr_meta = hpr_meta.merge(htask, on="pr_id", how="left")
    hpr_meta["agent"] = "Human"
    hpr_meta["is_human"] = True

    result = human_df.merge(
        hpr_meta[["pr_id", "agent", "language", "task_type", "is_human"]],
        on="pr_id",
        how="inner",
    )
    result = result.rename(columns={"language": "repo_language"})
    print(f"  Human rows after join: {len(result):,}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(offline: bool, zip_path: Path, limit: int | None) -> None:
    loader: ZipLoader | None = None
    if offline or zip_path.exists():
        loader = ZipLoader(zip_path)
        loader.open()

    try:
        index = build_test_commit_index(offline, loader, limit)
        index = attach_commit_order(index, offline, loader)

        agentic = attach_agentic_metadata(index, offline, loader)

        # Human PRs live in the same pr_commit_details but are identified via
        # human_pull_request. Re-run the join for human pr_ids.
        human = attach_human_metadata(index, offline, loader)

        corpus = pd.concat([agentic, human], ignore_index=True)

        # Compute has_temporal_data flag (PR has ≥2 commits touching test files)
        n_commits = (
            corpus.groupby("pr_id")["commit_sha"]
            .nunique()
            .rename("_n_test_commits")
        )
        corpus = corpus.merge(n_commits, on="pr_id", how="left")
        corpus["has_temporal_data"] = corpus["_n_test_commits"] >= 2
        corpus = corpus.drop(columns=["_n_test_commits"])

        # Language filter report
        lang_counts = corpus["repo_language"].value_counts()
        print("\nLanguage distribution:")
        print(lang_counts.head(10).to_string())
        primary_mask = corpus["repo_language"].isin(PRIMARY_LANGS)
        print(f"\nPrimary languages (JS/TS/Python): {primary_mask.sum():,} / {len(corpus):,} rows")

        corpus.to_parquet(CORPUS_PARQUET, index=False)
        print(f"\nSaved {len(corpus):,} rows → {CORPUS_PARQUET}")

        summary = (
            corpus.groupby(["agent", "repo_language"])
            .agg(
                n_commit_pairs=("commit_sha", "count"),
                n_prs=("pr_id", "nunique"),
                temporal_prs=("has_temporal_data", "sum"),
            )
            .reset_index()
            .sort_values(["agent", "n_prs"], ascending=[True, False])
        )
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved summary → {SUMMARY_CSV}")
        print()
        print(summary.to_string(index=False))

    finally:
        if loader:
            loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 corpus builder")
    parser.add_argument("--offline", action="store_true",
                        help="Force reading from local ZIP")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP,
                        metavar="PATH", help="Path to OFFICIALDATASET.zip")
    parser.add_argument("--limit", type=int, default=None,
                        metavar="N", help="Truncate pr_commit_details to N rows (debug)")
    args = parser.parse_args()
    main(offline=args.offline, zip_path=args.zip, limit=args.limit)
