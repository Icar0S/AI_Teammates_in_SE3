#!/usr/bin/env python3
"""
RQ3 — Script 00: Build Performance PR Corpus
=============================================
Constrói o corpus de PRs relacionados a performance para a RQ3, usando dois
sinais complementares:
  Signal A — PRs pré-classificados como tipo 'perf' em pr_task_type
  Signal B — PRs que tocam arquivos com padrões de performance nos nomes/conteúdo

O corpus inclui PRs agênticos (5 agentes) e grupo de controle humano.

Fontes de dados
---------------
  OFFICIALDATASET.zip (ou HuggingFace):
    pull_request, pr_task_type, pr_commit_details (filenames only),
    human_pull_request, human_pr_task_type, repository

Outputs
-------
  AIDev/rq3_corpus.parquet  — (pr_id, agent, is_human, task_type,
                               corpus_source, repo_language, html_url, repo_id)
  AIDev/rq3_corpus_summary.csv — contagens por agente e fonte do corpus

Usage
-----
  python rq3_00_build_corpus.py
  python rq3_00_build_corpus.py --offline
  python rq3_00_build_corpus.py --offline --limit 50000
  python rq3_00_build_corpus.py --zip CAMINHO/OUTRO.zip
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    CSV_DIR,
    FIG_DIR,
    NAME_MAPPING,
    ZipLoader,
    load_parquet,
)

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
CORPUS_PARQUET = OUT_DIR / "rq3_corpus.parquet"
SUMMARY_CSV = OUT_DIR / "rq3_corpus_summary.csv"

# ---------------------------------------------------------------------------
# Performance file patterns (reused/extended from rq3_perf_exploration.py)
# ---------------------------------------------------------------------------

FILENAME_PATTERNS: list[re.Pattern] = [
    re.compile(r"\.k6\.(js|ts)$", re.IGNORECASE),
    re.compile(r"(^|[/_-])k6[/_-]", re.IGNORECASE),
    re.compile(r"[/_-]k6\.(js|ts)$", re.IGNORECASE),
    re.compile(r"\.jmx$", re.IGNORECASE),
    re.compile(r"gatling", re.IGNORECASE),
    re.compile(r"locustfile\.py$", re.IGNORECASE),
    re.compile(r"[/_-]locust[/_-]", re.IGNORECASE),
    re.compile(
        r"(load[_-]?test|stress[_-]?test|perf[_-]?test|benchmark)",
        re.IGNORECASE,
    ),
    re.compile(r"[/_-](perf|load|stress|benchmark)[/_-]", re.IGNORECASE),
    re.compile(r"(performance|throughput)[_-]test", re.IGNORECASE),
]


def _matches_perf_filename(fname: str | None) -> bool:
    if not fname or not isinstance(fname, str):
        return False
    return any(p.search(fname) for p in FILENAME_PATTERNS)


# ---------------------------------------------------------------------------
# Normalise task-type table (handles HF 'id'/'type' and ZIP 'pr_id'/'task_type')
# ---------------------------------------------------------------------------

def _load_task_type(
    table_name: str,
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    df = load_parquet(table_name, offline, loader)
    rename: dict[str, str] = {}
    if "id" in df.columns and "pr_id" not in df.columns:
        rename["id"] = "pr_id"
    if "type" in df.columns and "task_type" not in df.columns:
        rename["type"] = "task_type"
    if rename:
        df = df.rename(columns=rename)
    return df[["pr_id", "task_type"]].copy()


# ---------------------------------------------------------------------------
# Build AI performance corpus
# ---------------------------------------------------------------------------

def _build_ai_corpus(
    offline: bool,
    loader: ZipLoader | None,
    limit: int | None,
) -> pd.DataFrame:
    print("\n[AI] Loading pull_request.parquet...")
    pr_df = load_parquet(
        "pull_request.parquet",
        offline,
        loader,
        columns=["id", "agent", "repo_id", "html_url"],
    ).rename(columns={"id": "pr_id"})
    pr_df["agent"] = pr_df["agent"].map(NAME_MAPPING).fillna(pr_df["agent"])
    ai_pr_ids = set(pr_df["pr_id"].unique())
    print(f"  AI PRs: {len(ai_pr_ids):,}")

    # -- Signal A: pre-classified 'perf' --
    print("[AI] Loading pr_task_type.parquet (Signal A)...")
    task_df = _load_task_type("pr_task_type.parquet", offline, loader)
    sig_a_ids = set(
        task_df[task_df["task_type"] == "perf"]["pr_id"].unique()
    )
    print(f"  Signal A (task_type==perf): {len(sig_a_ids):,} PRs")

    # -- Signal B: filename patterns in commit details --
    print("[AI] Loading pr_commit_details.parquet filenames (Signal B)...")
    details = load_parquet(
        "pr_commit_details.parquet",
        offline,
        loader,
        columns=["pr_id", "filename"],
    )
    if limit:
        details = details.head(limit)
    details = details[details["pr_id"].isin(ai_pr_ids)]
    mask = details["filename"].fillna("").astype(str).apply(_matches_perf_filename)
    sig_b_ids = set(details[mask]["pr_id"].unique())
    print(f"  Signal B (filename patterns): {len(sig_b_ids):,} PRs")

    # -- Union A ∪ B with source tracking --
    all_ids = sig_a_ids | sig_b_ids
    rows = []
    for pr_id in all_ids:
        in_a = pr_id in sig_a_ids
        in_b = pr_id in sig_b_ids
        if in_a and in_b:
            src = "both"
        elif in_a:
            src = "task_type"
        else:
            src = "filename"
        rows.append({"pr_id": pr_id, "corpus_source": src})
    corpus_ids = pd.DataFrame(rows)
    print(f"  Union A ∪ B: {len(corpus_ids):,} distinct PRs")

    # Attach task_type for all PRs (fill from task_df, NA if not classified)
    corpus_ids = corpus_ids.merge(
        task_df[["pr_id", "task_type"]], on="pr_id", how="left"
    )

    # Attach agent + repo metadata
    corpus_ids = corpus_ids.merge(pr_df, on="pr_id", how="left")
    corpus_ids["is_human"] = False
    return corpus_ids


# ---------------------------------------------------------------------------
# Build human performance corpus
# ---------------------------------------------------------------------------

def _build_human_corpus(
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    print("\n[Human] Loading human_pull_request.parquet...")
    hpr = load_parquet(
        "human_pull_request.parquet",
        offline,
        loader,
        columns=["id", "repo_id", "html_url"],
    ).rename(columns={"id": "pr_id"})
    print(f"  Human PRs: {len(hpr):,}")

    print("[Human] Loading human_pr_task_type.parquet (Signal A)...")
    htask = _load_task_type("human_pr_task_type.parquet", offline, loader)
    human_perf = htask[htask["task_type"] == "perf"][["pr_id", "task_type"]].copy()
    print(f"  Signal A (task_type==perf): {len(human_perf):,} PRs")

    if human_perf.empty:
        print("  Warning: no human 'perf' PRs found — human group will be empty.")
        return pd.DataFrame(
            columns=["pr_id", "task_type", "corpus_source", "repo_id",
                     "html_url", "agent", "is_human"]
        )

    human_perf["corpus_source"] = "task_type"
    corpus = human_perf.merge(hpr, on="pr_id", how="left")
    corpus["agent"] = "Human"
    corpus["is_human"] = True
    print(f"  Human perf corpus: {len(corpus):,} PRs")
    return corpus


# ---------------------------------------------------------------------------
# Attach repository language
# ---------------------------------------------------------------------------

def _attach_language(
    corpus: pd.DataFrame,
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    print("\nLoading repository.parquet for language info...")
    repo = load_parquet(
        "repository.parquet",
        offline,
        loader,
        columns=["id", "language"],
    ).rename(columns={"id": "repo_id"})
    corpus = corpus.merge(repo, on="repo_id", how="left")
    corpus = corpus.rename(columns={"language": "repo_language"})
    return corpus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(offline: bool, zip_path: Path, limit: int | None) -> None:
    loader: ZipLoader | None = None
    if offline or zip_path.exists():
        loader = ZipLoader(zip_path)
        loader.open()

    try:
        ai_corpus = _build_ai_corpus(offline, loader, limit)
        human_corpus = _build_human_corpus(offline, loader)

        corpus = pd.concat([ai_corpus, human_corpus], ignore_index=True)
        corpus = _attach_language(corpus, offline, loader)

        # Select and order final columns
        col_order = [
            "pr_id", "agent", "is_human", "task_type",
            "corpus_source", "repo_language", "html_url", "repo_id",
        ]
        for c in col_order:
            if c not in corpus.columns:
                corpus[c] = None
        corpus = corpus[col_order].drop_duplicates(subset=["pr_id"])

        corpus.to_parquet(CORPUS_PARQUET, index=False)
        print(f"\nSaved {len(corpus):,} rows → {CORPUS_PARQUET}")

        summary = (
            corpus.groupby(["agent", "corpus_source", "repo_language"])
            .agg(n_prs=("pr_id", "count"))
            .reset_index()
            .sort_values(["agent", "n_prs"], ascending=[True, False])
        )
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved summary → {SUMMARY_CSV}")

        print("\n--- Per-agent PR counts ---")
        agent_totals = (
            corpus.groupby("agent")
            .agg(n_prs=("pr_id", "count"))
            .reset_index()
            .sort_values("n_prs", ascending=False)
        )
        print(agent_totals.to_string(index=False))

        print("\n--- Source breakdown ---")
        src_totals = corpus["corpus_source"].value_counts()
        print(src_totals.to_string())

        if len(corpus[corpus["is_human"]]) < 50:
            print(
                "\nWarning: fewer than 50 human perf PRs found. "
                "Human comparison group will have limited statistical power."
            )

    finally:
        if loader:
            loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3 performance corpus builder")
    parser.add_argument(
        "--offline", action="store_true", help="Force reading from local ZIP"
    )
    parser.add_argument(
        "--zip", type=Path, default=DEFAULT_ZIP, metavar="PATH",
        help="Path to OFFICIALDATASET.zip",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Truncate pr_commit_details filename scan to first N rows (debug)",
    )
    args = parser.parse_args()
    main(offline=args.offline, zip_path=args.zip, limit=args.limit)
