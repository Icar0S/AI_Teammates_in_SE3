#!/usr/bin/env python3
"""
RQ4 — Script 00: Build Review-Comment Corpus
=============================================
Constrói o corpus de comentários de revisão para a RQ4, unificando os dois
snapshots de review comments do AIDev, ligando cada comentário ao seu PR e
classificando o artefato comentado (teste vs. produção vs. config vs. docs).

O protocolo (§3.6.1) previa extração via GitHub API; o dataset oficial já
publica os review comments, de modo que o corpus é construído offline.

Sinais construídos
------------------
  * artifact_kind   — test / production / config / docs (pelo path comentado)
  * reviewer_id     — username anonimizado (protocolo §3.6.1)
  * is_bot          — separa revisores humanos de bots de revisão
  * merged          — desfecho do PR (variável dependente da regressão)
  * thread_root_id  — raiz da thread, para análise de resolução

Fontes de dados
---------------
  OFFICIALDATASET.zip (ou HuggingFace):
    pr_review_comments, pr_review_comments_v2, pr_reviews,
    pull_request, pr_task_type, repository, pr_commit_details (tamanho do PR)

Outputs
-------
  AIDev/rq4_comments.parquet      — um comentário por linha + metadados do PR
  AIDev/rq4_pr_reviews.parquet    — estados de revisão agregados por PR
  AIDev/rq4_pr_size.parquet       — cache de tamanho do PR (leitura pesada)
  AIDev/rq4_corpus_summary.csv    — contagens por agente e tipo de artefato

Usage
-----
  python rq4_00_build_corpus.py --offline
  python rq4_00_build_corpus.py --offline --skip-size   # pula leitura pesada
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    NAME_MAPPING,
    ZipLoader,
    classify_artifact_kind,
    load_parquet,
)

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
COMMENTS_PARQUET = OUT_DIR / "rq4_comments.parquet"
REVIEWS_PARQUET = OUT_DIR / "rq4_pr_reviews.parquet"
SIZE_PARQUET = OUT_DIR / "rq4_pr_size.parquet"
SUMMARY_CSV = OUT_DIR / "rq4_corpus_summary.csv"

REVIEW_COMMENT_TABLES = [
    "pr_review_comments_v2.parquet",
    "pr_review_comments.parquet",
]

COMMENT_COLS = [
    "id", "pull_request_review_id", "user", "user_type", "path",
    "body", "pull_request_url", "created_at", "in_reply_to_id",
    "diff_hunk",
]


# ---------------------------------------------------------------------------
# Review comments — load both snapshots and deduplicate
# ---------------------------------------------------------------------------

def _load_review_comments(
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    frames = []
    for table in REVIEW_COMMENT_TABLES:
        try:
            df = load_parquet(table, offline, loader, columns=COMMENT_COLS)
        except Exception as exc:  # table absent in some snapshots
            print(f"  [skip] {table}: {exc}")
            continue
        df["source_table"] = table
        frames.append(df)
        print(f"  {table}: {len(df):,} rows")

    if not frames:
        raise RuntimeError("No review-comment table could be loaded.")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    # v2 is the newer snapshot; keep it when the same comment id appears twice
    merged = merged.drop_duplicates(subset=["id"], keep="first")
    print(f"  Deduplicated: {before:,} → {len(merged):,} distinct comments")
    return merged


# ---------------------------------------------------------------------------
# Link comments to pr_id via URL rewriting
# ---------------------------------------------------------------------------

def _api_url_to_html(url_series: pd.Series) -> pd.Series:
    """https://api.github.com/repos/O/R/pulls/N → https://github.com/O/R/pull/N"""
    return (
        url_series.astype(str)
        .str.replace(
            "https://api.github.com/repos/", "https://github.com/", regex=False
        )
        .str.replace("/pulls/", "/pull/", regex=False)
    )


def _attach_pr_metadata(
    comments: pd.DataFrame,
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    print("\nLoading pull_request.parquet...")
    pr = load_parquet(
        "pull_request.parquet",
        offline,
        loader,
        columns=["id", "agent", "repo_id", "html_url", "merged_at",
                 "state", "created_at"],
    ).rename(columns={"id": "pr_id", "created_at": "pr_created_at"})
    pr["agent"] = pr["agent"].map(NAME_MAPPING).fillna(pr["agent"])
    pr["merged"] = pr["merged_at"].notna()
    print(f"  {len(pr):,} PRs | merged: {pr['merged'].mean() * 100:.1f}%")

    comments["_html_url"] = _api_url_to_html(comments["pull_request_url"])
    url2id = dict(zip(pr["html_url"], pr["pr_id"]))
    comments["pr_id"] = comments["_html_url"].map(url2id)

    n_unmatched = comments["pr_id"].isna().sum()
    print(
        f"  Comments linked to a PR: "
        f"{len(comments) - n_unmatched:,} / {len(comments):,} "
        f"({(1 - n_unmatched / len(comments)) * 100:.1f}%)"
    )
    comments = comments.dropna(subset=["pr_id"]).copy()
    comments["pr_id"] = comments["pr_id"].astype("int64")

    keep = ["pr_id", "agent", "repo_id", "merged", "state", "pr_created_at"]
    comments = comments.merge(pr[keep], on="pr_id", how="left")
    return comments.drop(columns=["_html_url"])


def _attach_language_and_task(
    comments: pd.DataFrame,
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    print("Loading repository.parquet for language...")
    repo = load_parquet(
        "repository.parquet", offline, loader, columns=["id", "language"]
    ).rename(columns={"id": "repo_id", "language": "repo_language"})
    comments = comments.merge(repo, on="repo_id", how="left")

    print("Loading pr_task_type.parquet...")
    task = load_parquet(
        "pr_task_type.parquet", offline, loader, columns=["id", "type"]
    ).rename(columns={"id": "pr_id", "type": "task_type"})
    comments = comments.merge(task, on="pr_id", how="left")
    return comments


# ---------------------------------------------------------------------------
# Anonymisation (protocol §3.6.1) and thread reconstruction
# ---------------------------------------------------------------------------

def _anonymise_reviewers(comments: pd.DataFrame) -> pd.DataFrame:
    """Replace usernames with stable sequential ids (R0001, R0002, ...)."""
    users = sorted(comments["user"].dropna().astype(str).unique())
    mapping = {u: f"R{i + 1:04d}" for i, u in enumerate(users)}
    comments["reviewer_id"] = comments["user"].astype(str).map(mapping)
    print(f"  Anonymised {len(mapping):,} distinct reviewer usernames")
    return comments.drop(columns=["user"])


def _reconstruct_threads(comments: pd.DataFrame) -> pd.DataFrame:
    """Resolve each comment to its thread root and mark replied-to comments."""
    parent = dict(
        zip(comments["id"], comments["in_reply_to_id"].fillna(-1).astype("int64"))
    )

    def root_of(cid: int) -> int:
        seen = set()
        cur = cid
        while True:
            p = parent.get(cur, -1)
            if p == -1 or p in seen or p not in parent:
                return cur
            seen.add(cur)
            cur = p

    comments["thread_root_id"] = comments["id"].map(root_of)
    comments["is_reply"] = comments["in_reply_to_id"].notna()

    replied_to = set(comments["in_reply_to_id"].dropna().astype("int64"))
    comments["has_reply"] = comments["id"].isin(replied_to)

    thread_sizes = comments.groupby("thread_root_id").size().rename("thread_size")
    comments = comments.merge(
        thread_sizes, left_on="thread_root_id", right_index=True, how="left"
    )
    print(
        f"  Threads: {comments['thread_root_id'].nunique():,} | "
        f"replies: {comments['is_reply'].sum():,} | "
        f"comments that got a reply: {comments['has_reply'].sum():,}"
    )
    return comments


# ---------------------------------------------------------------------------
# PR-level review states (tone/intensity signal at review granularity)
# ---------------------------------------------------------------------------

def _build_pr_reviews(
    offline: bool,
    loader: ZipLoader | None,
) -> pd.DataFrame:
    print("\nLoading pr_reviews.parquet...")
    rev = load_parquet(
        "pr_reviews.parquet",
        offline,
        loader,
        columns=["id", "pr_id", "user", "user_type", "state", "submitted_at"],
    )
    print(f"  {len(rev):,} reviews across {rev['pr_id'].nunique():,} PRs")

    rev["is_bot"] = rev["user_type"].eq("Bot")
    human = rev[~rev["is_bot"]]

    def _counts(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        wide = (
            df.groupby(["pr_id", "state"]).size().unstack(fill_value=0)
            .rename(columns=lambda s: f"{prefix}_{str(s).lower()}")
        )
        return wide

    all_counts = _counts(rev, "n_review")
    human_counts = _counts(human, "n_human_review")

    out = all_counts.join(human_counts, how="outer").fillna(0).astype(int)
    out["n_reviews_total"] = out.filter(like="n_review_").sum(axis=1)
    out["n_distinct_reviewers"] = (
        rev.groupby("pr_id")["user"].nunique().reindex(out.index).fillna(0)
    ).astype(int)
    out["n_distinct_human_reviewers"] = (
        human.groupby("pr_id")["user"].nunique().reindex(out.index).fillna(0)
    ).astype(int)
    return out.reset_index()


# ---------------------------------------------------------------------------
# PR size control variable (heavy read — cached)
# ---------------------------------------------------------------------------

def _build_pr_size(
    offline: bool,
    loader: ZipLoader | None,
    pr_ids: set[int],
) -> pd.DataFrame:
    if SIZE_PARQUET.exists():
        print(f"\nReusing cached PR sizes → {SIZE_PARQUET.name}")
        return pd.read_parquet(SIZE_PARQUET)

    print("\nLoading pr_commit_details.parquet for PR size (heavy)...")
    det = load_parquet(
        "pr_commit_details.parquet",
        offline,
        loader,
        columns=["pr_id", "filename", "additions", "deletions"],
    )
    det = det[det["pr_id"].isin(pr_ids)]
    size = (
        det.groupby("pr_id")
        .agg(
            pr_files_changed=("filename", "nunique"),
            pr_additions=("additions", "sum"),
            pr_deletions=("deletions", "sum"),
        )
        .reset_index()
    )
    size["pr_churn"] = size["pr_additions"] + size["pr_deletions"]
    size.to_parquet(SIZE_PARQUET, index=False)
    print(f"  Sized {len(size):,} PRs → {SIZE_PARQUET.name}")
    return size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(offline: bool, zip_path: Path, skip_size: bool) -> None:
    loader: ZipLoader | None = None
    if offline or zip_path.exists():
        loader = ZipLoader(zip_path)
        loader.open()

    try:
        print("Loading review-comment snapshots...")
        comments = _load_review_comments(offline, loader)

        comments = _attach_pr_metadata(comments, offline, loader)
        comments = _attach_language_and_task(comments, offline, loader)

        print("\nClassifying commented artifacts...")
        comments["artifact_kind"] = comments["path"].apply(classify_artifact_kind)
        comments["is_test_comment"] = comments["artifact_kind"].eq("test")
        comments["is_bot"] = comments["user_type"].eq("Bot")

        print("\nAnonymising reviewers...")
        comments = _anonymise_reviewers(comments)

        print("\nReconstructing review threads...")
        comments = _reconstruct_threads(comments)

        comments.to_parquet(COMMENTS_PARQUET, index=False)
        print(f"\nSaved {len(comments):,} comments → {COMMENTS_PARQUET}")

        pr_reviews = _build_pr_reviews(offline, loader)
        pr_reviews.to_parquet(REVIEWS_PARQUET, index=False)
        print(f"Saved {len(pr_reviews):,} PR review summaries → {REVIEWS_PARQUET}")

        if not skip_size:
            _build_pr_size(offline, loader, set(comments["pr_id"].unique()))

        # -- Summary --
        summary = (
            comments.groupby(["agent", "artifact_kind", "is_bot"])
            .agg(n_comments=("id", "count"), n_prs=("pr_id", "nunique"))
            .reset_index()
            .sort_values(["agent", "n_comments"], ascending=[True, False])
        )
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved summary → {SUMMARY_CSV}")

        print("\n--- Comments by artifact kind ---")
        print(comments["artifact_kind"].value_counts().to_string())

        print("\n--- Test-file comments: human vs bot ---")
        test_only = comments[comments["is_test_comment"]]
        print(
            f"  human: {(~test_only['is_bot']).sum():,} | "
            f"bot: {test_only['is_bot'].sum():,} | "
            f"PRs: {test_only['pr_id'].nunique():,}"
        )

        print("\n--- Comments per agent ---")
        print(
            comments.groupby("agent")
            .agg(n_comments=("id", "count"), n_prs=("pr_id", "nunique"))
            .sort_values("n_comments", ascending=False)
            .to_string()
        )

        n_test_prs = test_only["pr_id"].nunique()
        if n_test_prs < 50:
            print(
                f"\nWarning: only {n_test_prs} PRs carry test-file comments — "
                "regression models will have limited power."
            )

    finally:
        if loader:
            loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4 review-comment corpus")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, metavar="PATH")
    parser.add_argument(
        "--skip-size", action="store_true",
        help="Skip the heavy pr_commit_details read used for PR size controls",
    )
    args = parser.parse_args()
    main(offline=args.offline, zip_path=args.zip, skip_size=args.skip_size)
