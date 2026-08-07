#!/usr/bin/env python3
"""
RQ4 — Script 02: Aggregate Feedback Features per PR
====================================================
Agrega os comentários classificados ao nível de PR e monta a matriz de
variáveis usada na regressão logística do protocolo (§3.6.3 / §4.3.1):

  variável dependente : merged (PR aceito ou não)
  preditores          : contagem de comentários por categoria da taxonomia
  covariáveis         : MSI (RQ1), índice de TDT (RQ2)
  controles           : tamanho do PR, agente, linguagem, nº de revisores

Também calcula, por PR, o contraste central da RQ4 — feedback em artefatos de
teste vs. feedback em código de produção.

Inputs
------
  AIDev/rq4_comments_classified.parquet   — rq4_01
  AIDev/rq4_pr_reviews.parquet            — rq4_00
  AIDev/rq4_pr_size.parquet               — rq4_00 (opcional)
  AIDev/rq1_msi_results.csv               — RQ1 (opcional)
  AIDev/rq2_tdt_per_pr.parquet            — RQ2 (opcional)

Outputs
-------
  AIDev/rq4_pr_features.parquet
  AIDev/rq4_feedback_summary.csv
  AIDev/rq4_coverage_report.csv   — quantos PRs têm cada covariável

Usage
-----
  python rq4_02_aggregate_feedback.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import REVIEW_CATEGORIES  # noqa: E402

OUT_DIR = ROOT_DIR / "AIDev"

CLASSIFIED_PARQUET = OUT_DIR / "rq4_comments_classified.parquet"
REVIEWS_PARQUET = OUT_DIR / "rq4_pr_reviews.parquet"
SIZE_PARQUET = OUT_DIR / "rq4_pr_size.parquet"
MSI_CSV = OUT_DIR / "rq1_msi_results.csv"
TDT_PARQUET = OUT_DIR / "rq2_tdt_per_pr.parquet"

FEATURES_PARQUET = OUT_DIR / "rq4_pr_features.parquet"
SUMMARY_CSV = OUT_DIR / "rq4_feedback_summary.csv"
COVERAGE_CSV = OUT_DIR / "rq4_coverage_report.csv"

ALL_CATEGORIES = REVIEW_CATEGORIES + ["unclassified"]


# ---------------------------------------------------------------------------
# Per-PR aggregation
# ---------------------------------------------------------------------------

def _category_counts(
    df: pd.DataFrame, prefix: str, index: pd.Index
) -> pd.DataFrame:
    """Wide count matrix of categories for the given comment subset."""
    if df.empty:
        return pd.DataFrame(
            0, index=index, columns=[f"{prefix}_{c}" for c in ALL_CATEGORIES]
        )
    wide = (
        df.groupby(["pr_id", "category"]).size().unstack(fill_value=0)
        .reindex(columns=ALL_CATEGORIES, fill_value=0)
        .rename(columns=lambda c: f"{prefix}_{c}")
    )
    return wide.reindex(index, fill_value=0)


def aggregate_per_pr(comments: pd.DataFrame) -> pd.DataFrame:
    pr_index = pd.Index(sorted(comments["pr_id"].unique()), name="pr_id")

    # The taxonomy only describes reviewer feedback; agent fix-reports
    # ("acknowledgement") and bare code suggestions are counted separately.
    feedback = comments[comments["role"].eq("feedback")]
    ack = comments[comments["role"].eq("acknowledgement")]

    test = feedback[feedback["is_test_comment"].fillna(False)]
    prod = feedback[feedback["artifact_kind"].eq("production")]
    human = feedback[~feedback["is_bot"].fillna(False)]
    human_test = test[~test["is_bot"].fillna(False)]

    base = pd.DataFrame(index=pr_index)
    base["n_comments"] = comments.groupby("pr_id").size().reindex(pr_index, fill_value=0)
    base["n_feedback_comments"] = (
        feedback.groupby("pr_id").size().reindex(pr_index, fill_value=0)
    )
    base["n_ack_comments"] = (
        ack.groupby("pr_id").size().reindex(pr_index, fill_value=0)
    )
    base["ack_ratio"] = (
        base["n_ack_comments"] / base["n_comments"].replace(0, np.nan)
    )
    base["n_test_comments"] = test.groupby("pr_id").size().reindex(pr_index, fill_value=0)
    base["n_prod_comments"] = prod.groupby("pr_id").size().reindex(pr_index, fill_value=0)
    base["n_human_comments"] = human.groupby("pr_id").size().reindex(pr_index, fill_value=0)
    base["n_human_test_comments"] = (
        human_test.groupby("pr_id").size().reindex(pr_index, fill_value=0)
    )
    base["n_bot_comments"] = base["n_comments"] - base["n_human_comments"]
    base["has_test_feedback"] = base["n_test_comments"] > 0

    # Category counts (feedback role only): overall, test files, human+test
    base = base.join(_category_counts(feedback, "cat", pr_index))
    base = base.join(_category_counts(test, "test_cat", pr_index))
    base = base.join(_category_counts(human_test, "human_test_cat", pr_index))

    # Tone / intensity
    for label, subset in (("", feedback), ("test_", test), ("human_", human)):
        if subset.empty:
            base[f"{label}mean_tone"] = np.nan
            base[f"{label}mean_intensity"] = np.nan
            continue
        g = subset.groupby("pr_id")
        base[f"{label}mean_tone"] = g["tone"].mean().reindex(pr_index)
        base[f"{label}mean_intensity"] = g["intensity"].mean().reindex(pr_index)

    # Resolution: share of comments that received a reply
    answered = comments.assign(_ans=comments["resolution"].eq("answered"))
    base["resolution_rate"] = (
        answered.groupby("pr_id")["_ans"].mean().reindex(pr_index)
    )
    if not test.empty:
        ans_t = test.assign(_ans=test["resolution"].eq("answered"))
        base["test_resolution_rate"] = (
            ans_t.groupby("pr_id")["_ans"].mean().reindex(pr_index)
        )
    else:
        base["test_resolution_rate"] = np.nan

    base["n_distinct_comment_reviewers"] = (
        comments.groupby("pr_id")["reviewer_id"].nunique().reindex(pr_index, fill_value=0)
    )

    # PR-level metadata (constant within a PR)
    meta = (
        comments.groupby("pr_id")
        .agg(
            agent=("agent", "first"),
            repo_language=("repo_language", "first"),
            task_type=("task_type", "first"),
            merged=("merged", "first"),
            state=("state", "first"),
        )
        .reindex(pr_index)
    )
    return base.join(meta).reset_index()


# ---------------------------------------------------------------------------
# Covariate joins
# ---------------------------------------------------------------------------

def attach_quality_metrics(pr_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    coverage: list[dict] = []

    if TDT_PARQUET.exists():
        tdt = pd.read_parquet(TDT_PARQUET)
        cols = ["pr_id", "tdt_first", "tdt_last", "delta_tdt", "tdt_max"]
        cols = [c for c in cols if c in tdt.columns]
        pr_df = pr_df.merge(tdt[cols], on="pr_id", how="left")
        n = pr_df["tdt_last"].notna().sum() if "tdt_last" in pr_df else 0
        print(f"  TDT joined: {n:,} / {len(pr_df):,} PRs")
        coverage.append({"covariate": "TDT", "n_prs": int(n),
                         "pct": round(n / len(pr_df) * 100, 2)})
    else:
        print(f"  [skip] {TDT_PARQUET.name} not found — TDT unavailable")
        coverage.append({"covariate": "TDT", "n_prs": 0, "pct": 0.0})

    if MSI_CSV.exists():
        msi = pd.read_csv(MSI_CSV, usecols=["pr_id", "msi_global",
                                            "cyclomatic_complexity",
                                            "assertion_count", "test_loc"])
        msi = msi.drop_duplicates(subset=["pr_id"])
        pr_df = pr_df.merge(msi, on="pr_id", how="left")
        n = pr_df["msi_global"].notna().sum()
        print(f"  MSI joined: {n:,} / {len(pr_df):,} PRs")
        coverage.append({"covariate": "MSI", "n_prs": int(n),
                         "pct": round(n / len(pr_df) * 100, 2)})
    else:
        print(f"  [skip] {MSI_CSV.name} not found — MSI unavailable")
        coverage.append({"covariate": "MSI", "n_prs": 0, "pct": 0.0})

    if SIZE_PARQUET.exists():
        size = pd.read_parquet(SIZE_PARQUET)
        pr_df = pr_df.merge(size, on="pr_id", how="left")
        n = pr_df["pr_churn"].notna().sum()
        print(f"  PR size joined: {n:,} / {len(pr_df):,} PRs")
        coverage.append({"covariate": "PR size", "n_prs": int(n),
                         "pct": round(n / len(pr_df) * 100, 2)})
    else:
        print(f"  [skip] {SIZE_PARQUET.name} not found — size control absent")
        coverage.append({"covariate": "PR size", "n_prs": 0, "pct": 0.0})

    if REVIEWS_PARQUET.exists():
        rev = pd.read_parquet(REVIEWS_PARQUET)
        pr_df = pr_df.merge(rev, on="pr_id", how="left")
        rev_cols = [c for c in pr_df.columns if c.startswith("n_review_")
                    or c.startswith("n_human_review_")]
        pr_df[rev_cols] = pr_df[rev_cols].fillna(0)
        n = int((pr_df.get("n_reviews_total", pd.Series(0)) > 0).sum())
        print(f"  Review states joined: {n:,} / {len(pr_df):,} PRs")
        coverage.append({"covariate": "Review states", "n_prs": n,
                         "pct": round(n / len(pr_df) * 100, 2)})

    return pr_df, coverage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CLASSIFIED_PARQUET.exists():
        print(f"ERROR: {CLASSIFIED_PARQUET} not found. Run rq4_01 first.")
        sys.exit(1)

    print("Loading rq4_comments_classified.parquet...")
    comments = pd.read_parquet(CLASSIFIED_PARQUET)
    print(f"  {len(comments):,} comments | {comments['pr_id'].nunique():,} PRs")

    print("\nAggregating per PR...")
    pr_df = aggregate_per_pr(comments)
    print(f"  {len(pr_df):,} PRs")

    print("\nAttaching quality covariates...")
    pr_df, coverage = attach_quality_metrics(pr_df)

    pr_df.to_parquet(FEATURES_PARQUET, index=False)
    print(f"\nSaved {len(pr_df):,} PRs → {FEATURES_PARQUET}")

    pd.DataFrame(coverage).to_csv(COVERAGE_CSV, index=False)
    print(f"Saved covariate coverage → {COVERAGE_CSV}")

    # -- Summary by agent --
    summary = (
        pr_df.groupby("agent")
        .agg(
            n_prs=("pr_id", "count"),
            merge_rate=("merged", "mean"),
            mean_comments=("n_comments", "mean"),
            mean_test_comments=("n_test_comments", "mean"),
            pct_with_test_feedback=("has_test_feedback", "mean"),
            mean_tone=("mean_tone", "mean"),
            mean_intensity=("mean_intensity", "mean"),
            mean_resolution=("resolution_rate", "mean"),
        )
        .reset_index()
        .sort_values("n_prs", ascending=False)
    )
    summary["merge_rate"] = (summary["merge_rate"] * 100).round(1)
    summary["pct_with_test_feedback"] = (
        summary["pct_with_test_feedback"] * 100
    ).round(1)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary → {SUMMARY_CSV}")

    print("\n--- Feedback summary by agent ---")
    print(summary.round(3).to_string(index=False))

    print("\n--- PRs with test-file feedback ---")
    with_test = pr_df[pr_df["has_test_feedback"]]
    print(f"  {len(with_test):,} PRs ({len(with_test) / len(pr_df) * 100:.1f}%)")
    print(f"  merge rate WITH test feedback:    "
          f"{with_test['merged'].mean() * 100:.1f}%")
    print(f"  merge rate WITHOUT test feedback: "
          f"{pr_df[~pr_df['has_test_feedback']]['merged'].mean() * 100:.1f}%")

    print("\n--- Category totals on test files (human reviewers) ---")
    for c in ALL_CATEGORIES:
        col = f"human_test_cat_{c}"
        if col in pr_df.columns:
            print(f"  {c:<16} {int(pr_df[col].sum()):,}")


if __name__ == "__main__":
    main()
