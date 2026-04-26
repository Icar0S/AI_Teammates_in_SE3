#!/usr/bin/env python3
"""
RQ2 — Script 02: Aggregate TDT Metrics
========================================
Calcula o índice TDT ponderado por commit e por PR, incluindo Delta-TDT
para PRs com múltiplos commits tocando arquivos de teste.

TDT_index = sum(weight[smell] * count[smell]) para cada smell no catálogo.
Delta-TDT = TDT(último commit) - TDT(primeiro commit) por PR.

Inputs
------
  AIDev/rq2_smells_raw.parquet — produzido por rq2_01_detect_smells.py
  AIDev/rq2_corpus.parquet     — metadados de agente, linguagem, task_type

Outputs
-------
  AIDev/rq2_tdt_per_commit.parquet — TDT por (pr_id, commit_sha)
  AIDev/rq2_tdt_per_pr.parquet     — TDT agregado por PR com Delta-TDT
  AIDev/rq2_tdt_summary.csv        — estatísticas descritivas por agente

Usage
-----
  python rq2_02_aggregate_tdt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    NAME_MAPPING,
    SMELL_CATALOG,
    compute_tdt_index,
)

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

SMELLS_PARQUET = OUT_DIR / "rq2_smells_raw.parquet"
CORPUS_PARQUET = OUT_DIR / "rq2_corpus.parquet"
TDT_COMMIT_PARQUET = OUT_DIR / "rq2_tdt_per_commit.parquet"
TDT_PR_PARQUET = OUT_DIR / "rq2_tdt_per_pr.parquet"
SUMMARY_CSV = OUT_DIR / "rq2_tdt_summary.csv"

SMELL_NAMES = list(SMELL_CATALOG.keys())


# ---------------------------------------------------------------------------
# Per-commit TDT
# ---------------------------------------------------------------------------

def compute_per_commit(smells: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Aggregate smell counts to (pr_id, commit_sha) level and compute TDT."""
    # Sum smell counts across all test files within the same (pr_id, commit_sha)
    agg_cols = {s: "sum" for s in SMELL_NAMES if s in smells.columns}
    commit_smells = (
        smells.groupby(["pr_id", "commit_sha"])
        .agg(agg_cols)
        .reset_index()
    )

    # Compute TDT index per row
    commit_smells["tdt_index"] = commit_smells.apply(
        lambda r: compute_tdt_index({s: r[s] for s in SMELL_NAMES}), axis=1
    )

    # Attach commit_order, agent, is_human from corpus
    meta_cols = ["pr_id", "commit_sha", "commit_order", "agent",
                 "is_human", "task_type", "repo_language"]
    # corpus may have duplicate (pr_id, commit_sha) when multiple test files
    # exist per commit — deduplicate to 1 row per pair for the join
    corpus_dedup = (
        corpus[meta_cols]
        .drop_duplicates(subset=["pr_id", "commit_sha"])
    )

    result = commit_smells.merge(corpus_dedup, on=["pr_id", "commit_sha"], how="left")
    result["agent"] = result["agent"].map(NAME_MAPPING).fillna(result["agent"])
    result = result.sort_values(["pr_id", "commit_order"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Per-PR TDT
# ---------------------------------------------------------------------------

def compute_per_pr(per_commit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate commit-level TDT to PR level; compute Delta-TDT."""
    rows = []
    for pr_id, group in per_commit.groupby("pr_id"):
        group = group.sort_values("commit_order")

        n = len(group)
        tdt_first = group.iloc[0]["tdt_index"]
        tdt_last = group.iloc[-1]["tdt_index"]
        delta_tdt = tdt_last - tdt_first
        tdt_max = group["tdt_index"].max()

        # Smell profile: total count of each smell across all commits in this PR
        smell_profile = {
            s: int(group[s].sum()) for s in SMELL_NAMES if s in group.columns
        }

        agent = group["agent"].iloc[0]
        is_human = group["is_human"].iloc[0] if "is_human" in group.columns else False
        task_type = group["task_type"].iloc[0] if "task_type" in group.columns else None

        rows.append({
            "pr_id": pr_id,
            "agent": agent,
            "is_human": bool(is_human),
            "task_type": task_type,
            "tdt_first": tdt_first,
            "tdt_last": tdt_last,
            "delta_tdt": delta_tdt,
            "tdt_max": tdt_max,
            "n_commits_with_tests": n,
            "delta_tdt_valid": n >= 2,
            "smell_profile_json": json.dumps(smell_profile),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(per_pr: pd.DataFrame) -> pd.DataFrame:
    stats_rows = []
    for agent, grp in per_pr.groupby("agent"):
        valid_delta = grp[grp["delta_tdt_valid"]]["delta_tdt"]
        stats_rows.append({
            "agent": agent,
            "n_prs": len(grp),
            "n_with_temporal": int(grp["delta_tdt_valid"].sum()),
            "tdt_median": grp["tdt_last"].median(),
            "tdt_mean": grp["tdt_last"].mean(),
            "tdt_q1": grp["tdt_last"].quantile(0.25),
            "tdt_q3": grp["tdt_last"].quantile(0.75),
            "delta_tdt_median": valid_delta.median() if len(valid_delta) else None,
            "delta_tdt_mean": valid_delta.mean() if len(valid_delta) else None,
        })
    return pd.DataFrame(stats_rows).sort_values("agent")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (SMELLS_PARQUET, CORPUS_PARQUET):
        if not path.exists():
            print(f"ERROR: {path} not found. Run preceding scripts first.")
            sys.exit(1)

    print("Loading rq2_smells_raw.parquet...")
    smells = pd.read_parquet(SMELLS_PARQUET)
    print(f"  {len(smells):,} rows")

    # Ensure all smell columns exist (fill 0 for any missing)
    for s in SMELL_NAMES:
        if s not in smells.columns:
            smells[s] = 0

    print("Loading rq2_corpus.parquet...")
    corpus = pd.read_parquet(CORPUS_PARQUET)
    print(f"  {len(corpus):,} rows")

    print("\nComputing per-commit TDT...")
    per_commit = compute_per_commit(smells, corpus)
    per_commit.to_parquet(TDT_COMMIT_PARQUET, index=False)
    print(f"  Saved {len(per_commit):,} rows → {TDT_COMMIT_PARQUET}")

    print("\nComputing per-PR TDT (Delta-TDT)...")
    per_pr = compute_per_pr(per_commit)
    per_pr.to_parquet(TDT_PR_PARQUET, index=False)
    print(f"  Saved {len(per_pr):,} rows → {TDT_PR_PARQUET}")

    print("\nComputing summary statistics...")
    summary = compute_summary(per_pr)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  Saved → {SUMMARY_CSV}")

    print("\n--- TDT Summary by Agent ---")
    print(summary.to_string(index=False))

    # Alert if temporal coverage is low
    pct_temporal = per_pr["delta_tdt_valid"].mean() * 100
    print(f"\nPRs with temporal data (≥2 commits): {pct_temporal:.1f}%")
    if pct_temporal < 20:
        print("  Warning: low temporal coverage — Delta-TDT analysis may have limited power.")

    zero_prs = (per_pr["tdt_last"] == 0).mean() * 100
    print(f"PRs with zero TDT (all commits): {zero_prs:.1f}%")


if __name__ == "__main__":
    main()
