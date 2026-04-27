#!/usr/bin/env python3
"""
RQ3 — Script 02: Aggregate PSQI per PR
=======================================
Agrega as métricas PSQI (Performance Script Quality Index) do nível de
arquivo para o nível de PR, calcula o score final normalizado (0–10) e
produz estatísticas descritivas por agente.

PSQI_score = mean(dim_load_type, dim_think_time, dim_payload,
                  dim_negative, dim_sla)  × 5   → escala 0–10

Inputs
------
  AIDev/rq3_features_raw.parquet  — produzido por rq3_01_extract_features.py
  AIDev/rq3_corpus.parquet        — metadados (agent, is_human, task_type, ...)

Outputs
-------
  AIDev/rq3_psqi_per_pr.parquet   — uma linha por PR com PSQI e metadados
  AIDev/rq3_psqi_summary.csv      — estatísticas descritivas por agente
  AIDev/rq3_tool_distribution.csv — contagem de ferramenta × agente

Usage
-----
  python rq3_02_aggregate_psqi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import NAME_MAPPING, PSQI_DIMENSIONS  # noqa: E402

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

FEATURES_PARQUET = OUT_DIR / "rq3_features_raw.parquet"
CORPUS_PARQUET = OUT_DIR / "rq3_corpus.parquet"
PSQI_PR_PARQUET = OUT_DIR / "rq3_psqi_per_pr.parquet"
SUMMARY_CSV = OUT_DIR / "rq3_psqi_summary.csv"
TOOL_DIST_CSV = OUT_DIR / "rq3_tool_distribution.csv"

DIMS = PSQI_DIMENSIONS  # ["dim_load_type", ..., "dim_sla"]
DIM_MEANS = [f"mean_{d}" for d in DIMS]


# ---------------------------------------------------------------------------
# Per-PR aggregation
# ---------------------------------------------------------------------------

def aggregate_per_pr(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate file-level PSQI scores to PR level."""
    rows = []
    for pr_id, grp in features.groupby("pr_id"):
        n_files = len(grp)

        # Mean of each dimension across all perf files in this PR
        dim_means = {
            f"mean_{d}": grp[d].mean() if d in grp.columns else 0.0
            for d in DIMS
        }

        # PSQI score (0–10): mean of all 5 dimension means × 5
        dim_vals = [dim_means[f"mean_{d}"] for d in DIMS]
        psqi_score = (sum(dim_vals) / len(DIMS)) * 5.0

        # Tool category info
        tool_counts = grp["tool_category"].value_counts()
        primary_tool = tool_counts.index[0] if not tool_counts.empty else "unknown"
        tool_list = sorted(grp["tool_category"].dropna().unique().tolist())

        row: dict = {
            "pr_id": pr_id,
            "psqi_score": psqi_score,
            "n_perf_files": n_files,
            "primary_tool": primary_tool,
            "tool_categories_json": json.dumps(tool_list),
            "score_profile_json": json.dumps(
                {d: round(dim_means[f"mean_{d}"], 4) for d in DIMS}
            ),
        }
        row.update(dim_means)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(psqi_pr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, grp in psqi_pr.groupby("agent"):
        rows.append({
            "agent": agent,
            "n_prs": len(grp),
            "psqi_median": grp["psqi_score"].median(),
            "psqi_mean": grp["psqi_score"].mean(),
            "psqi_std": grp["psqi_score"].std(),
            "psqi_q1": grp["psqi_score"].quantile(0.25),
            "psqi_q3": grp["psqi_score"].quantile(0.75),
            **{
                f"{m}_median": grp[m].median() if m in grp.columns else None
                for m in DIM_MEANS
            },
        })
    return pd.DataFrame(rows).sort_values("agent")


def compute_tool_distribution(psqi_pr: pd.DataFrame) -> pd.DataFrame:
    """Count primary_tool × agent for chi-square analysis."""
    return (
        psqi_pr.groupby(["agent", "primary_tool"])
        .size()
        .reset_index(name="count")
        .sort_values(["agent", "count"], ascending=[True, False])
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (FEATURES_PARQUET, CORPUS_PARQUET):
        if not path.exists():
            print(f"ERROR: {path} not found. Run preceding scripts first.")
            sys.exit(1)

    print("Loading rq3_features_raw.parquet...")
    features = pd.read_parquet(FEATURES_PARQUET)
    print(f"  {len(features):,} rows")

    # Ensure all dim columns exist
    for d in DIMS:
        if d not in features.columns:
            features[d] = 0

    print("Loading rq3_corpus.parquet...")
    corpus = pd.read_parquet(CORPUS_PARQUET)
    corpus["agent"] = corpus["agent"].map(NAME_MAPPING).fillna(corpus["agent"])
    print(f"  {len(corpus):,} corpus PRs")

    print("\nAggregating PSQI per PR...")
    psqi_pr = aggregate_per_pr(features)
    print(f"  {len(psqi_pr):,} PRs with performance files scored")

    # Attach corpus metadata
    meta_cols = ["pr_id", "agent", "is_human", "task_type", "repo_language"]
    corpus_meta = corpus[meta_cols].drop_duplicates(subset=["pr_id"])
    psqi_pr = psqi_pr.merge(corpus_meta, on="pr_id", how="left")

    # PRs in corpus but with no performance files found (no patch data)
    missing_prs = set(corpus["pr_id"]) - set(psqi_pr["pr_id"])
    if missing_prs:
        print(
            f"  Note: {len(missing_prs):,} corpus PRs have no extractable "
            "performance file patches (likely human PRs or missing patches)."
        )

    psqi_pr.to_parquet(PSQI_PR_PARQUET, index=False)
    print(f"Saved {len(psqi_pr):,} rows → {PSQI_PR_PARQUET}")

    print("\nComputing summary statistics by agent...")
    summary = compute_summary(psqi_pr)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved → {SUMMARY_CSV}")

    print("\nComputing tool distribution...")
    tool_dist = compute_tool_distribution(psqi_pr)
    tool_dist.to_csv(TOOL_DIST_CSV, index=False)
    print(f"Saved → {TOOL_DIST_CSV}")

    print("\n--- PSQI Summary by Agent ---")
    display_cols = ["agent", "n_prs", "psqi_median", "psqi_mean", "psqi_std"]
    print(summary[display_cols].to_string(index=False))

    print("\n--- Tool Distribution ---")
    pivot = tool_dist.pivot_table(
        index="agent", columns="primary_tool",
        values="count", fill_value=0
    )
    print(pivot.to_string())

    ai_prs = psqi_pr[~psqi_pr["is_human"].fillna(False)]
    human_prs = psqi_pr[psqi_pr["is_human"].fillna(False)]
    print(
        f"\nAI PRs with PSQI: {len(ai_prs):,} | "
        f"Human PRs with PSQI: {len(human_prs):,}"
    )

    zero_psqi = (psqi_pr["psqi_score"] == 0).mean() * 100
    print(f"PRs with PSQI == 0: {zero_psqi:.1f}%")


if __name__ == "__main__":
    main()
