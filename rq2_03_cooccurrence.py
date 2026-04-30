#!/usr/bin/env python3
"""
RQ2 — Script 03: Smell Co-occurrence Analysis
===============================================
Calcula a matriz de co-ocorrência de smells usando o coeficiente phi (φ)
e aplica clusterização hierárquica de Ward para identificar padrões compostos
de dívida técnica que emergem sistematicamente em testes agênticos vs. humanos.

Hipótese central: a TDT agêntica se manifesta em padrões COMPOSTOS de smells
(ex: EagerTest + MagicNumberTest + AssertionRoulette co-ocorrendo), invisíveis
à análise de smells individuais (cf. Milanese et al. effect size ≈ 0).

Inputs
------
  AIDev/rq2_tdt_per_pr.parquet — produzido por rq2_02_aggregate_tdt.py

Outputs
-------
  AIDev/rq2_cooccurrence_phi.csv       — pares de smells com phi, p-valor, Bonferroni
  AIDev/rq2_cooccurrence_clusters.csv  — cluster Ward por smell
  figs/rq2_cooccurrence_heatmap_agentic.png
  figs/rq2_cooccurrence_heatmap_human.png
  figs/rq2_cluster_dendrogram.png

Usage
-----
  python rq2_03_cooccurrence.py
  python rq2_03_cooccurrence.py --min-phi 0.15
  python rq2_03_cooccurrence.py --no-figs
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    FIG_DIR,
    PHI_THRESHOLD,
    SMELL_CATALOG,
)

OUT_DIR = ROOT_DIR / "AIDev"
TDT_PR_PARQUET = OUT_DIR / "rq2_tdt_per_pr.parquet"
PHI_CSV = OUT_DIR / "rq2_cooccurrence_phi.csv"
CLUSTER_CSV = OUT_DIR / "rq2_cooccurrence_clusters.csv"

FIG_DIR.mkdir(exist_ok=True)
SMELL_NAMES = list(SMELL_CATALOG.keys())
MIN_OCCURRENCES = 5  # skip phi for smells appearing in < 5 PRs


# ---------------------------------------------------------------------------
# Build binary presence matrix
# ---------------------------------------------------------------------------

def build_binary_matrix(per_pr: pd.DataFrame) -> pd.DataFrame:
    """Expand smell_profile_json into binary has_{smell} columns."""
    records = []
    for _, row in per_pr.iterrows():
        profile = json.loads(row["smell_profile_json"])
        rec = {"pr_id": row["pr_id"], "is_human": row["is_human"]}
        for s in SMELL_NAMES:
            rec[f"has_{s}"] = int(profile.get(s, 0) > 0)
        records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Phi coefficient computation
# ---------------------------------------------------------------------------

def _phi_pair(
    col_a: pd.Series,
    col_b: pd.Series,
) -> tuple[float, float]:
    """Return (phi, p_value) for two binary Series using chi2 contingency."""
    n11 = ((col_a == 1) & (col_b == 1)).sum()
    n10 = ((col_a == 1) & (col_b == 0)).sum()
    n01 = ((col_a == 0) & (col_b == 1)).sum()
    n00 = ((col_a == 0) & (col_b == 0)).sum()

    table = np.array([[n11, n10], [n01, n00]], dtype=float)

    # Guard against degenerate tables
    if table.min() == 0 and (n11 == 0 or n00 == 0):
        return 0.0, 1.0

    try:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        n = col_a.shape[0]
        phi = np.sqrt(chi2 / n) if chi2 > 0 else 0.0
        # Direction: positive if co-occur more than expected
        if n10 * n01 > 0:
            denom = np.sqrt((n11 + n10) * (n11 + n01) * (n10 + n00) * (n01 + n00))
            if denom > 0:
                phi_directed = (n11 * n00 - n10 * n01) / denom
                phi = phi_directed
        return float(phi), float(p)
    except Exception:
        return 0.0, 1.0


def compute_phi_matrix(binary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute phi for all C(10,2)=45 pairs. Return tidy DataFrame."""
    n_pairs = len(SMELL_NAMES) * (len(SMELL_NAMES) - 1) // 2
    alpha_bonferroni = 0.05 / n_pairs

    rows = []
    for s_a, s_b in combinations(SMELL_NAMES, 2):
        col_a = binary_df[f"has_{s_a}"]
        col_b = binary_df[f"has_{s_b}"]

        # Skip if either smell appears in too few PRs
        if col_a.sum() < MIN_OCCURRENCES or col_b.sum() < MIN_OCCURRENCES:
            continue

        phi, p = _phi_pair(col_a, col_b)
        rows.append({
            "smell_a": s_a,
            "smell_b": s_b,
            "phi": phi,
            "p_value": p,
            "significant_bonferroni": p < alpha_bonferroni,
            "notable": abs(phi) >= PHI_THRESHOLD,
        })

    return pd.DataFrame(rows)


def phi_to_square_matrix(phi_df: pd.DataFrame) -> pd.DataFrame:
    """Convert tidy phi pairs to symmetric square matrix (for heatmap)."""
    mat = pd.DataFrame(0.0, index=SMELL_NAMES, columns=SMELL_NAMES)
    for _, row in phi_df.iterrows():
        mat.loc[row["smell_a"], row["smell_b"]] = row["phi"]
        mat.loc[row["smell_b"], row["smell_a"]] = row["phi"]
    arr = mat.values.copy()
    np.fill_diagonal(arr, 1.0)
    mat = pd.DataFrame(arr, index=mat.index, columns=mat.columns)
    return mat


# ---------------------------------------------------------------------------
# Ward clustering
# ---------------------------------------------------------------------------

def run_ward_clustering(phi_matrix: pd.DataFrame) -> pd.DataFrame:
    """Run Ward's hierarchical clustering on smell distance matrix."""
    # Distance = 1 - |phi|
    dist_matrix = 1.0 - np.abs(phi_matrix.values)
    np.fill_diagonal(dist_matrix, 0.0)

    # Condensed distance form for scipy
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)
    linkage = hierarchy.linkage(condensed, method="ward")

    # Cut tree into 3 clusters
    cluster_ids = hierarchy.fcluster(linkage, t=3, criterion="maxclust")

    cluster_df = pd.DataFrame({
        "smell_name": SMELL_NAMES,
        "cluster_id": cluster_ids,
    })
    # Label each cluster by the smell with highest total |phi|
    cluster_df["cluster_label"] = cluster_df["cluster_id"].astype(str)

    return cluster_df, linkage


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _heatmap_fig(
    phi_matrix: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    mask = np.triu(np.ones_like(phi_matrix.values, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        phi_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.6,
        vmax=0.6,
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "φ (phi coefficient)"},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def _dendrogram_fig(linkage: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    hierarchy.dendrogram(
        linkage,
        labels=SMELL_NAMES,
        ax=ax,
        leaf_rotation=45,
        color_threshold=0.8,
    )
    ax.set_title("Ward's Hierarchical Clustering of Test Smell Co-occurrences", fontsize=12)
    ax.set_ylabel("Distance (1 - |φ|)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(min_phi: float, no_figs: bool) -> None:
    if not TDT_PR_PARQUET.exists():
        print(f"ERROR: {TDT_PR_PARQUET} not found. Run rq2_02_aggregate_tdt.py first.")
        sys.exit(1)

    print("Loading rq2_tdt_per_pr.parquet...")
    per_pr = pd.read_parquet(TDT_PR_PARQUET)
    print(f"  {len(per_pr):,} PRs")

    print("\nBuilding binary smell presence matrix...")
    binary_df = build_binary_matrix(per_pr)

    ai_binary = binary_df[~binary_df["is_human"]]
    human_binary = binary_df[binary_df["is_human"]]
    print(f"  Agentic: {len(ai_binary):,} | Human: {len(human_binary):,}")

    # Phi computation per subset
    print("\nComputing phi coefficients (agentic)...")
    phi_ai = compute_phi_matrix(ai_binary)
    phi_ai["subset"] = "agentic"

    print("Computing phi coefficients (human)...")
    phi_human = compute_phi_matrix(human_binary)
    phi_human["subset"] = "human"

    phi_all = pd.concat([phi_ai, phi_human], ignore_index=True)
    phi_all.to_csv(PHI_CSV, index=False)
    print(f"Saved {len(phi_all):,} rows → {PHI_CSV}")

    # Ward clustering on agentic matrix
    if len(phi_ai) > 0:
        mat_ai = phi_to_square_matrix(phi_ai)
        cluster_df, linkage = run_ward_clustering(mat_ai)
        cluster_df.to_csv(CLUSTER_CSV, index=False)
        print(f"Saved clusters → {CLUSTER_CSV}")
    else:
        print("  Warning: no phi pairs computed for agentic subset.")
        cluster_df = pd.DataFrame()
        linkage = None

    # Print top co-occurring pairs unique to agentic
    print(f"\nTop co-occurrence pairs in agentic (|φ| ≥ {min_phi}):")
    notable_ai = phi_ai[phi_ai["notable"]].sort_values("phi", ascending=False)
    if not notable_ai.empty:
        for _, r in notable_ai.head(10).iterrows():
            print(f"  {r['smell_a']} × {r['smell_b']:25} φ={r['phi']:.3f} "
                  f"sig={'*' if r['significant_bonferroni'] else ' '}")
    else:
        print("  No notable pairs found (threshold may be too high).")

    # Pairs unique to agentic (not notable in human)
    if len(phi_human) > 0:
        human_notable = set(
            zip(phi_human[phi_human["notable"]]["smell_a"],
                phi_human[phi_human["notable"]]["smell_b"])
        )
        unique_ai = notable_ai[
            ~notable_ai.apply(
                lambda r: (r["smell_a"], r["smell_b"]) in human_notable, axis=1
            )
        ]
        print(f"\nAgentic-unique notable pairs ({len(unique_ai)}):")
        for _, r in unique_ai.iterrows():
            print(f"  {r['smell_a']} × {r['smell_b']}")

    # Figures
    if not no_figs:
        print("\nGenerating figures...")
        if len(phi_ai) > 0:
            _heatmap_fig(
                phi_to_square_matrix(phi_ai),
                "Test Smell Co-occurrence (φ) — Agentic PRs",
                FIG_DIR / "rq2_cooccurrence_heatmap_agentic.png",
            )
        if len(phi_human) > 0:
            _heatmap_fig(
                phi_to_square_matrix(phi_human),
                "Test Smell Co-occurrence (φ) — Human PRs",
                FIG_DIR / "rq2_cooccurrence_heatmap_human.png",
            )
        if linkage is not None:
            _dendrogram_fig(linkage, FIG_DIR / "rq2_cluster_dendrogram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 co-occurrence analyser")
    parser.add_argument("--min-phi", type=float, default=PHI_THRESHOLD,
                        help="Minimum |phi| to flag as notable")
    parser.add_argument("--no-figs", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()
    main(min_phi=args.min_phi, no_figs=args.no_figs)
