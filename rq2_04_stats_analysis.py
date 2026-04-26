#!/usr/bin/env python3
"""
RQ2 — Script 04: Statistical Analysis and Figures
===================================================
Executa a bateria completa de testes estatísticos para responder à RQ2:
  - Mann-Whitney U + Cliff's delta (AI global vs. Human)
  - Per-agent vs. Human + Bonferroni
  - Kruskal-Wallis entre agentes + post-hoc Bonferroni pairwise
  - Estratificação por task_type (feat, fix, refactor, test)
  - Mesma bateria para Delta-TDT (PRs com ≥2 commits)
  - Fisher's exact test por smell (prevalência agêntica vs. humana)

Gera 4 figuras de publicação.

Inputs
------
  AIDev/rq2_tdt_per_pr.parquet
  AIDev/rq2_tdt_per_commit.parquet
  AIDev/rq2_cooccurrence_phi.csv  (opcional — para resumo)

Outputs
-------
  AIDev/rq2_stats_summary.csv
  figs/rq2_tdt_boxplot.png
  figs/rq2_delta_tdt_boxplot.png
  figs/rq2_smell_prevalence.png
  figs/rq2_temporal_trajectories.png

Usage
-----
  python rq2_04_stats_analysis.py
  python rq2_04_stats_analysis.py --no-figs
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    COLOR_MAP,
    FIG_DIR,
    NAME_MAPPING,
    PREFER_ORDER,
    SMELL_CATALOG,
    mannUandCliffdelta,
)

OUT_DIR = ROOT_DIR / "AIDev"
TDT_PR_PARQUET = OUT_DIR / "rq2_tdt_per_pr.parquet"
TDT_COMMIT_PARQUET = OUT_DIR / "rq2_tdt_per_commit.parquet"
PHI_CSV = OUT_DIR / "rq2_cooccurrence_phi.csv"
STATS_CSV = OUT_DIR / "rq2_stats_summary.csv"

FIG_DIR.mkdir(exist_ok=True)
SMELL_NAMES = list(SMELL_CATALOG.keys())
ELIGIBLE_TASK_TYPES = {"feat", "fix", "refactor", "test"}
AI_AGENTS = [a for a in PREFER_ORDER if a != "Human"]


# ---------------------------------------------------------------------------
# Statistical battery helpers
# ---------------------------------------------------------------------------

def _run_battery(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str,
    label_b: str,
    metric: str,
    all_stats: list[dict],
) -> None:
    """Run Mann-Whitney U + Cliff's delta, append to all_stats."""
    a = series_a.dropna().tolist()
    b = series_b.dropna().tolist()
    if len(a) < 5 or len(b) < 5:
        print(f"  [skip] {label_a} vs {label_b} — too few samples ({len(a)}, {len(b)})")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, p, d, size = mannUandCliffdelta(a, b)

    all_stats.append({
        "metric": metric,
        "group_a": label_a,
        "group_b": label_b,
        "n_a": len(a),
        "n_b": len(b),
        "median_a": np.median(a),
        "median_b": np.median(b),
        "mwu_u": u,
        "mwu_p": p,
        "cliffs_d": d,
        "cliffs_size": size,
    })


def bonferroni_correct(all_stats: list[dict], metric: str) -> None:
    """Add bonferroni_p column in-place for a given metric subset."""
    subset = [s for s in all_stats if s["metric"] == metric and "mwu_p" in s]
    if not subset:
        return
    pvals = [s["mwu_p"] for s in subset]
    _, pvals_corrected, _, _ = multipletests(pvals, method="bonferroni")
    for s, pc in zip(subset, pvals_corrected):
        s["bonferroni_p"] = pc
        s["significant_bonferroni"] = pc < 0.05


# ---------------------------------------------------------------------------
# Main statistical tests
# ---------------------------------------------------------------------------

def run_tdt_tests(per_pr: pd.DataFrame, all_stats: list[dict]) -> None:
    ai = per_pr[~per_pr["is_human"]]
    human = per_pr[per_pr["is_human"]]

    print("\n=== TDT Index — Global AI vs Human ===")
    _run_battery(ai["tdt_last"], human["tdt_last"],
                 "AI (all agents)", "Human", "tdt_index", all_stats)

    print("\n=== TDT Index — Per-agent vs Human ===")
    for agent_raw in AI_AGENTS:
        agent = NAME_MAPPING.get(agent_raw, agent_raw)
        grp = ai[ai["agent"] == agent]
        if grp.empty:
            continue
        _run_battery(grp["tdt_last"], human["tdt_last"],
                     agent, "Human", "tdt_index_per_agent", all_stats)
    bonferroni_correct(all_stats, "tdt_index_per_agent")

    print("\n=== TDT Index — Kruskal-Wallis across AI agents ===")
    groups = [ai[ai["agent"] == NAME_MAPPING.get(a, a)]["tdt_last"].dropna().tolist()
              for a in AI_AGENTS]
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) >= 2:
        h, p_kw = stats.kruskal(*groups)
        print(f"  H={h:.3f}  p={p_kw:.4f}")
        all_stats.append({
            "metric": "tdt_index_kruskal", "group_a": "all_agents",
            "mwu_u": h, "mwu_p": p_kw,
        })
        if p_kw < 0.05:
            print("  Post-hoc pairwise (Bonferroni):")
            ph_stats: list[dict] = []
            for (a1, g1), (a2, g2) in combinations(
                zip(AI_AGENTS, groups), 2
            ):
                _run_battery(pd.Series(g1), pd.Series(g2), a1, a2,
                             "tdt_index_posthoc", ph_stats)
            bonferroni_correct(ph_stats, "tdt_index_posthoc")
            all_stats.extend(ph_stats)

    print("\n=== TDT Index — Stratified by task_type ===")
    for tt in ELIGIBLE_TASK_TYPES:
        ai_tt = ai[ai["task_type"] == tt]
        hum_tt = human[human["task_type"] == tt]
        if len(ai_tt) < 5 or len(hum_tt) < 5:
            continue
        print(f"  task_type={tt}")
        _run_battery(ai_tt["tdt_last"], hum_tt["tdt_last"],
                     f"AI ({tt})", f"Human ({tt})", f"tdt_index_by_task_{tt}", all_stats)


def run_delta_tdt_tests(per_pr: pd.DataFrame, all_stats: list[dict]) -> None:
    valid = per_pr[per_pr["delta_tdt_valid"]]
    ai = valid[~valid["is_human"]]
    human = valid[valid["is_human"]]

    n_valid = len(valid)
    n_total = len(per_pr)
    print(f"\n=== Delta-TDT — {n_valid}/{n_total} PRs with temporal data ===")

    if len(ai) < 5 or len(human) < 5:
        print("  Too few samples for Delta-TDT tests.")
        return

    _run_battery(ai["delta_tdt"], human["delta_tdt"],
                 "AI (all agents)", "Human", "delta_tdt", all_stats)

    for agent_raw in AI_AGENTS:
        agent = NAME_MAPPING.get(agent_raw, agent_raw)
        grp = ai[ai["agent"] == agent]
        if grp.empty:
            continue
        _run_battery(grp["delta_tdt"], human["delta_tdt"],
                     agent, "Human", "delta_tdt_per_agent", all_stats)
    bonferroni_correct(all_stats, "delta_tdt_per_agent")


def run_smell_prevalence_tests(per_pr: pd.DataFrame, all_stats: list[dict]) -> None:
    """Fisher's exact test per smell (presence vs. absence × agentic vs. human)."""
    print("\n=== Smell Prevalence — Fisher's Exact Test ===")
    ai = per_pr[~per_pr["is_human"]]
    human = per_pr[per_pr["is_human"]]

    p_values = []
    smell_rows = []
    for s in SMELL_NAMES:
        # Parse smell_profile_json to get counts
        def _has_smell(row, s=s):
            try:
                profile = json.loads(row["smell_profile_json"])
                return int(profile.get(s, 0) > 0)
            except Exception:
                return 0

        ai_has = ai.apply(_has_smell, axis=1)
        hum_has = human.apply(_has_smell, axis=1)

        n11 = ai_has.sum()           # AI with smell
        n10 = len(ai_has) - n11      # AI without
        n01 = hum_has.sum()          # Human with smell
        n00 = len(hum_has) - n01     # Human without

        table = [[n11, n10], [n01, n00]]
        try:
            _, p = stats.fisher_exact(table)
        except Exception:
            p = 1.0

        pct_ai = n11 / len(ai_has) * 100 if len(ai_has) else 0
        pct_hum = n01 / len(hum_has) * 100 if len(hum_has) else 0
        print(f"  {s:<25} AI={pct_ai:4.1f}% Hum={pct_hum:4.1f}% p={p:.4f}")

        p_values.append(p)
        smell_rows.append({
            "metric": "smell_prevalence",
            "group_a": "AI",
            "group_b": "Human",
            "smell": s,
            "n_a": int(n11),
            "n_b": int(n01),
            "pct_a": round(pct_ai, 2),
            "pct_b": round(pct_hum, 2),
            "mwu_p": p,
        })

    # Bonferroni
    _, pvals_corrected, _, _ = multipletests(p_values, method="bonferroni")
    for row, pc in zip(smell_rows, pvals_corrected):
        row["bonferroni_p"] = pc
        row["significant_bonferroni"] = pc < 0.05

    all_stats.extend(smell_rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _agent_order(per_pr: pd.DataFrame) -> list[str]:
    """Return agents present in the data in protocol order."""
    order = [NAME_MAPPING.get(a, a) for a in PREFER_ORDER]
    present = set(per_pr["agent"].unique())
    return [a for a in order if a in present]


def fig_tdt_boxplot(per_pr: pd.DataFrame, output: Path) -> None:
    order = _agent_order(per_pr)
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [per_pr[per_pr["agent"] == a]["tdt_last"].dropna().tolist() for a in order]
    colors = [COLOR_MAP.get(a, "#999999") for a in order]

    vp = ax.violinplot(data, positions=range(len(order)), showmedians=False,
                       showextrema=False)
    for i, (body, col) in enumerate(zip(vp["bodies"], colors)):
        body.set_facecolor(col)
        body.set_alpha(0.6)

    bp = ax.boxplot(data, positions=range(len(order)), widths=0.15,
                    patch_artist=True, medianprops={"color": "black", "linewidth": 2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("TDT Index (last commit)")
    ax.set_title("Test Technical Debt Index by Agent (RQ2)")
    plt.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output}")


def fig_delta_tdt_boxplot(per_pr: pd.DataFrame, output: Path) -> None:
    valid = per_pr[per_pr["delta_tdt_valid"]]
    order = _agent_order(valid)
    data = [valid[valid["agent"] == a]["delta_tdt"].dropna().tolist() for a in order]
    colors = [COLOR_MAP.get(a, "#999999") for a in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.5,
                    patch_artist=True, medianprops={"color": "black", "linewidth": 2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="No change")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Delta-TDT (last − first commit)")
    ax.set_title("Test Debt Evolution Across PR Lifecycle (Delta-TDT)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output}")


def fig_smell_prevalence(per_pr: pd.DataFrame, output: Path) -> None:
    order = _agent_order(per_pr)
    prevalence: dict[str, list[float]] = {s: [] for s in SMELL_NAMES}

    for agent in order:
        grp = per_pr[per_pr["agent"] == agent]
        for s in SMELL_NAMES:
            pcts = []
            for _, row in grp.iterrows():
                try:
                    profile = json.loads(row["smell_profile_json"])
                    pcts.append(int(profile.get(s, 0) > 0))
                except Exception:
                    pass
            prevalence[s].append(np.mean(pcts) * 100 if pcts else 0)

    x = np.arange(len(order))
    width = 0.7 / len(SMELL_NAMES)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, s in enumerate(SMELL_NAMES):
        ax.bar(x + i * width, prevalence[s], width,
               label=s, alpha=0.85)

    ax.set_xticks(x + width * len(SMELL_NAMES) / 2)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("% PRs with smell present")
    ax.set_title("Test Smell Prevalence by Agent (RQ2)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output}")


def fig_temporal_trajectories(per_commit: pd.DataFrame, output: Path) -> None:
    """Median TDT across 5 commit-position quantile bins per agent."""
    order = _agent_order(per_commit)

    # Normalize commit position per PR to [0, 1]
    def _norm_pos(grp):
        mn, mx = grp["commit_order"].min(), grp["commit_order"].max()
        if mx == mn:
            grp = grp.copy()
            grp["norm_pos"] = 0.5
        else:
            grp = grp.copy()
            grp["norm_pos"] = (grp["commit_order"] - mn) / (mx - mn)
        return grp

    pc = per_commit.groupby("pr_id", group_keys=False).apply(_norm_pos)

    # Bin into 5 quantiles
    pc["pos_bin"] = pd.cut(pc["norm_pos"], bins=5,
                           labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    x = range(len(bins))

    for agent in order:
        grp = pc[pc["agent"] == agent]
        medians = [grp[grp["pos_bin"] == b]["tdt_index"].median() for b in bins]
        ax.plot(x, medians, marker="o", label=agent,
                color=COLOR_MAP.get(agent, "#999999"), linewidth=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(bins)
    ax.set_ylabel("Median TDT Index")
    ax.set_xlabel("Commit Position in PR Lifecycle")
    ax.set_title("TDT Evolution Across PR Lifecycle (by Agent)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(no_figs: bool) -> None:
    for path in (TDT_PR_PARQUET, TDT_COMMIT_PARQUET):
        if not path.exists():
            print(f"ERROR: {path} not found. Run preceding scripts first.")
            sys.exit(1)

    print("Loading data...")
    per_pr = pd.read_parquet(TDT_PR_PARQUET)
    per_commit = pd.read_parquet(TDT_COMMIT_PARQUET)
    per_pr["agent"] = per_pr["agent"].map(NAME_MAPPING).fillna(per_pr["agent"])
    per_commit["agent"] = per_commit["agent"].map(NAME_MAPPING).fillna(per_commit["agent"])

    print(f"  per_pr: {len(per_pr):,} | per_commit: {len(per_commit):,}")

    all_stats: list[dict] = []

    run_tdt_tests(per_pr, all_stats)
    run_delta_tdt_tests(per_pr, all_stats)
    run_smell_prevalence_tests(per_pr, all_stats)

    summary = pd.DataFrame(all_stats)
    summary.to_csv(STATS_CSV, index=False)
    print(f"\nSaved {len(summary):,} stat rows → {STATS_CSV}")

    if not no_figs:
        print("\nGenerating figures...")
        fig_tdt_boxplot(per_pr, FIG_DIR / "rq2_tdt_boxplot.png")
        fig_delta_tdt_boxplot(per_pr, FIG_DIR / "rq2_delta_tdt_boxplot.png")
        fig_smell_prevalence(per_pr, FIG_DIR / "rq2_smell_prevalence.png")
        fig_temporal_trajectories(per_commit, FIG_DIR / "rq2_temporal_trajectories.png")

    # Final summary
    print("\n=== FINDINGS SUMMARY ===")
    sig = summary[summary.get("significant_bonferroni", False) == True] if "significant_bonferroni" in summary.columns else pd.DataFrame()
    print(f"Significant results after Bonferroni correction: {len(sig)}")
    if not sig.empty:
        for _, r in sig.iterrows():
            print(f"  {r.get('metric', '')} | {r.get('group_a', '')} vs {r.get('group_b', '')} "
                  f"| p_bonf={r.get('bonferroni_p', '?'):.4f} "
                  f"| d={r.get('cliffs_d', '?')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 statistical analysis")
    parser.add_argument("--no-figs", action="store_true", help="Skip figure generation")
    args = parser.parse_args()
    main(no_figs=args.no_figs)
