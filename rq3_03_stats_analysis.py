#!/usr/bin/env python3
"""
RQ3 — Script 03: Statistical Analysis and Publication Figures
=============================================================
Bateria completa de testes estatísticos para responder à RQ3:
  1. PSQI global — Mann-Whitney U + Cliff's delta (AI vs. Human)
  2. PSQI por agente — MWU + Bonferroni vs. Human
  3. Kruskal-Wallis entre agentes AI + post-hoc pairwise
  4. Por dimensão PSQI — MWU para cada uma das 5 dimensões (AI vs. Human)
  5. Distribuição de ferramentas — chi-quadrado + Cramér's V (AI vs. Human)
  6. Estratificação por task_type (perf, feat, fix)

Gera 4 figuras de publicação.

Inputs
------
  AIDev/rq3_psqi_per_pr.parquet
  AIDev/rq3_tool_distribution.csv

Outputs
-------
  AIDev/rq3_stats_summary.csv
  figs/rq3_psqi_boxplot.png
  figs/rq3_dimension_radar.png
  figs/rq3_tool_distribution.png
  figs/rq3_dimension_boxplot.png

Usage
-----
  python rq3_03_stats_analysis.py
  python rq3_03_stats_analysis.py --no-figs
"""

from __future__ import annotations

import argparse
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
    PERF_TOOL_ORDER,
    PREFER_ORDER,
    PSQI_DIMENSIONS,
    mannUandCliffdelta,
)

OUT_DIR = ROOT_DIR / "AIDev"
FIG_DIR.mkdir(exist_ok=True)

PSQI_PR_PARQUET = OUT_DIR / "rq3_psqi_per_pr.parquet"
TOOL_DIST_CSV = OUT_DIR / "rq3_tool_distribution.csv"
STATS_CSV = OUT_DIR / "rq3_stats_summary.csv"

DIMS = PSQI_DIMENSIONS
DIM_MEANS = [f"mean_{d}" for d in DIMS]
DIM_LABELS = {
    "dim_load_type": "Load Type",
    "dim_think_time": "Think Time",
    "dim_payload": "Payload Var.",
    "dim_negative": "Neg. Scenarios",
    "dim_sla": "SLA Assertions",
}

AI_AGENTS = [a for a in PREFER_ORDER if a != "Human"]
ELIGIBLE_TASK_TYPES = {"perf", "feat", "fix"}

_MIN_SAMPLES = 5


# ---------------------------------------------------------------------------
# Statistical battery helpers  (mirror rq2_04 patterns)
# ---------------------------------------------------------------------------

def _run_battery(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str,
    label_b: str,
    metric: str,
    all_stats: list[dict],
) -> None:
    a = series_a.dropna().tolist()
    b = series_b.dropna().tolist()
    if len(a) < _MIN_SAMPLES or len(b) < _MIN_SAMPLES:
        print(
            f"  [skip] {label_a} vs {label_b} — "
            f"too few samples ({len(a)}, {len(b)})"
        )
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
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "mwu_u": u,
        "mwu_p": p,
        "cliffs_d": d,
        "cliffs_size": size,
    })


def _bonferroni(all_stats: list[dict], metric: str) -> None:
    subset = [s for s in all_stats if s["metric"] == metric and "mwu_p" in s]
    if not subset:
        return
    _, corrected, _, _ = multipletests(
        [s["mwu_p"] for s in subset], method="bonferroni"
    )
    for s, pc in zip(subset, corrected):
        s["bonferroni_p"] = pc
        s["significant_bonferroni"] = pc < 0.05


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_psqi_tests(psqi_pr: pd.DataFrame, all_stats: list[dict]) -> None:
    ai = psqi_pr[~psqi_pr["is_human"].fillna(False)]
    human = psqi_pr[psqi_pr["is_human"].fillna(False)]

    print("\n=== PSQI Score — Global AI vs Human ===")
    _run_battery(
        ai["psqi_score"], human["psqi_score"],
        "AI (all agents)", "Human", "psqi_global", all_stats,
    )

    print("\n=== PSQI Score — Per-agent vs Human ===")
    for agent_raw in AI_AGENTS:
        agent = NAME_MAPPING.get(agent_raw, agent_raw)
        grp = ai[ai["agent"] == agent]
        if grp.empty:
            continue
        _run_battery(
            grp["psqi_score"], human["psqi_score"],
            agent, "Human", "psqi_per_agent", all_stats,
        )
    _bonferroni(all_stats, "psqi_per_agent")

    print("\n=== PSQI Score — Kruskal-Wallis across AI agents ===")
    groups = [
        ai[ai["agent"] == NAME_MAPPING.get(a, a)]["psqi_score"]
        .dropna().tolist()
        for a in AI_AGENTS
    ]
    groups = [g for g in groups if len(g) >= _MIN_SAMPLES]
    if len(groups) >= 2:
        h, p_kw = stats.kruskal(*groups)
        print(f"  H={h:.3f}  p={p_kw:.4f}")
        all_stats.append({
            "metric": "psqi_kruskal",
            "group_a": "all_agents",
            "mwu_u": h,
            "mwu_p": p_kw,
        })
        if p_kw < 0.05:
            print("  Post-hoc pairwise (Bonferroni):")
            ph: list[dict] = []
            for (a1, g1), (a2, g2) in combinations(
                zip(AI_AGENTS, groups), 2
            ):
                _run_battery(
                    pd.Series(g1), pd.Series(g2),
                    a1, a2, "psqi_posthoc", ph,
                )
            _bonferroni(ph, "psqi_posthoc")
            all_stats.extend(ph)

    print("\n=== PSQI Score — Stratified by task_type ===")
    for tt in ELIGIBLE_TASK_TYPES:
        ai_tt = ai[ai["task_type"] == tt]
        hum_tt = human[human["task_type"] == tt]
        if len(ai_tt) < _MIN_SAMPLES or len(hum_tt) < _MIN_SAMPLES:
            continue
        print(f"  task_type={tt}")
        _run_battery(
            ai_tt["psqi_score"], hum_tt["psqi_score"],
            f"AI ({tt})", f"Human ({tt})", f"psqi_task_{tt}", all_stats,
        )


def run_dimension_tests(psqi_pr: pd.DataFrame, all_stats: list[dict]) -> None:
    ai = psqi_pr[~psqi_pr["is_human"].fillna(False)]
    human = psqi_pr[psqi_pr["is_human"].fillna(False)]

    print("\n=== Per-Dimension PSQI — AI vs Human (Bonferroni over 5 dims) ===")
    for d in DIMS:
        col = f"mean_{d}"
        if col not in psqi_pr.columns:
            continue
        _run_battery(
            ai[col], human[col],
            "AI (all agents)", "Human", f"dim_{d}", all_stats,
        )
    for d in DIMS:
        _bonferroni(all_stats, f"dim_{d}")


def run_tool_chi_square(
    psqi_pr: pd.DataFrame, all_stats: list[dict]
) -> None:
    """Chi-square test for tool strategy distribution (AI vs Human)."""
    print("\n=== Tool Strategy Distribution — Chi-square ===")
    ai = psqi_pr[~psqi_pr["is_human"].fillna(False)]
    human = psqi_pr[psqi_pr["is_human"].fillna(False)]

    if len(human) < _MIN_SAMPLES:
        print("  [skip] too few human PRs for chi-square")
        return

    tools = psqi_pr["primary_tool"].dropna().unique().tolist()
    ai_counts = ai["primary_tool"].value_counts().reindex(tools, fill_value=0)
    hum_counts = human["primary_tool"].value_counts().reindex(tools, fill_value=0)

    contingency = np.array([ai_counts.values, hum_counts.values])
    # Remove zero-sum columns
    nonzero = contingency.sum(axis=0) > 0
    contingency = contingency[:, nonzero]

    if contingency.shape[1] < 2:
        print("  [skip] insufficient tool variety for chi-square")
        return

    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))
    print(f"  chi2={chi2:.3f}  dof={dof}  p={p:.4f}  Cramér's V={cramers_v:.3f}")
    all_stats.append({
        "metric": "tool_distribution_chi2",
        "group_a": "AI (all agents)",
        "group_b": "Human",
        "n_a": int(ai_counts.sum()),
        "n_b": int(hum_counts.sum()),
        "mwu_u": chi2,
        "mwu_p": p,
        "cliffs_d": cramers_v,
        "cliffs_size": (
            "negligible" if cramers_v < 0.1
            else "small" if cramers_v < 0.3
            else "medium" if cramers_v < 0.5
            else "large"
        ),
    })


# ---------------------------------------------------------------------------
# Figure 1 — PSQI violin + box plot by agent
# ---------------------------------------------------------------------------

def fig_psqi_boxplot(psqi_pr: pd.DataFrame) -> None:
    ordered_agents = [
        NAME_MAPPING.get(a, a)
        for a in PREFER_ORDER
        if NAME_MAPPING.get(a, a) in psqi_pr["agent"].unique()
    ]
    data = [
        psqi_pr[psqi_pr["agent"] == ag]["psqi_score"].dropna().tolist()
        for ag in ordered_agents
    ]
    colors = [COLOR_MAP.get(ag, "#888888") for ag in ordered_agents]

    fig, ax = plt.subplots(figsize=(11, 5))
    positions = range(1, len(ordered_agents) + 1)

    parts = ax.violinplot(
        [d for d in data if d], positions=[p for p, d in zip(positions, data) if d],
        showmedians=False, showextrema=False,
    )
    visible_positions = [p for p, d in zip(positions, data) if d]
    visible_colors = [c for c, d in zip(colors, data) if d]
    for pc, col in zip(parts["bodies"], visible_colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.35)

    bp = ax.boxplot(
        [d if d else [0] for d in data],
        positions=list(positions),
        widths=0.35,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(ordered_agents, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("PSQI Score (0–10)", fontsize=10)
    ax.set_title(
        "Performance Script Quality Index by Agent (RQ3)", fontsize=11
    )
    ax.set_ylim(-0.5, 10.5)
    ax.axhline(5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = FIG_DIR / "rq3_psqi_boxplot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Radar chart of 5 PSQI dimensions per agent
# ---------------------------------------------------------------------------

def fig_dimension_radar(psqi_pr: pd.DataFrame) -> None:
    ordered_agents = [
        NAME_MAPPING.get(a, a)
        for a in PREFER_ORDER
        if NAME_MAPPING.get(a, a) in psqi_pr["agent"].unique()
    ]

    labels = [DIM_LABELS[d] for d in DIMS]
    N = len(DIMS)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for agent in ordered_agents:
        grp = psqi_pr[psqi_pr["agent"] == agent]
        values = [
            grp[f"mean_{d}"].mean() if f"mean_{d}" in grp.columns else 0
            for d in DIMS
        ]
        values += values[:1]
        color = COLOR_MAP.get(agent, "#888888")
        ax.plot(angles, values, "o-", linewidth=2, label=agent, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5", "1.0", "1.5", "2.0"], fontsize=7)
    ax.set_title("PSQI Dimensions per Agent (mean score 0–2)", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "rq3_dimension_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Stacked bar: % PRs using each tool per agent
# ---------------------------------------------------------------------------

def fig_tool_distribution(psqi_pr: pd.DataFrame) -> None:
    ordered_agents = [
        NAME_MAPPING.get(a, a)
        for a in PREFER_ORDER
        if NAME_MAPPING.get(a, a) in psqi_pr["agent"].unique()
    ]
    tools = [
        t for t in PERF_TOOL_ORDER
        if t in psqi_pr["primary_tool"].values
    ]

    pivot = (
        psqi_pr.groupby(["agent", "primary_tool"])
        .size()
        .unstack(fill_value=0)
        .reindex(ordered_agents, fill_value=0)
        .reindex(columns=tools, fill_value=0)
    )
    # Convert to percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    tool_colors = {
        "k6": "#E69F00",
        "locust": "#009E73",
        "jmeter": "#CC79A7",
        "gatling": "#56B4E9",
        "pytest_benchmark": "#F0E442",
        "generic_perf": "#999999",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(ordered_agents))
    for tool in tools:
        if tool not in pivot_pct.columns:
            continue
        vals = pivot_pct[tool].values
        ax.bar(
            ordered_agents, vals, bottom=bottom,
            label=tool, color=tool_colors.get(tool, "#888888"),
        )
        bottom += vals

    ax.set_ylabel("% of PRs", fontsize=10)
    ax.set_title("Performance Tool Distribution by Agent (RQ3)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "rq3_tool_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Per-dimension box plots (5 subplots, AI vs Human)
# ---------------------------------------------------------------------------

def fig_dimension_boxplot(psqi_pr: pd.DataFrame) -> None:
    ai = psqi_pr[~psqi_pr["is_human"].fillna(False)]
    human = psqi_pr[psqi_pr["is_human"].fillna(False)]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
    for ax, dim in zip(axes, DIMS):
        col = f"mean_{dim}"
        data_ai = ai[col].dropna().tolist() if col in ai.columns else []
        data_hum = human[col].dropna().tolist() if col in human.columns else []
        data = [data_ai, data_hum]
        labels = ["AI", "Human"]
        colors_bp = [
            COLOR_MAP.get("OpenAI Codex", "#D55E00"),
            COLOR_MAP.get("Human", "#56B4E9"),
        ]
        bp = ax.boxplot(
            [d if d else [0] for d in data],
            labels=labels,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, col_color in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(col_color)
            patch.set_alpha(0.6)
        ax.set_title(DIM_LABELS[dim], fontsize=9)
        ax.set_ylim(-0.1, 2.1)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Mean Score (0–2)", fontsize=10)
    fig.suptitle(
        "PSQI Dimensions: AI (pooled) vs Human (RQ3)", fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / "rq3_dimension_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(no_figs: bool) -> None:
    if not PSQI_PR_PARQUET.exists():
        print(
            f"ERROR: {PSQI_PR_PARQUET} not found. "
            "Run rq3_02_aggregate_psqi.py first."
        )
        sys.exit(1)

    print("Loading rq3_psqi_per_pr.parquet...")
    psqi_pr = pd.read_parquet(PSQI_PR_PARQUET)
    psqi_pr["agent"] = psqi_pr["agent"].map(NAME_MAPPING).fillna(psqi_pr["agent"])
    print(f"  {len(psqi_pr):,} PRs loaded")

    ai = psqi_pr[~psqi_pr["is_human"].fillna(False)]
    human = psqi_pr[psqi_pr["is_human"].fillna(False)]
    print(f"  AI: {len(ai):,} | Human: {len(human):,}")

    all_stats: list[dict] = []

    run_psqi_tests(psqi_pr, all_stats)
    run_dimension_tests(psqi_pr, all_stats)
    run_tool_chi_square(psqi_pr, all_stats)

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"\nSaved {len(stats_df):,} test results → {STATS_CSV}")

    if not no_figs:
        print("\n--- Generating figures ---")
        try:
            fig_psqi_boxplot(psqi_pr)
            fig_dimension_radar(psqi_pr)
            fig_tool_distribution(psqi_pr)
            fig_dimension_boxplot(psqi_pr)
        except Exception as exc:
            print(f"  Warning: figure generation error: {exc}")

    print("\n--- Key Results Summary ---")
    global_row = stats_df[stats_df["metric"] == "psqi_global"]
    if not global_row.empty:
        r = global_row.iloc[0]
        print(
            f"  Global PSQI (AI vs Human): "
            f"median_AI={r.get('median_a', 'N/A'):.2f}, "
            f"median_Human={r.get('median_b', 'N/A'):.2f}, "
            f"p={r.get('mwu_p', 'N/A'):.4f}, "
            f"Cliff's d={r.get('cliffs_d', 'N/A'):.3f} ({r.get('cliffs_size', '')})"
        )

    sig_agents = stats_df[
        (stats_df["metric"] == "psqi_per_agent")
        & (stats_df.get("significant_bonferroni", pd.Series(False)))
    ]
    if not sig_agents.empty:
        print("  Agents with significant PSQI difference vs Human:")
        for _, row in sig_agents.iterrows():
            print(f"    {row['group_a']}: p_bonf={row.get('bonferroni_p', '?'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3 statistical analysis")
    parser.add_argument(
        "--no-figs", action="store_true",
        help="Skip figure generation",
    )
    args = parser.parse_args()
    main(no_figs=args.no_figs)
