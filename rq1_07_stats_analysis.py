#!/usr/bin/env python3
"""
RQ1 — Script 07: Análise Estatística e Figuras de Publicação
=============================================================
Executa todos os testes estatísticos do protocolo (seção 3.3.4) e
gera figuras prontas para publicação em ESE/EMSE.

Testes realizados
-----------------
  1. Mann-Whitney U + Cliff's delta — IA global vs humanos
  2. Mann-Whitney U por agente vs humanos (5×) com correção de Bonferroni
  3. Kruskal-Wallis entre os 5 agentes
  4. Post-hoc Bonferroni (10 pares) via statsmodels
  5. Análise estratificada por task_type (feat / fix / refactor)

Figuras geradas (figs/)
-----------------------
  rq1_msi_boxplot.png      — MSI por agente (side-by-side)
  rq1_msi_by_operator.png  — heatmap agente × operador de mutação
  rq1_msi_by_tasktype.png  — boxplot agrupado por task_type × agente

Outputs tabulares
-----------------
  AIDev/rq1_stats_summary.csv — todos os resultados estatísticos

Usage
-----
  python rq1_07_stats_analysis.py
  python rq1_07_stats_analysis.py --no-figs   # apenas CSV
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"
FIGS_DIR = ROOT_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

MSI_RESULTS_CSV = OUT_DIR / "rq1_msi_results.csv"
STATS_CSV = OUT_DIR / "rq1_stats_summary.csv"

sys.path.insert(0, str(ROOT_DIR))
from analysis.helper import COLOR_MAP, NAME_MAPPING, PREFER_ORDER  # noqa: E402
from cliffs_delta import cliffs_delta  # noqa: E402

AGENT_ORDER = [n for n in PREFER_ORDER if n != "Human"]
ELIGIBLE_TASK_TYPES = ["feat", "fix", "refactor", "test"]


# ---------------------------------------------------------------------------
# Helpers estatísticos
# ---------------------------------------------------------------------------

def effect_size_label(d: float) -> str:
    """Rótulo qualitativo do d de Cliff (Vargha & Delaney, 2000)."""
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.330:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def mann_whitney_with_cliff(
    group1: pd.Series,
    group2: pd.Series,
    label1: str,
    label2: str,
) -> dict:
    g1 = group1.dropna().tolist()
    g2 = group2.dropna().tolist()
    if len(g1) < 3 or len(g2) < 3:
        return {
            "test": "Mann-Whitney-U",
            "group1": label1, "group2": label2,
            "n1": len(g1), "n2": len(g2),
            "U": None, "p_value": None,
            "cliffs_d": None, "effect_size": "insufficient_data",
        }
    U, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    d, _ = cliffs_delta(g1, g2)
    return {
        "test": "Mann-Whitney-U",
        "group1": label1, "group2": label2,
        "n1": len(g1), "n2": len(g2),
        "U": round(U, 2), "p_value": round(p, 6),
        "cliffs_d": round(d, 4), "effect_size": effect_size_label(d),
    }


def kruskal_wallis_test(groups: dict[str, pd.Series]) -> dict:
    valid = {k: v.dropna().tolist() for k, v in groups.items() if len(v.dropna()) >= 3}
    if len(valid) < 2:
        return {"test": "Kruskal-Wallis", "H": None, "p_value": None, "groups": list(valid.keys())}
    H, p = stats.kruskal(*valid.values())
    return {
        "test": "Kruskal-Wallis",
        "groups": list(valid.keys()),
        "n_groups": len(valid),
        "H": round(H, 4),
        "p_value": round(p, 6),
    }


def bonferroni_pairwise(groups: dict[str, pd.Series]) -> list[dict]:
    pairs = list(combinations(groups.keys(), 2))
    raw_results = [mann_whitney_with_cliff(groups[a], groups[b], a, b) for a, b in pairs]
    n_comparisons = len(pairs)

    # Aplicar correção de Bonferroni (multiplicar p por n_comparisons, clip a 1.0)
    corrected = []
    for r in raw_results:
        r = r.copy()
        if r["p_value"] is not None:
            r["p_bonferroni"] = round(min(r["p_value"] * n_comparisons, 1.0), 6)
            r["significant_bonferroni"] = r["p_bonferroni"] < 0.05
        else:
            r["p_bonferroni"] = None
            r["significant_bonferroni"] = False
        corrected.append(r)

    return corrected


# ---------------------------------------------------------------------------
# Geração de figuras
# ---------------------------------------------------------------------------

def fig_msi_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = df[df["msi_global"].notna()].copy()
    plot_df["agent_label"] = plot_df["agent"].map(lambda a: NAME_MAPPING.get(a, a))
    plot_df["is_human_label"] = plot_df["is_human"].map({True: "Human", False: "AI"})

    # Ordenar agentes
    agent_order_labels = [NAME_MAPPING.get(a, a) for a in AGENT_ORDER]
    all_order = ["Human"] + agent_order_labels

    colors = [COLOR_MAP.get(a, "#888888") for a in all_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=plot_df,
        x="agent_label",
        y="msi_global",
        order=[o for o in all_order if o in plot_df["agent_label"].unique()],
        palette={a: COLOR_MAP.get(a, "#888888") for a in all_order},
        ax=ax,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
    )
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("MSI (%)", fontsize=12)
    ax.set_title("Mutation Score Indicator (MSI) by Agent", fontsize=14)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path}")


def fig_msi_by_operator(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Explodir msi_by_operator (JSON) em linhas
    records = []
    for _, row in df[df["msi_global"].notna()].iterrows():
        op_data = {}
        try:
            op_data = json.loads(row.get("msi_by_operator", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        agent = NAME_MAPPING.get(str(row.get("agent", "")), str(row.get("agent", "")))
        for op, msi in op_data.items():
            records.append({"agent": agent, "operator": op, "msi": msi})

    if not records:
        print("  [aviso] Sem dados msi_by_operator para gerar heatmap")
        return

    op_df = pd.DataFrame(records)
    pivot = op_df.pivot_table(values="msi", index="operator", columns="agent", aggfunc="median")

    # Ordenar colunas conforme PREFER_ORDER
    ordered_cols = [NAME_MAPPING.get(a, a) for a in AGENT_ORDER if NAME_MAPPING.get(a, a) in pivot.columns]
    if "Human" in pivot.columns:
        ordered_cols = ["Human"] + ordered_cols
    pivot = pivot.reindex(columns=ordered_cols)

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_cols) * 1.5), max(5, len(pivot) * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=0, vmax=100, ax=ax,
        linewidths=0.5, cbar_kws={"label": "Median MSI (%)"},
    )
    ax.set_title("MSI by Mutation Operator and Agent", fontsize=13)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mutation Operator")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path}")


def fig_msi_by_tasktype(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if "task_type" not in df.columns:
        print("  [aviso] Coluna task_type ausente — figura task_type ignorada")
        return

    plot_df = df[
        df["msi_global"].notna() &
        df["task_type"].isin(ELIGIBLE_TASK_TYPES)
    ].copy()
    plot_df["agent_label"] = plot_df["agent"].map(lambda a: NAME_MAPPING.get(a, a))

    if len(plot_df) == 0:
        print("  [aviso] Sem dados para figura task_type")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    agent_order_labels = [
        NAME_MAPPING.get(a, a)
        for a in AGENT_ORDER
        if NAME_MAPPING.get(a, a) in plot_df["agent_label"].unique()
    ]
    if "Human" in plot_df["agent_label"].unique():
        agent_order_labels = ["Human"] + agent_order_labels

    sns.boxplot(
        data=plot_df,
        x="agent_label",
        y="msi_global",
        hue="task_type",
        order=agent_order_labels,
        hue_order=ELIGIBLE_TASK_TYPES,
        palette="Set2",
        ax=ax,
        flierprops={"marker": ".", "markersize": 3},
    )
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("MSI (%)", fontsize=12)
    ax.set_title("MSI by Agent and Task Type", fontsize=14)
    ax.legend(title="Task Type", loc="upper right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 Script 07 — Análise Estatística")
    parser.add_argument("--no-figs", action="store_true", help="Pular geração de figuras")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 07: Análise Estatística")
    print("=" * 65)

    if not MSI_RESULTS_CSV.exists():
        print(f"[ERRO] {MSI_RESULTS_CSV} não encontrado. Execute rq1_06_aggregate_msi.py.")
        raise SystemExit(1)

    df = pd.read_csv(MSI_RESULTS_CSV)
    df["pr_id"] = df["pr_id"].astype(int)
    df["agent"] = df["agent"].map(lambda a: NAME_MAPPING.get(str(a), str(a)))

    print(f"\n  Total de PRs: {len(df):,}")
    print(f"  Com MSI:      {df['msi_global'].notna().sum():,}")

    ai_df = df[df["is_human"] == False].copy()    # noqa: E712
    human_df = df[df["is_human"] == True].copy()  # noqa: E712

    all_stats: list[dict] = []

    # ── 1. Mann-Whitney U global (IA vs Humanos) ──────────────────────────
    print("\n[1] Mann-Whitney U: IA global vs Humanos")
    r = mann_whitney_with_cliff(ai_df["msi_global"], human_df["msi_global"], "AI", "Human")
    r["description"] = "AI (all agents) vs Human"
    all_stats.append(r)
    print(f"  U={r['U']}, p={r['p_value']}, d={r['cliffs_d']} ({r['effect_size']})")

    # ── 2. Mann-Whitney por agente vs Humanos (5×) + Bonferroni ──────────
    print("\n[2] Mann-Whitney por agente vs Humanos (Bonferroni)")
    agent_names = [NAME_MAPPING.get(a, a) for a in ai_df["agent"].unique()]
    agent_rows = []
    for agent in sorted(agent_names):
        ag_df = ai_df[ai_df["agent"] == agent]
        r = mann_whitney_with_cliff(ag_df["msi_global"], human_df["msi_global"], agent, "Human")
        r["description"] = f"{agent} vs Human"
        agent_rows.append(r)

    n_agents = len(agent_rows)
    for r in agent_rows:
        if r["p_value"] is not None:
            r["p_bonferroni"] = round(min(r["p_value"] * n_agents, 1.0), 6)
            r["significant_bonferroni"] = r["p_bonferroni"] < 0.05
        all_stats.append(r)
        print(f"  {r['group1']}: U={r['U']}, p={r['p_value']}, "
              f"p_bonf={r.get('p_bonferroni', '?')}, d={r['cliffs_d']} ({r['effect_size']})")

    # ── 3. Kruskal-Wallis entre os 5 agentes ─────────────────────────────
    print("\n[3] Kruskal-Wallis entre agentes de IA")
    agent_groups = {
        agent: ai_df[ai_df["agent"] == agent]["msi_global"]
        for agent in sorted(ai_df["agent"].unique())
    }
    kw = kruskal_wallis_test(agent_groups)
    kw["description"] = "Kruskal-Wallis across AI agents"
    all_stats.append(kw)
    print(f"  H={kw.get('H')}, p={kw.get('p_value')}, groups={kw.get('n_groups')}")

    # ── 4. Post-hoc Bonferroni (pairwise entre agentes) ───────────────────
    print("\n[4] Post-hoc pairwise (Bonferroni) entre agentes")
    pairwise = bonferroni_pairwise(agent_groups)
    for r in pairwise:
        r["description"] = f"post-hoc: {r['group1']} vs {r['group2']}"
        all_stats.append(r)
        sig = "✓" if r.get("significant_bonferroni") else " "
        print(f"  {sig} {r['group1']} vs {r['group2']}: "
              f"p_bonf={r.get('p_bonferroni', '?')}, d={r['cliffs_d']} ({r['effect_size']})")

    # ── 5. Análise estratificada por task_type ────────────────────────────
    if "task_type" in df.columns:
        print("\n[5] MSI mediano por agente × task_type")
        task_df = df[df["task_type"].isin(ELIGIBLE_TASK_TYPES)]
        pivot = task_df.pivot_table(
            values="msi_global",
            index="agent",
            columns="task_type",
            aggfunc="median",
        ).round(1)
        print(pivot.to_string())

        # Adicionar ao stats summary
        for task_type in ELIGIBLE_TASK_TYPES:
            for agent, group in df[df["task_type"] == task_type].groupby("agent"):
                h_task = human_df[human_df.get("task_type", pd.Series(dtype=str)) == task_type]["msi_global"] \
                    if "task_type" in human_df.columns else human_df["msi_global"]
                r = mann_whitney_with_cliff(group["msi_global"], h_task, str(agent), "Human")
                r["description"] = f"{agent} vs Human | task={task_type}"
                r["task_type"] = task_type
                all_stats.append(r)

    # ── Salvar stats summary ───────────────────────────────────────────────
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"\n  Stats summary salvo: {STATS_CSV} ({len(stats_df)} linhas)")

    # ── Figuras ────────────────────────────────────────────────────────────
    if not args.no_figs:
        try:
            import matplotlib  # noqa: PLC0415
            matplotlib.use("Agg")
            print("\n[Figuras]")
            fig_msi_boxplot(df, FIGS_DIR / "rq1_msi_boxplot.png")
            fig_msi_by_operator(df, FIGS_DIR / "rq1_msi_by_operator.png")
            fig_msi_by_tasktype(df, FIGS_DIR / "rq1_msi_by_tasktype.png")
        except ImportError as exc:
            print(f"  [aviso] matplotlib/seaborn não disponível: {exc}")

    print(f"\n{'='*65}")
    print(" CONCLUÍDO — Pipeline RQ1")
    print(f"{'='*65}")
    print(f"  Resultados: {MSI_RESULTS_CSV}")
    print(f"  Stats:      {STATS_CSV}")
    print(f"  Figuras:    {FIGS_DIR}/rq1_*.png")


if __name__ == "__main__":
    main()
