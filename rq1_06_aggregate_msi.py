#!/usr/bin/env python3
"""
RQ1 — Script 06: Agregação de MSI + Métricas de Qualidade de Teste
===================================================================
Unifica stryker_results.csv + mutmut_results.csv, junta metadados de PR
(agente, task_type, linguagem), calcula complexidade ciclomática via lizard,
conta assertions e linhas de teste. Produz tabela mestra e correlações.

Métricas calculadas (protocolo seção 3.3.4)
------------------------------------------
  msi_global          — já vem dos scripts anteriores
  cyclomatic_complexity — média por PR via lizard (arquivos de teste)
  assertion_count      — contagem via regex nos arquivos de teste
  test_loc             — linhas de código nos arquivos de teste

Correlações de Spearman (por agente + global)
----------------------------------------------
  msi_global vs assertion_count
  msi_global vs cyclomatic_complexity
  msi_global vs test_loc

Outputs
-------
  AIDev/rq1_msi_results.csv    — tabela mestra de análise
  AIDev/rq1_correlations.csv   — correlações de Spearman (tidy format)

Usage
-----
  python rq1_06_aggregate_msi.py
  python rq1_06_aggregate_msi.py --no-lizard   # pular cálculo de CC (se lizard não instalado)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"
REPOS_DIR = ROOT_DIR / "repos"

load_dotenv(ROOT_DIR / ".env")

STRYKER_RESULTS = OUT_DIR / "stryker_results.csv"
MUTMUT_RESULTS = OUT_DIR / "mutmut_results.csv"
ELIGIBLE_CSV = OUT_DIR / "rq1_eligible.csv"
CLONE_LOG_CSV = OUT_DIR / "clone_log.csv"

MSI_RESULTS_CSV = OUT_DIR / "rq1_msi_results.csv"
CORRELATIONS_CSV = OUT_DIR / "rq1_correlations.csv"

HF_BASE = "hf://datasets/hao-li/AIDev"

sys.path.insert(0, str(ROOT_DIR))
from analysis.helper import NAME_MAPPING  # noqa: E402

# Regex para contagem de assertions (JS/TS + Python)
ASSERTION_RE = re.compile(
    r"\b(expect|assert|toBe|toEqual|toStrictEqual|toContain|toHaveBeenCalled"
    r"|assertEqual|assertIn|assertRaises|assertIsNone|assertIsNotNone"
    r"|assertTrue|assertFalse|assertAlmostEqual|verify|check)\s*[\(\.]",
    re.IGNORECASE,
)

DONE_STATUSES = {"ok", "partial"}


# ---------------------------------------------------------------------------
# Carregar e unificar resultados
# ---------------------------------------------------------------------------

def load_mutation_results() -> pd.DataFrame:
    dfs = []
    for path in [STRYKER_RESULTS, MUTMUT_RESULTS]:
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["status"].isin(DONE_STATUSES)].copy()
            print(f"  {path.name}: {len(df)} linhas válidas")
            dfs.append(df)
        else:
            print(f"  [aviso] {path.name} não encontrado, pulando")

    if not dfs:
        print("[ERRO] Nenhum resultado de mutação encontrado.")
        raise SystemExit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined["pr_id"] = combined["pr_id"].astype(int)
    print(f"  Total combinado: {len(combined)} PRs")
    return combined


# ---------------------------------------------------------------------------
# Enriquecer com metadados de PR
# ---------------------------------------------------------------------------

def enrich_with_pr_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona task_type, is_human, linguagem a partir de rq1_eligible.csv."""
    eligible = pd.read_csv(ELIGIBLE_CSV)
    eligible["pr_id"] = eligible["pr_id"].astype(int)

    cols = ["pr_id", "agent", "is_human", "task_type", "mutation_tool"]
    eligible_sub = eligible[[c for c in cols if c in eligible.columns]].copy()

    merged = df.merge(eligible_sub, on="pr_id", how="left", suffixes=("", "_eligible"))

    # Preferir agent do eligible (mais confiável)
    if "agent_eligible" in merged.columns:
        merged["agent"] = merged["agent_eligible"].fillna(merged["agent"])
        merged.drop(columns=["agent_eligible"], inplace=True)

    # Normalizar nomes de agentes
    merged["agent"] = merged["agent"].map(lambda a: NAME_MAPPING.get(a, a))

    missing_task = merged["task_type"].isna().sum() if "task_type" in merged.columns else len(merged)
    if missing_task > 0:
        print(f"  [aviso] {missing_task} PRs sem task_type")

    return merged


# ---------------------------------------------------------------------------
# Complexidade ciclomática via lizard
# ---------------------------------------------------------------------------

def compute_cyclomatic_complexity(
    df: pd.DataFrame,
    clone_log: pd.DataFrame,
    use_lizard: bool,
) -> pd.Series:
    """
    Para cada PR, calcula a média de complexidade ciclomática dos arquivos de teste.
    Requer que o worktree esteja clonado (usa clone_log para localizar).
    Retorna Series indexada por pr_id.
    """
    if not use_lizard:
        return pd.Series(dtype=float)

    try:
        import lizard  # noqa: PLC0415
    except ImportError:
        print("  [aviso] lizard não instalado — CC será None. Execute: pip install lizard")
        return pd.Series(dtype=float)

    wt_map = clone_log[clone_log["status"] == "ok"].set_index("pr_id")["worktree_path"].to_dict()
    eligible = pd.read_csv(ELIGIBLE_CSV)[["pr_id", "test_files"]].copy()
    eligible["pr_id"] = eligible["pr_id"].astype(int)
    tf_map = eligible.set_index("pr_id")["test_files"].to_dict()

    cc_values: dict[int, float] = {}

    for pr_id in df["pr_id"].unique():
        worktree_path = wt_map.get(int(pr_id))
        test_files_json = tf_map.get(int(pr_id), "[]")

        if not worktree_path:
            continue

        wt = Path(worktree_path)
        test_files: list[str] = json.loads(test_files_json) if test_files_json else []
        cc_list: list[float] = []

        for tf in test_files:
            full_path = wt / tf
            if not full_path.exists():
                continue
            try:
                result = lizard.analyze_file(str(full_path))
                if result and result.function_list:
                    for fn in result.function_list:
                        cc_list.append(fn.cyclomatic_complexity)
            except Exception:
                continue

        if cc_list:
            cc_values[int(pr_id)] = round(sum(cc_list) / len(cc_list), 2)

    return pd.Series(cc_values, name="cyclomatic_complexity")


# ---------------------------------------------------------------------------
# Contagem de assertions e LOC
# ---------------------------------------------------------------------------

def compute_test_metrics(df: pd.DataFrame, clone_log: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula assertion_count e test_loc para cada PR.
    Lê os arquivos de teste diretamente dos worktrees clonados.
    """
    wt_map = clone_log[clone_log["status"] == "ok"].set_index("pr_id")["worktree_path"].to_dict()
    eligible = pd.read_csv(ELIGIBLE_CSV)[["pr_id", "test_files"]].copy()
    eligible["pr_id"] = eligible["pr_id"].astype(int)
    tf_map = eligible.set_index("pr_id")["test_files"].to_dict()

    records: list[dict] = []

    for pr_id in df["pr_id"].unique():
        worktree_path = wt_map.get(int(pr_id))
        test_files_json = tf_map.get(int(pr_id), "[]")
        test_files: list[str] = json.loads(test_files_json) if test_files_json else []

        total_assertions = 0
        total_loc = 0

        if worktree_path:
            wt = Path(worktree_path)
            for tf in test_files:
                full_path = wt / tf
                if not full_path.exists():
                    continue
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    total_assertions += len(ASSERTION_RE.findall(content))
                    total_loc += len(content.splitlines())
                except Exception:
                    continue

        records.append({
            "pr_id": int(pr_id),
            "assertion_count": total_assertions,
            "test_loc": total_loc,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Correlações de Spearman
# ---------------------------------------------------------------------------

def compute_spearman_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula correlações de Spearman entre msi_global e cada métrica,
    estratificado por agente e também globalmente.
    """
    metrics = ["assertion_count", "test_loc", "cyclomatic_complexity"]
    records: list[dict] = []

    def _corr(group: pd.DataFrame, label: str) -> None:
        valid = group.dropna(subset=["msi_global"])
        for metric in metrics:
            if metric not in valid.columns:
                continue
            sub = valid.dropna(subset=[metric])
            if len(sub) < 5:
                continue
            r, p = stats.spearmanr(sub["msi_global"], sub[metric])
            records.append({
                "scope": label,
                "metric": metric,
                "n": len(sub),
                "spearman_r": round(r, 4),
                "p_value": round(p, 6),
                "significant": p < 0.05,
            })

    # Global
    _corr(df, "global")

    # Por agente
    for agent, group in df.groupby("agent"):
        _corr(group, str(agent))

    # Por is_human
    for is_human, group in df.groupby("is_human"):
        label = "Human" if is_human else "AI"
        _corr(group, label)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 Script 06 — Agregação de MSI")
    parser.add_argument("--no-lizard", action="store_true", help="Pular cálculo de CC")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 06: Agregação de MSI")
    print("=" * 65)

    # ── 1. Carregar resultados de mutação ──────────────────────────────────
    print("\n[1/5] Carregando resultados de mutação ...")
    df = load_mutation_results()

    # ── 2. Enriquecer com metadados de PR ──────────────────────────────────
    print("\n[2/5] Enriquecendo com metadados (task_type, agente) ...")
    if ELIGIBLE_CSV.exists():
        df = enrich_with_pr_metadata(df)
    else:
        print("  [aviso] rq1_eligible.csv não encontrado — metadados incompletos")

    # Carregar clone_log para localizar worktrees
    clone_log = pd.read_csv(CLONE_LOG_CSV) if CLONE_LOG_CSV.exists() else pd.DataFrame()
    if not clone_log.empty:
        clone_log["pr_id"] = clone_log["pr_id"].astype(int)

    # ── 3. Complexidade ciclomática ────────────────────────────────────────
    print("\n[3/5] Calculando complexidade ciclomática (lizard) ...")
    cc_series = compute_cyclomatic_complexity(df, clone_log, use_lizard=not args.no_lizard)
    if not cc_series.empty:
        df = df.merge(cc_series.reset_index().rename(columns={"index": "pr_id"}),
                      on="pr_id", how="left")
        print(f"  CC calculado para {cc_series.notna().sum()} PRs")
    else:
        df["cyclomatic_complexity"] = None

    # ── 4. Assertions e LOC ────────────────────────────────────────────────
    print("\n[4/5] Calculando assertion_count e test_loc ...")
    if not clone_log.empty and ELIGIBLE_CSV.exists():
        test_metrics = compute_test_metrics(df, clone_log)
        df = df.merge(test_metrics, on="pr_id", how="left")
        has_assertions = (df["assertion_count"] > 0).sum()
        print(f"  assertion_count > 0 em {has_assertions} PRs")
    else:
        df["assertion_count"] = None
        df["test_loc"] = None
        print("  [aviso] clone_log não disponível — métricas de arquivo serão None")

    # ── 5. Correlações de Spearman ─────────────────────────────────────────
    print("\n[5/5] Calculando correlações de Spearman ...")
    corr_df = compute_spearman_correlations(df)
    corr_df.to_csv(CORRELATIONS_CSV, index=False)
    print(f"  Salvo: {CORRELATIONS_CSV}")
    if len(corr_df) > 0:
        print("\n  Correlações globais (msi_global vs métrica):")
        global_corr = corr_df[corr_df["scope"] == "global"]
        print(global_corr[["metric", "spearman_r", "p_value", "significant", "n"]].to_string(index=False))

    # ── Salvar tabela mestra ───────────────────────────────────────────────
    df.to_csv(MSI_RESULTS_CSV, index=False)
    print(f"\n  Tabela mestra salva: {MSI_RESULTS_CSV}")

    # ── Resumo ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(" RESUMO — rq1_msi_results.csv")
    print(f"{'='*65}")
    print(f"  Total de PRs: {len(df):,}")
    print(f"  Com MSI calculado: {df['msi_global'].notna().sum():,}")
    print(f"\n  Mediana MSI por agente:")
    if "agent" in df.columns:
        print(df.groupby("agent")["msi_global"].median().round(1).to_string())
    print(f"\n  Mediana MSI — IA vs Humanos:")
    if "is_human" in df.columns:
        print(df.groupby("is_human")["msi_global"].median().round(1).to_string())

    print("\nPróximo passo: python rq1_07_stats_analysis.py")


if __name__ == "__main__":
    main()
