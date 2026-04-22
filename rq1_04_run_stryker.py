#!/usr/bin/env python3
"""
RQ1 — Script 04: Execução do Stryker e Coleta de MSI (JS/TS)
=============================================================
Executa `npx stryker run` em cada worktree preparado pelo Script 03,
parseia stryker-report.json e salva MSI por PR.

Métricas coletadas (protocolo seção 3.3.4)
------------------------------------------
  msi_global      — killed / total * 100
  msi_by_operator — MSI por operador de mutação (JSON)
  killed, survived, timeout_mutants, no_coverage
  duration_seconds

Checkpoint
----------
  AIDev/stryker_results.csv — pular pr_ids com status 'ok' ou 'partial'

Usage
-----
  python rq1_04_run_stryker.py
  python rq1_04_run_stryker.py --agent "Claude Code"
  python rq1_04_run_stryker.py --limit 5
  python rq1_04_run_stryker.py --dry-run   # valida setup sem executar mutação
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"

SETUP_LOG_CSV = OUT_DIR / "stryker_setup_log.csv"
ELIGIBLE_CSV = OUT_DIR / "rq1_eligible.csv"
RESULTS_CSV = OUT_DIR / "stryker_results.csv"

STRYKER_TIMEOUT = 3_600  # 1h por PR (cobre ~120 mutantes a 30s cada)

load_dotenv(ROOT_DIR / ".env")

RESULT_COLS = [
    "pr_id", "agent", "repo_full_name",
    "total_mutants", "killed", "survived", "timeout_mutants", "no_coverage",
    "msi_global", "msi_by_operator",
    "status", "error_msg", "duration_seconds",
]


# ---------------------------------------------------------------------------
# Parsing do relatório Stryker
# ---------------------------------------------------------------------------

def _parse_stryker_report(report_path: Path) -> dict:
    """
    Parseia stryker-report.json e retorna métricas de MSI.
    Schema Stryker v8: {"mutants": [{"id", "status", "mutatorName", ...}]}
    """
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    mutants = report.get("mutants", [])
    if not mutants:
        return {
            "total_mutants": 0, "killed": 0, "survived": 0,
            "timeout_mutants": 0, "no_coverage": 0,
            "msi_global": None, "msi_by_operator": "{}",
        }

    total = len(mutants)
    killed = sum(1 for m in mutants if m.get("status") == "Killed")
    survived = sum(1 for m in mutants if m.get("status") == "Survived")
    timeout_count = sum(1 for m in mutants if m.get("status") == "Timeout")
    no_coverage = sum(1 for m in mutants if m.get("status") == "NoCoverage")

    msi_global = round(killed / total * 100, 2) if total > 0 else None

    # MSI por operador de mutação
    by_op: dict[str, dict[str, int]] = defaultdict(lambda: {"killed": 0, "total": 0})
    for m in mutants:
        op = m.get("mutatorName", "Unknown")
        by_op[op]["total"] += 1
        if m.get("status") == "Killed":
            by_op[op]["killed"] += 1

    msi_by_op = {
        op: round(v["killed"] / v["total"] * 100, 2)
        for op, v in by_op.items()
        if v["total"] > 0
    }

    return {
        "total_mutants": total,
        "killed": killed,
        "survived": survived,
        "timeout_mutants": timeout_count,
        "no_coverage": no_coverage,
        "msi_global": msi_global,
        "msi_by_operator": json.dumps(msi_by_op),
    }


# ---------------------------------------------------------------------------
# Execução de um único worktree
# ---------------------------------------------------------------------------

def run_stryker(
    pr_id: int,
    agent: str,
    repo_full_name: str,
    worktree_path: str,
    dry_run: bool = False,
) -> dict:
    wt = Path(worktree_path)
    report_path = wt / "stryker-report.json"

    base = {
        "pr_id": pr_id,
        "agent": agent,
        "repo_full_name": repo_full_name,
        "total_mutants": None,
        "killed": None,
        "survived": None,
        "timeout_mutants": None,
        "no_coverage": None,
        "msi_global": None,
        "msi_by_operator": "{}",
        "status": "",
        "error_msg": "",
        "duration_seconds": 0,
    }

    if not wt.exists():
        return {**base, "status": "worktree_missing", "error_msg": str(wt)}

    if not (wt / "stryker.config.json").exists():
        return {**base, "status": "config_missing", "error_msg": "stryker.config.json não encontrado"}

    if dry_run:
        return {**base, "status": "dry_run_ok"}

    # Remover relatório anterior
    if report_path.exists():
        report_path.unlink()

    t0 = time.time()
    print(f"    npx stryker run ... ", end="", flush=True)

    try:
        result = subprocess.run(
            ["npx", "stryker", "run", "--config", "stryker.config.json"],
            cwd=str(wt),
            capture_output=True,
            text=True,
            timeout=STRYKER_TIMEOUT,
        )
        duration = round(time.time() - t0, 1)
        exit_ok = result.returncode == 0

    except subprocess.TimeoutExpired:
        duration = STRYKER_TIMEOUT
        print(f"TIMEOUT ({STRYKER_TIMEOUT}s)")
        # Verificar se relatório parcial foi gerado
        if report_path.exists():
            try:
                metrics = _parse_stryker_report(report_path)
                return {**base, **metrics, "status": "timeout_global", "duration_seconds": duration}
            except Exception:
                pass
        return {**base, "status": "timeout_no_report", "duration_seconds": duration}

    except Exception as exc:
        return {**base, "status": "run_error", "error_msg": str(exc)[:300]}

    print(f"{'OK' if exit_ok else 'exit=' + str(result.returncode)} ({duration}s)")

    # Tentar parsear o relatório mesmo com exit code != 0
    if not report_path.exists():
        err_snippet = (result.stderr or result.stdout)[:400]
        return {
            **base,
            "status": "no_report",
            "error_msg": err_snippet,
            "duration_seconds": duration,
        }

    try:
        metrics = _parse_stryker_report(report_path)
    except Exception as exc:
        return {**base, "status": "parse_error", "error_msg": str(exc), "duration_seconds": duration}

    status = "ok" if exit_ok else "partial"
    return {**base, **metrics, "status": status, "duration_seconds": duration}


# ---------------------------------------------------------------------------
# Checkpoint logger
# ---------------------------------------------------------------------------

class ResultLogger:
    DONE_STATUSES = {"ok", "partial", "dry_run_ok"}

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._done: set[int] = set()
        self._fh = None
        self._writer = None

    def load(self) -> None:
        if self._path.exists():
            df = pd.read_csv(self._path)
            done = df[df["status"].isin(self.DONE_STATUSES)]
            self._done = set(done["pr_id"].astype(int))
            print(f"  [checkpoint] {len(self._done)} PRs já processadas")
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        self._fh = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=RESULT_COLS)
        if write_header:
            self._writer.writeheader()

    def already_done(self, pr_id: int) -> bool:
        return pr_id in self._done

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in RESULT_COLS})
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 Script 04 — Execução Stryker")
    parser.add_argument("--agent", help="Filtrar por agente")
    parser.add_argument("--limit", type=int, help="Limitar N PRs")
    parser.add_argument("--dry-run", action="store_true", help="Validar setup sem executar mutação")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 04: Execução Stryker (JS/TS)")
    print("=" * 65)

    for required in [SETUP_LOG_CSV, ELIGIBLE_CSV]:
        if not required.exists():
            print(f"[ERRO] {required} não encontrado.")
            raise SystemExit(1)

    setup_log = pd.read_csv(SETUP_LOG_CSV)
    ready = setup_log[setup_log["status"] == "ready"].copy()
    print(f"\n  Worktrees com status=ready: {len(ready):,}")

    eligible = pd.read_csv(ELIGIBLE_CSV)[["pr_id", "agent", "repo_full_name"]].copy()
    eligible["pr_id"] = eligible["pr_id"].astype(int)
    ready["pr_id"] = ready["pr_id"].astype(int)

    work_items = ready.merge(eligible, on="pr_id", how="left")

    if args.agent:
        work_items = work_items[work_items["agent"] == args.agent]
        print(f"  Filtrado para agente: {args.agent} ({len(work_items)})")

    if args.limit:
        work_items = work_items.head(args.limit)
        print(f"  Limitado a {args.limit} PRs")

    logger = ResultLogger(RESULTS_CSV)
    logger.load()

    total = len(work_items)
    ok_count = partial_count = err_count = 0

    for i, (_, row) in enumerate(work_items.iterrows(), 1):
        pr_id = int(row["pr_id"])
        if logger.already_done(pr_id):
            continue

        print(f"\n[{i}/{total}] {row.get('repo_full_name', '?')} PR#{pr_id} [{row.get('agent', '?')}]")

        result = run_stryker(
            pr_id=pr_id,
            agent=str(row.get("agent", "")),
            repo_full_name=str(row.get("repo_full_name", "")),
            worktree_path=str(row["worktree_path"]),
            dry_run=args.dry_run,
        )
        logger.write(result)

        s = result["status"]
        if s == "ok":
            ok_count += 1
            print(f"  MSI={result['msi_global']}% | mutants={result['total_mutants']}")
        elif s == "partial":
            partial_count += 1
            print(f"  [partial] MSI={result['msi_global']}%")
        elif s == "dry_run_ok":
            ok_count += 1
        else:
            err_count += 1
            print(f"  [ERRO] {s}: {result.get('error_msg', '')[:120]}")

    logger.close()

    print(f"\n{'='*65}")
    print(" RESULTADO")
    print(f"{'='*65}")
    print(f"  OK (completo):  {ok_count}")
    print(f"  Parcial:        {partial_count}")
    print(f"  Erros/timeout:  {err_count}")
    print(f"  Resultados em:  {RESULTS_CSV}")

    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        done = df[df["status"].isin(["ok", "partial"])]
        if len(done) > 0:
            print(f"\n  MSI global — mediana: {done['msi_global'].median():.1f}%")
            print(f"  MSI global — média:   {done['msi_global'].mean():.1f}%")

    print("\nPróximo passo: python rq1_05_run_mutmut.py")


if __name__ == "__main__":
    main()
