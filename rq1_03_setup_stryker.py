#!/usr/bin/env python3
"""
RQ1 — Script 03: Preparação do Ambiente Stryker (JS/TS)
========================================================
Para cada worktree JS/TS elegível já clonado, instala dependências npm
e gera stryker.config.json customizado para aquela PR específica.

NÃO executa mutação — apenas prepara o ambiente.
A execução está no Script 04 (rq1_04_run_stryker.py).

Pré-requisitos
--------------
  - Node >= 18 e npm no PATH
  - rq1_eligible.csv gerado (Script 02)
  - Worktrees clonados em repos/ (rq1_clone_repos.py --csv AIDev/rq1_eligible.csv)

Checkpoint
----------
  AIDev/stryker_setup_log.csv
    colunas: pr_id, repo_full_name, worktree_path, status, error_msg, config_path

Usage
-----
  python rq1_03_setup_stryker.py
  python rq1_03_setup_stryker.py --agent "Claude Code"
  python rq1_03_setup_stryker.py --limit 10   # teste rápido
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"
REPOS_DIR = ROOT_DIR / "repos"

ELIGIBLE_CSV = OUT_DIR / "rq1_eligible.csv"
CLONE_LOG_CSV = OUT_DIR / "clone_log.csv"
SETUP_LOG_CSV = OUT_DIR / "stryker_setup_log.csv"

NPM_INSTALL_TIMEOUT = 180   # segundos
STRYKER_INSTALL_TIMEOUT = 120

load_dotenv(ROOT_DIR / ".env")

sys.path.insert(0, str(ROOT_DIR))
from rq1_clone_repos import repo_to_dirname  # noqa: E402


# ---------------------------------------------------------------------------
# Inferir test runner a partir de package.json
# ---------------------------------------------------------------------------

KNOWN_RUNNERS = ["vitest", "jest", "mocha", "jasmine", "karma"]

def detect_test_runner(worktree: Path) -> str:
    pkg_path = worktree / "package.json"
    if not pkg_path.exists():
        return "vitest"  # default — a maioria dos repos filtrados usa vitest

    try:
        pkg = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "vitest"

    all_deps = {
        **pkg.get("dependencies", {}),
        **pkg.get("devDependencies", {}),
    }
    for runner in KNOWN_RUNNERS:
        if runner in all_deps:
            return runner

    # Verificar scripts npm
    scripts = pkg.get("scripts", {})
    test_script = scripts.get("test", "")
    for runner in KNOWN_RUNNERS:
        if runner in test_script:
            return runner

    return "vitest"


# ---------------------------------------------------------------------------
# Inferir globs de código fonte a partir dos arquivos de teste
# ---------------------------------------------------------------------------

def infer_source_globs(test_files: list[str]) -> tuple[list[str], list[str]]:
    """
    Retorna (mutate_globs, exclude_globs) para o stryker.config.json.
    Baseado nos caminhos dos arquivos de teste, infere os diretórios fonte.
    """
    source_dirs: set[str] = set()

    for tf in test_files:
        parts = Path(tf).parts
        # Inferir diretório fonte: src/, lib/, app/, etc.
        for i, part in enumerate(parts):
            if part.lower() in ("src", "lib", "app", "core", "packages"):
                source_dirs.add(part)
                break
        else:
            # Se não encontrar diretório padrão, usar o primeiro componente
            if len(parts) > 1:
                source_dirs.add(parts[0])

    if not source_dirs:
        source_dirs = {"src", "lib"}

    mutate = [f"{d}/**/*.{{ts,tsx,js,jsx}}" for d in sorted(source_dirs)]
    exclude = [
        "**/*.test.ts", "**/*.test.tsx", "**/*.test.js",
        "**/*.spec.ts", "**/*.spec.tsx", "**/*.spec.js",
        "**/*.d.ts",
        "node_modules/**",
    ]
    return mutate, exclude


# ---------------------------------------------------------------------------
# Geração de stryker.config.json
# ---------------------------------------------------------------------------

def generate_stryker_config(
    test_files: list[str],
    test_runner: str,
    output_file: Path,
) -> None:
    mutate_globs, exclude_globs = infer_source_globs(test_files)

    config = {
        "mutate": mutate_globs + [f"!{e}" for e in exclude_globs],
        "testRunner": test_runner,
        "timeoutMS": 30_000,
        "timeoutFactor": 2.0,
        "mutationOperators": [
            "ArithmeticOperator",
            "BooleanLiteral",
            "ConditionalExpression",
            "EqualityOperator",
            "LogicalOperator",
            "StringLiteral",
        ],
        "reporters": ["json", "clear-text"],
        "jsonReporter": {
            "fileName": "stryker-report.json",
        },
        "coverageAnalysis": "perTest",
        "concurrency": max(1, (os.cpu_count() or 2) - 1),
        "tempDirName": ".stryker-tmp",
        "logLevel": "warn",
    }
    output_file.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Setup de um único worktree
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[bool, str]:
    """Executa comando e retorna (sucesso, stderr_ou_saída)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, (result.stderr or result.stdout)[:500]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout após {timeout}s"
    except Exception as exc:
        return False, str(exc)


def setup_worktree(
    pr_id: int,
    repo_full_name: str,
    worktree_path: str,
    test_files_json: str,
) -> dict:
    wt = Path(worktree_path)
    test_files: list[str] = json.loads(test_files_json) if test_files_json else []
    config_path = str(wt / "stryker.config.json")

    base = {
        "pr_id": pr_id,
        "repo_full_name": repo_full_name,
        "worktree_path": worktree_path,
        "status": "",
        "error_msg": "",
        "config_path": config_path,
    }

    if not wt.exists():
        return {**base, "status": "worktree_missing", "error_msg": f"não existe: {wt}"}

    pkg_json = wt / "package.json"
    if not pkg_json.exists():
        return {**base, "status": "no_package_json", "error_msg": "package.json não encontrado"}

    # 1. npm install
    print(f"    npm install ...", end=" ", flush=True)
    ok, err = _run(["npm", "install", "--ignore-scripts", "--prefer-offline"], wt, NPM_INSTALL_TIMEOUT)
    if not ok:
        # Tentar sem --prefer-offline
        ok, err = _run(["npm", "install", "--ignore-scripts"], wt, NPM_INSTALL_TIMEOUT)
    if not ok:
        print("ERRO")
        return {**base, "status": "npm_install_error", "error_msg": err}
    print("OK")

    # 2. Verificar/instalar Stryker
    print(f"    verificando Stryker ...", end=" ", flush=True)
    ok, _ = _run(["npx", "--yes", "stryker", "--version"], wt, 60)
    if not ok:
        print(f"    instalando @stryker-mutator/core ...", end=" ", flush=True)
        test_runner = detect_test_runner(wt)
        stryker_runner_pkg = f"@stryker-mutator/{test_runner}-runner"
        ok, err = _run(
            ["npm", "install", "--save-dev",
             "@stryker-mutator/core", stryker_runner_pkg],
            wt, STRYKER_INSTALL_TIMEOUT,
        )
        if not ok:
            print("ERRO")
            return {**base, "status": "stryker_install_error", "error_msg": err}
    print("OK")

    # 3. Gerar stryker.config.json
    test_runner = detect_test_runner(wt)
    config_out = wt / "stryker.config.json"
    generate_stryker_config(test_files, test_runner, config_out)

    return {**base, "status": "ready", "error_msg": "", "config_path": str(config_out)}


# ---------------------------------------------------------------------------
# Logger com checkpoint
# ---------------------------------------------------------------------------

class SetupLogger:
    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._done: set[int] = set()
        self._fh = None
        self._writer = None

    def load(self) -> None:
        if self._path.exists():
            df = pd.read_csv(self._path)
            done_mask = df["status"].isin(["ready", "no_package_json"])
            self._done = set(df[done_mask]["pr_id"].astype(int))
            print(f"  [checkpoint] {len(self._done)} worktrees já configurados")
        self._fh = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=["pr_id", "repo_full_name", "worktree_path", "status", "error_msg", "config_path"],
        )
        if not self._path.exists() or self._path.stat().st_size == 0:
            self._writer.writeheader()

    def already_done(self, pr_id: int) -> bool:
        return pr_id in self._done

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 Script 03 — Setup Stryker")
    parser.add_argument("--agent", help="Filtrar por agente")
    parser.add_argument("--limit", type=int, help="Limitar N worktrees (teste)")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 03: Setup Stryker (JS/TS)")
    print("=" * 65)

    for required in [ELIGIBLE_CSV, CLONE_LOG_CSV]:
        if not required.exists():
            print(f"[ERRO] {required} não encontrado.")
            raise SystemExit(1)

    eligible = pd.read_csv(ELIGIBLE_CSV)
    clone_log = pd.read_csv(CLONE_LOG_CSV)

    # Apenas worktrees clonados com sucesso
    cloned = clone_log[clone_log["status"] == "ok"][["pr_id", "worktree_path"]].copy()
    cloned["pr_id"] = cloned["pr_id"].astype(int)

    # Apenas PRs JS/TS (mutation_tool == 'stryker')
    js_eligible = eligible[eligible.get("mutation_tool", "stryker") == "stryker"].copy()
    if "mutation_tool" not in eligible.columns:
        js_eligible = eligible.copy()

    js_eligible["pr_id"] = js_eligible["pr_id"].astype(int)

    # Join para obter worktree_path
    work_items = js_eligible.merge(cloned, on="pr_id", how="inner")
    print(f"\n  Worktrees JS/TS prontos para setup: {len(work_items):,}")

    if args.agent:
        work_items = work_items[work_items["agent"] == args.agent]
        print(f"  Filtrado para agente: {args.agent} ({len(work_items)})")

    if args.limit:
        work_items = work_items.head(args.limit)
        print(f"  Limitado a {args.limit} itens")

    logger = SetupLogger(SETUP_LOG_CSV)
    logger.load()

    total = len(work_items)
    ok_count = 0
    err_count = 0

    for i, (_, row) in enumerate(work_items.iterrows(), 1):
        pr_id = int(row["pr_id"])
        if logger.already_done(pr_id):
            continue

        print(f"\n  [{i}/{total}] {row['repo_full_name']} PR#{pr_id}")
        result = setup_worktree(
            pr_id=pr_id,
            repo_full_name=str(row["repo_full_name"]),
            worktree_path=str(row["worktree_path"]),
            test_files_json=str(row.get("test_files", "[]")),
        )
        logger.write(result)

        if result["status"] == "ready":
            ok_count += 1
        else:
            err_count += 1
            print(f"    [ERRO] status={result['status']}: {result['error_msg']}")

    logger.close()

    print(f"\n{'='*65}")
    print(" RESULTADO")
    print(f"{'='*65}")
    print(f"  Configurados com sucesso: {ok_count}")
    print(f"  Com erros:                {err_count}")
    print(f"  Log: {SETUP_LOG_CSV}")
    print("\nPróximo passo: python rq1_04_run_stryker.py")


if __name__ == "__main__":
    main()
