#!/usr/bin/env python3
"""
RQ1 — Script 05: Execução do mutmut e Coleta de MSI (Python)
=============================================================
Para cada worktree Python elegível (clonado via rq1_clone_repos.py),
cria venv isolado, instala dependências, executa mutmut 2.x,
parseia resultados e salva MSI por PR.

Schema de output (mesmo de stryker_results.csv + mutation_tool_used)
--------------------------------------------------------------------
  pr_id, agent, repo_full_name,
  total_mutants, killed, survived, timeout_mutants, no_coverage,
  msi_global, msi_by_operator (JSON),
  status, error_msg, duration_seconds, mutation_tool_used

Checkpoint
----------
  AIDev/mutmut_results.csv — pular pr_ids com status 'ok' ou 'partial'

Usage
-----
  python rq1_05_run_mutmut.py
  python rq1_05_run_mutmut.py --limit 5
  python rq1_05_run_mutmut.py --keep-venv   # não remove venv após execução
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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

ELIGIBLE_CSV = OUT_DIR / "rq1_eligible.csv"
CLONE_LOG_CSV = OUT_DIR / "clone_log.csv"
RESULTS_CSV = OUT_DIR / "mutmut_results.csv"

VENV_NAME = ".venv_mutmut"
PIP_TIMEOUT = 180
MUTMUT_TIMEOUT = 3_600   # 1h por PR
MUTANT_TIMEOUT_THRESHOLD = 300  # s por mutante para ativar Cosmic Ray

load_dotenv(ROOT_DIR / ".env")

RESULT_COLS = [
    "pr_id", "agent", "repo_full_name",
    "total_mutants", "killed", "survived", "timeout_mutants", "no_coverage",
    "msi_global", "msi_by_operator",
    "status", "error_msg", "duration_seconds", "mutation_tool_used",
]

# Regex para classificar operadores de mutação a partir do diff do mutante
OP_PATTERNS = [
    ("ArithmeticOperator", re.compile(r"[+\-*/] ?→ ?[+\-*/]|(?<![=!<>])[+\-][^=]")),
    ("ComparisonOperator", re.compile(r"==|!=|>=|<=|>|<")),
    ("BooleanLiteral",     re.compile(r"\bTrue\b|\bFalse\b")),
    ("StringLiteral",      re.compile(r'["\'].*["\']')),
    ("LogicalOperator",    re.compile(r"\band\b|\bor\b|\bnot\b")),
]


def classify_operator(diff_snippet: str) -> str:
    for name, pattern in OP_PATTERNS:
        if pattern.search(diff_snippet):
            return name
    return "Other"


# ---------------------------------------------------------------------------
# Helpers de execução
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None) -> tuple[bool, str, str]:
    """Retorna (ok, stdout, stderr)."""
    import os
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        r = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=run_env,
        )
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"timeout após {timeout}s"
    except Exception as exc:
        return False, "", str(exc)


def _python_exe(wt: Path) -> Path:
    """Retorna caminho do Python no venv (Windows ou Unix)."""
    scripts = wt / VENV_NAME / "Scripts"
    if scripts.exists():
        return scripts / "python.exe"
    return wt / VENV_NAME / "bin" / "python"


def _pip_exe(wt: Path) -> Path:
    scripts = wt / VENV_NAME / "Scripts"
    if scripts.exists():
        return scripts / "pip.exe"
    return wt / VENV_NAME / "bin" / "pip"


def _mutmut_exe(wt: Path) -> Path:
    scripts = wt / VENV_NAME / "Scripts"
    if scripts.exists():
        return scripts / "mutmut.exe"
    return wt / VENV_NAME / "bin" / "mutmut"


# ---------------------------------------------------------------------------
# Inferência do diretório fonte a partir dos arquivos de teste
# ---------------------------------------------------------------------------

def infer_src_paths(test_files: list[str], wt: Path) -> list[str]:
    """Heurística para descobrir o diretório de código fonte."""
    candidates: list[str] = []

    for tf in test_files:
        parts = Path(tf).parts
        # test_foo.py → mesmo diretório
        # tests/test_foo.py → diretório pai (src/ ou package name)
        if "tests" in parts or "test" in parts:
            pkg_dirs = [p for p in (wt / d for d in ("src", "lib", "app")) if p.is_dir()]
            if pkg_dirs:
                candidates.extend(str(d.relative_to(wt)) for d in pkg_dirs)
                continue
            # Tentar encontrar pacote Python pela presença de __init__.py
            for d in wt.iterdir():
                if d.is_dir() and (d / "__init__.py").exists() and d.name not in ("tests", "test"):
                    candidates.append(d.name)
        else:
            parent = Path(tf).parent
            if parent != Path("."):
                candidates.append(str(parent))

    # Deduplicar e verificar existência
    seen: set[str] = set()
    valid: list[str] = []
    for c in candidates:
        if c not in seen and (wt / c).is_dir():
            seen.add(c)
            valid.append(c)

    if not valid:
        # Fallback: qualquer subdir com __init__.py (exceto tests)
        for d in wt.iterdir():
            if d.is_dir() and (d / "__init__.py").exists() and "test" not in d.name.lower():
                valid.append(d.name)

    return valid or ["."]


# ---------------------------------------------------------------------------
# Setup do ambiente Python
# ---------------------------------------------------------------------------

def setup_venv(wt: Path) -> tuple[bool, str]:
    py = sys.executable
    ok, _, err = _run([py, "-m", "venv", VENV_NAME], wt, 60)
    if not ok:
        return False, f"venv: {err}"

    pip = str(_pip_exe(wt))

    # Instalar dependências do projeto
    for req_file in ["requirements.txt", "requirements-dev.txt", "requirements_test.txt"]:
        req_path = wt / req_file
        if req_path.exists():
            ok, _, err = _run([pip, "install", "-r", str(req_path)], wt, PIP_TIMEOUT)
            if not ok:
                print(f"      [aviso] pip install -r {req_file} falhou: {err[:100]}")
            break
    else:
        # Sem requirements.txt — tentar setup.py / pyproject.toml
        if (wt / "pyproject.toml").exists() or (wt / "setup.py").exists():
            ok, _, err = _run([pip, "install", "-e", ".[test]"], wt, PIP_TIMEOUT)
            if not ok:
                _run([pip, "install", "-e", "."], wt, PIP_TIMEOUT)

    # Garantir pytest e mutmut
    _run([pip, "install", "pytest", "mutmut"], wt, PIP_TIMEOUT)
    return True, ""


# ---------------------------------------------------------------------------
# Execução e parsing mutmut
# ---------------------------------------------------------------------------

def _get_mutant_ids_by_status(mutmut_exe: str, wt: Path, status: str) -> list[int]:
    """Retorna lista de IDs de mutantes com dado status."""
    ok, stdout, _ = _run([mutmut_exe, "results"], wt, 60)
    if not ok:
        return []
    ids = []
    in_section = False
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(f"--- {status}"):
            in_section = True
            continue
        if line.startswith("---"):
            in_section = False
        if in_section and line.startswith("mutmut #"):
            try:
                ids.append(int(line.split("#")[1].strip()))
            except (IndexError, ValueError):
                pass
    return ids


def parse_mutmut_results(mutmut_exe: str, wt: Path) -> dict:
    """Parseia resultados do mutmut e retorna métricas."""
    ok, stdout, err = _run([mutmut_exe, "results"], wt, 60)
    if not ok:
        return {"status": "parse_error", "error_msg": err[:300]}

    # Parsing da linha de sumário: "xx survived, yy killed, ..."
    total = killed = survived = timeout_count = 0

    survived_re = re.compile(r"(\d+) survived")
    killed_re = re.compile(r"(\d+) killed")
    timeout_re = re.compile(r"(\d+) timeout")
    suspicious_re = re.compile(r"(\d+) suspicious")

    for line in stdout.splitlines():
        m = survived_re.search(line)
        if m:
            survived = int(m.group(1))
        m = killed_re.search(line)
        if m:
            killed = int(m.group(1))
        m = timeout_re.search(line)
        if m:
            timeout_count = int(m.group(1))
        m = suspicious_re.search(line)
        if m:
            survived += int(m.group(1))  # suspicious = survived para MSI

    total = killed + survived + timeout_count
    msi_global = round(killed / total * 100, 2) if total > 0 else None

    # Classificação de operadores via diff dos mutantes sobreviventes (sample)
    by_op: dict[str, dict[str, int]] = defaultdict(lambda: {"killed": 0, "total": 0})
    survived_ids = _get_mutant_ids_by_status(mutmut_exe, wt, "SURVIVED")
    killed_ids = _get_mutant_ids_by_status(mutmut_exe, wt, "KILLED")

    for mid in survived_ids[:50]:  # limitar para evitar chamadas excessivas
        ok2, diff, _ = _run([mutmut_exe, "show", str(mid)], wt, 10)
        op = classify_operator(diff) if ok2 else "Other"
        by_op[op]["total"] += 1

    for mid in killed_ids[:50]:
        ok2, diff, _ = _run([mutmut_exe, "show", str(mid)], wt, 10)
        op = classify_operator(diff) if ok2 else "Other"
        by_op[op]["total"] += 1
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
        "no_coverage": 0,
        "msi_global": msi_global,
        "msi_by_operator": json.dumps(msi_by_op),
        "mutation_tool_used": "mutmut",
    }


# ---------------------------------------------------------------------------
# Execução de um único worktree
# ---------------------------------------------------------------------------

def run_mutmut_on_worktree(
    pr_id: int, agent: str, repo_full_name: str,
    worktree_path: str, test_files_json: str,
    keep_venv: bool = False,
) -> dict:
    wt = Path(worktree_path)
    test_files: list[str] = json.loads(test_files_json or "[]")

    base = {
        "pr_id": pr_id, "agent": agent, "repo_full_name": repo_full_name,
        "total_mutants": None, "killed": None, "survived": None,
        "timeout_mutants": None, "no_coverage": None,
        "msi_global": None, "msi_by_operator": "{}",
        "status": "", "error_msg": "", "duration_seconds": 0,
        "mutation_tool_used": "mutmut",
    }

    if not wt.exists():
        return {**base, "status": "worktree_missing"}

    # Setup venv
    print("    setup venv ...", end=" ", flush=True)
    ok, err = setup_venv(wt)
    if not ok:
        print("ERRO")
        return {**base, "status": "venv_error", "error_msg": err}
    print("OK")

    mutmut = str(_mutmut_exe(wt))
    src_paths = infer_src_paths(test_files, wt)
    paths_arg = ",".join(src_paths[:3])  # mutmut aceita múltiplos com vírgula

    print(f"    mutmut run (src={paths_arg}) ...", end=" ", flush=True)
    t0 = time.time()

    ok, stdout, stderr = _run(
        [mutmut, "run", f"--paths-to-mutate={paths_arg}",
         "--test-time-multiplier=2.0"],
        wt, MUTMUT_TIMEOUT,
    )
    duration = round(time.time() - t0, 1)

    if "timeout" in stderr.lower() and not ok:
        print(f"TIMEOUT ({duration}s)")
        return {**base, "status": "timeout_no_report", "duration_seconds": duration}

    print(f"{'OK' if ok else 'done'} ({duration}s)")

    # Parsear resultados
    metrics = parse_mutmut_results(mutmut, wt)
    if metrics.get("status") == "parse_error":
        return {**base, **metrics, "duration_seconds": duration}

    # Limpar venv para economizar disco
    if not keep_venv:
        venv_dir = wt / VENV_NAME
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)

    status = "ok" if ok else "partial"
    return {**base, **metrics, "status": status, "duration_seconds": duration}


# ---------------------------------------------------------------------------
# Checkpoint logger (idêntico ao Script 04)
# ---------------------------------------------------------------------------

class ResultLogger:
    DONE_STATUSES = {"ok", "partial"}

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._done: set[int] = set()
        self._fh = None
        self._writer = None

    def load(self) -> None:
        if self._path.exists():
            df = pd.read_csv(self._path)
            self._done = set(df[df["status"].isin(self.DONE_STATUSES)]["pr_id"].astype(int))
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
    parser = argparse.ArgumentParser(description="RQ1 Script 05 — mutmut (Python)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-venv", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 05: Execução mutmut (Python)")
    print("=" * 65)

    for required in [ELIGIBLE_CSV, CLONE_LOG_CSV]:
        if not required.exists():
            print(f"[ERRO] {required} não encontrado.")
            raise SystemExit(1)

    eligible = pd.read_csv(ELIGIBLE_CSV)
    clone_log = pd.read_csv(CLONE_LOG_CSV)

    cloned = clone_log[clone_log["status"] == "ok"][["pr_id", "worktree_path"]].copy()
    cloned["pr_id"] = cloned["pr_id"].astype(int)

    py_eligible = eligible[eligible.get("mutation_tool", pd.Series(dtype=str)) == "mutmut"].copy()
    if "mutation_tool" not in eligible.columns:
        print("[AVISO] Coluna mutation_tool ausente — usando todas as PRs")
        py_eligible = eligible.copy()
    py_eligible["pr_id"] = py_eligible["pr_id"].astype(int)

    work_items = py_eligible.merge(cloned, on="pr_id", how="inner")
    print(f"\n  Worktrees Python disponíveis: {len(work_items):,}")

    if args.limit:
        work_items = work_items.head(args.limit)
        print(f"  Limitado a {args.limit} PRs")

    logger = ResultLogger(RESULTS_CSV)
    logger.load()

    total = len(work_items)
    ok_count = err_count = 0

    for i, (_, row) in enumerate(work_items.iterrows(), 1):
        pr_id = int(row["pr_id"])
        if logger.already_done(pr_id):
            continue

        print(f"\n[{i}/{total}] {row.get('repo_full_name', '?')} PR#{pr_id} [{row.get('agent', '?')}]")

        result = run_mutmut_on_worktree(
            pr_id=pr_id,
            agent=str(row.get("agent", "")),
            repo_full_name=str(row.get("repo_full_name", "")),
            worktree_path=str(row["worktree_path"]),
            test_files_json=str(row.get("test_files", "[]")),
            keep_venv=args.keep_venv,
        )
        logger.write(result)

        s = result["status"]
        if s in ("ok", "partial"):
            ok_count += 1
            print(f"  MSI={result['msi_global']}% | mutants={result['total_mutants']}")
        else:
            err_count += 1
            print(f"  [ERRO] {s}: {str(result.get('error_msg', ''))[:120]}")

    logger.close()

    print(f"\n{'='*65}")
    print(f"  Processados com sucesso: {ok_count}")
    print(f"  Erros:                   {err_count}")
    print(f"  Resultados em:           {RESULTS_CSV}")
    print("\nPróximo passo: python rq1_06_aggregate_msi.py")


if __name__ == "__main__":
    main()
