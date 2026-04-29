#!/usr/bin/env python3
"""
RQ1 — Script 05: Execução cosmic-ray (Python, Windows-compatible)
==================================================================
mutmut não suporta Windows nativamente (requer WSL).
Este script usa cosmic-ray 8.x que roda em qualquer SO.

Schema de output (mesmo schema de stryker_results.csv)
------------------------------------------------------
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
import shutil
import sqlite3
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
CR_INIT_TIMEOUT = 300
CR_EXEC_TIMEOUT = 600   # 10 min por PR
CR_PER_MUTANT_TIMEOUT = 10.0  # s por mutante

load_dotenv(ROOT_DIR / ".env")

RESULT_COLS = [
    "pr_id", "agent", "repo_full_name",
    "total_mutants", "killed", "survived", "timeout_mutants", "no_coverage",
    "msi_global", "msi_by_operator",
    "status", "error_msg", "duration_seconds", "mutation_tool_used",
]

# Exclusões padrão para detecção de módulo fonte
_DIR_EXCLUSIONS = {
    "tests", "test", "testing", ".git", ".venv_mutmut", "__pycache__",
    "docs", "doc", "examples", "example", "migrations", "scripts",
    "script", "benchmark", "benchmarks", "data", "fixtures",
}


# ---------------------------------------------------------------------------
# Helpers de execução
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str], cwd: Path, timeout: int, env: dict | None = None,
) -> tuple[bool, str, str]:
    """Retorna (ok, stdout, stderr)."""
    import os
    run_env = os.environ.copy()
    run_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        run_env.update(env)
    try:
        r = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=run_env, encoding="utf-8", errors="replace",
        )
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"timeout após {timeout}s"
    except Exception as exc:
        return False, "", str(exc)


def _python_exe(wt: Path) -> Path:
    scripts = wt / VENV_NAME / "Scripts"
    if scripts.exists():
        return scripts / "python.exe"
    return wt / VENV_NAME / "bin" / "python"


def _pip_exe(wt: Path) -> Path:
    scripts = wt / VENV_NAME / "Scripts"
    if scripts.exists():
        return scripts / "pip.exe"
    return wt / VENV_NAME / "bin" / "pip"


def _cr_exe(wt: Path) -> Path:
    scripts = wt / VENV_NAME / "Scripts"
    if scripts.exists():
        return scripts / "cosmic-ray.exe"
    return wt / VENV_NAME / "bin" / "cosmic-ray"


# ---------------------------------------------------------------------------
# Detecção do módulo fonte para cosmic-ray
# ---------------------------------------------------------------------------

def _test_to_source_path(test_file: str, wt: Path) -> str | None:
    """
    Derive the source file path (POSIX, relative to wt) from a test file.
    cosmic-ray accepts file paths in module-path.

    Strategy: strip leading 'tests/'/'test/', strip 'test_' prefix,
    check if that reconstructed path exists; fall back to rglob.
    """
    parts = list(Path(test_file).parts)
    if not parts:
        return None

    if parts[0].lower() in ("tests", "test", "testing"):
        parts = parts[1:]
    if not parts:
        return None

    fname = parts[-1]
    # Exclude pytest infrastructure files — they have no source counterpart
    if fname in ("conftest.py", "pytest.ini", "setup.cfg"):
        return None

    if fname.startswith("test_"):
        src_fname = fname[5:]
    elif fname.endswith("_test.py"):
        src_fname = fname[:-8] + ".py"
    else:
        src_fname = fname

    # Direct reconstruction: tests/pkg/sub/test_foo.py → pkg/sub/foo.py
    candidate = wt.joinpath(*(parts[:-1] + [src_fname]))
    if candidate.is_file():
        return candidate.relative_to(wt).as_posix()

    # rglob fallback (outside test/venv dirs)
    for hit in sorted(wt.rglob(src_fname)):
        hit_str = hit.as_posix()
        if (
            "test" not in hit_str.lower()
            and ".venv" not in hit_str
            and "__pycache__" not in hit_str
        ):
            return hit.relative_to(wt).as_posix()

    return None


def find_module_for_cosmic_ray(
    test_files: list[str],
    wt: Path,
) -> tuple[str | None, dict[str, str]]:
    """
    Retorna (module_path_str, extra_env) para a config do cosmic-ray.
    module_path_str é um caminho POSIX relativo ao worktree ou nome de
    módulo importável. Prefere arquivos específicos inferidos dos test_files.
    """
    # 1. Tentar derivar arquivo fonte específico de cada test file
    for tf in test_files:
        src = _test_to_source_path(tf, wt)
        if src:
            return src, {"PYTHONPATH": str(wt)}

    # 2. Fallback: pacotes de nível superior com __init__.py
    top_pkgs = [
        d.name for d in sorted(wt.iterdir())
        if d.is_dir()
        and (d / "__init__.py").exists()
        and d.name.lower() not in _DIR_EXCLUSIONS
        and not d.name.startswith(".")
    ]
    if top_pkgs:
        return top_pkgs[0], {"PYTHONPATH": str(wt)}

    # 3. Layout src/ ou lib/
    for srcdir in ("src", "lib"):
        src = wt / srcdir
        if src.is_dir():
            pkgs = [
                d.name for d in sorted(src.iterdir())
                if d.is_dir()
                and (d / "__init__.py").exists()
                and d.name.lower() not in _DIR_EXCLUSIONS
            ]
            if pkgs:
                return pkgs[0], {"PYTHONPATH": str(src)}

    # 4. Módulo simples (arquivo .py de nível superior)
    simple = [
        f.stem for f in sorted(wt.glob("*.py"))
        if f.stem not in ("setup", "conftest", "conf")
        and not f.stem.startswith("test_")
    ]
    if simple:
        return simple[0], {"PYTHONPATH": str(wt)}

    return None, {}


def infer_test_command(test_files: list[str], wt: Path) -> str:
    """
    Retorna comando pytest para a PR usando o caminho absoluto do
    Python do venv (forward slashes para compatibilidade com TOML).
    """
    py = str(_python_exe(wt)).replace("\\", "/")
    to = int(CR_PER_MUTANT_TIMEOUT * 2)

    existing = [tf for tf in test_files if (wt / tf).exists()]
    if existing:
        files_str = " ".join(
            tf.replace("\\", "/") for tf in existing[:5]
        )
        return f"{py} -m pytest {files_str} -x -q --tb=no --timeout={to}"

    for test_dir in ("tests", "test", "testing"):
        if (wt / test_dir).is_dir():
            return f"{py} -m pytest {test_dir} -x -q --tb=no --timeout={to}"

    return f"{py} -m pytest . -x -q --tb=no --timeout={to}"


# ---------------------------------------------------------------------------
# Setup do ambiente Python
# ---------------------------------------------------------------------------

def setup_venv(wt: Path) -> tuple[bool, str]:
    cr_exe = _cr_exe(wt)

    if not cr_exe.exists():
        # Create fresh venv
        py = sys.executable
        ok, _, err = _run([py, "-m", "venv", VENV_NAME], wt, 60)
        if not ok:
            return False, f"venv: {err}"

        # Evaluate pip AFTER venv creation so Scripts/ exists on Windows
        pip = str(_pip_exe(wt))

        # Install ALL requirement files found (not just the first one)
        req_files = (
            "requirements.txt", "requirements-dev.txt",
            "requirements_test.txt", "requirements-test.txt",
        )
        found_any_req = False
        for req_file in req_files:
            req_path = wt / req_file
            if req_path.exists():
                found_any_req = True
                ok2, _, e2 = _run(
                    [pip, "install", "-r", str(req_path)], wt, PIP_TIMEOUT
                )
                if not ok2:
                    print(
                        f"      [aviso] pip install -r {req_file} falhou: "
                        f"{e2[:80]}"
                    )

        if not found_any_req:
            has_setup = (
                (wt / "pyproject.toml").exists()
                or (wt / "setup.py").exists()
            )
            if has_setup:
                ok2, _, _ = _run(
                    [pip, "install", "-e", ".[test]"], wt, PIP_TIMEOUT
                )
                if not ok2:
                    _run([pip, "install", "-e", "."], wt, PIP_TIMEOUT)

    # Always ensure essential test tools — re-evaluate pip so path is correct
    # even when the venv already existed (cr_exe.exists() was True above).
    pip = str(_pip_exe(wt))
    ok_ens, _, err_ens = _run(
        [pip, "install", "--quiet",
         "pytest", "pytest-timeout", "pytest-asyncio", "anyio",
         "cosmic-ray"],
        wt, PIP_TIMEOUT,
    )
    if not ok_ens:
        print(f"      [aviso] ensure-install falhou: {err_ens[:120]}")
    return True, ""


# ---------------------------------------------------------------------------
# Execução cosmic-ray e parsing
# ---------------------------------------------------------------------------

def _write_cr_config(
    wt: Path,
    module_name: str,
    test_cmd: str,
    config_file: Path,
) -> None:
    # Use explicit \n to avoid Windows \r\n breaking TOML parsers
    content = (
        f'[cosmic-ray]\n'
        f'module-path = "{module_name}"\n'
        f'timeout = {CR_PER_MUTANT_TIMEOUT}\n'
        f'test-command = "{test_cmd}"\n'
        f'\n'
        f'[cosmic-ray.distributor]\n'
        f'name = "local"\n'
    )
    with open(config_file, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def parse_cosmic_ray_results(session_db: Path) -> dict:
    """Parseia resultados do cosmic-ray via SQLite."""
    if not session_db.exists():
        return {"status": "parse_error", "error_msg": "session db missing"}

    conn = sqlite3.connect(str(session_db))
    try:
        rows = conn.execute("""
            SELECT ms.operator_name,
                   wr.worker_outcome,
                   wr.test_outcome
            FROM work_items wi
            JOIN mutation_specs ms ON wi.job_id = ms.job_id
            LEFT JOIN work_results wr ON wi.job_id = wr.job_id
        """).fetchall()
    except Exception as exc:
        conn.close()
        return {"status": "parse_error", "error_msg": str(exc)[:200]}
    conn.close()

    total = killed = survived = timeout_count = 0
    by_op: dict[str, dict[str, int]] = defaultdict(
        lambda: {"killed": 0, "total": 0}
    )

    for op_name, worker_outcome, test_outcome in rows:
        if worker_outcome is None:
            continue  # não executado (pr timeout antes de terminar)

        op_short = op_name.split("/")[-1] if op_name else "Unknown"
        wo = worker_outcome.upper() if worker_outcome else ""
        to = test_outcome.upper() if test_outcome else ""

        if wo == "TIMEOUT":
            timeout_count += 1
            total += 1
        elif wo == "NORMAL":
            total += 1
            by_op[op_short]["total"] += 1
            if to == "KILLED":
                killed += 1
                by_op[op_short]["killed"] += 1
            elif to == "SURVIVED":
                survived += 1
        # "EXCEPTION" → mutante incompetente, ignorar no MSI

    msi_global = round(killed / total * 100, 2) if total > 0 else None
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
        "mutation_tool_used": "cosmic-ray",
    }


def run_cr_on_worktree(
    pr_id: int,
    agent: str,
    repo_full_name: str,
    worktree_path: str,
    test_files_json: str,
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
        "mutation_tool_used": "cosmic-ray",
    }

    if not wt.exists():
        return {**base, "status": "worktree_missing"}

    # ── 1. Setup venv ─────────────────────────────────────────────────────
    print("    setup venv ...", end=" ", flush=True)
    ok, err = setup_venv(wt)
    if not ok:
        print("ERRO")
        return {**base, "status": "venv_error", "error_msg": err}
    print("OK")

    # ── 2. Detectar módulo fonte ───────────────────────────────────────────
    module_name, extra_env = find_module_for_cosmic_ray(test_files, wt)
    if not module_name:
        return {
            **base,
            "status": "no_module",
            "error_msg": "nenhum módulo Python encontrado",
        }

    test_cmd = infer_test_command(test_files, wt)
    cr_exe = str(_cr_exe(wt))
    config_file = wt / "cr_rq1.toml"
    session_db = wt / "cr_rq1.sqlite"

    # Inject venv Scripts/bin into PATH so "python" in test-command
    # resolves to the worktree's venv Python, not the system Python.
    import os as _os
    venv_scripts = wt / VENV_NAME / "Scripts"
    if not venv_scripts.exists():
        venv_scripts = wt / VENV_NAME / "bin"
    extra_env["PATH"] = (
        str(venv_scripts) + _os.pathsep + _os.environ.get("PATH", "")
    )

    # Limpar sessão anterior se existir; se bloqueada, usar nome único
    if session_db.exists():
        try:
            session_db.unlink()
        except PermissionError:
            import time as _time
            session_db = wt / f"cr_rq1_{int(_time.time())}.sqlite"

    _write_cr_config(wt, module_name, test_cmd, config_file)

    print(
        f"    cosmic-ray (module={module_name}) ...",
        end=" ", flush=True,
    )
    t0 = time.time()

    # ── 3. cosmic-ray init ─────────────────────────────────────────────────
    ok_init, _, err_init = _run(
        [cr_exe, "init", str(config_file), str(session_db)],
        wt, CR_INIT_TIMEOUT, env=extra_env,
    )
    if not ok_init:
        print(f"init ERRO ({round(time.time()-t0, 1)}s)")
        return {
            **base, "status": "cr_init_error",
            "error_msg": err_init[:200],
            "duration_seconds": round(time.time() - t0, 1),
        }

    # ── 4. cosmic-ray baseline ────────────────────────────────────────────
    ok_base, out_base, err_base = _run(
        [cr_exe, "baseline", str(config_file)],
        wt, 120, env=extra_env,
    )
    if not ok_base:
        print(f"baseline FALHOU ({round(time.time()-t0, 1)}s)")
        combined = (err_base + "\n" + out_base).strip()[:300]
        return {
            **base, "status": "baseline_fail",
            "error_msg": combined,
            "duration_seconds": round(time.time() - t0, 1),
        }

    # ── 5. cosmic-ray exec ─────────────────────────────────────────────────
    ok_exec, out_exec, err_exec = _run(
        [cr_exe, "exec", str(config_file), str(session_db)],
        wt, CR_EXEC_TIMEOUT, env=extra_env,
    )
    duration = round(time.time() - t0, 1)

    if not ok_exec and "timeout" in err_exec.lower():
        print(f"TIMEOUT ({duration}s)")
        status_str = "partial"
    else:
        if err_exec.strip():
            print(f"\n    [exec stderr]: {err_exec[:200]}", flush=True)
        print(f"{'OK' if ok_exec else 'done'} ({duration}s)")
        status_str = "ok" if ok_exec else "partial"

    # ── 6. Parsear resultados ──────────────────────────────────────────────
    metrics = parse_cosmic_ray_results(session_db)
    if metrics.get("status") == "parse_error":
        return {
            **base, **metrics,
            "duration_seconds": duration,
        }

    # ── 7. Limpar artefatos ────────────────────────────────────────────────
    for artifact in (config_file, session_db):
        if artifact.exists():
            artifact.unlink(missing_ok=True)

    if not keep_venv:
        venv_dir = wt / VENV_NAME
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)

    return {
        **base, **metrics,
        "status": status_str,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------------------------
# Checkpoint logger
# ---------------------------------------------------------------------------

class ResultLogger:
    DONE_STATUSES = {"ok", "partial", "baseline_fail", "no_module", "venv_error"}

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._done: set[int] = set()
        self._fh = None
        self._writer = None

    def load(self) -> None:
        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                df = pd.read_csv(self._path)
                mask = df["status"].isin(self.DONE_STATUSES)
                self._done = set(df[mask]["pr_id"].astype(int))
                print(
                    f"  [checkpoint] {len(self._done)} PRs já processadas"
                )
            except Exception:
                print("  [checkpoint] CSV corrompido, iniciando do zero")
        write_header = (
            not self._path.exists() or self._path.stat().st_size == 0
        )
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
    parser = argparse.ArgumentParser(
        description="RQ1 Script 05 — cosmic-ray (Python)"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-venv", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 05: Execução cosmic-ray (Python)")
    print("=" * 65)

    for required in [ELIGIBLE_CSV, CLONE_LOG_CSV]:
        if not required.exists():
            print(f"[ERRO] {required} não encontrado.")
            raise SystemExit(1)

    eligible = pd.read_csv(ELIGIBLE_CSV)
    clone_log = pd.read_csv(CLONE_LOG_CSV)

    cloned = clone_log[clone_log["status"] == "ok"][
        ["pr_id", "worktree_path"]
    ].copy()
    cloned["pr_id"] = cloned["pr_id"].astype(int)

    if "mutation_tool" in eligible.columns:
        py_eligible = eligible[eligible["mutation_tool"] == "mutmut"].copy()
    else:
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

        print(
            f"\n[{i}/{total}] {row.get('repo_full_name', '?')} "
            f"PR#{pr_id} [{row.get('agent', '?')}]"
        )

        result = run_cr_on_worktree(
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
            print(
                f"  MSI={result['msi_global']}% "
                f"| mutants={result['total_mutants']}"
            )
        else:
            err_count += 1
            print(
                f"  [ERRO] {s}: "
                f"{str(result.get('error_msg', ''))[:120]}"
            )

    logger.close()

    print(f"\n{'='*65}")
    print(f"  Processados com sucesso: {ok_count}")
    print(f"  Erros:                   {err_count}")
    print(f"  Resultados em:           {RESULTS_CSV}")
    print("\nPróximo passo: python rq1_06_aggregate_msi.py")


if __name__ == "__main__":
    main()
