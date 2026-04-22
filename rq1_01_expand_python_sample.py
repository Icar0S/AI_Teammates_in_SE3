#!/usr/bin/env python3
"""
RQ1 — Script 01: Expansão da Amostra para Python
=================================================
Estende rq1_sample.csv (atualmente só vitest JS/TS) com PRs de
repositórios Python, detectando arquivos de teste via padrões de
nomenclatura (test_*.py / *_test.py).

Colunas adicionadas / atualizadas
----------------------------------
  mutation_tool  — 'stryker' (JS/TS, backfill) | 'mutmut' (Python, novo)
  task_type      — tipo de tarefa da PR (feat/fix/refactor/…)

Fonte de dados
--------------
  Primária: HuggingFace (hf://datasets/hao-li/AIDev/)
    - Apenas o subconjunto AIDev-pop (repos com >100 estrelas) possui
      pr_commit_details.parquet completo (711k linhas).
    - Para o dataset completo usa all_pull_request.parquet +
      all_repository.parquet, mas sem os diffs de commits.
  Fallback: OFFICIALDATASET.zip (leitura offline via zipfile)
    - Ativado automaticamente se HuggingFace falhar ou com --offline.

Lógica de checkpoint
--------------------
  - Lê rq1_sample.csv existente e pula pr_ids já presentes.
  - Faz append (nunca sobrescreve).
  - Idempotente: pode ser re-executado com segurança.

Usage
-----
  python rq1_01_expand_python_sample.py
  python rq1_01_expand_python_sample.py --offline   # usa ZIP local
  python rq1_01_expand_python_sample.py --limit 500 # apenas N novas linhas (teste)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT_DIR / ".env")

SAMPLE_CSV = OUT_DIR / "rq1_sample.csv"
DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
HF_BASE = "hf://datasets/hao-li/AIDev"

# Regex para arquivos de teste Python (protocolo: critérios de Yoshimoto et al.)
PYTHON_TEST_RE = re.compile(
    r"(test_[^/]+\.py|[^/]+_test\.py|[^/]+test[^/]*\.py)$",
    re.IGNORECASE,
)

# Reutilizar mapeamento de nomes de agentes do script original
# (importa diretamente para evitar duplicação)
sys.path.insert(0, str(ROOT_DIR))
from rq1_sample_selection import (  # noqa: E402
    NAME_MAP,
    load_last_commit_sha,
    full_name_from_html_url,
    normalise_agent,
)


# ---------------------------------------------------------------------------
# Data loading (HuggingFace ou ZIP offline)
# ---------------------------------------------------------------------------

class _ZipLoader:
    """Carrega parquets do OFFICIALDATASET.zip em memória."""

    def __init__(self, zip_path: Path) -> None:
        self._zip_path = zip_path
        self._zf: zipfile.ZipFile | None = None
        self._names: list[str] = []

    def open(self) -> None:
        self._zf = zipfile.ZipFile(self._zip_path, "r")
        self._names = [m.filename for m in self._zf.infolist()]
        print(f"  [offline] ZIP aberto: {len(self._names)} arquivos")

    def close(self) -> None:
        if self._zf:
            self._zf.close()

    def read_parquet(self, basename: str, columns: list[str] | None = None) -> pd.DataFrame:
        matches = [n for n in self._names if n.endswith(basename)]
        if not matches:
            raise FileNotFoundError(f"{basename} não encontrado no ZIP")
        data = self._zf.read(matches[0])
        df = pd.read_parquet(io.BytesIO(data), columns=columns)
        return df


def _load(
    filename: str,
    offline: bool,
    loader: _ZipLoader | None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Carrega parquet do HuggingFace ou do ZIP offline."""
    if offline and loader is not None:
        return loader.read_parquet(filename, columns=columns)
    try:
        return pd.read_parquet(f"{HF_BASE}/{filename}", columns=columns)
    except Exception as exc:
        if loader is not None:
            print(f"  [fallback] HuggingFace falhou ({exc}), usando ZIP ...")
            return loader.read_parquet(filename, columns=columns)
        raise


# ---------------------------------------------------------------------------
# Detecção de arquivos de teste Python
# ---------------------------------------------------------------------------

def find_python_test_pr_ids(
    offline: bool,
    loader: _ZipLoader | None,
) -> tuple[dict[int, list[str]], set[int]]:
    """
    Lê pr_commit_details e retorna:
      - dict pr_id → lista de arquivos de teste Python
      - set de pr_ids elegíveis
    """
    print("  Carregando pr_commit_details.parquet ...")
    details = _load("pr_commit_details.parquet", offline, loader, columns=["pr_id", "filename"])

    mask = details["filename"].apply(
        lambda f: bool(PYTHON_TEST_RE.search(f)) if isinstance(f, str) else False
    )
    py_test_details = details[mask]

    pr_test_files: dict[int, list[str]] = (
        py_test_details.groupby("pr_id")["filename"]
        .apply(list)
        .to_dict()
    )
    print(f"  PRs com arquivos de teste Python detectados: {len(pr_test_files):,}")
    return pr_test_files, set(pr_test_files.keys())


# ---------------------------------------------------------------------------
# Carregamento de metadados de PR
# ---------------------------------------------------------------------------

def load_ai_python_prs(
    python_test_pr_ids: set[int],
    offline: bool,
    loader: _ZipLoader | None,
) -> pd.DataFrame:
    """Retorna PRs de IA em repos Python com testes detectados."""
    print("  Carregando pull_request.parquet (pop) ...")
    pr_df = _load(
        "pull_request.parquet", offline, loader,
        columns=["id", "agent", "html_url", "merged_at"],
    )
    pr_df["agent"] = pr_df["agent"].map(lambda a: NAME_MAP.get(a, a))
    pr_df["repo_full_name"] = pr_df["html_url"].apply(full_name_from_html_url)

    print("  Carregando repository.parquet ...")
    repo_df = _load("repository.parquet", offline, loader, columns=["full_name", "language"])

    # Join PR → repositório
    merged = pr_df.merge(repo_df, left_on="repo_full_name", right_on="full_name", how="left")

    # Filtros: Python, merged, com arquivos de teste detectados
    python_mask = merged["language"].str.lower() == "python"
    merged_mask = merged["merged_at"].notna()
    test_mask = merged["id"].isin(python_test_pr_ids)

    result = merged[python_mask & merged_mask & test_mask].copy()
    print(f"  PRs de IA Python com testes (pop): {len(result):,}")
    return result


def load_human_python_prs(
    offline: bool,
    loader: _ZipLoader | None,
) -> pd.DataFrame:
    """Retorna PRs humanas em repos Python com type 'test'."""
    print("  Carregando human_pull_request.parquet ...")
    h_pr = _load(
        "human_pull_request.parquet", offline, loader,
        columns=["id", "html_url", "merged_at"],
    )
    h_pr["agent"] = "Human"
    h_pr["repo_full_name"] = h_pr["html_url"].apply(full_name_from_html_url)

    print("  Carregando human_pr_task_type.parquet ...")
    h_task = _load("human_pr_task_type.parquet", offline, loader, columns=["id", "type"])

    merged = h_pr.merge(h_task, on="id", how="inner")

    print("  Carregando repository.parquet ...")
    repo_df = _load("repository.parquet", offline, loader, columns=["full_name", "language"])
    merged = merged.merge(repo_df, left_on="repo_full_name", right_on="full_name", how="left")

    python_mask = merged["language"].str.lower() == "python"
    merged_mask = merged["merged_at"].notna()
    test_mask = merged["type"].isin(["test", "feat", "fix", "refactor"])

    result = merged[python_mask & merged_mask & test_mask].copy()
    print(f"  PRs humanas Python (type=test/feat/fix/refactor): {len(result):,}")
    return result


def load_task_types(
    pr_ids: set[int],
    offline: bool,
    loader: _ZipLoader | None,
) -> dict[int, str]:
    """Carrega task_type das PRs de IA (pr_task_type.parquet)."""
    print("  Carregando pr_task_type.parquet ...")
    task_df = _load("pr_task_type.parquet", offline, loader, columns=["id", "type"])
    filtered = task_df[task_df["id"].isin(pr_ids)]
    return dict(zip(filtered["id"], filtered["type"]))


# ---------------------------------------------------------------------------
# Construção das novas linhas
# ---------------------------------------------------------------------------

def build_python_rows(
    pr_meta: pd.DataFrame,
    pr_test_files: dict[int, list[str]],
    sha_df: pd.DataFrame,
    is_human: bool,
    task_map: dict[int, str],
) -> pd.DataFrame:
    """Constrói DataFrame com o schema de rq1_sample.csv para PRs Python."""
    df = pr_meta.copy()

    # Arquivos de teste (vazio para humanas — não temos commit_details delas)
    if is_human:
        df["test_files"] = df["id"].apply(lambda pid: json.dumps([]))
    else:
        df["test_files"] = df["id"].apply(
            lambda pid: json.dumps(pr_test_files.get(pid, []))
        )

    # SHA do último commit
    df = df.merge(sha_df.rename(columns={"pr_id": "id"}), on="id", how="left")

    # Tipo de tarefa
    df["task_type"] = df["id"].map(task_map).fillna("unknown")

    df["is_human"] = is_human
    df["is_vitest"] = False
    df["mutation_tool"] = "mutmut"

    cols_map = {
        "id": "pr_id",
        "html_url": "pr_html_url",
        "full_name": "repo_full_name",
    }
    df = df.rename(columns=cols_map)

    # Garantir todas as colunas necessárias
    needed = [
        "pr_id", "agent", "repo_full_name", "merge_commit_sha",
        "test_files", "is_vitest", "pr_html_url", "merged_at",
        "is_human", "task_type", "mutation_tool",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    return df[needed]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Expande rq1_sample.csv com PRs Python")
    parser.add_argument("--offline", action="store_true", help="Usar ZIP offline")
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help="Caminho do ZIP")
    parser.add_argument("--limit", type=int, default=0, help="Limitar N novas linhas (teste)")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 01: Expansão da Amostra para Python")
    print("=" * 65)

    # ── Carregar amostra existente para checkpoint ─────────────────────────
    existing_ids: set[int] = set()
    if SAMPLE_CSV.exists():
        existing_df = pd.read_csv(SAMPLE_CSV)
        existing_ids = set(existing_df["pr_id"].astype(int))
        print(f"\n[checkpoint] {len(existing_ids)} pr_ids já no rq1_sample.csv")

        # Backfill mutation_tool se a coluna não existir
        if "mutation_tool" not in existing_df.columns:
            existing_df["mutation_tool"] = "stryker"
            existing_df.to_csv(SAMPLE_CSV, index=False)
            print("  [backfill] Adicionada coluna mutation_tool='stryker' nas linhas existentes")

        if "task_type" not in existing_df.columns:
            existing_df["task_type"] = "unknown"
            existing_df.to_csv(SAMPLE_CSV, index=False)
            print("  [backfill] Adicionada coluna task_type='unknown' nas linhas existentes")
    else:
        print("\n[aviso] rq1_sample.csv não encontrado — execute rq1_sample_selection.py primeiro.")
        print("        Criando arquivo novo com apenas as linhas Python.")

    # ── Inicializar loader offline se necessário ───────────────────────────
    loader: _ZipLoader | None = None
    zip_path = Path(args.zip)
    if args.offline or not zip_path.exists():
        if zip_path.exists():
            loader = _ZipLoader(zip_path)
            loader.open()
        elif args.offline:
            print(f"[ERRO] ZIP não encontrado: {zip_path}")
            raise SystemExit(1)

    try:
        # ── 1. Detectar PRs Python com arquivos de teste ───────────────────
        print("\n[1/5] Detectando arquivos de teste Python nos diffs ...")
        pr_test_files, python_test_pr_ids = find_python_test_pr_ids(args.offline, loader)

        # Remover pr_ids já no sample
        new_ids = python_test_pr_ids - existing_ids
        print(f"  PRs Python novas (não duplicadas): {len(new_ids):,}")

        if not new_ids:
            print("\n  Nenhuma PR nova para adicionar. Saindo.")
            return

        # ── 2. Metadados de PRs de IA Python ──────────────────────────────
        print("\n[2/5] Carregando metadados de PRs de IA Python ...")
        ai_python = load_ai_python_prs(new_ids, args.offline, loader)
        ai_python = ai_python[~ai_python["id"].isin(existing_ids)]

        # ── 3. Metadados de PRs humanas Python ────────────────────────────
        print("\n[3/5] Carregando metadados de PRs humanas Python ...")
        human_python = load_human_python_prs(args.offline, loader)
        human_python = human_python[~human_python["id"].isin(existing_ids)]

        all_new_ids = set(ai_python["id"]) | set(human_python["id"])
        if not all_new_ids:
            print("\n  Nenhuma PR nova após filtragem. Saindo.")
            return

        # ── 4. SHAs e task types ───────────────────────────────────────────
        print("\n[4/5] Carregando SHAs e tipos de tarefa ...")
        sha_df = load_last_commit_sha(all_new_ids)

        ai_task_map = load_task_types(set(ai_python["id"]), args.offline, loader)
        # Para humanas usa a coluna 'type' já carregada
        human_task_map = dict(zip(human_python["id"], human_python.get("type", pd.Series(dtype=str))))

        # ── 5. Construção das novas linhas ─────────────────────────────────
        print("\n[5/5] Construindo novas linhas ...")
        ai_rows = build_python_rows(ai_python, pr_test_files, sha_df, False, ai_task_map)
        human_rows = build_python_rows(human_python, pr_test_files, sha_df, True, human_task_map)

        new_rows = pd.concat([ai_rows, human_rows], ignore_index=True)

        if args.limit > 0:
            new_rows = new_rows.head(args.limit)
            print(f"  [limite] Truncando para {args.limit} linhas")

        print(f"\n  Novas linhas Python (IA):     {len(new_rows[~new_rows['is_human']]):,}")
        print(f"  Novas linhas Python (humanas): {len(new_rows[new_rows['is_human']]):,}")

        # ── Append ao CSV ──────────────────────────────────────────────────
        write_header = not SAMPLE_CSV.exists()
        new_rows.to_csv(SAMPLE_CSV, mode="a", header=write_header, index=False)
        print(f"\n  Append concluído em: {SAMPLE_CSV}")

        # ── Resumo final ───────────────────────────────────────────────────
        final_df = pd.read_csv(SAMPLE_CSV)
        print(f"\n{'='*65}")
        print(" RESUMO FINAL — rq1_sample.csv")
        print(f"{'='*65}")
        print(f"  Total de linhas: {len(final_df):,}")
        print(f"  PRs únicas:      {final_df['pr_id'].nunique():,}")
        print(f"\n  Por ferramenta de mutação:")
        if "mutation_tool" in final_df.columns:
            print(final_df.groupby("mutation_tool")["pr_id"].count().to_string())
        print(f"\n  Por agente:")
        print(final_df.groupby("agent")["pr_id"].count().to_string())

        dups = final_df.duplicated(subset=["pr_id"]).sum()
        if dups > 0:
            print(f"\n  [AVISO] {dups} linhas duplicadas por pr_id detectadas!")
        else:
            print(f"\n  [OK] Sem pr_ids duplicados.")

    finally:
        if loader:
            loader.close()

    print("\nScript 01 concluído. Próximo passo: rq1_02_eligibility_filter.py")


if __name__ == "__main__":
    main()
