#!/usr/bin/env python3
"""
RQ1 — Fase 2: Clone dos Repositórios
======================================
Lê rq1_sample.csv e clona cada repositório único no estado exato do
merge_commit_sha da PR, reproduzindo o ambiente que o Stryker precisa.

Por que clonar?
---------------
O dataset AIDev contém apenas diffs (patches). O Stryker precisa do projeto
completo em disco para:
  - instalar dependências via npm install
  - executar a suíte de testes no baseline (deve passar antes de mutar)
  - injetar mutantes no src e re-executar os testes

Estrutura de diretórios criada
-------------------------------
  repos/
    owner__repo/          ← clone completo no HEAD do repo
      .git/
      src/
      package.json
      ...
    owner__repo__<sha7>/  ← worktree isolado no commit exato da PR
      ...

Cada PR usa um worktree Git separado do mesmo clone base, evitando
re-clonar N vezes o mesmo repositório quando há múltiplas PRs do mesmo repo.

Checkpoints
-----------
  AIDev/clone_log.csv — registra resultado de cada repo/PR:
    repo_full_name, pr_id, sha, status, error_msg

  Repos e worktrees já existentes são pulados automaticamente.

Usage
-----
  python rq1_clone_repos.py

  # Para processar apenas um agente específico:
  python rq1_clone_repos.py --agent "Claude Code"

  # Para limitar o número de repos (teste rápido):
  python rq1_clone_repos.py --limit 5

Pré-requisitos
--------------
  - git instalado e no PATH
  - GITHUB_TOKEN no .env (escopo public_repo — para checar disponibilidade)
  - rq1_sample.csv gerado pela Fase 1
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"
REPOS_DIR = ROOT_DIR / "repos"
SAMPLE_CSV = OUT_DIR / "rq1_sample.csv"
CLONE_LOG = OUT_DIR / "clone_log.csv"

OUT_DIR.mkdir(exist_ok=True)
REPOS_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT_DIR / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GIT_TIMEOUT = 300  # segundos por operação git

# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass
class CloneResult:
    repo_full_name: str
    pr_id: int
    sha: str
    status: str
    error_msg: str = ""
    worktree_path: str = ""


# ---------------------------------------------------------------------------
# GitHub API — verificação de disponibilidade
# ---------------------------------------------------------------------------


def check_repo_available(full_name: str) -> bool:
    """
    Checa via GitHub API se o repo ainda existe e é público.
    Evita tentativas de clone que vão falhar com 404/403.
    """
    url = f"https://api.github.com/repos/{full_name}"
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return not data.get("private", True)  # só repos públicos
        return False
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Operações Git
# ---------------------------------------------------------------------------


def repo_to_dirname(full_name: str) -> str:
    """'owner/repo' → 'owner__repo' (seguro para nome de diretório)."""
    return full_name.replace("/", "__")


def clone_repo(full_name: str, clone_dir: Path) -> tuple[bool, str]:
    """
    Clona o repo para clone_dir usando HTTPS com token (se disponível).
    Retorna (sucesso, mensagem_de_erro).
    """
    token_prefix = f"{GITHUB_TOKEN}@" if GITHUB_TOKEN else ""
    url = f"https://{token_prefix}github.com/{full_name}.git"

    cmd = [
        "git",
        "clone",
        "--filter=blob:none",
        "--no-single-branch",  # mantém refs de todos os branches
        url,
        str(clone_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout durante git clone"
    except FileNotFoundError:
        return False, "git não encontrado no PATH"


def create_worktree(
    clone_dir: Path, sha: str, worktree_dir: Path
) -> tuple[bool, str]:  # noqa: E501
    """
    Cria um worktree Git isolado no commit SHA especificado.
    Permite que múltiplas PRs do mesmo repo usem commits diferentes
    sem interferência, sem precisar clonar o repo múltiplas vezes.
    """
    cmd = [
        "git",
        "-C",
        str(clone_dir),
        "worktree",
        "add",
        "--detach",
        str(worktree_dir),
        sha,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, ""

        # SHA pode não estar disponível localmente no partial clone — busca
        fetch_result = subprocess.run(
            ["git", "-C", str(clone_dir), "fetch", "--unshallow"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        if fetch_result.returncode != 0:
            return False, f"fetch falhou: {fetch_result.stderr.strip()}"

        # Tenta novamente após o fetch
        result2 = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result2.returncode == 0:
            return True, ""
        return False, result2.stderr.strip()

    except subprocess.TimeoutExpired:
        return False, "timeout durante git worktree add"


# ---------------------------------------------------------------------------
# Log de resultados
# ---------------------------------------------------------------------------


class CloneLogger:
    """Escreve clone_log.csv com checkpointing incremental."""

    FIELDS = [
        "repo_full_name",
        "pr_id",
        "sha",
        "status",
        "error_msg",
        "worktree_path",
    ]  # noqa: E501

    def __init__(self, path: Path) -> None:
        self.path = path
        self._done: dict[tuple[str, int], str] = {}  # (repo, pr_id) → status

        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                self._done[(row["repo_full_name"], int(row["pr_id"]))] = row[
                    "status"
                ]  # noqa: E501
            print(f"Checkpoint: {len(self._done)} entradas anteriores carrega")

        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if not path.exists() or path.stat().st_size == 0:
            self._writer.writeheader()

    def already_done(self, repo: str, pr_id: int) -> bool:
        status = self._done.get((repo, pr_id))
        return status in ("ok", "skipped")

    def write(self, result: CloneResult) -> None:
        self._writer.writerow(
            {
                "repo_full_name": result.repo_full_name,
                "pr_id": result.pr_id,
                "sha": result.sha,
                "status": result.status,
                "error_msg": result.error_msg,
                "worktree_path": result.worktree_path,
            }
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def process_repos(sample: pd.DataFrame) -> None:
    logger = CloneLogger(CLONE_LOG)
    counters = {
        "ok": 0,
        "skipped": 0,
        "repo_unavailable": 0,
        "clone_error": 0,
        "checkout_error": 0,
    }

    # Agrupa por repo para clonar uma vez e criar N worktrees
    repos = sample.groupby("repo_full_name")
    total_repos = len(repos)

    for repo_idx, (full_name, group) in enumerate(repos, 1):
        full_name = str(full_name)
        print(f"\n[{repo_idx}/{total_repos}] {full_name} ({len(group)} PRs)")

        clone_dir = REPOS_DIR / repo_to_dirname(full_name)

        # ── Verificação de disponibilidade ───────────────────────────────
        if not clone_dir.exists():
            print("Verificando disponibilidade no GitHub ...")
            if not check_repo_available(full_name):
                print("INDISPONÍVEL")
                for _, pr_row in group.iterrows():
                    result = CloneResult(
                        repo_full_name=full_name,
                        pr_id=int(pr_row["pr_id"]),
                        sha=str(pr_row.get("merge_commit_sha", "")),
                        status="repo_unavailable",
                        error_msg="repo não existe ou é privado",
                    )
                    logger.write(result)
                    counters["repo_unavailable"] += 1
                continue

            # ── Clone base ────────────────────────────────────────────────
            print(f"  Clonando para {clone_dir} ...")
            ok, err = clone_repo(full_name, clone_dir)
            if not ok:
                print(f"  ERRO no clone: {err}")
                for _, pr_row in group.iterrows():
                    result = CloneResult(
                        repo_full_name=full_name,
                        pr_id=int(pr_row["pr_id"]),
                        sha=str(pr_row.get("merge_commit_sha", "")),
                        status="clone_error",
                        error_msg=err,
                    )
                    logger.write(result)
                    counters["clone_error"] += 1
                continue
            print("Clone OK.")
        else:
            print("Clone base já existe, pulando git clone.")

        # ── Worktrees por PR ──────────────────────────────────────────────
        for _, pr_row in group.iterrows():
            pr_id = int(pr_row["pr_id"])
            sha = str(pr_row.get("merge_commit_sha", ""))

            if logger.already_done(full_name, pr_id):
                print(f"  PR {pr_id}: já processada, pulando.")
                counters["skipped"] += 1
                continue

            if not sha or sha == "nan":
                print(f"  PR {pr_id}: SHA ausente, pulando.")
                result = CloneResult(
                    repo_full_name=full_name,
                    pr_id=pr_id,
                    sha=sha,
                    status="checkout_error",
                    error_msg="merge_commit_sha ausente",
                )
                logger.write(result)
                counters["checkout_error"] += 1
                continue

            sha7 = sha[:7]
            worktree_dir = REPOS_DIR / f"{repo_to_dirname(full_name)}__{sha7}"

            if worktree_dir.exists():
                print(f"  PR {pr_id} (SHA {sha7}): worktree existe pulando")
                result = CloneResult(
                    repo_full_name=full_name,
                    pr_id=pr_id,
                    sha=sha,
                    status="skipped",
                    worktree_path=str(worktree_dir),
                )
                logger.write(result)
                counters["skipped"] += 1
                continue

            print(f"  PR {pr_id}: criando worktree @ {sha7} ...")
            ok, err = create_worktree(clone_dir, sha, worktree_dir)
            if ok:
                print(f"  OK → {worktree_dir}")
                result = CloneResult(
                    repo_full_name=full_name,
                    pr_id=pr_id,
                    sha=sha,
                    status="ok",
                    worktree_path=str(worktree_dir),
                )
                counters["ok"] += 1
            else:
                print(f"  ERRO no checkout: {err}")
                result = CloneResult(
                    repo_full_name=full_name,
                    pr_id=pr_id,
                    sha=sha,
                    status="checkout_error",
                    error_msg=err,
                )
                counters["checkout_error"] += 1

            logger.write(result)

    logger.close()

    # ── Resumo ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" RESUMO DA FASE 2")
    print("=" * 65)
    print(f"  Worktrees criados com sucesso : {counters['ok']}")
    print(f"  Já existiam (skipped)         : {counters['skipped']}")
    print(f"  Repos indisponíveis           : {counters['repo_unavailable']}")
    print(f"  Erros de clone                : {counters['clone_error']}")
    print(f"  Erros de checkout             : {counters['checkout_error']}")
    print(f"\n  Log completo: {CLONE_LOG}")
    print(f"  Worktrees em: {REPOS_DIR}")
    print("\nFase 2 concluída. Próximo passo: rq1_setup_stryker.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 Fase 2 — Clone repo")
    parser.add_argument(
        "--agent", help="Filtrar por agente específico (ex: 'Claude Code')"
    )
    parser.add_argument(
        "--limit", type=int, help="Limitar número de repos (para testes)"
    )
    parser.add_argument(
        "--csv",
        default=str(SAMPLE_CSV),
        help="CSV de entrada (default: AIDev/rq1_sample.csv). "
        "Use AIDev/rq1_eligible.csv para clonar apenas PRs elegíveis.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERRO: {csv_path} não encontrado.")
        print(
            "Execute rq1_sample_selection.py"
            " (e rq1_02_eligibility_filter.py) antes desta fase."
        )
        sys.exit(1)

    if not GITHUB_TOKEN:
        print("AVISO: GITHUB_TOKEN não definido.")
        print("Repos indisponíveis não serão detectados antes do clone.")
        print("Adicione GITHUB_TOKEN=ghp_... no arquivo .env\n")

    sample = pd.read_csv(csv_path)

    if args.agent:
        sample = sample[sample["agent"] == args.agent].copy()
        print(f"Filtrado para agente: {args.agent} ({len(sample)} PRs)")

    if args.limit:
        repos_limitados = sample["repo_full_name"].unique()[: args.limit]
        sample = sample[sample["repo_full_name"].isin(repos_limitados)].copy()
        print(f"Limitado a {args.limit} repos ({len(sample)} PRs)")

    print("=" * 65)
    print(" RQ1 — Fase 2: Clone dos Repositórios")
    print("=" * 65)
    print(f"\n  PRs na amostra : {len(sample):,}")
    print(f"  Repos únicos   : {sample['repo_full_name'].nunique():,}")
    print(f"  Destino        : {REPOS_DIR}")

    process_repos(sample)


if __name__ == "__main__":
    main()
