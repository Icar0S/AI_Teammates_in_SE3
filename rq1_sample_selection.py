#!/usr/bin/env python3
"""
RQ1 — Fase 1: Seleção de Amostras
==================================
Produz rq1_sample.csv com as PRs de teste (IA + humanos) elegíveis para
execução do Stryker na Fase 2.

Critérios de inclusão
---------------------
  1. PR do tipo "test" (pr_task_type.parquet) e com merged_at preenchido
  2. Contém pelo menos 1 arquivo *.test.ts / *.spec.ts / *.test.tsx / *.spec.tsx
  3. Repositório usa vitest como test runner
  4. (apenas PRs de IA) repositório com >= MIN_AI_PRS_PER_REPO PRs de IA

Colunas do output
-----------------
  pr_id, agent, repo_full_name, merge_commit_sha,
  test_files, is_vitest, pr_html_url, merged_at, is_human

Usage
-----
  python rq1_sample_selection.py

Credentials
-----------
  HF_TOKEN carregado automaticamente do .env na raiz do projeto.
"""

from __future__ import annotations

import json
import re
import sys
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

HF_BASE = "hf://datasets/hao-li/AIDev"

# Padrões de arquivo de teste TypeScript
TEST_FILE_RE = re.compile(r"\.(test|spec)\.(ts|tsx)$", re.IGNORECASE)

# Padrões que indicam uso de vitest no repositório
VITEST_INDICATORS = [
    re.compile(r'"vitest"', re.IGNORECASE),           # em package.json
    re.compile(r"from ['\"]vitest['\"]"),              # import em qualquer .ts
    re.compile(r"vitest\.config\.(ts|js|mts|mjs)"),   # arquivo de config
    re.compile(r'"test":\s*"vitest'),                  # script npm "test": "vitest"
]

# Poder estatístico: mínimo de PRs de IA com testes por repositório
MIN_AI_PRS_PER_REPO = 5

# Mapeamento de nomes de agentes (consistente com helper.py)
NAME_MAP = {
    "OpenAI_Codex": "OpenAI Codex",
    "Codex": "OpenAI Codex",
    "Claude_Code": "Claude Code",
    "Copilot": "GitHub Copilot",
    "GitHub Copilot": "GitHub Copilot",
    "Cursor": "Cursor",
    "Devin": "Devin",
    "Human": "Human",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_agent(name: str) -> str:
    return NAME_MAP.get(name, name)


def full_name_from_html_url(url: str | None) -> str:
    """
    Extrai 'owner/repo' de qualquer URL GitHub.
    Funciona para PRs:  https://github.com/owner/repo/pull/123
    E para API:         https://api.github.com/repos/owner/repo
    Retorna "" se o padrão não bater.
    """
    if not url or not isinstance(url, str):
        return ""
    m = re.search(r"github\.com/(?:repos/)?([^/]+/[^/]+)(?:/|$)", url)
    return m.group(1) if m else ""


def is_test_file(filename: str | None) -> bool:
    if not filename or not isinstance(filename, str):
        return False
    return bool(TEST_FILE_RE.search(filename))


def has_vitest_signal(patch: str | None) -> bool:
    """Retorna True se o patch contém algum indicador de vitest."""
    if not patch or not isinstance(patch, str):
        return False
    return any(p.search(patch) for p in VITEST_INDICATORS)


def extract_test_files(group: pd.DataFrame) -> list[str]:
    """Retorna lista de paths de arquivos de teste em um grupo de commit_details."""
    return group.loc[group["filename"].apply(is_test_file), "filename"].tolist()


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_pr_metadata() -> pd.DataFrame:
    """Retorna PRs de IA com tipo 'test' e merged_at preenchido."""
    print("  Carregando pull_request.parquet ...")
    pr_df = pd.read_parquet(
        f"{HF_BASE}/pull_request.parquet",
        columns=["id", "agent", "html_url", "merged_at"],
    )
    pr_df["agent"] = pr_df["agent"].map(normalise_agent).fillna(pr_df["agent"])
    pr_df["repo_full_name"] = pr_df["html_url"].apply(full_name_from_html_url)

    print("  Carregando pr_task_type.parquet ...")
    task_df = pd.read_parquet(f"{HF_BASE}/pr_task_type.parquet", columns=["id", "type"])

    merged = pr_df.merge(task_df, on="id", how="inner")
    test_merged = merged[
        (merged["type"] == "test") & (merged["merged_at"].notna())
    ].copy()

    print(f"  PRs de IA tipo 'test' e merged: {len(test_merged):,}")
    return test_merged


def load_human_pr_metadata() -> pd.DataFrame:
    """Retorna PRs humanas com tipo 'test' e merged_at preenchido."""
    print("  Carregando human_pull_request.parquet ...")
    # human_pull_request não tem repo_id — usa html_url para extrair repo_full_name
    human_pr_df = pd.read_parquet(
        f"{HF_BASE}/human_pull_request.parquet",
        columns=["id", "html_url", "merged_at"],
    )
    human_pr_df["agent"] = "Human"
    human_pr_df["repo_full_name"] = human_pr_df["html_url"].apply(full_name_from_html_url)

    print("  Carregando human_pr_task_type.parquet ...")
    human_task_df = pd.read_parquet(
        f"{HF_BASE}/human_pr_task_type.parquet", columns=["id", "type"]
    )

    merged = human_pr_df.merge(human_task_df, on="id", how="inner")
    test_merged = merged[
        (merged["type"] == "test") & (merged["merged_at"].notna())
    ].copy()

    print(f"  PRs humanas tipo 'test' e merged: {len(test_merged):,}")
    return test_merged


def load_repo_metadata() -> pd.DataFrame:
    """Retorna metadados dos repositórios (full_name, language)."""
    print("  Carregando repository.parquet ...")
    return pd.read_parquet(
        f"{HF_BASE}/repository.parquet",
        columns=["full_name", "language"],
    )


def load_commit_details(pr_ids: set[int]) -> pd.DataFrame:
    """Carrega commit_details filtrando apenas as PRs de interesse."""
    print("  Carregando pr_commit_details.parquet (711k linhas) ...")
    details_df = pd.read_parquet(
        f"{HF_BASE}/pr_commit_details.parquet",
        columns=["pr_id", "filename", "patch"],
    )
    filtered = details_df[details_df["pr_id"].isin(pr_ids)].copy()
    print(f"  Linhas após filtro por pr_ids: {len(filtered):,}")
    return filtered


def load_last_commit_sha(pr_ids: set[int]) -> pd.DataFrame:
    """
    Retorna o SHA do último commit de cada PR como proxy do merge_commit_sha.
    Usa pr_commits.parquet (88k linhas).
    """
    print("  Carregando pr_commits.parquet ...")
    commits_df = pd.read_parquet(f"{HF_BASE}/pr_commits.parquet")

    # Inspeciona colunas disponíveis na primeira execução
    sha_col = _detect_sha_column(commits_df)
    if sha_col is None:
        print("  AVISO: coluna SHA não encontrada em pr_commits — merge_commit_sha ficará vazio.")
        return pd.DataFrame(columns=["pr_id", "merge_commit_sha"])

    filtered = commits_df[commits_df["pr_id"].isin(pr_ids)][["pr_id", sha_col]].copy()

    # Mantém apenas o último commit por PR (maior posição = mais recente)
    # Se houver coluna de posição/ordem, usa; caso contrário, pega o último por índice
    order_col = _detect_order_column(commits_df)
    if order_col:
        filtered = filtered.sort_values(["pr_id", order_col])

    last_sha = (
        filtered.groupby("pr_id")[sha_col]
        .last()
        .reset_index()
        .rename(columns={sha_col: "merge_commit_sha"})
    )
    return last_sha


def _detect_sha_column(df: pd.DataFrame) -> str | None:
    """Detecta o nome da coluna de SHA entre variantes possíveis do schema."""
    for candidate in ["sha", "commit_sha", "commit_id", "oid"]:
        if candidate in df.columns:
            return candidate
    # Busca por qualquer coluna com 'sha' no nome
    sha_cols = [c for c in df.columns if "sha" in c.lower()]
    return sha_cols[0] if sha_cols else None


def _detect_order_column(df: pd.DataFrame) -> str | None:
    for candidate in ["position", "order", "index", "commit_order"]:
        if candidate in df.columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Processamento principal
# ---------------------------------------------------------------------------

def build_sample(
    pr_meta: pd.DataFrame,
    repo_df: pd.DataFrame,
    details_df: pd.DataFrame,
    sha_df: pd.DataFrame,
    is_human: bool,
    precomputed_vitest_repos: set[str] | None = None,
) -> pd.DataFrame:
    """
    Constrói as linhas de amostra para um grupo de PRs (IA ou humano).
    Retorna DataFrame com uma linha por PR elegível.

    precomputed_vitest_repos
        Quando fornecido (caso humano), pula a detecção de vitest e usa
        diretamente esses repos — necessário porque pr_commit_details.parquet
        só contém PRs de IA, não tendo entradas para PRs humanas.
    """
    label = "humanas" if is_human else "de IA"

    # Enriquece com linguagem do repo (join por full_name — funciona para IA e humano)
    pr_enriched = pr_meta.merge(
        repo_df[["full_name", "language"]],
        left_on="repo_full_name",
        right_on="full_name",
        how="left",
    )

    # Filtra apenas repos TypeScript (linguagem principal)
    ts_mask = pr_enriched["language"].str.lower().isin(["typescript", "javascript"])
    pr_ts = pr_enriched[ts_mask].copy()
    print(f"  PRs {label} em repos TypeScript/JavaScript: {len(pr_ts):,}")

    # ── Detecção de vitest ────────────────────────────────────────────────
    if precomputed_vitest_repos is not None:
        # Caso humano: repos vitest já identificados via PRs de IA do mesmo repo
        pr_ts["is_vitest"] = pr_ts["repo_full_name"].isin(precomputed_vitest_repos)
        print("  Vitest herdado dos repos de IA (sem re-detecção nos diffs)")
    else:
        # Caso IA: detecta pelo conteúdo dos diffs
        all_pr_ids = set(pr_ts["id"])
        details_sub = details_df[details_df["pr_id"].isin(all_pr_ids)]

        vitest_pr_ids: set[int] = set(
            details_sub[details_sub["patch"].apply(has_vitest_signal)]["pr_id"].unique()
        )
        pr_vitest_map = pr_ts[["id", "repo_full_name"]].copy()
        pr_vitest_map["has_vitest_signal"] = pr_vitest_map["id"].isin(vitest_pr_ids)
        detected_vitest_repos: set[str] = set(
            pr_vitest_map[pr_vitest_map["has_vitest_signal"]]["repo_full_name"].unique()
        )
        pr_ts["is_vitest"] = pr_ts["repo_full_name"].isin(detected_vitest_repos)

    vitest_prs = pr_ts[pr_ts["is_vitest"]].copy()
    print(f"  PRs {label} em repos com vitest detectado: {len(vitest_prs):,}")

    # ── Critério de poder estatístico (apenas IA) ─────────────────────────
    if not is_human:
        repo_counts = vitest_prs.groupby("repo_full_name")["id"].count()
        eligible_repos = set(repo_counts[repo_counts >= MIN_AI_PRS_PER_REPO].index)
        vitest_prs = vitest_prs[vitest_prs["repo_full_name"].isin(eligible_repos)].copy()
        print(
            f"  PRs de IA após critério >= {MIN_AI_PRS_PER_REPO} PRs/repo: {len(vitest_prs):,}"
        )

    # ── Extração de arquivos de teste ─────────────────────────────────────
    eligible_ids = set(vitest_prs["id"])
    details_eligible = details_df[details_df["pr_id"].isin(eligible_ids)]

    if not details_eligible.empty:
        test_files_per_pr: dict[int, list[str]] = (
            details_eligible[details_eligible["filename"].apply(is_test_file)]
            .groupby("pr_id")["filename"]
            .apply(list)
            .to_dict()
        )
        vitest_prs["test_files"] = vitest_prs["id"].map(test_files_per_pr)
        final = vitest_prs[
            vitest_prs["test_files"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ].copy()
    else:
        # PRs humanas: commit_details não disponível neste dataset —
        # inclui todas confiando na classificação LLM (type == "test")
        # A verificação de arquivos ocorrerá na Fase 2 após o clone
        vitest_prs["test_files"] = [[] for _ in range(len(vitest_prs))]
        final = vitest_prs.copy()
        print(f"  AVISO: commit_details indisponível para PRs humanas — incluindo todas ({len(final)})")

    print(f"  PRs {label} com >= 1 arquivo de teste: {len(final):,}")

    # ── Join com SHA do último commit ─────────────────────────────────────
    final = final.merge(
        sha_df.rename(columns={"pr_id": "id"}),
        on="id",
        how="left",
    )

    # ── Formatação final ──────────────────────────────────────────────────
    final["is_human"] = is_human
    final["test_files"] = final["test_files"].apply(json.dumps)

    return final[
        [
            "id",
            "agent",
            "full_name",
            "merge_commit_sha",
            "test_files",
            "is_vitest",
            "html_url",
            "merged_at",
            "is_human",
        ]
    ].rename(columns={"id": "pr_id", "full_name": "repo_full_name", "html_url": "pr_html_url"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_path = OUT_DIR / "rq1_sample.csv"

    if out_path.exists():
        print(f"rq1_sample.csv já existe em {out_path}.")
        print("Apague o arquivo para regenerar. Saindo.")
        sys.exit(0)

    print("=" * 65)
    print(" RQ1 — Fase 1: Seleção de Amostras")
    print("=" * 65)

    # ── 1. Metadados ──────────────────────────────────────────────────────
    print("\n[1/5] Carregando metadados de PRs e repositórios ...")
    ai_pr_meta = load_pr_metadata()
    human_pr_meta = load_human_pr_metadata()
    repo_df = load_repo_metadata()

    all_pr_ids = set(ai_pr_meta["id"]) | set(human_pr_meta["id"])

    # ── 2. Commit details (filtrado) ──────────────────────────────────────
    print("\n[2/5] Carregando commit details para PRs de teste ...")
    details_df = load_commit_details(all_pr_ids)

    # ── 3. SHAs dos commits ───────────────────────────────────────────────
    print("\n[3/5] Extraindo SHA do último commit por PR ...")
    sha_df = load_last_commit_sha(all_pr_ids)

    # ── 4. Construção das amostras ────────────────────────────────────────
    print("\n[4/5] Construindo amostra de PRs de IA ...")
    ai_sample = build_sample(
        ai_pr_meta, repo_df, details_df, sha_df, is_human=False
    )

    print("\n[4/5] Construindo amostra de PRs humanas (baseline) ...")
    # Repos vitest já identificados pela etapa de IA — reutiliza sem re-detectar,
    # pois pr_commit_details.parquet não contém entradas de PRs humanas
    ai_vitest_repos = set(ai_sample["repo_full_name"].unique())
    human_repos_ok = human_pr_meta[
        human_pr_meta["repo_full_name"].isin(ai_vitest_repos)
    ].copy()

    human_sample = build_sample(
        human_repos_ok, repo_df, details_df, sha_df, is_human=True,
        precomputed_vitest_repos=ai_vitest_repos,
    )

    # ── 5. Output ─────────────────────────────────────────────────────────
    print("\n[5/5] Consolidando e salvando ...")
    sample = pd.concat([ai_sample, human_sample], ignore_index=True)

    sample.to_csv(out_path, index=False)

    # ── Resumo ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" RESULTADO")
    print("=" * 65)

    ai_rows = sample[~sample["is_human"]]
    human_rows = sample[sample["is_human"]]

    print(f"\n  PRs de IA elegíveis    : {len(ai_rows):,}")
    print(f"  PRs humanas (baseline) : {len(human_rows):,}")
    print(f"  Total no sample        : {len(sample):,}")
    print(f"  Repositórios únicos    : {sample['repo_full_name'].nunique():,}")

    print("\n── Distribuição por agente (IA) ──────────────────────────────")
    agent_counts = (
        ai_rows.groupby("agent")
        .agg(prs=("pr_id", "count"), repos=("repo_full_name", "nunique"))
        .reset_index()
    )
    print(agent_counts.to_string(index=False))

    repos_sem_sha = sample["merge_commit_sha"].isna().sum()
    if repos_sem_sha:
        print(f"\n  AVISO: {repos_sem_sha} PRs sem SHA — verifique schema de pr_commits.parquet")

    print(f"\n  Salvo em: {out_path}")
    print("\nFase 1 concluída. Próximo passo: rq1_clone_repos.py")


if __name__ == "__main__":
    main()