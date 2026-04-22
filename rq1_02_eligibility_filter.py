#!/usr/bin/env python3
"""
RQ1 — Script 02: Filtro de Elegibilidade + Amostragem Estratificada
====================================================================
Aplica os 4 critérios do protocolo (seção 3.3.2) em nível de repositório,
produz rq1_eligible.csv + log de exclusões, e aplica amostragem
estratificada (200 PRs por agente, seed=42).

Critérios (inferidos do dataset — sem executar repositórios)
-------------------------------------------------------------
  C1. CI configurado    — repo possui .github/workflows/*.yml ou Jenkinsfile
                          detectados em qualquer PR do pr_commit_details
  C2. Testes passando   — proxy: PR merged + C1 satisfeito
  C3. Containerizável   — repo possui Dockerfile / docker-compose.yml /
                          requirements.txt / pyproject.toml / package.json
  C4. Linguagem         — JS/TS (mutation_tool=stryker) ou Python (mutmut)
                          já codificado nas linhas do rq1_sample.csv

Amostragem estratificada
------------------------
  - Por agente: 200 PRs máximo (seed=42), estratificado por task_type
  - Humanos: pool completo (raramente excede 400)
  - Apenas task_type in (feat, fix, refactor, test) segundo protocolo

Outputs
-------
  AIDev/rq1_eligible.csv       — PRs passando todos os critérios
  AIDev/rq1_eligibility_log.csv — todas as PRs com motivo de exclusão

Usage
-----
  python rq1_02_eligibility_filter.py
  python rq1_02_eligibility_filter.py --offline     # usar ZIP local
  python rq1_02_eligibility_filter.py --no-sample   # pular amostragem (depuração)
"""

from __future__ import annotations

import argparse
import io
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
ELIGIBLE_CSV = OUT_DIR / "rq1_eligible.csv"
LOG_CSV = OUT_DIR / "rq1_eligibility_log.csv"
DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
HF_BASE = "hf://datasets/hao-li/AIDev"

MAX_PER_AGENT = 200
RANDOM_SEED = 42
ELIGIBLE_TASK_TYPES = {"feat", "fix", "refactor", "test"}

CI_FILE_RE = (
    r"(\.github/workflows/[^/]+\.ya?ml$|^Jenkinsfile$|"
    r"\.circleci/config\.yml$|\.travis\.yml$|Makefile$)"
)
CONTAINER_FILE_RE = (
    r"(Dockerfile$|docker-compose\.ya?ml$|"
    r"requirements\.txt$|pyproject\.toml$|setup\.py$|"
    r"package\.json$|pom\.xml$|build\.gradle$|build\.gradle\.kts$)"
)

import re  # noqa: E402
CI_RE = re.compile(CI_FILE_RE, re.IGNORECASE)
CONTAINER_RE = re.compile(CONTAINER_FILE_RE, re.IGNORECASE)

sys.path.insert(0, str(ROOT_DIR))
from rq1_sample_selection import full_name_from_html_url  # noqa: E402


# ---------------------------------------------------------------------------
# Offline loader
# ---------------------------------------------------------------------------

class _ZipLoader:
    def __init__(self, zip_path: Path) -> None:
        self._zip_path = zip_path
        self._zf: zipfile.ZipFile | None = None
        self._names: list[str] = []

    def open(self) -> None:
        self._zf = zipfile.ZipFile(self._zip_path, "r")
        self._names = [m.filename for m in self._zf.infolist()]

    def close(self) -> None:
        if self._zf:
            self._zf.close()

    def read_parquet(self, basename: str, columns: list[str] | None = None) -> pd.DataFrame:
        matches = [n for n in self._names if n.endswith(basename)]
        if not matches:
            raise FileNotFoundError(f"{basename} não encontrado no ZIP")
        data = self._zf.read(matches[0])
        return pd.read_parquet(io.BytesIO(data), columns=columns)


def _load(filename: str, offline: bool, loader: _ZipLoader | None,
          columns: list[str] | None = None) -> pd.DataFrame:
    if offline and loader:
        return loader.read_parquet(filename, columns=columns)
    try:
        return pd.read_parquet(f"{HF_BASE}/{filename}", columns=columns)
    except Exception as exc:
        if loader:
            print(f"  [fallback] {exc} — usando ZIP ...")
            return loader.read_parquet(filename, columns=columns)
        raise


# ---------------------------------------------------------------------------
# Inferência de propriedades em nível de repositório
# ---------------------------------------------------------------------------

def build_repo_flags(offline: bool, loader: _ZipLoader | None) -> pd.DataFrame:
    """
    Varre pr_commit_details para detectar, por repositório, se:
      - tem CI configurado
      - é containerizável

    Retorna DataFrame: repo_full_name | has_ci | has_container
    """
    print("  Carregando pr_commit_details.parquet para detecção de CI/container ...")
    details = _load("pr_commit_details.parquet", offline, loader, columns=["pr_id", "filename"])
    print(f"  Linhas em pr_commit_details: {len(details):,}")

    # Precisamos mapear pr_id → repo_full_name
    print("  Carregando pull_request.parquet para mapeamento pr_id → repo ...")
    pr_df = _load("pull_request.parquet", offline, loader, columns=["id", "html_url"])
    pr_df["repo_full_name"] = pr_df["html_url"].apply(full_name_from_html_url)
    pr_to_repo = pr_df.set_index("id")["repo_full_name"].to_dict()

    details["repo_full_name"] = details["pr_id"].map(pr_to_repo)
    details = details.dropna(subset=["repo_full_name"])

    # Detectar CI e container por arquivo
    details["is_ci_file"] = details["filename"].apply(
        lambda f: bool(CI_RE.search(f)) if isinstance(f, str) else False
    )
    details["is_container_file"] = details["filename"].apply(
        lambda f: bool(CONTAINER_RE.search(f)) if isinstance(f, str) else False
    )

    ci_repos = set(details[details["is_ci_file"]]["repo_full_name"].unique())
    container_repos = set(details[details["is_container_file"]]["repo_full_name"].unique())

    print(f"  Repositórios com CI detectado:      {len(ci_repos):,}")
    print(f"  Repositórios containerizáveis:       {len(container_repos):,}")

    all_repos = set(pr_to_repo.values()) - {""}
    repo_flags = pd.DataFrame({"repo_full_name": sorted(all_repos)})
    repo_flags["has_ci"] = repo_flags["repo_full_name"].isin(ci_repos)
    repo_flags["has_container"] = repo_flags["repo_full_name"].isin(container_repos)

    return repo_flags


# ---------------------------------------------------------------------------
# Aplicação dos critérios
# ---------------------------------------------------------------------------

def apply_criteria(sample: pd.DataFrame, repo_flags: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de critério a cada PR e coluna 'eligible' (bool).
    Retorna DataFrame enriquecido para geração do log.
    """
    df = sample.copy()

    # Join com flags de repositório
    df = df.merge(repo_flags, on="repo_full_name", how="left")
    df["has_ci"] = df["has_ci"].fillna(False)
    df["has_container"] = df["has_container"].fillna(False)

    # C4: linguagem já codificada
    df["c4_language"] = df["mutation_tool"].isin(["stryker", "mutmut"])

    # C1: CI configurado
    df["c1_ci"] = df["has_ci"]

    # C2: proxy — PR merged + CI (merged_at já foi filtrado no sample_selection)
    df["c2_tests_pass"] = df["c1_ci"]

    # C3: containerizável
    df["c3_container"] = df["has_container"]

    # Elegível: todos os critérios satisfeitos
    df["eligible"] = df["c1_ci"] & df["c2_tests_pass"] & df["c3_container"] & df["c4_language"]

    # Motivo de exclusão (primeiro critério falho, da prioridade mais alta)
    def exclusion_reason(row: pd.Series) -> str:
        if not row["c4_language"]:
            return "linguagem_nao_suportada"
        if not row["c1_ci"]:
            return "sem_ci"
        if not row["c3_container"]:
            return "nao_containerizavel"
        return ""

    df["exclusion_reason"] = df.apply(exclusion_reason, axis=1)
    return df


# ---------------------------------------------------------------------------
# Amostragem estratificada
# ---------------------------------------------------------------------------

def stratified_sample(eligible: pd.DataFrame, max_per_agent: int, seed: int) -> pd.DataFrame:
    """
    Amostragem estratificada por agente × task_type.
    Humanos: incluir todos (pool pequeno).
    PRs com task_type desconhecido/inelegível: prioridade mais baixa.
    """
    human_prs = eligible[eligible["is_human"] == True].copy()  # noqa: E712
    ai_prs = eligible[eligible["is_human"] == False].copy()    # noqa: E712

    # Filtrar por task_type elegível (protocolo: feat/fix/refactor/test)
    ai_eligible_task = ai_prs[ai_prs["task_type"].isin(ELIGIBLE_TASK_TYPES)].copy()
    ai_other_task = ai_prs[~ai_prs["task_type"].isin(ELIGIBLE_TASK_TYPES)].copy()

    selected_rows = []

    for agent, group in ai_eligible_task.groupby("agent"):
        if len(group) <= max_per_agent:
            selected_rows.append(group)
            continue

        # Amostragem estratificada por task_type dentro do agente
        task_counts = group["task_type"].value_counts(normalize=True)
        sampled_parts = []
        remaining = max_per_agent

        for task_type, frac in task_counts.items():
            n = max(1, round(frac * max_per_agent))
            task_group = group[group["task_type"] == task_type]
            n = min(n, len(task_group), remaining)
            if n > 0:
                sampled_parts.append(task_group.sample(n, random_state=seed))
                remaining -= n
            if remaining <= 0:
                break

        agent_sample = pd.concat(sampled_parts, ignore_index=True)

        # Completar com outras task_types se ainda faltarem linhas
        if len(agent_sample) < max_per_agent and len(ai_other_task) > 0:
            filler = ai_other_task[ai_other_task["agent"] == agent]
            need = max_per_agent - len(agent_sample)
            if len(filler) > 0:
                agent_sample = pd.concat(
                    [agent_sample, filler.sample(min(need, len(filler)), random_state=seed)],
                    ignore_index=True,
                )

        selected_rows.append(agent_sample)

    sampled_ai = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    final = pd.concat([sampled_ai, human_prs], ignore_index=True)
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Filtro de elegibilidade RQ1")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument("--no-sample", action="store_true", help="Pular amostragem")
    args = parser.parse_args()

    print("=" * 65)
    print(" RQ1 — Script 02: Filtro de Elegibilidade")
    print("=" * 65)

    # ── Carregar amostra ───────────────────────────────────────────────────
    if not SAMPLE_CSV.exists():
        print(f"[ERRO] {SAMPLE_CSV} não encontrado. Execute os scripts anteriores.")
        raise SystemExit(1)

    sample = pd.read_csv(SAMPLE_CSV)
    print(f"\n  Total de linhas em rq1_sample.csv: {len(sample):,}")

    # Garantir colunas necessárias
    if "mutation_tool" not in sample.columns:
        sample["mutation_tool"] = "stryker"
    if "task_type" not in sample.columns:
        sample["task_type"] = "unknown"

    # ── Loader offline ─────────────────────────────────────────────────────
    loader: _ZipLoader | None = None
    zip_path = Path(args.zip)
    if args.offline and zip_path.exists():
        loader = _ZipLoader(zip_path)
        loader.open()

    try:
        # ── 1. Flags de repositório ────────────────────────────────────────
        print("\n[1/3] Detectando propriedades de CI/container por repositório ...")
        repo_flags = build_repo_flags(args.offline, loader)

        # ── 2. Aplicar critérios ───────────────────────────────────────────
        print("\n[2/3] Aplicando critérios de elegibilidade ...")
        annotated = apply_criteria(sample, repo_flags)

        # Salvar log completo
        annotated.to_csv(LOG_CSV, index=False)
        print(f"  Log salvo: {LOG_CSV}")

        # Resumo por critério
        print("\n  Critérios (% de PRs satisfazendo):")
        for col, label in [
            ("c4_language", "C4 - Linguagem suportada"),
            ("c1_ci",       "C1 - CI configurado"),
            ("c3_container","C3 - Containerizável"),
            ("eligible",    "TODOS (elegível)"),
        ]:
            pct = annotated[col].mean() * 100
            n = annotated[col].sum()
            print(f"    {label:<35}: {n:>5} ({pct:.1f}%)")

        eligible = annotated[annotated["eligible"]].copy()
        print(f"\n  PRs elegíveis (antes da amostragem): {len(eligible):,}")

        if len(eligible) == 0:
            print("\n  [AVISO] Nenhuma PR elegível encontrada.")
            print("  Considere revisar os critérios ou executar com --no-sample.")
            return

        # Distribuição por agente antes da amostragem
        print("\n  Elegíveis por agente:")
        print(eligible.groupby("agent")["pr_id"].count().to_string())

        # ── 3. Amostragem estratificada ────────────────────────────────────
        if args.no_sample:
            final = eligible
            print("\n[3/3] Amostragem pulada (--no-sample)")
        else:
            print(f"\n[3/3] Amostragem estratificada (máx {MAX_PER_AGENT}/agente, seed={RANDOM_SEED}) ...")
            final = stratified_sample(eligible, MAX_PER_AGENT, RANDOM_SEED)

        final.to_csv(ELIGIBLE_CSV, index=False)
        print(f"\n  Salvo: {ELIGIBLE_CSV}")

        # Resumo final
        print(f"\n{'='*65}")
        print(" RESUMO — rq1_eligible.csv")
        print(f"{'='*65}")
        print(f"  Total elegíveis: {len(final):,}")
        print(f"  PRs de IA:       {len(final[final['is_human']==False]):,}")  # noqa: E712
        print(f"  PRs humanas:     {len(final[final['is_human']==True]):,}")   # noqa: E712

        print("\n  Por agente:")
        print(final.groupby("agent")["pr_id"].count().to_string())

        print("\n  Por ferramenta de mutação:")
        print(final.groupby("mutation_tool")["pr_id"].count().to_string())

        print("\n  Por task_type:")
        print(final.groupby("task_type")["pr_id"].count().to_string())

        missing_sha = final["merge_commit_sha"].isna().sum()
        if missing_sha:
            print(f"\n  [AVISO] {missing_sha} PRs sem merge_commit_sha (clone pode falhar)")

    finally:
        if loader:
            loader.close()

    print("\nScript 02 concluído. Próximo passo:")
    print("  python rq1_clone_repos.py --csv AIDev/rq1_eligible.csv")


if __name__ == "__main__":
    main()
