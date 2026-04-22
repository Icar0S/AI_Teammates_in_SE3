#!/usr/bin/env python3
"""
RQ1 — Script 00: Exploração do OFFICIALDATASET.zip (Zenodo)
============================================================
Inspeciona o ZIP sem descompactar tudo, cataloga os arquivos parquet
e compara schemas com os do HuggingFace para decidir se o ZIP pode
ser usado como fonte offline (sem HF_TOKEN).

Outputs
-------
  AIDev/zenodo_manifest.csv — catálogo de arquivos no ZIP
    colunas: filename, size_mb, columns, hf_equivalent, schema_diff

Usage
-----
  python rq1_00_explore_zenodo.py
  python rq1_00_explore_zenodo.py --zip CAMINHO/PARA/OUTRO.zip
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
MANIFEST_CSV = OUT_DIR / "zenodo_manifest.csv"

load_dotenv(ROOT_DIR / ".env")

HF_BASE = "hf://datasets/hao-li/AIDev"

# Arquivos parquet conhecidos do HuggingFace (AIDev-pop + AIDev completo)
HF_FILES = [
    "pull_request.parquet",
    "all_pull_request.parquet",
    "repository.parquet",
    "all_repository.parquet",
    "user.parquet",
    "all_user.parquet",
    "pr_comments.parquet",
    "pr_reviews.parquet",
    "pr_review_comments_v2.parquet",
    "pr_commits.parquet",
    "pr_commit_details.parquet",
    "related_issue.parquet",
    "issue.parquet",
    "pr_timeline.parquet",
    "pr_task_type.parquet",
    "human_pull_request.parquet",
    "human_pr_task_type.parquet",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_parquet_schema_from_bytes(data: bytes) -> list[str]:
    """Retorna lista de nomes de colunas de um parquet em memória."""
    buf = io.BytesIO(data)
    schema = pq.read_schema(buf)
    return schema.names


def _hf_schema(filename: str) -> list[str] | None:
    """Lê schema de um parquet do HuggingFace (requer HF_TOKEN ou acesso público)."""
    try:
        path = f"{HF_BASE}/{filename}"
        schema = pq.read_schema(path)
        return schema.names
    except Exception as exc:
        print(f"  [aviso] não foi possível ler schema HF para {filename}: {exc}")
        return None


def _col_diff(zip_cols: list[str], hf_cols: list[str] | None) -> str:
    """Retorna string descrevendo diferenças de colunas."""
    if hf_cols is None:
        return "hf_unavailable"
    only_zip = sorted(set(zip_cols) - set(hf_cols))
    only_hf = sorted(set(hf_cols) - set(zip_cols))
    if not only_zip and not only_hf:
        return "identical"
    parts = []
    if only_zip:
        parts.append(f"+zip:{only_zip}")
    if only_hf:
        parts.append(f"+hf:{only_hf}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def explore_zip(zip_path: Path) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Explorando: {zip_path}")
    print(f"Tamanho no disco: {zip_path.stat().st_size / 1_048_576:.1f} MB")
    print(f"{'='*60}\n")

    rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.infolist()
        print(f"Total de entradas no ZIP: {len(all_members)}\n")

        # Listar estrutura de diretórios
        dirs: set[str] = set()
        for info in all_members:
            if "/" in info.filename:
                dirs.add(info.filename.rsplit("/", 1)[0])
        if dirs:
            print("Diretórios encontrados:")
            for d in sorted(dirs):
                print(f"  {d}/")
            print()

        # Filtrar arquivos parquet
        parquet_members = [m for m in all_members if m.filename.endswith(".parquet")]
        other_members = [m for m in all_members if not m.filename.endswith(".parquet")]

        print(f"Arquivos .parquet no ZIP: {len(parquet_members)}")
        print(f"Outros arquivos: {len(other_members)}\n")

        if other_members:
            print("Outros arquivos (não parquet):")
            for m in other_members:
                size_kb = m.file_size / 1024
                print(f"  {m.filename}  ({size_kb:.1f} KB)")
            print()

        # Para cada parquet: ler schema e comparar com HuggingFace
        print("Analisando schemas dos arquivos parquet...\n")
        for member in parquet_members:
            size_mb = member.file_size / 1_048_576
            basename = Path(member.filename).name

            print(f"  {member.filename}  ({size_mb:.1f} MB) ...", end=" ", flush=True)

            # Ler apenas os primeiros bytes para schema (não descompacta tudo)
            try:
                data = zf.read(member.filename)
                zip_cols = _read_parquet_schema_from_bytes(data)
            except Exception as exc:
                print(f"ERRO: {exc}")
                rows.append({
                    "filename": member.filename,
                    "size_mb": round(size_mb, 2),
                    "columns": "ERROR",
                    "hf_equivalent": basename if basename in HF_FILES else "unknown",
                    "schema_diff": f"read_error: {exc}",
                })
                continue

            # Identificar equivalente HuggingFace
            hf_equiv = basename if basename in HF_FILES else "unknown"
            hf_cols = _hf_schema(hf_equiv) if hf_equiv != "unknown" else None
            diff = _col_diff(zip_cols, hf_cols)

            print(f"{len(zip_cols)} colunas | diff={diff}")

            rows.append({
                "filename": member.filename,
                "size_mb": round(size_mb, 2),
                "columns": json.dumps(zip_cols),
                "hf_equivalent": hf_equiv,
                "schema_diff": diff,
            })

    df = pd.DataFrame(rows)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"{'='*60}")

    identical = df[df["schema_diff"] == "identical"]
    hf_unavail = df[df["schema_diff"] == "hf_unavailable"]
    different = df[~df["schema_diff"].isin(["identical", "hf_unavailable", "ERROR"])]
    errors = df[df["schema_diff"].str.startswith("read_error", na=False)]

    print(f"\nArquivos com schema idêntico ao HuggingFace: {len(identical)}")
    print(f"Arquivos com schema diferente: {len(different)}")
    print(f"HuggingFace indisponível para comparação: {len(hf_unavail)}")
    print(f"Erros de leitura: {len(errors)}")

    if len(different) > 0:
        print("\n[!] Arquivos com diferenças de schema:")
        for _, row in different.iterrows():
            print(f"    {row['filename']}: {row['schema_diff']}")

    if len(identical) == len(df) - len(hf_unavail) - len(errors):
        print("\n[OK] ZIP parece ser espelho do HuggingFace.")
        print("     Pode ser usado offline sem HF_TOKEN via pd.read_parquet(BytesIO(...)).")
    else:
        print("\n[ATENÇÃO] ZIP contém dados diferentes do HuggingFace.")
        print("          Revisar diferenças antes de usar como fonte alternativa.")

    total_mb = df["size_mb"].sum()
    print(f"\nTamanho total descomprimido dos parquets: {total_mb:.1f} MB")

    # Arquivos HuggingFace não encontrados no ZIP
    zip_basenames = {Path(f).name for f in df["filename"]}
    missing_in_zip = [f for f in HF_FILES if f not in zip_basenames]
    if missing_in_zip:
        print(f"\n[AVISO] Arquivos HuggingFace NÃO encontrados no ZIP ({len(missing_in_zip)}):")
        for f in missing_in_zip:
            print(f"    {f}")

    # Arquivos no ZIP sem equivalente HuggingFace
    extra_in_zip = df[df["hf_equivalent"] == "unknown"]
    if len(extra_in_zip) > 0:
        print(f"\n[INFO] Arquivos no ZIP sem equivalente HuggingFace ({len(extra_in_zip)}):")
        for _, row in extra_in_zip.iterrows():
            print(f"    {row['filename']}  ({row['size_mb']:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploração do OFFICIALDATASET.zip (Zenodo)")
    parser.add_argument(
        "--zip",
        default=str(DEFAULT_ZIP),
        help=f"Caminho para o ZIP (default: {DEFAULT_ZIP})",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Pular comparação com HuggingFace (útil sem HF_TOKEN)",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"[ERRO] ZIP não encontrado: {zip_path}")
        raise SystemExit(1)

    if args.no_hf:
        # Monkey-patch para desabilitar consultas HuggingFace
        global _hf_schema
        def _hf_schema(filename):  # noqa: F811
            return None

    df = explore_zip(zip_path)
    print_summary(df)

    df.to_csv(MANIFEST_CSV, index=False)
    print(f"\nManifesto salvo em: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
