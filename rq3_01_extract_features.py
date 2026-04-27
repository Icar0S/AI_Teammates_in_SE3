#!/usr/bin/env python3
"""
RQ3 — Script 01: Extract PSQI Features from Performance Diffs
=============================================================
Para cada arquivo de performance no corpus, extrai as 5 dimensões do
Performance Script Quality Index (PSQI) a partir das linhas adicionadas
(+) no unified diff do patch.

Dimensões PSQI (escala 0–2, protocolo §3.5.3):
  dim_load_type   — variedade de padrões de carga modelados
  dim_think_time  — simulação de tempo entre ações
  dim_payload     — variação nos dados de entrada
  dim_negative    — cenários de erro e boundary
  dim_sla         — thresholds e assertions de SLA

Também detecta a categoria de ferramenta por arquivo:
  k6, locust, jmeter, gatling, pytest_benchmark, generic_perf

Inputs
------
  AIDev/rq3_corpus.parquet          — produzido por rq3_00_build_corpus.py
  OFFICIALDATASET.zip (ou HF)       — pr_commit_details.parquet (com patch)

Outputs
-------
  AIDev/rq3_features_raw.parquet    — (pr_id, filename, tool_category,
                                       dim_load_type, dim_think_time,
                                       dim_payload, dim_negative, dim_sla)
  AIDev/rq3_extract_log.csv         — estatísticas de processamento

Usage
-----
  python rq3_01_extract_features.py
  python rq3_01_extract_features.py --offline
  python rq3_01_extract_features.py --offline --resume
  python rq3_01_extract_features.py --offline --limit 5000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    ZipLoader,
    load_parquet,
)

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
CORPUS_PARQUET = OUT_DIR / "rq3_corpus.parquet"
FEATURES_PARQUET = OUT_DIR / "rq3_features_raw.parquet"
LOG_CSV = OUT_DIR / "rq3_extract_log.csv"

PSQI_DIMS = ["dim_load_type", "dim_think_time", "dim_payload",
             "dim_negative", "dim_sla"]

# ---------------------------------------------------------------------------
# Performance file & content patterns
# ---------------------------------------------------------------------------

_FNAME_PATTERNS: list[re.Pattern] = [
    re.compile(r"\.k6\.(js|ts)$", re.IGNORECASE),
    re.compile(r"(^|[/_-])k6[/_-]", re.IGNORECASE),
    re.compile(r"[/_-]k6\.(js|ts)$", re.IGNORECASE),
    re.compile(r"\.jmx$", re.IGNORECASE),
    re.compile(r"gatling", re.IGNORECASE),
    re.compile(r"locustfile\.py$", re.IGNORECASE),
    re.compile(r"[/_-]locust[/_-]", re.IGNORECASE),
    re.compile(
        r"(load[_-]?test|stress[_-]?test|perf[_-]?test|benchmark)",
        re.IGNORECASE,
    ),
    re.compile(r"[/_-](perf|load|stress|benchmark)[/_-]", re.IGNORECASE),
    re.compile(r"(performance|throughput)[_-]test", re.IGNORECASE),
]

_K6_CONTENT: list[re.Pattern] = [
    re.compile(r"\bimport\s+.*from\s+['\"]k6['\"]", re.IGNORECASE),
    re.compile(r"\bimport\s+.*from\s+['\"]k6/", re.IGNORECASE),
    re.compile(r"from\s+['\"]k6/http['\"]", re.IGNORECASE),
    re.compile(r"\bhttp\.get\b.*\bcheck\b", re.IGNORECASE),
    re.compile(r"\bramping_arrival_rate\b|\bramping_vus\b", re.IGNORECASE),
    re.compile(r"\bthresholds\s*:", re.IGNORECASE),
]


def _matches_perf(fname: str | None, patch: str | None) -> bool:
    if fname and any(p.search(fname) for p in _FNAME_PATTERNS):
        return True
    if patch and any(p.search(patch) for p in _K6_CONTENT):
        return True
    return False


def _tool_category(fname: str, added_text: str) -> str:
    f = fname.lower()
    if re.search(r"\.k6\.(js|ts)|[/_-]k6[/_-]|[/_-]k6\.(js|ts)", f):
        return "k6"
    if re.search(r"from\s+['\"]k6", added_text, re.IGNORECASE):
        return "k6"
    if re.search(r"\.jmx$", f):
        return "jmeter"
    if re.search(r"gatling", f):
        return "gatling"
    if re.search(r"locustfile\.py|[/_-]locust[/_-]", f):
        return "locust"
    # pytest-benchmark: check content
    if re.search(
        r"@pytest\.mark\.benchmark|benchmark\.pedantic\(|def test_.*benchmark",
        added_text, re.IGNORECASE,
    ):
        return "pytest_benchmark"
    # Locust from content
    if re.search(r"\bHttpUser\b|\b@task\b|wait_time\s*=\s*between\(", added_text):
        return "locust"
    return "generic_perf"


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------

def _extract_added_lines(patch: str) -> list[str]:
    """Return content of '+' lines from a unified diff, excluding '+++' headers."""
    lines = []
    for line in patch.splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return lines


# ---------------------------------------------------------------------------
# PSQI dimension scorers (return int 0, 1, or 2)
# ---------------------------------------------------------------------------

# -- Load Type (dim_load_type) --
# 0 = steady-state only (or nothing), 1 = single non-steady pattern,
# 2 = combination of ≥2 distinct patterns
_LOAD_PATTERNS = {
    "ramp": re.compile(
        r"\b(ramp[-_]?up|ramping_vus|ramping_arrival_rate|stages\s*:)",
        re.IGNORECASE,
    ),
    "spike": re.compile(r"\bspike\b", re.IGNORECASE),
    "soak": re.compile(r"\b(soak|endurance)\b", re.IGNORECASE),
    "stress": re.compile(
        r"\b(stress[_-]?test|high[_-]?load|breaking[_-]?point)\b",
        re.IGNORECASE,
    ),
    "scenario": re.compile(
        r"\b(scenario|executor|constant_arrival_rate)\b", re.IGNORECASE
    ),
}


def _score_load_type(text: str) -> int:
    matched = [k for k, pat in _LOAD_PATTERNS.items() if pat.search(text)]
    if len(matched) >= 2:
        return 2
    if len(matched) == 1:
        return 1
    return 0


# -- Think Time (dim_think_time) --
_THINK_FIXED = re.compile(
    r"\b(sleep\s*\(\s*\d+|wait_time\s*=\s*\d+|think_time\s*=\s*\d+"
    r"|time\.sleep\s*\(\s*\d+)",
    re.IGNORECASE,
)
_THINK_VARIABLE = re.compile(
    r"\b(between\s*\(|uniform\s*\(|normal\s*\(|random\s*\(|"
    r"wait_time\s*=\s*between|constant_pacing|sleep_time_func)",
    re.IGNORECASE,
)


def _score_think_time(text: str) -> int:
    if _THINK_VARIABLE.search(text):
        return 2
    if _THINK_FIXED.search(text):
        return 1
    return 0


# -- Payload Variation (dim_payload) --
_PAYLOAD_EXTERNAL = re.compile(
    r"\b(SharedArray\s*\(|open\s*\([^)]*csv|pd\.read_csv|json\.load\s*\("
    r"|data_file|csv_file|fixture_file|params_file)",
    re.IGNORECASE,
)
_PAYLOAD_PARAM = re.compile(
    r"\b(formData|params\s*=\s*\{|body\s*=\s*\w+|payload\s*=\s*\w+"
    r"|json\s*=\s*\w+|data\s*=\s*\w+)",
    re.IGNORECASE,
)


def _score_payload(text: str) -> int:
    if _PAYLOAD_EXTERNAL.search(text):
        return 2
    if _PAYLOAD_PARAM.search(text):
        return 1
    return 0


# -- Negative Scenarios (dim_negative) --
_NEG_PATTERNS = [
    re.compile(r"\b(status\s*==\s*[45]\d{2}|check\s*\([^)]*4\d{2}"
               r"|check\s*\([^)]*5\d{2})", re.IGNORECASE),
    re.compile(r"\b(expect_error|assert_raises|with_exception"
               r"|assertRaises|pytest\.raises)", re.IGNORECASE),
    re.compile(r"\b(boundary|edge[_-]?case|invalid[_-]?input"
               r"|out[_-]?of[_-]?range|overflow)", re.IGNORECASE),
    re.compile(r"\b(rate[_-]?limit|throttl|429|503|timeout_error"
               r"|connection_refused)", re.IGNORECASE),
]


def _score_negative(text: str) -> int:
    hits = sum(1 for p in _NEG_PATTERNS if p.search(text))
    if hits >= 2:
        return 2
    if hits == 1:
        return 1
    return 0


# -- SLA Assertions (dim_sla) --
_SLA_LINKED = re.compile(
    r"\b(slo|sla|budget|error_rate\s*[<>]=?\s*[\d.]+|"
    r"p\s*\(\s*9[05]\s*\)|p99|p95|p90|percentile\s*\(\s*9)",
    re.IGNORECASE,
)
_SLA_THRESHOLD = re.compile(
    r"\b(thresholds\s*:|threshold\s*=|latency\s*[<>]=?"
    r"|response_time\s*[<>]=?|duration\s*[<>]=?\s*\d)",
    re.IGNORECASE,
)


def _score_sla(text: str) -> int:
    if _SLA_LINKED.search(text):
        return 2
    if _SLA_THRESHOLD.search(text):
        return 1
    return 0


def score_psqi(added_text: str) -> dict[str, int]:
    return {
        "dim_load_type": _score_load_type(added_text),
        "dim_think_time": _score_think_time(added_text),
        "dim_payload": _score_payload(added_text),
        "dim_negative": _score_negative(added_text),
        "dim_sla": _score_sla(added_text),
    }


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

def process_corpus(
    offline: bool,
    loader: ZipLoader | None,
    limit: int | None,
    resume: bool,
) -> None:
    print("Loading rq3_corpus.parquet...")
    corpus = pd.read_parquet(CORPUS_PARQUET)
    corpus_pr_ids = set(corpus["pr_id"].unique())
    print(f"  Corpus: {len(corpus_pr_ids):,} distinct PRs")

    already_done: set[tuple] = set()
    if resume and FEATURES_PARQUET.exists():
        existing = pd.read_parquet(
            FEATURES_PARQUET, columns=["pr_id", "filename"]
        )
        already_done = set(zip(existing["pr_id"], existing["filename"]))
        print(f"  Resuming: {len(already_done):,} (pr_id, filename) already done.")

    print("Loading pr_commit_details.parquet (patch column)...")
    details = load_parquet(
        "pr_commit_details.parquet",
        offline,
        loader,
        columns=["pr_id", "filename", "patch"],
    )
    if limit:
        details = details.head(limit)

    # Filter to corpus PRs only
    details = details[details["pr_id"].isin(corpus_pr_ids)].copy()
    print(f"  Rows for corpus PRs: {len(details):,}")

    # Keep only performance-matching rows
    details["_fname"] = details["filename"].fillna("").astype(str)
    details["_patch_str"] = details["patch"].fillna("").astype(str)
    details["_is_perf"] = details.apply(
        lambda r: _matches_perf(r["_fname"], r["_patch_str"]), axis=1
    )
    perf_rows = details[details["_is_perf"]].copy()
    print(f"  Performance-matching rows: {len(perf_rows):,}")

    records = []
    n_skipped = n_null = n_error = 0

    for _, row in perf_rows.iterrows():
        pr_id = row["pr_id"]
        filename = str(row["filename"])
        patch = row["patch"]

        if resume and (pr_id, filename) in already_done:
            n_skipped += 1
            continue

        if not isinstance(patch, str) or not patch.strip():
            n_null += 1
            continue

        try:
            added_lines = _extract_added_lines(patch)
            added_text = "\n".join(added_lines)
            scores = score_psqi(added_text)
            tool_cat = _tool_category(filename, added_text)
        except Exception as exc:
            n_error += 1
            print(f"  Error on pr={pr_id} file={filename}: {exc}")
            scores = {d: 0 for d in PSQI_DIMS}
            tool_cat = "generic_perf"

        record: dict = {"pr_id": pr_id, "filename": filename,
                        "tool_category": tool_cat}
        record.update(scores)
        records.append(record)

    print(
        f"\nProcessed: {len(records):,} | Skipped: {n_skipped:,} | "
        f"Null patches: {n_null:,} | Errors: {n_error:,}"
    )

    result_df = pd.DataFrame(records)

    if resume and FEATURES_PARQUET.exists():
        old = pd.read_parquet(FEATURES_PARQUET)
        result_df = pd.concat([old, result_df], ignore_index=True)

    result_df.to_parquet(FEATURES_PARQUET, index=False)
    print(f"Saved {len(result_df):,} rows → {FEATURES_PARQUET}")

    log = pd.DataFrame([{
        "corpus_prs": len(corpus_pr_ids),
        "commit_detail_rows_for_corpus": len(details),
        "perf_matching_rows": len(perf_rows),
        "processed": len(records),
        "skipped_resume": n_skipped,
        "null_patch": n_null,
        "errors": n_error,
    }])
    log.to_csv(LOG_CSV, index=False)
    print(f"Log → {LOG_CSV}")

    if not result_df.empty:
        print("\nTool category distribution:")
        print(result_df["tool_category"].value_counts().to_string())
        print("\nMean PSQI dimension scores (0–2):")
        for dim in PSQI_DIMS:
            if dim in result_df.columns:
                print(f"  {dim:<20} {result_df[dim].mean():.3f}")


def main(
    offline: bool,
    zip_path: Path,
    limit: int | None,
    resume: bool,
) -> None:
    if not CORPUS_PARQUET.exists():
        print(
            f"ERROR: {CORPUS_PARQUET} not found. "
            "Run rq3_00_build_corpus.py first."
        )
        sys.exit(1)

    loader: ZipLoader | None = None
    if offline or zip_path.exists():
        loader = ZipLoader(zip_path)
        loader.open()

    try:
        process_corpus(offline, loader, limit, resume)
    finally:
        if loader:
            loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3 PSQI feature extractor")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--zip", type=Path, default=DEFAULT_ZIP, metavar="PATH"
    )
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-processed (pr_id, filename) pairs",
    )
    args = parser.parse_args()
    main(
        offline=args.offline,
        zip_path=args.zip,
        limit=args.limit,
        resume=args.resume,
    )
