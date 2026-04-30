#!/usr/bin/env python3
"""
RQ2 — Script 01: Test Smell Detection
======================================
Detecta 10 test smells em patches (unified diff) de arquivos de teste,
sem necessidade de clonar repositórios. Análise puramente estática baseada
nas linhas adicionadas (+) de cada diff.

Catálogo (Palomba / Bavota / Verdecchia):
  AssertionRoulette, EagerTest, MagicNumberTest, UnknownTest,
  RedundantAssertion, DuplicateAssert, VerboseTest,
  MissingExceptionTest, ResourceOptimism, GeneralFixture

Inputs
------
  AIDev/rq2_corpus.parquet — produzido por rq2_00_build_corpus.py
  OFFICIALDATASET.zip (ou HuggingFace) — pr_commit_details.parquet

Outputs
-------
  AIDev/rq2_smells_raw.parquet — (pr_id, commit_sha, filename, language,
                                   AssertionRoulette, ..., [10 colunas])
  AIDev/rq2_detect_log.csv — linhas processadas, erros, pulados

Usage
-----
  python rq2_01_detect_smells.py
  python rq2_01_detect_smells.py --offline
  python rq2_01_detect_smells.py --offline --resume
  python rq2_01_detect_smells.py --offline --limit 1000
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
    SMELL_CATALOG,
    TEST_FILE_PATTERNS,
    ZipLoader,
    load_parquet,
)

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_ZIP = ROOT_DIR / "OFFICIALDATASET.zip"
CORPUS_PARQUET = OUT_DIR / "rq2_corpus.parquet"
SMELLS_PARQUET = OUT_DIR / "rq2_smells_raw.parquet"
LOG_CSV = OUT_DIR / "rq2_detect_log.csv"

SMELL_NAMES = list(SMELL_CATALOG.keys())

# ---------------------------------------------------------------------------
# Patch parsing utilities
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


def _split_into_test_functions(
    added_lines: list[str],
    lang: str,
) -> list[tuple[str, list[str]]]:
    """
    Split added lines into per-function blocks.
    Returns list of (func_name, lines_in_block).
    Heuristic: split on 'def test_' (Python) or 'it(/'test(' (JS/TS).
    """
    if lang == "python":
        boundary = re.compile(r"^\s*def\s+(test_\w+|setUp|tearDown)\s*\(")
    else:
        boundary = re.compile(
            r"(^\s*(it|test|describe|beforeEach|afterEach)\s*[\(\'])"
        )

    blocks: list[tuple[str, list[str]]] = []
    current_name = "__preamble__"
    current: list[str] = []

    for line in added_lines:
        m = boundary.match(line)
        if m:
            if current:
                blocks.append((current_name, current))
            current_name = line.strip()[:60]
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append((current_name, current))

    return blocks


def _detect_language(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "py":
        return "python"
    if ext in ("js", "jsx", "ts", "tsx"):
        return "js"
    if ext == "java":
        return "java"
    return "other"


# ---------------------------------------------------------------------------
# Smell detectors
# ---------------------------------------------------------------------------

_RE_PY_ASSERT = re.compile(r"\b(assert|self\.assert\w+)\b")
_RE_PY_ASSERT_MSG = re.compile(
    r"self\.assert\w+\([^,]+,[^)]+\)|assert\s+\w.*,\s*['\"]"
)
_RE_PY_METHOD_CALL = re.compile(r"\b(\w+)\s*\(")
_ASSERT_METHODS = {
    "assert", "assertEqual", "assertNotEqual", "assertTrue", "assertFalse",
    "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone",
    "assertIn", "assertNotIn", "assertRaises", "assertRaisesRegex",
    "assertWarns", "assertGreater", "assertGreaterEqual",
    "assertLess", "assertLessEqual", "assertAlmostEqual",
}
_PY_BUILTINS = {
    "print", "len", "range", "str", "int", "float", "list", "dict",
    "set", "tuple", "type", "isinstance", "hasattr", "getattr",
    "setattr", "open", "super", "self", "cls",
}
_RE_PY_MAGIC = re.compile(r"\b(assert|self\.assert\w+).*\b([2-9]\d{1,}|\d{3,})\b")
_RE_PY_REDUNDANT = re.compile(
    r"assert\s+True\b|self\.assertTrue\(\s*True\s*\)"
    r"|self\.assertEqual\(([^,]+),\s*\1\s*\)"
)
_RE_PY_RESOURCE = re.compile(r"\b(open\s*\(|requests\.|socket\.|urllib\.)")
_RE_PY_TRY = re.compile(r"^\s*try\s*:")
_RE_PY_EXCEPT = re.compile(r"^\s*(except|with\s+pytest\.raises)")

_RE_JS_EXPECT = re.compile(r"\bexpect\s*\(")
_RE_JS_MAGIC = re.compile(r"\.(toBe|toEqual)\s*\(\s*([2-9]\d{1,}|\d{3,})\s*\)")
_RE_JS_REDUNDANT = re.compile(
    r"\.toBe\s*\(\s*true\s*\)|expect\s*\(\s*(\w+)\s*\)\.toBe\s*\(\s*\1\s*\)"
)
_RE_JS_RESOURCE = re.compile(r"\b(fetch\s*\(|fs\.|axios\.|http\.)")
_RE_JS_CATCH = re.compile(r"\.catch\s*\(|try\s*\{")
_RE_JS_EXCEPTION = re.compile(r"\.(toThrow|toThrowError|rejects)\b")


def _detect_python(func_name: str, lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {s: 0 for s in SMELL_NAMES}

    assert_lines = [l for l in lines if _RE_PY_ASSERT.search(l)]
    non_empty = [l for l in lines if l.strip()]

    # AssertionRoulette: ≥3 asserts without descriptive message
    if len(assert_lines) >= 3:
        has_msg = any(_RE_PY_ASSERT_MSG.search(l) for l in assert_lines)
        if not has_msg:
            counts["AssertionRoulette"] = 1

    # EagerTest: ≥4 distinct non-assert/builtin method calls
    if "test_" in func_name.lower() or "__preamble__" not in func_name:
        all_calls = {
            m.group(1) for l in lines for m in [_RE_PY_METHOD_CALL.search(l)] if m
        }
        sut_calls = all_calls - _ASSERT_METHODS - _PY_BUILTINS
        if len(sut_calls) >= 4:
            counts["EagerTest"] = 1

    # MagicNumberTest
    magic_hits = sum(1 for l in assert_lines if _RE_PY_MAGIC.search(l))
    counts["MagicNumberTest"] = magic_hits

    # UnknownTest
    if "def test_" in func_name and not assert_lines:
        counts["UnknownTest"] = 1

    # RedundantAssertion
    redundant = sum(1 for l in lines if _RE_PY_REDUNDANT.search(l))
    counts["RedundantAssertion"] = redundant

    # DuplicateAssert
    normalized = [re.sub(r"\s+", " ", l.strip()) for l in assert_lines]
    seen: set[str] = set()
    dupes = 0
    for n in normalized:
        if n in seen:
            dupes += 1
        seen.add(n)
    counts["DuplicateAssert"] = dupes

    # VerboseTest
    if len(non_empty) > 20 and "test_" in func_name.lower():
        counts["VerboseTest"] = 1

    # MissingExceptionTest
    name_lower = func_name.lower()
    if any(k in name_lower for k in ("raise", "error", "exception", "invalid")):
        has_exc = any("assertraises" in l.lower() or "pytest.raises" in l.lower()
                      for l in lines)
        if not has_exc:
            counts["MissingExceptionTest"] = 1

    # ResourceOptimism
    resource_lines = [l for l in lines if _RE_PY_RESOURCE.search(l)]
    if resource_lines:
        has_guard = any(
            _RE_PY_TRY.match(l) or _RE_PY_EXCEPT.match(l) for l in lines
        )
        if not has_guard:
            counts["ResourceOptimism"] = len(resource_lines)

    # GeneralFixture
    if "setUp" in func_name or "fixture" in func_name.lower():
        stmt_lines = [l for l in non_empty if not l.strip().startswith("#")]
        if len(stmt_lines) > 10:
            counts["GeneralFixture"] = 1

    return counts


def _detect_js(func_name: str, lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {s: 0 for s in SMELL_NAMES}

    expect_lines = [l for l in lines if _RE_JS_EXPECT.search(l)]
    non_empty = [l for l in lines if l.strip()]

    # AssertionRoulette: ≥3 expect() calls, no custom message argument
    if len(expect_lines) >= 3:
        has_msg = any(
            re.search(r",\s*['\"]", l) for l in expect_lines
        )
        if not has_msg:
            counts["AssertionRoulette"] = 1

    # EagerTest: ≥4 lines with distinct function invocations outside expect
    non_expect = [l for l in lines if not _RE_JS_EXPECT.search(l)]
    calls = {
        m.group(1)
        for l in non_expect
        for m in [re.search(r"\b(\w+)\s*\(", l)] if m
    }
    calls -= {"it", "test", "describe", "beforeEach", "afterEach",
               "expect", "beforeAll", "afterAll", "console"}
    if len(calls) >= 4:
        counts["EagerTest"] = 1

    # MagicNumberTest
    magic_hits = sum(1 for l in expect_lines if _RE_JS_MAGIC.search(l))
    counts["MagicNumberTest"] = magic_hits

    # UnknownTest
    is_test_block = any(k in func_name for k in ("it(", "test(", "it('", 'it("'))
    if is_test_block and not expect_lines:
        counts["UnknownTest"] = 1

    # RedundantAssertion
    redundant = sum(1 for l in lines if _RE_JS_REDUNDANT.search(l))
    counts["RedundantAssertion"] = redundant

    # DuplicateAssert
    normalized = [re.sub(r"\s+", " ", l.strip()) for l in expect_lines]
    seen: set[str] = set()
    dupes = 0
    for n in normalized:
        if n in seen:
            dupes += 1
        seen.add(n)
    counts["DuplicateAssert"] = dupes

    # VerboseTest
    if len(non_empty) > 20 and is_test_block:
        counts["VerboseTest"] = 1

    # MissingExceptionTest
    name_lower = func_name.lower()
    if any(k in name_lower for k in ("throw", "error", "exception", "invalid", "reject")):
        if not _RE_JS_EXCEPTION.search(" ".join(lines)):
            counts["MissingExceptionTest"] = 1

    # ResourceOptimism
    resource_lines = [l for l in lines if _RE_JS_RESOURCE.search(l)]
    if resource_lines:
        has_guard = any(_RE_JS_CATCH.search(l) for l in lines)
        if not has_guard:
            counts["ResourceOptimism"] = len(resource_lines)

    # GeneralFixture
    if "beforeEach" in func_name or "beforeAll" in func_name:
        stmt_lines = [l for l in non_empty if not l.strip().startswith("//")]
        if len(stmt_lines) > 10:
            counts["GeneralFixture"] = 1

    return counts


def detect_smells(patch: str, filename: str) -> dict[str, int]:
    """Return aggregated smell counts for a single test file patch."""
    lang = _detect_language(filename)
    if lang not in ("python", "js"):
        return {s: 0 for s in SMELL_NAMES}

    added = _extract_added_lines(patch)
    if not added:
        return {s: 0 for s in SMELL_NAMES}

    blocks = _split_into_test_functions(added, lang)
    totals: dict[str, int] = {s: 0 for s in SMELL_NAMES}

    detector = _detect_python if lang == "python" else _detect_js

    for name, block_lines in blocks:
        result = detector(name, block_lines)
        for s in SMELL_NAMES:
            totals[s] += result[s]

    return totals


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_corpus(
    offline: bool,
    loader: ZipLoader | None,
    limit: int | None,
    resume: bool,
) -> None:
    print("Loading rq2_corpus.parquet...")
    corpus = pd.read_parquet(CORPUS_PARQUET)
    corpus_pairs = set(zip(corpus["pr_id"], corpus["commit_sha"]))
    print(f"  Corpus: {len(corpus_pairs):,} (pr_id, commit_sha) pairs.")

    already_done: set[tuple] = set()
    if resume and SMELLS_PARQUET.exists():
        existing = pd.read_parquet(SMELLS_PARQUET,
                                   columns=["pr_id", "commit_sha", "filename"])
        already_done = set(
            zip(existing["pr_id"], existing["commit_sha"], existing["filename"])
        )
        print(f"  Resuming: {len(already_done):,} already processed.")

    print("Loading pr_commit_details (patch column)...")
    details = load_parquet(
        "pr_commit_details.parquet",
        offline,
        loader,
        columns=["pr_id", "sha", "filename", "patch"],
    )
    details = details.rename(columns={"sha": "commit_sha"})
    if limit:
        details = details.head(limit)

    # Filter to corpus pairs and test files only
    details = details[
        details.apply(
            lambda r: (r["pr_id"], r["commit_sha"]) in corpus_pairs, axis=1
        )
    ]
    details = details[details["filename"].apply(
        lambda f: bool(TEST_FILE_PATTERNS.search(str(f)))
    )]
    print(f"  Relevant patch rows: {len(details):,}")

    records = []
    n_skipped = 0
    n_error = 0
    n_null_patch = 0

    for _, row in details.iterrows():
        pr_id = row["pr_id"]
        commit_sha = row["commit_sha"]
        filename = str(row["filename"])
        patch = row["patch"]

        if resume and (pr_id, commit_sha, filename) in already_done:
            n_skipped += 1
            continue

        if not isinstance(patch, str) or not patch.strip():
            n_null_patch += 1
            continue

        try:
            smell_counts = detect_smells(patch, filename)
        except Exception as exc:
            n_error += 1
            print(f"  Error on {filename} pr={pr_id}: {exc}")
            smell_counts = {s: 0 for s in SMELL_NAMES}

        record = {
            "pr_id": pr_id,
            "commit_sha": commit_sha,
            "filename": filename,
            "language": _detect_language(filename),
        }
        record.update(smell_counts)
        records.append(record)

    print(f"\nProcessed: {len(records):,} | Skipped: {n_skipped:,} | "
          f"Null patches: {n_null_patch:,} | Errors: {n_error:,}")

    result_df = pd.DataFrame(records)

    if resume and SMELLS_PARQUET.exists():
        old = pd.read_parquet(SMELLS_PARQUET)
        result_df = pd.concat([old, result_df], ignore_index=True)

    result_df.to_parquet(SMELLS_PARQUET, index=False)
    print(f"Saved {len(result_df):,} rows → {SMELLS_PARQUET}")

    log = pd.DataFrame([{
        "total_rows_in_corpus": len(corpus_pairs),
        "relevant_patch_rows": len(details),
        "processed": len(records),
        "skipped_resume": n_skipped,
        "null_patch": n_null_patch,
        "errors": n_error,
    }])
    log.to_csv(LOG_CSV, index=False)
    print(f"Log → {LOG_CSV}")

    # Quick sanity check
    if not result_df.empty:
        print("\nSmell prevalence (% rows with count > 0):")
        for s in SMELL_NAMES:
            if s in result_df.columns:
                pct = (result_df[s] > 0).mean() * 100
                print(f"  {s:<25} {pct:5.1f}%")


def main(offline: bool, zip_path: Path, limit: int | None, resume: bool) -> None:
    if not CORPUS_PARQUET.exists():
        print(f"ERROR: {CORPUS_PARQUET} not found. Run rq2_00_build_corpus.py first.")
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
    parser = argparse.ArgumentParser(description="RQ2 smell detector")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, metavar="PATH")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed (pr_id, commit_sha, filename) triples")
    args = parser.parse_args()
    main(offline=args.offline, zip_path=args.zip, limit=args.limit, resume=args.resume)
