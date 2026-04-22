#!/usr/bin/env python3
"""
RQ3 Viability Probe — Performance Artifact Discovery in AIDev-pop
==================================================================
Answers the question: "Are there enough k6/JMeter performance scripts in
AI-authored PRs to sustain RQ3 as originally framed?"

Outputs
-------
  1. Console table: counts of performance artefacts per agent (file-pattern match)
  2. Console table: existing 'perf' PRs per agent (from pre-classified pr_task_type)
  3. rq3_perf_files.csv       — every matched file row (for manual inspection)
  4. rq3_perf_pr_summary.csv  — one row per PR with match counts
  5. rq3_decision.txt         — plain-text recommendation: keep / reformulate RQ3

Usage
-----
  python rq3_perf_exploration.py

Credentials
-----------
  HF_TOKEN is loaded automatically from the .env file in the project root.
  To override, set the env var before running:
    HF_TOKEN=hf_... python rq3_perf_exploration.py

"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

# Load HF_TOKEN from .env (no-op if already set in the environment)
load_dotenv(ROOT_DIR / ".env")

HF_BASE = "hf://datasets/hao-li/AIDev"

# ---------------------------------------------------------------------------
# Performance file patterns (filename-level)
# Rationale: covers k6 (*.k6.js, *.k6.ts), JMeter (*.jmx),
#            Gatling (*.scala in perf dirs), Locust (locustfile.py),
#            and generic naming conventions used in CI performance suites.
# ---------------------------------------------------------------------------
FILENAME_PATTERNS: list[re.Pattern] = [
    # k6 scripts
    re.compile(r"\.k6\.(js|ts)$", re.IGNORECASE),
    re.compile(r"(^|[/_-])k6[/_-]", re.IGNORECASE),
    re.compile(r"[/_-]k6\.(js|ts)$", re.IGNORECASE),
    # JMeter
    re.compile(r"\.jmx$", re.IGNORECASE),
    # Gatling
    re.compile(r"gatling", re.IGNORECASE),
    # Locust
    re.compile(r"locustfile\.py$", re.IGNORECASE),
    re.compile(r"[/_-]locust[/_-]", re.IGNORECASE),
    # Generic performance / load-test naming
    re.compile(
        r"(load[_-]?test|stress[_-]?test|perf[_-]?test|benchmark)", re.IGNORECASE
    ),
    re.compile(r"[/_-](perf|load|stress|benchmark)[/_-]", re.IGNORECASE),
    re.compile(r"(performance|throughput)[_-]test", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# k6 API patterns inside patch content
# Catches scripts that may have generic filenames but contain k6 API calls.
# ---------------------------------------------------------------------------
K6_CONTENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bimport\s+.*from\s+['\"]k6['\"]", re.IGNORECASE),
    re.compile(r"\bimport\s+.*from\s+['\"]k6/", re.IGNORECASE),
    re.compile(r"\bhttp\.get\b.*\bcheck\b", re.IGNORECASE),  # k6 http + check idiom
    re.compile(r"\bVirtualUsers\b|\bvus\b", re.IGNORECASE),
    re.compile(r"\bramping_arrival_rate\b|\bramping_vus\b", re.IGNORECASE),
    re.compile(r"\bthresholds\s*:", re.IGNORECASE),
    re.compile(r"from\s+['\"]k6/http['\"]", re.IGNORECASE),
]

# Agent name normaliser (matches helper.py NAME_MAPPING)
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

AGENT_ORDER = ["OpenAI Codex", "GitHub Copilot", "Cursor", "Devin", "Claude Code"]

# Viability threshold: minimum distinct PRs to consider RQ3 feasible as-is
MIN_PRS_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalise_agent(name: str) -> str:
    return NAME_MAP.get(name, name)


def matches_filename(fname: str | None) -> bool:
    if not fname or not isinstance(fname, str):
        return False
    return any(p.search(fname) for p in FILENAME_PATTERNS)


def matches_k6_content(patch: str | None) -> bool:
    if not patch or not isinstance(patch, str):
        return False
    return any(p.search(patch) for p in K6_CONTENT_PATTERNS)


def match_category(fname: str | None) -> str:
    """Return the category of the first matching pattern."""
    if not fname or not isinstance(fname, str):
        return "none"
    fname_lower = fname.lower()
    if re.search(r"\.k6\.(js|ts)|[/_-]k6[/_-]|[/_-]k6\.(js|ts)", fname_lower):
        return "k6"
    if re.search(r"\.jmx$", fname_lower):
        return "JMeter"
    if re.search(r"gatling", fname_lower):
        return "Gatling"
    if re.search(r"locustfile\.py|[/_-]locust[/_-]", fname_lower):
        return "Locust"
    if re.search(
        r"load[_-]?test|stress[_-]?test|perf[_-]?test|benchmark|"
        r"[/_-](perf|load|stress|benchmark)[/_-]|performance[_-]test|throughput[_-]test",
        fname_lower,
    ):
        return "generic_perf"
    return "none"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 65)
    print(" RQ3 Viability Probe — AIDev-pop Performance Artifact Search")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 1. Load PR metadata + pre-classified task types
    # -----------------------------------------------------------------------
    print("\n[1/4] Loading pull_request.parquet and pr_task_type.parquet ...")
    pr_df = pd.read_parquet(
        f"{HF_BASE}/pull_request.parquet",
        columns=["id", "agent", "repo_id", "html_url"],
    )
    pr_df["agent"] = pr_df["agent"].map(normalise_agent).fillna(pr_df["agent"])

    task_df = pd.read_parquet(f"{HF_BASE}/pr_task_type.parquet", columns=["id", "type"])

    # Merge task type into PR frame
    pr_full = pr_df.merge(task_df, on="id", how="left")

    # -----------------------------------------------------------------------
    # 2. Summarise pre-classified 'perf' PRs (already available, free signal)
    # -----------------------------------------------------------------------
    print("\n[2/4] Summarising pre-classified 'perf' PRs ...")
    perf_task_prs = pr_full[pr_full["type"] == "perf"].copy()

    perf_by_agent = (
        perf_task_prs.groupby("agent")
        .agg(perf_pr_count=("id", "count"))
        .reindex(AGENT_ORDER, fill_value=0)
        .reset_index()
    )

    print("\n── Pre-classified 'perf' PRs (from pr_task_type.parquet) ─────")
    print(perf_by_agent.to_string(index=False))
    print(f"\n   Total 'perf' PRs: {len(perf_task_prs)}")

    # -----------------------------------------------------------------------
    # 3. Scan pr_commit_details for performance file patterns
    # -----------------------------------------------------------------------
    print("\n[3/4] Loading pr_commit_details.parquet (711k rows) ...")
    print("      This may take a few minutes depending on bandwidth ...")

    # Load only the columns we need to reduce memory footprint
    details_df = pd.read_parquet(
        f"{HF_BASE}/pr_commit_details.parquet",
        columns=["pr_id", "filename", "patch"],
    )

    print(f"      Loaded {len(details_df):,} file-level rows.")

    # Join agent name
    pr_agent = pr_df[["id", "agent", "html_url"]].rename(columns={"id": "pr_id"})  # type: ignore
    details_df = details_df.merge(pr_agent, on="pr_id", how="inner")

    # ── 3a. Filename-based match ──────────────────────────────────────────
    print("      Scanning filenames ...")
    details_df["fname_match"] = details_df["filename"].fillna("").astype(str).apply(matches_filename)
    details_df["category"] = details_df["filename"].fillna("").astype(str).apply(match_category)

    # ── 3b. k6 content match (only on JS/TS files not already matched) ───
    print("      Scanning patch content for k6 API patterns ...")
    unmatched_jsts = ~details_df["fname_match"] & details_df[
        "filename"
    ].str.lower().str.contains(r"\.(js|ts)$", na=False)
    details_df.loc[unmatched_jsts, "content_k6"] = (
        details_df[unmatched_jsts]["patch"]
        .fillna("")
        .astype(str)
        .apply(matches_k6_content)
    )
    details_df["content_k6"] = details_df["content_k6"].fillna(False)

    # Union of both signals
    details_df["any_match"] = details_df["fname_match"] | details_df["content_k6"]

    # Override category for content-only k6 hits
    content_only = details_df["content_k6"] & ~details_df["fname_match"]
    details_df.loc[content_only, "category"] = "k6_content_only"

    # -----------------------------------------------------------------------
    # 4. Aggregate and report
    # -----------------------------------------------------------------------
    print("\n[4/4] Aggregating results ...")

    matched_files = details_df[details_df["any_match"]].copy()

    # Per-agent file count
    files_by_agent = (
        matched_files.groupby("agent")
        .agg(matched_files_count=("pr_id", "count"))
        .reindex(AGENT_ORDER, fill_value=0)
        .reset_index()
    )

    # Per-agent distinct PR count
    prs_by_agent = (
        matched_files.groupby("agent")["pr_id"]
        .nunique()
        .reindex(AGENT_ORDER, fill_value=0)
        .reset_index()
        .rename(columns={"pr_id": "distinct_prs"})
    )

    # Category breakdown
    category_breakdown = (
        matched_files.groupby(["agent", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(AGENT_ORDER, fill_value=0)
    )

    # ── Console output ─────────────────────────────────────────────────────
    summary = files_by_agent.merge(prs_by_agent, on="agent")
    summary = summary.merge(perf_by_agent, on="agent")

    print("\n" + "=" * 65)
    print(" RESULTS")
    print("=" * 65)

    print("\n── Performance artifact matches by agent ─────────────────────")
    print(
        f"{'Agent':<20} {'Matched Files':>14} {'Distinct PRs':>13} {'perf-type PRs':>14}"
    )
    print("-" * 65)
    for _, row in summary.iterrows():
        print(
            f"{row['agent']:<20} {int(row['matched_files_count']):>14,}"
            f" {int(row['distinct_prs']):>13,}"
            f" {int(row['perf_pr_count']):>14,}"
        )
    print("-" * 65)
    print(
        f"{'TOTAL':<20} {int(summary['matched_files_count'].sum()):>14,}"
        f" {int(matched_files['pr_id'].nunique()):>13,}"
        f" {int(summary['perf_pr_count'].sum()):>14,}"
    )

    print("\n── Category breakdown (matched files) ────────────────────────")
    print(category_breakdown.to_string())

    # ── Decision logic ─────────────────────────────────────────────────────
    total_k6_prs = int(
        matched_files[matched_files["category"].isin(["k6", "k6_content_only"])][
            "pr_id"
        ].nunique()
    )
    total_matched_prs = int(matched_files["pr_id"].nunique())

    print("\n" + "=" * 65)
    print(" RQ3 DECISION")
    print("=" * 65)
    if total_k6_prs >= MIN_PRS_THRESHOLD:
        verdict = "VIABLE"
        recommendation = textwrap.dedent(
            f"""\
            VERDICT: RQ3 is VIABLE as originally framed (k6 focus).
            Found {total_k6_prs} distinct PRs with k6 scripts — above the
            minimum threshold of {MIN_PRS_THRESHOLD}.
            Proceed with k6 analysis: extract VUs, stages, thresholds.
        """
        )
    elif total_matched_prs >= MIN_PRS_THRESHOLD:
        verdict = "REFORMULATE"
        recommendation = textwrap.dedent(
            f"""\
            VERDICT: Reformulate RQ3.
            k6-specific PRs: {total_k6_prs} (below threshold of {MIN_PRS_THRESHOLD}).
            BUT total performance-related PRs (all tools): {total_matched_prs}.
            Suggested reformulation:
              "Como agentes de IA abordam artefatos de performance (commits
               do tipo 'perf', benchmarks, scripts de carga) comparado a
               especialistas humanos? Quais estratégias de otimização são
               mais frequentes?"
            Include: perf-type PRs ({int(summary['perf_pr_count'].sum())}),
                     benchmark files, and generic load-test scripts.
        """
        )
    else:
        verdict = "DROP OR PIVOT"
        recommendation = textwrap.dedent(
            f"""\
            VERDICT: Drop or significantly pivot RQ3.
            Both k6-specific ({total_k6_prs}) and total performance PRs
            ({total_matched_prs}) are below the minimum threshold of {MIN_PRS_THRESHOLD}.
            Suggested pivot:
              Replace RQ3 with analysis of how AI agents handle performance-
              sensitive code changes (identified via static analysis of
              algorithmic complexity patterns), or merge RQ3 concerns into
              RQ2 as a "performance test smell" sub-category.
        """
        )

    print(f"\n  {recommendation}")

    # -----------------------------------------------------------------------
    # 5. Save artefacts
    # -----------------------------------------------------------------------
    files_out = OUT_DIR / "rq3_perf_files.csv"
    summary_out = OUT_DIR / "rq3_perf_pr_summary.csv"
    decision_out = OUT_DIR / "rq3_decision.txt"

    matched_files.drop(columns=["patch"]).to_csv(files_out, index=False)
    summary.to_csv(summary_out, index=False)
    decision_out.write_text(
        f"Verdict: {verdict}\n\n{recommendation}\n\n"
        f"Total k6 PRs: {total_k6_prs}\n"
        f"Total perf-pattern PRs: {total_matched_prs}\n"
        f"Total pre-classified perf PRs: {int(summary['perf_pr_count'].sum())}\n",
        encoding="utf-8",
    )

    print(f"\n  Saved matched files  → {files_out}")
    print(f"  Saved PR summary     → {summary_out}")
    print(f"  Saved decision memo  → {decision_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
