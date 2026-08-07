"""Shared helper functions for this project."""

from __future__ import annotations

import io
import json
import pathlib
import re
import zipfile
from typing import Any

import pandas as pd
from scipy import stats

from cliffs_delta import cliffs_delta


ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = ROOT_DIR / "clean_data"
PR_DIR = DATA_DIR / "prs"
DEVELOPER_DIR = DATA_DIR / "developers"
COMMENTS_DIR = DATA_DIR / "pr_comments"
REVIEWS_DIR = DATA_DIR / "pr_reviews"
COMMITS_DIR = DATA_DIR / "pr_commits"
COMMIT_DETAILS_DIR = DATA_DIR / "commit_details"
TIMELINE_DIR = DATA_DIR / "pr_timelines"
ISSUE_TIMELINE_DIR = DATA_DIR / "issue_timelines"
DEV_REPO_PR_DIR = DATA_DIR / "dev_repo_prs"
ISSUES_DIR = DATA_DIR / "issues"
REPO_PR_DIR = DATA_DIR / "repo_prs"
ANALYSIS_DIR = ROOT_DIR / "analysis"
CSV_DIR = ROOT_DIR / "AIDev"
FIG_DIR = ROOT_DIR / "figs"
PROJECT_META_DIR = DATA_DIR / "project_metadata"
PROJECT_META_RAW_DIR = DATA_DIR / "project_metadata_raw"
PROTOTYPE_DIR = DATA_DIR / "prototype"
SCOPE_DIR = ROOT_DIR / "AIDev-pop"

COLOR_MAP = {
    "Human": "#56B4E9",
    "OpenAI_Codex": "#D55E00",
    "OpenAI Codex": "#D55E00",
    "Codex": "#F0E442",
    "Devin": "#009E73",
    "Copilot": "#0072B2",
    "GitHub Copilot": "#0072B2",
    "Cursor": "#785EF0",
    "Claude_Code": "#DC267F",
    "Claude Code": "#DC267F",
}

NAME_MAPPING = {
    "OpenAI_Codex": "OpenAI Codex",
    "Codex": "OpenAI Codex",
    "Devin": "Devin",
    "Copilot": "GitHub Copilot",
    "Cursor": "Cursor",
    "Claude_Code": "Claude Code",
    "Claude": "Claude Code",
    "Human": "Human",
    "OpenAI Codex": "OpenAI Codex",
    "GitHub Copilot": "GitHub Copilot",
    "Claude Code": "Claude Code",
}
for key, value in NAME_MAPPING.items():
    if key not in COLOR_MAP:
        COLOR_MAP[key] = COLOR_MAP[value]

PREFER_ORDER = [
    "Human",
    "OpenAI_Codex",
    "Devin",
    "Copilot",
    "Cursor",
    "Claude_Code"
]

FLOW_ORDER = [
    "feat",  # new functionality
    "fix",  # bug fixes
    "perf",  # performance work
    "refactor",  # code reshaping
    "style",  # lint / formatting
    "docs",  # documentation
    "test",  # testing
    "chore",  # misc maintenance
    "build",  # packaging / build sys
    "ci",  # continuous-integration
    "other",  # continuous-integration
]


def save_json(data: Any, filepath: str | pathlib.Path, indent: int = 2) -> None:
    """Write *data* to *filepath* as JSON."""

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def read_json(filepath: str | pathlib.Path) -> Any:
    """Return JSON data loaded from *filepath*."""

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_pr_df(df: pd.DataFrame, agent: str) -> pd.DataFrame:
    """Return *df* after applying agent-specific PR filters."""

    agent_key = agent.lower()

    if agent_key == "claude_code":
        mask = df["body"].str.contains("Co-Authored-By: Claude", case=False, na=False)
        return df.loc[mask]
    if agent_key == "copilot":
        mask = df["user"].str.lower() == "copilot"
        return df.loc[mask]
    return df


def mannUandCliffdelta(dist1, dist2):
    d, size = cliffs_delta(dist1, dist2)
    print(f"Cliff's delta: {size}, d={d}")
    u, p = stats.mannwhitneyu(dist1, dist2, alternative="two-sided")
    print(f"Mann-Whitney-U-test: u={u} p={p}")
    return u, p, d, size


# ---------------------------------------------------------------------------
# RQ2 — Test Technical Debt catalog and helpers
# ---------------------------------------------------------------------------

SMELL_CATALOG: dict[str, dict] = {
    "AssertionRoulette":    {"weight": 0.10, "source": "Palomba"},
    "EagerTest":            {"weight": 0.12, "source": "Palomba"},
    "MagicNumberTest":      {"weight": 0.07, "source": "Palomba"},
    "UnknownTest":          {"weight": 0.15, "source": "Palomba"},
    "RedundantAssertion":   {"weight": 0.08, "source": "Palomba"},
    "DuplicateAssert":      {"weight": 0.06, "source": "Palomba"},
    "VerboseTest":          {"weight": 0.06, "source": "Bavota"},
    "MissingExceptionTest": {"weight": 0.09, "source": "Bavota"},
    "ResourceOptimism":     {"weight": 0.11, "source": "Verdecchia"},
    "GeneralFixture":       {"weight": 0.08, "source": "Verdecchia"},
}

TEST_FILE_PATTERNS: re.Pattern = re.compile(
    r"(test_[^/]+\.py$|[^/]+_test\.py$|tests/[^/]+\.py$"
    r"|[^/]+\.test\.[jt]sx?$|[^/]+\.spec\.[jt]sx?$"
    r"|Test[^/]+\.java$|[^/]+Test\.java$)",
    re.IGNORECASE,
)

PHI_THRESHOLD: float = 0.20

HF_BASE = "hf://datasets/hao-li/AIDev"


def compute_tdt_index(smell_counts: dict) -> float:
    """Return weighted TDT index from a dict of {smell_name: count}."""
    return sum(
        SMELL_CATALOG[s]["weight"] * smell_counts.get(s, 0)
        for s in SMELL_CATALOG
    )


class ZipLoader:
    """Read parquet files directly from a zip archive without full extraction."""

    def __init__(self, zip_path: pathlib.Path) -> None:
        self._zip_path = zip_path
        self._zf: zipfile.ZipFile | None = None
        self._names: list[str] = []

    def open(self) -> None:
        self._zf = zipfile.ZipFile(self._zip_path, "r")
        self._names = [m.filename for m in self._zf.infolist()]

    def close(self) -> None:
        if self._zf:
            self._zf.close()

    def __enter__(self) -> "ZipLoader":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def read_parquet(
        self,
        basename: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        matches = [n for n in self._names if n.endswith(basename)]
        if not matches:
            raise FileNotFoundError(f"{basename} not found in ZIP")
        data = self._zf.read(matches[0])  # type: ignore[union-attr]
        return pd.read_parquet(io.BytesIO(data), columns=columns)


def load_parquet(
    filename: str,
    offline: bool,
    loader: ZipLoader | None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a parquet file from the ZIP (offline) or HuggingFace."""
    if offline and loader:
        return loader.read_parquet(filename, columns=columns)
    try:
        return pd.read_parquet(f"{HF_BASE}/{filename}", columns=columns)
    except Exception as exc:
        if loader:
            print(f"  [fallback ZIP] {exc}")
            return loader.read_parquet(filename, columns=columns)
        raise


# ---------------------------------------------------------------------------
# RQ3 — Performance Script Quality Index catalog
# ---------------------------------------------------------------------------

PSQI_DIMENSIONS: list[str] = [
    "dim_load_type",
    "dim_think_time",
    "dim_payload",
    "dim_negative",
    "dim_sla",
]

PERF_TOOL_ORDER: list[str] = [
    "k6",
    "locust",
    "jmeter",
    "gatling",
    "pytest_benchmark",
    "generic_perf",
]


# ---------------------------------------------------------------------------
# RQ4 — Review-comment taxonomy (Bacchelli & Bird 2013, adapted to tests)
# ---------------------------------------------------------------------------

REVIEW_CATEGORIES: list[str] = [
    "superficial",      # style, naming, formatting
    "functional",       # wrong test logic, bad assertion, setup/teardown
    "coverage",         # missing cases, absent boundary scenarios
    "maintainability",  # test smells, coupling, duplication
    "confidence",       # questions the validity/usefulness of the test
    "process",          # PR flow itself: timing, scope, rebase, CI
    "directive",        # emergent: terse imperative with no diagnosis
]

REVIEW_CATEGORY_LABELS: dict[str, str] = {
    "superficial": "Superficial",
    "functional": "Functional",
    "coverage": "Coverage",
    "maintainability": "Maintainability",
    "confidence": "Confidence",
    "process": "Process",
    "directive": "Directive",
    "unclassified": "Unclassified",
}

# A review comment is not always feedback: in agentic PRs a large share are the
# agent reporting back a fix. Mixing the two inflates "feedback" counts, so the
# role is scored separately and the taxonomy is applied to `feedback` only.
REVIEW_ROLES: list[str] = [
    "feedback",         # a reviewer raising something
    "acknowledgement",  # author/agent replying: "fixed in commit abc123"
    "code_suggestion",  # bare ```suggestion block, no prose to classify
]

# Ordinal tone scale: -1 critical / 0 neutral / +1 constructive
TONE_LABELS: dict[int, str] = {
    -1: "critical",
    0: "neutral",
    1: "constructive",
}

ARTIFACT_KINDS: list[str] = ["test", "production", "config", "docs"]

# Non-test paths that must not be counted as production source code
CONFIG_FILE_PATTERNS: re.Pattern = re.compile(
    r"(\.(ya?ml|json|toml|ini|cfg|lock|env)$"
    r"|(^|/)(Dockerfile|Makefile|\.gitignore)$"
    r"|(^|/)\.github/)",
    re.IGNORECASE,
)

DOC_FILE_PATTERNS: re.Pattern = re.compile(
    r"\.(md|mdx|rst|txt|adoc)$", re.IGNORECASE
)


def classify_artifact_kind(path: str | None) -> str:
    """Bucket a commented file path into test / config / docs / production."""
    if not path or not isinstance(path, str):
        return "production"
    if TEST_FILE_PATTERNS.search(path):
        return "test"
    if DOC_FILE_PATTERNS.search(path):
        return "docs"
    if CONFIG_FILE_PATTERNS.search(path):
        return "config"
    return "production"
