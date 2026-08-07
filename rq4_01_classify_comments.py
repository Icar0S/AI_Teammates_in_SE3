#!/usr/bin/env python3
"""
RQ4 — Script 01: Classify Review Comments
==========================================
Aplica a taxonomia de Bacchelli & Bird (2013), adaptada a artefatos de teste
(protocolo §3.6.2), a cada comentário de revisão, e deriva as três dimensões
restantes da RQ4: tom, intensidade e resolução.

Dimensões produzidas
--------------------
  category        — superficial | functional | coverage | maintainability
                    | confidence | process | unclassified
  category_scores — pontuação bruta de cada categoria (JSON, auditável)
  tone            — -1 crítico | 0 neutro | +1 construtivo
  intensity       — 0–4: comprimento, sugestão de código, perguntas, severidade
  resolution      — unanswered | answered  (proxy: existência de resposta
                    na thread; o GitHub `resolved` não consta do dataset)

Classificação por regras
------------------------
Cada categoria tem um léxico de padrões ponderados. Vence a categoria de maior
pontuação; empates seguem a ordem de prioridade de CATEGORY_PRIORITY (as
categorias mais específicas ganham das mais genéricas). Comentários sem
nenhum acerto ficam como `unclassified`.

O classificador é determinístico e auditável — os scores por categoria são
gravados para permitir a reconciliação humana prevista no protocolo. O script
também exporta uma amostra estratificada para dupla codificação manual e
cálculo do kappa de Cohen (protocolo §4.3.1).

Inputs
------
  AIDev/rq4_comments.parquet   — produzido por rq4_00_build_corpus.py

Outputs
-------
  AIDev/rq4_comments_classified.parquet
  AIDev/rq4_manual_coding_sample.csv   — amostra para kappa de Cohen
  AIDev/rq4_classify_log.csv

Usage
-----
  python rq4_01_classify_comments.py
  python rq4_01_classify_comments.py --sample-size 300 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import REVIEW_CATEGORIES  # noqa: E402

# ---------------------------------------------------------------------------
# Role detection — feedback vs. the agent reporting back a fix
# ---------------------------------------------------------------------------
# Em PRs agênticos boa parte dos "review comments" não é feedback: é o próprio
# agente respondendo ("Fixed in commit abc123"). Tratá-los como feedback infla
# as contagens da taxonomia, então o papel é pontuado antes da categoria.

_ACK_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(fixed|done|applied|addressed|resolved|implemented|updated|"
        r"changed|corrected|reverted|removed|added)\b[^.]{0,80}?\b"
        r"(in\s+)?commit\s+[0-9a-f]{6,}", re.IGNORECASE),
    re.compile(r"\bcommit\s+[0-9a-f]{7,}\b", re.IGNORECASE),
    re.compile(r"^\s*(done|fixed|fixed!|thanks|thank you|ok|okay|sure|"
               r"good catch|nice catch|agreed|ack|👍|✅|sgtm)\s*[.!]?\s*$",
               re.IGNORECASE),
    re.compile(r"^\s*(i'?ve|i have|we'?ve|we have)\s+"
               r"(updated|added|changed|fixed|removed|refactored|moved)\b",
               re.IGNORECASE),
    re.compile(r"^\s*(applied|addressed|implemented)\s+"
               r"(this|the|your)\s+(suggestion|comment|feedback)",
               re.IGNORECASE),
    re.compile(r"\bhas been (fixed|addressed|resolved|updated)\b",
               re.IGNORECASE),
]

# Imperativos curtos e sem diagnóstico: acionáveis, mas não diagnosticáveis
# pela taxonomia de Bacchelli & Bird. Categoria indutiva emergente, prevista
# no protocolo §3.6.2 ("categorias emergentes ... registradas como novas").
_DIRECTIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*@?\w*\s*(please\s+)?"
               r"(remove|delete|drop|undo|revert|move|rename|keep|use|add|"
               r"fix|change|revert|restore|extract|inline|split)\b"
               r"[^.?!]{0,40}$", re.IGNORECASE),
    re.compile(r"^\s*(no|nope|yes|yep|what|why|undo|revert|remove|delete|"
               r"nit|same|ditto|typo)\s*[.?!]?\s*$", re.IGNORECASE),
]

_BARE_SUGGESTION = re.compile(
    r"^\s*```suggestion\s*[\r\n]*```\s*$", re.IGNORECASE)

OUT_DIR = ROOT_DIR / "AIDev"
OUT_DIR.mkdir(exist_ok=True)

COMMENTS_PARQUET = OUT_DIR / "rq4_comments.parquet"
CLASSIFIED_PARQUET = OUT_DIR / "rq4_comments_classified.parquet"
SAMPLE_CSV = OUT_DIR / "rq4_manual_coding_sample.csv"
LOG_CSV = OUT_DIR / "rq4_classify_log.csv"

# Mais específicas primeiro: em empate, "confidence" ganha de "functional",
# que ganha de "superficial", e assim por diante.
CATEGORY_PRIORITY = [
    "confidence",
    "coverage",
    "functional",
    "maintainability",
    "process",
    "superficial",
]


# ---------------------------------------------------------------------------
# Lexicons — (regex, weight) per category
# ---------------------------------------------------------------------------

_LEXICON: dict[str, list[tuple[str, float]]] = {
    # (1) Superficial: estilo, nomenclatura, formatação
    "superficial": [
        (r"\btypos?\b|\bspell(ing|ed)?\b|\bmisspell", 2.0),
        (r"\bnit\b|\bnitpick\b|\bnitpicking\b", 2.0),
        (r"\b(rename|naming|name it|better name|variable name)\b", 1.5),
        (r"\b(format+ing|indent(ation|ed)?|whitespace|trailing\s+(comma|space))\b", 1.5),
        (r"\b(lint(er|ing)?|prettier|eslint|black|gofmt|rubocop)\b", 1.5),
        (r"\b(unused|unnecessary)\s+(import|variable|parameter)\b", 1.5),
        (r"\b(semicolon|capitali[sz]|uppercase|lowercase|camelcase|snake_case)\b", 1.0),
        (r"\b(comment|docstring|documentation)\s+(typo|wording|style)\b", 1.0),
        (r"\b(reorder|sort)\s+(the\s+)?imports?\b", 1.0),
        # observed vocabulary
        (r"\b(add|adding)\s+(jsdoc|docstrings?|comments?)\b", 1.5),
        (r"\bend\s+files?\s+with\s+a\s+newline\b|\bnewline\s+at\s+end\b", 1.5),
        (r"\b(translate|localis|localiz)\w*\s+(the\s+)?(code\s+)?comment", 1.5),
        (r"\b(wrong|bad|inconsistent)\s+format(ting)?\b", 1.5),
        (r"\bdrop\s+(any\s+|the\s+)?comments?\b", 1.5),
    ],
    # (2) Functional: lógica do teste, asserção, setup/teardown
    "functional": [
        (r"\b(this|it|that)\s+(will|would|could|might)\s+(fail|break|throw|error)\b", 2.5),
        (r"\b(wrong|incorrect|invalid)\s+(assert|assertion|expect|value|result|logic)\b", 2.5),
        (r"\b(assert|assertion|expect)\w*\s+(is|are|looks?)\s+(wrong|incorrect|backwards)\b", 2.5),
        (r"\boff[- ]by[- ]one\b|\bshould\s+be\s+[`'\"]?\S+[`'\"]?\s+not\b", 2.0),
        (r"\b(setup|teardown|set_?up|tear_?down|before(each|all)?|after(each|all)?)\b", 1.5),
        (r"\b(fixture|mock|stub|spy|patch)\w*\s+(is|are|was|were)?\s*(wrong|incorrect|missing|misused|not)\b", 2.0),
        (r"\b(race\s+condition|deadlock|non[- ]?deterministic|timing\s+issue)\b", 2.0),
        (r"\b(null|none|nil|undefined)\s+(check|handling|pointer|deref)\b", 1.5),
        (r"\b(bug|broken|regression|breaks?)\b", 1.5),
        (r"\b(expected|actual)\s+(value|result)s?\s+(is|are|seems?|looks?)\b", 1.5),
        (r"\bexception\s+(is\s+)?(not\s+)?(raised|thrown|caught|expected)\b", 1.5),
        # observed vocabulary
        (r"\buse\s+[`'\"]?\w[\w.()]*[`'\"]?\s+(here|instead|everywhere)\b", 1.5),
        (r"\b(should|must|need\s+to)\s+use\b", 1.5),
        (r"\bkeep\s+calling\b|\binstead\s+of\s+[`'\"]?\w", 1.5),
        (r"\b(this|it)\s+is\s+not\s+using\b", 2.0),
        (r"\b(check|assert)\w*\s+(for\s+)?(the\s+)?exact\s+"
         r"(expected\s+)?value\b", 1.5),
        (r"\bthread[- ]?safe|synchroni[sz]ation|concurrent\b", 1.5),
    ],
    # (3) Coverage: casos ausentes, cenários de fronteira
    "coverage": [
        (r"\b(missing|need|needs|add|adding|write)\s+(a\s+|more\s+|some\s+)?(unit\s+|integration\s+)?tests?\b", 2.5),
        (r"\bnot\s+(covered|tested)\b|\buncovered\b|\buntested\b", 2.5),
        (r"\b(edge|corner|boundary)\s+cases?\b", 2.5),
        (r"\bwhat\s+(about|happens)\s+(if|when)\b", 2.0),
        (r"\b(also|should\s+we|can\s+we|could\s+we)\s+test\b", 2.0),
        (r"\b(test|cover)\s+(the\s+)?(case|scenario|path)\s+where\b", 2.0),
        (r"\b(negative|failure|error)\s+(case|scenario|path)s?\b", 2.0),
        (r"\b(only\s+)?happy\s+path\b", 2.0),
        (r"\b(test\s+)?coverage\b", 1.5),
        (r"\b(empty|null|zero|negative|large|unicode)\s+(input|value|list|array|string)\b", 1.5),
        # observed vocabulary
        (r"\b(could|can|should)\s+(these|this|the)\s+tests?\b", 2.0),
        (r"\badd\s+this\s+test\s+under\b|\badd\s+a\s+\w+\s+test\b", 2.0),
        (r"\btests?\s+(for|covering)\s+(the\s+)?\w+\s+(case|path|api)\b", 1.5),
    ],
    # (4) Maintainability: smells, acoplamento, duplicação
    "maintainability": [
        (r"\b(duplicat(e|ed|ion)|copy[- ]?paste|repeated|DRY)\b", 2.5),
        (r"\b(refactor|extract|split|decompose)\s+(this|it|into|the)\b", 2.0),
        (r"\b(hard[- ]?cod(e|ed|ing)|magic\s+(number|string|value))\b", 2.0),
        (r"\b(too|overly)\s+(long|complex|big|large|verbose)\b", 2.0),
        (r"\b(coupl(ed|ing)|tightly\s+coupled|depends?\s+on\s+internal)\b", 2.0),
        (r"\b(parametri[sz]e|table[- ]driven|test\s+matrix|data[- ]driven)\b", 2.0),
        (r"\b(helper|utility|shared)\s+(function|method|fixture)\b", 1.5),
        (r"\b(reuse|reusable|extract\s+into)\b", 1.5),
        (r"\b(maintain(able|ability)|readab(le|ility))\b", 1.5),
        (r"\bmultiple\s+assert(ion)?s?\b|\bassertion\s+roulette\b", 1.5),
        # observed vocabulary
        (r"\b(into|as)\s+a\s+(seperate|separate)\s+(function|method|file)\b", 2.0),
        (r"\bmove\s+(this|these|the)\s+\w*\s*(file|test|tests|class)\b", 2.0),
        (r"\b(different|another|its\s+own)\s+file\b", 1.5),
        (r"\bnext\s+to\s+the\s+\w+\s+file\b", 1.5),
        (r"\b(refactor|reorgani[sz]e)\b", 1.5),
    ],
    # (5) Confidence: questiona a validade/utilidade do teste
    "confidence": [
        (r"\bdoes\s+(this|it)\s+(actually|really|even)\s+test\b", 3.0),
        (r"\b(this|the)\s+test\s+(always|never)\s+(passes|fails|runs)\b", 3.0),
        (r"\b(tautolog|vacuous|meaningless|pointless|useless)\w*\b", 3.0),
        (r"\bdoesn'?t\s+(assert|verify|check|test)\s+(anything|much|the)\b", 3.0),
        (r"\bno\s+(real\s+)?assert(ion)?s?\b", 2.5),
        (r"\b(flaky|flakiness|intermittent(ly)?\s+fail)\b", 2.5),
        (r"\bfalse\s+(positive|negative|sense\s+of\s+security)\b", 2.5),
        (r"\bwhy\s+(do\s+we\s+|are\s+we\s+)?(test|testing|need)\s+this\b", 2.5),
        (r"\b(mock(ing|s|ed)?\s+(everything|too\s+much|the\s+whole))\b", 2.5),
        (r"\b(add|adds|provide|bring)\s+(any\s+)?value\b", 2.0),
        (r"\bis\s+this\s+test\s+(useful|meaningful|necessary|needed)\b", 2.5),
        (r"\btrivial\s+test\b|\bsmoke\s+test\s+only\b", 2.0),
        # observed vocabulary
        (r"\b(i\s+)?do(n'?t| not)\s+trust\b", 2.5),
        (r"\bfeel\s+more\s+confident\b|\bnot\s+confident\b", 2.5),
        (r"\bdo\s+we\s+(really\s+)?need\s+(this|it|these)\b", 2.5),
        (r"\bdon'?t\s+think\s+(this|it|that)\s+is\s+(great|good|right)\b", 2.0),
        (r"\bwhat\s+does\s+this\s+(mean|do|test)\b", 2.0),
        (r"\bis\s+this\s+(really\s+)?(correct|right|intended)\b", 2.0),
    ],
    # (6) Process: fluxo do PR
    "process": [
        (r"\b(rebase|merge\s+conflict|squash|force[- ]push|cherry[- ]pick)\b", 2.5),
        (r"\b(out\s+of\s+scope|separate\s+(PR|pull\s+request)|another\s+PR)\b", 2.5),
        (r"\b(CI|pipeline|workflow|build)\s+(is\s+)?(failing|red|broken|green)\b", 2.5),
        (r"\b(changelog|release\s+notes?|version\s+bump)\b", 2.0),
        (r"\b(revert|roll\s?back)\s+(this|that|the)\b", 2.0),
        (r"\b(update|fix)\s+the\s+(PR\s+)?description\b", 2.0),
        (r"\b(depend(ency|encies)|lock\s?file|package\.json|requirements\.txt)\b", 1.5),
        (r"\b(branch|target\s+branch|base\s+branch)\b", 1.5),
        (r"\b(please\s+)?(split|break)\s+(this\s+)?(PR|pull\s+request)\b", 2.0),
        # observed vocabulary
        (r"\b(remove|delete|drop)\s+(this\s+)?\w*\s*from\s+this\s+PR\b", 2.5),
        (r"\b(this|that)\s+(file|change)\s+(should\s+not|shouldn'?t)\s+be\b", 2.0),
        (r"\bunrelated\s+(to\s+)?(this\s+)?(PR|change)\b", 2.0),
    ],
}

_COMPILED: dict[str, list[tuple[re.Pattern, float]]] = {
    cat: [(re.compile(p, re.IGNORECASE), w) for p, w in pats]
    for cat, pats in _LEXICON.items()
}

# Patterns that only count when the comment sits on a TEST file — they
# disambiguate generic wording that would otherwise read as production talk.
_TEST_CONTEXT_BOOST: dict[str, list[tuple[re.Pattern, float]]] = {
    "coverage": [
        (re.compile(r"\b(assert|expect)\w*\b", re.I), 0.5),
        (re.compile(r"\bcase\b", re.I), 0.5),
    ],
    "confidence": [
        (re.compile(r"\bmock\w*\b", re.I), 0.5),
    ],
}


# ---------------------------------------------------------------------------
# Tone lexicons
# ---------------------------------------------------------------------------

_CONSTRUCTIVE = [
    re.compile(r"\b(please|could\s+you|would\s+you|can\s+we|shall\s+we)\b", re.I),
    re.compile(r"\b(thanks?|thank\s+you|nice|great|good\s+(catch|point|work)|lgtm)\b", re.I),
    re.compile(r"\b(suggestion|consider|maybe|perhaps|might\s+want|what\s+do\s+you\s+think|wdyt)\b", re.I),
    re.compile(r"\b(i\s+think|imo|imho|in\s+my\s+opinion|just\s+a\s+thought)\b", re.I),
    re.compile(r"\bnit\b|\boptional\b|\bfeel\s+free\b|\bup\s+to\s+you\b", re.I),
    re.compile(r"[👍🙏✅😄🎉💯]", re.I),
]

_CRITICAL = [
    re.compile(r"\b(wrong|broken|bad|terrible|awful|useless|pointless)\b", re.I),
    re.compile(r"\b(must|never|do\s+not|don'?t|should\s+not|shouldn'?t)\b", re.I),
    re.compile(r"\b(why\s+would\s+you|makes\s+no\s+sense|this\s+is\s+incorrect)\b", re.I),
    re.compile(r"!{2,}", re.I),
    re.compile(r"\b(blocker|blocking|reject|unacceptable)\b", re.I),
]

_SEVERITY = re.compile(
    r"\b(must|required|blocker|blocking|critical|security|vulnerab|urgent|"
    r"data\s+loss|breaking\s+change)\b",
    re.IGNORECASE,
)

_CODE_SUGGESTION = re.compile(r"```suggestion|```[a-z]*\n|^\s{4,}\S", re.MULTILINE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_QUOTE_LINE = re.compile(r"^\s*>.*$", re.MULTILINE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _strip_noise(body: str) -> str:
    """Remove code fences, quoted replies and HTML comments before matching."""
    txt = _HTML_COMMENT.sub(" ", body)
    txt = _CODE_FENCE.sub(" ", txt)
    txt = _QUOTE_LINE.sub(" ", txt)
    return txt


def score_role(raw_body: str, clean_text: str) -> str:
    """feedback | acknowledgement | code_suggestion (see REVIEW_ROLES)."""
    if _BARE_SUGGESTION.match(raw_body.strip()):
        return "code_suggestion"
    if not clean_text.strip():
        # all prose was inside code fences → a suggestion delivered as a diff
        return "code_suggestion"
    for pat in _ACK_PATTERNS:
        if pat.search(clean_text):
            return "acknowledgement"
    return "feedback"


def score_categories(text: str, on_test_file: bool) -> dict[str, float]:
    scores = {c: 0.0 for c in REVIEW_CATEGORIES}
    for cat, patterns in _COMPILED.items():
        for pat, weight in patterns:
            if pat.search(text):
                scores[cat] += weight
    if on_test_file:
        for cat, patterns in _TEST_CONTEXT_BOOST.items():
            for pat, weight in patterns:
                if pat.search(text):
                    scores[cat] += weight
    # Terse imperative with no diagnostic content — only when nothing else fired
    if max(scores.values()) <= 0:
        stripped = text.strip()
        if any(p.match(stripped) for p in _DIRECTIVE_PATTERNS):
            scores["directive"] = 1.0
    return scores


def pick_category(scores: dict[str, float]) -> str:
    best = max(scores.values())
    if best <= 0:
        return "unclassified"
    tied = [c for c, v in scores.items() if v == best]
    if len(tied) == 1:
        return tied[0]
    for cat in CATEGORY_PRIORITY:
        if cat in tied:
            return cat
    return tied[0]


def score_tone(text: str) -> int:
    pos = sum(1 for p in _CONSTRUCTIVE if p.search(text))
    neg = sum(1 for p in _CRITICAL if p.search(text))
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def score_intensity(raw_body: str, clean_text: str) -> int:
    """0–4 composite: verbosity, code suggestion, questions, severity."""
    score = 0
    if len(clean_text.split()) > 50:
        score += 1
    if _CODE_SUGGESTION.search(raw_body):
        score += 1
    if clean_text.count("?") >= 1:
        score += 1
    if _SEVERITY.search(clean_text):
        score += 1
    return score


# ---------------------------------------------------------------------------
# Manual coding sample (protocol §4.3.1 — Cohen's kappa)
# ---------------------------------------------------------------------------

def export_manual_sample(
    df: pd.DataFrame, sample_size: int, seed: int
) -> pd.DataFrame:
    """Stratified sample (category × artifact_kind) for two-rater coding."""
    pool = df[df["body"].notna()].copy()
    # Oversample test-file comments: they are the RQ's object of study
    test_pool = pool[pool["is_test_comment"]]
    prod_pool = pool[~pool["is_test_comment"]]

    n_test = min(len(test_pool), int(sample_size * 0.6))
    n_prod = min(len(prod_pool), sample_size - n_test)

    def _strat(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        """Proportional allocation across categories, ≥1 row per category.

        Iterates the groups directly: groupby().apply() drops the grouping
        column in pandas ≥2.2, which would strip `category` from the output.
        """
        if frame.empty or n <= 0:
            return frame.head(0)
        frac = n / len(frame)
        parts = []
        for _, g in frame.groupby("category"):
            k = min(len(g), max(1, int(round(len(g) * frac))))
            parts.append(g.sample(k, random_state=seed))
        if not parts:
            return frame.head(0)
        return pd.concat(parts).head(n)

    sample = pd.concat([_strat(test_pool, n_test), _strat(prod_pool, n_prod)])
    sample = sample.sample(frac=1.0, random_state=seed)  # shuffle for blinding

    cols = [
        "id", "pr_id", "agent", "artifact_kind", "path", "is_bot",
        "body", "role", "category", "tone", "intensity", "resolution",
    ]
    out = sample[cols].copy()
    # Blank columns for the two human raters + reconciliation
    out["rater1_category"] = ""
    out["rater2_category"] = ""
    out["reconciled_category"] = ""
    out["rater_notes"] = ""
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sample_size: int, seed: int, limit: int | None) -> None:
    if not COMMENTS_PARQUET.exists():
        print(f"ERROR: {COMMENTS_PARQUET} not found. Run rq4_00 first.")
        sys.exit(1)

    print("Loading rq4_comments.parquet...")
    df = pd.read_parquet(COMMENTS_PARQUET)
    if limit:
        df = df.head(limit)
    print(f"  {len(df):,} comments")

    bodies = df["body"].fillna("").astype(str)
    clean = bodies.apply(_strip_noise)
    on_test = df["is_test_comment"].fillna(False).tolist()

    print("Scoring comment role (feedback vs acknowledgement)...")
    df["role"] = [
        score_role(b, c) for b, c in zip(bodies.tolist(), clean.tolist())
    ]
    print(f"  {df['role'].value_counts().to_dict()}")

    print("Scoring taxonomy categories...")
    all_scores = [
        score_categories(t, tf) for t, tf in zip(clean.tolist(), on_test)
    ]
    df["category"] = [pick_category(s) for s in all_scores]
    # The taxonomy describes reviewer feedback; agent fix-reports and bare
    # code suggestions carry no diagnostic prose to categorise.
    df.loc[df["role"] != "feedback", "category"] = "n/a_" + df.loc[
        df["role"] != "feedback", "role"
    ]
    df["category_scores_json"] = [
        json.dumps({k: round(v, 2) for k, v in s.items()}) for s in all_scores
    ]
    df["category_confidence"] = [max(s.values()) for s in all_scores]

    print("Scoring tone and intensity...")
    df["tone"] = clean.apply(score_tone)
    df["intensity"] = [
        score_intensity(b, c) for b, c in zip(bodies.tolist(), clean.tolist())
    ]
    df["body_words"] = clean.apply(lambda t: len(t.split()))

    print("Deriving resolution proxy...")
    df["resolution"] = df["has_reply"].map(
        {True: "answered", False: "unanswered"}
    ).fillna("unanswered")

    df.to_parquet(CLASSIFIED_PARQUET, index=False)
    print(f"\nSaved {len(df):,} classified comments → {CLASSIFIED_PARQUET}")

    print(f"\nExporting manual coding sample (n≈{sample_size})...")
    sample = export_manual_sample(df, sample_size, seed)
    sample.to_csv(SAMPLE_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(sample):,} rows → {SAMPLE_CSV}")

    # -- Reporting --
    fb = df[df["role"] == "feedback"]

    print("\n--- Comment roles ---")
    print(df["role"].value_counts().to_string())
    print(f"  feedback share: {len(fb) / len(df) * 100:.1f}%")

    print("\n--- Category distribution (feedback comments only) ---")
    print(fb["category"].value_counts().to_string())

    print("\n--- Category distribution: TEST vs PRODUCTION (feedback only) ---")
    sub = fb[fb["artifact_kind"].isin(["test", "production"])]
    pivot = (
        pd.crosstab(sub["category"], sub["artifact_kind"], normalize="columns")
        * 100
    ).round(1)
    print(pivot.to_string())

    print("\n--- Human reviewers only, test files (feedback only) ---")
    ht = fb[fb["is_test_comment"] & ~fb["is_bot"].fillna(False)]
    print(f"  n = {len(ht):,}")
    if not ht.empty:
        print(ht["category"].value_counts().to_string())

    print("\n--- Tone / intensity / resolution ---")
    print(f"  tone       mean={df['tone'].mean():+.3f}  "
          f"{df['tone'].value_counts().to_dict()}")
    print(f"  intensity  mean={df['intensity'].mean():.3f}")
    print(f"  resolution {df['resolution'].value_counts().to_dict()}")

    unclassified = (fb["category"] == "unclassified").mean() * 100
    print(f"\nUnclassified among feedback comments: {unclassified:.1f}%")

    log = pd.DataFrame([{
        "n_comments": len(df),
        "n_feedback": int((df["role"] == "feedback").sum()),
        "n_acknowledgement": int((df["role"] == "acknowledgement").sum()),
        "n_code_suggestion": int((df["role"] == "code_suggestion").sum()),
        "n_test_comments": int(df["is_test_comment"].sum()),
        "n_bot_comments": int(df["is_bot"].fillna(False).sum()),
        "pct_unclassified_of_feedback": round(unclassified, 2),
        "mean_tone": round(float(df["tone"].mean()), 4),
        "mean_intensity": round(float(df["intensity"].mean()), 4),
        "manual_sample_size": len(sample),
        "seed": seed,
    }])
    log.to_csv(LOG_CSV, index=False)
    print(f"Log → {LOG_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4 comment classifier")
    parser.add_argument("--sample-size", type=int, default=300, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    args = parser.parse_args()
    main(sample_size=args.sample_size, seed=args.seed, limit=args.limit)
