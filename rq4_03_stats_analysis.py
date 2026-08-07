#!/usr/bin/env python3
"""
RQ4 — Script 03: Statistical Analysis and Publication Figures
==============================================================
Bateria estatística para responder à RQ4:

  1. Distribuição de categorias — teste vs. produção (qui-quadrado + Cramér's V)
     → hipótese central: o feedback em artefatos de teste segue padrões
       distintos do feedback em código de produção
  2. Tom e intensidade — teste vs. produção (Mann-Whitney U + Cliff's delta)
  3. Revisores humanos vs. bots em arquivos de teste (qui-quadrado)
  4. Resolução por categoria (taxa de resposta + qui-quadrado)
  5. Regressão logística: merged ~ categorias + TDT + controles (§3.6.3)
  6. Correlações de Spearman entre feedback e qualidade (MSI, TDT)
  7. Kruskal-Wallis entre agentes

Gera 5 figuras de publicação.

Inputs
------
  AIDev/rq4_comments_classified.parquet
  AIDev/rq4_pr_features.parquet

Outputs
-------
  AIDev/rq4_stats_summary.csv
  AIDev/rq4_logit_coefficients.csv
  figs/rq4_category_test_vs_prod.png
  figs/rq4_tone_intensity.png
  figs/rq4_logit_forest.png
  figs/rq4_correlation_heatmap.png
  figs/rq4_resolution_by_category.png

Usage
-----
  python rq4_03_stats_analysis.py
  python rq4_03_stats_analysis.py --no-figs
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from helper import (  # noqa: E402
    FIG_DIR,
    NAME_MAPPING,
    REVIEW_CATEGORIES,
    REVIEW_CATEGORY_LABELS,
    mannUandCliffdelta,
)

OUT_DIR = ROOT_DIR / "AIDev"
FIG_DIR.mkdir(exist_ok=True)

CLASSIFIED_PARQUET = OUT_DIR / "rq4_comments_classified.parquet"
FEATURES_PARQUET = OUT_DIR / "rq4_pr_features.parquet"
STATS_CSV = OUT_DIR / "rq4_stats_summary.csv"
LOGIT_CSV = OUT_DIR / "rq4_logit_coefficients.csv"

ALL_CATEGORIES = REVIEW_CATEGORIES + ["unclassified"]
_MIN_SAMPLES = 5
_MIN_LOGIT_PRS = 50


def _cramers_v(chi2: float, contingency: np.ndarray) -> float:
    n = contingency.sum()
    k = min(contingency.shape) - 1
    if n == 0 or k == 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * k)))


def _effect_label(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return ("negligible" if v < 0.1 else "small" if v < 0.3
            else "medium" if v < 0.5 else "large")


def _run_battery(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str,
    label_b: str,
    metric: str,
    all_stats: list[dict],
) -> None:
    a = pd.Series(series_a).dropna().tolist()
    b = pd.Series(series_b).dropna().tolist()
    if len(a) < _MIN_SAMPLES or len(b) < _MIN_SAMPLES:
        print(f"  [skip] {label_a} vs {label_b} — too few samples "
              f"({len(a)}, {len(b)})")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, p, d, size = mannUandCliffdelta(a, b)
    all_stats.append({
        "metric": metric, "group_a": label_a, "group_b": label_b,
        "n_a": len(a), "n_b": len(b),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "stat": u, "p_value": p, "effect": d, "effect_size": size,
    })


# ---------------------------------------------------------------------------
# 1. Category distribution — test vs production
# ---------------------------------------------------------------------------

def test_vs_production_categories(
    comments: pd.DataFrame, all_stats: list[dict]
) -> pd.DataFrame:
    print("\n=== 1. Category distribution — TEST vs PRODUCTION ===")
    print("  (feedback-role comments only; agent fix-reports excluded)")
    sub = comments[
        comments["artifact_kind"].isin(["test", "production"])
        & comments["role"].eq("feedback")
    ]
    ct = pd.crosstab(sub["category"], sub["artifact_kind"])
    ct = ct.reindex(ALL_CATEGORIES, fill_value=0)
    ct = ct.loc[ct.sum(axis=1) > 0]

    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print("  [skip] insufficient variety")
        return ct

    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    v = _cramers_v(chi2, ct.values)
    print(f"  chi2={chi2:.2f}  dof={dof}  p={p:.3e}  Cramér's V={v:.3f} "
          f"({_effect_label(v)})")
    all_stats.append({
        "metric": "category_test_vs_prod_chi2",
        "group_a": "test files", "group_b": "production files",
        "n_a": int(ct["test"].sum()) if "test" in ct else 0,
        "n_b": int(ct["production"].sum()) if "production" in ct else 0,
        "stat": chi2, "p_value": p, "effect": v, "effect_size": _effect_label(v),
    })

    pct = (ct / ct.sum() * 100).round(2)
    print("\n  % distribution:")
    print(pct.to_string())

    # Per-category standardised residuals: which categories drive the effect?
    expected = stats.contingency.expected_freq(ct.values)
    resid = (ct.values - expected) / np.sqrt(expected)
    resid_df = pd.DataFrame(resid, index=ct.index, columns=ct.columns).round(2)
    print("\n  standardised residuals (>|2| = notable):")
    print(resid_df.to_string())

    for cat in ct.index:
        if "test" not in ct.columns:
            continue
        all_stats.append({
            "metric": "category_share_test_vs_prod",
            "group_a": f"{cat} (test)", "group_b": f"{cat} (production)",
            "n_a": int(ct.loc[cat, "test"]),
            "n_b": int(ct.loc[cat, "production"]),
            "mean_a": float(pct.loc[cat, "test"]),
            "mean_b": float(pct.loc[cat, "production"]),
            "effect": float(resid_df.loc[cat, "test"]),
            "effect_size": "std_residual",
        })
    return ct


# ---------------------------------------------------------------------------
# 2. Tone / intensity / resolution — test vs production
# ---------------------------------------------------------------------------

def tone_intensity_tests(
    comments: pd.DataFrame, all_stats: list[dict]
) -> None:
    print("\n=== 2. Tone / intensity — TEST vs PRODUCTION ===")
    test = comments[comments["artifact_kind"].eq("test")]
    prod = comments[comments["artifact_kind"].eq("production")]

    for col, metric in (("tone", "tone_test_vs_prod"),
                        ("intensity", "intensity_test_vs_prod"),
                        ("body_words", "verbosity_test_vs_prod")):
        if col not in comments.columns:
            continue
        print(f"  -- {col} --")
        _run_battery(test[col], prod[col], "test files",
                     "production files", metric, all_stats)

    print("\n=== 2b. Human reviewers only — TEST vs PRODUCTION ===")
    h = comments[~comments["is_bot"].fillna(False)]
    ht = h[h["artifact_kind"].eq("test")]
    hp = h[h["artifact_kind"].eq("production")]
    for col, metric in (("tone", "tone_human_test_vs_prod"),
                        ("intensity", "intensity_human_test_vs_prod")):
        if col not in comments.columns:
            continue
        print(f"  -- {col} (humans) --")
        _run_battery(ht[col], hp[col], "test files (human)",
                     "production files (human)", metric, all_stats)


# ---------------------------------------------------------------------------
# 3. Human vs bot reviewers on test files
# ---------------------------------------------------------------------------

def human_vs_bot(comments: pd.DataFrame, all_stats: list[dict]) -> None:
    print("\n=== 3. Human vs bot reviewers on TEST files ===")
    test = comments[
        comments["artifact_kind"].eq("test") & comments["role"].eq("feedback")
    ].copy()
    if test.empty:
        print("  [skip] no test comments")
        return
    test["reviewer_kind"] = np.where(
        test["is_bot"].fillna(False), "bot", "human"
    )
    ct = pd.crosstab(test["category"], test["reviewer_kind"])
    ct = ct.loc[ct.sum(axis=1) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print("  [skip] insufficient variety")
        return
    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    v = _cramers_v(chi2, ct.values)
    print(f"  chi2={chi2:.2f}  dof={dof}  p={p:.3e}  Cramér's V={v:.3f} "
          f"({_effect_label(v)})")
    print((ct / ct.sum() * 100).round(1).to_string())
    all_stats.append({
        "metric": "category_human_vs_bot_test_chi2",
        "group_a": "human", "group_b": "bot",
        "n_a": int(ct.get("human", pd.Series(0)).sum()),
        "n_b": int(ct.get("bot", pd.Series(0)).sum()),
        "stat": chi2, "p_value": p, "effect": v,
        "effect_size": _effect_label(v),
    })


# ---------------------------------------------------------------------------
# 3b. Role distribution — does the agent respond differently on tests?
# ---------------------------------------------------------------------------

def role_distribution(comments: pd.DataFrame, all_stats: list[dict]) -> None:
    print("\n=== 3b. Comment role — TEST vs PRODUCTION ===")
    sub = comments[comments["artifact_kind"].isin(["test", "production"])]
    ct = pd.crosstab(sub["role"], sub["artifact_kind"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print("  [skip] insufficient variety")
        return
    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    v = _cramers_v(chi2, ct.values)
    print((ct / ct.sum() * 100).round(1).to_string())
    print(f"  chi2={chi2:.2f}  dof={dof}  p={p:.3e}  Cramér's V={v:.3f} "
          f"({_effect_label(v)})")
    all_stats.append({
        "metric": "role_test_vs_prod_chi2",
        "group_a": "test files", "group_b": "production files",
        "n_a": int(ct["test"].sum()) if "test" in ct else 0,
        "n_b": int(ct["production"].sum()) if "production" in ct else 0,
        "stat": chi2, "p_value": p, "effect": v,
        "effect_size": _effect_label(v),
    })


# ---------------------------------------------------------------------------
# 4. Resolution by category
# ---------------------------------------------------------------------------

def resolution_analysis(
    comments: pd.DataFrame, all_stats: list[dict]
) -> pd.DataFrame:
    print("\n=== 4. Resolution (reply) rate by category ===")
    df = comments.copy()
    df["answered"] = df["resolution"].eq("answered")

    by_cat = (
        df.groupby(["artifact_kind", "category"])["answered"]
        .agg(["mean", "count"])
        .reset_index()
    )
    by_cat = by_cat[by_cat["artifact_kind"].isin(["test", "production"])]
    by_cat["mean"] = (by_cat["mean"] * 100).round(1)
    print(by_cat.pivot(index="category", columns="artifact_kind",
                       values="mean").to_string())

    test = df[df["artifact_kind"].eq("test")]
    prod = df[df["artifact_kind"].eq("production")]
    if len(test) >= _MIN_SAMPLES and len(prod) >= _MIN_SAMPLES:
        ct = np.array([
            [test["answered"].sum(), (~test["answered"]).sum()],
            [prod["answered"].sum(), (~prod["answered"]).sum()],
        ])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = _cramers_v(chi2, ct)
        print(f"\n  test answered={test['answered'].mean() * 100:.1f}%  "
              f"prod answered={prod['answered'].mean() * 100:.1f}%")
        print(f"  chi2={chi2:.2f}  p={p:.3e}  Cramér's V={v:.3f}")
        all_stats.append({
            "metric": "resolution_test_vs_prod_chi2",
            "group_a": "test files", "group_b": "production files",
            "n_a": len(test), "n_b": len(prod),
            "mean_a": float(test["answered"].mean()),
            "mean_b": float(prod["answered"].mean()),
            "stat": chi2, "p_value": p, "effect": v,
            "effect_size": _effect_label(v),
        })
    return by_cat


# ---------------------------------------------------------------------------
# 5. Logistic regression — merge outcome
# ---------------------------------------------------------------------------

def logistic_regression(pr_df: pd.DataFrame, all_stats: list[dict]) -> pd.DataFrame:
    print("\n=== 5. Logistic regression — merged ~ feedback + quality ===")

    df = pr_df.copy()
    df["merged"] = df["merged"].astype(float)

    predictors: list[str] = []
    for c in REVIEW_CATEGORIES:
        col = f"cat_{c}"
        if col in df.columns and df[col].sum() > 0:
            predictors.append(col)

    for extra in ("n_test_comments", "mean_intensity", "resolution_rate"):
        if extra in df.columns and df[extra].notna().any():
            predictors.append(extra)

    # Controls
    if "pr_churn" in df.columns and df["pr_churn"].notna().any():
        df["log_pr_churn"] = np.log1p(df["pr_churn"].fillna(0))
        predictors.append("log_pr_churn")
    if "tdt_last" in df.columns and df["tdt_last"].notna().sum() >= _MIN_LOGIT_PRS:
        df["tdt_last_filled"] = df["tdt_last"].fillna(df["tdt_last"].median())
        predictors.append("tdt_last_filled")
    else:
        print("  [note] TDT has too few PRs to enter the model")

    if "msi_global" in df.columns:
        n_msi = df["msi_global"].notna().sum()
        if n_msi >= _MIN_LOGIT_PRS:
            df["msi_filled"] = df["msi_global"].fillna(df["msi_global"].median())
            predictors.append("msi_filled")
        else:
            print(f"  [note] MSI available for only {n_msi} PRs — excluded "
                  f"from the model (threshold {_MIN_LOGIT_PRS})")

    # Agent dummies (drop the largest as reference)
    agent_dummies = pd.get_dummies(df["agent"], prefix="agent", dtype=float)
    if not agent_dummies.empty:
        ref = df["agent"].value_counts().idxmax()
        ref_col = f"agent_{ref}"
        if ref_col in agent_dummies.columns:
            agent_dummies = agent_dummies.drop(columns=[ref_col])
        print(f"  agent reference category: {ref}")
        df = pd.concat([df, agent_dummies], axis=1)
        predictors += list(agent_dummies.columns)

    model_df = df[predictors + ["merged"]].dropna()
    if len(model_df) < _MIN_LOGIT_PRS:
        print(f"  [skip] only {len(model_df)} complete cases")
        return pd.DataFrame()

    X = sm.add_constant(model_df[predictors].astype(float))
    y = model_df["merged"].astype(float)
    print(f"  n={len(model_df):,}  merged={y.mean() * 100:.1f}%  "
          f"predictors={len(predictors)}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:
        print(f"  [error] model failed to converge: {exc}")
        return pd.DataFrame()

    coefs = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "z": res.tvalues.values,
        "p_value": res.pvalues.values,
    })
    coefs["odds_ratio"] = np.exp(coefs["coef"])
    coefs["or_ci_low"] = np.exp(coefs["coef"] - 1.96 * coefs["std_err"])
    coefs["or_ci_high"] = np.exp(coefs["coef"] + 1.96 * coefs["std_err"])
    _, p_adj, _, _ = multipletests(coefs["p_value"], method="bonferroni")
    coefs["p_bonferroni"] = p_adj
    coefs["significant"] = coefs["p_bonferroni"] < 0.05

    print(f"  pseudo R²={res.prsquared:.4f}  LLR p={res.llr_pvalue:.3e}")
    print("\n" + coefs[["term", "coef", "odds_ratio", "p_value",
                        "p_bonferroni", "significant"]].round(4).to_string(index=False))

    all_stats.append({
        "metric": "logit_model_fit",
        "group_a": "merged ~ feedback + quality + controls",
        "n_a": len(model_df),
        "stat": res.prsquared, "p_value": res.llr_pvalue,
        "effect": res.llf, "effect_size": "pseudo_r2",
    })
    return coefs


# ---------------------------------------------------------------------------
# 6. Correlations with quality metrics
# ---------------------------------------------------------------------------

def quality_correlations(pr_df: pd.DataFrame, all_stats: list[dict]) -> pd.DataFrame:
    print("\n=== 6. Spearman correlations — feedback vs quality ===")
    feedback_cols = (
        [f"test_cat_{c}" for c in REVIEW_CATEGORIES]
        + ["n_test_comments", "mean_tone", "mean_intensity", "resolution_rate"]
    )
    feedback_cols = [c for c in feedback_cols if c in pr_df.columns]
    quality_cols = [c for c in ("tdt_last", "delta_tdt", "tdt_max",
                                "msi_global", "assertion_count")
                    if c in pr_df.columns]

    rows = []
    for q in quality_cols:
        sub = pr_df[pr_df[q].notna()]
        if len(sub) < 10:
            print(f"  [skip] {q}: only {len(sub)} PRs")
            continue
        print(f"  -- {q} (n={len(sub):,}) --")
        for f in feedback_cols:
            vals = sub[[f, q]].dropna()
            if len(vals) < 10 or vals[f].nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = stats.spearmanr(vals[f], vals[q])
            rows.append({"quality": q, "feedback": f, "n": len(vals),
                         "spearman_rho": rho, "p_value": p})
    if not rows:
        return pd.DataFrame()

    corr = pd.DataFrame(rows)
    _, p_adj, _, _ = multipletests(corr["p_value"].fillna(1.0),
                                   method="bonferroni")
    corr["p_bonferroni"] = p_adj
    corr["significant"] = corr["p_bonferroni"] < 0.05
    sig = corr[corr["significant"]]
    print(f"\n  {len(sig)} / {len(corr)} correlations significant after "
          f"Bonferroni")
    if not sig.empty:
        print(sig.round(4).to_string(index=False))

    for _, r in corr.iterrows():
        all_stats.append({
            "metric": "spearman_feedback_quality",
            "group_a": r["feedback"], "group_b": r["quality"],
            "n_a": int(r["n"]), "stat": r["spearman_rho"],
            "p_value": r["p_value"], "effect": r["spearman_rho"],
            "effect_size": "spearman_rho",
        })
    return corr


# ---------------------------------------------------------------------------
# 7. Per-agent differences
# ---------------------------------------------------------------------------

def agent_differences(pr_df: pd.DataFrame, all_stats: list[dict]) -> None:
    print("\n=== 7. Kruskal-Wallis across agents ===")
    for col in ("n_test_comments", "mean_tone", "mean_intensity",
                "resolution_rate"):
        if col not in pr_df.columns:
            continue
        groups = [
            g[col].dropna().tolist()
            for _, g in pr_df.groupby("agent")
            if g[col].notna().sum() >= _MIN_SAMPLES
        ]
        if len(groups) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h, p = stats.kruskal(*groups)
        print(f"  {col:<22} H={h:8.3f}  p={p:.4e}")
        all_stats.append({
            "metric": f"kruskal_agent_{col}", "group_a": "all agents",
            "n_a": sum(len(g) for g in groups), "stat": h, "p_value": p,
        })


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_category_test_vs_prod(ct: pd.DataFrame) -> None:
    if ct.empty or "test" not in ct.columns:
        return
    pct = ct / ct.sum() * 100
    cats = [c for c in ALL_CATEGORIES if c in pct.index]
    labels = [REVIEW_CATEGORY_LABELS.get(c, c) for c in cats]
    x = np.arange(len(cats))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, pct.loc[cats, "test"], width,
           label="Test files", color="#DC267F", alpha=0.85)
    ax.bar(x + width / 2, pct.loc[cats, "production"], width,
           label="Production files", color="#0072B2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% of comments", fontsize=10)
    ax.set_title("Review-comment categories: test vs production artifacts (RQ4)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = FIG_DIR / "rq4_category_test_vs_prod.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def fig_tone_intensity(comments: pd.DataFrame) -> None:
    kinds = ["test", "production"]
    colors = ["#DC217F", "#0072B2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, col, title in zip(
        axes, ["tone", "intensity", "body_words"],
        ["Tone (-1 critical … +1 constructive)", "Intensity (0–4)",
         "Verbosity (words)"],
    ):
        if col not in comments.columns:
            continue
        data = [comments[comments["artifact_kind"].eq(k)][col].dropna().tolist()
                for k in kinds]
        bp = ax.boxplot([d if d else [0] for d in data],
                        tick_labels=["Test", "Production"],
                        patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        if col == "body_words":
            ax.set_yscale("log")

    fig.suptitle("Feedback tone, intensity and verbosity by artifact kind (RQ4)",
                 fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / "rq4_tone_intensity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def fig_logit_forest(coefs: pd.DataFrame) -> None:
    if coefs.empty:
        return
    df = coefs[coefs["term"] != "const"].copy()
    if df.empty:
        return
    df = df.sort_values("odds_ratio")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(df))))
    colors = ["#D55E00" if s else "#888888" for s in df["significant"]]
    ax.errorbar(
        df["odds_ratio"], y,
        xerr=[df["odds_ratio"] - df["or_ci_low"],
              df["or_ci_high"] - df["odds_ratio"]],
        fmt="o", markersize=5, capsize=3, linewidth=1.2, ecolor="#999999",
        linestyle="none", color="black", zorder=3,
    )
    ax.scatter(df["odds_ratio"], y, c=colors, s=45, zorder=4)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["term"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for PR being merged (log scale, 95% CI)",
                  fontsize=9)
    ax.set_title("Logistic regression: what predicts merge? (RQ4)", fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = FIG_DIR / "rq4_logit_forest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def fig_correlation_heatmap(corr: pd.DataFrame) -> None:
    if corr.empty:
        return
    pivot = corr.pivot(index="feedback", columns="quality",
                       values="spearman_rho")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * pivot.shape[1] + 3),
                                    max(4, 0.42 * pivot.shape[0] + 1)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                   aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > 0.3 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("Feedback patterns vs test quality (RQ4)", fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / "rq4_correlation_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def fig_resolution_by_category(by_cat: pd.DataFrame) -> None:
    if by_cat.empty:
        return
    pivot = by_cat.pivot(index="category", columns="artifact_kind",
                         values="mean")
    cats = [c for c in ALL_CATEGORIES if c in pivot.index]
    if not cats:
        return
    pivot = pivot.loc[cats]
    labels = [REVIEW_CATEGORY_LABELS.get(c, c) for c in cats]
    x = np.arange(len(cats))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    if "test" in pivot.columns:
        ax.bar(x - width / 2, pivot["test"].fillna(0), width,
               label="Test files", color="#DC267F", alpha=0.85)
    if "production" in pivot.columns:
        ax.bar(x + width / 2, pivot["production"].fillna(0), width,
               label="Production files", color="#0072B2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% of comments that received a reply", fontsize=10)
    ax.set_title("Resolution proxy by comment category (RQ4)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = FIG_DIR / "rq4_resolution_by_category.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(no_figs: bool) -> None:
    for path in (CLASSIFIED_PARQUET, FEATURES_PARQUET):
        if not path.exists():
            print(f"ERROR: {path} not found. Run preceding scripts first.")
            sys.exit(1)

    print("Loading classified comments...")
    comments = pd.read_parquet(CLASSIFIED_PARQUET)
    comments["agent"] = comments["agent"].map(NAME_MAPPING).fillna(
        comments["agent"])
    print(f"  {len(comments):,} comments | {comments['pr_id'].nunique():,} PRs")

    print("Loading PR features...")
    pr_df = pd.read_parquet(FEATURES_PARQUET)
    pr_df["agent"] = pr_df["agent"].map(NAME_MAPPING).fillna(pr_df["agent"])
    print(f"  {len(pr_df):,} PRs | merged={pr_df['merged'].mean() * 100:.1f}%")

    all_stats: list[dict] = []

    ct = test_vs_production_categories(comments, all_stats)
    tone_intensity_tests(comments, all_stats)
    human_vs_bot(comments, all_stats)
    role_distribution(comments, all_stats)
    by_cat = resolution_analysis(comments, all_stats)
    coefs = logistic_regression(pr_df, all_stats)
    corr = quality_correlations(pr_df, all_stats)
    agent_differences(pr_df, all_stats)

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"\nSaved {len(stats_df):,} test results → {STATS_CSV}")

    if not coefs.empty:
        coefs.to_csv(LOGIT_CSV, index=False)
        print(f"Saved logit coefficients → {LOGIT_CSV}")

    if not no_figs:
        print("\n--- Generating figures ---")
        for fn, args in (
            (fig_category_test_vs_prod, (ct,)),
            (fig_tone_intensity, (comments,)),
            (fig_logit_forest, (coefs,)),
            (fig_correlation_heatmap, (corr,)),
            (fig_resolution_by_category, (by_cat,)),
        ):
            try:
                fn(*args)
            except Exception as exc:
                print(f"  Warning: {fn.__name__} failed: {exc}")

    print("\n--- Key Results ---")
    chi_row = stats_df[stats_df["metric"] == "category_test_vs_prod_chi2"]
    if not chi_row.empty:
        r = chi_row.iloc[0]
        print(f"  Test vs production category distribution: "
              f"chi2={r['stat']:.1f}, p={r['p_value']:.2e}, "
              f"V={r['effect']:.3f} ({r['effect_size']})")
    if not coefs.empty:
        sig = coefs[coefs["significant"] & (coefs["term"] != "const")]
        print(f"  Significant merge predictors (Bonferroni): {len(sig)}")
        for _, r in sig.iterrows():
            print(f"    {r['term']:<28} OR={r['odds_ratio']:.3f} "
                  f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4 statistical analysis")
    parser.add_argument("--no-figs", action="store_true")
    args = parser.parse_args()
    main(no_figs=args.no_figs)
