<p align="center">
  <img src="figs/aidev_logo.png" alt="Description" width="300"/>
</p>

# AIDev: Studying AI Coding Agents on GitHub (The Rise of AI Teammates in Software Engineering 3.0)

📢 We’re hosting the **MSR 2026 Mining Challenge** (co-located with **ICSE 2026** in **Rio de Janeiro, Brazil**). Details and submissions:  
⚠️⚠️⚠️ [https://2026.msrconf.org/track/msr-2026-mining-challenge](https://2026.msrconf.org/track/msr-2026-mining-challenge) ⚠️⚠️⚠️

[![Paper](https://img.shields.io/badge/arXiv-2507.15003-b31b1b.svg)](https://arxiv.org/abs/2507.15003)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-dataset-blue?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/hao-li/AIDev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16919272.svg)](https://doi.org/10.5281/zenodo.16919272)

- **Paper:** https://arxiv.org/abs/2507.15003
- **HuggingFace Dataset:** https://huggingface.co/datasets/hao-li/AIDev
- **Example Notebooks:** 

| Description                          | Notebook Link                                                                                       | Open in Colab                                                                                                                |
|--------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Basic usage                          | [load_AIDev.ipynb](analysis/load_AIDev.ipynb)                                                       | <a href="https://colab.research.google.com/github/SAILResearch/AI_Teammates_in_SE3/blob/main/analysis/load_AIDev.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Dataset overview                     | [dataset_overview.ipynb](analysis/dataset_overview.ipynb)                                           | <a href="https://colab.research.google.com/github/SAILResearch/AI_Teammates_in_SE3/blob/main/analysis/productivity.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Analysis of programming usage        | [language_usage.ipynb](analysis/language_usage.ipynb)                                               | <a href="https://colab.research.google.com/github/SAILResearch/AI_Teammates_in_SE3/blob/main/analysis/language_usage.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| PR merge rate and turnaround time    | [productivity.ipynb](analysis/productivity.ipynb)                                                   | <a href="https://colab.research.google.com/github/SAILResearch/AI_Teammates_in_SE3/blob/main/analysis/productivity.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

> ⚠️ **Update (Aug 10, 2025):** The dataset has been refreshed to include data up to **August 1, 2025**, ensuring 
> our dataset reflects the most recent trends in coding agents.

This repository contains the replication package for the paper "The Rise of AI Teammates in Software Engineering (SE)
3.0: How Autonomous Coding Agents Are Reshaping SE". Due to the size limit of GitHub repositories, the full dataset is
not included here. You can find our full dataset on HuggingFace: https://huggingface.co/datasets/hao-li/AIDev

> If you're interested in the raw data of AIDev-pop, you can find them here: https://drive.google.com/file/d/1l0_RjS7ZT0Y27V3mv0oJK-jfeRkhq5l5/view?usp=drive_link

## Overview

The overview of the AIDev dataset is as follows:

|                  | #PR         | #Developer | #Repo       |
| ---------------- | ----------- |------------|-------------|
| `OpenAI Codex`   | 814,522     | 61,653     | 84,704      |
| `Devin`          | 29,744      | NA         | 4,747       |
| `GitHub Copilot` | 50,447      | NA         | 14,492      |
| `Cursor`         | 32,941      | 9,658      | 12,699      |
| `Claude Code`    | 5,137       | 1,643      | 1,915       |
| **Total**        | **932,791** | **72,189** | **116,211** |

![](./figs/pr_cumulative_all.png)

## Repository Structure

```
├── AIDev-pop/              # AIDev-pop subset of AIDev
├── analysis/              # Analysis scripts and Jupyter notebooks
├── figs/                  # Generated figures and results
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Research Pipelines (GESAD/UECE — ESE Special Issue)

This repository includes two empirical mining pipelines investigating AI-generated test quality.
All scripts read from `OFFICIALDATASET.zip` (offline) or fall back to HuggingFace automatically.

### Prerequisites

```bash
pip install -r requirements.txt
# Copy .env.example → .env and set HF_TOKEN (only needed for online mode)
```

Place `OFFICIALDATASET.zip` (downloaded from [Zenodo](https://doi.org/10.5281/zenodo.16919272)) in the project root.

---

### RQ1 — Functional Effectiveness (Mutation Score Indicator)

> *Qual é a capacidade real de detecção de falhas dos testes gerados por agentes de IA, medida pelo MSI?*

| Script | Purpose |
|--------|---------|
| `rq1_00_explore_zenodo.py` | Inspect the ZIP without full extraction — lists files, reads parquet schemas, compares with HuggingFace |
| `rq1_sample_selection.py` | Extract initial candidate test PRs from AIDev (vitest / pytest patterns) |
| `rq1_01_expand_python_sample.py` | Expand sample to Python repos via `test_*.py` / `*_test.py` patterns |
| `rq1_02_eligibility_filter.py` | Apply 4 protocol criteria (CI, passing tests, containerisable, JS/TS or Python) + stratified sampling (200/agent, seed=42) |
| `rq1_clone_repos.py` | Clone repos to exact merge commit SHA, create git worktrees |
| `rq1_03_setup_stryker.py` | Prepare JS/TS worktrees: `npm install`, detect test runner, generate `stryker.config.json` |
| `rq1_04_run_stryker.py` | Execute Stryker, parse `stryker-report.json`, compute MSI global and per operator |
| `rq1_05_run_mutmut.py` | Create venv, install deps, run `mutmut`, classify mutation operators via diff |
| `rq1_06_aggregate_msi.py` | Unify results, compute cyclomatic complexity (lizard), count assertions, Spearman correlations |
| `rq1_07_stats_analysis.py` | Mann-Whitney U, Cliff's delta, Kruskal-Wallis, Bonferroni post-hoc + 3 figures |

**Execution order:**

```bash
python rq1_00_explore_zenodo.py
python rq1_sample_selection.py --offline
python rq1_01_expand_python_sample.py --offline
python rq1_02_eligibility_filter.py --offline

# Requires Docker and git
python rq1_clone_repos.py --csv AIDev/rq1_eligible.csv

# JS/TS and Python mutation testing can run in parallel
python rq1_03_setup_stryker.py &
python rq1_05_run_mutmut.py &
wait

python rq1_04_run_stryker.py
python rq1_06_aggregate_msi.py
python rq1_07_stats_analysis.py
```

**Outputs:** `AIDev/rq1_*.csv`, `figs/rq1_*.png`

---

### RQ2 — Test Technical Debt (TDT)

> *Quais padrões de dívida técnica emergem em testes agênticos, e como evoluem ao longo do ciclo de vida dos PRs?*

Detection is **patch-based** (no repo cloning required). Analyses 10 test smells across the unified catalog
(Palomba / Bavota / Verdecchia) and introduces two dimensions absent from prior work:
co-occurrence patterns (phi coefficient + Ward clustering) and temporal evolution (Delta-TDT).

| Script | Purpose |
|--------|---------|
| `rq2_00_build_corpus.py` | Build (pr_id, commit_sha) corpus of all commits touching test files; attaches agent, language, task_type |
| `rq2_01_detect_smells.py` | Detect 10 smells from unified diff patches (no cloning); supports `--resume` for large runs |
| `rq2_02_aggregate_tdt.py` | Compute weighted TDT index per commit and Delta-TDT per PR |
| `rq2_03_cooccurrence.py` | Phi coefficient matrix + Ward's hierarchical clustering to find composite smell patterns |
| `rq2_04_stats_analysis.py` | Full statistical battery (Mann-Whitney U, Cliff's delta, Kruskal-Wallis, Bonferroni) + 4 figures |

**Execution order:**

```bash
python rq2_00_build_corpus.py --offline          # ~few minutes
python rq2_01_detect_smells.py --offline --resume # longest step — resumable
python rq2_02_aggregate_tdt.py
python rq2_03_cooccurrence.py                     # can run in parallel with rq2_04
python rq2_04_stats_analysis.py
```

**Debug mode** (test with a small slice of data):

```bash
python rq2_00_build_corpus.py --offline --limit 5000
python rq2_01_detect_smells.py --offline --limit 1000 --resume
```

**Outputs:** `AIDev/rq2_*.{parquet,csv}`, `figs/rq2_*.png`

**Smell catalog (10 smells, weighted TDT index):**

| Smell | Source | Weight |
|-------|--------|--------|
| UnknownTest | Palomba | 0.15 |
| EagerTest | Palomba | 0.12 |
| ResourceOptimism | Verdecchia | 0.11 |
| AssertionRoulette | Palomba | 0.10 |
| MissingExceptionTest | Bavota | 0.09 |
| RedundantAssertion | Palomba | 0.08 |
| GeneralFixture | Verdecchia | 0.08 |
| MagicNumberTest | Palomba | 0.07 |
| VerboseTest | Bavota | 0.06 |
| DuplicateAssert | Palomba | 0.06 |

---

### RQ3 — Performance Modelling Strategies (PSQI)

> *Quais estratégias de modelagem e otimização de performance os agentes de IA adotam em commits do tipo `perf`,
> e como diferem das adotadas por desenvolvedores humanos?*

Detection is **patch-based** (no repo cloning). Introduces the **Performance Script Quality Index (PSQI)** —
five ordinal dimensions (0–2) derived from performance-engineering recommendations (Jain, 1991; Feitelson, 2015),
aggregated per PR and normalised to 0–10.

| Script | Purpose |
|--------|---------|
| `rq3_perf_exploration.py` | Feasibility probe — measures how many k6-specific PRs exist (produced `rq3_decision.txt`) |
| `rq3_00_build_corpus.py` | Build perf PR corpus from two signals: `task_type == perf` (A) ∪ perf filename patterns (B) |
| `rq3_01_extract_features.py` | Score the 5 PSQI dimensions from `+` lines of unified diffs; classify tool (k6/locust/jmeter/gatling/pytest-benchmark/generic) |
| `rq3_02_aggregate_psqi.py` | Aggregate file-level scores to PR level, compute PSQI (0–10), tool distribution |
| `rq3_03_stats_analysis.py` | Mann-Whitney U, Cliff's delta, Kruskal-Wallis + Bonferroni post-hoc, chi-square/Cramér's V + 4 figures |

**Execution order:**

```bash
python rq3_00_build_corpus.py --offline           # ~2 min
python rq3_01_extract_features.py --offline       # longest step — supports --resume
python rq3_02_aggregate_psqi.py
python rq3_03_stats_analysis.py
```

**Outputs:** `AIDev/rq3_*.{parquet,csv}`, `figs/rq3/rq3_*.png`

**PSQI dimensions (0–2 each; protocol §3.5.3):**

| Dimension | Score 0 | Score 1 | Score 2 |
|-----------|---------|---------|---------|
| `dim_load_type` | steady-state only | one pattern (ramp/spike/soak/stress) | ≥2 combined patterns |
| `dim_think_time` | absent | fixed sleep | variable/distribution |
| `dim_payload` | hardcoded | parameterised | external dataset |
| `dim_negative` | absent | one boundary/error signal | ≥2 signals |
| `dim_sla` | absent | raw threshold | SLO/percentile-linked |

**Corpus (1,533 PRs):**

| Agent | PRs in corpus | PRs with scoreable diffs |
|-------|--------------:|-------------------------:|
| OpenAI Codex | 1,007 | 803 |
| GitHub Copilot | 209 | 162 |
| Devin | 164 | 107 |
| Cursor | 43 | 21 |
| Claude Code | 22 | 18 |
| Human | 88 | **0** — see limitations |

> ⚠️ **Known limitations of the current RQ3 run**
>
> 1. **No human comparison group is possible offline.** `pr_commit_details.parquet` contains diffs for
>    33,580 agentic PRs and **zero** human PRs, so all AI-vs-Human tests in `rq3_03` are skipped. Answering the
>    comparative half of RQ3 requires fetching human PR diffs from the GitHub API.
> 2. **Corpus precision is low.** The broad path rule `[/_-](perf|load|stress|benchmark)[/_-]` captures 60.6% of
>    files on its own, and 61.3% of captured files use *load* in the data-loading/ETL sense
>    (e.g. `airbyte-cdk/bulk/core/load/...`) rather than load testing; 26.4% are docs/config/fixtures rather than
>    scripts. Only ~24% (969 files across 194 PRs) are plausibly genuine performance scripts.
> 3. **Genuine load-testing tooling is nearly absent** — 10 k6 files and 7 pytest-benchmark files in the whole
>    corpus, consistent with the `rq3_decision.txt` feasibility probe.

---

### RQ4 — Human-Agent Collaboration in Review

> *Quais padrões de feedback de revisão (tipo, tom, intensidade, resolução) emergem em PRs com artefatos de teste
> agênticos, e como se correlacionam com MSI, TDT e as decisões de merge?*

Applies the **Bacchelli & Bird (2013)** review-comment taxonomy, adapted to test artifacts (protocol §3.6.2).
The protocol assumed GitHub API extraction; the dataset already ships the review comments, so the pipeline runs
fully offline.

| Script | Purpose |
|--------|---------|
| `rq4_00_build_corpus.py` | Unify both review-comment snapshots, link to PRs by URL rewriting, bucket each commented path (test/production/config/docs), anonymise reviewers, rebuild threads, cache PR size |
| `rq4_01_classify_comments.py` | Score comment **role**, then the 6-category taxonomy + tone + intensity + resolution; export a blind sample for two-rater Cohen's κ |
| `rq4_02_aggregate_feedback.py` | Per-PR feature matrix; join MSI (RQ1) and TDT (RQ2) |
| `rq4_03_stats_analysis.py` | Chi-square/Cramér's V, Mann-Whitney U/Cliff's delta, logistic regression, Spearman, Kruskal-Wallis + 5 figures |

**Execution order:**

```bash
python rq4_00_build_corpus.py --offline        # heavy: reads pr_commit_details for PR size
python rq4_00_build_corpus.py --offline --skip-size   # faster, drops the size control
python rq4_01_classify_comments.py
python rq4_02_aggregate_feedback.py
python rq4_03_stats_analysis.py
```

**Outputs:** `AIDev/rq4_*.{parquet,csv}`, `figs/rq4/rq4_*.png`

**Comment roles.** In agentic PRs a sizeable share of "review comments" are not feedback but the agent reporting
back (*"Fixed in commit abc123"*). Counting these as feedback inflates the taxonomy, so role is scored first and
the taxonomy applies to `feedback` only:

| Role | n | Share |
|------|--:|------:|
| `feedback` — a reviewer raising something | 21,372 | 79.8% |
| `acknowledgement` — author/agent reporting a fix | 3,788 | 14.1% |
| `code_suggestion` — bare ```` ```suggestion ```` block, no prose | 1,619 | 6.0% |

**Taxonomy** (6 a-priori categories + `directive`, an emergent inductive category for terse imperatives such as
*"remove this"*, as anticipated by protocol §3.6.2).

**Corpus:** 26,779 comments → 4,451 PRs; 1,602 on test files (688 human / 914 bot) across 410 PRs.

> ⚠️ **Known limitations of the current RQ4 run**
>
> 1. **MSI cannot enter the regression.** Only 12 of the mutation-tested PRs have review comments (3 with
>    test-file comments), so `rq4_03` excludes any covariate below 50 PRs and reports MSI descriptively. The
>    MSI half of the RQ is therefore unanswered — this traces back to RQ1's small mutation run, not to RQ4.
>    TDT is unaffected (1,235 PRs).
> 2. **64% of feedback comments remain unclassified** by the rule-based lexicons. Test-file comments are
>    *more* classifiable (49.5%) than production ones (64.2%), so this does not bias against the main contrast,
>    but the category shares are lower bounds.
> 3. **Resolution is a proxy.** GitHub's `resolved` flag is absent from the dataset, so resolution is
>    approximated by whether a comment received a reply in its thread.
> 4. **The classifier is deterministic, not an LLM.** It is auditable (per-category scores are written to
>    `category_scores_json`) but unvalidated: `rq4_manual_coding_sample.csv` exists so the two-rater κ ≥ 0.70
>    required by protocol §4.3.1 can be computed before the category distributions are treated as findings.

---

## Key Findings

The key findings from the analysis of are based on AIDev-pop, a subset of the AIDev dataset.

### AIDev-pop: Filtered (>100 stars)

|                  | #PR        | #Developer | #Repo     |
| ---------------- | ---------- |------------| --------- |
| `OpenAI Codex`   | 21,799     | 1,284      | 1,248     |
| `Devin`          | 4,827      | NA         | 288       |
| `GitHub Copilot` | 4,970      | NA         | 1,012     |
| `Cursor`         | 1,541      | 363        | 327       |
| `Claude Code`    | 459        | 236        | 213       |
| **Total**        | **33,596** | **1,796**  | **2,807** |

![](./figs/pr_cumulative_popular.png)

### Productivity in the Coding Agents Era

![pr_merge_compare_radar2.png](figs%2Fpr_merge_compare_radar2.png)

### Turnaround Time

![](figs%2Fturnaround_distribution.png)

[//]: # ()
[//]: # (![]&#40;figs/copilot_job_completion_time.png&#41;)

[//]: # ()
[//]: # (**GitHub Copilot delivers half of its PRs within 12.8 minutes**. While 75% of jobs are completed within 18.5 minutes, the distribution)

[//]: # (exhibits a long tail extending up to 60 minutes, with the 95th percentile exceeding one hour. )
[//]: # ()
[//]: # (### Who Review the PRs?)

[//]: # ()
[//]: # (![]&#40;./figs/reviewer_classification_stacked.png&#41;)

[//]: # ()
[//]: # (**Human reviewers remain dominant across Agentic-PRs yet GitHub Copilot drives a shift)

[//]: # (toward automated hybrid collaboration in review**. Both Human-PRs and Agentic-PRs receive)

[//]: # (no explicit review in the majority of cases &#40;75.3% and 58.2%, respectively&#41;, while the second most common category)

[//]: # (involves reviews conducted solely by humans, at 14.7% and 21.8%. Notably, bot reviewers are more prevalent in Agentic-)

[//]: # (PRs &#40;20.1%&#41; than in Human-PRs &#40;10.0%&#41;. )

[//]: # ()
[//]: # (![]&#40;./figs/bot_heatmap.png&#41;)

[//]: # ()
[//]: # (**Autonomous Coding Agents and their paired review bots often originate from the same)

[//]: # (provider, forming closed PR-review loops that streamline workflows; but risk reinforcing provider-specific)

[//]: # (biases.** We analyze the top 10 most active review bots to understand on which Autonomous Coding Agents they operate.)

[//]: # (The heatmap illustrates a strong association between review bots and Autonomous Coding Agents from the same provider.)

### Language Usage

![](figs%2Ftotal_language_percentages_top.png)

**Autonomous Coding Agents exhibit distinct language preferences reflecting domain specialization in their capabilities.** 
TypeScript is the most common language across all agents, underscoring its popularity in AI-assisted development.
However, notable divergences emerge: OpenAI Codex shows a pronounced skew toward Python, while GitHub
Copilot heavily favours C#, likely reflecting their respective integrations and user bases.

## Dataset Schema

![](figs%2Fdataset_schema.png)

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@misc{li2025aiteammates,
      title={The Rise of AI Teammates in Software Engineering (SE) 3.0: How Autonomous Coding Agents Are Reshaping Software Engineering}, 
      author={Hao Li and Haoxiang Zhang and Ahmed E. Hassan},
      year={2025},
      eprint={2507.15003},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2507.15003}, 
}
```
