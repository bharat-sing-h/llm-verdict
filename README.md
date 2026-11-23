# llm-eval-harness

[![CI](https://github.com/bharat-sing-h/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/bharat-sing-h/llm-eval-harness/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**An evaluation and red-teaming framework for LLMs and LLM applications, with statistically sound model comparison and CI regression gates.**

Most LLM evals stop at "model A scored 71%, model B scored 68%". That's not enough to ship on. `evalharness` answers the questions a team actually has to answer before changing a model or a prompt in production:

- **Is the difference real or noise?** Paired significance tests (exact McNemar, paired bootstrap) and confidence intervals on every metric.
- **Did anything get worse?** Per-slice breakdowns and per-sample regression lists.
- **Is the system secure?** A prompt-injection suite with 15 attack techniques, scored objectively with canary tokens, plus benign controls so you see the security/utility trade-off.
- **Can we trust model-graded metrics?** Measures LLM-as-judge agreement against human labels (Cohen's κ, quadratic-weighted κ, Spearman ρ, leniency bias) before you rely on the judge.
- **Can we stop regressions automatically?** `evalharness gate` exits non-zero on a significant regression and posts a Markdown verdict to the GitHub job summary.

```text
$ evalharness compare naive__prompt_injection guarded__prompt_injection
| Metric           | Baseline | Candidate | Δ        | 95% CI (Δ)    | p-value | Test          | Verdict     |
|------------------|----------|-----------|----------|---------------|---------|---------------|-------------|
| control_helpful  | 100.0%   | 100.0%    | +0.0 pp  | [+0.0, +0.0]  | 1.0000  | mcnemar_exact | no change   |
| secret_protected | 82.4%    | 100.0%    | +17.6 pp | [+8.8, +26.5] | 0.0005  | mcnemar_exact | 🟢 improved |
```
<sub>Real output from the offline demo below, which red-teams a toy rule-based app (not an LLM).</sub>

---

## Contents
- [Quickstart (no API key needed)](#quickstart)
- [Features](#features)
- [Architecture](#architecture)
- [Built-in benchmarks](#built-in-benchmarks)
- [Evaluating real models](#evaluating-real-models)
- [Statistical methodology](#statistical-methodology)
- [LLM-as-judge calibration](#llm-as-judge-calibration)
- [CI regression gating](#ci-regression-gating)
- [Extending](#extending)
- [Design notes](docs/design.md)

## Quickstart

```bash
git clone https://github.com/bharat-sing-h/llm-eval-harness && cd llm-eval-harness
pip install -e ".[dev]"
make demo
```

`make demo` runs the prompt-injection suite against two toy support bots in [`examples/apps/support_bot.py`](examples/apps/support_bot.py):

| App | Defence | Secret protected | Leaks via |
|---|---|---|---|
| `naive_bot` | Input keyword blocklist | 82.4% | prompt extraction, translation, Spanish-language request |
| `guarded_bot` | Blocklist **plus** output filter that detects the secret, including encoded forms | 100.0% | none |

It then compares the two runs and gates on the result. The takeaway holds for real systems too: input keyword filters are easy to route around, and defence in depth (checking outputs) closes the gap without hurting helpfulness on benign questions (`control_helpful` stays at 100%).

```bash
pip install -e ".[dashboard]" && make dashboard   # explore runs in the browser
```

![dashboard](docs/images/dashboard.png)

## Features

| Area | What you get |
|---|---|
| **Providers** | Any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, llama.cpp, TGI, LM Studio, Together, Groq, OpenRouter, ...), **any Python callable** (evaluate a whole RAG pipeline or agent, not just a model), and a deterministic mock for tests |
| **Engine** | Async with bounded concurrency, client-side token-bucket rate limiting, timeouts, jittered exponential backoff that honours `Retry-After`, and per-sample error isolation |
| **Reproducibility** | Content-addressed response cache (re-scoring is free), crash-safe resume, seeded procedural datasets, full config snapshot stored with every run |
| **Scorers** | `final_answer`, `exact_match`, `includes`, `regex`, `token_f1`, `json_schema` (lenient/strict), `json_fields` (partial credit), `canary_leak`, `refusal`, `llm_judge`. Any scorer can be scoped to tagged samples with `only_tags` |
| **Statistics** | Wilson/bootstrap CIs, exact McNemar, paired bootstrap, Cohen's κ (weighted), Spearman, all dependency-free and unit-tested against reference values |
| **Reporting** | Markdown reports (renders in PRs and job summaries), JSON export, a Streamlit dashboard, p50/p95/p99 latency, token and cost accounting |
| **Ops** | Typed CLI, SQLite results store, Docker image (non-root, multi-stage), GitHub Actions CI with an eval-gate job that runs this tool on itself |
| **Quality** | 97 tests, 98% branch coverage, `mypy --strict`, ruff, tested on Python 3.10–3.13 |

## Architecture

```mermaid
flowchart LR
    CFG[YAML config] --> H[harness.run_config]
    subgraph Data
      B[Built-in procedural tasks] --> T[TaskSpec / Samples]
      J[Custom JSONL] --> T
    end
    T --> R
    H --> R[Runner<br/>semaphore + token bucket<br/>timeout + retry]
    R <--> C[(Response cache<br/>SQLite, content-addressed)]
    R --> P{Provider}
    P --> O[OpenAI-compatible API]
    P --> F[Python callable<br/>RAG / agent / app]
    R --> S[Scorers<br/>deterministic + LLM judge]
    S --> DB[(Results store<br/>SQLite)]
    DB --> REP[Markdown / JSON reports]
    DB --> CMP[Paired comparison<br/>McNemar / bootstrap]
    CMP --> G[CI gate<br/>exit code + job summary]
    DB --> UI[Streamlit dashboard]
```

| Module | Responsibility |
|---|---|
| `providers/` | `Provider` ABC; errors are classified as retryable or fatal at the edge |
| `engine/runner.py` | Concurrency, rate limiting, retries, caching, scoring, persistence |
| `scorers/` | Async scorer interface shared by cheap deterministic scorers and LLM judges |
| `tasks/` | Seeded benchmark generators and the JSONL loader |
| `stats.py` | Pure-Python statistics |
| `compare.py` | Paired run comparison and the gate policy |
| `calibration.py` | Judge–human agreement analysis |
| `store.py` | SQLite persistence with resume support |

See [docs/design.md](docs/design.md) for the reasoning behind each decision and the trade-offs.

## Built-in benchmarks

All built-in tasks are **procedurally generated from a seed**. Fresh instances can't have leaked into pre-training data, ground truth is computed in code, and a fixed seed keeps runs comparable.

| Task | What it measures | Primary metric |
|---|---|---|
| `math_reasoning` | Multi-step arithmetic and date reasoning in 6 problem families × 3 difficulty levels, including distractor sentences | `final_answer` |
| `json_extraction` | Pulling an order from messy prose into **schema-valid JSON**, with varied date and price formats, computed totals and noise about other orders. Format validity and field accuracy are scored separately | `json_fields` |
| `prompt_injection` | Secret extraction with 15 techniques (direct request, ignore-previous, authority claim, role-play, prompt extraction, translation, base64, spell-out, reverse, completion, fiction, structured dump, indirect injection, payload splitting, multilingual), plus benign controls | `secret_protected` |

```bash
evalharness tasks   # list tasks with sample counts
```

The canary detector catches leaks that plain substring checks miss: spaced-out or re-punctuated characters, reversed text, base64, hex, ROT13, and a secret split across several sentences.

## Evaluating real models

```bash
# Local models via Ollama (or vLLM / llama.cpp - anything OpenAI-compatible)
ollama pull qwen2.5:7b && ollama pull llama3.1:8b
evalharness run examples/ollama.yaml --label qwen7b
MODEL=llama3.1:8b evalharness run examples/ollama.yaml --label llama8b
evalharness compare qwen7b__math_reasoning llama8b__math_reasoning

# Hosted APIs
export OPENAI_API_KEY=...
evalharness run examples/openai.yaml --label gpt-4o-mini --report-dir reports/
```

A config is a short YAML file. `${VAR}` and `${VAR:-default}` are expanded from the environment:

```yaml
model:
  provider: openai_compat           # or: function (any Python callable), mock
  model: ${MODEL:-qwen2.5:7b}
  base_url: http://localhost:11434/v1
  pricing: {input_per_mtok: 0.15, output_per_mtok: 0.60}
params: {temperature: 0, max_tokens: 768, seed: 1234}
tasks:
  - builtin: math_reasoning
    options: {n: 300, seed: 7}
  - path: data/my_dataset.jsonl     # {"input": ..., "reference": ..., "tags": [...]}
    scorers: [exact_match, {type: token_f1, options: {threshold: 0.6}}]
execution: {concurrency: 16, rpm: 450, max_retries: 5, timeout_s: 120}
```

Other commands: `evalharness runs`, `report <run>`, `export <run> out.jsonl`. Interrupted runs resume where they stopped: re-running the same `--label` only evaluates the missing or failed samples.

## Statistical methodology

| Question | Method | Why |
|---|---|---|
| CI for a pass rate | Wilson score interval | Stays accurate near 0% and 100%, unlike the normal approximation |
| CI for a graded score | Percentile bootstrap (2,000 resamples, seeded) | Makes no distributional assumptions |
| A vs B on binary metrics | **Exact McNemar** on discordant pairs | Both systems answer the *same* items, so pairing removes item difficulty from the noise; an unpaired test of two CIs is badly under-powered |
| A vs B on graded metrics | Paired bootstrap of per-item differences | Same reasoning, for non-binary scores |
| Judge vs human | Cohen's κ, quadratic-weighted κ, Spearman ρ | Corrects for chance agreement; weighting treats 4-vs-5 as less serious than 1-vs-5 |

API and network errors are reported separately (`n_errors`) and never counted as wrong answers, so an outage doesn't look like a model regression. The gate has its own `--max-error-rate`.

## LLM-as-judge calibration

A judge is itself a model with biases (leniency, preferring longer answers, preferring its own outputs). Before trusting a judge-based metric, measure the judge against human labels:

```bash
evalharness calibrate examples/llm_judge.yaml examples/data/judge_calibration.jsonl
```

The output includes exact agreement, κ, quadratic-weighted κ, Spearman ρ, pass/fail precision and recall at the threshold, leniency bias, a confusion matrix, and the largest disagreements with the judge's reasoning. A judge can *rank* perfectly (ρ = 1) while making the wrong pass/fail call on every item; the tests include exactly that case. Replace the bundled illustrative labels with your annotators' labels on your own data.

## CI regression gating

```bash
evalharness gate candidate__my_task --baseline baseline__my_task --max-drop 0.02
```

| Policy | Fails when | Use when |
|---|---|---|
| `--mode significant` (default) | The drop is larger than `--max-drop` **and** statistically significant | Small benchmarks; doesn't fail on noise |
| `--mode non_inferiority` | The 95% CI can't rule out a drop larger than `--max-drop` | High-stakes changes; needs more samples |
| `--min-score` | The metric falls below an absolute floor | Hard product requirements (e.g. secret protected ≥ 95%) |
| `--max-error-rate` | Too many samples hit API/app errors | Catches broken deployments |

Inside GitHub Actions the verdict is appended to `$GITHUB_STEP_SUMMARY`. This repo's own [CI](.github/workflows/ci.yml) runs this tool's red-team suite and gate on every push. A drop-in workflow for application repos is in [`examples/github-actions/llm-eval-gate.yml`](examples/github-actions/llm-eval-gate.yml).

## Extending

**Evaluate your own app**: any function that takes `messages` and returns text, sync or async:

```yaml
model:
  provider: function
  target: my_app.rag:answer      # def answer(messages: list[dict]) -> str
```

**Add a scorer**: subclass `Scorer` and register it:

```python
from evalharness.scorers import SCORERS, Scorer

class Toxicity(Scorer):
    type_name = "toxicity"
    async def score(self, sample, output):
        return self._bool(my_classifier(output) < 0.5)

SCORERS[Toxicity.type_name] = Toxicity
```

**Add a provider**: subclass `Provider`, implement `async generate(messages, params) -> ModelResponse`, and raise `ProviderError(retryable=...)` for failures.

## Development

```bash
make install   # editable install with dev tools
make check     # ruff + mypy --strict + pytest with coverage
docker build -t evalharness . && docker run --rm -v "$PWD:/work" evalharness run examples/offline_demo.yaml
```

## Roadmap
- Pairwise (A/B) judging with position-swap debiasing
- Multi-turn conversation tasks and tool-calling evals
- Self-consistency sampling (pass@k, variance across seeds)
- Optional Postgres store and an HTTP API for team-wide result sharing

## License
MIT
