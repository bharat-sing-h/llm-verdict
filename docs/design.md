# Design notes

These notes cover the decisions that shaped the harness, the alternatives that were considered, and the trade-offs that were accepted.

## 1. Goals and non-goals

**Goals:** trustworthy comparisons between two systems, cheap iteration on metrics, the same tool on a laptop and in CI, and evaluation of *applications* (RAG pipelines, guarded bots), not just raw models.

**Non-goals (for now):** distributed execution across machines, training-time evaluation hooks, a hosted multi-user service.

## 2. Data model

Everything that crosses a component boundary is a pydantic model (`Sample`, `ModelResponse`, `Score`, `SampleResult`). Data is validated once at the edge and then serialises losslessly to SQLite and JSONL.

- `Sample.reference` is deliberately `Any`. Each scorer decides what it means: a string, a number, a JSON schema plus expected values, a canary. The alternative, a typed reference per task, made custom datasets awkward to write.
- `Score.value = None` means "not applicable" and is **excluded** from aggregation. This lets one task mix sample types (attacks vs. controls) without polluting metrics with fake zeros.
- `tags` are free-form `key:value` strings. Every metric is automatically broken down by tag, so slicing needs no extra code.

## 3. Execution engine

One asyncio task per sample, bounded by:

1. **Semaphore**: maximum in-flight requests (protects the server and memory).
2. **Token bucket**: requests per minute. Staying just under a provider's quota is cheaper than tripping 429s and backing off. Waiters queue FIFO behind a lock, so no request starves.
3. **Timeout** per attempt, converted to a retryable error.
4. **Retry** with full-jitter exponential backoff (as described in AWS's "Exponential Backoff and Jitter"). A server's `Retry-After` header overrides the computed delay.

Providers must raise `ProviderError(retryable=bool)`. The classification is made where the protocol knowledge lives (HTTP 429/5xx are retryable; 400/401 are not), which keeps the retry loop generic.

**Failure isolation:** a provider error becomes `SampleResult.error`. A crashing scorer becomes `Score(value=None, explanation=...)`. Neither aborts the run. On `KeyboardInterrupt` or cancellation the run is marked `interrupted`. Every completed sample is already persisted, so `resume` only re-runs what's missing, including samples that previously errored.

Why asyncio rather than threads? The workload is I/O-bound, asyncio gives cheap concurrency for thousands of in-flight requests, and both the HTTP client (`httpx`) and judge calls are async-native. SQLite writes are synchronous, but they are small and local, and they happen on the event-loop thread, so there is a single writer.

## 4. Caching

Responses are cached by `sha256(provider identity, messages, generation params)`.

- Changing a scorer and re-running costs **zero** API calls, which makes iterating on metrics cheap.
- Only `temperature == 0` requests are cached. Caching sampled outputs would silently remove the variance you might be trying to measure.
- Cached responses are flagged `cached=True` and excluded from latency percentiles and cost, so performance numbers always reflect real calls.

Bug worth recording: `ResponseCache` defines `__len__`, so an *empty* cache is falsy, and an early version used `if self.cache:`, which silently disabled caching until the cache was non-empty (that is, forever). The fix is `is not None` checks, plus a regression test. Python's truthiness rules make this an easy mistake on any container-like class.

## 5. Statistics

Implemented in pure Python (`stats.py`) to keep the dependency footprint small for CI, and tested against reference values.

- **Wilson** intervals for pass rates. The normal approximation produces zero-width or out-of-range intervals at 0% and 100%, which are common in safety evaluations.
- **Paired tests** for comparisons. Two systems evaluated on the same items have strongly correlated scores, because item difficulty dominates. Comparing two independent CIs ignores that correlation and needs far more samples to detect the same difference. The exact McNemar test uses only the discordant pairs; the paired bootstrap resamples per-item differences.
- Bootstrap resampling is seeded, so reports are reproducible byte for byte.

**Errors are not wrong answers.** Quality metrics exclude errored samples, and the error rate is reported and gated separately. Mixing them would make an API outage look like a model regression.

## 6. Gate policy

Two modes, because teams want different risk trade-offs:

- `significant`: fail only on a statistically significant drop larger than `max_drop`. This doesn't flake on small benchmarks, but it can miss small real regressions (limited power).
- `non_inferiority`: pass only if the 95% CI's lower bound on the change is above `-max_drop`. Stricter, but it needs enough samples to make the CI narrow, which is exactly the property you want for high-stakes changes.

## 7. Red-teaming design

Using a **canary** (a random per-sample secret) instead of an LLM judge makes attack success objective, cheap and reproducible. The detector deliberately looks for the obfuscations attackers request (spacing, reversal, base64, hex, ROT13, fragmentation), because a naive substring check under-reports leaks exactly where defences are weakest.

**Benign controls** are part of the same task. A model that refuses everything scores 100% on security and is useless, so `control_helpful` and `control_refused` are tracked next to `secret_protected`, and a hardening change has to keep both.

## 8. Procedural datasets

Static public benchmarks leak into pre-training corpora. Seeded generators produce fresh instances with exact, code-computed ground truth and adjustable difficulty. The trade-off is less linguistic diversity than human-written items, which is why custom JSONL datasets are a first-class input.

Ground truth is checked independently in tests (for example, weekday answers are recomputed with `datetime`). One generator bug was caught this way: a "leftover students" problem was meant to always have a non-zero remainder but could produce zero.

## 9. Things deliberately left out

- **A vector DB or RAG framework dependency.** RAG systems are evaluated through the `function` provider, so the harness stays framework-agnostic.
- **pandas/numpy in the core.** They're only needed for the dashboard (`[dashboard]` extra). The core installs in seconds in CI.
- **Postgres.** SQLite is zero-ops, transactional and easy to upload as a CI artifact. The store sits behind a small class, so swapping it out is localised.
