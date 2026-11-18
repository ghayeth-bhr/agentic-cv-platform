# Agentic CV Platform: CI/CD, Load Testing, Failure-Injection (Roadmap Item 9, Slice 3 — Complete)

Adds failure-injection tests covering resilience of outbox delivery, export worker/job recovery, and label workflow persistence failures. Completes roadmap item 9 (slices 1–3). Defaults and runtime behavior unchanged.

## What this branch provides

- **`tests/test_failure_injection.py`**
  - **Outbox dispatcher:** publish failure is recorded (`attempts`/`last_error`) and delivery continues to subsequent events (no queue starvation).
  - **Export worker:** injected export failure marks the affected job failed while subsequent queued jobs still succeed; expired-lease (simulated worker death) is requeued/recovered and completed on next run.
  - **Label workflow:** annotation-persistence failure is surfaced as a failed job with `LabelRunFailed` outbox event.

## Validation

- Focused: `tests/test_failure_injection.py` → **4 passed**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **172 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Item 9 summary (complete)

- **Slice 1:** GitHub Actions CI pipeline for compile + default tests on Python 3.11/3.13 (`.github/workflows/ci.yml`).
- **Slice 2:** Lightweight local API load-test script (`scripts/load_test.py`, mixed workload, latency/error reporting).
- **Slice 3:** Failure-injection tests for outbox/worker/workflow resilience (**this slice**).

## Limitations (out of scope for this baseline)

- CI runs default tests only (no live-service integration matrix, no lint/type-check tooling).
- Load testing is a local smoke script (not long-soak/distributed); failure injection is deterministic/unit-integration level (not chaos/fault-injection at infrastructure level).

## Roadmap status

- **Items 6, 6b, 6c, 7, 8, 9: complete.**
- **Next:** Item 10 — SAM 3 licensing and hosted-product review.

---
