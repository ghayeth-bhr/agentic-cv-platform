# Agentic CV Platform: CI/CD, Load Testing, Failure-Injection (Roadmap Item 9, Slice 1 — CI Pipeline)

Adds a baseline GitHub Actions CI workflow to run compile and default test checks automatically. This is the first slice of roadmap item 9; SQLite/local defaults and existing behavior remain unchanged.

## What this branch provides

- **`.github/workflows/ci.yml`**
  - Triggers on `push` to `main` and on `pull_request`.
  - Matrix on Ubuntu with Python **3.11** and **3.13**.
  - Steps: checkout, set up Python, install package + test deps (`httpx`, `pytest`), `python -m compileall -q src/agentic_cv tests`, `python -m pytest -q` (default suite, no live services).

## Validation

- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **168 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.
- Workflow file added as repository CI configuration (GitHub Actions execution is external to this local run).

## Limitations / next boundary

- CI runs default tests only (no live Postgres/broker/S3/OIDC integration jobs, no lint/type-check tooling configured in this slice).
- Further item 9 slices can add load testing and failure-injection testing.

## Roadmap status

- **Items 6, 6b, 6c, 7, 8: complete.**
- **Item 9: slice 1 (CI pipeline) delivered in this branch.**
- **Next:** Item 9 slices (load testing, failure-injection testing), then Item 10.

---
