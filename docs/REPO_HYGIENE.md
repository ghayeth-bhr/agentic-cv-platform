# Repository hygiene: decision and execution log

Status: **owner approved option A; partially executed.** The GitHub default-branch change still needs a repository admin because `gh` and a GitHub token are not available in this environment.

## Original verified state

| Fact | Value |
|---|---|
| GitHub default branch | `feature/platform-foundation-contracts` (tip `8021578`, “Document foundation contracts branch”) |
| `main` before the merge | `f0c4998` (“Add SAM3 licensing and hosted-product review”) |
| `main` was an ancestor of the P0 branch | yes |
| Commits from `main` to the P0 tip | 7 (UI workbench branch + 3 P0 commits) |
| Remote refs before cleanup | 45 (44 feature/fix + `main`) |
| Remote refs merged into `main` | 42 **including `origin/main` itself**, so 41 feature branches to prune |
| Remote branches not merged into `main` | `feature/local-resumable-upload`, `feature/platform-foundation-contracts`, `feature/ui-workbench-redesign` |

Correction: the initial proposal said “42 merged branches.” That count included `origin/main`. The executable list was 41 feature branches; `main` was never a deletion candidate.

## Owner decision

Option **A**: fast-forward `main`, set `main` as the GitHub default, prune only fully merged branches, and keep the unmerged branches.

## Executed

1. Fast-forwarded and pushed `main`:

   ```text
   f0c4998..2855e2b  main -> main
   ```

   `main` is now the validated P0 tip (`2855e2b`), which includes the UI workbench and P0 changes.

2. Deleted **41 fully merged feature branches**, after verifying each was an ancestor of `main`:

   ```text
   git push origin --delete ...   # 41 branches
   exit: 0
   ```

3. Kept:

   - `main`
   - `fix/runtime-foundation` (the P0 branch; now merged, retained for traceability)
   - `feature/ui-workbench-redesign` (now merged; retained because it was one of the three branches the owner asked to keep)
   - `feature/platform-foundation-contracts` (still the GitHub default and still unmerged)
   - `feature/local-resumable-upload` (still unmerged; inspect before deleting)

## Remaining actions

1. **Set the GitHub default branch to `main`.** This requires repo admin. No `gh` binary or `GH_TOKEN`/`GITHUB_TOKEN` is available here, so it was not executed. Use:

   - GitHub → Settings → Branches → Default branch → `main`, or
   - `gh repo edit ghayeth-bhr/agentic-cv-platform --default-branch main` on a machine with an authenticated `gh`.

2. **Review `feature/platform-foundation-contracts` after the default changes.** It has unique commits (`8021578`) and is not merged. Do not delete it until those commits are accounted for.

3. **Review `feature/local-resumable-upload`.** It is not merged; it may be an early slice superseded by later work, but that is an assumption, not a verified fact.

4. **Protect `main`** (recommended): require a pull request and a passing CI check.

## Final remote refs after pruning

```text
origin/feature/local-resumable-upload
origin/feature/platform-foundation-contracts
origin/feature/ui-workbench-redesign
origin/fix/runtime-foundation
origin/main
```

`git ls-remote --symref origin HEAD` still reports `refs/heads/feature/platform-foundation-contracts` until step 1 is done manually.

## Risks

- Branch deletion is irreversible from the local clone unless the commits are reachable from another ref. The 41 deleted branches were verified as ancestors of `main` before deletion.
- The default branch remains misleading until the manual GitHub settings change is completed.
