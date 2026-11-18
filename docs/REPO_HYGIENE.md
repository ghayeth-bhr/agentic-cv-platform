# Repository hygiene proposal (owner decision required)

Status: **proposal only — nothing in this document was executed.**

Verified on the P0 branch (`fix/runtime-foundation`, stacked on `feature/ui-workbench-redesign`):

| Fact | Value |
|---|---|
| GitHub default branch | `feature/platform-foundation-contracts` (tip `8021578`, “Document foundation contracts branch”) |
| `main` | `f0c4998` (“Add SAM3 licensing and hosted-product review”) |
| `main` is an ancestor of the working branch | yes |
| Commits from `main` to the working branch | 7 (includes the UI workbench branch and P0) |
| Remote refs | 45 (44 feature/fix + `main`) |
| Remote branches merged into `main` | 42 |
| Remote branches **not** merged into `main` | `feature/local-resumable-upload`, `feature/platform-foundation-contracts`, `feature/ui-workbench-redesign` |

## Why this matters

- New visitors land on an old feature branch, not `main`.
- `main` is behind the actual UI/runtime work, so cloning the default branch gives a misleading baseline.
- 42 merged branches make the branch list unusable and hide the 3 branches that still need attention.

## Recommended sequence (after P0 is pushed and the owner approves)

1. **Fast-forward `main`** to the validated P0 tip (this also brings in the UI workbench branch):

   ```bash
   git switch main
   git merge --ff-only fix/runtime-foundation
   git push origin main
   ```

   Do not rewrite history. If `--ff-only` refuses, stop and report instead of forcing a merge.

2. **Set the GitHub default branch to `main`** (repo admin required):

   - GitHub → Settings → Branches → Default branch → `main`, or
   - `gh repo edit ghayeth-bhr/agentic-cv-platform --default-branch main` if `gh` is authenticated.

3. **Prune only branches fully merged into `main`.** Generate the list with:

   ```bash
   git branch -r --merged main
   ```

   Then delete each `origin/<branch>` with `git push origin --delete <branch>`. Keep:
   - `feature/local-resumable-upload` (not merged; inspect before deleting),
   - `feature/platform-foundation-contracts` (current default until step 2; contains unique commits),
   - any branch not listed by `--merged`.

4. **Protect `main`** (optional but recommended): require a pull request and a passing CI check before merge.

## Risks / open questions

- Branch deletion is irreversible from the local clone unless the commits are reachable from another ref. Confirm the merged list before deleting.
- `feature/local-resumable-upload` is not merged. It may be an early slice superseded by later work, but it must be inspected, not assumed.
- `feature/platform-foundation-contracts` is the current default and is not merged. Do not delete it until the default branch is changed and its unique commits are accounted for.
- P0 has not been pushed yet when this document was written; merge only after its validation is reported.

## Decision requested

Approve one of:

- **A (recommended):** fast-forward `main` to P0, set `main` as default, delete the 42 merged branches, keep the 3 unmerged branches.
- **B:** merge to `main`, set default, but keep all branches.
- **C:** keep the current branch layout; do not merge or prune yet.

No merge, push, default-branch change, or branch deletion will be performed until the owner chooses.
