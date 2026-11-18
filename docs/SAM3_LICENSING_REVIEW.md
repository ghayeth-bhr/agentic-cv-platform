# SAM 3 Licensing and Hosted-Product Review (Roadmap Item 10)

**Status:** Review deliverable (documentation/checklist). **Not legal advice.** Final licensing/compliance decisions and any hosted-product clearance require review by the owner and qualified counsel.

## Scope

This document records a practical review of SAM 3 usage considerations for this project as of 2026-11-18, focused on:
- Gated Hugging Face model access (`facebook/sam3`)
- License and terms awareness (SAM License / Meta terms)
- Implications for a hosted/productized offering

## Current project context (factual)

- The platform integrates SAM 3 through a lazy, provider-injected adapter and local inference/service paths (Transformers/HF route), with tests designed to run without loading real weights by default.
- Prior project notes indicate: local SAM 3 inference was evaluated via `facebook/sam3` (gated on Hugging Face), and **access was approved for the owner** in that context. Peak VRAM measured ~2.08 GB in a prior test run (environment-specific).
- Default automated test suite does **not** require SAM 3 weights, GPU, or network access; real-model execution is opt-in/manual and environment-dependent.

## Key considerations

### 1) Gated weights and access control
- `facebook/sam3` on Hugging Face is a **gated** model: access, download, and use are subject to the repository’s gate terms and acceptance flow.
- **Action/verification:** ensure only authorized users/environments hold credentials/tokens that can access gated weights; do not redistribute weights or tokens. For team/CI/hosted environments, confirm each party’s access is permitted under the applicable terms.

### 2) License/terms (SAM License and related terms)
- SAM 3 weights and associated materials are governed by Meta’s applicable license/terms (commonly referred to as the “SAM License” for SAM releases) and, where applicable, Hugging Face repository terms.
- **Action/verification:** review the exact, current license/terms attached to `facebook/sam3` (and any specific model revision used) before production or hosted deployment. Terms can differ by version/release and may include restrictions (e.g. use-case, redistribution, attribution/notice, acceptable use).
- **Non-claim:** this document does **not** assert that current or planned use is compliant; compliance must be determined against the actual terms in force for the chosen model/version and deployment model.

### 3) Hosted/product implications
Before offering a hosted product or exposing SAM 3-powered features to third parties, evaluate (with counsel as appropriate):
- **Distribution/serving model:** whether serving outputs via an API/product constitutes use permitted under the license/terms for your intended users and jurisdictions.
- **Redistribution:** do not bundle/redistribute gated weights unless explicitly permitted; prefer runtime download in authorized environments or documented customer-provided access where appropriate.
- **Downstream obligations:** attribution, notices, and any required pass-through terms for users of the service.
- **Data/privacy/security:** prompts/images/labels sent to model-serving components; retention, isolation, and third-party processing disclosures.
- **Model/version pinning and auditability:** record model ID/revision used in production for reproducibility and license/terms traceability.
- **Acceptable use and risk:** safeguards against prohibited use cases per applicable terms and your product policies.

## Recommended owner actions (checklist)

- [ ] Identify the exact SAM 3 model ID(s) and revision(s) intended for use (e.g. `facebook/sam3` + pinned revision).
- [ ] Retrieve and archive the current Hugging Face model card terms and the applicable Meta SAM license/terms for that revision.
- [ ] Confirm access/entitlement status for all deployment contexts (local dev, CI, staging, production, any hosted service).
- [ ] Decide and document the serving architecture (local-only vs hosted) and whether it triggers additional license/terms obligations.
- [ ] Obtain legal/compliance review for the intended hosted-product use case before external/production launch.
- [ ] Define attribution/notice and user-facing terms needed for the product, if any.
- [ ] Document operational controls: no weight redistribution, credential handling, model pinning, and audit logging of model/version used.

## Outcome

- **Delivered:** licensing/hosted-product review checklist and documented considerations for SAM 3 gated weights and SAM License/terms.
- **Not delivered/claimed:** legal approval, license compliance certification, or clearance for hosted deployment. These remain explicit owner/counsel actions before any hosted product release.
