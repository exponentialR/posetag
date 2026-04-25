# PoseTag JOSS Readiness Checklist

This document is a living checklist for preparing PoseTag for a future
Journal of Open Source Software (JOSS) submission.

This checklist is a planning document, not a claim that PoseTag is currently
ready for JOSS submission.

It is intended to help track repository, packaging, documentation,
reproducibility, governance, and paper-preparation work without changing the
scientific contract of the software.

## How to use this checklist

- Update status as work progresses.
- Add links or short notes as evidence where useful.
- Prefer concrete evidence over assumptions.
- If a requirement is only partially met, mark it clearly and record the gap.

Suggested status markers:

- `[ ]` Not yet satisfied
- `[~]` In progress or partially satisfied
- `[x]` Satisfied

---

## 1. Repository Availability and Public Access

- [ ] Repository is public and accessible without special credentials.
- [ ] Repository can be cloned by an external user from the canonical public URL.
- [ ] Default branch and active development branch are clear to outside users.
- [ ] Repository metadata uses the canonical PoseTag identity consistently.
- [ ] Archived or legacy repository names, if any, are explained or redirected.

Notes:

- Evidence:
- Open questions:

---

## 2. Licensing

- [ ] Repository includes an OSI-approved licence file.
- [ ] Package metadata and repository licence information agree.
- [ ] Third-party assets, models, sample data, and generated artefacts have
  compatible licensing or clear exclusions.
- [ ] Any bundled dependencies or copied code have appropriate attribution.

Notes:

- Evidence:
- Open questions:

---

## 3. Issue Tracker and Public Development

- [ ] Public issue tracker is enabled and visible.
- [ ] Users can report bugs, ask questions, and request features publicly.
- [ ] At least six months of public development history exists before
  submission.
- [ ] Commit and issue history shows active software development rather than a
  one-time code dump.
- [ ] Important design or maintenance decisions are discoverable from issues,
  PRs, or project docs.

Notes:

- Evidence:
- Open questions:

---

## 4. Demonstrated Research Use

- [ ] PoseTag has been used in real research, internal experiments, or
  reproducible pilot studies.
- [ ] There is a clear statement of the research problem PoseTag supports.
- [ ] Example outputs or datasets demonstrate the software in practice.
- [ ] Any claims about scientific use are supportable with citations, reports,
  or public artefacts.

Notes:

- Evidence:
- Open questions:

---

## 5. Installability

- [ ] Fresh-clone installation works on a supported Python version.
- [ ] Editable install works for development workflows.
- [ ] Install instructions are documented and match the actual package state.
- [ ] Core CLI entry points resolve after installation.
- [ ] Optional dependencies and extras are documented clearly.
- [ ] Platform-specific dependencies are called out explicitly.

Notes:

- Evidence:
- Open questions:

---

## 6. Documentation

- [ ] README explains what PoseTag does and who it is for.
- [ ] Installation instructions are current and tested.
- [ ] Core workflows are documented end to end.
- [ ] Scientific conventions are documented explicitly.
- [ ] Inputs, outputs, file layouts, and required metadata are described.
- [ ] Common failure modes and troubleshooting guidance are documented.
- [ ] Canonical PoseTag commands, package names, and migration notes are
  consistent across docs.

Notes:

- Evidence:
- Open questions:

---

## 7. Tests and Continuous Integration

- [ ] Automated tests exist for critical scientific and schema invariants.
- [ ] Test suite runs in CI on supported platforms or Python versions.
- [ ] CI status is visible publicly.
- [ ] Packaging or install smoke checks run in CI.
- [ ] Failures are actionable and not ignored routinely.
- [ ] Coverage is sufficient for the current maturity of the project.

Notes:

- Evidence:
- Open questions:

---

## 8. Contribution Process and Governance

- [ ] Contribution guidance exists for new contributors.
- [ ] Maintainer expectations for issues, PRs, and review are documented.
- [ ] Code of conduct or equivalent contributor expectations are available.
- [ ] Development setup is documented for contributors.
- [ ] Decision-making and review responsibilities are clear enough for an
  external reviewer to understand project stewardship.

Notes:

- Evidence:
- Open questions:

---

## 9. Releases and Changelog

- [ ] Tagged releases exist or a release plan is documented.
- [ ] Versioning approach is clear and consistent.
- [ ] Changes between releases are documented in a changelog or release notes.
- [ ] Release artefacts are reproducible and correspond to repository tags.
- [ ] Breaking changes or migration steps are documented where relevant.

Notes:

- Evidence:
- Open questions:

---

## 10. Citation Metadata

- [ ] `CITATION.cff` or equivalent citation metadata exists.
- [ ] Citation metadata matches repository authorship and title.
- [ ] Citation guidance is visible to users in the repository.
- [ ] DOI strategy is defined if releases will be archived through Zenodo or a
  similar service.

Notes:

- Evidence:
- Open questions:

---

## 11. JOSS Paper Materials

- [ ] JOSS paper repository materials are prepared in the expected format.
- [ ] Paper includes summary, statement of need, and references.
- [ ] Paper claims match what the software currently does.
- [ ] Installation and usage described in the paper match the repository.
- [ ] Author list, affiliations, and acknowledgements are ready.
- [ ] Review-facing reproducibility notes are collected.

Notes:

- Evidence:
- Open questions:

---

## 12. AI Usage Disclosure

- [ ] Repository contains a clear disclosure describing any AI-assisted repository preparation, maintenance, documentation, or development work.
- [ ] Scope of AI assistance is described accurately.
- [ ] Human review, editing, testing, and validation responsibilities are
  stated explicitly.
- [ ] Human responsibility for scientific, architectural, legal, and
  publication decisions is stated explicitly.
- [ ] JOSS paper materials can be updated consistently with the repository
  disclosure if needed.

Notes:

- Evidence:
- Open questions:

---

## Recommended Pre-Submission Review Pass

Before submitting to JOSS, confirm all of the following:

- [ ] A new external user can install and run the documented PoseTag workflow.
- [ ] Scientific assumptions and output conventions are explicit and tested.
- [ ] Repository naming, package naming, and CLI naming are consistent.
- [ ] Public metadata, citation files, and paper materials agree.
- [ ] AI usage disclosure is present and aligned with the final paper text.

