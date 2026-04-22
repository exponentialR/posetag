# PoseTag Phase 0 Rename Plan

## Scope

This phase removes repository-wide naming drift from the legacy GT-6DoF-ATag
identity and establishes the canonical PoseTag naming across metadata, package
namespace direction, CLI direction, and docs without changing scientific
behaviour.

## Legacy Names Found

### Package and metadata

- `pyproject.toml`
  - project name: `gt6dof-atag`
  - duplicated `[project.scripts]` sections
  - script targets under `gt6dof_atag.cli.*`
  - `gtat*` command names
  - `readme = "README.md"` while the file on disk is `readme.md`

### Package/module paths

- package directory: `src/gt6dof_atag/`
- imports in packaged CLI code: `from gt6dof_atag...`
- internal comments referencing `src/gt6dof_atag/...`

### User-facing CLI and help text

- CLI module `src/gt6dof_atag/cli/gtat.py`
  - help text describes `GT-6DoF-ATag`
  - parser/program name is `gtat`
  - usage examples start with `gtat ...`
- `pyproject.toml` defines legacy commands beginning with `gtat-`

### Docs and repository presentation

- `readme.md`
  - title: `GT-6DoF-ATag`
  - shield URLs include `GT-6DoF-ATag`
  - examples teach script-era invocations such as `python -m src.*`
  - wording and paths still imply the legacy identity

### Project/config naming surfaces

- `utils/project_config.py`
  - docstring references `GT-6DoF-ATag`
  - config dir app name: `gt6dof_atag`
  - env vars: `GTAT_PROJECT`, `GTAT_PROJECTS_DIR`
  - default projects home: `~/gt-6dof`
  - docs mention `~/.config/gt6dof_atag/config.yaml`
- `utils/project_paths.py`
  - docs and env handling use `GTAT_PROJECT`
- `utils/charuco_calibrate.py`
  - docstrings mention `GTAT_PROJECT`
  - packaged import fallback uses `gt6dof_atag.utils.project_config`
  - fallback project directory `cwd/gtat_project`
- `utils/logger.py` and `utils/capture_utils.py`
  - logger name `gtat`

## Target Replacements

### Canonical replacements

- display/project name: `PoseTag`
- distribution/package direction: `posetag`
- primary CLI command: `posetag`
- canonical config app name: `posetag`
- canonical env vars:
  - `POSETAG_PROJECT`
  - `POSETAG_PROJECTS_DIR`
- canonical default projects home: `~/posetag`

### Temporary compatibility aliases to retain

- import namespace alias: `gt6dof_atag` -> `posetag`
- legacy CLI commands:
  - `gtat`
  - `gtat-gen-tags`
  - `gtat-calib-charuco`
  - `gtat-make-board`
  - `gtat-capture-face`
  - `gtat-annotate`
  - `gtat-collect`
- legacy env vars:
  - `GTAT_PROJECT`
  - `GTAT_PROJECTS_DIR`
- legacy config/projects locations as read fallback only:
  - `~/.config/gt6dof_atag/config.yaml`
  - `~/gt-6dof`

## Cosmetic vs Public API Changes

### Purely cosmetic/internal

- README title and narrative text
- comments and module docstrings
- logger names
- internal package comments that mention old paths

### Public API / compatibility-sensitive

- package import namespace (`gt6dof_atag` -> `posetag`)
- CLI entry point names (`gtat*` -> `posetag*` and `posetag`)
- distribution metadata name
- config directory/app name
- environment variable names
- default projects home directory

## Compatibility Strategy

1. Add a canonical `src/posetag/` package.
2. Keep `src/gt6dof_atag/` as a short-lived compatibility package that
   re-exports from `posetag` and carries deprecation comments.
3. Add importable package shims for utilities needed by packaged CLI code so
   entry points resolve without relying on repo-root-only imports.
4. Make `posetag` the canonical CLI surface in metadata.
5. Retain legacy `gtat*` entry points temporarily, pointing at compatibility
   wrappers that delegate to the same implementation.
6. Prefer new env vars and new config/projects locations, but read legacy env
   vars and legacy config/home locations as fallbacks.
7. Update docs to teach only `PoseTag` / `posetag`, with a small migration note
   for legacy users.

## Risks

- Packaging risk: current metadata is invalid because `[project.scripts]` is
  duplicated and the packaged `gtat` target imports a non-existent packaged
  module path.
- Installability risk: much runtime logic still lives in top-level modules and
  repo-root `utils/`, so the canonical package needs import shims rather than a
  broad refactor in this phase.
- Compatibility risk: changing default config/project locations could hide
  existing projects unless legacy locations remain discoverable.
- Documentation risk: the repository currently teaches stale script paths and
  could leave users in a half-renamed state unless README and migration notes
  are updated together.

## Execution Order

1. Fix metadata and establish the canonical `posetag` package namespace.
2. Add compatibility shims for `gt6dof_atag` imports and legacy CLI names.
3. Reconcile project/config naming helpers to prefer `PoseTag`/`posetag` while
   preserving legacy fallbacks.
4. Update README and add a migration document.
5. Add tests for import aliases, CLI target resolution, and metadata coherence.
6. Run smoke checks to confirm the rename did not alter scientific logic.
