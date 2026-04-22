# PoseTag Migration

## Canonical Identity

Use these names going forward:

- display/project name: `PoseTag`
- distribution name: `posetag`
- import namespace: `posetag`
- primary CLI: `posetag`

## Old to New

| Legacy surface | Canonical surface |
| --- | --- |
| `GT-6DoF-ATag` | `PoseTag` |
| `gt6dof-atag` | `posetag` |
| `gt6dof_atag` | `posetag` |
| `gtat` | `posetag` |
| `gtat-gen-tags` | `posetag-gen-tags` |
| `gtat-calib-charuco` | `posetag-calib-charuco` |
| `gtat-make-board` | `posetag-make-board` |
| `gtat-capture-face` | `posetag-capture-face` |
| `gtat-annotate` | `posetag-annotate` |
| `gtat-collect` | `posetag-collect` |
| `GTAT_PROJECT` | `POSETAG_PROJECT` |
| `GTAT_PROJECTS_DIR` | `POSETAG_PROJECTS_DIR` |
| `~/.config/gt6dof_atag/` | `~/.config/posetag/` |
| `~/gt-6dof/` | `~/posetag/` |

## Compatibility Retained in Phase 0

The following aliases still work intentionally while the rename settles:

- `import gt6dof_atag`
- legacy `gtat*` console scripts
- legacy `GTAT_PROJECT` and `GTAT_PROJECTS_DIR` environment variables
- legacy config discovery under `~/.config/gt6dof_atag/`
- legacy project-home discovery under `~/gt-6dof/`

Those aliases now delegate to the canonical PoseTag implementation and should be
treated as temporary migration shims, not the preferred interface.

## Notes for Users

- New docs, examples, and packaging metadata use only `PoseTag` and `posetag`.
- Scientific behaviour is unchanged by this rename phase.
- Later phases can remove the legacy aliases once the package and CLI migration
  is complete and downstream users have had time to update.
