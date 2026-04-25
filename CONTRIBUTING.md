# Contributing to PoseTag

Thanks for contributing to PoseTag.

PoseTag is scientific Python software for generating reproducible ground-truth
6-DoF object poses. Contributions should prioritise scientific correctness,
reproducibility, installability, and documentation accuracy over superficial
cleanup.

## Project Priorities

When proposing changes, please keep these principles in mind:

- preserve scientific correctness before convenience
- preserve reproducibility before broad refactoring
- keep pose conventions, units, schemas, and output assumptions explicit
- prefer small, reviewable changes over wide cleanup passes
- use canonical PoseTag naming where possible

The core pose composition contract is:

```text
T_cam_object = T_cam_board @ T_board_object
```

Do not change pose maths, coordinate conventions, quaternion ordering, angle
units, translation units, or output schemas casually. If a future change needs
to affect those areas, it should be discussed explicitly and documented with
tests.

## Opening Useful Issues

Please use the GitHub issue templates to keep reports reproducible and useful.

- Use `Bug report` for reproducible failures or incorrect behaviour.
- Use `Workflow validation` to report whether a documented workflow worked in
  practice, including exact commands and environment details.
- Use `Research use case` to document public research or evaluation contexts in
  which PoseTag is being used.
- Use `Feature request` for proposed improvements or missing capabilities.

Helpful issue reports usually include:

- the exact commands or steps you ran
- relevant input files, media types, calibration assumptions, and resolutions
- the observed output or error
- the expected behaviour
- the operating system, Python version, and install method
- any scientific or reproducibility impact

## Running Tests

Run the current automated test suite from the repository root:

```bash
python3 -m unittest discover -s tests
```

If you touch packaging, naming compatibility, scientific logic, schemas, or
workflow-critical behaviour, please add or update tests where possible.

## Validating Workflow Steps

If you change documentation, developer infrastructure, or GitHub workflow
files, please validate the relevant steps and report the exact commands used.

For GitHub Actions workflow linting, run:

```bash
./scripts/check-workflows.sh
```

That script uses a local `actionlint` binary when available, or Docker with the
`rhysd/actionlint` image when Docker is running.

For user-facing workflow validation, please record:

- what workflow stage you validated
- the commands and inputs used
- the environment used
- what outputs were produced
- whether the documented instructions matched actual behaviour

Opening a `Workflow validation` issue is a good way to capture that evidence in
public.

## Proposing Changes

Before opening a pull request, please:

1. read the relevant code and docs carefully
2. identify the scientific or workflow invariant that must be preserved
3. keep the change set as small and coherent as practical
4. update tests and docs when public behaviour changes
5. explain the motivation, scope, and verification in the pull request

Good pull requests usually make it easy to answer:

- what changed
- why it changed
- how it was tested or validated
- whether any naming, schema, or workflow behaviour changed

## Documentation and Open Development Evidence

PoseTag is being prepared for a reproducible scientific software release and a
future JOSS submission. Public issues, validation reports, contributor
discussion, and documented fixes all help build evidence of healthy open
development.

If you identify a documentation mismatch, installation problem, workflow gap,
or reproducibility concern, reporting it clearly is a valuable contribution.
