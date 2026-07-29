# Contributing

Thanks for considering a contribution. This project is still young — issues,
PRs, and questions are all welcome.

## Setup

```bash
git clone https://github.com/bradpatton/agon.git
cd agon
uv sync --extra dev
pre-commit install
```

If you're working on `UltralyticsDetector`, `BoTSORTTracker`, or anything
else behind the `[train]` extra, see the [README's Development
section](README.md#development) for why the `Dockerfile` exists and when
you'll need it — those paths need torch, which doesn't have wheels for every
platform.

## Before opening a PR

```bash
pytest
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/agon
```

All of the above run in CI; running them locally first saves a round trip.
`pre-commit install` handles the lint/format checks automatically on commit.

## Guidelines

- **Tests for new logic.** Pure-Python/synthetic-input tests, matching what's
  already in `tests/` — no video or model file should be required to run the
  suite. If you're adding a new `Detector`/`PitchCalibrator`/`TeamClassifier`/
  `FrameTracker` backend, it should satisfy the existing protocol in
  `src/agon/interfaces.py` (or `detection/base.py` for
  `FrameTracker`) so it's swappable via config without touching
  `pipeline.py`.
- **Comments explain *why*, not *what*.** Only add one when there's a
  non-obvious constraint, a workaround for a specific bug, or a limitation a
  reader would otherwise be surprised by. Well-named code should speak for
  itself otherwise.
- **Don't claim more accuracy than a change actually has.** This codebase has
  a few components (`PitchKeypointCalibrator`, the pixel-based
  `TeamAssigner`) with real, documented limitations. If you improve one,
  update its docstring's limitations section rather than just removing it —
  and if you're not sure whether a change actually fixes the underlying
  problem or just moves it, say so in the PR description.
- **Breaking changes to `agon.export.schema`** need a
  `SCHEMA_VERSION` bump (see that module's docstring) — the export format is
  a public contract for downstream ML tooling, not an internal detail.

## Reporting bugs / requesting features

Open an issue with as much repro detail as you have (footage
characteristics, config used, checkpoint type — soccer-specific vs. generic
COCO matters a lot for what's actually reachable). See the issue templates
for the specific fields that help most.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
