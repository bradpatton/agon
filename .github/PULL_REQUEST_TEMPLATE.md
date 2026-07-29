## What this does

## Why

## Testing
- [ ] `pytest` passes
- [ ] `ruff check src/ tests/` and `ruff format --check src/ tests/` pass
- [ ] `mypy src/agon` passes
- [ ] New logic has tests (pure-Python/synthetic-input, no video/model file required — see CONTRIBUTING.md)
- [ ] If this touches `agon.export.schema`: `SCHEMA_VERSION` bumped if the change is breaking

## Notes for reviewers
Anything non-obvious: accuracy tradeoffs, known limitations of the approach, what wasn't validated (e.g. only tested against a generic COCO checkpoint, not a soccer-specific one).
