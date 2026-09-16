# Contributing

## Development setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src scripts tests
```

Keep reusable logic in `src/edcr/` and thin experiment entry points in `scripts/`. Add tests for
behavior changes, especially changes to split handling, sealed-evaluation rules, calibration, or
metric definitions.

## Research integrity

Several experiments use frozen protocols and label-sealed evaluations. A contribution must not
silently change a frozen decision, read held-out labels early, or overwrite the historical record.
Document new analyses separately and state whether they are confirmatory, exploratory, or post hoc.

## Data hygiene

Do not contribute datasets, weights, generated images, caches, or per-example derived records.
Before opening a change, inspect the staged files and run a credential scan appropriate to your
environment.
