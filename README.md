# OOA

OOA is a research codebase for studying when an
input-dependent selector can improve over fixed prediction actions in multi-label recognition.
The repository contains the reusable Python package, experiment entry points, tests, frozen
protocol documents, and lightweight configuration needed to reproduce the code path.

> The datasets, model weights, cached features, run directories, generated images, and manuscript
> submission packages are intentionally not included. See [DATA.md](DATA.md) for the expected local
> layout and redistribution boundaries.

## Repository layout

```text
configs/                 Experiment configuration
docs/                    Protocols, preregistrations, and result summaries
scripts/                 Data preparation, training, evaluation, and analysis entry points
src/edcr/                Reusable Python package
tests/                   Unit and protocol-regression tests
experiment_registry.jsonl  Chronological experiment registry
```

## Installation

Python 3.11–3.14 is supported. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

PyTorch and OpenCLIP may download pretrained weights on first use. Dataset preparation commands
may also download public datasets; review their terms before running them.

## Verification

The unit and protocol-regression suite does not require the full research datasets:

```bash
pytest
ruff check src scripts tests
```

At the time this public snapshot was prepared, the suite contained 101 passing tests.

## Reproducing experiments

Start with the environment and configuration audit, then prepare a local COCO layout:

```bash
python scripts/audit.py --config configs/pilot.yaml --data-root /path/to/coco
python scripts/prepare.py --config configs/pilot.yaml --data-root /path/to/coco
python scripts/audit_splits.py --config configs/pilot.yaml
```

Expected COCO layout:

```text
COCO_ROOT/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  train2017/
  val2017/
```

The experiment series is cumulative and contains sealed-evaluation constraints. Before running
later-stage scripts, the relevant preregistration under
`docs/`, and `experiment_registry.jsonl`. In particular, do not use sealed evaluation labels for
model fitting or hyperparameter selection.

Most scripts write generated files beneath `data/`, `runs/`, `reports/`, `output/`, or `paper/`.
Those paths are ignored by default to prevent accidental publication of large or restricted
artifacts.

## Scope of the public snapshot

Included:

- reusable source code and command-line scripts;
- tests and dependency lockfile;
- experiment configuration, protocols, preregistrations, and summaries;
- the text-only experiment registry.

Excluded:

- COCO, Pascal VOC, Open Images, and CIFAR-100 files or derived per-example manifests;
- model checkpoints, pretrained weights, feature arrays, and run caches;
- generated intervention images and contact sheets;
- journal submission packages, build products, and local scratch files.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks. Please report suspected credential,
privacy, or dataset-redistribution issues privately as described in [SECURITY.md](SECURITY.md).

## License

The original code and documentation in this repository are released under the [MIT License](LICENSE).
Third-party datasets, pretrained weights, models, and dependencies retain their own licenses and
terms; they are not relicensed by this repository.
