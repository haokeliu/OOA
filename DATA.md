# Data and artifact policy

This repository does not redistribute datasets, pretrained weights, generated images, or derived
per-example records. Obtain each resource from its official distributor and comply with its
license, access conditions, and citation requirements.

The experiment scripts use these resources:

- COCO 2017 images and instance annotations;
- Pascal VOC 2007 and 2012 images and annotations;
- Open Images V7 images and human-verified labels;
- CIFAR-100;
- pretrained weights used by PyTorch, torchvision, and OpenCLIP.

The default local working paths are under `data/`. Common generated artifacts are written beneath
`data/`, `runs/`, `reports/`, `output/`, and `paper/`; all are ignored by Git in this public
snapshot. The preparation scripts create the required derived manifests and caches locally.

Do not commit:

- downloaded archives or extracted dataset files;
- annotations or manifests containing per-example third-party records;
- checkpoints, feature tensors, serialized estimators, or cached predictions;
- generated or edited dataset images;
- secrets, access tokens, signed URLs, or machine-specific paths.

Before publishing a reproduction artifact, audit every added file for dataset redistribution
rights, personal data, secrets, and unexpectedly large binaries.
