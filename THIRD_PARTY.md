# Third-party materials

## Paper and official implementation

Jia-Qi Yang, De-Chuan Zhan, and Le Gan. **Beyond Probability Partitions: Calibrating Neural Networks with Semantic Aware Grouping.** NeurIPS 2023.

- Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/b693a240cf1009bff9fa4422141c9392-Paper-Conference.pdf
- Official repository: https://github.com/ThyrixYang/group_calibration
- Fixed commit: `1fb9c6bb7b98e3960b3a9281ef25e8fcedfb1a88`

The setup script obtains this repository and its two CIFAR-10 / ResNet-152 caches directly from the authors. Upstream files remain in an ignored checkout under `code/upstream/` and are not vendored into this repository's history. The pinned checkout contains no standalone LICENSE file; this project does not assign a new license to the authors' implementation or data.

The local runner imports the pinned implementation and adds experiment scheduling, split/feature-shuffle controls, runtime observations, and result recording. The original methods and evaluator remain the reproduction baseline. Runtime observation hooks verify gradients, parameters, temperatures, and group sizes while preserving the fitted algorithm.

## Dataset caches

The cached features, logits, and labels are supplied by the official repository. Setup verifies their SHA-256 hashes against `results/provenance.json` before use. The experiment uses CIFAR-10 and the authors' pretrained ResNet-152; the classifier is not retrained here.

## Software dependencies

Dependency versions are listed in `requirements/analysis.txt` and `requirements/gpu-lock.txt`. Individual packages retain their respective licenses. The GPU lock records the original experiment environment; the analysis lock supports CSV plotting separately.
