# Group calibration and feature correspondence

Does group calibration benefit from matching each image to its own internal features? We study this question using the authors' cached CIFAR-10 / ResNet-152 outputs.

[Paper: Beyond Probability Partitions (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/b693a240cf1009bff9fa4422141c9392-Paper-Conference.pdf) · [Official code](https://github.com/ThyrixYang/group_calibration)

**Start here: [main.ipynb](main.ipynb).** Open it in Jupyter or VS Code with Python 3.11/3.12 and run all cells. It installs the analysis dependencies, reads the saved measurements, and redraws the figure below on CPU.

## Results

Four conditions × five paired seeds produced 20 GPU runs. Values are mean ± sample standard deviation; lower ECE and NLL are better.

| Method | ECE | NLL |
|---|---:|---:|
| Uncalibrated | 0.024763 ± 0.000547 | 0.173049 ± 0.002638 |
| Global TS | 0.007889 ± 0.001860 | 0.144950 ± 0.001370 |
| GC+TS | 0.007251 ± 0.002144 | 0.143622 ± 0.001840 |
| Shuffled GC+TS | 0.008059 ± 0.002199 | 0.145304 ± 0.001391 |

![Paired ECE measurements](figures/paired_ece.png)

GC+TS improves ECE over TS in 4/5 seeds and improves both ECE and NLL over shuffled GC+TS in 5/5. These results support a contribution from sample–feature correspondence in this setting. Predicted classes remain unchanged; mean accuracy is 95.7822%.

[Per-seed results](results/per_seed.csv) · [Summary](results/summary.csv) · [Paired differences](results/paired_differences.csv) · [Paired summary](results/paired_summary.csv)

## Reproduce the figure

From the repository root, use the Notebook above or these commands:

```bash
bash code/setup.sh --analysis
source .venv-analysis/bin/activate
python code/plot.py --input results/per_seed.csv --output figures
```

Both routes use the same plotting script and export PNG, SVG, and PDF. The Notebook displays the PNG inline and saves exports to `outputs/notebook/`.

## Reproduce the experiments

The fixed classifier supplies features, logits, and labels. Group learning, final temperature fitting, and evaluation use 5,000 / 1,000 / 9,000 samples. GC uses 2 groups, 20 partitions, an optimization limit of 100, and regularization 0.1. The shuffled condition permutes feature rows independently within each split while retaining logits and labels; paired methods share splits and initialization.

Use Linux, Python 3.11, and one allocated Slurm GPU. Installation can run before allocation; fitting uses logical `cuda:0`. Run smoke successfully before the main batch.

```bash
bash code/setup.sh --gpu
source .venv-gpu/bin/activate
python code/preflight.py
python code/run.py --level smoke --method gc_ts --seeds 0 \
  --manifest-out outputs/smoke_manifest.json
python code/run.py --level main --method all --seeds 0 1 2 3 4 \
  --manifest-out outputs/main_manifest.json
python code/summarize.py --manifest outputs/main_manifest.json \
  --output outputs/reproduced
python code/validate.py --manifest outputs/main_manifest.json \
  --summary outputs/reproduced
python code/plot.py --input outputs/reproduced/per_seed.csv \
  --output outputs/reproduced/figures
```

New runs save their own configurations, predictions, metrics, and logs under `outputs/`. Use new manifest names for another batch. Summarization requires all 20 completed combinations; validation checks pairing and recomputes metrics.

## Sources and checks

Setup fetches official commit `1fb9c6bb7b98e3960b3a9281ef25e8fcedfb1a88` and verifies both cache hashes. [Provenance](results/provenance.json) records the original environment, protocol, hashes, and completed checks; [THIRD_PARTY.md](THIRD_PARTY.md) documents attribution.

[Verification](VERIFICATION.md): 36 tests passed, 436 saved-artifact checks passed, and all four CSVs were reconstructed byte-for-byte. Run the software tests with `python -m unittest discover -s tests -v`.

## Contributors

- Wei Liu: research question, alternative hypothesis, experiment design, and core conclusions.
- Yuanman Mu: paper reading, method explanation, and evidence review.
- Liuwen Kong: reproduction, environment setup, GPU execution, and checks.
- Jiansong Zhang: statistics, figures, and report integration.
