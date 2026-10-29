# Verification record

Checked on **2026-10-28**. Machine-readable results are in [`results/provenance.json`](results/provenance.json), under `release_validation`.

| Check | Result | Scope |
|---|---|---|
| Offline tests | **36 passed** | Original 26 statistical/provenance tests plus 10 manifest and complete-batch tests |
| Fresh analysis installation | **Passed** | Python 3.12, macOS arm64; isolated environment from `requirements/analysis.txt`; `pip check` passed |
| README plotting command | **Passed** | Created PNG, SVG, and PDF from committed per-seed measurements |
| Summary reconstruction | **Passed** | All four CSVs reproduced byte-for-byte from the original 20 run metadata files |
| Saved-prediction audit | **436/436 checks passed** | CPU audit of the original 20 runs using PyTorch 2.4.1 and TorchMetrics 1.9.0 |
| Scientific runner comparison | **Passed** | All original runner functions/classes match by Python AST, except the manifest orchestration in `run_one` and `main` |
| Fresh GPU batch from reorganized repository | **Not run** | Requires a new GPU allocation and Linux/CUDA setup |

The saved-artifact audit checked cache hashes and dimensions, source hashes, saved source snapshots, pairing, all split and shuffle indices, regenerated initialization, positive finite temperatures, group coverage, optimizer observations, recorded GPU execution, predictions, and recomputed metrics. The largest absolute metric difference was **1.49 × 10⁻⁸**, below the `1e-7` tolerance. Float32 softmax reconstruction uses `atol=1e-7, rtol=1e-6` to accommodate CPU-kernel differences across architectures.

Original data and run records were read without alteration. Synthetic test fixtures lived in temporary directories and were never added to experiment results.

## Remaining release check

Run these commands from a fresh repository copy after obtaining an allocated GPU. Installation can run beforehand:

```bash
bash code/setup.sh --gpu
source .venv-gpu/bin/activate
python code/preflight.py
python code/run.py --level smoke --method gc_ts --seeds 0 \
  --manifest-out outputs/smoke_manifest.json
```

After smoke succeeds:

```bash
python code/run.py --level main --method all --seeds 0 1 2 3 4 \
  --manifest-out outputs/main_manifest.json
python code/summarize.py --manifest outputs/main_manifest.json \
  --output outputs/reproduced
python code/validate.py --manifest outputs/main_manifest.json \
  --summary outputs/reproduced
python code/plot.py --input outputs/reproduced/per_seed.csv \
  --output outputs/reproduced/figures
```

Compare each reproduced method/seed with `results/per_seed.csv`, keeping new results separate. Record the environment, installation receipt, successful validation report, and any numerical differences before updating the GPU verification status. The reported original results remain unchanged.
