# Verification record

Checked on **2026-10-28**. Details are recorded under `release_validation` in [results/provenance.json](results/provenance.json).

| Check | Result | Scope |
|---|---|---|
| Software tests | **36 passed** | Statistics, provenance, manifests, and complete-batch selection |
| Clean analysis installation | **Passed** | Python 3.12, macOS arm64; locked dependencies and `pip check` |
| Plotting command | **Passed** | PNG, SVG, and PDF generated from the saved measurements |
| Result notebook | **Passed** | All cells executed in an independent Python 3.12 CPU environment; embedded PNG identical to CLI and committed figure |
| Summary reconstruction | **Passed** | Four CSVs reproduced byte-for-byte from the original 20 run records |
| Saved-artifact audit | **436/436 passed** | Original predictions, metrics, hashes, splits, initialization, and optimizer records; audited on CPU |
| Scientific runner comparison | **Passed** | Original fitting functions unchanged; manifest orchestration isolated |

The largest recomputed metric difference was **1.49 × 10⁻⁸**, within the `1e-7` tolerance. Original data and run records were preserved.
