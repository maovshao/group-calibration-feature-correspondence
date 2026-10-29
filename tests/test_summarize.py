#!/usr/bin/env python3
"""Offline software tests only: synthetic fixtures are NOT experiment results.

All CLI fixture files and CSVs exist only inside TemporaryDirectory under /tmp.
This suite never writes synthetic metadata to the project's outputs/runs.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SUMMARY_PATH = Path(__file__).resolve().parents[1] / "code/summarize.py"
SPEC = importlib.util.spec_from_file_location("summary_under_test", SUMMARY_PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def synthetic_run(method="gc_ts", seed=0, **changes):
    """Deliberately synthetic, in-memory metadata for software assertions."""
    row = {
        "software_test_only": True,
        "status": "completed",
        "level": "main",
        "method": method,
        "seed": seed,
        "run_id": "SYNTHETIC_SOFTWARE_TEST_%s_%s" % (method, seed),
        "upstream_commit": "a" * 40,
        "data_sha256": {"val": "b" * 64, "test": "c" * 64},
        "code_sha256": {"code/run.py": "d" * 64, "code/configs/main.yaml": "e" * 64},
        "split_sha256": str(seed) * 64,
        "initialization_hashes": ["f" * 64] * 20 if method.startswith("gc_") else [],
        "metrics": {"ece": 0.02, "nll": 0.5, "accuracy": 0.9,
                    "brier": 0.1, "prediction_changes": 0},
    }
    row.update(changes)
    return row


class SummarizerStatisticsTests(unittest.TestCase):
    def test_sample_standard_deviation_uses_ddof_one(self):
        runs = [synthetic_run(seed=0), synthetic_run(seed=1)]
        runs[0]["metrics"]["ece"] = 0.1
        runs[1]["metrics"]["ece"] = 0.3
        rows, stats, paired, _, missing = summary.summarize(runs)
        result = next(r for r in stats if r["metric"] == "ece")
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mean"], 0.2)
        self.assertAlmostEqual(result["sample_std_ddof1"], math.sqrt(0.02))
        self.assertNotAlmostEqual(result["sample_std_ddof1"], 0.1)
        self.assertEqual([r["seed"] for r in rows], [0, 1])
        self.assertEqual(paired, [])
        self.assertEqual(len(missing), 18)

    def test_single_seed_standard_deviation_is_na(self):
        _, stats, _, _, _ = summary.summarize([synthetic_run()])
        self.assertTrue(all(r["n"] == 1 and r["sample_std_ddof1"] == "NA" for r in stats))

    def test_paired_directions_and_single_pair_na(self):
        ts = synthetic_run("ts")
        gc = synthetic_run("gc_ts")
        shuffled = synthetic_run("gc_ts_shuffle")
        ts["metrics"].update(ece=0.03, nll=0.8)
        shuffled["metrics"].update(ece=0.01, nll=0.4)
        _, _, paired, stats, _ = summary.summarize([shuffled, gc, ts])
        by_comparison = {r["comparison"]: r for r in paired}
        self.assertAlmostEqual(by_comparison["delta_gc"]["ece_difference"], 0.01)
        self.assertAlmostEqual(by_comparison["delta_gc"]["nll_difference"], 0.3)
        self.assertAlmostEqual(by_comparison["delta_feature"]["ece_difference"], -0.01)
        self.assertAlmostEqual(by_comparison["delta_feature"]["nll_difference"], -0.1)
        self.assertTrue(all(r["n"] == 1 and r["sample_std_ddof1"] == "NA" for r in stats))

    def test_paired_summary_uses_sample_standard_deviation(self):
        runs = []
        for seed, difference in ((0, 0.01), (1, 0.03)):
            ts, gc = synthetic_run("ts", seed), synthetic_run("gc_ts", seed)
            ts["metrics"]["ece"] = gc["metrics"]["ece"] + difference
            runs.extend([ts, gc])
        _, _, paired, stats, _ = summary.summarize(runs)
        result = next(r for r in stats if r["metric"] == "ece_difference")
        self.assertEqual(len(paired), 2)
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mean"], 0.02)
        self.assertAlmostEqual(result["sample_std_ddof1"], math.sqrt(0.0002))

    def test_only_intersecting_seeds_are_paired(self):
        runs = [synthetic_run("ts", 0), synthetic_run("gc_ts", 1),
                synthetic_run("gc_ts_shuffle", 2)]
        _, stats, paired, paired_stats, missing = summary.summarize(runs)
        self.assertEqual(paired, [])
        self.assertEqual(paired_stats, [])
        self.assertTrue(all(r["n"] == 1 for r in stats))
        self.assertEqual(len(missing), 17)

    def test_all_twenty_pairs_are_complete_without_imputation(self):
        runs = [synthetic_run(method, seed) for method in summary.METHODS for seed in range(5)]
        rows, stats, paired, paired_stats, missing = summary.summarize(runs)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(paired), 10)
        self.assertTrue(all(r["n"] == 5 for r in stats + paired_stats))
        self.assertEqual(missing, [])


class SummarizerSelectionTests(unittest.TestCase):
    def test_blocked_failed_invalid_smoke_pilot_excluded(self):
        ignored = [{"status": status, "level": level}
                   for status, level in (("blocked", "main"), ("failed", "main"),
                                         ("invalid", "main"), ("completed", "smoke"),
                                         ("completed", "pilot"))]
        rows, stats, paired, paired_stats, missing = summary.summarize(ignored)
        self.assertEqual((rows, stats, paired, paired_stats), ([], [], [], []))
        self.assertEqual(len(missing), 20)

    def test_pilot_not_automatically_reused_or_counted_as_duplicate(self):
        main = synthetic_run()
        pilot = synthetic_run(level="pilot", run_id="SYNTHETIC_PILOT_ONLY")
        rows, *_ = summary.summarize([pilot, main])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], main["run_id"])

    def test_duplicate_completed_attempts_rejected(self):
        original = synthetic_run()
        repeated = synthetic_run(run_id="SYNTHETIC_ATTEMPT_2")
        with self.assertRaisesRegex(ValueError, "Multiple completed attempts"):
            summary.summarize([original, repeated])

    def test_failed_attempt_does_not_hide_completed_run(self):
        failed = synthetic_run(status="failed", run_id="SYNTHETIC_FAILED")
        rows, *_ = summary.summarize([failed, synthetic_run()])
        self.assertEqual(len(rows), 1)

    def test_unknown_methods_and_invalid_seed_types_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected method/seed"):
            summary.summarize([synthetic_run("unplanned_variant")])
        for seed in (-1, 5, True, False, 0.0, "0"):
            with self.subTest(seed=repr(seed)):
                with self.assertRaisesRegex(ValueError, "Unexpected method/seed"):
                    summary.summarize([synthetic_run(seed=seed)])

    def test_nonfinite_boolean_and_missing_metrics_rejected(self):
        for metric in summary.METRICS:
            for bad_value in (None, True, float("nan"), float("inf"), -float("inf"), "0.1"):
                with self.subTest(metric=metric, value=repr(bad_value)):
                    row = synthetic_run()
                    row["metrics"][metric] = bad_value
                    with self.assertRaisesRegex(ValueError, "Nonfinite or missing completed metric"):
                        summary.summarize([row])
            row = synthetic_run()
            del row["metrics"][metric]
            with self.assertRaisesRegex(ValueError, "Nonfinite or missing completed metric"):
                summary.summarize([row])
        for invalid_metrics in (None, [], "missing"):
            with self.subTest(metrics=invalid_metrics):
                with self.assertRaisesRegex(ValueError, "Nonfinite or missing completed metric"):
                    summary.summarize([synthetic_run(metrics=invalid_metrics)])


class SummarizerProvenanceTests(unittest.TestCase):
    def test_missing_empty_or_wrong_type_split_hash_rejected(self):
        for bad_hash in (None, "", " ", [], {}, 0):
            with self.subTest(value=repr(bad_hash)):
                with self.assertRaisesRegex(ValueError, "identical splits"):
                    summary.summarize([synthetic_run(split_sha256=bad_hash)])
        row = synthetic_run()
        del row["split_sha256"]
        with self.assertRaisesRegex(ValueError, "identical splits"):
            summary.summarize([row])

    def test_different_splits_within_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical splits"):
            summary.summarize([synthetic_run("ts"), synthetic_run("gc_ts", split_sha256="9" * 64)])

    def test_different_splits_between_seeds_allowed(self):
        rows, *_ = summary.summarize([synthetic_run(seed=0), synthetic_run(seed=1)])
        self.assertEqual(len(rows), 2)

    def test_missing_empty_or_mismatched_initialization_hashes_rejected(self):
        for bad_hashes in (None, [], {}, [""], [" "], [None], ["9" * 64] * 20):
            with self.subTest(value=repr(bad_hashes)):
                with self.assertRaisesRegex(ValueError, "initializations"):
                    summary.summarize([synthetic_run("gc_ts"),
                                       synthetic_run("gc_ts_shuffle", initialization_hashes=bad_hashes)])
        for empty_matching in ([""], {"partition_0": ""}):
            with self.subTest(matching=empty_matching):
                with self.assertRaisesRegex(ValueError, "initializations"):
                    summary.summarize([synthetic_run("gc_ts", initialization_hashes=empty_matching),
                                       synthetic_run("gc_ts_shuffle", initialization_hashes=empty_matching)])
        row = synthetic_run("gc_ts")
        del row["initialization_hashes"]
        with self.assertRaisesRegex(ValueError, "initializations"):
            summary.summarize([row, synthetic_run("gc_ts_shuffle")])

    def test_matching_ordered_partition_hashes_allowed(self):
        first = synthetic_run("gc_ts")
        second = synthetic_run("gc_ts_shuffle")
        first["initialization_hashes"] = ["1" * 64, "2" * 64]
        second["initialization_hashes"] = copy.deepcopy(first["initialization_hashes"])
        self.assertEqual(len(summary.summarize([first, second])[2]), 1)
        second["initialization_hashes"].reverse()
        with self.assertRaisesRegex(ValueError, "initializations"):
            summary.summarize([first, second])

    def test_missing_or_empty_code_data_and_upstream_provenance_rejected(self):
        for key in ("upstream_commit", "code_sha256", "data_sha256"):
            for bad_value in (None, "", " ", {}, [], {"some_file": ""}, {"some_file": None}):
                with self.subTest(key=key, value=repr(bad_value)):
                    with self.assertRaisesRegex(ValueError, "provenance"):
                        summary.summarize([synthetic_run(**{key: bad_value})])
            missing = synthetic_run()
            del missing[key]
            with self.assertRaisesRegex(ValueError, "provenance"):
                summary.summarize([missing])

    def test_changed_code_data_and_upstream_provenance_rejected_across_seeds(self):
        variations = {
            "upstream_commit": "9" * 40,
            "code_sha256": {"code/run.py": "9" * 64},
            "data_sha256": {"val": "9" * 64, "test": "c" * 64},
        }
        for key, value in variations.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    summary.summarize([synthetic_run(seed=0), synthetic_run(seed=1, **{key: value})])

    def test_equivalent_provenance_mapping_order_allowed(self):
        first, second = synthetic_run(seed=0), synthetic_run(seed=1)
        second["data_sha256"] = dict(reversed(list(second["data_sha256"].items())))
        second["code_sha256"] = dict(reversed(list(second["code_sha256"].items())))
        self.assertEqual(len(summary.summarize([first, second])[0]), 2)

    def test_matching_versions_and_legacy_records_allowed(self):
        versions = {"torch": {"version": "2.0.TEST", "import_spec_found": True},
                    "numpy": {"version": "1.0.TEST", "import_spec_found": True}}
        first = synthetic_run(seed=0, versions=versions)
        second = synthetic_run(seed=1, versions=dict(reversed(list(versions.items()))))
        self.assertEqual(len(summary.summarize([first, second])[0]), 2)
        legacy = [synthetic_run(seed=0), synthetic_run(seed=1)]
        self.assertEqual(len(summary.summarize(legacy)[0]), 2)
        ignored = synthetic_run(status="blocked", versions={"torch": "DIFFERENT.TEST"})
        self.assertEqual(len(summary.summarize([first, second, ignored])[0]), 2)

    def test_changed_missing_or_empty_versions_rejected_when_present(self):
        versions = {"torch": {"version": "2.0.TEST", "import_spec_found": True}}
        first = synthetic_run(seed=0, versions=versions)
        for other in (None, {}, "", {"torch": ""}, {"torch": {}},
                      {"torch": {"version": "2.1.TEST", "import_spec_found": True}},
                      {"torch": {"version": "2.0.TEST", "import_spec_found": False}}):
            with self.subTest(versions=other):
                with self.assertRaisesRegex(ValueError, "provenance: versions"):
                    summary.summarize([first, synthetic_run(seed=1, versions=other)])
        with self.assertRaisesRegex(ValueError, "provenance: versions"):
            summary.summarize([first, synthetic_run(seed=1)])
        with self.assertRaisesRegex(ValueError, "provenance: versions"):
            summary.summarize([synthetic_run(versions={})])


class SummarizerCliTests(unittest.TestCase):
    def run_cli(self, base, rows, extra=()):
        inputs = base / "synthetic_inputs"
        output = base / "synthetic_summary"
        inputs.mkdir(exist_ok=True)
        for i, row in enumerate(rows):
            folder = inputs / str(i)
            folder.mkdir()
            (folder / "metadata.json").write_text(json.dumps(row))
        result = subprocess.run([sys.executable, str(SUMMARY_PATH), "--runs", str(inputs),
                                 "--output", str(output), "--allow-partial", *extra], capture_output=True, text=True)
        return result, output, inputs

    def test_no_valid_main_runs_write_status_but_no_numeric_csv(self):
        for rows in ([], [synthetic_run(status="blocked"), synthetic_run(level="smoke"),
                          synthetic_run(level="pilot")]):
            with self.subTest(input_count=len(rows)), tempfile.TemporaryDirectory(prefix="gc_summary_software_test_") as tmp:
                result, output, _ = self.run_cli(Path(tmp), rows)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(output.glob("*.csv")), [])
                status = json.loads((output / "summary_status.json").read_text())
                self.assertEqual(status["status"], "no_completed_main_runs")
                self.assertFalse(status["numeric_outputs_written"])
                self.assertEqual(len(status["missing"]), 20)

    def test_empty_selection_preserves_and_discloses_stale_csv(self):
        with tempfile.TemporaryDirectory(prefix="gc_summary_software_test_") as tmp:
            base = Path(tmp)
            output = base / "synthetic_summary"
            output.mkdir()
            old = output / "summary.csv"
            old.write_text("SYNTHETIC SOFTWARE TEST SENTINEL,DO NOT OVERWRITE\n")
            original = old.read_bytes()
            result, _, _ = self.run_cli(base, [])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(old.read_bytes(), original)
            report = json.loads((output / "summary_status.json").read_text())
            self.assertFalse(report["numeric_outputs_written"])
            self.assertEqual(report["preexisting_csv_not_modified"], ["summary.csv"])

    def test_invalid_provenance_fails_before_creating_csv(self):
        with tempfile.TemporaryDirectory(prefix="gc_summary_software_test_") as tmp:
            result, output, _ = self.run_cli(Path(tmp), [synthetic_run(split_sha256=None)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("identical splits", result.stderr)
            self.assertFalse(output.exists())

    def test_manifest_can_explicitly_choose_one_duplicate_attempt(self):
        with tempfile.TemporaryDirectory(prefix="gc_summary_software_test_") as tmp:
            base = Path(tmp)
            result, output, inputs = self.run_cli(base, [synthetic_run(),
                                                       synthetic_run(run_id="SYNTHETIC_SECOND_ATTEMPT")])
            self.assertNotEqual(result.returncode, 0)
            manifest = base / "synthetic_manifest.json"
            manifest.write_text(json.dumps([str(inputs / "1" / "metadata.json")]))
            result = subprocess.run([sys.executable, str(SUMMARY_PATH), "--manifest", str(manifest),
                                     "--output", str(output), "--allow-partial"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "summary_status.json").read_text())
            self.assertEqual(report["completed_main_method_seed_pairs"], 1)
            self.assertEqual(report["status"], "partial")
            self.assertIn("SYNTHETIC_SECOND_ATTEMPT", (output / "per_seed.csv").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
