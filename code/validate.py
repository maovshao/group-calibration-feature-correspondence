#!/usr/bin/env python3
"""Audit a selected 20-run batch from saved predictions on CPU; never fit a model."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys

from summarize import METHODS, METRICS, load_manifest_paths, summarize

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1fb9c6bb7b98e3960b3a9281ef25e8fcedfb1a88"
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compare_csv(path, expected):
    with path.open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    if len(actual) != len(expected):
        return False
    for row, reference in zip(actual, expected):
        if set(row) != set(reference):
            return False
        for key, value in reference.items():
            if isinstance(value, (int, float)):
                if not math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-15):
                    return False
            elif row[key] != str(value):
                return False
    return True


def audit(root, paths, summary_folder):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchmetrics.functional import calibration_error

    torch.set_num_threads(1)
    records = [read(path) for path in paths]
    rows, stats, paired, paired_stats, missing = summarize(records)
    checks = {
        "exact_main_grid": len(records) == len(rows) == 20 and not missing,
        "all_completed_main": all(r.get("status") == "completed" and r.get("level") == "main" for r in records),
    }
    if not all(checks.values()):
        raise ValueError("Validation requires exactly four completed main conditions for seeds 0-4")
    upstream = root / "code/upstream"
    def git(*args):
        return subprocess.check_output(["git", "-C", str(upstream), *args])
    checks["pinned_upstream_clean"] = (
        git("rev-parse", "HEAD").decode().strip() == COMMIT
        and not git("status", "--porcelain", "--untracked-files=all").strip())
    expected_hashes = read(ROOT / "results/provenance.json")["data_sha256"]
    caches, hashes = {}, {}
    for split, size in (("val", 5000), ("test", 10000)):
        path = upstream / f"datasets/cifar10_resnet152_0.9576_{split}.pkl"
        hashes[split] = digest(path)
        checks[f"{split}_cache_hash"] = hashes[split] == expected_hashes[split]
        if not checks[f"{split}_cache_hash"]:
            raise ValueError(f"Cache integrity check failed before loading {split}")
        with path.open("rb") as stream:
            caches[split] = pickle.load(stream)
        checks[f"{split}_cache_shape"] = (
            list(caches[split]["features"].shape) == [size, 2048]
            and list(caches[split]["logits"].shape) == [size, 10]
            and list(caches[split]["labels"].shape) == [size])
    initialized = {}
    for seed in range(5):
        torch.manual_seed(seed)
        initialized[seed] = []
        for _ in range(20):
            h = hashlib.sha256()
            for parameter in torch.nn.Linear(2048, 2, bias=False).parameters():
                t = parameter.detach().cpu().contiguous()
                h.update(str(tuple(t.shape)).encode())
                h.update(str(t.dtype).encode())
                h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
            initialized[seed].append(h.hexdigest())
    report = {"validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "Saved-artifact audit on CPU; no fitting or GPU computation.",
              "checks": checks, "runs": [], "metrics_absolute_tolerance": 1e-7,
              "probability_reconstruction_tolerance": {"absolute": 1e-7, "relative": 1e-6}}
    for path, record in zip(paths, records):
        folder = path.parent
        observations = read(folder / "observations.json")
        before = {f.name: digest(f) for f in folder.iterdir() if f.is_file()}
        with np.load(folder / "indices.npz", allow_pickle=False) as saved:
            indices = {k: saved[k] for k in saved.files}
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            prediction = {k: saved[k] for k in saved.files}
        c = {
            "completed_exit_zero": record.get("exit_code") == 0 and (folder / "exit_code.txt").read_text().strip() == "0",
            "metric_artifact_matches_metadata": read(folder / "metrics.json") == record["metrics"],
            "configuration_matches_metadata": read(folder / "configuration.json") == record["configuration"],
            "upstream_commit": record["upstream_commit"] == COMMIT,
            "data_hashes": record["data_sha256"] == hashes,
            "no_runtime_warnings": not record.get("warnings") and not observations["warnings"] and not observations["invalid_reasons"],
        }
        plan = record["configuration"]["project_level"]
        effective = record["configuration"]["effective_upstream"]["method"]
        c["main_configuration"] = (
            plan["num_groups"] == effective["num_groups"] == 2
            and plan["num_partitions"] == effective["num_partitions"] == 20
            and plan["group_optimizer_steps"] == effective["optimizer"]["steps"] == 100
            and plan["weight_decay"] == effective["w_net"]["weight_decay"] == 0.1
            and plan["device"] == "cuda:0"
            and record["configuration"]["ece_bins"] == 15
            and record["configuration"]["probability_nll_epsilon"] == 1e-10
            and record["configuration"]["brier_class_reduction"] == "sum")
        c["source_integrity"] = all(
            digest(root / ("code/" + rel if rel.startswith("upstream/") else rel)) == expected
            and (rel not in record["local_source_snapshots"]
                 or digest(folder / record["local_source_snapshots"][rel]) == expected)
            for rel, expected in record["code_sha256"].items())
        shuffled = list(range(10000))
        np.random.RandomState(record["seed"]).shuffle(shuffled)
        c["split_indices"] = (
            np.array_equal(indices["D_cal"], shuffled[:1000])
            and np.array_equal(indices["D_eval"], shuffled[1000:])
            and np.array_equal(indices["D_val"], np.arange(5000)))
        h = hashlib.sha256()
        for name in ("D_cal", "D_eval"):
            a = indices[name].astype("<i8")
            h.update(name.encode() + str(len(a)).encode() + b"\0" + a.tobytes())
        c["split_hash"] = h.hexdigest() == record["split_sha256"]
        raw_logits = caches["test"]["logits"][indices["D_eval"]]
        labels = torch.from_numpy(prediction["labels"])
        prob = torch.from_numpy(prediction["probabilities"])
        original = raw_logits.argmax(1)
        c["prediction_shape"] = list(prob.shape) == [9000, 10] and list(labels.shape) == [9000]
        c["probabilities_valid"] = (
            bool(torch.isfinite(prob).all()) and bool(((prob >= 0) & (prob <= 1)).all())
            and bool(torch.allclose(prob.sum(1), torch.ones(9000), atol=1e-5, rtol=1e-5)))
        c["labels_and_original_predictions"] = (
            torch.equal(labels, caches["test"]["labels"][indices["D_eval"]])
            and np.array_equal(prediction["original_predictions"], original.numpy())
            and np.array_equal(prediction["evaluation_indices"], indices["D_eval"]))
        c["predicted_classes"] = np.array_equal(prediction["predictions"], prob.argmax(1).numpy())
        preflight = record["preflight"]
        cuda = preflight["cuda"]
        c["recorded_gpu_runtime"] = (
            not preflight["blockers"] and cuda["available"] and cuda["device_count"] >= 1
            and cuda["device"] == record["device"] == "cuda:0"
            and cuda["allocation_and_backward"] == "passed")
        c["positive_finite_temperatures"] = all(
            t["finite"] and t["positive"] and not t["fallback"]
            and all(math.isfinite(v) and v > 0 for v in t["values"])
            for t in observations["temperatures"])
        c["finite_optimizer_evidence"] = all(
            s["status"] == "returned" and s["closure_calls"] > 0
            and s["finite_losses"] and s["finite_gradients"] and s["finite_parameters_after"]
            and not s["missing_gradient"] and s.get("finite_final_ts_gradients", True)
            for s in observations["optimizer_steps"])
        if record["method"].startswith("gc_ts"):
            steps = [s for s in observations["optimizer_steps"] if s["kind"] == "group_training"]
            temps = [s for s in observations["optimizer_steps"] if s["kind"] == "group_temperature"]
            c["group_fitting_updates"] = len(steps) == 20 and len(temps) == 40 and all(
                s["optimizer_has_all_model_parameters"] and s["model_parameters_changed"]
                and s["model_devices"] == ["cuda:0"]
                for s in steps)
            c["initialized_parameters"] = (
                initialized[record["seed"]] == record["initialization_hashes"]
                == observations["initialization_hashes"])
            counts = observations["group_counts"]
            c["group_coverage"] = len(counts) == 40 and all(g["calibration"] > 0 for g in counts) and all(
                sum(g[split] for g in counts if g["partition"] == partition) == size
                for partition in range(20) for split, size in (("calibration", 1000), ("evaluation", 9000)))
        if record["method"] == "gc_ts_shuffle":
            with np.load(folder / "permutations.npz", allow_pickle=False) as saved:
                permutations = {k: saved[k] for k in saved.files}
            children = np.random.SeedSequence(record["seed"]).spawn(3)
            c["independent_feature_permutations"] = set(permutations) == {"D_val", "D_cal", "D_eval"} and all(
                np.array_equal(permutations[name], np.random.default_rng(child).permutation(size))
                for name, size, child in zip(("D_val", "D_cal", "D_eval"), (5000, 1000, 9000), children))
        logits = None
        if record["method"] == "uncalibrated":
            logits = raw_logits
        elif record["method"] == "ts":
            logits = raw_logits / observations["temperatures"][0]["values"][0]
        if logits is not None:
            reconstructed = logits.softmax(1)
            # Float32 softmax kernels differ slightly across CPU architectures.
            c["direct_probability_reconstruction"] = bool(torch.allclose(prob, reconstructed, atol=1e-7, rtol=1e-6))
        one_hot = F.one_hot(labels, 10)
        metrics = {
            "ece": float(calibration_error(prob, labels, task="multiclass", num_classes=10, n_bins=15)),
            "nll": float(F.cross_entropy(logits, labels)) if logits is not None else float(-(torch.log(prob + 1e-10) * one_hot).sum(1).mean()),
            "accuracy": float((prob.argmax(1) == labels).float().mean()),
            "brier": float((prob - one_hot).square().sum(1).mean()),
            "prediction_changes": int((prob.argmax(1) != original).sum()),
        }
        differences = {name: abs(metrics[name] - record["metrics"][name]) for name in METRICS}
        c["saved_prediction_metrics"] = all(d <= 1e-7 for d in differences.values())
        c["run_files_unchanged"] = all(digest(folder / name) == expected for name, expected in before.items())
        report["runs"].append({"run_id": record["run_id"], "checks": c,
                               "recomputed_metrics": metrics, "absolute_differences": differences})
    for filename, expected in (("per_seed.csv", rows), ("summary.csv", stats),
                               ("paired_differences.csv", paired), ("paired_summary.csv", paired_stats)):
        checks["summary_" + filename] = compare_csv(summary_folder / filename, expected)
    all_checks = list(checks.values()) + [v for row in report["runs"] for v in row["checks"].values()]
    report.update(status="passed" if all(all_checks) else "failed",
                  passed_checks=sum(bool(v) for v in all_checks), total_checks=len(all_checks))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT, help="Artifact root; defaults to this repository")
    parser.add_argument("--output", type=Path, help="New audit JSON; default is a timestamped file under the summary directory")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.output or args.summary / ("validation_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    if destination.exists():
        parser.error("Refusing to overwrite an existing validation report")
    try:
        paths = load_manifest_paths(args.manifest, root)
        report = audit(root, paths, args.summary)
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "report": str(destination),
                      "passed_checks": report.get("passed_checks"), "total_checks": report.get("total_checks"),
                      "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
