#!/usr/bin/env python3
"""Summarize saved, completed main runs only; no invented or imputed results."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("uncalibrated", "ts", "gc_ts", "gc_ts_shuffle")
METRICS = ("ece", "nll", "accuracy", "brier", "prediction_changes")


def nonempty_provenance(value):
    """Accept stored hash strings or nonempty collections, never empty evidence."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value) and all(isinstance(k, str) and k.strip() and nonempty_provenance(v)
                                   for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(nonempty_provenance(v) for v in value)
    return False


def nonempty_versions(value):
    """Recognize version strings or preflight's version/import-status records."""
    if not isinstance(value, dict) or not value:
        return False
    for package, record in value.items():
        version = record.get("version") if isinstance(record, dict) else record
        if not isinstance(package, str) or not package.strip() or not isinstance(version, str) or not version.strip():
            return False
    return True


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs):
    selected = {}
    for row in runs:
        if row["status"] != "completed" or row["level"] != "main":
            continue
        key = (row["method"], row["seed"])
        if key[0] not in METHODS or type(key[1]) is not int or key[1] not in range(5):
            raise ValueError("Unexpected method/seed in completed main run: " + str(key))
        if key in selected:
            raise ValueError("Multiple completed attempts for %r; use --manifest to select explicitly" % (key,))
        for name in METRICS:
            metrics = row.get("metrics")
            value = metrics.get(name) if isinstance(metrics, dict) else None
            if type(value) not in (float, int) or not (-float("inf") < value < float("inf")):
                raise ValueError("Nonfinite or missing completed metric: " + name)
        selected[key] = row
    records = list(selected.values())
    for name in ("upstream_commit", "data_sha256", "code_sha256"):
        # Mixing code/data implementations silently would invalidate paired comparisons.
        variants = {json.dumps(r.get(name), sort_keys=True) for r in records}
        if len(variants) > 1 or any(not nonempty_provenance(r.get(name)) for r in records):
            raise ValueError("Inconsistent or missing provenance: " + name)
    # Legacy records may all omit versions. Once recorded, every selected run
    # must provide the same dependency evidence, including import status.
    if any("versions" in r for r in records):
        variants = {json.dumps(r.get("versions"), sort_keys=True) for r in records}
        if len(variants) > 1 or any(not nonempty_versions(r.get("versions")) for r in records):
            raise ValueError("Inconsistent or missing provenance: versions")
    for seed in range(5):
        group = [r for r in records if r["seed"] == seed]
        splits = [r.get("split_sha256") for r in group]
        if any(not isinstance(s, str) or not s.strip() for s in splits) or len(set(splits)) > 1:
            raise ValueError("Cannot establish identical splits for seed " + str(seed))
        if ("gc_ts", seed) in selected and ("gc_ts_shuffle", seed) in selected:
            a, b = selected["gc_ts", seed], selected["gc_ts_shuffle", seed]
            if (not nonempty_provenance(a.get("initialization_hashes"))
                    or a["initialization_hashes"] != b.get("initialization_hashes")):
                raise ValueError("GC initializations not demonstrably paired for seed " + str(seed))
    rows, stats, paired = [], [], []
    for method in METHODS:
        group = sorted((r for r in records if r["method"] == method), key=lambda r: r["seed"])
        for row in group:
            rows.append({"method": method, "seed": row["seed"], "run_id": row["run_id"], **{m: row["metrics"][m] for m in METRICS}})
        if not group:
            continue
        for metric in METRICS:
            values = [r["metrics"][metric] for r in group]
            stats.append({"method": method, "metric": metric, "n": len(values),
                          "mean": statistics.mean(values),
                          "sample_std_ddof1": statistics.stdev(values) if len(values) > 1 else "NA"})
    comparisons = (("delta_gc", "ts", "gc_ts"),
                   ("delta_feature", "gc_ts_shuffle", "gc_ts"))
    for label, first, second in comparisons:
        for seed in range(5):
            if (first, seed) in selected and (second, seed) in selected:
                a, b = selected[first, seed], selected[second, seed]
                paired.append({"comparison": label, "seed": seed,
                               "first_method": first, "second_method": second,
                               "ece_difference": a["metrics"]["ece"] - b["metrics"]["ece"],
                               "nll_difference": a["metrics"]["nll"] - b["metrics"]["nll"]})
    paired_stats = []
    for label, _, _ in comparisons:
        group = [r for r in paired if r["comparison"] == label]
        for metric in ("ece_difference", "nll_difference"):
            values = [r[metric] for r in group]
            if values:
                paired_stats.append({"comparison": label, "metric": metric, "n": len(values),
                                     "mean": statistics.mean(values),
                                     "sample_std_ddof1": statistics.stdev(values) if len(values) > 1 else "NA"})
    missing = [{"method": method, "seed": seed} for method in METHODS for seed in range(5)
               if (method, seed) not in selected]
    return rows, stats, paired, paired_stats, missing


def load_manifest_paths(path, root=ROOT):
    """Resolve an explicit batch selection; every entry identifies one run."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or any(not isinstance(x, str) or not x.strip() for x in entries):
        raise ValueError("Manifest must be a JSON list of metadata path strings")
    paths = [(Path(root) / entry).resolve() for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest contains duplicate metadata paths")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--manifest", type=Path, help="JSON list of metadata paths relative to the repository root")
    parser.add_argument("--plot", action="store_true", help="Render figures through code/plot.py")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Inspect an incomplete selection; formal summaries require all 20 runs by default")
    args = parser.parse_args()
    try:
        paths = load_manifest_paths(args.manifest) if args.manifest else sorted(args.runs.glob("*/metadata.json"))
        runs = [json.loads(p.read_text()) for p in paths]
        rows, stats, paired, paired_stats, missing = summarize(runs)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    invalid_inputs = [r.get("run_id", "unknown") for r in runs
                      if r.get("status") != "completed" or r.get("level") != "main"]
    incomplete = bool(missing or invalid_inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "no_completed_main_runs" if not rows else ("partial" if incomplete else "complete"),
              "completed_main_method_seed_pairs": len(rows), "expected_main_method_seed_pairs": 20,
              "input_status_counts": {s: sum(r["status"] == s for r in runs) for s in sorted({r["status"] for r in runs})},
              "source_metadata": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
              "missing": missing, "invalid_inputs": invalid_inputs,
              "standard_deviation": "sample standard deviation, ddof=1; n=1 -> NA",
              "interpretation": "Five seeds repeat calibration and splits of one fixed classifier."}
    write_numbers = bool(rows) and (args.allow_partial or not incomplete)
    report["numeric_outputs_written"] = write_numbers
    if not write_numbers:
        report["preexisting_csv_not_modified"] = [p.name for p in args.output.glob("*.csv")]
    else:
        write_csv(args.output / "per_seed.csv", rows, ["method", "seed", "run_id", *METRICS])
        write_csv(args.output / "summary.csv", stats, ["method", "metric", "n", "mean", "sample_std_ddof1"])
        write_csv(args.output / "paired_differences.csv", paired, ["comparison", "seed", "first_method", "second_method", "ece_difference", "nll_difference"])
        write_csv(args.output / "paired_summary.csv", paired_stats, ["comparison", "metric", "n", "mean", "sample_std_ddof1"])
        if args.plot:
            subprocess.run([sys.executable, str(ROOT / "code/plot.py"),
                            "--input", str(args.output / "per_seed.csv"),
                            "--output", str(args.output / "figures")], check=True)
    (args.output / "summary_status.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "completed_pairs": len(rows), "missing_pairs": len(missing)}))
    if incomplete and not args.allow_partial:
        print("Incomplete or invalid main selection; no numeric CSVs were written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
