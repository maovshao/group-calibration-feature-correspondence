#!/usr/bin/env python3
"""Draw paired ECE measurements from a per-seed CSV; no fitting is performed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = {
    "title_left": "Global vs. group-wise calibration",
    "title_right": "Original vs. shuffled features",
    "xlabel": "ECE (lower is better)",
    "seed_label": "Seed",
    "mean_label": "Mean",
    "method_ts": "Global TS",
    "method_gc": "Original GC+TS",
    "method_shuffle": "Shuffled GC+TS",
}
METHODS = ("ts", "gc_ts", "gc_ts_shuffle")


def load_measurements(path: Path):
    """Require all five paired measurements and reject ambiguous duplicates."""
    import math

    records = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            method = row["method"]
            if method not in METHODS:
                continue
            seed, value = int(row["seed"]), float(row["ece"])
            if seed not in range(5) or not math.isfinite(value) or value < 0:
                raise ValueError("Expected finite, nonnegative ECE for seeds 0--4")
            if (method, seed) in records:
                raise ValueError(f"Duplicate method/seed: {method}/{seed}")
            records[method, seed] = value
    expected = {(method, seed) for method in METHODS for seed in range(5)}
    if set(records) != expected:
        raise ValueError(f"Missing paired measurements: {sorted(expected - records.keys())}")
    return {method: [records[method, seed] for seed in range(5)] for method in METHODS}


def draw(measurements, output: Path, labels, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    from matplotlib import font_manager

    families = ["DejaVu Sans"]
    if any(any(ord(char) > 255 for char in value) for value in labels.values()):
        installed = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in ("Arial Unicode MS", "Noto Sans CJK SC", "PingFang SC", "Heiti TC"):
            if candidate in installed:
                families.insert(0, candidate)
                break
    plt.rcParams.update({
        "font.family": families, "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=True)
    ink, blue, orange = "#243748", "#136B93", "#BB6728"
    all_values = [value for values in measurements.values() for value in values]
    span = max(all_values) - min(all_values)
    limits = (max(0, min(all_values) - 0.10 * span), max(all_values) + 0.10 * span)
    y_positions = [5, 4, 3, 2, 1, -0.25]
    row_labels = [f"{labels['seed_label']} {seed}" for seed in range(5)] + [labels["mean_label"]]
    comparisons = [("ts", "title_left", "method_ts"),
                   ("gc_ts_shuffle", "title_right", "method_shuffle")]
    for ax, (other, title_key, method_key) in zip(axes, comparisons):
        baseline = measurements[other]
        original = measurements["gc_ts"]
        other_values = baseline + [sum(baseline) / 5]
        gc_values = original + [sum(original) / 5]
        ax.axhspan(-0.65, 0.15, color="#F0F4F6", zorder=0)
        for y, left, right in zip(y_positions, other_values, gc_values):
            ax.plot([left, right], [y, y], color="#97A5AF", linewidth=1.7, zorder=2)
        ax.scatter(other_values[:5], y_positions[:5], color=orange, marker="s",
                   s=42, edgecolor="white", linewidth=0.65, zorder=4)
        ax.scatter(gc_values[:5], y_positions[:5], color=blue, marker="o",
                   s=50, edgecolor="white", linewidth=0.65, zorder=5)
        ax.scatter([other_values[-1]], [y_positions[-1]], color=orange, marker="s",
                   s=58, edgecolor="white", linewidth=0.65, zorder=4)
        ax.scatter([gc_values[-1]], [y_positions[-1]], color=blue, marker="o",
                   s=66, edgecolor="white", linewidth=0.65, zorder=5)
        ax.set_title(labels[title_key], loc="left", fontsize=13, weight="bold", color=ink, pad=48)
        handles = [Line2D([], [], linestyle="", marker="o", color=blue, markersize=7,
                          label=labels["method_gc"]),
                   Line2D([], [], linestyle="", marker="s", color=orange, markersize=6,
                          label=labels[method_key])]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.015),
                  frameon=False, fontsize=10.5, handletextpad=0.5, labelspacing=0.35,
                  borderaxespad=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(row_labels, fontsize=11, color=ink)
        ax.tick_params(axis="y", length=0, pad=9, labelleft=True)
        ax.tick_params(axis="x", labelsize=11, colors=ink, length=3)
        ax.set_xlim(*limits)
        ax.set_ylim(-0.8, 5.5)
        ax.xaxis.set_major_locator(MultipleLocator(0.002))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(axis="x", color="#DDE3E8", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_color("#9AA7B1")
        ax.set_xlabel(labels["xlabel"], fontsize=11, labelpad=8, color=ink)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.75, bottom=0.18, wspace=0.28)
    output.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output / f"paired_ece.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Per-seed CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--labels", type=Path, help="Optional JSON label overrides")
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"),
                        default=("png", "svg", "pdf"))
    args = parser.parse_args()
    labels = dict(LABELS)
    if args.labels:
        overrides = json.loads(args.labels.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or set(overrides) - set(labels):
            parser.error("Label overrides must be an object containing known label keys")
        if not all(isinstance(value, str) for value in overrides.values()):
            parser.error("Label values must be strings")
        labels.update(overrides)
    draw(load_measurements(args.input), args.output, labels, args.formats)
    print(f"Wrote paired ECE figures to {args.output}")


if __name__ == "__main__":
    main()
