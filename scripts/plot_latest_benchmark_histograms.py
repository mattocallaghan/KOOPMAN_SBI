#!/usr/bin/env python
"""Plot method metric means and standard deviations from newest benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METRICS = [
    "c2st",
    "mmd",
    "posterior_mean_error",
    "posterior_variance_ratio",
    "sampling_time_per_generated_sample_ms",
]


def _latest_comparison(task_dir: Path) -> Path | None:
    candidates = list(task_dir.glob("benchmark_suite/*/benchmark/comparison_summary.csv"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_methods(benchmark_dir: Path, rows: list[dict[str, str]]) -> list[str]:
    manifest_path = benchmark_dir / "benchmark_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        return [str(item["label"]) for item in manifest]
    suffixes = ("_c2st", "_mmd", "_posterior_mean_error")
    methods = []
    for key in rows[0] if rows else {}:
        if key.endswith(suffixes):
            methods.append(key.rsplit("_", 1)[0])
    return sorted(set(methods))


def _display_name(name: str) -> str:
    return name.replace("_", " ")


def _plot_task(task: str, comparison_path: Path, output_dir: Path, metrics: list[str]) -> list[dict[str, Any]]:
    rows = _read_rows(comparison_path)
    methods = _read_methods(comparison_path.parent, rows)
    if not rows or not methods:
        return []

    values: dict[str, dict[str, list[float]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        values[method] = {}
        for metric in metrics:
            key = f"{method}_{metric}"
            metric_values = [float(row[key]) for row in rows if row.get(key, "") not in ("", None)]
            if not metric_values:
                continue
            values[method][metric] = metric_values
            summary_rows.append(
                {
                    "task": task,
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(metric_values)),
                    "std": float(np.std(metric_values)),
                    "num_observations": len(metric_values),
                    "source": str(comparison_path),
                }
            )

    available_metrics = [metric for metric in metrics if any(metric in values[method] for method in methods)]
    if not available_metrics:
        return summary_rows
    ncols = 2
    nrows = int(np.ceil(len(available_metrics) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(methods), 1)))
    ideal_values = {
        "c2st": 0.5,
        "posterior_mean_error": 0.0,
        "posterior_variance_ratio": 1.0,
        "mmd": 0.0,
    }
    for index, metric in enumerate(available_metrics):
        axis = axes[index // ncols][index % ncols]
        plotted_methods = [method for method in methods if values[method].get(metric)]
        means = [float(np.mean(values[method][metric])) for method in plotted_methods]
        stds = [float(np.std(values[method][metric])) for method in plotted_methods]
        axis.bar(
            np.arange(len(plotted_methods)),
            means,
            yerr=stds,
            capsize=4,
            color=colors[: len(plotted_methods)],
            alpha=0.8,
        )
        axis.set_xticks(np.arange(len(plotted_methods)))
        axis.set_xticklabels([_display_name(method) for method in plotted_methods], rotation=35, ha="right")
        if metric in ideal_values:
            axis.axhline(ideal_values[metric], color="black", linestyle="--", linewidth=1, label="ideal")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("method")
        axis.set_ylabel("mean ± std across observations")
        axis.grid(alpha=0.25)
        if metric in ideal_values:
            axis.legend(fontsize=8)
    for index in range(len(available_metrics), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    figure.suptitle(f"{task}: latest benchmark method means ± standard deviations", fontsize=15)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{task}_method_metric_mean_std.png", dpi=180)
    plt.close(figure)
    return summary_rows


def _write_wide_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["method"]))
        grouped.setdefault(key, {"task": key[0], "method": key[1]})
        grouped[key][row["metric"]] = row["mean"]
    columns = [
        "task",
        "method",
        "c2st",
        "mmd",
        "posterior_mean_error",
        "posterior_variance_ratio",
        "sampling_time_per_generated_sample_ms",
    ]
    wide_rows = [grouped[key] for key in sorted(grouped)]
    with (output_dir / "benchmark_summary_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(wide_rows)
    with (output_dir / "benchmark_summary_table.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in wide_rows:
            cells = []
            for column in columns:
                value = row.get(column, "")
                cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(cells) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot histograms from the latest benchmark run for each task.")
    parser.add_argument("--logs-root", default="logs", help="Benchmark logs directory.")
    parser.add_argument("--output-dir", default="logs/latest_benchmark_histograms")
    parser.add_argument("--tasks", nargs="*", help="Optional task names; defaults to every task with benchmark data.")
    args = parser.parse_args()
    logs_root = Path(args.logs_root)
    if not logs_root.is_absolute():
        logs_root = ROOT / logs_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    task_names = args.tasks or sorted(path.name for path in logs_root.iterdir() if path.is_dir())
    all_summary_rows: list[dict[str, Any]] = []
    found = 0
    for task in task_names:
        comparison_path = _latest_comparison(logs_root / task)
        if comparison_path is None:
            continue
        found += 1
        print(f"{task}: {comparison_path}")
        all_summary_rows.extend(_plot_task(task, comparison_path, output_dir, DEFAULT_METRICS))
    if not found:
        raise FileNotFoundError(f"No benchmark comparison files found under {logs_root}")
    summary_path = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["task", "method", "metric", "mean", "std", "num_observations", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summary_rows)
    _write_wide_summary(output_dir, all_summary_rows)
    print(f"Wrote {found} task plots to {output_dir}")


if __name__ == "__main__":
    main()
