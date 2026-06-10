#!/usr/bin/env python3
"""Redraw the SST-2 ADP/accuracy curve in a paper-ready style."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/ram/tmp/matplotlib-cache")


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_weight_distance_results"
    / "bert"
    / "sst2"
    / "4_weight_distance_summary.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_weight_distance_results"
    / "bert"
    / "sst2"
    / "4_line_scale_vs_accuracy"
)


def load_points(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    points = [
        (float(row["line_scale"]), float(row["mean_metric"]))
        for row in summary.get("line", [])
    ]
    if not points:
        raise ValueError(f"No line data found in {path}")
    return sorted(points)


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(points: list[tuple[float, float]], output_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    configure_matplotlib()

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=(3.55, 2.35), constrained_layout=True)
    ax.set_facecolor("white")

    ax.plot(
        xs,
        ys,
        color="#2563EB",
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.45,
        zorder=3,
    )

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(max(0.0, min(ys) - 0.025), min(1.0, max(ys) + 0.02))
    ax.set_xlabel("ADP")
    ax.set_ylabel("model-stealing accuracy")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, linestyle=(0, (2.2, 2.2)))
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45, linestyle=(0, (2.2, 2.2)))
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(axis="both", which="major", length=3.0, color="#222222")

    written = []
    for suffix in (".png", ".pdf", ".svg"):
        path = output_stem.with_suffix(suffix)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
        if suffix == ".png":
            save_kwargs["dpi"] = 400
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = plot(load_points(args.input), args.output)
    for path in written:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
