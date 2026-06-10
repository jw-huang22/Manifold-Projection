#!/usr/bin/env python3
"""Plot ADP vs model-stealing accuracy from weight-distance summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/ram/tmp/matplotlib-cache")


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "evaluate_weight_distance_results"
MODEL_ALIASES = {
    "bert": "bert",
    "bert-base-cased": "bert",
    "gpt2": "gpt2_base",
    "gpt2-base": "gpt2_base",
    "gpt2_base": "gpt2_base",
    "vit": "ViT",
    "vit-base": "ViT",
    "ViT": "ViT",
}
MODEL_LABELS = {
    "bert": "BERT-Base",
    "gpt2_base": "GPT-2 Base",
    "ViT": "ViT-Base",
}
DEFAULT_DATASETS = {
    "bert": ("mnli", "qnli", "qqp", "sst2"),
    "gpt2_base": ("mnli", "qnli", "qqp", "sst2"),
    "ViT": ("cifar_10", "cifar_100", "food101"),
}
DATASET_LABELS = {
    "mnli": "MNLI",
    "qnli": "QNLI",
    "qqp": "QQP",
    "sst2": "SST-2",
    "cifar_10": "CIFAR-10",
    "cifar_100": "CIFAR-100",
    "food101": "Food-101",
}


def normalize_model(raw_model: str) -> str:
    try:
        return MODEL_ALIASES[raw_model]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unsupported model {raw_model!r}. Use one of: {supported}") from exc


def load_points(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    points = [
        (float(row["constructed_adp"]), float(row["mean_metric"]))
        for row in summary.get("adp_curve", {}).get("points", [])
    ]
    if not points:
        raise ValueError(f"No adp_curve.points data found in {path}")
    return sorted(points)


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(
    series: dict[str, list[tuple[float, float]]],
    output_stem: Path,
    model: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    configure_matplotlib()

    colors = {
        "mnli": "#0072B2",
        "qnli": "#D55E00",
        "qqp": "#009E73",
        "sst2": "#CC79A7",
        "cifar_10": "#0072B2",
        "cifar_100": "#D55E00",
        "food101": "#009E73",
    }
    markers = {
        "mnli": "o",
        "qnli": "s",
        "qqp": "^",
        "sst2": "D",
        "cifar_10": "h",
        "cifar_100": "P",
        "food101": "X",
    }
    linestyles = {
        "mnli": "-",
        "qnli": "--",
        "qqp": "-.",
        "sst2": ":",
        "cifar_10": "-",
        "cifar_100": "--",
        "food101": "-.",
    }

    is_paper_bert = model == "bert"
    figsize = (3.35, 2.18) if is_paper_bert else (1.55, 1.18)
    line_width = 1.55 if is_paper_bert else 1.25
    marker_size = 3.25 if is_paper_bert else 2.25
    marker_edge_width = 0.75 if is_paper_bert else 0.65
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_facecolor("white")

    all_xs = []
    all_ys = []
    for dataset, points in series.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        all_xs.extend(xs)
        all_ys.extend(ys)
        (line,) = ax.plot(
            xs,
            ys,
            color=colors.get(dataset, "#111827"),
            marker=markers.get(dataset, "o"),
            linestyle=linestyles.get(dataset, "-"),
            markersize=marker_size,
            markerfacecolor="white",
            markeredgewidth=marker_edge_width,
            markevery=2,
            linewidth=line_width,
            label=DATASET_LABELS.get(dataset, dataset.upper()),
            zorder=3,
        )
        dash_patterns = {
            "qnli": (3.2, 1.6),
            "qqp": (5.0, 1.6, 1.1, 1.6),
            "sst2": (1.1, 1.6),
            "cifar_100": (3.2, 1.6),
            "food101": (5.0, 1.6, 1.1, 1.6),
        }
        if dataset in dash_patterns:
            line.set_dashes(dash_patterns[dataset])
            line.set_dash_capstyle("round")

    x_padding = (0.015 if is_paper_bert else 0.02) * (max(all_xs) - min(all_xs))
    ax.set_xlim(min(all_xs) - x_padding, max(all_xs) + x_padding)
    ax.set_ylim(max(0.0, min(all_ys) - 0.035), min(1.0, max(all_ys) + 0.025))
    ax.set_xlabel("ADP")
    ax.set_ylabel("Accuracy" if is_paper_bert else "Acc.")
    if model != "bert":
        ax.set_title(MODEL_LABELS.get(model, model), pad=2.0)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    if is_paper_bert:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    grid_y_width = 0.42 if is_paper_bert else 0.35
    grid_x_width = 0.34 if is_paper_bert else 0.3
    ax.grid(axis="y", color="#D9D9D9", linewidth=grid_y_width, linestyle=(0, (1.4, 2.4)), alpha=0.55)
    ax.grid(axis="x", color="#ECECEC", linewidth=grid_x_width, linestyle=(0, (1.4, 2.6)), alpha=0.45)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.tick_params(
        axis="both",
        which="major",
        length=3.0 if is_paper_bert else 2.4,
        color="#303030",
        pad=2.0 if is_paper_bert else 1.5,
    )
    ax.legend(
        frameon=is_paper_bert,
        fancybox=False,
        framealpha=0.92,
        edgecolor="#CFCFCF",
        facecolor="white",
        loc="upper right" if is_paper_bert else "lower left",
        ncol=2 if len(series) > 3 else 1,
        handlelength=1.45 if is_paper_bert else 1.25,
        handletextpad=0.35 if is_paper_bert else 0.25,
        columnspacing=0.70 if is_paper_bert else 0.45,
        borderaxespad=0.32 if is_paper_bert else 0.12,
        borderpad=0.25 if is_paper_bert else 0.4,
        labelspacing=0.22 if is_paper_bert else 0.15,
    )

    written = []
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        path = output_stem.with_suffix(suffix)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.025 if is_paper_bert else 0.015}
        if suffix == ".png":
            save_kwargs["dpi"] = 400
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def build_series(results_dir: Path, datasets: list[str]) -> dict[str, list[tuple[float, float]]]:
    return {
        dataset: load_points(results_dir / dataset / "weight_distance_summary.json")
        for dataset in datasets
    }


def draw_series_on_axis(
    ax,
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    show_ylabel: bool,
    show_legend: bool,
    compact: bool,
):
    import matplotlib.ticker as mticker

    colors = {
        "mnli": "#0072B2",
        "qnli": "#D55E00",
        "qqp": "#009E73",
        "sst2": "#CC79A7",
        "cifar_10": "#0072B2",
        "cifar_100": "#D55E00",
        "food101": "#009E73",
    }
    markers = {
        "mnli": "o",
        "qnli": "s",
        "qqp": "^",
        "sst2": "D",
        "cifar_10": "h",
        "cifar_100": "P",
        "food101": "X",
    }
    linestyles = {
        "mnli": "-",
        "qnli": "--",
        "qqp": "-.",
        "sst2": ":",
        "cifar_10": "-",
        "cifar_100": "--",
        "food101": "-.",
    }

    all_xs = []
    all_ys = []
    handles = []
    labels = []
    for dataset, points in series.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        all_xs.extend(xs)
        all_ys.extend(ys)
        (handle,) = ax.plot(
            xs,
            ys,
            color=colors.get(dataset, "#111827"),
            marker=markers.get(dataset, "o"),
            linestyle=linestyles.get(dataset, "-"),
            markersize=2.15 if compact else 2.45,
            markerfacecolor="white",
            markeredgewidth=0.62,
            markevery=3 if compact else 2,
            linewidth=1.05 if compact else 1.15,
            label=DATASET_LABELS.get(dataset, dataset.upper()),
            zorder=3,
        )
        dash_patterns = {
            "qnli": (3.2, 1.6),
            "qqp": (5.0, 1.6, 1.1, 1.6),
            "sst2": (1.1, 1.6),
            "cifar_100": (3.2, 1.6),
            "food101": (5.0, 1.6, 1.1, 1.6),
        }
        if dataset in dash_patterns:
            handle.set_dashes(dash_patterns[dataset])
            handle.set_dash_capstyle("round")
        handles.append(handle)
        labels.append(DATASET_LABELS.get(dataset, dataset.upper()))

    x_padding = 0.018 * (max(all_xs) - min(all_xs))
    y_min = max(0.0, min(all_ys) - 0.04)
    y_max = min(1.0, max(all_ys) + 0.035)
    ax.set_xlim(min(all_xs) - x_padding, max(all_xs) + x_padding)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, pad=1.8, fontsize=6.8, fontweight="semibold")
    ax.set_xlabel("ADP", labelpad=1.0)
    ax.set_ylabel("Acc." if show_ylabel else "", labelpad=1.0)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.32, linestyle=(0, (1.2, 2.2)), alpha=0.55)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.28, linestyle=(0, (1.2, 2.4)), alpha=0.45)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.tick_params(axis="both", which="major", length=2.2, color="#303030", pad=1.3)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)

    legend = None
    if show_legend:
        legend = ax.legend(
            frameon=False,
            fancybox=False,
            loc="lower left",
            ncol=1,
            handlelength=1.12,
            handletextpad=0.24,
            borderaxespad=0.18,
            labelspacing=0.08,
            fontsize=5.0,
        )
    return handles, labels, legend


def plot_paper_combined(results_root: Path, output_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.6,
        }
    )

    model_series = {
        model: build_series(results_root / model, list(DEFAULT_DATASETS[model]))
        for model in ("bert", "gpt2_base", "ViT")
    }

    fig = plt.figure(figsize=(3.35, 2.52))
    gs = fig.add_gridspec(
        2,
        4,
        left=0.115,
        right=0.985,
        bottom=0.125,
        top=0.895,
        wspace=0.64,
        hspace=0.44,
    )
    ax_bert = fig.add_subplot(gs[0, 0:2])
    ax_gpt = fig.add_subplot(gs[0, 2:4])
    ax_vit = fig.add_subplot(gs[1, 1:3])

    glue_handles, glue_labels, _ = draw_series_on_axis(
        ax_bert,
        model_series["bert"],
        title=MODEL_LABELS["bert"],
        show_ylabel=True,
        show_legend=False,
        compact=True,
    )
    draw_series_on_axis(
        ax_gpt,
        model_series["gpt2_base"],
        title=MODEL_LABELS["gpt2_base"],
        show_ylabel=False,
        show_legend=False,
        compact=True,
    )
    vit_handles, vit_labels, _ = draw_series_on_axis(
        ax_vit,
        model_series["ViT"],
        title=MODEL_LABELS["ViT"],
        show_ylabel=True,
        show_legend=False,
        compact=True,
    )
    vit_pos = ax_vit.get_position()
    ax_vit.set_position(
        [
            vit_pos.x0,
            vit_pos.y0 - 0.035,
            vit_pos.width,
            vit_pos.height,
        ]
    )

    fig.legend(
        glue_handles,
        glue_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.992),
        ncol=4,
        handlelength=1.35,
        handletextpad=0.28,
        columnspacing=0.72,
        borderaxespad=0.0,
        fontsize=5.6,
    )
    ax_vit.legend(
        vit_handles,
        vit_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.14),
        ncol=3,
        handlelength=1.18,
        handletextpad=0.22,
        columnspacing=0.48,
        borderaxespad=0.0,
        labelspacing=0.05,
        fontsize=5.2,
    )
    written = []
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        path = output_stem.with_suffix(suffix)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.025}
        if suffix == ".png":
            save_kwargs["dpi"] = 450
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all",
        help="model to plot: bert, gpt2-base, vit-base, or all",
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="override datasets for a single --model run",
    )
    parser.add_argument(
        "--paper-combined",
        action="store_true",
        help="plot BERT/GPT-2 on the first row and centered ViT on the second row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paper_combined:
        output = args.output_root / "adp_vs_model_stealing_accuracy_three_models_single_column"
        for path in plot_paper_combined(args.results_root, output):
            print(f"Saved: {path}")
        return

    models = list(DEFAULT_DATASETS) if args.model == "all" else [normalize_model(args.model)]
    if args.datasets is not None and len(models) != 1:
        raise ValueError("--datasets can only be used with a single --model")

    for model in models:
        datasets = args.datasets if args.datasets is not None else list(DEFAULT_DATASETS[model])
        series = build_series(args.results_root / model, datasets)
        dataset_tag = "datasets" if len(datasets) != 4 else "four_datasets"
        output = args.output_root / model / f"adp_vs_model_stealing_accuracy_{dataset_tag}"
        for path in plot(series, output, model):
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
