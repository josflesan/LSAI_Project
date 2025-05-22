import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import parse_all
from typing import List


def plot_bar(df: pd.DataFrame):
    print("PLOT BAR")
    pass


def plot_line(df: pd.DataFrame, runs: List[str]) -> None:
    # Set up plot style
    plt.figure(figsize=(10, 6), dpi=150)
    sns.set_style("darkgrid")

    run_order = sorted(
        runs, key=lambda run: df[df["type"] == run]["loss"].mean(), reverse=True
    )

    # Create main plots
    for idx, run in enumerate(run_order):
        local_df = df[df["type"] == run]

        # Fill area to highlight differences if plotting more than one run
        if args.type == "area":
            plt.fill_between(
                local_df["step"].astype(dtype=int),
                local_df["loss"].astype(dtype=float),
                alpha=0.8,
                color=colours[idx],
                zorder=idx,
                label=f"{run}",
            )

        else:
            plt.plot(
                local_df["step"],
                local_df["loss"],
                marker="o",
                linestyle="-",
                color=colours[idx],
                linewidth=1.4,
                markersize=5,
                label=f"{run}",
            )

    # Configure titles and axes
    plt.title(
        f"{args.title} [{args.nodes} Nodes | {args.gpus} GPUs | {args.seqlen} SeqLen | {args.batch_size} Batch Size]",
        loc="left",
        fontweight="bold",
        fontsize=12,
        family="Arial",
        pad=22,
    )
    plt.ylabel("Training Loss", loc="top", rotation=0, fontsize=10, family="Arial")
    plt.xlabel("Step", fontsize=10, family="Arial")
    plt.gca().yaxis.set_label_coords(0.090, 1.02)
    plt.xlim(0, args.end)

    if len(runs) > 1:
        plt.ylim(0, None)  # Set ylim for the area chart
    plt.grid(axis="x")

    # Add legend
    plt.legend()

    # Output
    plt.tight_layout()
    plt.savefig(f"{args.out}_{args.type}.png", format="png", dpi=300)


def main(filenames: List[str], runs: List[str], plot_type: str):
    # Parse the files
    df = parse_all(filenames, runs)

    if plot_type == "line" or plot_type == "area":
        plot_line(df, runs)
    elif plot_type == "bar":
        plot_bar(df)
    else:
        raise NotImplementedError(
            "The plot type requested is not supported, options are: [line, bar]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plotting Function for Tensor and Data Parallel Runs"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        type=str,
        help="File path to the training logs",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=str,
        help="Names of the runs for each of the log files passed",
    )
    parser.add_argument(
        "--type", type=str, choices=["line", "area", "bar"], help="Type of plot to run"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output path for the plot"
    )
    parser.add_argument("--title", type=str, help="Title for the plot")

    # Run metadata
    parser.add_argument(
        "--end", type=int, default=1000, help="Final step size for the run"
    )
    parser.add_argument(
        "--gpus", type=int, default=4, help="Number of GPUs used to run the experiment"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes used to run the experiment",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=4096,
        help="Sequence Length with which the model was trained",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch Size of the Run"
    )

    args = parser.parse_args()

    # Define colour palette
    colours = sns.color_palette("husl", len(args.runs))

    main(args.files, args.runs, args.type)
