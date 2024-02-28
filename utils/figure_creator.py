""" Sheet that contains methods to draw certain figures """
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
import os
from constants import NUMBER_OF_TRAJECTORIES

COLOR_GRID_HASH = "#69b3a2"
COLOR_DISK_HASH = "violet"
COLOR_TRUE = "#3399e6"


def draw_hash_similarity_runtime(path: str, path_to_reference: str = "") -> None:
    """
    Method that draws a figure of the runtime of the hash similarity computation:

    ### Params:
    ---
    path : str (abspath)
        The Path to the csv file containing the runtimes
    path_to_reference_values : str (abspath)
        The Path to the csv file containing the reference runtimes

    """

    timing_data = pd.read_csv(path, index_col=0)
    reference_data = (
        pd.read_csv(path_to_reference, index_col=0) if path_to_reference else None
    )

    mean_timing = timing_data.mean(axis=0)
    data_sizes = mean_timing.index.to_numpy(int)
    data_runtimes = mean_timing.values

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    if path_to_reference:
        ax2 = ax.twinx()
        rd = reference_data.mean(axis=0)
        r_sizes = rd.index.to_numpy(int)
        r_runtimes = rd.values
        ax2.plot(r_sizes, r_runtimes, "o", color=COLOR_TRUE, lw=2)

        degree = 2
        print(data_sizes[: len(r_runtimes)])
        coeffs = np.polyfit(data_sizes[: len(r_runtimes)], r_runtimes, degree)
        p = np.poly1d(coeffs)
        ax2.plot(data_sizes, [p(n) for n in data_sizes], color=COLOR_TRUE, lw=2)

        ax2.set_ylabel(
            "True similarity computation time (s)", fontsize=14, color=COLOR_TRUE
        )
        ax2.tick_params(axis="y", labelcolor=COLOR_TRUE)

    ax2.plot(data_sizes, data_runtimes, "xr", lw=2)
    ax.set_xlabel("Dataset size", fontsize=14)
    ax.set_ylabel(
        "Hash similarity computation time (s)", fontsize=14, color=COLOR_GRID_HASH
    )
    ax.tick_params(axis="y", labelcolor=COLOR_GRID_HASH)

    degree = 4
    coeffs = np.polyfit(data_sizes, data_runtimes, degree)
    p = np.poly1d(coeffs)
    ax2.plot(data_sizes, [p(n) for n in data_sizes], color=COLOR_GRID_HASH, lw=2)
    plt.show()


def draw_hash_similarity_runtime_logarithmic(
    city: str, path_grid: str, path_disk: str, path_to_reference: str = ""
) -> None:
    """
    Method that draws a figure of the runtime of the hash similarity computation, logarithmic y-scale:

    ### Params:
    ---
    city : str
        The city
    path_grid : str (abspath)
        The Path to the csv file containing the grid runtimes
    path_disk : str (abspath)
        Path to the csv file containing the disk runtimes
    path_to_reference_values : str (abspath)
        The Path to the csv file containing the reference runtimes

    """

    grid_timing_data = pd.read_csv(path_grid, index_col=0)
    disk_timing_data = pd.read_csv(path_disk, index_col=0)
    reference_data = (
        pd.read_csv(path_to_reference, index_col=0) if path_to_reference else None
    )

    grid_mean_timing = grid_timing_data.mean(axis=0)
    grid_data_sizes = grid_mean_timing.index.to_numpy(int)
    grid_data_runtimes = grid_mean_timing.values

    disk_mean_timing = disk_timing_data.mean(axis=0)
    disk_data_sizes = disk_mean_timing.index.to_numpy(int)
    disk_data_runtimes = disk_mean_timing.values

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    rd = reference_data.mean(axis=0)
    r_sizes = rd.index.to_numpy(int)
    r_runtimes = rd.values
    ax.plot(r_sizes, r_runtimes, "or", markersize=8)

    degree = 2
    print(grid_data_sizes[: len(r_runtimes)])
    coeffs = np.polyfit(grid_data_sizes[: len(r_runtimes)], r_runtimes, degree)
    p = np.poly1d(coeffs)
    ax.plot(
        grid_data_sizes,
        [p(n) for n in grid_data_sizes],
        color=COLOR_TRUE,
        lw=3,
        label="DTW Similarity",
    )
    ax.plot(grid_data_sizes, grid_data_runtimes, "xr", markersize=12)
    ax.plot(disk_data_sizes, disk_data_runtimes, "xr", markersize=12)

    ax.set_xlabel("Dataset size - number of trajectories", fontsize=18)
    ax.set_ylabel("Similarity computation time (s)", fontsize=18)

    ax.set_yscale("log")
    ax.tick_params(axis="y")
    ax.tick_params(axis="both", which="major", labelsize=18)

    degree = 5
    grid_coeffs = np.polyfit(grid_data_sizes, grid_data_runtimes, degree)
    grid_p = np.poly1d(grid_coeffs)
    ax.plot(
        grid_data_sizes,
        [grid_p(n) for n in grid_data_sizes],
        color=COLOR_GRID_HASH,
        lw=3,
        label="Grid Hash Similarity",
    )

    disk_coeffs = np.polyfit(disk_data_sizes, disk_data_runtimes, degree)
    disk_p = np.poly1d(disk_coeffs)
    ax.plot(
        disk_data_sizes,
        [disk_p(n) for n in disk_data_sizes],
        color=COLOR_DISK_HASH,
        lw=3,
        label="Disk Hash Similarity",
    )

    ax.text(
        0.10,
        0.97,
        f"{city}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=18,
        color="grey",
    )

    ax.legend(
        loc="lower right",
        ncols=1,
        fontsize=18,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        handletextpad=0.5,
        borderaxespad=0.2,
        columnspacing=1,
    )
    plt.grid(axis="y", which="major")
    plt.show()


def draw_similarity_correlation(
    hash_sim_path: str, city: str, hash_type: str, reference_measure: str
) -> None:
    """
    Method that draws a similarity correlation graph visualising the correlation between the true similarities and the hashed similarities.

    ---
    ### Params:
    hash_sim_path : str (abspath)
        The Path to the csv file containing the hashed similarities
    city : str ("porto" | "rome")
        The city
    hash_type : str ("grid" | "disk")
        The hash method
    reference_measure : str ("dtw" | "frechet")
    """
    # Defining helper functions:

    def _mirrorDiagonal(M: np.ndarray) -> np.ndarray:
        """Flips and mirrors a two-dimenional np.array"""
        return M.values + np.rot90(np.fliplr(M.values))

    porto_dtw = _mirrorDiagonal(
        pd.read_csv(
            os.path.abspath(
                f"../prosjektoppgave/benchmarks/similarities/porto/porto-dtw-{NUMBER_OF_TRAJECTORIES}.csv"
            ),
            index_col=0,
        )
    ).flatten()
    # porto_fre = _mirrorDiagonal(
    #     pd.read_csv(
    #         os.path.abspath("../benchmarks/similarities/porto/porto-frechet.csv"), index_col=0
    #     )
    # ).flatten()
    rome_dtw = _mirrorDiagonal(
        pd.read_csv(
            os.path.abspath(
                f"../prosjektoppgave/benchmarks/similarities/rome/rome-dtw-{NUMBER_OF_TRAJECTORIES}.csv"
            ),
            index_col=0,
        )
    ).flatten()
    # rome_fre = _mirrorDiagonal(
    #     pd.read_csv(
    #         os.path.abspath("../benchmarks/similarities/rome/rome-frechet.csv"), index_col=0
    #     )
    # ).flatten()

    true_sims = {
        # "porto": {"dtw": porto_dtw, "frechet": porto_fre},
        "porto": {"dtw": porto_dtw},
        # "rome": {"dtw": rome_dtw, "frechet": rome_fre},
        "rome": {"dtw": rome_dtw},
    }

    hist_arr = {
        "porto": {
            "grid": {
                "dtw": (np.arange(0, 12, 0.2), np.arange(0, 3, 0.05)),
                # "frechet": (np.arange(0, 12, 0.2), np.arange(0, 0.08, 0.001)),
            },
            "disk": {
                "dtw": (np.arange(0, 4, 0.05), np.arange(0, 3, 0.05)),
                # "frechet": (np.arange(0, 4, 0.05), np.arange(0, 0.08, 0.001)),
            },
        },
        "rome": {
            "grid": {
                "dtw": (np.arange(0, 15, 0.2), np.arange(0, 6, 0.05)),
                # "frechet": (np.arange(0, 15, 0.2), np.arange(0, 0.10, 0.001)),
            },
            "disk": {
                "dtw": (np.arange(0, 3, 0.05), np.arange(0, 6, 0.05)),
                # "frechet": (np.arange(0, 3, 0.05), np.arange(0, 0.10, 0.001)),
            },
        },
    }

    hash_sim = _mirrorDiagonal(pd.read_csv(hash_sim_path, index_col=0)).flatten()
    print("hash sim")
    print(hash_sim)
    print("true sim")
    print(true_sims[city][reference_measure])
    corr = np.corrcoef(hash_sim, true_sims[city][reference_measure])[0][1]

    print(
        "Similarity correlation: ",
        np.corrcoef(hash_sim, true_sims[city][reference_measure])[0][1],
    )

    x = hash_sim

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    ax.hist2d(
        x,
        true_sims[city][reference_measure],
        bins=hist_arr[city][hash_type][reference_measure],
        cmap="turbo",
    )
    ax.set_ylabel(f"{reference_measure.upper()} distance", fontsize=18)
    ax.set_xlabel(f"{hash_type.capitalize()} scheme distance", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.text(
        0.99,
        0.04,
        f"{hash_type.capitalize()}/{city.capitalize()} - Correlation: {'{:.2f}'.format(corr)}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=16,
        color="white",
    )
    # ax.hist2d(x, true_sims[city]["fre"], bins=hist_arr[city][hash_type]["fre"], cmap="turbo")

    plt.show()


if __name__ == "__main__":
    # hash_sim_porto = os.path.abspath("./code/experiments/similarities/grid_porto.csv")
    hash_sim_porto = os.path.abspath(
        "../prosjektoppgave/hashed_similarities/output/grid/porto/grid_porto.csv"
    )
    dtw_sim_porto = os.path.abspath(
        f"../prosjektoppgave/benchmarks/similarities/porto/porto-dtw-{NUMBER_OF_TRAJECTORIES}.csv"
    )

    print(hash_sim_porto)
    print(dtw_sim_porto)

    draw_similarity_correlation(hash_sim_porto, "porto", "grid", "dtw")
