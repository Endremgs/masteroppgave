"""
This file contains methods for finding an optimal/working grid resolution / layers

Each experiment will be run in 20 parallell jobs
"""

# Importing nescessary modules

import numpy as np
import pandas as pd
import sys, os
from matplotlib import pyplot as plt
from multiprocessing import Pool

currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from schemes.lsh_grid import GridLSH

from utils.similarity_measures.distance import compute_hash_similarity

from constants import (
    PORTO_OUTPUT_FOLDER,
    ROME_OUTPUT_FOLDER,
    KOLUMBUS_OUTPUT_FOLDER,
    P_MAX_LAT,
    P_MIN_LAT,
    P_MAX_LON,
    P_MIN_LON,
    R_MAX_LAT,
    R_MIN_LAT,
    R_MAX_LON,
    R_MIN_LON,
    K_MAX_LAT,
    K_MIN_LAT,
    K_MAX_LON,
    K_MIN_LON,
    SIMILARITIES_OUTPUT_FOLDER_KOLUMBUS,
    SIMILARITIES_OUTPUT_FOLDER_PORTO,
    SIMILARITIES_OUTPUT_FOLDER_ROME,
    NUMBER_OF_TRAJECTORIES,
    COLOR_MAP,
)

# Defining some constants

PORTO_CHOSEN_DATA = f"../{PORTO_OUTPUT_FOLDER}/"
PORTO_META_FILE = f"../{PORTO_OUTPUT_FOLDER}/META-{NUMBER_OF_TRAJECTORIES}.TXT"

ROME_CHOSEN_DATA = f"../{ROME_OUTPUT_FOLDER}/"
ROME_META_FILE = f"../{ROME_OUTPUT_FOLDER}/META-{NUMBER_OF_TRAJECTORIES}.TXT"

KOLUMBUS_CHOSEN_DATA = f"../{KOLUMBUS_OUTPUT_FOLDER}/"
KOLUMBUS_META_FILE = f"../{KOLUMBUS_OUTPUT_FOLDER}/META-{NUMBER_OF_TRAJECTORIES}.TXT"


# Defining helper functions:
def _mirrorDiagonal(M: np.ndarray) -> np.ndarray:
    """Flips and mirrors a two-dimenional np.array"""
    return M.values + np.rot90(np.fliplr(M.values))


# NOTE: Check .stack().values meaning
# True similarities:

P_DTW = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_PORTO}/porto-dtw-{NUMBER_OF_TRAJECTORIES}.csv",
        index_col=0,
    )
).flatten()
P_FRE = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_PORTO}/porto-frechet-{NUMBER_OF_TRAJECTORIES}.csv",
        index_col=0,
    )
).flatten()

R_DTW = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_ROME}/rome-dtw-{NUMBER_OF_TRAJECTORIES}.csv",
        index_col=0,
    )
).flatten()
R_FRE = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_ROME}/rome-frechet-{NUMBER_OF_TRAJECTORIES}.csv",
        index_col=0,
    )
).flatten()

K_DTW = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_KOLUMBUS}/kolumbus-dtw-{NUMBER_OF_TRAJECTORIES}.csv",
        index_col=0,
    )
).flatten()
K_FRE = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_KOLUMBUS}/kolumbus-frechet-{NUMBER_OF_TRAJECTORIES}.csv",
        index_col=0,
    )
).flatten()

REFERENCE = {
    "portodtw": P_DTW,
    "portofrechet": P_FRE,
    "romedtw": R_DTW,
    "romefrechet": R_FRE,
    "kolumbusdtw": K_DTW,
    "kolumbusfrechet": K_FRE,
}


def _constructGrid(
    city: str, res: float, layers: int, meta_file: str, chosen_data: str
) -> GridLSH:
    """Constructs a grid hash object over the given city"""
    if city.lower() == "porto":
        return GridLSH(
            f"GP_{layers}-{'{:.2f}'.format(res)}",
            P_MIN_LAT,
            P_MAX_LAT,
            P_MIN_LON,
            P_MAX_LON,
            res,
            layers,
            meta_file,
            chosen_data,
        )
    elif city.lower() == "rome":
        return GridLSH(
            f"GR_{layers}-{'{:.2f}'.format(res)}",
            R_MIN_LAT,
            R_MAX_LAT,
            R_MIN_LON,
            R_MAX_LON,
            res,
            layers,
            meta_file,
            chosen_data,
        )
    elif city.lower() == "kolumbus":
        return GridLSH(
            f"GK_{layers}-{'{:.2f}'.format(res)}",
            K_MIN_LAT,
            K_MAX_LAT,
            K_MIN_LON,
            K_MAX_LON,
            res,
            layers,
            meta_file,
            chosen_data,
        )
    else:
        raise ValueError(f"City/dataset argument {city} not supported")


def _fun_wrapper_corr(args):
    city, res, lay, measure, reference = args
    Grid = _constructGrid(city, res, lay)
    hashes = Grid.compute_dataset_hashes()

    hashed_similarity = compute_hash_similarity(
        hashes=hashes, scheme="grid", measure=measure, parallel=False
    )

    hashed_array = _mirrorDiagonal(hashed_similarity).flatten()
    truesim_array = REFERENCE[city.lower() + reference.lower()]
    corr = np.corrcoef(hashed_array, truesim_array)[0][1]
    return corr


def _compute_grid_res_layers(
    city: str,
    layers: list[int],
    resolution: list[float],
    measure: str = "py_dtw_manhattan",
    reference: str = "dtw",
    parallell_jobs: int = 20,
):
    """Computations for the visualisation"""

    pool = Pool()

    results = []
    for lay in layers:
        result = []
        for res in np.arange(*resolution):
            print(f"L: {lay}", "{:.2f}".format(res), end="\r")
            # edits = _mirrorDiagonal(MEASURE[measure](hashes)).flatten()

            # corr = np.corrcoef(edits, REFERENCE[city.lower()+reference.lower()])[0][1]
            corrs = pool.map(
                _fun_wrapper_corr,
                [(city, res, lay, measure, reference) for _ in range(parallell_jobs)],
            )
            corr = np.average(np.array(corrs))
            std = np.std(np.array(corrs))
            result.append([corr, res, std])

        results.append([result, lay])

    return results


def plot_grid_res_layers(
    city: str,
    layers: list[int],
    resolution: list[float],
    measure: str = "dtw",
    reference: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution and layers for the grid hashes

    Param
    ---
    city : str
        Either "porto" or "rome", throws error unless
    layers : list[int]
        The layers that will be visualised -> [x, y, z...]
    resolution : list[float]
        The resolution that will be visualised -> [min, max, step]
    measure : str (default py_dtw_manhattan)
        The measure that will be used. Either dtw -> "dtw" or frechet -> "frechet"
    reference : str (default dtw)
        The true similarities that will be used as reference. Either dtw or frechet
    parallel_jobs : int (default 20)
        Yhe number of parallel jobs that will create the data foundation
    """

    results = _compute_grid_res_layers(
        city, layers, resolution, measure, reference, parallel_jobs
    )

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    for layer_element in results:
        corrs, layer = layer_element

        corre, res, std = list(zip(*corrs))
        corre = np.array(corre)
        res = np.array(res)
        std = np.array(std)
        color = COLOR_MAP[layer]
        ax1.plot(
            res,
            corre,
            c=color,
            label=f"{layer} layers",
            lw=2,
        )
        ax2.plot(res, std, c=color, alpha=0.3, ls="dashed")
        # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    ax1.legend(
        loc="lower right",
        ncols=5,
        fontsize=16,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        handletextpad=0.5,
        borderaxespad=0.2,
        columnspacing=1,
    )
    ax2.text(
        0.99,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Grid) - {reference.upper()} True\nSize: {NUMBER_OF_TRAJECTORIES}\nJobs: {parallel_jobs} ",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=12,
        color="black",
    )
    ax1.set_xlabel("Grid tile width (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax2.set_ylabel("Standard deviation - Dashed lines", fontsize=16)
    ax1.set_ylim([0.2, 1.0])
    # Dynamic y-axis limits based on values
    # ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax2.set_ylim([0.0, 0.1])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()


def _compute_grid_sizes(
    city: str,
    layer: int,
    resolution: float,
    measure: str = "dtw",
    reference: str = "dtw",
    sizes: list[int] = [],
):
    """"""

    def get_meta_file(city: str, size: int) -> str:
        if city.lower() == "porto":
            return f"../{PORTO_OUTPUT_FOLDER}/META-{size}.txt"
        elif city.lower() == "rome":
            return f"../{ROME_OUTPUT_FOLDER}/META-{size}.txt"
        elif city.lower() == "kolumbus":
            return f"../{KOLUMBUS_OUTPUT_FOLDER}/META-{size}.txt"

    def get_correlation():

        return corr

    OUTPUT_FOLDER = None
    CHOSEN_DATA = None
    if city.lower() == "porto":
        OUTPUT_FOLDER = SIMILARITIES_OUTPUT_FOLDER_PORTO
        CHOSEN_DATA = PORTO_CHOSEN_DATA
    elif city.lower() == "rome":
        OUTPUT_FOLDER = SIMILARITIES_OUTPUT_FOLDER_ROME
        CHOSEN_DATA = ROME_CHOSEN_DATA
    elif city.lower() == "kolumbus":
        OUTPUT_FOLDER = SIMILARITIES_OUTPUT_FOLDER_KOLUMBUS
        CHOSEN_DATA = KOLUMBUS_CHOSEN_DATA

    results = []
    for dataset_size in sizes:
        Grid = _constructGrid(
            city,
            resolution,
            layer,
            meta_file=get_meta_file(city=city, size=dataset_size),
            chosen_data=CHOSEN_DATA,
        )
        hashes = Grid.compute_dataset_hashes()
        hashed_similarity = compute_hash_similarity(
            hashes=hashes, scheme="grid", measure=measure, parallel=False
        )
        hashed_array = _mirrorDiagonal(hashed_similarity).flatten()

        true_sim_array = _mirrorDiagonal(
            pd.read_csv(
                f"../{OUTPUT_FOLDER}/{city.lower()}-{reference}-{dataset_size}.csv",
                index_col=0,
            )
        ).flatten()
        print("Size: ", dataset_size)
        correlations = np.corrcoef(hashed_array, true_sim_array)[0][1]
        print("Correlations: ", correlations)
        corr = np.average(np.array(correlations))
        std = np.std(np.array(correlations))
        results.append([corr, resolution, std, dataset_size])
    print(results)
    return results


def plot_grid_sizes(
    city: str,
    layer: int,
    sizes: list[int],
    resolution: float,
    measure: str = "dtw",
    reference: str = "dtw",
):
    """Visualises the 'optimal' values for resolution and layers for the grid hashes

    Param
    ---
    # TODO: fill in
    """

    results = _compute_grid_sizes(
        city=city,
        layer=layer,
        resolution=resolution,
        measure=measure,
        reference=reference,
        sizes=sizes,
    )

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    for element in results:

        corre, res, std, dataset_size = element
        print(corre, res, std, dataset_size)
        corre = np.array(corre)
        res = np.array(res)
        std = np.array(std)
        color = COLOR_MAP[layer]
        ax1.plot(
            res,
            corre,
            c=color,
            label=f"{layer} layers",
            lw=2,
        )
        ax2.plot(res, std, c=color, alpha=0.3, ls="dashed")
        # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    ax1.legend(
        loc="lower right",
        ncols=5,
        fontsize=16,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        handletextpad=0.5,
        borderaxespad=0.2,
        columnspacing=1,
    )
    ax2.text(
        0.99,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Grid) - {reference.upper()} True\nSize: {str(sizes)}\n ",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=12,
        color="black",
    )
    ax1.set_xlabel("Number of trajectories", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax2.set_ylabel("Standard deviation - Dashed lines", fontsize=16)
    ax1.set_ylim([0, 1.0])
    # Dynamic y-axis limits based on values
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax2.set_xlim([sizes[0], sizes[-1]])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()
