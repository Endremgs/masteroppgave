"""
This file contains methods for finding an optimal/working number of disks and their diameter / layers

Each experiment will be run in 20 parallell jobs
"""

# Importing nescessary modules

import numpy as np
import pandas as pd
import sys, os
from matplotlib import pyplot as plt
from multiprocessing import Pool
import scipy

currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from schemes.lsh_disk import DiskLSH

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
)

# Defining some constants

PORTO_CHOSEN_DATA = f"../{PORTO_OUTPUT_FOLDER}/"
PORTO_META_TEST = f"../{PORTO_OUTPUT_FOLDER}/META-50.TXT"

ROME_CHOSEN_DATA = f"../{ROME_OUTPUT_FOLDER}/"
ROME_META_TEST = f"../{ROME_OUTPUT_FOLDER}/META-50.TXT"

KOLUMBUS_CHOSEN_DATA = f"../{KOLUMBUS_OUTPUT_FOLDER}/"
KOLUMBUS_META_TEST = f"../{KOLUMBUS_OUTPUT_FOLDER}/META-50.TXT"


# Defining helper functions:
def _mirrorDiagonal(M: np.ndarray) -> np.ndarray:
    """Flips and mirrors a two-dimenional np.array"""
    return M.values + np.rot90(np.fliplr(M.values))


# NOTE: Check .stack().values meaning
# True similarities:

P_DTW = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_PORTO}/porto-dtw-testset.csv", index_col=0
    )
).flatten()
P_FRE = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_PORTO}/porto-frechet-testset.csv", index_col=0
    )
).flatten()

R_DTW = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_ROME}/rome-dtw-testset.csv", index_col=0
    )
).flatten()
R_FRE = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_ROME}/rome-frechet-testset.csv", index_col=0
    )
).flatten()
K_DTW = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_KOLUMBUS}/kolumbus-dtw-testset.csv",
        index_col=0,
    )
).flatten()
K_FRE = _mirrorDiagonal(
    pd.read_csv(
        f"../{SIMILARITIES_OUTPUT_FOLDER_KOLUMBUS}/kolumbus-frechet-testset.csv",
        index_col=0,
    )
).flatten()
NULL_TEST_CSV = _mirrorDiagonal(
    pd.read_csv("../benchmarks/true_similarities/null-testset.csv", index_col=0)
).flatten()

REFERENCE = {
    "portodtw": P_DTW,
    "portofrechet": P_FRE,
    "romedtw": R_DTW,
    "romefrechet": R_FRE,
    "kolumbusdtw": K_DTW,
    "kolumbusfrechet": K_FRE,
    "null_testset": NULL_TEST_CSV,
}


def _constructDisk(city: str, diameter: float, layers: int, disks: int = 50) -> DiskLSH:
    """Constructs a disk hash object over the given city"""
    if city.lower() == "porto":
        return DiskLSH(
            f"GP_{layers}-{'{:.2f}'.format(diameter)}",
            P_MIN_LAT,
            P_MAX_LAT,
            P_MIN_LON,
            P_MAX_LON,
            disks,
            layers,
            diameter,
            PORTO_META_TEST,
            PORTO_CHOSEN_DATA,
        )
    elif city.lower() == "rome":
        return DiskLSH(
            f"GR_{layers}-{'{:.2f}'.format(diameter)}",
            R_MIN_LAT,
            R_MAX_LAT,
            R_MIN_LON,
            R_MAX_LON,
            disks,
            layers,
            diameter,
            ROME_META_TEST,
            ROME_CHOSEN_DATA,
        )
    elif city.lower() == "kolumbus":
        return DiskLSH(
            f"GK_{layers}-{'{:.2f}'.format(diameter)}",
            K_MIN_LAT,
            K_MAX_LAT,
            K_MIN_LON,
            K_MAX_LON,
            disks,
            layers,
            diameter,
            KOLUMBUS_META_TEST,
            KOLUMBUS_CHOSEN_DATA,
        )
    else:
        raise ValueError(f"City/dataset argument {city} not supported")


def _compute_hashes(disk: DiskLSH, measure: str = "dtw") -> dict[str, list]:
    if measure == "dtw" or measure == "frechet":
        return disk.compute_dataset_hashes_with_KD_tree_numerical()
    else:
        raise ValueError("Preferred similarity measure not supported")


def _fun_wrapper_corr(args):
    city, dia, lay, disks, measure, reference = args
    Disk = _constructDisk(city=city, diameter=dia, layers=lay, disks=disks)
    hashes = _compute_hashes(Disk, measure)
    hashed_similarity = compute_hash_similarity(
        hashes=hashes, scheme="disk", measure=measure, parallel=False
    )

    hashed_array = _mirrorDiagonal(hashed_similarity).flatten()
    truesim_array = REFERENCE[city.lower() + reference.lower()]
    corr = np.corrcoef(hashed_array, truesim_array)[0][1]
    return corr


def _test_wrapper_corr(args):
    city, dia, lay, measure, reference = args
    Disk = _constructDisk(city, dia, lay)
    hashes = _compute_hashes(Disk, measure)
    hash_array = _mirrorDiagonal(MEASURE[measure](hashes)).flatten()
    truesim_array_dtw = REFERENCE[city.lower() + "dtw"]
    truesim_array_frechet = REFERENCE[city.lower() + "frechet"]
    null_values = REFERENCE["null_testset"]
    # spearman_corr = scipy.stats.spearmanr(hash_array, truesim_array_dtw)
    # print("Spearman_corr", spearman_corr)
    # kendall_corr = scipy.stats.kendalltau(hash_array, truesim_array_dtw)
    # print("Kendall_corr", kendall_corr)

    test_corr = np.corrcoef(hash_array, truesim_array_dtw)[0][1]
    return test_corr


def _compute_disk_diameter_layers(
    city: str,
    layers: list[int],
    diameter: list[float],
    disks: int = 100,
    measure: str = "dtw",
    reference: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation"""

    pool = Pool()

    results = []
    for lay in layers:
        result = []
        for dia in np.arange(*diameter):
            print(f"L: {lay}", "{:.2f}".format(dia), end="\r")
            corrs = pool.map(
                _fun_wrapper_corr,
                [
                    (city, dia, lay, disks, measure, reference)
                    for _ in range(parallel_jobs)
                ],
            )

            # NOTE: The below function call can be used to verify the correlation methods by measuring
            # correlation between authentic similarity values and approximately zero values or other combinations
            # corrs = pool.map(
            #     _test_wrapper_corr,
            #     [(city, dia, lay, measure, reference) for _ in range(parallell_jobs)],
            # )
            corr = np.average(np.array(corrs))
            std = np.std(np.array(corrs))
            result.append([corr, dia, std])

        results.append([result, lay])
    return results


def plot_disk_dia_layers(
    city: str,
    layers: list[int],
    diameter: list[float],
    disks: int = 100,
    measure: str = "dtw",
    reference: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution and layers for the disk hashes

    Param
    ---
    city : str
        Either "porto" or "rome", throws error unless
    layers : list[int]
        The layers that will be visualised -> [x, y, z...]
    diameter : list[float]
        The diameter that will be visualised -> [min, max, step]
    disks : int (default 100)
    measure : str (default dtw)
        The measure that will be used. Either dtw or frechet -> "dtw" or "frechet"
    reference : str (default dtw)
        The true similarities that will be used as reference. Either dtw or frechet
    parallel_jobs : int (default 20)
        Yhe number of parallel jobs that will create the data foundation
    """

    results = _compute_disk_diameter_layers(
        city, layers, diameter, disks, measure, reference, parallel_jobs
    )

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    for layer_element in results:
        corrs, layer = layer_element

        corre, dia, std = list(zip(*corrs))
        corre = np.array(corre)
        dia = np.array(dia)
        std = np.array(std)
        ax1.plot(
            dia,
            corre,
            c=cmap(float(layer - 1) / (1.2 * N)),
            label=f"{layer} layers",
            lw=2,
        )
        ax2.plot(dia, std, c=cmap(float(layer - 1) / (1.2 * N)), ls="dashed")
        # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    ax1.legend(
        loc="center right",
        ncols=2,
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
        f"{city.capitalize()}: {measure.upper()} (Disk) - {reference.upper()} True",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=14,
        color="grey",
    )
    ax1.set_xlabel("Disk diameter (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
    ax2.set_ylabel("Standard deviation \- Dashed lines", fontsize=18)
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()


## Methods for visualusing the effect of number of disks


# def _fun_wrapper_corr_disks(args):
#     city, dia, lay, disks, measure, reference = args
#     Disk = _constructDisk(city, dia, lay, disks)
#     hashes = _compute_hashes(Disk, measure)

#     edits = _mirrorDiagonal(MEASURE[measure](hashes)).flatten()
#     corr = np.corrcoef(edits, REFERENCE[city.lower() + reference.lower()])[0][1]
#     return corr


# def _compute_disk_numbers(
#     city: str,
#     layers: int,
#     diameter: int,
#     disks: list[int],
#     measure: str = "py_dtw_euclidean",
#     reference: str = "dtw",
#     parallell_jobs: int = 20,
# ):
#     """Computations for the visualisation of the number of disks"""

#     pool = Pool()

#     results = []

#     for disk_number in disks:
#         print(f"DN: {disk_number}", end="\r")
#         corrs = pool.map(
#             _fun_wrapper_corr_disks,
#             [
#                 (city, diameter, layers, disk_number, measure, reference)
#                 for _ in range(parallell_jobs)
#             ],
#         )
#         corr = np.average(np.array(corrs))
#         std = np.std(np.array(corrs))
#         results.append([corr, disk_number, std])

#     return results


# def plot_disk_numbers(
#     city: str,
#     layers: int,
#     diameter: int,
#     disks: list[int],
#     measure: str = "dtw",
#     reference: str = "dtw",
#     parallel_jobs: int = 20,
# ):
#     """Visualises the effect of adjusting the number of disks

#     Param
#     ---
#     city : str
#         Either "porto" or "rome", throws error unless
#     layers : int
#         The layers that will be visualised
#     diameter : list[float]
#         The resolution that will be used for the disks
#     disks : list[int]
#         The number of disks that will be plotted
#     measure : str (default py_dtw_euclidean)
#         The measure that will be used. Either dtw or frechet -> "dtw" or "frechet"
#     reference : str (default dtw)
#         The true similarities that will be used as reference. Either dtw or frechet
#     parallel_jobs : int (default 20)
#         Yhe number of parallell jobs that will create the data foundation
#     """

#     results = _compute_disk_numbers(
#         city, layers, diameter, disks, measure, reference, parallel_jobs
#     )

#     fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
#     ax2 = ax1.twinx()
#     # fig.set_size_inches(10,8)
#     cmap = plt.get_cmap("gist_ncar")
#     N = len(results)

#     corre, num_disks, std = list(zip(*results))
#     corre = np.array(corre)
#     num_disks = np.array(num_disks)
#     std = np.array(std)
#     ax1.plot(num_disks, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
#     ax2.plot(num_disks, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
#     # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

#     # Now styling the figure
#     # ax1.legend(loc="lower left", ncols=3)
#     ax2.text(
#         0.99,
#         0.99,
#         f"{city.capitalize()}: {measure.upper()} (Disk) - {reference.upper()} True",
#         ha="right",
#         va="top",
#         transform=ax2.transAxes,
#         fontsize=12,
#         color="grey",
#     )
#     ax1.set_xlabel("Number of disks", fontsize=18)
#     ax1.set_ylabel("Pearson correlation coefficient - Solid line", fontsize=18)
#     ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
#     ax2.set_ylabel("Standard deviation - Dashed line", fontsize=18)
#     ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
#     ax1.tick_params(axis="both", which="major", labelsize=16)
#     ax2.tick_params(axis="both", which="major", labelsize=16)

#     plt.show()
