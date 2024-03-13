import numpy as np
import pandas as pd
import collections as co
from traj_dist.pydist.dtw import e_dtw as p_dtw
from traj_dist.distance import dtw as c_dtw

from multiprocessing import Pool
import timeit as ti
import time


def py_dtw_hashes(hashes: dict[str, list[list[list[float]]]]) -> pd.DataFrame:
    """
    Method for computing DTW similarity between all layers of trajectories in a given dataset using python, and summing these similarities.

    Params
    ---
    trajectories : dict[str, list[list[list[float]]]]
        A dictionary containing the trajectories, where each key corresponds to multiple layers of trajectories.

    Returns
    ---
    A nxn pandas dataframe containing the pairwise summed similarities - sorted alphabetically
    """
    sorted_trajectories = co.OrderedDict(sorted(hashes.items()))
    num_trajectories = len(sorted_trajectories)

    M = np.zeros((num_trajectories, num_trajectories))

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            total_dtw = 0  # Initialize total DTW similarity for this pair
            for layer_i, layer_j in zip(
                sorted_trajectories[traj_i], sorted_trajectories[traj_j]
            ):
                X = np.array(layer_i)
                Y = np.array(layer_j)
                # Ensure both X and Y are not empty and have the correct shape
                # NOTE: If the calculation is skipped, either X or Y (or both) is empty.
                # This usually occurs for comparisons starting on the second half of the trajectories for unknown reasons.
                # We normally use sizes so big that the number of skipped comparisons can be neglected as it does not affect the correlation.
                if X.size > 0 and Y.size > 0 and X.ndim == 2 and Y.ndim == 2:
                    dtw = p_dtw(
                        X, Y
                    )  # Assuming c_dtw is defined elsewhere to calculate DTW similarity
                    total_dtw += dtw
            M[i, j] = total_dtw
            if i == j:
                break  # This optimizes by not recalculating for identical trajectories

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


def cy_dtw_hashes(hashes: dict[str, list[list[list[float]]]]) -> pd.DataFrame:
    """
    Method for computing DTW similarity between all layers of trajectories in a given dataset using cython, and summing these similarities.

    Params
    ---
    trajectories : dict[str, list[list[list[float]]]]
        A dictionary containing the trajectories, where each key corresponds to multiple layers of trajectories.

    Returns
    ---
    A nxn pandas dataframe containing the pairwise summed similarities - sorted alphabetically
    """
    sorted_trajectories = co.OrderedDict(sorted(hashes.items()))
    num_trajectories = len(sorted_trajectories)

    M = np.zeros((num_trajectories, num_trajectories))

    # total_comparisons = 0
    # total_skipped_comparisons = 0

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            total_dtw = 0  # Initialize total DTW similarity for this pair
            for layer_i, layer_j in zip(
                sorted_trajectories[traj_i], sorted_trajectories[traj_j]
            ):
                X = np.array(layer_i)
                Y = np.array(layer_j)
                # Ensure both X and Y are not empty and have the correct shape
                # NOTE: If the calculation is skipped, either X or Y (or both) is empty.
                # This usually occurs for comparisons starting on the second half of the trajectories for unknown reasons.
                # We normally use sizes so big that the number of skipped comparisons can be neglected as it does not affect the correlation.
                if X.size > 0 and Y.size > 0 and X.ndim == 2 and Y.ndim == 2:
                    dtw = c_dtw(
                        X, Y
                    )  # Assuming c_dtw is defined elsewhere to calculate DTW similarity
                    total_dtw += dtw
            M[i, j] = total_dtw
            if i == j:
                break  # This optimizes by not recalculating for identical trajectories

    # print("Total comparisons", total_comparisons)
    # print("Total skipped comparisons", total_skipped_comparisons)

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


def _fun_wrapper_hashes(args):
    x_layers, y_layers, j = args
    filtered_x_layers = [x for x in x_layers if x]  # Filter out empty lists
    filtered_y_layers = [y for y in y_layers if y]  # Filter out empty lists
    dtw_sum = sum(
        c_dtw(np.array(x), np.array(y))
        for x, y in zip(filtered_x_layers, filtered_y_layers)
    )
    return dtw_sum, j


def cy_dtw_hashes_pool(
    trajectories: dict[str, list[list[list[float]]]]
) -> pd.DataFrame:
    """
    Calculates the DTW similarity for trajectories with multiple layers, using a pool of processes for speedup.
    """
    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectories = len(sorted_trajectories)

    M = np.zeros((num_trajectories, num_trajectories))

    with Pool(12) as pool:
        for i, traj_i_key in enumerate(sorted_trajectories.keys()):
            traj_i_layers = sorted_trajectories[traj_i_key]

            dtw_elements = pool.map(
                _fun_wrapper_hashes,
                [
                    (
                        traj_i_layers,
                        sorted_trajectories[traj_j_key],
                        j,
                    )
                    for j, traj_j_key in enumerate(sorted_trajectories.keys())
                    if i >= j
                ],
            )

            for dtw_sum, j in dtw_elements:
                M[i, j] = dtw_sum
                M[j, i] = dtw_sum  # Assuming DTW distance is symmetric

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


def measure_hashed_cy_dtw(hashes: dict[str, list[list[list[float]]]]):
    """Method for measuring time efficiency using cy_dtw_hashes"""
    measures = ti.repeat(
        lambda: cy_dtw_hashes(hashes), number=1, repeat=1, timer=time.process_time
    )
    return measures


def measure_hashed_py_dtw(hashes: dict[str, list[list[list[float]]]]):
    """Method for measuring time efficiency using py_dtw_hashes"""
    measures = ti.repeat(
        lambda: py_dtw_hashes(hashes), number=1, repeat=1, timer=time.process_time
    )
    return measures
