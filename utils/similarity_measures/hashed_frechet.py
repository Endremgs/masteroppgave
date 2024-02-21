import numpy as np
import pandas as pd
import collections as co

from multiprocessing import Pool
from traj_dist.distance import frechet as c_frechet


def cy_frechet_hashes(hashes: dict[str, list[list[list[float]]]]) -> pd.DataFrame:
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

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            total_dtw = 0  # Initialize total DTW similarity for this pair
            for layer_i, layer_j in zip(
                sorted_trajectories[traj_i], sorted_trajectories[traj_j]
            ):
                X = np.array(layer_i)
                Y = np.array(layer_j)
                dtw = c_frechet(
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


def _fun_wrapper_hashes(args):
    x_layers, y_layers, j = args
    frechet_sum = sum(
        c_frechet(np.array(x), np.array(y)) for x, y in zip(x_layers, y_layers)
    )
    return frechet_sum, j


def cy_frechet_hashes_pool(
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
