""" Sheet containing DTW methods related to true similarity creation """

import numpy as np
import pandas as pd
import collections as co
from traj_dist.pydist.dtw import e_dtw as p_dtw
from traj_dist.distance import dtw as c_dtw

from utils.helpers import trajectory_distance as td

from multiprocessing import Pool
import timeit as ti
import time


def py_dtw(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Method for computing DTW similarity between all trajectories in a given dataset using python.

    Params
    ---
    trajectories : dict[str, list[list[float]]]
        A dictionary containing the trajectories

    Returns
    ---
    A nxn pandas dataframe containing the pairwise similarities - sorted alphabetically
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectoris = len(sorted_trajectories)

    M = np.zeros((num_trajectoris, num_trajectoris))

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            X = np.array(sorted_trajectories[traj_i])
            Y = np.array(sorted_trajectories[traj_j])
            dtw = p_dtw(X, Y)
            M[i, j] = dtw
            if i == j:
                break

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


def measure_py_dtw(args):
    """Method for measuring time efficiency using py_dtw

    Params
    ---
    args : (trajectories: dict[str, list[list[float]]], number: int, repeat: int) list
    """
    trajectories, number, repeat = args

    measures = ti.repeat(
        lambda: py_dtw(trajectories),
        number=number,
        repeat=repeat,
        timer=time.process_time,
    )
    return measures


def cy_dtw(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Method for computing DTW similarity between all trajectories in a given dataset using cython.

    Params
    ---
    trajectories : dict[str, list[list[float]]]
        A dictionary containing the trajectories

    Returns
    ---
    A nxn pandas dataframe containing the pairwise similarities - sorted alphabetically
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectoris = len(sorted_trajectories)

    M = np.zeros((num_trajectoris, num_trajectoris))

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            X = np.array(sorted_trajectories[traj_i])
            Y = np.array(sorted_trajectories[traj_j])
            dtw = c_dtw(X, Y)
            M[i, j] = dtw
            if i == j:
                break

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


# Helper function for dtw parallell programming for speedy computations
def _fun_wrapper(args):
    x, y, j = args
    dtw = c_dtw(x, y)
    return dtw, j


def cy_dtw_pool(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Same as above, but using a pool of procesess for speedup
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectoris = len(sorted_trajectories)

    M = np.zeros((num_trajectoris, num_trajectoris))

    pool = Pool(12)

    for i, traj_i in enumerate(sorted_trajectories.keys()):

        dtw_elements = pool.map(
            _fun_wrapper,
            [
                (
                    np.array(sorted_trajectories[traj_i]),
                    np.array(sorted_trajectories[traj_j]),
                    j,
                )
                for j, traj_j in enumerate(sorted_trajectories.keys())
                if i >= j
            ],
        )

        for element in dtw_elements:
            M[i, element[1]] = element[0]

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

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            total_dtw = 0  # Initialize total DTW similarity for this pair
            for layer_i, layer_j in zip(
                sorted_trajectories[traj_i], sorted_trajectories[traj_j]
            ):
                X = np.array(layer_i)
                Y = np.array(layer_j)
                dtw = c_dtw(
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
    dtw_sum = sum(c_dtw(np.array(x), np.array(y)) for x, y in zip(x_layers, y_layers))
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


def measure_cy_dtw(args):
    """Method for measuring time efficiency using py_dtw"""

    trajectories, number, repeat = args

    measures = ti.repeat(
        lambda: cy_dtw(trajectories),
        number=number,
        repeat=repeat,
        timer=time.process_time,
    )
    return measures
