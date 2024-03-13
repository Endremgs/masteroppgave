""" Sheet containing DTW methods related to true similarity creation """

import numpy as np
import pandas as pd
import collections as co
from traj_dist.pydist.dtw import e_dtw as p_dtw
from traj_dist.distance import dtw as c_dtw

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


def measure_cy_dtw(args):
    """Method for measuring time efficiency using cy_dtw"""

    trajectories, number, repeat = args

    measures = ti.repeat(
        lambda: cy_dtw(trajectories),
        number=number,
        repeat=repeat,
        timer=time.process_time,
    )
    return measures


def measure_py_dtw(args):
    """Method for measuring time efficiency using py_dtw"""

    trajectories, number, repeat = args

    measures = ti.repeat(
        lambda: py_dtw(trajectories),
        number=number,
        repeat=repeat,
        timer=time.process_time,
    )
    return measures
