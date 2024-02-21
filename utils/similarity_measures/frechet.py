""" Sheet containing Frechet methods related to true similarity creation """

import numpy as np
import pandas as pd
import collections as co

from multiprocessing import Pool
import timeit as ti
import time

from traj_dist.pydist.frechet import frechet as p_frechet
from traj_dist.distance import frechet as c_frechet


def py_frechet(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Method for computing frechet similarity between all trajectories in a given dataset using python.

    Params
    ---
    trajectories : dict[str, list[list[float]]]
        A dictionary containing the trajectories

    Returns
    ---
    A nxn pandas dataframe containing the pairwise similarities - sorted alphabetically
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectories = len(sorted_trajectories)

    M = np.zeros((num_trajectories, num_trajectories))

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            X = np.array(sorted_trajectories[traj_i])
            Y = np.array(sorted_trajectories[traj_j])
            frechet = p_frechet(X, Y)
            M[i, j] = frechet
            if i == j:
                break

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


def measure_py_frechet(args):
    """Method for measuring time efficiency using py_dtw"""
    trajectories, number, repeat = args

    measures = ti.repeat(
        lambda: py_frechet(trajectories),
        number=number,
        repeat=repeat,
        timer=time.process_time,
    )
    return measures


def cy_frechet(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Method for computing frechet similarity between all trajectories in a given dataset using cython.

    Params
    ---
    trajectories : dict[str, list[list[float]]]
        A dictionary containing the trajectories

    Returns
    ---
    A nxn pandas dataframe containing the pairwise similarities - sorted alphabetically
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectories = len(sorted_trajectories)

    M = np.zeros((num_trajectories, num_trajectories))

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            X = np.array(sorted_trajectories[traj_i])
            Y = np.array(sorted_trajectories[traj_j])
            frech = c_frechet(X, Y)
            M[i, j] = frech
            if i == j:
                break

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


# Helper function for dtw parallell programming for speedy computations
def _fun_wrapper(args):
    x, y, j = args
    frechet = c_frechet(x, y)
    return frechet, j


def cy_frechet_pool(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Same as above, but using a pool of procesess for speedup
    """
    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectories = len(sorted_trajectories)

    M = np.zeros((num_trajectories, num_trajectories))

    pool = Pool(12)

    for i, traj_i in enumerate(sorted_trajectories.keys()):
        if (i % 5) == 0:
            print(f"Cy Pool Frechet: {i}/{num_trajectories}")
        frech_elements = pool.map(
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

        for element in frech_elements:
            M[i, element[1]] = element[0]

    df = pd.DataFrame(
        M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys()
    )

    return df


# def measure_cy_frechet(args):
#     """Method for measuring time efficiency using py_dtw"""
#     trajectories, number, repeat = args
#     measures = ti.repeat(
#         lambda: cy_frechet(trajectories),
#         number=number,
#         repeat=repeat,
#         timer=time.process_time,
#     )
#     return measures
