import numpy as np
import pandas as pd
import collections as co

from multiprocessing import Pool

from .hashed_dtw import dtw_euclidean
from .hashed_dtw import dtw_manhattan
from utils.similarity_measures.frechet import (
    cy_frechet_hashes,
    cy_frechet_hashes_pool,
)
from utils.similarity_measures.dtw import cy_dtw, cy_dtw_hashes_pool


def _fun_wrapper_dtw_manhattan(args):
    x, y, j = args
    e_dist = dtw_manhattan(x, y)[0]
    return e_dist, j


def py_dtw_manhattan(hashes: dict[str, list[list[str]]]) -> pd.DataFrame:
    sorted_hashes = co.OrderedDict(sorted(hashes.items()))
    num_hashes = len(sorted_hashes)

    M = np.zeros((num_hashes, num_hashes))
    for i, hash_i in enumerate(sorted_hashes.keys()):
        for j, hash_j in enumerate(sorted_hashes.keys()):
            X = np.array(sorted_hashes[hash_i], dtype=object)
            Y = np.array(sorted_hashes[hash_j], dtype=object)
            e_dist = dtw_manhattan(X, Y)[0]
            M[i, j] = e_dist
            if i == j:
                break

    df = pd.DataFrame(M, index=sorted_hashes.keys(), columns=sorted_hashes.keys())

    return df


def py_dtw_manhattan_parallel(hashes: dict[str, list[list[str]]]) -> pd.DataFrame:

    sorted_hashes = co.OrderedDict(sorted(hashes.items()))
    num_hashes = len(sorted_hashes)

    M = np.zeros((num_hashes, num_hashes))
    pool = Pool(12)

    for i, hash_i in enumerate(sorted_hashes.keys()):
        elements = pool.map(
            _fun_wrapper_dtw_manhattan,
            [
                (
                    np.array(sorted_hashes[hash_i], dtype=object),
                    np.array(sorted_hashes[traj_j], dtype=object),
                    j,
                )
                for j, traj_j in enumerate(sorted_hashes.keys())
                if i >= j
            ],
        )

        for element in elements:
            M[i, element[1]] = element[0]

    df = pd.DataFrame(M, index=sorted_hashes.keys(), columns=sorted_hashes.keys())

    return df


def py_dtw_euclidean(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """Coordinate dtw as hashes"""
    sorted_hashes = co.OrderedDict(sorted(hashes.items()))
    num_hashes = len(sorted_hashes)

    M = np.zeros((num_hashes, num_hashes))
    for i, hash_i in enumerate(sorted_hashes.keys()):
        for j, hash_j in enumerate(sorted_hashes.keys()):
            X = np.array(sorted_hashes[hash_i], dtype=object)
            Y = np.array(sorted_hashes[hash_j], dtype=object)
            e_dist = dtw_euclidean(X, Y)
            M[i, j] = e_dist
            if i == j:
                break

    df = pd.DataFrame(M, index=sorted_hashes.keys(), columns=sorted_hashes.keys())

    return df


def transform_np_numerical_disk_hashes_to_non_np(
    hashes: dict[str, list[list[float]]]
) -> dict[str, list[list[list[float]]]]:
    """Transforms the numerical disk hashes to a format that fits the true dtw similarity measure (non numpy input)"""
    transformed_data = {
        key: [[array.tolist() for array in sublist] for sublist in value]
        for key, value in hashes.items()
    }
    return transformed_data


def compute_hash_similarity(
    hashes: dict[str, list[list[list[float]]]],
    scheme: str,
    measure: str,
    paralell: bool = False,
) -> pd.DataFrame:
    if scheme == "disk":
        hashes = transform_np_numerical_disk_hashes_to_non_np(hashes)
    if measure == "dtw":
        if paralell:
            return cy_dtw_hashes_pool(hashes)
        return cy_dtw(hashes)
    elif measure == "frechet":
        if paralell:
            return cy_frechet_hashes_pool(hashes)
        return cy_frechet_hashes(hashes)
