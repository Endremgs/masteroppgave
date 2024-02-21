import numpy as np
import pandas as pd
import collections as co

from multiprocessing import Pool
import timeit as ti
import time

from collections import OrderedDict
from .hashed_dtw import dtw_euclidean
from .hashed_dtw import dtw_manhattan
from utils.similarity_measures.frechet import cy_frechet_pool, cy_frechet
from utils.similarity_measures.dtw import cy_dtw_pool


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
) -> OrderedDict:
    """Transforms the numerical disk hashes to a format that fits the true dtw similarity measure (non numpy input)"""
    transformed_data = OrderedDict()
    for key, layer in hashes.items():
        transformed_points = []
        for points in layer:
            transformed_traj = [point.tolist() for point in points]
            for point in transformed_traj:
                transformed_points.append(point)
        transformed_data[key] = transformed_points
    return transformed_data


def frechet_disk(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """Frechet distance for disk hashes (Used for correlation computation due to parallell jobs)"""
    transformed_data = transform_np_numerical_disk_hashes_to_non_np(hashes)
    return cy_frechet(transformed_data)


def frechet_disk_parallel(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """Frechet distance for disk hashes computed in parallell"""
    transformed_data = transform_np_numerical_disk_hashes_to_non_np(hashes)
    return cy_frechet_pool(transformed_data)


def dtw_disk_parallel(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """DTW distance for disk hashes computed in parallell"""
    transformed_data = transform_np_numerical_disk_hashes_to_non_np(hashes)
    return cy_dtw_pool(transformed_data)


def disk_coordinates(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """The hashed disk coordinates"""
    hashed_coordinates = transform_np_numerical_disk_hashes_to_non_np(hashes)
    return hashed_coordinates
