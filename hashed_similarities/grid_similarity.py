""" 
Sheet that will be used to measure the time needed to generate similarities of the grid-hashes 

Will run N processess in parallell to measure time efficiency
"""

from multiprocessing import Pool
import time
import timeit as ti
import pandas as pd
import os, sys

currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from schemes.lsh_grid import GridLSH

from utils.similarity_measures.distance import (
    compute_hash_similarity,
)

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
)

PORTO_CHOSEN_DATA = f"../{PORTO_OUTPUT_FOLDER}/"
ROME_CHOSEN_DATA = f"../{ROME_OUTPUT_FOLDER}/"
KOLUMBUS_CHOSEN_DATA = f"../{KOLUMBUS_OUTPUT_FOLDER}/"


def PORTO_META(size: int):
    return f"../{PORTO_OUTPUT_FOLDER}/META-{size}.txt"


def ROME_META(size: int):
    return f"../{ROME_OUTPUT_FOLDER}/META-{size}.txt"


def KOLUMBUS_META(size: int):
    return f"../{KOLUMBUS_OUTPUT_FOLDER}/META-{size}.txt"


# MEASURE = {
#     "py_dtw_manhattan": py_dtw_manhattan_parallel,
# }


def _constructGrid(city: str, res: float, layers: int, size: int) -> GridLSH:
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
            PORTO_META(size),
            PORTO_CHOSEN_DATA,
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
            ROME_META(size),
            ROME_CHOSEN_DATA,
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
            KOLUMBUS_META(size),
            KOLUMBUS_CHOSEN_DATA,
        )
    else:
        raise ValueError("City argument must be either porto or rome")


def generate_grid_hash_similarity(
    city: str, res: float, layers: int, measure: str = "dtw", size: int = 50
) -> pd.DataFrame:
    """Generates the full grid hash similarities and saves it as a dataframe"""

    Grid = _constructGrid(city, res, layers, size)
    hashes = Grid.compute_dataset_hashes()
    similarities = compute_hash_similarity(
        hashes=hashes, scheme="grid", measure=measure, parallel=True
    )

    return similarities


# TODO - measure computation time
# def _computeSimilarities(args) -> list:
#     hashes, measure = args
#     elapsed_time = ti.timeit(
#         lambda: MEASURE[measure](hashes), number=1, timer=time.process_time
#     )
#     return elapsed_time


# def measure_grid_hash_similarity_computation_time(
#     city: str,
#     size: int,
#     res: float,
#     layers: int,
#     measure: str = "py_dtw_manhattan",
#     parallell_jobs: int = 10,
# ) -> list:
#     """
#     Method to measure the execution time of similarity computation of the hashes

#     Param
#     ---
#     city : str
#         Either "porto" or "rome"
#     size : int
#         The dataset-size that will be computed
#     res : float
#         The grid resolution
#     layers : int
#         The number of layers that will be used
#     measure : str (Either "py_dtw_manhattan" - "py_dtw_manhattan" default)
#         The measure that will be used for computation
#     parallell_jobs : int
#         The number of jobs that will be run
#     """
#     times = []
#     with Pool(parallell_jobs) as pool:
#         Grid = _constructGrid(city, res, layers, size)
#         hashes = Grid.compute_dataset_hashes()
#         time_measurement = pool.map(
#             _computeSimilarities, [(hashes, measure) for _ in range(parallell_jobs)]
#         )
#         times.extend(time_measurement)
#     return times
