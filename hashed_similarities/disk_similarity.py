"""
Sheet that will be used to measure the time needed to generate similarities of the disk hashes

Will run N processess in parallell to measure time efficiency
"""

import time
import timeit as ti
import pandas as pd
import os, sys

currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from schemes.lsh_disk import DiskLSH

from utils.similarity_measures.distance import compute_hash_similarity, disk_coordinates

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


def _constructDisk(
    city: str, diameter: float, layers: int, disks: int, size: int
) -> DiskLSH:
    """Constructs a disk hash object over the given city"""
    if city.lower() == "porto":
        return DiskLSH(
            f"DP_{layers}-{'{:.2f}'.format(diameter)}",
            P_MIN_LAT,
            P_MAX_LAT,
            P_MIN_LON,
            P_MAX_LON,
            disks,
            layers,
            diameter,
            PORTO_META(size),
            PORTO_CHOSEN_DATA,
        )
    elif city.lower() == "rome":
        return DiskLSH(
            f"GDR_{layers}-{'{:.2f}'.format(diameter)}",
            R_MIN_LAT,
            R_MAX_LAT,
            R_MIN_LON,
            R_MAX_LON,
            disks,
            layers,
            diameter,
            ROME_META(size),
            ROME_CHOSEN_DATA,
        )
    elif city.lower() == "kolumbus":
        return DiskLSH(
            f"GDK_{layers}-{'{:.2f}'.format(diameter)}",
            K_MIN_LAT,
            K_MAX_LAT,
            K_MIN_LON,
            K_MAX_LON,
            disks,
            layers,
            diameter,
            KOLUMBUS_META(size),
            KOLUMBUS_CHOSEN_DATA,
        )
    else:
        raise ValueError("City argument must be either porto or rome")


def generate_disk_hash_similarity(
    city: str,
    diameter: float,
    layers: int,
    disks: int,
    measure: str = "dtw",
    size: int = 50,
) -> pd.DataFrame:
    """Generates the full disk hash similarities and saves it as a dataframe"""

    Disk = _constructDisk(city, diameter, layers, disks, size)
    hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
    similarities = compute_hash_similarity(
        hashes=hashes, scheme="disk", measure=measure, parallel=True
    )

    return similarities


def generate_disk_hash_similarity_coordinates(
    city: str,
    diameter: float,
    layers: int,
    disks: int,
    measure: str = "dtw",
    size: int = 50,
) -> any:
    """Generates the full disk hash similarities and saves it as a dataframe"""

    Disk = _constructDisk(city, diameter, layers, disks, size)
    hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
    # Disk.print_disks()
    hashed_coordinates = disk_coordinates(hashes)
    all_disks_coordinates = Disk.disks

    return hashed_coordinates, all_disks_coordinates


# TODO - measure computation time
# def _computeSimilarities(args) -> list:
#     hashes, measure = args
#     elapsed_time = ti.timeit(
#         lambda: MEASURE[measure](hashes), number=1, timer=time.process_time
#     )
#     return elapsed_time


# def measure_disk_hash_similarity_computation_time(
#     city: str,
#     size: int,
#     diameter: float,
#     layers: int,
#     disks: int,
#     hashtype: str,
#     measure: str = "dtw",
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
#     diameter : float
#         The disks diameter
#     layers : int
#         The number of layers that will be used
#     disks : int
#         The number of disks that will be used at each layer
#     hashtype : str
#         "normal" | "quadrants" | "kd"
#     measure : str (Either "ed" or "dtw" - "dtw" default)
#         The measure that will be used for computation
#     parallell_jobs : int
#         The number of jobs that will be run
#     """

#     execution_times = []

#     with Pool(parallell_jobs) as pool:
#         Disk = _constructDisk(city, diameter, layers, disks, size)

#         if measure == "dtw" and hashtype == "kd":
#             hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
#         elif measure == "ed" and hashtype == "normal":
#             hashes = Disk.compute_dataset_hashes()
#         elif measure == "ed" and hashtype == "quadrants":
#             hashes = Disk.compute_dataset_hashes_with_quad_tree()
#         elif measure == "ed" and hashtype == "kd":
#             hashes = Disk.compute_dataset_hashes_with_KD_tree()
#         else:
#             raise ValueError(
#                 "Cannot construct disk hashes as input parameters are uncertain"
#             )

#         time_measurement = pool.map(
#             _computeSimilarities, [(hashes, measure) for _ in range(parallell_jobs)]
#         )
#         execution_times.extend(time_measurement)

#     return execution_times
