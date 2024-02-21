import pandas as pd
from utils.similarity_measures.hashed_frechet import (
    cy_frechet_hashes,
    cy_frechet_hashes_pool,
)
from utils.similarity_measures.hashed_dtw import cy_dtw_hashes, cy_dtw_hashes_pool


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
    parallel: bool = False,
) -> pd.DataFrame:
    if scheme == "disk":
        hashes = transform_np_numerical_disk_hashes_to_non_np(hashes)
    # NOTE - if scheme =="grid" then the hashes are already in the correct format, I.E non numpy
    if measure == "dtw":
        if parallel:
            return cy_dtw_hashes_pool(hashes)
        else:
            return cy_dtw_hashes(hashes)
    elif measure == "frechet":
        if parallel:
            return cy_frechet_hashes_pool(hashes)
        else:
            return cy_frechet_hashes(hashes)


def disk_coordinates(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """The hashed disk coordinates"""
    hashed_coordinates = transform_np_numerical_disk_hashes_to_non_np(hashes)
    return hashed_coordinates
