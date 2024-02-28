import numpy as np
import pandas as pd
import os

from grid_similarity import generate_grid_hash_similarity

# from experiments.disk_similarity import generate_disk_hash_similarity
from constants import SIMILARITIES_OUTPUT_FOLDER_PORTO, SIMILARITIES_OUTPUT_FOLDER_ROME


def compute_correlation_similarity(city: str, scheme: str, runs: int):
    """Standalone method that computes the correlation with true similarity values from 10 runs"""

    # Defining helper functions:

    def _mirrorDiagonal(M: np.ndarray) -> np.ndarray:
        """Flips and mirrors a two-dimenional np.array"""
        return M.values + np.rot90(np.fliplr(M.values))

    porto_dtw = _mirrorDiagonal(
        pd.read_csv(
            os.path.abspath(
                f"../{SIMILARITIES_OUTPUT_FOLDER_PORTO}/porto-dtw-50.csv"
            ),
            index_col=0,
        )
    ).flatten()
    porto_fre = _mirrorDiagonal(
        pd.read_csv(
            os.path.abspath(
                f"../{SIMILARITIES_OUTPUT_FOLDER_PORTO}/porto-frechet-50.csv"
            ),
            index_col=0,
        )
    ).flatten()
    # rome_dtw = _mirrorDiagonal(pd.read_csv(os.path.abspath("../benchmarks/similarities/rome-dtw.csv"), index_col=0)).flatten()
    # rome_fre = _mirrorDiagonal(pd.read_csv(os.path.abspath("../benchmarks/similarities/rome-frechet.csv"), index_col=0)).flatten()

    #    similarities = {
    #        "grid" : {
    #            "porto" : generate_grid_hash_similarity("porto", 1.6, 5),
    #            "rome" : generate_grid_hash_similarity("rome", 1.2, 4)
    #        },
    #        "disk" : {
    #            "porto" : generate_disk_hash_similarity("porto", 2.2, 4, 60),
    #            "rome" : generate_disk_hash_similarity("rome", 1.6, 5, 50)
    #        }
    #    }

    true_sims = {
        "porto": {"dtw": porto_dtw, "fre": porto_fre},
        # "rome": {"dtw": rome_dtw, "fre": rome_fre},
    }

    # Computing similarities n times:

    # hash_sims = _mirrorDiagonal(similarities[city.lower()][scheme.lower()]).flatten()

    correlation_dtw = []
    correlation_fre = []

    for run in range(runs):
        print("Run :", run)
        hash_sims = None
        if city.lower() == "porto" and scheme.lower() == "grid":
            hash_sims = generate_grid_hash_similarity("porto", 1.6, 5)
        # elif city.lower() == "porto" and scheme.lower() == "disk":
        #     hash_sims = generate_disk_hash_similarity("porto", 2.2, 4, 60)
        # elif city.lower() == "rome" and scheme.lower() == "grid":
        #     hash_sims = generate_grid_hash_similarity("rome", 1.2, 4)
        # else:
        #     hash_sims = generate_disk_hash_similarity("rome", 1.6, 5, 50)

        h_sims = _mirrorDiagonal(hash_sims).flatten()
        correlation_dtw.append(np.corrcoef(h_sims, true_sims[city]["dtw"])[0][1])
        correlation_fre.append(np.corrcoef(h_sims, true_sims[city]["fre"])[0][1])
        print("DTW: ", np.corrcoef(h_sims, true_sims[city]["dtw"])[0][1])
        print("Frechet: ", np.corrcoef(h_sims, true_sims[city]["fre"])[0][1])

    print(city, scheme, ": (min, max, avg, std)")
    print(
        "DTW:",
        min(correlation_dtw),
        max(correlation_dtw),
        np.average(correlation_dtw),
        np.std(correlation_dtw),
    )
    print(
        "FRE:",
        min(correlation_fre),
        max(correlation_fre),
        np.average(correlation_fre),
        np.std(correlation_fre),
    )
