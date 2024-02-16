import numpy as np

from utils.helpers import trajectory_distance as td


def _get_num_value(string: str, base: str):
    c1, c2 = [*string]
    return (ord(c1) - ord(base)) * 26 + ord(c2) - ord(base)


# TODO: Confirm that this is Manhattan, and not Hamming distance as previously stated
def _get_alphabetical_grid_distance(hash_x, hash_y):
    """
    Calculates and returns the Manhattan distance between the hashes from the grids
    """

    if len(hash_x) != len(hash_y):
        raise ValueError("Hashes have wrong length")

    hx_1 = hash_x[:2]
    hx_2 = hash_x[2:]
    hy_1 = hash_y[:2]
    hy_2 = hash_y[2:]

    x = abs(_get_num_value(hx_1, "A") - _get_num_value(hy_1, "A"))
    y = abs(_get_num_value(hx_2, "a") - _get_num_value(hy_2, "a"))

    return x + y


# TODO: Confirm that this is DTW, and not edit distance penalty, and change docstring accordingly
def dtw_manhattan(hash_x: np.ndarray, hash_y: np.ndarray) -> float:
    """
    Computes the edit distance with penalty between two trajectory hashes (Grid | Disk hash)\n
    Runs in layers x O(n^2) time, where n is the length of one hash

    Param
    ---
    hash_x : np array list(list(str))
        The full hash of trajectory x
    hash_y : np array list(list(str))
        The full hash of trajectory y

    Returns
    ---
    Their combined edit distance (sum of number of edits divided by longest sequence) and total number of edits (float, float)

    Notes
    ---
    Rewritten from https://github.com/bguillouet/traj-dist/blob/master/traj_dist/pydist/edr.py
    """

    x_len = len(hash_x)
    y_len = len(hash_y)

    if x_len != y_len:
        raise ValueError(
            "Number of layers are different for the hashes. Unable to compute edit distance"
        )

    cost = 0
    c = 0

    for layer in range(x_len):
        X = hash_x[layer]
        Y = hash_y[layer]
        X_len = len(X)
        Y_len = len(Y)
        M = np.zeros((X_len + 1, Y_len + 1))

        # Edge case if one of hashes is empty
        if (X_len == 0 or Y_len == 0) and X_len != Y_len:
            cost += 1
            c += max(X_len, Y_len)
            continue
        # Edge case if both hashes are empty
        if X_len == 0 and Y_len == 0:
            cost += 0

            continue

        for i in range(1, X_len + 1):
            for j in range(1, Y_len + 1):
                s = _get_alphabetical_grid_distance(X[i - 1], Y[j - 1])
                # d = _get_alphabetical_grid_distance(X[i], Y[j-1])
                # r = _get_alphabetical_grid_distance(X[i-1], Y[j])
                if i == 1:
                    M[i - 1][j - 1] = s
                elif j == 1:
                    M[i - 1][j - 1] = s

                # if X[i-1] == Y[j-1]: subcost = 0
                # else: subcost = s

                M[i, j] = s + min(M[i][j - 1], M[i - 1][j], M[i - 1][j - 1])
        # print(M)
        cost += float(M[X_len][Y_len]) / max([X_len, Y_len])
        c += float(M[X_len][Y_len])

    return cost, c


# DTW for hashes. Used by disk hashes
# TODO: change docstring accordingly
def dtw_euclidean(hash_x: np.ndarray, hash_y: np.ndarray) -> float:
    """
    Computes the edit distance with penalty between two trajectory hashes (Grid | Disk hash)\n
    Runs in layers x O(n^2) time, where n is the length of one hash

    Param
    ---
    hash_x : np array list(list(float))
        The full hash of trajectory x as layers of list of coordinates
    hash_y : np array list(list(str))
        The full hash of trajectory y as layers with list of coordinates

    Returns
    ---
    Their dtw

    Notes
    ---
    Rewritten from https://github.com/bguillouet/traj-dist/blob/master/traj_dist/pydist/edr.py
    """

    x_len = len(hash_x)
    y_len = len(hash_y)

    if x_len != y_len:
        raise ValueError(
            "Number of layers are different for the hashes. Unable to compute edit distance"
        )

    cost = 0

    for layer in range(x_len):
        X = hash_x[layer]
        Y = hash_y[layer]
        X_len = len(X)
        Y_len = len(Y)

        if X_len == 0 or Y_len == 0:
            cost += 0.5
            continue

        M = np.zeros((X_len + 1, Y_len + 1))
        M[1:, 0] = float("inf")
        M[0, 1:] = float("inf")

        for i in range(1, X_len + 1):
            for j in range(1, Y_len + 1):
                s = td.get_euclidean_distance(X[i - 1], Y[j - 1])

                M[i, j] = s + min(M[i][j - 1], M[i - 1][j], M[i - 1][j - 1])
        # print(M)
        cost += float(M[X_len][Y_len])

    return cost


if __name__ == "__main__":
    assert _get_num_value("ZZ", "A") == 675
    assert _get_num_value("BJ", "A") == 35
    assert _get_num_value("AB", "A") == 1
    assert _get_num_value("AA", "A") == 0
    assert _get_num_value("zz", "a") == 675
    assert _get_num_value("bj", "a") == 35
    assert _get_num_value("ab", "a") == 1
    assert _get_num_value("aa", "a") == 0

    assert _get_alphabetical_grid_distance("ACad", "ACad") == 0
    assert _get_alphabetical_grid_distance("ABan", "ABam") == 1
    assert _get_alphabetical_grid_distance("ACan", "ABam") == 2
    assert _get_alphabetical_grid_distance("ABan", "BCai") == 32

    print("All tests passed")
