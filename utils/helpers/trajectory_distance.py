""" Sheet containing distance methods related to trajectories and their hashes """


from haversine import haversine, Unit
import math

# Helper function to compute distance between a list of coordinates (Trajectory distance)
# Haversine distance used


def calculate_trajectory_distance(positions: list[tuple[float]]) -> float:
    """
    Calculate the trajectory distance for a trajectory

    :param: List of coordinates (lat, lon)

    :return: Float (km) -> Combined distance between all pairs of points in km
    """
    distance = 0
    for i in range(1, len(positions)):
        from_location = positions[i - 1]
        to_location = positions[i]

        distance += haversine(from_location, to_location, unit=Unit.KILOMETERS)
    return distance


def get_latitude_difference(distance: float) -> float:
    """
    Calculate the difference in latitude decimal degrees corresponding to a given distance

    Param
    ---
    distance : float (km)

    Returns
    ---
    latitude decimal degree difference as float
    """
    return distance / 110.574


def get_longitude_difference(distance: float, latitude: float) -> float:
    """
    Calculate the difference in longitude decimal degrees corresponding to a given distance

    Param
    ---
    distance : float (km)
        The distance for which the computation should be made
    latitude : float
        The latitude at which the computation is to be made
    Returns
    ---
    longitide decimal degree difference as float
    """
    return distance / (111.320 * math.cos(math.radians(latitude)))


def get_euclidean_distance(pos1: list[float], pos2: list[float]) -> float:
    """
    Calculcate the euclidean distance between any two points
    **Using this method for geo-coordinates is a simplification, and will result in some errors compared to the true distance

    Param
    ---
    pos1 : list(float) [lat, lon]
        The start coordinate
    pos2 : list(float) [lat, lon]
        The end coordinate
    """
    distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    return distance
