"""
File for a grid-based LSH scheme class in python.

Takes min/max lat/lon as argument -> Could potentially make this integrated in the future
"""

import random
import os, sys


from colorama import init as colorama_init, Fore, Style

from itertools import groupby

import timeit as ti
import time


currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from utils.helpers import trajectory_distance as td
from utils.helpers import alphabetical_number as an
from utils.helpers import metafile_handler as mfh
from utils.helpers import file_handler as fh
from .lsh_interface import LSHInterface


class GridLSH(LSHInterface):
    """
    A class for a grid-based LSH function for trajectory data
    """

    def __init__(
        self,
        name: str,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        resolution: float,
        layers: int,
        meta_file: str,
        data_path: str,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            The name of the grid
        min_lat : float
            The minimum latitude coordinate in the dataset
        max_lat : float
            The maximum latitude coordinate in the dataset
        min_lon : float
            The minimum longitude coordinate in the dataset
        max_lon : float
            The maximum longitude coordinate in the dataset
        resolution: float
            The preferred resolution for the grid (km)
        layers: int
            The number of layers that will be created
        meta_file: str
            A file containing the file-names that should be hashed through this class. Should be in the same folder as the data_path
        data_path: str
            The folder where the trajectories are stored
        """

        # First, initiating the direct variables
        self.name = name
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.resolution = resolution
        self.layers = layers
        self.meta_file = meta_file
        self.data_path = data_path

        # Second, instantiate the indirect variables required for the scheme
        self.lat_len = td.calculate_trajectory_distance(
            [(self.min_lat, self.min_lon), (self.max_lat, self.min_lon)]
        )
        self.lon_len = td.calculate_trajectory_distance(
            [(self.min_lat, self.min_lon), (self.min_lat, self.max_lon)]
        )

        self.lat_res = td.get_latitude_difference(self.resolution)
        self.lon_res = td.get_longitude_difference(self.resolution, self.min_lat)

        # List of length layers with the distortion for each layer
        self.distortion = self._compute_grid_distortion(self.resolution, self.layers)

        # Dict with the grid points coordinates for each layer for latitude and longitude as a 2D-list
        # Ex: {0: [[lat0, lat1, ...], [lon0, lon1, ...]] }
        self.grid = dict()
        # Calculate number of cells in the grid in both direction
        lat_cells = int((self.max_lat - self.min_lat) // self.lat_res)
        lon_cells = int((self.max_lon - self.min_lon) // self.lon_res)
        for layer in range(self.layers):
            # Find distortion specific for each layer and in decimal degrees to shift cells in the grid
            distortion = self.distortion[layer]
            lat_distort = td.get_latitude_difference(distortion)
            lon_distort = td.get_longitude_difference(distortion, self.min_lat)
            latitude_cells, longitude_cells = self.generate_cell_coordinates(
                self.min_lat,
                self.min_lon,
                self.lat_res,
                self.lon_res,
                lat_cells,
                lon_cells,
                lat_distort,
                lon_distort,
            )
            self.grid[layer] = [latitude_cells, longitude_cells]

        self.hashes = dict()

    def __str__(self) -> str:
        """Prints information about the grid"""
        lat_cells = int((self.max_lat - self.min_lat) // self.lat_res)
        lon_cells = int((self.max_lon - self.min_lon) // self.lon_res)

        return (
            f"Grid: {self.name}\nCovering: "
            f"{self.lat_len, self.lon_len} km \n"
            f"Resolution: {self.resolution} km \n"
            f"Distortion: {self.distortion} km \n"
            f"Dimensions: {lat_cells, lon_cells} cells"
        )

    # Defining some getters and setters

    def set_meta_file(self, meta_file: str) -> None:
        """Additional set method for the meta_file attribute"""
        self.meta_file = meta_file

    def _compute_grid_distortion(self, resolution: float, layers: int) -> list[float]:
        """Compute a random grid distortion off the resolution for the number of layers"""

        # Distortion should be a random float in the interval [0, resolution)
        distortion = [random.random() * resolution for x in range(layers)]
        return distortion

    def generate_cell_coordinates(
        self,
        min_lat,
        min_lon,
        lat_res,
        lon_res,
        lat_count,
        lon_count,
        lat_distort,
        lon_distort,
    ):
        # First cell set to minimum + distortion
        latitude_cells = [min_lat + lat_distort + i * lat_res for i in range(lat_count)]
        longitude_cells = [
            min_lon + lon_distort + j * lon_res for j in range(lon_count)
        ]
        return latitude_cells, longitude_cells

    def _create_trajectory_hash(self, trajectory: list[list[float]]) -> list[list[str]]:
        """Creates a hash for one trajectory for all layers, returns it as a list of length layers with a list for each hashed layer"""

        # Snap trajectories to grid:
        trajectory_hash = []

        for layer in range(self.layers):

            hashes_in_layer = []
            for coordinate in trajectory:
                lat, lon = coordinate
                hashed_coordinate = td.find_nearest_gridpoint(
                    (lat, lon), self.grid[layer][0], self.grid[layer][1]
                )
                hashes_in_layer.append(hashed_coordinate)

            trajectory_hash.append(hashes_in_layer)

        # Then remove consecutive duplicates and return result:
        result = []
        for hash in trajectory_hash:
            result.append([el[0] for el in groupby(hash)])
        return result

    def compute_dataset_hashes(self) -> dict[str, list]:
        """Method for computing the grid hashes for a given dataset and stores it in a dictionary

        Params
        ---
        meta_file_path : str
            The path to the dataset metafile

        Returns
        ---
        A dictionary containing the hashes
        """
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        # Starting to hash the trajectories
        for key in trajectories:
            self.hashes[key] = self._create_trajectory_hash(trajectories[key])

        return self.hashes

    def measure_hash_computation(self, repeat: int, number: int) -> list:
        """Method for measuring the computation time of the grid hashes. Does not change the object nor its attributes."""
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        hashes = dict()

        def compute_hashes(trajectories, hashes):
            for key in trajectories:
                hashes[key] = self._create_trajectory_hash(trajectories[key])
            return

        measures = ti.repeat(
            lambda: compute_hashes(trajectories, hashes),
            number=number,
            repeat=repeat,
            timer=time.process_time,
        )
        return (measures, len(hashes))

    def print_hashes(self):
        """Method that prints the created hashes"""

        if len(self.hashes) == 0:
            print("No hashes created yet")
        else:
            colorama_init()
            for key in self.hashes:
                print(
                    f"{Fore.GREEN}{key}{Style.RESET_ALL}:  {Fore.BLUE}{self.hashes[key][0]}{Style.RESET_ALL} "
                )
                for hash in self.hashes[key][1:]:
                    print(f"\t{Fore.BLUE}{hash}{Style.RESET_ALL}")


if __name__ == "__main__":
    Grid = GridLSH(
        "G1",
        min_lat=41.14,
        max_lat=41.19,
        min_lon=-8.66,
        max_lon=-8.57,
        resolution=0.25,
        meta_file="meta.txt",
        data_path="/data",
    )
    print(Grid)
