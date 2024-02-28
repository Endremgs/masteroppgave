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
        # print("Minimum latitude", self.min_lat)
        # print("Maximum latitude", self.max_lat)
        # print("Minimum longitude", self.min_lon)
        # print("Maximum longitude", self.max_lon)
        # print("Resolution", self.resolution)
        # print("Layers", self.layers)

        # Second, instantiate the indirect variables required for the scheme

        self.lat_len = td.calculate_trajectory_distance(
            [(self.min_lat, self.min_lon), (self.max_lat, self.min_lon)]
        )
        self.lon_len = td.calculate_trajectory_distance(
            [(self.min_lat, self.min_lon), (self.min_lat, self.max_lon)]
        )
        # print("Lat len", self.lat_len)
        # print("Lon len", self.lon_len)

        self.lat_res = td.get_latitude_difference(self.resolution)
        self.lon_res = td.get_longitude_difference(self.resolution, self.min_lat)
        # print("Lat res", self.lat_res)
        # print("Lon res", self.lon_res)

        self.distortion = self._compute_grid_distortion(self.resolution, self.layers)
        # print("Distortion", self.distortion)

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
        start_lat = min_lat + lat_distort
        start_lon = min_lon + lon_distort
        latitude_cells = [start_lat + i * lat_res for i in range(lat_count)]
        longitude_cells = [start_lon + j * lon_res for j in range(lon_count)]
        # latitude_cells = []
        # longitude_cells = []

        return latitude_cells, longitude_cells

    def _create_trajectory_hash(self, trajectory: list[list[float]]) -> list[list[str]]:
        """Creates a hash for one trajectory for all layers, returns it as a list of length layers with a list for each hashed layer"""

        # Snap trajectories to grid:
        hashes = []

        for layer in range(self.layers):
            # TODO: Perhaps the lists should be initialized here to utilise the distortion on the various layers
            distortion = self.distortion[layer]
            # print("Distortion", distortion)
            lat_distort = td.get_latitude_difference(distortion)
            lon_distort = td.get_longitude_difference(distortion, self.min_lat)
            # print("Lat distort", lat_distort)
            # print("Lon distort", lon_distort, "\n")

            lat_cells = int((self.max_lat - self.min_lat) // self.lat_res)
            lon_cells = int((self.max_lon - self.min_lon) // self.lon_res)
            # TODO: initialize two lists (lat, lon) where each element represents a coordinate of the start of a cell.
            # First element is min_lat, second element is min_lat + lat_res, third element is previous element + lat_res, etc.
            # Same goes for lon.
            self.latitude_cells, self.longitude_cells = self.generate_cell_coordinates(
                self.min_lat,
                self.min_lon,
                self.lat_res,
                self.lon_res,
                lat_cells,
                lon_cells,
                lat_distort,
                lon_distort,
            )
            # print("Latitudee cells", self.latitude_cells)
            # print("Longitude cells", self.longitude_cells)

            hash = []
            for coordinate in trajectory:
                lat, lon = coordinate
                hashed_coordinate = td.find_nearest_gridpoint(
                    (lat, lon), self.latitude_cells, self.longitude_cells
                )
                hash.append(hashed_coordinate)

            hashes.append(hash)
        # Then remove consecutive duplicates and return result:
        result = []

        for hash in hashes:
            result.append([el[0] for el in groupby(hash)])
        total_hashed_coordinates = 0
        for layer in result:
            total_hashed_coordinates += len(layer)
        # print("Total hashed coordinates", total_hashed_coordinates)
        return result, total_hashed_coordinates

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
        # print(self, "\n")
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        count = 0
        # Starting to hash the trajectories
        for key in trajectories:
            self.hashes[key], num_hashed_coordinates = self._create_trajectory_hash(
                trajectories[key]
            )
            count += num_hashed_coordinates
            # if count == 1:
            #     break
        # print("len(trajectories)", len(trajectories))
        num_avg_hashed_coordinates = count / len(trajectories)
        # print("Average number of hashed coordinates", num_avg_hashed_coordinates)

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
