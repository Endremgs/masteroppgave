"""
File for a disk-based LSH scheme class in python.

Takes min/max lat/lon as argument -> Could potentially make this integrated in the future
"""

import random
import sys, os

from matplotlib import pyplot as plt
from matplotlib import lines

from colorama import init as colorama_init, Fore, Style
from scipy import spatial as sp

import timeit as ti
import time


currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.helpers import metafile_handler as mfh
from utils.helpers import file_handler as fh


from utils.helpers import trajectory_distance as td
from utils.helpers import alphabetical_number as an
from utils.helpers import metafile_handler as mfh
from utils.helpers import file_handler as fh
from .lsh_interface import LSHInterface


class Disk:
    """Class structure for disk-based LSH"""

    def __init__(self, name: int, lat: float, lon: float) -> None:
        self.name = name
        self.lat = lat
        self.lon = lon


class DiskLSH(LSHInterface):
    """
    A class for a disk-based LSH function for trajectory data
    """

    def __init__(
        self,
        name: str,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        disks: int,
        layers: int,
        diameter: float,
        meta_file: str,
        data_path: str,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            The name of the disk
        min_lat : float
            The minimum latitude coordinate in the dataset
        max_lat : float
            The maximum latitude coordinate in the dataset
        min_lon : float
            The minimum longitude coordinate in the dataset
        max_lon : float
            The maximum longitude coordinate in the dataset
        disks : int
            The number of disks at each layer
        layers: int
            The number of layers that will be created
        diameter: float
            The preferred diameter of a disk in the scheme (km)
        meta_file: str
            A file containing the file-names that should be hashed through this class. Should be in the same folder as the data_path
        data_path: str
            The folder where the trajectories are stored
        """

        # First, initializing the direct variables

        self.name = name
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.num_disks = disks
        self.layers = layers
        self.diameter = diameter
        self.meta_file = meta_file
        self.data_path = data_path

        # Second, instantiate the indirect variables required for the scheme:
        self.lat_len = td.calculate_trajectory_distance(
            [(self.min_lat, self.min_lon), (self.max_lat, self.min_lon)]
        )
        self.lon_len = td.calculate_trajectory_distance(
            [(self.min_lat, self.min_lon), (self.min_lat, self.max_lon)]
        )

        # Lastly, instantiate and compute the disks that will represent the hash function
        self.disks = self._instantiate_disks(self.layers, self.num_disks)
        self.hashes = dict()

        # Attributes for quad-tree-like structure
        self.split_lat = (self.max_lat + self.min_lat) / 2
        self.split_lon = (self.min_lon + self.max_lon) / 2
        self.disks_qt = self._instantiate_disks_qt(self.layers, self.num_disks)

        # Attributes for utilizing a KD-tree during disk against point matching
        self.KDTrees = self._instantiate_KD_tree(self.layers)

    def __str__(self) -> str:
        """Prints information about the disks"""

        return (
            f"Disk-scheme: {self.name} \n"
            f"Covering: {self.lat_len, self.lon_len} km \n"
            f"Diameter: {self.diameter} km\n"
            f"Layers: {self.layers} \n"
        )

    def set_meta_file(self, meta_file: str) -> None:
        """Resets the meta_file"""
        self.meta_file = meta_file

    def _instantiate_disks(self, layers: int, num_disks: int) -> dict[str, list]:
        """Instantiates the random disks that will be present at each layer"""
        disks = dict()
        for layer in range(layers):
            disks_list = []
            for disk in range(num_disks):
                lat = random.uniform(self.min_lat, self.max_lat)
                lon = random.uniform(self.min_lon, self.max_lon)
                disks_list.append([lat, lon])

            disks[layer] = disks_list
        return disks

    def _instantiate_disks_qt(
        self, layers: int, num_disks: int
    ) -> dict[str, list[list]]:
        """Instantiates the random disks that will be present at each layer with a quad-like structure"""
        disks = dict()
        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in range(layers):
            disks_list = [[], [], [], []]
            for i, disk in enumerate(self.disks[layer]):
                lat, lon = disk
                Dsk = Disk(i, lat, lon)

                # Must now determine which quadrant the Disk intersects with and add them into that one
                match self._get_quadrant(lat, lon, self.split_lat, self.split_lon):
                    case 0:  # First quadrant upper left
                        disks_list[0].append(Dsk)

                        intersect_right = (
                            td.get_euclidean_distance((lat, lon), (lat, self.split_lon))
                            <= radius
                        )
                        intersect_bottom = (
                            td.get_euclidean_distance((lat, lon), (self.split_lat, lon))
                            <= radius
                        )
                        intersect_bottom_right = (
                            td.get_euclidean_distance(
                                (lat, lon), (self.split_lat, self.split_lon)
                            )
                            <= radius
                        )

                        if intersect_right:
                            disks_list[1].append(
                                Dsk
                            )  # Also intersects with second quadrant
                        if intersect_bottom:
                            disks_list[2].append(Dsk)  # Third quadrant
                        if intersect_bottom_right:
                            disks_list[3].append(Dsk)  # Fourth quadrant

                    case 1:
                        # Second quadrant upper right
                        disks_list[1].append(Dsk)

                        intersect_left = (
                            td.get_euclidean_distance((lat, lon), (lat, self.split_lon))
                            <= radius
                        )
                        intersect_bottom = (
                            td.get_euclidean_distance((lat, lon), (self.split_lat, lon))
                            <= radius
                        )
                        intersect_bottom_left = (
                            td.get_euclidean_distance(
                                (lat, lon), (self.split_lat, self.split_lon)
                            )
                            <= radius
                        )

                        if intersect_left:
                            disks_list[0].append(
                                Dsk
                            )  # Also intersects with first quadrant
                        if intersect_bottom:
                            disks_list[3].append(Dsk)  # fourth quadrant
                        if intersect_bottom_left:
                            disks_list[2].append(Dsk)  # third quadrant

                    case 2:
                        # Third quadrant bottom left
                        disks_list[2].append(Dsk)

                        intersect_right = (
                            td.get_euclidean_distance((lat, lon), (lat, self.split_lon))
                            <= radius
                        )
                        intersect_top = (
                            td.get_euclidean_distance((lat, lon), (self.split_lat, lon))
                            <= radius
                        )
                        intersect_top_right = (
                            td.get_euclidean_distance(
                                (lat, lon), (self.split_lat, self.split_lon)
                            )
                            <= radius
                        )

                        if intersect_right:
                            disks_list[3].append(
                                Dsk
                            )  # Also intersects with fourth quadrant
                        if intersect_top:
                            disks_list[0].append(Dsk)  # first quadrant
                        if intersect_top_right:
                            disks_list[1].append(Dsk)  # second quadrant

                    case 3:
                        # Fourth quadrant bottom right
                        disks_list[3].append(Dsk)

                        intersect_left = (
                            td.get_euclidean_distance((lat, lon), (lat, self.split_lon))
                            <= radius
                        )
                        intersect_top = (
                            td.get_euclidean_distance((lat, lon), (self.split_lat, lon))
                            <= radius
                        )
                        intersect_top_left = (
                            td.get_euclidean_distance(
                                (lat, lon), (self.split_lat, self.split_lon)
                            )
                            <= radius
                        )

                        if intersect_left:
                            disks_list[2].append(
                                Dsk
                            )  # Also intersects with third quadrant
                        if intersect_top:
                            disks_list[1].append(Dsk)  # second quadrant
                        if intersect_top_left:
                            disks_list[0].append(Dsk)  # first quadrant
                    case _:
                        raise Exception("Somethin went wrong during init of quad-disks")
            disks[layer] = disks_list

        return disks

    def _instantiate_KD_tree(self, layers: int) -> dict[str, sp.cKDTree]:
        """Instantiates the random disks that will be present at each layer with a quad-like structure

        Utilizes the originial disks and create a tree structure for each layer
        """
        if not layers:
            raise Exception("Cannot instantiate KD-tree on empty disk map")

        trees = dict()
        for layer in range(layers):
            tree = sp.KDTree(self.disks[layer])
            trees[layer] = tree

        return trees

    def _create_trajectory_hash(self, trajectory: list[list[float]]) -> list[list[str]]:
        """Creates a hash for one trajectory for all layers. Returns it as a alist of length layers with a list of hashed point for each layer"""
        hashes = []
        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in self.disks.keys():
            hash = []  # The created hash
            within = []  # The disks that the trajectory are currently within
            disks = self.disks[layer]
            for coordinate in trajectory:
                print(trajectory)
                lat, lon = coordinate

                # If next point no longer in disk: Remove from within list
                for disk in within:
                    if td.get_euclidean_distance([lat, lon], disk) > radius:
                        within.remove(disk)

                # If next point inside disk: Append to hash if not still within disk
                # Can speed up substantially by applying a tree-structure here, naive implementation for now:
                for i, disk in enumerate(disks):
                    if td.get_euclidean_distance([lat, lon], disk) <= radius:
                        if disk not in within:
                            within.append(disk)
                            diskHash = an.get_alphabetical_value(i)
                            hash.append(diskHash)
            hashes.append(hash)
        return hashes

    def _create_trajectory_hash_numerical(
        self, trajectory: list[list[float]]
    ) -> list[list[str]]:
        """Creates a hash for one trajectory for all layers. Returns it as a alist of length layers with a list of hashed point for each layer"""

        hashes = []
        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in self.disks.keys():
            hash = []  # The created hash
            within = []  # The disks that the trajectory are currently within
            disks = self.disks[layer]
            for coordinate in trajectory:
                lat, lon = coordinate

                # If next point no longer in disk: Remove from within list
                for disk in within:
                    if td.get_euclidean_distance([lat, lon], disk) > radius:
                        within.remove(disk)

                # If next point inside disk: Append to hash if not still within disk
                # Can speed up substantially by applying a tree-structure here, naive implementation for now:
                for i, disk in enumerate(disks):
                    if td.get_euclidean_distance([lat, lon], disk) <= radius:
                        if disk not in within:
                            within.append(disk)
                            diskHash = disk
                            hash.append(diskHash)
            hashes.append(hash)
        return hashes

    def _create_trajectory_hash_with_quad_tree(
        self, trajectory: list[list[float]]
    ) -> list[list[str]]:
        """Same as above, but utilises a quad-tree-like structure for faster computation"""
        hashes = []
        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in self.disks_qt.keys():
            hash = []
            within = []
            disks = self.disks_qt[layer]
            for coordinate in trajectory:
                lat, lon = coordinate
                quadrant = self._get_quadrant(lat, lon, self.split_lat, self.split_lon)

                for disk in within:
                    if (
                        td.get_euclidean_distance([lat, lon], [disk.lat, disk.lon])
                        > radius
                    ):
                        within.remove(disk)

                for disk in disks[quadrant]:
                    if (
                        td.get_euclidean_distance([lat, lon], [disk.lat, disk.lon])
                        <= radius
                    ):
                        if disk not in within:
                            within.append(disk)
                            diskHash = an.get_alphabetical_value(disk.name)
                            hash.append(diskHash)
            hashes.append(hash)
        return hashes

    def _create_trajectory_hash_with_quad_tree_numerical(
        self, trajectory: list[list[float]]
    ) -> list[list[str]]:
        """Same as above, but utilises a quad-tree-like structure for faster computation -numerical"""
        hashes = []
        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in self.disks_qt.keys():
            hash = []
            within = []
            disks = self.disks_qt[layer]
            for coordinate in trajectory:
                lat, lon = coordinate
                quadrant = self._get_quadrant(lat, lon, self.split_lat, self.split_lon)

                for disk in within:
                    if (
                        td.get_euclidean_distance([lat, lon], [disk.lat, disk.lon])
                        > radius
                    ):
                        within.remove(disk)

                for disk in disks[quadrant]:
                    if (
                        td.get_euclidean_distance([lat, lon], [disk.lat, disk.lon])
                        <= radius
                    ):
                        if disk not in within:
                            within.append(disk)
                            diskHash = disk
                            hash.append(diskHash)
            hashes.append(hash)
        return hashes

    def _create_trajectory_hash_with_KD_tree(
        self, trajectory: list[list[float]]
    ) -> list[list[str]]:
        """Same as above, but utilises a KD-tree-like for faster computation"""
        hashes = []
        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in self.disks.keys():
            hash = []
            within = []
            tree = self.KDTrees[layer]
            for coordinate in trajectory:
                lat, lon = coordinate
                for disk in within:
                    dsklat, dsklon = self.disks[layer][disk]
                    # print(dsklat, dsklon)
                    if td.get_euclidean_distance([lat, lon], [dsklat, dsklon]) > radius:
                        # print("Removing disk")
                        # print(dsklat, dsklon)
                        within.remove(disk)

                # Gives disk index
                intersect_disks = tree.query_ball_point([lat, lon], radius)
                for disk in intersect_disks:
                    if disk not in within:
                        within.append(disk)
                        diskHash = an.get_alphabetical_value(disk)
                        hash.append(diskHash)
            hashes.append(hash)
        return hashes

    def _create_trajectory_hash_with_KD_tree_numerical(
        self, trajectory: list[list[float]]
    ) -> list[list[str]]:
        """Same as above, but creates hashes that consists of the disks coordinates"""
        hashes = []
        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in self.disks.keys():
            hash = []
            within = []
            tree = self.KDTrees[layer]
            for coordinate in trajectory:
                lat, lon = coordinate
                for disk in within:
                    dsklat, dsklon = self.disks[layer][disk]
                    if td.get_euclidean_distance([lat, lon], [dsklat, dsklon]) > radius:
                        within.remove(disk)

                # Gives disk index
                intersect_disks = tree.query_ball_point([lat, lon], radius)
                for disk in intersect_disks:
                    if disk not in within:
                        within.append(disk)
                        diskHash = tree.data[disk]
                        hash.append(diskHash)
            hashes.append(hash)
        return hashes

    def compute_dataset_hashes(self) -> dict[str, list]:
        """Method for computing the disk hashes for a given dataset. Stores the hashes in a dictionary

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

        # Beginning to hash trajectories
        for key in trajectories:
            self.hashes[key] = self._create_trajectory_hash(trajectories[key])

        return self.hashes

    def compute_dataset_hashes_with_quad_tree(self) -> dict[str, list]:
        """Same as above, but utilises a quad-tree-like-structure for faster computation"""
        # TODO
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)

        # Beginning to hash trajectories
        hashes = dict()
        for key in trajectories:
            hashes[key] = self._create_trajectory_hash(trajectories[key])

        return hashes

    def compute_dataset_hashes_with_KD_tree(self) -> dict[str, list]:
        """Same as above, but utilises a KD-tree for faster computation"""
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)

        # Beginning to hash trajectories
        hashes = dict()
        for key in trajectories:
            hashes[key] = self._create_trajectory_hash_with_KD_tree(trajectories[key])

        return hashes

    def compute_dataset_hashes_with_KD_tree_numerical(self) -> dict[str, list]:
        """Same as aboce, but returns the hashes as the disks center coordinates"""
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)

        # Beginning to hash trajectories
        hashes = dict()
        for key in trajectories:
            hashes[key] = self._create_trajectory_hash_with_KD_tree_numerical(
                trajectories[key]
            )

        return hashes

    def measure_hash_computation(self, number: int, repeat: int) -> list[list, int]:
        """Method for measuring the computation time of the disk hashes. Does not change the object nor its attributes."""
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        hashes = dict()

        def compute_hashes(trajectories: dict, hashes: dict):
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

    def measure_hash_computation_numerical(
        self, number: int, repeat: int
    ) -> list[list, int]:
        """Method for measuring the computation time of the disk hashes. Does not change the object nor its attributes."""
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        hashes = dict()

        def compute_hashes(trajectories: dict, hashes: dict):
            for key in trajectories:
                hashes[key] = self._create_trajectory_hash_numerical(trajectories[key])
            return

        measures = ti.repeat(
            lambda: compute_hashes(trajectories, hashes),
            number=number,
            repeat=repeat,
            timer=time.process_time,
        )
        return (measures, len(hashes))

    def measure_hash_computation_with_quad_tree(self, number: int, repeat: int) -> None:
        """Same as above, but using quad-structure for speed improvement"""
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        hashes = dict()

        def compute_hashes(trajectories: dict, hashes: dict):
            for key in trajectories:
                hashes[key] = self._create_trajectory_hash_with_quad_tree(
                    trajectories[key]
                )
            return

        measures = ti.repeat(
            lambda: compute_hashes(trajectories, hashes),
            number=number,
            repeat=repeat,
            timer=time.process_time,
        )
        return (measures, len(hashes))

    def measure_hash_computation_with_quad_tree_numerical(
        self, number: int, repeat: int
    ) -> None:
        """Same as above, but using quad-structure for speed improvement"""
        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        hashes = dict()

        def compute_hashes(trajectories: dict, hashes: dict):
            for key in trajectories:
                hashes[key] = self._create_trajectory_hash_with_quad_tree_numerical(
                    trajectories[key]
                )
            return

        measures = ti.repeat(
            lambda: compute_hashes(trajectories, hashes),
            number=number,
            repeat=repeat,
            timer=time.process_time,
        )
        return (measures, len(hashes))

    def measure_hash_computation_with_KD_tree(self, number: int, repeat: int) -> None:
        """Same as above, but using KD-tree for speed improvement"""

        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        hashes = dict()

        def compute_hashes(trajectories: dict, hashes: dict):
            for key in trajectories:
                hashes[key] = self._create_trajectory_hash_with_KD_tree(
                    trajectories[key]
                )
            return

        measures = ti.repeat(
            lambda: compute_hashes(trajectories, hashes),
            number=number,
            repeat=repeat,
            timer=time.process_time,
        )

        return (measures, len(hashes))

    def measure_hash_computation_with_KD_tree_numerical(
        self, number: int, repeat: int
    ) -> None:
        """Same as above, but using KD-tree for speed improvement"""

        files = mfh.read_meta_file(self.meta_file)
        trajectories = fh.load_trajectory_files(files, self.data_path)
        hashes = dict()

        def compute_hashes(trajectories: dict, hashes: dict):
            for key in trajectories:
                hashes[key] = self._create_trajectory_hash_with_KD_tree_numerical(
                    trajectories[key]
                )
            return

        measures = ti.repeat(
            lambda: compute_hashes(trajectories, hashes),
            number=number,
            repeat=repeat,
            timer=time.process_time,
        )

        return (measures, len(hashes))

    def print_hashes(self) -> None:
        """Printing the hashes"""
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

    def print_disks(self):
        print("Number of layers", len(self.disks))
        for layer in self.disks:
            print("Layer: ", layer)
            print("Number of disks in layer: ", len(self.disks[layer]))
            for disk in self.disks[layer]:
                print(f"Disk \t{disk}")

    def visualise_hashes(self, trajectory=None) -> None:
        """Method to visualise hashes"""

        radius = td.get_latitude_difference(self.diameter / 2)
        for layer in self.disks.keys():
            plt.rcParams["figure.autolayout"] = True

            fig, ax = plt.subplots()
            for disk in self.disks[layer]:
                x, y = disk

                ax.add_patch(plt.Circle((y, x), radius, fill=False))

            if trajectory:
                print("Trajectory")
                lats, lons = list(zip(*trajectory))
                ax.add_line(lines.Line2D(lons, lats))

            plt.ylim(self.min_lat - 0.02, self.max_lat + 0.02)
            plt.xlim(self.min_lon - 0.02, self.max_lon + 0.02)
            plt.xlim()
            plt.show()

    def _get_quadrant(self, lat: float, lon: float, split_lat, split_lon):
        """Helper function that returns the corresponding quadrant that a point is in"""
        if lat >= split_lat and lon <= split_lon:
            return 0  # First quadrant upper left
        elif lat >= split_lat and lon > split_lon:
            return 1  # Second quadrant upper right
        elif lat < split_lat and lon <= split_lon:
            return 2  # Third quadrant bottom left
        elif lat < split_lat and lon > split_lon:
            return 3  # Fourth quadrant bottom right
        else:
            raise Exception(
                "Somethin went wrong during quadrant fetching (Should be impossible)"
            )


if __name__ == "__main__":
    DiskLSH = DiskLSH("Disk1", 41.88, 41.93, 12.44, 12.53, 50, 4, 2, "meta.txt", "data")
