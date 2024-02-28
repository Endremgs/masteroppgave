# Some helper functions for controling and saving the trajectories


# Save function
def save_current_trajectory(
    OUTPUT_FOLDER: str,
    file_name: str,
    trajectory: list[tuple[float]],
    trajectory_file_prefix: str = "R",
) -> None:
    with open(f"{OUTPUT_FOLDER}/{trajectory_file_prefix}_{file_name}.txt", "w") as file:
        counter = 0
        for coordinate in trajectory:
            counter += 1
            lat, lon = coordinate
            # if counter % 10 == 0:
            #     file.write("%s, %s\n" % (lat, lon))
            file.write("%s, %s\n" % (lat, lon))
        file.close()
        return


# Saving the hashes to files
def save_trajectory_hashes(output_folder: str, hashes: dict[str, list]) -> None:
    for key in hashes:
        with open(f"{output_folder}/{key}.txt", "w") as file:
            for hash in hashes[key]:
                file.write("%s\n" % hash)
            file.close()
