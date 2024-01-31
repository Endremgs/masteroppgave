# Some helper functions for controling and saving the trajectories


# Save function
def save_current_trajectory(
    OUTPUT_FOLDER: str,
    file_name: str,
    trajectory: list[tuple[float]],
    trajectory_file_prefix: str = "R",
) -> None:
    with open(f"{OUTPUT_FOLDER}/{trajectory_file_prefix}_{file_name}.txt", "w") as file:
        for coordinate in trajectory:
            lat, lon = coordinate
            file.write("%s, %s\n" % (lat, lon))
        file.close()
        return
