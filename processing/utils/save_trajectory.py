# Some helper functions for controling and saving the trajectories


# Save function
def save_current_trajectory(
    OUTPUT_FOLDER: str,
    file_name: str,
    trajectory: list[tuple[float]],
    split_coordinate: bool,
) -> None:
    with open(f"{OUTPUT_FOLDER}/R_{file_name}.txt", "w") as file:
        for coordinate in trajectory:
            lat, lon = coordinate.split(",") if split_coordinate else coordinate
            file.write("%s, %s\n" % (lat, lon))
        file.close()
        return
