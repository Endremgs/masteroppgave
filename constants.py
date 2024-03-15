### COORDINATES ###

# Porto coordinates
P_MAX_LON = -8.57
P_MIN_LON = -8.66
P_MAX_LAT = 41.19
P_MIN_LAT = 41.14

# Rome coordinates
R_MAX_LON = 12.53
R_MIN_LON = 12.44
R_MAX_LAT = 41.93
R_MIN_LAT = 41.88

# TODO - Decide on coordinates to use for kolumbus
# Kolumbus coordinates
K_MAX_LON = 5.80
K_MIN_LON = 5.70
K_MAX_LAT = 59.10
K_MIN_LAT = 58.85

### DATA PATHS ###

# Porto data paths
PORTO_OUTPUT_FOLDER = "dataset/porto/output"
SIMILARITIES_OUTPUT_FOLDER_PORTO = "benchmarks/true_similarities/porto"

# Rome data paths
ROME_OUTPUT_FOLDER = "dataset/rome/output"
SIMILARITIES_OUTPUT_FOLDER_ROME = "benchmarks/true_similarities/rome"

# Kolumbus data paths
KOLUMBUS_OUTPUT_FOLDER = "dataset/kolumbus/output"
SIMILARITIES_OUTPUT_FOLDER_KOLUMBUS = "benchmarks/true_similarities/kolumbus"

NUMBER_OF_TRAJECTORIES = 100
COLOR_MAP = {
    0: "red",
    1: "blue",
    2: "green",
    3: "cyan",
    4: "magenta",
    5: "black",
    6: "darkorange",
    7: "darkviolet",
    8: "yellow",
}

COLOR_MAP_DATASET = {
    "porto": "blue",
    "rome": "green",
    "kolumbus": "red",
}
