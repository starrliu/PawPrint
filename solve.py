from pawprint.data import TrajectoryCollection

SCALE_FROM_PIXEL_TO_CM = 0.259

# Load the data
tc = TrajectoryCollection(
    trajectory_path="data_1hour_10mice_20250423/gt.csv",
    fps=30,
    scale=SCALE_FROM_PIXEL_TO_CM,
)
# Get the speed distribution of each identity
speeds = tc.to_speed(window_size=5, mode="mean")