"""TrajectoryData class for loading and further processing.

TrajectoryData class is used to load trajectory data from files.
Once loaded, it can be used to process the data for further analysis.
Trajectory data with idtracker.ai format and timestamp data are required.

Typical usage example:
    trajectory_data = TrajectoryData(trajectory_path='path/to/trajectory.csv',
                                      timestamp_path='path/to/timestamps.csv')
"""

import pandas as pd
import numpy as np


class Trajectory:
    """Trajectory class for loading and further processing."""

    def __init__(self, trajectory_data: dict, identity: int, fps: int):
        self.identity = identity
        self.fps = fps
        # Check the trajectory_data format
        self._check_format(trajectory_data)
        # Store the trajectory data
        self.trajectory_data = trajectory_data

    def _check_format(self, trajectory_data):
        # Check the type of trajectory_data
        if not isinstance(trajectory_data, dict):
            raise TypeError("trajectory_data must be a dictionary.")

        # Check if the trajectory_data has the required columns
        required_columns = ["x", "y"]
        for col in required_columns:
            if col not in trajectory_data:
                raise ValueError(f"Missing required column: {col}")
        if len(trajectory_data["x"]) != len(trajectory_data["y"]):
            raise ValueError("x and y have different len")
        for col in trajectory_data:
            if col not in required_columns:
                raise ValueError(f"Unexpected column: {col}")

        # Check if the trajectory_data has the correct data types
        # x: numeric, y: numeric
        if not all(isinstance(x, (int, float)) for x in trajectory_data["x"]):
            raise ValueError("Column 'x' must be numeric.")
        if not all(isinstance(x, (int, float)) for x in trajectory_data["y"]):
            raise ValueError("Column 'y' must be numeric.")

    def _calculate_linear_speed(self, x: list, y: list, t: list) -> float:
        """Calculate speed using linear regression."""
        coeffs = np.polyfit(t, x, 1), np.polyfit(t, y, 1)
        vx, vy = coeffs[0][0], coeffs[1][0]
        return np.sqrt(vx**2 + vy**2)

    def _calculate_mean_speed(self, x: list, y: list, t: list) -> float:
        """Calculate speed using mean of neighboring points."""
        dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
        speed = np.sqrt(dx**2 + dy**2) / dt
        return np.mean(speed)

    def _calculate_single_speed(self, x: list, y: list, t: list) -> float:
        """Calculate speed using first and last points."""
        dx, dy, dt = x[-1] - x[0], y[-1] - y[0], t[-1] - t[0]
        return np.sqrt(dx**2 + dy**2) / dt

    def to_speed(self, window_size: int = 5, mode: str = "linear") -> list:
        """Convert trajectory data to speed data.
        The speed is calculated within a window of size window_size.
        The speed is calculated using the following methods:
            - linear: linear regression of the window data. The speed is the slope of the line.
            - mean: mean of the speed calculated from the neighboring points.
            - single: the speed is calculated from the first and last points of the window.

        Args:
            window_size (int, optional): window size to calculate speed. Defaults to 5.
            mode (str, optional): mode to calculate speed, can be "linear", "mean", or "single".
            Defaults to "linear".

        Raises:
            ValueError: _window_size_ must be at least 2.
            ValueError: _mode_ must be "linear", "mean", or "single".
            ValueError: Less data points than window size.

        Returns:
            list: list of speed values with length n - window_size + 1
        """
        # Calculate the speed

        if window_size < 2:
            raise ValueError("Window size must be at least 2.")

        if mode not in ["linear", "mean", "single"]:
            raise ValueError("Mode must be 'linear', 'mean', or 'single'.")

        n = len(self)
        if n < window_size:
            raise ValueError("Less data points than window size.")

        speeds = []
        for i in range(n - window_size + 1):
            x = self.trajectory_data["x"][i : i + window_size]
            y = self.trajectory_data["y"][i : i + window_size]
            t = np.arange(window_size) / self.fps

            # If there are NaN values in the data window, skip this window
            if np.isnan(x).any() or np.isnan(y).any():
                continue

            speed = None
            if mode == "linear":
                speed = self._calculate_linear_speed(x, y, t)
            elif mode == "mean":
                speed = self._calculate_mean_speed(x, y, t)
            elif mode == "single":
                speed = self._calculate_single_speed(x, y, t)
            speeds.append(speed)
        return speeds

    def __len__(self):
        return len(self.trajectory_data["x"])


class TrajectoryCollection:
    """TrajectoryCollection class for loading and further processing."""

    def __init__(self, trajectory_path: str, fps: int, scale: float = 1.0):
        """Initialize the TrajectoryCollection class.

        Args:
            trajectory_path (str): path to the trajectory data file.
            fps (int): frames per second of the video.
            scale (float, optional): scale factor for the trajectory data.
            (can be used to convert from pixels to cm).
        """
        traj_data = pd.read_csv(trajectory_path, sep=",")

        self.identities = [int(col[1:]) for col in traj_data.columns if "x" in col]
        self.trajectories = {}
        for identity in self.identities:
            tmp = {}
            tmp["x"] = (traj_data[f"x{identity}"] * scale).tolist()
            tmp["y"] = (traj_data[f"y{identity}"] * scale).tolist()
            self.trajectories[identity] = Trajectory(tmp, identity, fps)
        self.fps = fps

    def to_speed(self, window_size: int = 5, mode: str = "linear") -> dict:
        """Convert trajectory data to speed data for all trajectories.
        The speed is calculated within a window of size window_size.
        The speed is calculated using the following methods:
            - linear: linear regression of the window data. The speed is the slope of the line.
            - mean: mean of the speed calculated from the neighboring points.
            - single: the speed is calculated from the first and last points of the window.

        Args:
            window_size (int, optional): window size to calculate speed. Defaults to 5.
            mode (str, optional): mode to calculate speed, can be "linear", "mean", or "single".
            Defaults to "linear".

        Returns:
            dict: dictionary with identity as key and list of speed values as value
        """
        speeds = {}
        for identity in self.identities:
            speeds[identity] = self.trajectories[identity].to_speed(window_size, mode)
        return speeds

    def to_distance(self, first_identity: int, second_identity: int) -> list:
        """Calculate distance between two trajectories.
        Args:
            first_identity (int): identity of the first trajectory.
            second_identity (int): identity of the second trajectory.
        Returns:
            list: list of distance values with length n
        """
        # TODO: implement
        return
