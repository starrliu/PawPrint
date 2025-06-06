"""Test the data module."""

import pytest
import pandas as pd

from pawprint.data import Trajectory
from pawprint.data import TrajectoryCollection


class TestTrajectory:
    """Test the Trajectory class."""

    @pytest.fixture
    def trajectory_data(self):
        """Fixture for trajectory data."""
        # Create a sample trajectory data
        data = {
            "x": list(range(10)),
            "y": list(range(10)),
        }
        return data

    def test_init(self, trajectory_data):
        """Test the initialization of the Trajectory class."""
        # Create a sample trajectory data
        Trajectory(trajectory_data, 1, 30)

    def test_speed(self, trajectory_data):
        """Test the speed calculation."""
        traj_df = trajectory_data
        traj_df["x"][2] = float("nan")
        traj_df["y"][2] = float("nan")
        traj = Trajectory(traj_df, 1, 30)

        speed_gt = 30 * 2 ** 0.5  # 30 fps, 1 second, 2 units of distance
        # Test linear speed
        speed = traj.to_speed(3, "linear")
        print(speed)
        assert len(speed) == 5
        assert all(abs(s - speed_gt) < 1e-5 for s in speed)
        # Test mean speed
        speed = traj.to_speed(3, "mean")
        assert len(speed) == 5
        assert all(abs(s - speed_gt) < 1e-5 for s in speed)
        # Test single speed
        speed = traj.to_speed(3, "single")
        assert len(speed) == 5
        assert all(abs(s - speed_gt) < 1e-5 for s in speed)

        with pytest.raises(ValueError):
            traj.to_speed(1, "linear")
        with pytest.raises(ValueError):
            traj.to_speed(3, "invalid_mode")
        with pytest.raises(ValueError):
            traj.to_speed(11, "linear")


class TestTrajectoryCollection:
    """Test the TrajectoryCollection class."""

    @pytest.fixture
    def trajectory_collection_data(self, tmp_path):
        """Fixture for creating a sample trajectory collection."""
        # 创建临时轨迹文件
        data = {}
        data["x1"] = list(range(10))
        data["y1"] = [0] * 10
        data["x2"] = [0] * 10
        data["y2"] = list(range(10))

        data["x1"][2] = float("nan")
        data["y1"][2] = float("nan")

        df = pd.DataFrame(data)
        file_path = tmp_path / "trajectory.csv"
        df.to_csv(file_path, index=False, na_rep="NaN")
        return str(file_path)

    def test_init(self, trajectory_collection_data):
        """Test the initialization of the Trajectory Collection class."""
        TrajectoryCollection(trajectory_collection_data, fps=30, scale=0.259)

    def test_to_distance(self, trajectory_collection_data):
        """Test the distance calculation."""
        tc_path = trajectory_collection_data
        tc = TrajectoryCollection(tc_path, fps=30, scale=0.259)

        distances = tc.to_distance(1, 2)
        print(distances)

        distance_gt = 2 ** 0.5 * 0.259
        assert len(distances) == 10
        for i in range(10):
            if i == 2:
                assert pd.isna(distances[i])
            else:
                assert abs(distances[i] - distance_gt * i) < 1e-5

        with pytest.raises(IndexError):
            tc.to_distance(5, 2)
        with pytest.raises(IndexError):
            tc.to_distance(1, 5)

    def test_to_speed(self, trajectory_collection_data):
        """Test the speed calculation."""
        tc_path = trajectory_collection_data
        tc = TrajectoryCollection(tc_path, fps=30, scale=0.259)

        speed_gt = 30 * 1 * 0.259
        # Test linear speed
        speed = tc.to_speed(3, "linear")
        print(speed)
        assert len(speed[1]) == 5
        assert all(abs(s - speed_gt) < 1e-5 for s in speed[1])
        assert len(speed[2]) == 8
        assert all(abs(s - speed_gt) < 1e-5 for s in speed[2])
        # Test mean speed
        speed = tc.to_speed(3, "mean")
        assert len(speed[1]) == 5
        assert all(abs(s - speed_gt) < 1e-5 for s in speed[1])
        assert len(speed[2]) == 8
        assert all(abs(s - speed_gt) < 1e-5 for s in speed[2])
        # Test single speed
        speed = tc.to_speed(3, "single")
        assert len(speed[1]) == 5
        assert all(abs(s - speed_gt) < 1e-5 for s in speed[1])
        assert len(speed[2]) == 8
        assert all(abs(s - speed_gt) < 1e-5 for s in speed[2])
