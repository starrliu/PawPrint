"""Test the data module."""

import pytest

from pawprint.data import Trajectory


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

        speed_gt = 30 * 2**0.5  # 30 fps, 1 second, 2 units of distance
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

    def test_init(self):
        # TODO
        return

    def test_to_distance(self):
        # TODO
        return

    def test_to_speed(self):
        # TODO
        return
