"""Test the data module."""

import pytest
import numpy as np
import os
import pandas as pd

from pawprint.data import Trajectory, TrajectoryCollection


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

    @pytest.fixture
    def sample_trajectory_collection(self):
        """Fixture for creating a sample trajectory collection."""
        
        # Create sample trajectory data
        n_frames = 100
        data = {
            'x1': list(range(n_frames)),  # Linear motion
            'y1': list(range(n_frames)),
            'x2': [x + 3 for x in range(n_frames)],  # Parallel motion, offset by 3
            'y2': [y + 4 for y in range(n_frames)],
            'x3': [10] * n_frames,  # Stationary point
            'y3': [10] * n_frames
        }
        
        # Add some NaN values
        data['x2'][5:8] = [np.nan] * 3  # Add NaN sequence
        data['y2'][5:8] = [np.nan] * 3
        data['x3'][50] = np.nan  # Add single NaN
        data['y3'][50] = np.nan
        
        # Create DataFrame and save to temporary file
        df = pd.DataFrame(data)
        temp_file = "test_trajectories.csv"
        df.to_csv(temp_file, index=False)
        
        # Create TrajectoryCollection
        collection = TrajectoryCollection(temp_file, fps=30)
        
        os.remove(temp_file)
        
        return collection

    def test_to_distance(self, sample_trajectory_collection):
        """Test the distance calculation between trajectories."""
        # Test constant distance between parallel trajectories
        distances = sample_trajectory_collection.to_distance(1, 2)
        valid_distances = [d for d in distances[:5] if not np.isnan(d)]  # Get first 5 valid distances
        expected_distance = 5.0  # sqrt(3^2 + 4^2) = 5
        assert all(abs(d - expected_distance) < 1e-5 for d in valid_distances)
        assert np.isnan(distances[5])  # Should be NaN where trajectory 2 has NaN
        assert np.isnan(distances[6])
        assert np.isnan(distances[7])
        
        # Test distance to stationary point
        distances = sample_trajectory_collection.to_distance(1, 3)
        assert not np.isnan(distances[0])  # Should be valid at start
        assert np.isnan(distances[50])  # Should be NaN at frame 50
        
        # Test invalid identities
        with pytest.raises(ValueError, match="not found in trajectories"):
            sample_trajectory_collection.to_distance(1, 999)  # Non-existent ID
        
        with pytest.raises(ValueError, match="not found in trajectories"):
            sample_trajectory_collection.to_distance(999, 1)  # Non-existent ID
        
        # Test distance calculation correctness
        frame0_distance = distances[0]
        expected_distance = np.sqrt((0 - 10)**2 + (0 - 10)**2)  # Distance from (0,0) to (10,10)
        assert abs(frame0_distance - expected_distance) < 1e-5

    def test_to_speed(self):
        # TODO
        return

    def test_to_approach(self, sample_trajectory_collection):
        """Test the approach calculation between trajectories."""
        # Test with default threshold (5.0)
        # Trajectories 1 and 2 are parallel with constant distance of 5.0
        approaches = sample_trajectory_collection.to_approach(1, 2)
        # First 5 frames should be 0 (distance = 5.0 = threshold)
        assert all(a == 0 for a in approaches[:5])
        # Frames 5-7 should be -1 (NaN in trajectory 2)
        assert all(a == -1 for a in approaches[5:8])

        # Test with larger threshold (10.0)
        # Now the distance (5.0) should be considered "close"
        approaches = sample_trajectory_collection.to_approach(1, 2, threshold=10.0)
        # First 5 frames should be 1 (distance < threshold)
        assert all(a == 1 for a in approaches[:5])
        # Frames 5-7 should still be -1 (NaN values)
        assert all(a == -1 for a in approaches[5:8])

        # Test with smaller threshold (3.0)
        # Now the distance (5.0) should be considered "far"
        approaches = sample_trajectory_collection.to_approach(1, 2, threshold=3.0)
        # First 5 frames should be 0 (distance > threshold)
        assert all(a == 0 for a in approaches[:5])

        # Test with stationary point (trajectory 3)
        approaches = sample_trajectory_collection.to_approach(1, 3, threshold=15.0)
        # Frame 0: distance = sqrt(200) ≈ 14.14, should be 1 with threshold 15
        assert approaches[0] == 1
        # Frame 50 should be -1 (NaN in trajectory 3)
        assert approaches[50] == -1

        # Test invalid identities
        with pytest.raises(ValueError, match="not found in trajectories"):
            sample_trajectory_collection.to_approach(1, 999)

        with pytest.raises(ValueError, match="not found in trajectories"):
            sample_trajectory_collection.to_approach(999, 1)

        # Test negative threshold
        with pytest.raises(ValueError, match="Threshold must be positive"):
            sample_trajectory_collection.to_approach(1, 2, threshold=-1.0)
