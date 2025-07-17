"""Chasing event detection."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .base import Event, EventDetection
from ..data import TrajectoryCollection

from tqdm import tqdm

@dataclass
class ChasingEvent(Event):
    """Chasing event class.

    Args:
        Event: event base class

    Attributes:
        subject_identity: the identity of the subject (chaser)
        object_identity: the identity of the object (chasee)
        subject_moving_distance: the moving distance of the subject
        object_moving_distance: the moving distance of the object
        final_distance: the final distance between the subject and the object
        correlation: the correlation between the trajectories of the subject and the object,
            defined as the proportion of the frames in time window that satisfies the chasing conditions.
    """

    subject_identity: int
    object_identity: int

    subject_moving_distance: float
    object_moving_distance: float

    final_distance: float

    correlation: float

    chasing_direction: float
    chased_direction: float

    def to_dict(self):
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "subject_identity": self.subject_identity,
            "object_identity": self.object_identity,
            "subject_moving_distance": self.subject_moving_distance,
            "object_moving_distance": self.object_moving_distance,
            "final_distance": self.final_distance,
            "correlation": self.correlation,
            "chasing_direction": self.chasing_direction,
            "chased_direction": self.chased_direction,
        }


class ChasingDetector(EventDetection):
    """Chasing event detection class.
    
    Args:
        EventDetection: event detection base class

    Attributes:
        trajectory_collection: trajectory collection
        distance_threshold: distance threshold for close mouse pairs, defaults to 30.0 cm
        theta_threshold: chasing direction threshold (cosθ), defaults to 0.8
        phi_threshold: chased direction threshold (cosϕ), defaults to 0.0
        min_movement: minimum movement threshold, defaults to 30.0 cm
        correlation_threshold: minimum correlation threshold, defaults to 0.6
        window_size: window size, defaults to 120
    """

    def __init__(
        self,
        trajectory_collection: TrajectoryCollection,
        distance_threshold: float = 30.0,
        theta_threshold: float = 0.8,
        phi_threshold: float = 0.0,
        min_movement: float = 30.0,
        correlation_threshold: float = 0.6,
        window_size: int = 120,
    ):
        """Initialize the ChasingDetector class.

        Args:
            trajectory_collection (TrajectoryCollection): trajectory collection
            distance_threshold (float, optional): distance threshold for close pairs. Defaults to 30.0 cm
            theta_threshold (float, optional): chasing direction threshold (cosθ). Defaults to 0.8
            phi_threshold (float, optional): chased direction threshold (cosϕ). Defaults to 0.0
            min_movement (float, optional): minimum movement threshold. Defaults to 30.0 cm
            correlation_threshold (float, optional): minimum correlation threshold. Defaults to 0.6
            window_size (int, optional): window size. Defaults to 120
        """
        super().__init__(trajectory_collection)
        self.distance_threshold = distance_threshold
        self.theta_threshold = theta_threshold
        self.phi_threshold = phi_threshold
        self.min_movement = min_movement
        self.correlation_threshold = correlation_threshold
        self.window_size = window_size

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions.
        
        Args:
            pos1: Position 1 (x, y)
            pos2: Position 2 (x, y)
            
        Returns:
            float: Euclidean distance
        """
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _calculate_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate velocity vectors from trajectory.
        
        Args:
            trajectory: Trajectory array of shape (n_frames, 2)
            
        Returns:
            np.ndarray: Velocity vectors of shape (n_frames-1, 2)
        """
        return np.diff(trajectory, axis=0)

    def _calculate_speed(self, velocity: np.ndarray) -> np.ndarray:
        """Calculate speed from velocity vectors.
        
        Args:
            velocity: Velocity vectors of shape (n_frames, 2)
            
        Returns:
            np.ndarray: Speed values of shape (n_frames,)
        """
        return np.sqrt(np.sum(velocity ** 2, axis=1))

    def _calculate_cos_theta(self, velocity_a: np.ndarray, position_diff: np.ndarray) -> np.ndarray:
        """Calculate cosine of angle between velocity and position difference.
        
        Args:
            velocity_a: Velocity vectors of mouse A
            position_diff: Position difference vectors (A - B)
            
        Returns:
            np.ndarray: Cosine values
        """
        # Normalize vectors
        velocity_norm = np.linalg.norm(velocity_a, axis=1)
        position_norm = np.linalg.norm(position_diff, axis=1)
        
        # Avoid division by zero
        velocity_norm = np.where(velocity_norm == 0, 1e-10, velocity_norm)
        position_norm = np.where(position_norm == 0, 1e-10, position_norm)
        
        # Calculate dot product
        dot_product = np.sum(velocity_a * position_diff, axis=1)
        
        # Calculate cosine
        cos_theta = dot_product / (velocity_norm * position_norm)
        
        return np.clip(cos_theta, -1.0, 1.0)

    def _get_trajectory_data(self, identity: int) -> np.ndarray:
        """Get trajectory data for a specific identity.
        
        Args:
            identity: Mouse identity
            
        Returns:
            np.ndarray: Trajectory array of shape (n_frames, 2)
        """
        trajectory = self.trajectory_collection.trajectories[identity]
        x_data = trajectory.trajectory_data["x"]
        y_data = trajectory.trajectory_data["y"]
        return np.column_stack([x_data, y_data])

    def _calculate_movement_distance(self, trajectory: np.ndarray) -> float:
        """Calculate total movement distance, ignoring NaN values.
        
        Args:
            trajectory: Trajectory array of shape (n_frames, 2)
            
        Returns:
            float: Total movement distance
        """
        velocity = self._calculate_velocity(trajectory)
        # Filter out NaN values
        valid_velocity = velocity[~np.isnan(velocity).any(axis=1)]
        if len(valid_velocity) == 0:
            return 0.0
        distances = np.sqrt(np.sum(valid_velocity ** 2, axis=1))
        return np.sum(distances)

    def _is_valid_frame(self, traj_a_frame: np.ndarray, traj_b_frame: np.ndarray) -> bool:
        """Check if both trajectory frames are valid (not NaN).
        
        Args:
            traj_a_frame: Frame data for trajectory A
            traj_b_frame: Frame data for trajectory B
            
        Returns:
            bool: True if both frames are valid
        """
        return not (np.isnan(traj_a_frame).any() or np.isnan(traj_b_frame).any())

    def detect(self) -> List[ChasingEvent]:
        """Detect chasing events.

        Returns:
            List[ChasingEvent]: List of detected chasing events
        """
        events = []
        identities = self.trajectory_collection.identities

        # Get all trajectory data
        trajectories = {}
        for identity in identities:
            trajectories[identity] = self._get_trajectory_data(identity)

        # Check all possible pairs
        for i, identity_a in tqdm(enumerate(identities)):
            for j, identity_b in tqdm(enumerate(identities)):
                if i >= j:  # Avoid duplicate pairs and self-pairs
                    continue
                 
                traj_a = trajectories[identity_a]
                traj_b = trajectories[identity_b]
                
                # Ensure trajectories have same length
                min_length = min(len(traj_a), len(traj_b))
                traj_a = traj_a[:min_length]
                traj_b = traj_b[:min_length]
                
                # Calculate distances between mice for each frame, handling NaN values
                distances = []
                for frame in range(min_length):
                    if self._is_valid_frame(traj_a[frame], traj_b[frame]):
                        distance = self._calculate_distance(traj_a[frame], traj_b[frame])
                        distances.append(distance)
                    else:
                        distances.append(np.nan)
                distances = np.array(distances)

                # Calculate velocities
                velocity_a = self._calculate_velocity(traj_a)
                velocity_b = self._calculate_velocity(traj_b)

                # Sliding window detection
                for start_frame in range(0, min_length - self.window_size, self.window_size):
                    end_frame = start_frame + self.window_size

                    # Get window data
                    window_distances = distances[start_frame:end_frame]
                    window_velocity_a = velocity_a[start_frame:end_frame-1]
                    window_velocity_b = velocity_b[start_frame:end_frame-1]

                    # Calculate position differences
                    window_traj_a = traj_a[start_frame:end_frame-1]
                    window_traj_b = traj_b[start_frame:end_frame-1]
                    position_diff_ab = window_traj_b - window_traj_a  # A -> B
                    position_diff_ba = window_traj_a - window_traj_b  # B -> A

                                        # Create validity mask for frames without NaN values
                    valid_frames = np.array([
                        self._is_valid_frame(window_traj_a[k], window_traj_b[k])
                        for k in range(len(window_traj_a))
                    ])
                    
                    # Also check if distances are valid (not NaN) - only for the frames we have velocity data
                    valid_distances = ~np.isnan(window_distances[:-1])
                    valid_frames = valid_frames & valid_distances

                    # Check if the first valid distance is within threshold
                    if len(window_distances[:-1][valid_frames]) == 0:
                        continue
                    first_valid_distance = window_distances[:-1][valid_frames][0]
                    if first_valid_distance > self.distance_threshold:
                        continue

                    # Need at least 90% valid frames
                    if np.sum(valid_frames) < self.window_size * 0.9:
                        continue

                    # Calculate cosine angles only for valid frames
                    cos_theta_a = self._calculate_cos_theta(window_velocity_a, position_diff_ab)
                    cos_theta_b = self._calculate_cos_theta(window_velocity_b, position_diff_ba)

                    # Calculate escape direction angles (cosϕ)
                    cos_phi_a = self._calculate_cos_theta(window_velocity_a, -position_diff_ab)
                    cos_phi_b = self._calculate_cos_theta(window_velocity_b, -position_diff_ba)

                    # Handle NaN values in cosine calculations
                    valid_cos_theta_a = ~np.isnan(cos_theta_a)
                    valid_cos_theta_b = ~np.isnan(cos_theta_b)
                    valid_cos_phi_a = ~np.isnan(cos_phi_a)
                    valid_cos_phi_b = ~np.isnan(cos_phi_b)

                    # Check movement thresholds (already handles NaN in _calculate_movement_distance)
                    movement_a = self._calculate_movement_distance(window_traj_a)
                    movement_b = self._calculate_movement_distance(window_traj_b)

                    if movement_a < self.min_movement or movement_b < self.min_movement:
                        continue

                    # Filter out NaN values for cosine calculations
                    valid_cos_theta_a_frames = ~np.isnan(cos_theta_a)
                    valid_cos_theta_b_frames = ~np.isnan(cos_theta_b)
                    
                    if np.sum(valid_cos_theta_a_frames) == 0 or np.sum(valid_cos_theta_b_frames) == 0:
                        continue
                    
                    # Calculate average cosθ for both mice (higher cosθ = more likely to be chaser)
                    avg_cos_theta_a = np.mean(cos_theta_a[valid_cos_theta_a_frames])
                    avg_cos_theta_b = np.mean(cos_theta_b[valid_cos_theta_b_frames])
                    
                    if avg_cos_theta_a > avg_cos_theta_b:
                        # A is chaser (moving towards B), B is chasee
                        chaser_id = identity_a
                        chasee_id = identity_b
                        cos_theta_chaser = cos_theta_a
                        cos_phi_chasee = cos_phi_b
                        valid_cos_theta_chaser = valid_cos_theta_a
                        valid_cos_phi_chasee = valid_cos_phi_b
                        chaser_movement = movement_a
                        chasee_movement = movement_b
                    else:
                        # B is chaser (moving towards A), A is chasee
                        chaser_id = identity_b
                        chasee_id = identity_a
                        cos_theta_chaser = cos_theta_b
                        cos_phi_chasee = cos_phi_a
                        valid_cos_theta_chaser = valid_cos_theta_b
                        valid_cos_phi_chasee = valid_cos_phi_a
                        chaser_movement = movement_b
                        chasee_movement = movement_a

                    # Apply chasing criteria (only for valid frames)
                    chasing_conditions = (
                        (cos_theta_chaser > self.theta_threshold) &     # cosθ > threshold
                        valid_cos_theta_chaser &    # Valid cosθ calculation
                        (cos_phi_chasee < self.phi_threshold) &     # cosϕ < threshold
                        valid_cos_phi_chasee    # Valid cosϕ calculation
                    )

                    # Calculate correlation (proportion of valid frames satisfying conditions)
                    correlation = np.sum(chasing_conditions) / len(chasing_conditions)

                    # Check if correlation exceeds threshold
                    if correlation > self.correlation_threshold:
                        # Find the final valid distance
                        valid_final_distances = window_distances[~np.isnan(window_distances)]
                        if len(valid_final_distances) == 0:
                            continue
                        final_distance = valid_final_distances[-1]

                        # Create chasing event
                        event = ChasingEvent(
                            start_frame=start_frame,
                            end_frame=end_frame,
                            subject_identity=chaser_id,
                            object_identity=chasee_id,
                            subject_moving_distance=chaser_movement,
                            object_moving_distance=chasee_movement,
                            final_distance=final_distance,
                            correlation=correlation,
                            chasing_direction=avg_cos_theta_a,
                            chased_direction=avg_cos_theta_b,
                        )
                        events.append(event)

        return events
