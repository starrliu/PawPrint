"""Approach event detection."""

from dataclasses import dataclass

import numpy as np

from .base import Event, EventDetection
from ..data import TrajectoryCollection
from ..utils import calculate_movement


@dataclass
class ApproachEvent(Event):
    """Approach event class.

    Args:
        Event: event base class

    Attributes:
        subject_identity: the identity of the subject (approacher)
        object_identity: the identity of the object (approached)
    """

    subject_identity: int
    object_identity: int

    def to_dict(self):
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "subject_identity": self.subject_identity,
            "object_identity": self.object_identity,
        }


class ApproachDetector(EventDetection):
    """Approach event detection class.

    Approach event is defined as one mouse starts from a long distance and gradually approaches another mouse.
    Args:
        EventDetection: event detection base class

    Attributes:
        init_threshold: the initial threshold for approach
        min_threshold: the minimum threshold for approach
        window_size: the window size for approach
    """

    def __init__(
        self,
        trajectory_collection: TrajectoryCollection,
        init_threshold: float = 30,
        end_threshold: float = 5,
        window_size: int = 120,
    ):
        super().__init__(trajectory_collection)
        self.init_threshold = init_threshold
        self.min_threshold = end_threshold
        self.window_size = window_size

    def detect(self):
        """Detect approach events.

        Returns:
            list[ApproachEvent]: List of detected approach events.
        """
        events = []
        identities = self.trajectory_collection.identities

        # Iterate through all possible pairs of identities
        for i, first_id in enumerate(identities):
            for second_id in identities[i + 1:]:
                # Get distances between the pair
                distances = self.trajectory_collection.to_distance(first_id, second_id)
                
                # Skip if not enough frames
                if len(distances) < self.window_size:
                    continue

                # Slide window through the trajectory
                for window_start in range(
                    0, len(distances) - self.window_size, self.window_size
                ):
                    window = distances[window_start : window_start + self.window_size]
                    
                    # Skip windows with too many NaN values
                    if (
                        sum(np.isnan(x) for x in window) > self.window_size * 0.2
                    ):  # Allow 20% NaN
                        continue

                    # Find the first frame where distance > init_threshold
                    start_frame = None
                    for idx, dist in enumerate(window):
                        if not np.isnan(dist) and dist > self.init_threshold:
                            start_frame = window_start + idx
                            break
                    
                    if start_frame is None:
                        continue  # No valid start frame found
                    
                    # Find the first frame after start_frame where distance < end_threshold
                    end_frame = None
                    for idx in range(start_frame - window_start + 1, len(window)):
                        dist = window[idx]
                        if not np.isnan(dist) and dist < self.min_threshold:
                            end_frame = window_start + idx
                            break
                    
                    if end_frame is None:
                        continue  # No valid end frame found

                    # Determine which animal is the approacher by comparing their movement
                    first_traj = self.trajectory_collection.trajectories[first_id]
                    second_traj = self.trajectory_collection.trajectories[second_id]
                    
                    # Get trajectory segments for the actual event duration
                    first_x = first_traj.trajectory_data["x"][
                        start_frame : end_frame + 1
                    ]
                    first_y = first_traj.trajectory_data["y"][
                        start_frame : end_frame + 1
                    ]
                    second_x = second_traj.trajectory_data["x"][
                        start_frame : end_frame + 1
                    ]
                    second_y = second_traj.trajectory_data["y"][
                        start_frame : end_frame + 1
                    ]
                    
                    # Calculate total movement for each animal
                    first_moved = calculate_movement(first_x, first_y)
                    second_moved = calculate_movement(second_x, second_y)
                    
                    # Create approach event with the animal that moved more as the subject
                    subject_id = (
                        first_id if first_moved > second_moved else second_id
                    )
                    object_id = second_id if subject_id == first_id else first_id
                    
                    event = ApproachEvent(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        subject_identity=subject_id,
                        object_identity=object_id,
                    )
                    events.append(event)

        return events
