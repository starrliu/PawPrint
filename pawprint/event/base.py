"""A base class for event detection.

Typical usage example:
    trajectory_collection = TrajectoryCollection(trajectory_data)
    event_detection = EventDetection(trajectory_collection)
    event_detection.detect()
"""

from dataclasses import dataclass

from ..data import TrajectoryCollection


@dataclass
class Event:
    """Event class."""

    start_frame: int
    end_frame: int


class EventDetection:
    """Event detection base class."""

    def __init__(self, trajectory_collection: TrajectoryCollection):
        self.trajectory_collection = trajectory_collection

    def detect(self):
        """Detect events."""
        raise NotImplementedError("Subclasses must implement detect() method.")
