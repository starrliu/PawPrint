"""Chasing event detection."""

from dataclasses import dataclass

import numpy as np

from .base import Event, EventDetection
from ..data import TrajectoryCollection


@dataclass
class ChasingEvent(Event):
    """Chasing event class.

    Args:
        Event: event base class

    Attributes:
        subject_identity: the identity of the subject
        object_identity: the identity of the object
        subject_moving_distance: the moving distance of the subject
        object_moving_distance: the moving distance of the object
        final_distance: the final distance between the subject and the object
        correlation: the correlation between the trajectories of the subject and the object
    """

    subject_identity: int
    object_identity: int

    subject_moving_distance: float
    object_moving_distance: float

    final_distance: float

    correlation: float


class ChasingDetector(EventDetection):
    """Chasing event detection class.
    Args:
        EventDetection: event detection base class

    Attributes:
        trajectory_collection: trajectory collection
        min_distance: minimum distance threshold, defaults to 30.0 cm
        min_correlation: minimum correlation threshold, defaults to 0.7
        min_movement: minimum movement threshold, defaults to 60.0 cm
        window_size: window size, defaults to 120
    """

    def __init__(
        self,
        trajectory_collection: TrajectoryCollection,
        min_distance: float = 30.0,
        min_correlation: float = 0.7,
        min_movement: float = 60.0,
        theta_high: float = 0.5,
        theta_low: float = 0.2,
        window_size: int = 120,
    ):
        """Initialize the ChasingDetector class.

        Args:
            trajectory_collection (TrajectoryCollection): trajectory collection
            min_distance (float, optional): minimum distance threshold. Defaults to 30.0 cm
            min_correlation (float, optional): minimum correlation threshold. Defaults to 0.7
            min_movement (float, optional): minimum movement threshold. Defaults to 60.0 cm
            theta_high (float): chaser threhold. Defaults to 0.5
            theta_low (float): chasee threhold. Defaults to 0.2
            window_size (int, optional): window size. Defaults to 120
        """
        super().__init__(trajectory_collection)
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.min_distance = min_distance
        self.min_correlation = min_correlation
        self.min_movement = min_movement
        self.window_size = window_size

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate the distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _calculate_movement(self, positions: list) -> list:
        """Calculate the movement of a trajectory."""
        movement = []
        for i in range(len(positions) - 1):
            movement.append(self._calculate_distance(positions[i], positions[i + 1]))
        return movement

    def _calculate_correlation(self, delta_a: list, delta_b: list) -> float:
        delta_ax = [x for (x, _) in delta_a]
        delta_ay = [y for (_, y) in delta_a]
        delta_bx = [x for (x, _) in delta_b]
        delta_by = [y for (_, y) in delta_b]
        corr_x = np.corrcoef(delta_ax, delta_bx)[0, 1]
        corr_y = np.corrcoef(delta_ay, delta_by)[0, 1]
        return (corr_x + corr_y) / 2

    def _calculate_diff(self, traj: list) -> list:
        dtraj = traj[1:] - traj[:-1]
        return np.asarray(dtraj)

    def _calculate_theta(self, delta_a: list, delta_ab: list) -> list:
        delta_ax = delta_a[:, 0]
        delta_ay = delta_a[:, 1]
        delta_abx = delta_ab[:, 0]
        delta_aby = delta_ab[:, 1]

        return (delta_ax * delta_abx + delta_ay * delta_aby) / (
            np.sqrt((delta_ax ** 2 + delta_ay ** 2) * (delta_abx ** 2 + delta_aby ** 2))
        )

    def detect(self) -> list[ChasingEvent]:
        """检测追逐行为。

        Returns:
            List[ChasingEvent]: 检测到的追逐行为事件列表
        """
        events = []

        # 遍历所有可能的追逐者-被追逐者对
        for a in self.trajectory_collection.identities:
            for b in self.trajectory_collection.identities:
                if a >= b:
                    continue
                print(a, b)

                traj_a = self.trajectory_collection.trajectories[a]
                traj_a = np.asarray(
                    list(zip(traj_a.trajectory_data["x"], traj_a.trajectory_data["y"]))
                )
                traj_b = self.trajectory_collection.trajectories[b]
                traj_b = np.asarray(
                    list(zip(traj_b.trajectory_data["x"], traj_b.trajectory_data["y"]))
                )

                delta_a = self._calculate_diff(traj_a)
                delta_b = self._calculate_diff(traj_b)
                delta_ab = traj_a - traj_b
                delta_ab = delta_ab[:-1]

                theta_ab = self._calculate_theta(delta_a, delta_ab)
                theta_ba = self._calculate_theta(delta_b, -delta_ab)

                distances = self.trajectory_collection.to_distance(a, b)

                movement_a = self._calculate_movement(traj_a)
                movement_b = self._calculate_movement(traj_b)

                # 在滑动窗口中检测追逐行为
                for start_frame in range(len(delta_a) - self.window_size):
                    end_frame = start_frame + self.window_size

                    movement_a_val = sum(movement_a[start_frame : end_frame - 1])

                    if not movement_a_val > self.min_movement:
                        continue

                    movement_b_val = sum(movement_b[start_frame : end_frame - 1])

                    if not movement_b_val > self.min_movement:
                        continue

                    final_distance = distances[end_frame - 1]

                    if not final_distance > self.min_distance:
                        continue

                    theta_ab_val = (
                        np.sum(theta_ab[start_frame:end_frame]) / self.window_size
                    )
                    theta_ba_val = (
                        np.sum(theta_ba[start_frame:end_frame]) / self.window_size
                    )
                    if not (
                        max(theta_ab_val, theta_ba_val) > self.theta_high
                        and min(theta_ab_val, theta_ba_val) < self.theta_low
                    ):
                        continue

                    correlation = self._calculate_correlation(
                        delta_a[start_frame : end_frame - 1],
                        delta_b[start_frame : end_frame - 1],
                    )
                    if not correlation > self.min_correlation:
                        continue

                    chasingevent = ChasingEvent(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        subject_identity=a,
                        object_identity=b,
                        subject_moving_distance=movement_a_val,
                        object_moving_distance=movement_b_val,
                        final_distance=final_distance,
                        correlation=correlation,
                    )
                    if theta_ba_val > theta_ab_val:
                        chasingevent.subject_identity = b
                        chasingevent.object_identity = a
                        chasingevent.subject_moving_distance = movement_b_val
                        chasingevent.object_moving_distance = movement_a_val
                    events.append(chasingevent)
        print(len(events))
        return events
