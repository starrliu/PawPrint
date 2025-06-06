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
        window_size: int = 120,
    ):
        """Initialize the ChasingDetector class.

        Args:
            trajectory_collection (TrajectoryCollection): trajectory collection
            min_distance (float, optional): minimum distance threshold. Defaults to 30.0 cm
            min_correlation (float, optional): minimum correlation threshold. Defaults to 0.7
            min_movement (float, optional): minimum movement threshold. Defaults to 60.0 cm
            window_size (int, optional): window size. Defaults to 120
        """
        super().__init__(trajectory_collection)
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

    def _init_correlation(self, traj1: list, traj2: list) -> tuple:

        dx1 = np.diff([x for x, _ in traj1])
        dy1 = np.diff([y for _, y in traj1])
        dx2 = np.diff([x for x, _ in traj2])
        dy2 = np.diff([y for _, y in traj2])

        tmpx = [dx1 * dx1, dx1 * dx2, dx2 * dx2, dx1, dx2]
        tmpy = [dy1 * dy1, dy1 * dy2, dy2 * dy2, dy1, dy2]

        return (tmpx, tmpy)

    def _calculate_correlation(self, tmp: list, l: int, r: int, n: int) -> float:
        """Calculate the correlation between two lists."""

        x11 = np.sum(tmp[0][l:r])
        x12 = np.sum(tmp[1][l:r])
        x22 = np.sum(tmp[2][l:r])
        x1 = np.sum(tmp[3][l:r])
        x2 = np.sum(tmp[4][l:r])

        x1 /= n
        x2 /= n

        cov = np.asarray(
            [
                [x11 - n * x1 * x1, x12 - n * x1 * x2],
                [x12 - n * x1 - x2, x22 - n * x2 * x2],
            ]
        )
        cov /= n - 1

        std1 = np.sqrt(cov[0, 0])
        std2 = np.sqrt(cov[1, 1])
        corr = cov / np.outer([std1, std2], [std1, std2])

        return corr[0, 1]

    def _test_correlation(
        self,
        tmp: list,
        l: int,
        r: int,
        correlation: float,
    ) -> int:
        ground = np.corrcoef(tmp[3][l:r], tmp[4][l:r])[0, 1]
        if np.isnan(ground):
            return np.isnan(correlation)

        if np.isnan(correlation):
            return 0

        return abs(correlation - ground) < 1e-5

    def detect(self) -> list[ChasingEvent]:
        """检测追逐行为。

        Returns:
            List[ChasingEvent]: 检测到的追逐行为事件列表
        """
        events = []

        # 遍历所有可能的追逐者-被追逐者对
        for chaser_id in self.trajectory_collection.identities:
            for chasee_id in self.trajectory_collection.identities:
                print(chaser_id, chasee_id)
                if chaser_id >= chasee_id:
                    continue

                chaser = self.trajectory_collection.trajectories[chaser_id]
                chasee = self.trajectory_collection.trajectories[chasee_id]

                chaser_positions = list(
                    zip(
                        chaser.trajectory_data["x"],
                        chaser.trajectory_data["y"],
                    )
                )
                chasee_positions = list(
                    zip(
                        chasee.trajectory_data["x"],
                        chasee.trajectory_data["y"],
                    )
                )

                distances = self.trajectory_collection.to_distance(chaser_id, chasee_id)

                chaser_movement = self._calculate_movement(chaser_positions)
                chasee_movement = self._calculate_movement(chasee_positions)

                # 预处理计算 correlation 所需的量
                (tmp_x, tmp_y) = self._init_correlation(
                    chaser_positions, chasee_positions
                )

                # 在滑动窗口中检测追逐行为
                for start_frame in range(len(chaser) - self.window_size):
                    end_frame = start_frame + self.window_size

                    # 计算窗口内的所需值
                    chaser_movement_val = sum(
                        chaser_movement[start_frame : end_frame - 1]
                    )
                    chasee_movement_val = sum(
                        chasee_movement[start_frame : end_frame - 1]
                    )

                    final_distance = distances[end_frame - 1]

                    correlation_x = self._calculate_correlation(
                        tmp_x, start_frame, end_frame - 1, self.window_size - 1
                    )
                    correlation_y = self._calculate_correlation(
                        tmp_y, start_frame, end_frame - 1, self.window_size - 1
                    )

                    # 检测优化的 correlation 是否正确
                    # if not self._test_correlation(
                    #     tmp_x, start_frame, end_frame - 1, correlation_x
                    # ):
                    #     raise ValueError(f"correlation_x wrong {start_frame,end_frame}")
                    # if not self._test_correlation(
                    #     tmp_y, start_frame, end_frame - 1, correlation_y
                    # ):
                    #     raise ValueError(f"correlation_y wrong {start_frame,end_frame}")

                    correlation = (correlation_x + correlation_y) / 2
                    if (
                        final_distance > self.min_distance
                        and chaser_movement_val > self.min_movement
                        and chasee_movement_val > self.min_movement
                        and abs(correlation) > self.min_correlation
                    ):
                        if correlation < 0:
                            correlation = -correlation
                            chaser_id, chasee_id = chasee_id, chaser_id
                            chaser_movement_val, chasee_movement_val = (
                                chasee_movement_val,
                                chaser_movement_val,
                            )
                        chasingevent = ChasingEvent(
                            start_frame=start_frame,
                            end_frame=end_frame,
                            subject_identity=chaser_id,
                            object_identity=chasee_id,
                            subject_moving_distance=chaser_movement_val,
                            object_moving_distance=chasee_movement_val,
                            final_distance=final_distance,
                            correlation=correlation,
                        )
                        events.append(chasingevent)
                        if chaser_id >= chasee_id:
                            chaser_id, chasee_id = chasee_id, chaser_id

        return events
