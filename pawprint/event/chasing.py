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


# TODO: 实现ChasingDetector类，如下内容由LLM自动生成，仅为代码框架，建议根据实际需求修改
# 一些hint:
# 1. 由于需要对每对小鼠进行遍历，所以这个算法的时间复杂度为O(n^2)，n为小鼠的只数。所以，最好进行一些剪枝。比如，
#    当对subject进行遍历时，如果subject的移动距离小于min_movement，那么就可以跳过这个subject。
# 2. 可以先根据视频选出一些追逐的片段，然后对这些片段进行测试。
# 3. default的参数都可以修改，不一定合适。
# 4. 这段代码可能比较长，最好保持代码结构清晰。可以多定义一些函数，使得代码更加清晰。
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

    def _calculate_movement(self, positions: list) -> float:
        """Calculate the movement of a trajectory."""
        total_distance = 0
        for i in range(len(positions) - 1):
            total_distance += self._calculate_distance(positions[i], positions[i + 1])
        return total_distance

    def _calculate_correlation(self, traj1: list, traj2: list) -> float:
        """Calculate the correlation between two trajectories."""
        # 计算位移向量
        dx1 = np.diff([x for x, _ in traj1])
        dy1 = np.diff([y for _, y in traj1])
        dx2 = np.diff([x for x, _ in traj2])
        dy2 = np.diff([y for _, y in traj2])

        # 计算方向向量的相关性
        corr_x = np.corrcoef(dx1, dx2)[0, 1]
        corr_y, _ = np.corrcoef(dy1, dy2)
        return (corr_x + corr_y) / 2

    def detect(self) -> list[ChasingEvent]:
        """检测追逐行为。

        Returns:
            List[ChasingEvent]: 检测到的追逐行为事件列表
        """
        events = []

        # 遍历所有可能的追逐者-被追逐者对
        for chaser_id in self.trajectory_collection.identities:
            for chasee_id in self.trajectory_collection.identities:
                if chaser_id == chasee_id:
                    continue

                chaser = self.trajectory_collection.trajectories[chaser_id]
                chasee = self.trajectory_collection.trajectories[chasee_id]

                # 在滑动窗口中检测追逐行为
                for start_frame in range(len(chaser) - self.window_size):
                    end_frame = start_frame + self.window_size

                    # 提取窗口内的轨迹
                    chaser_positions = list(
                        zip(
                            chaser.trajectory_data["x"][start_frame:end_frame],
                            chaser.trajectory_data["y"][start_frame:end_frame],
                        )
                    )
                    chasee_positions = list(
                        zip(
                            chasee.trajectory_data["x"][start_frame:end_frame],
                            chasee.trajectory_data["y"][start_frame:end_frame],
                        )
                    )

                    # TODO: 实现追逐行为的判定逻辑
                    # 1. 计算移动距离
                    # 2. 计算平均距离
                    # 3. 计算轨迹相关性
                    # 4. 根据阈值判断是否为追逐行为
                    # 5. 如果满足条件，创建并添加ChasingEvent

        return events
