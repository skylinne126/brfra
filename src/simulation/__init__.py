"""仿真层：虚拟老年队列、人体运动学、测试动作真值、单目传感噪声模型。"""

from .cohort import Subject, generate_cohort
from .kinematics import SkeletonModel, build_standing_skeleton
from .motion import simulate_quiet_stance, simulate_single_leg_stance, simulate_tug
from .sensors import MonocularCamera, SensorStream, simulate_sensor_stream

__all__ = [
    "Subject",
    "generate_cohort",
    "SkeletonModel",
    "build_standing_skeleton",
    "simulate_quiet_stance",
    "simulate_single_leg_stance",
    "simulate_tug",
    "MonocularCamera",
    "SensorStream",
    "simulate_sensor_stream",
]
