"""
人体运动学模型。

以 Winter 人体测量学比例由身高推算各分段长度，构建 20 关键点骨架；
提供站立姿态生成、腿部二连杆逆运动学求解与躯干前倾/偏航变换，
为静态平衡与 TUG 测试动作的 3D 真值轨迹提供几何基础。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from config import JOINT_INDEX, JOINT_NAMES, SEGMENT_LENGTH_RATIO


def rotation_z(angle_rad: float) -> np.ndarray:
    """绕 z 轴（竖直轴）旋转，用于人体偏航。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_x(angle_rad: float) -> np.ndarray:
    """绕 x 轴（横向轴）旋转，用于躯干前倾。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass
class SkeletonModel:
    """单个受试者的骨架几何参数（世界坐标系：x 横向，y 前进方向，z 竖直向上）。"""

    height_m: float
    mass_kg: float
    lengths: Dict[str, float]

    @classmethod
    def from_subject(cls, subject) -> "SkeletonModel":
        h = subject.height_cm / 100.0
        lengths = {name: ratio * h for name, ratio in SEGMENT_LENGTH_RATIO.items()}
        return cls(height_m=h, mass_kg=subject.mass_kg, lengths=lengths)

    # ------------------------------------------------------------------
    # 基础姿态
    # ------------------------------------------------------------------
    def rest_pose(self) -> np.ndarray:
        """标准直立姿态（骨盆位于原点，局部坐标）。"""
        pts = np.zeros((len(JOINT_NAMES), 3), dtype=float)
        hip_half = 0.098 * self.height_m
        shoulder_half = 0.129 * self.height_m
        foot_h = 0.039 * self.height_m
        trunk = self.lengths["trunk"]
        thigh = self.lengths["thigh"]
        shank = self.lengths["shank"]

        pts[JOINT_INDEX["pelvis"]] = (0.0, 0.0, 0.0)
        pts[JOINT_INDEX["l_hip"]] = (hip_half, 0.0, 0.0)
        pts[JOINT_INDEX["r_hip"]] = (-hip_half, 0.0, 0.0)
        pts[JOINT_INDEX["l_knee"]] = (hip_half, 0.0, -thigh)
        pts[JOINT_INDEX["r_knee"]] = (-hip_half, 0.0, -thigh)
        pts[JOINT_INDEX["l_ankle"]] = (hip_half, 0.0, -thigh - shank)
        pts[JOINT_INDEX["r_ankle"]] = (-hip_half, 0.0, -thigh - shank)
        pts[JOINT_INDEX["l_toe"]] = (hip_half, 0.72 * self.lengths["foot"], -thigh - shank - foot_h)
        pts[JOINT_INDEX["r_toe"]] = (-hip_half, 0.72 * self.lengths["foot"], -thigh - shank - foot_h)

        pts[JOINT_INDEX["neck"]] = (0.0, 0.0, trunk)
        pts[JOINT_INDEX["thorax"]] = (0.0, 0.0, 0.72 * trunk)
        pts[JOINT_INDEX["head"]] = (0.0, 0.0, trunk + self.lengths["head"])

        for side, sign in (("l", 1.0), ("r", -1.0)):
            sh = (sign * shoulder_half, 0.0, trunk - 0.02 * self.height_m)
            el = (sign * shoulder_half, 0.0, sh[2] - self.lengths["upper_arm"])
            wr = (sign * shoulder_half, 0.0, el[2] - self.lengths["forearm"])
            hd = (sign * shoulder_half, 0.0, wr[2] - self.lengths["hand"])
            pts[JOINT_INDEX[f"{side}_shoulder"]] = sh
            pts[JOINT_INDEX[f"{side}_elbow"]] = el
            pts[JOINT_INDEX[f"{side}_wrist"]] = wr
            pts[JOINT_INDEX[f"{side}_hand"]] = hd
        return pts

    def pelvis_to_ground(self) -> float:
        """直立时骨盆到地面的高度。"""
        return self.lengths["thigh"] + self.lengths["shank"] + 0.039 * self.height_m

    # ------------------------------------------------------------------
    # 变换与逆运动学
    # ------------------------------------------------------------------
    @staticmethod
    def local_to_world(points_local: np.ndarray, pelvis_world: np.ndarray,
                       yaw_rad: float, lean_rad: float) -> np.ndarray:
        """局部骨架 → 世界坐标：先绕 x 轴前倾，再绕 z 轴偏航，最后平移。"""
        rot = rotation_z(yaw_rad) @ rotation_x(lean_rad)
        return points_local @ rot.T + pelvis_world

    def solve_leg_ik(self, hip: np.ndarray, ankle: np.ndarray,
                     knee_dir: np.ndarray) -> np.ndarray:
        """二连杆逆运动学：由髋、踝位置求膝关节位置（膝向 knee_dir 弯曲）。"""
        thigh = self.lengths["thigh"]
        shank = self.lengths["shank"]
        vec = ankle - hip
        dist = float(np.linalg.norm(vec))
        dist = float(np.clip(dist, 1e-4, 0.999 * (thigh + shank)))
        axis = vec / dist
        # 目标距离被裁剪后重新求踝位置，保证连杆长度一致
        a = (thigh ** 2 - shank ** 2 + dist ** 2) / (2.0 * dist)
        h = float(np.sqrt(max(thigh ** 2 - a ** 2, 0.0)))
        # 构造垂直于 axis 的弯曲方向
        perp = knee_dir - np.dot(knee_dir, axis) * axis
        norm = float(np.linalg.norm(perp))
        if norm < 1e-8:
            perp = np.array([0.0, 1.0, 0.0])
            perp = perp - np.dot(perp, axis) * axis
            norm = float(np.linalg.norm(perp)) + 1e-12
        perp = perp / norm
        return hip + a * axis + h * perp


def build_standing_skeleton(model: SkeletonModel,
                            pelvis_world: np.ndarray,
                            yaw_rad: float = 0.0,
                            lean_rad: float = 0.0,
                            left_ankle: Optional[np.ndarray] = None,
                            right_ankle: Optional[np.ndarray] = None,
                            arm_swing_l: float = 0.0,
                            arm_swing_r: float = 0.0,
                            trunk_twist: float = 0.0) -> np.ndarray:
    """
    生成一帧世界坐标骨架。

    参数
    ----
    pelvis_world : 骨盆世界坐标 (3,)
    yaw_rad      : 人体朝向（0 表示面向 y 轴正方向）
    lean_rad     : 躯干前倾角（正值为前倾）
    left_ankle / right_ankle : 左右踝关节世界坐标，None 表示使用直立默认位置
    arm_swing_l / arm_swing_r : 手臂摆动角（绕横向轴）
    trunk_twist  : 躯干扭转角（绕竖直轴）
    """
    local = model.rest_pose()
    world = model.local_to_world(local, pelvis_world, yaw_rad, lean_rad)

    # 腿部：以目标踝位置做逆运动学，膝向人体前方弯曲
    rot = rotation_z(yaw_rad)
    forward = rot @ np.array([0.0, 1.0, 0.0])
    for side, ankle_target in (("l", left_ankle), ("r", right_ankle)):
        if ankle_target is None:
            continue
        hip = world[JOINT_INDEX[f"{side}_hip"]]
        ankle = np.asarray(ankle_target, dtype=float)
        knee = model.solve_leg_ik(hip, ankle, forward)
        world[JOINT_INDEX[f"{side}_knee"]] = knee
        world[JOINT_INDEX[f"{side}_ankle"]] = ankle
        world[JOINT_INDEX[f"{side}_toe"]] = ankle + forward * (0.72 * model.lengths["foot"]) \
            - np.array([0.0, 0.0, 0.039 * model.height_m])

    # 手臂：绕肩关节摆动
    for side, swing in (("l", arm_swing_l), ("r", arm_swing_r)):
        if abs(swing) < 1e-9:
            continue
        sh = world[JOINT_INDEX[f"{side}_shoulder"]]
        for joint in ("elbow", "wrist", "hand"):
            rel = world[JOINT_INDEX[f"{side}_{joint}"]] - sh
            rotated = rotation_x(swing) @ rel
            world[JOINT_INDEX[f"{side}_{joint}"]] = sh + rotated

    # 躯干扭转
    if abs(trunk_twist) > 1e-9:
        base = world[JOINT_INDEX["pelvis"]]
        rot_twist = rotation_z(trunk_twist)
        for name in ("neck", "thorax", "head", "l_shoulder", "r_shoulder",
                     "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"):
            world[JOINT_INDEX[name]] = base + rot_twist @ (world[JOINT_INDEX[name]] - base)
    return world


def skeleton_segment_lengths(joints: np.ndarray) -> Dict[str, float]:
    """由一帧骨架计算各骨骼段长度，用于时序一致性约束。"""
    pairs = {
        "head_neck": ("head", "neck"),
        "neck_pelvis": ("neck", "pelvis"),
        "shoulder_l": ("neck", "l_shoulder"),
        "shoulder_r": ("neck", "r_shoulder"),
        "upper_arm_l": ("l_shoulder", "l_elbow"),
        "upper_arm_r": ("r_shoulder", "r_elbow"),
        "forearm_l": ("l_elbow", "l_wrist"),
        "forearm_r": ("r_elbow", "r_wrist"),
        "thigh_l": ("l_hip", "l_knee"),
        "thigh_r": ("r_hip", "r_knee"),
        "shank_l": ("l_knee", "l_ankle"),
        "shank_r": ("r_knee", "r_ankle"),
        "hip_width": ("l_hip", "r_hip"),
    }
    return {name: float(np.linalg.norm(joints[JOINT_INDEX[a]] - joints[JOINT_INDEX[b]]))
            for name, (a, b) in pairs.items()}


def bone_pairs() -> Tuple[Tuple[int, int], ...]:
    """骨架连线，供可视化使用。"""
    named = [
        ("head", "neck"), ("neck", "thorax"), ("thorax", "pelvis"),
        ("neck", "l_shoulder"), ("neck", "r_shoulder"),
        ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("l_wrist", "l_hand"),
        ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("r_wrist", "r_hand"),
        ("pelvis", "l_hip"), ("pelvis", "r_hip"),
        ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
        ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"),
    ]
    return tuple((JOINT_INDEX[a], JOINT_INDEX[b]) for a, b in named)
