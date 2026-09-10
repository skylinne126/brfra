"""
人体测量学先验：分段质量分布与分段质心位置。

Dempster / Winter 人体惯性参数表用于把关节三维坐标加权成人体重心（COM）；
同一张表还作为单目深度估计的"人体尺寸先验"，用于消解单目深度的尺度歧义。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from config import JOINT_INDEX, SEGMENT_DEFS, SEGMENT_LENGTH_RATIO

# 标准分段质量比例
STANDARD_MASS_RATIOS: Dict[str, float] = {name: ratio for name, _, _, ratio, _ in SEGMENT_DEFS}

# 用于尺度恢复的分段长度先验（相对身高），键为关节对
SCALE_PRIOR_PAIRS: Tuple[Tuple[str, str, float], ...] = (
    ("head", "neck", SEGMENT_LENGTH_RATIO["head"]),
    ("neck", "pelvis", SEGMENT_LENGTH_RATIO["trunk"]),
    ("l_shoulder", "l_elbow", SEGMENT_LENGTH_RATIO["upper_arm"]),
    ("r_shoulder", "r_elbow", SEGMENT_LENGTH_RATIO["upper_arm"]),
    ("l_elbow", "l_wrist", SEGMENT_LENGTH_RATIO["forearm"]),
    ("r_elbow", "r_wrist", SEGMENT_LENGTH_RATIO["forearm"]),
    ("l_hip", "l_knee", SEGMENT_LENGTH_RATIO["thigh"]),
    ("r_hip", "r_knee", SEGMENT_LENGTH_RATIO["thigh"]),
    ("l_knee", "l_ankle", SEGMENT_LENGTH_RATIO["shank"]),
    ("r_knee", "r_ankle", SEGMENT_LENGTH_RATIO["shank"]),
)


def segment_com_points(joints: np.ndarray,
                       mass_ratios: Optional[Dict[str, float]] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    由一帧（或多帧）关节坐标计算各分段质心位置与对应质量权重。

    joints 形状为 (J, 3) 或 (T, J, 3)。
    """
    ratios = mass_ratios if mass_ratios is not None else STANDARD_MASS_RATIOS
    pts = []
    weights = []
    for name, proximal, distal, _ratio, com_frac in SEGMENT_DEFS:
        p = joints[..., JOINT_INDEX[proximal], :]
        d = joints[..., JOINT_INDEX[distal], :]
        pts.append(p + com_frac * (d - p))
        weights.append(ratios[name])
    return np.stack(pts, axis=-2), np.asarray(weights, dtype=float)


def compute_com(joints: np.ndarray,
                mass_ratios: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    计算人体三维重心（分段质量加权）。

    joints : (J, 3) 或 (T, J, 3)
    返回   : (3,) 或 (T, 3)
    """
    seg_points, weights = segment_com_points(joints, mass_ratios)
    w = weights / weights.sum()
    return np.tensordot(seg_points, w, axes=([-2], [0]))


def segment_lengths(joints: np.ndarray) -> Dict[str, float]:
    """一帧骨架的分段长度（米）。"""
    return {name: float(np.linalg.norm(joints[JOINT_INDEX[a]] - joints[JOINT_INDEX[b]]))
            for a, b, _ in SCALE_PRIOR_PAIRS}


def height_prior_from_lengths(joints: np.ndarray, height_m: float) -> Dict[str, float]:
    """按受试者身高生成各分段长度先验（米）。"""
    return {name: ratio * height_m for _, _, ratio in SCALE_PRIOR_PAIRS}
