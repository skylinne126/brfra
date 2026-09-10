"""
MoSca 复现：四维（3D + 时间）人体轨迹重建。

论文出处
--------
#882  Lei J, Weng Y, Harley A W, et al. MoSca: Dynamic Gaussian Fusion from
      Casual Videos via 4D Motion Scaffolds[C]. Proceedings of the IEEE/CVF
      Conference on Computer Vision and Pattern Recognition (CVPR), 2025:
      6165-6177.

原论文思想与本模块的对应
------------------------
原论文把单目视频提升为紧凑、平滑的 Motion Scaffold（运动支架）表示来编码运动
与形变，将几何与外观从形变场中解耦，并把高斯锚定在支架上做全局融合；相机焦距
与位姿由光束法平差求解。

本项目以 20 关节人体骨架充当“运动支架”：时间二阶差分项对应支架的平滑运动
约束，骨骼长度一致性项对应支架的结构约束，全局四维优化对应原论文的支架全局
融合；本任务只需轨迹而无需新视角合成，故略去高斯泼溅渲染分支。

复现要点
--------
MoSca 的核心思想是把动态场景重建表述为一个带时序正则的全局优化问题：
在数据项之外引入时间二阶差分平滑项与结构（骨骼长度）一致性项，
从而在遮挡、单目深度抖动条件下仍能得到时间上连贯的动态重建结果。

本模块在 RollingDepth 输出的逐帧三维关节基础上求解

    min_X  w_data Σ_t m_t‖x_t - z_t‖²
         + w_time Σ_t ‖x_{t+1} - 2x_t + x_{t-1}‖²
         + w_bone Σ_t Σ_(i,j) (‖x_t^i - x_t^j‖ - L_ij)² / L_ij²

其中 m_t 为可见性掩码，遮挡帧只由时序正则与骨骼约束填补。
采用向量化梯度下降（Adam）求解，全部计算在 (T, J, 3) 张量上完成。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import CFG, JOINT_INDEX, SimConfig
from src.rollingdepth.anthropometry import SCALE_PRIOR_PAIRS


@dataclass
class FourDResult:
    """四维重建结果。"""

    joints3d: np.ndarray          # (T, J, 3) 时序一致化后的三维关节
    bone_error_before: float      # 优化前骨骼长度平均误差（m）
    bone_error_after: float       # 优化后骨骼长度平均误差（m）
    jitter_before: float          # 优化前时间抖动指标（m/frame²）
    jitter_after: float           # 优化后时间抖动指标（m/frame²）
    filled_frames: int            # 被填补的遮挡帧数
    iterations: int


class Trajectory4DOptimizer:
    """四维轨迹时序一致性优化器。"""

    def __init__(self, cfg: SimConfig = CFG):
        self.cfg = cfg

    # ------------------------------------------------------------------
    @staticmethod
    def _bone_error(joints: np.ndarray, pairs: List[Tuple[int, int, float]]) -> float:
        """骨骼长度平均绝对误差。"""
        errors = []
        for i, j, length in pairs:
            cur = np.linalg.norm(joints[:, i] - joints[:, j], axis=1)
            errors.append(np.abs(cur - length))
        return float(np.mean(np.concatenate(errors))) if errors else 0.0

    @staticmethod
    def _jitter(joints: np.ndarray) -> float:
        """时间二阶差分均方根，衡量轨迹抖动程度。"""
        if joints.shape[0] < 3:
            return 0.0
        d2 = joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
        return float(np.sqrt(np.mean(np.sum(d2 ** 2, axis=-1))))

    def reconstruct(self, joints3d: np.ndarray, visible: np.ndarray,
                    height_m: float) -> FourDResult:
        """
        对逐帧三维关节做四维重建。

        joints3d : (T, J, 3) RollingDepth 输出的三维关节
        visible  : (T, J) 可见性掩码
        height_m : 受试者身高，用于骨骼长度先验
        """
        cfg = self.cfg
        pairs = [(JOINT_INDEX[a], JOINT_INDEX[b], ratio * height_m)
                 for a, b, ratio in SCALE_PRIOR_PAIRS]

        mask = visible.astype(float)[..., None]        # (T, J, 1)
        filled = int((~visible).sum())
        x = joints3d.copy()

        # 遮挡关节先用线性插值初始化，避免优化从零开始
        idx = np.arange(x.shape[0])
        for j in range(x.shape[1]):
            bad = ~np.isfinite(x[:, j, 0]) | (~visible[:, j])
            if bad.all():
                x[:, j, :] = 0.0
            elif bad.any():
                for k in range(3):
                    x[bad, j, k] = np.interp(idx[bad], idx[~bad], x[~bad, j, k])

        bone_before = self._bone_error(x, pairs)
        jitter_before = self._jitter(x)

        m = np.zeros_like(x)
        v = np.zeros_like(x)
        w_data = cfg.mosca_data_weight
        w_time = cfg.mosca_time_weight
        w_bone = cfg.mosca_bone_weight

        for _ in range(cfg.mosca_iters):
            grad = w_data * mask * (x - joints3d)

            if x.shape[0] >= 3:
                d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
                grad[2:] += w_time * d2
                grad[1:-1] += -2.0 * w_time * d2
                grad[:-2] += w_time * d2

            for i, j, length in pairs:
                vec = x[:, i] - x[:, j]
                norm = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6)
                rel = (norm[:, 0] - length) / length
                coeff = (2.0 * w_bone * rel / (length * norm[:, 0]))[:, None]
                grad[:, i] += coeff * vec
                grad[:, j] -= coeff * vec

            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            x -= cfg.mosca_step * m / (np.sqrt(v) + 1e-8)

        return FourDResult(
            joints3d=x,
            bone_error_before=bone_before,
            bone_error_after=self._bone_error(x, pairs),
            jitter_before=jitter_before,
            jitter_after=self._jitter(x),
            filled_frames=filled,
            iterations=cfg.mosca_iters,
        )


def heading_series(joints3d: np.ndarray) -> np.ndarray:
    """
    由肩轴与髋轴估计人体朝向角（弧度，已解缠绕）。

    转身阶段朝向角的变化率是判断转身质量的关键观测。
    """
    lat_hip = joints3d[:, JOINT_INDEX["l_hip"], :2] - joints3d[:, JOINT_INDEX["r_hip"], :2]
    lat_sho = joints3d[:, JOINT_INDEX["l_shoulder"], :2] - joints3d[:, JOINT_INDEX["r_shoulder"], :2]
    lat = 0.5 * (lat_hip + lat_sho)
    angle = np.arctan2(lat[:, 1], lat[:, 0])
    return np.unwrap(angle)


def phase_summary(phase: np.ndarray, fps: float) -> Dict[str, float]:
    """统计各阶段持续时间（秒）。"""
    labels, counts = np.unique(phase, return_counts=True)
    return {str(label): float(count / fps) for label, count in zip(labels, counts)}
