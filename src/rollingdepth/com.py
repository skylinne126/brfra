"""
三维重心（COM）估计与轨迹预处理。

由 RollingDepth 恢复的三维关节坐标，按 Dempster 分段质量分布加权得到人体重心，
再按压力中心（COP）分析的标准流程做低通滤波与去均值处理，供晃动指标计算使用。
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt

from config import CFG, SimConfig
from src.rollingdepth.anthropometry import STANDARD_MASS_RATIOS, compute_com


@dataclass
class ComResult:
    """重心估计结果。"""

    com_raw: np.ndarray        # (T, 3) 未滤波重心（米）
    com_filtered: np.ndarray   # (T, 3) 低通滤波后重心
    com_centered: np.ndarray   # (T, 3) 去均值后的水平晃动分量（米）
    fps: float

    @property
    def com_xy(self) -> np.ndarray:
        return self.com_centered[:, :2]


class ComEstimator:
    """分段质量加权的三维重心估计器。"""

    def __init__(self, fps: float, cfg: SimConfig = CFG):
        self.fps = float(fps)
        self.cfg = cfg

    def _lowpass(self, signal: np.ndarray) -> np.ndarray:
        """零相位 Butterworth 低通滤波（截止频率默认 6 Hz）。"""
        nyq = 0.5 * self.fps
        cutoff = min(self.cfg.lowpass_cutoff_hz, 0.45 * self.fps)
        if self.fps <= 2.0 * cutoff + 1e-6:
            return signal.copy()
        b, a = butter(4, cutoff / nyq, btype="low")
        pad = min(signal.shape[0] - 1, 3 * max(len(a), len(b)))
        return filtfilt(b, a, signal, axis=0, padlen=max(pad, 1))

    def estimate(self, joints3d: np.ndarray,
                 mass_ratios: Optional[Dict[str, float]] = None) -> ComResult:
        """
        由三维关节序列估计重心轨迹。

        joints3d : (T, J, 3)
        """
        ratios = mass_ratios if mass_ratios is not None else STANDARD_MASS_RATIOS
        com_raw = compute_com(joints3d, ratios)
        com_filt = self._lowpass(com_raw)
        centered = com_filt - com_filt.mean(axis=0, keepdims=True)
        return ComResult(com_raw=com_raw, com_filtered=com_filt,
                         com_centered=centered, fps=self.fps)


def com_from_joints(joints3d: np.ndarray,
                    mass_ratios: Optional[Dict[str, float]] = None) -> np.ndarray:
    """便捷函数：直接由关节序列计算重心（不做滤波）。"""
    return compute_com(joints3d, mass_ratios)
