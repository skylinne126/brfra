"""
重心晃动（postural sway）量化分析。

在三维重心轨迹基础上计算静态平衡的标准量化指标：
晃动幅度（RMS、极差）、95% 置信椭圆面积、晃动路径长度与平均摆动速度、
中位功率频率，以及睁眼/闭眼对照的 Romberg 商。
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
from scipy.signal import welch

CHI2_95_2DOF = 5.991   # 二维自由度 95% 卡方分位数


@dataclass
class SwayMetrics:
    """静态平衡晃动指标（位移单位 mm，速度单位 mm/s，频率单位 Hz）。"""

    duration: float          # 有效分析时长（s）
    n_samples: int
    rms_ml: float            # 左右方向晃动 RMS
    rms_ap: float            # 前后方向晃动 RMS
    rms_total: float         # 合矢量晃动 RMS
    range_ml: float          # 左右方向极差
    range_ap: float          # 前后方向极差
    a95: float               # 95% 置信椭圆面积
    path_length: float       # 晃动路径总长度
    mean_velocity: float     # 平均摆动速度（摆动路径/时长）
    mpf: float               # 中位功率频率
    mean_height: float       # 平均重心高度（m，用于竖直方向稳定性描述）

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SwayAnalyzer:
    """重心晃动指标计算器。"""

    def __init__(self, fps: float, detrend: bool = True):
        self.fps = float(fps)
        self.detrend = detrend

    def analyze(self, com: np.ndarray) -> SwayMetrics:
        """
        计算晃动指标。

        com : (T, 3) 重心轨迹（米），水平面为 x（左右 ML）、y（前后 AP）。
        """
        com = np.asarray(com, dtype=float)
        n = com.shape[0]
        duration = n / self.fps if n > 1 else 1.0 / self.fps
        xy = com[:, :2] - (com[:, :2].mean(axis=0, keepdims=True) if self.detrend else 0.0)

        rms_ml = float(np.sqrt(np.mean(xy[:, 0] ** 2)) * 1000.0)
        rms_ap = float(np.sqrt(np.mean(xy[:, 1] ** 2)) * 1000.0)
        rms_total = float(np.sqrt(np.mean(xy[:, 0] ** 2 + xy[:, 1] ** 2)) * 1000.0)
        range_ml = float((xy[:, 0].max() - xy[:, 0].min()) * 1000.0)
        range_ap = float((xy[:, 1].max() - xy[:, 1].min()) * 1000.0)

        # 95% 置信椭圆面积
        if n > 2:
            cov = np.cov(xy.T)
            area = float(np.pi * CHI2_95_2DOF * np.sqrt(max(np.linalg.det(cov), 0.0)) * 1e6)
        else:
            area = 0.0

        # 摆动路径与平均速度
        if n > 1:
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            path = float(step.sum() * 1000.0)
            velocity = float(step.sum() / duration * 1000.0)
        else:
            path, velocity = 0.0, 0.0

        return SwayMetrics(
            duration=float(duration), n_samples=int(n),
            rms_ml=rms_ml, rms_ap=rms_ap, rms_total=rms_total,
            range_ml=range_ml, range_ap=range_ap, a95=area,
            path_length=path, mean_velocity=velocity,
            mpf=self._median_power_frequency(xy),
            mean_height=float(com[:, 2].mean()),
        )

    def _median_power_frequency(self, xy: np.ndarray) -> float:
        """中位功率频率：功率谱累计功率达到 50% 处对应的频率。"""
        n = xy.shape[0]
        if n < 16 or self.fps <= 0:
            return 0.0
        nperseg = int(min(n, max(16, 2 ** int(np.floor(np.log2(n / 4))))))
        freqs, psd = welch(xy, fs=self.fps, nperseg=nperseg, axis=0)
        power = psd.sum(axis=1)
        total = power.sum()
        if total <= 0:
            return 0.0
        cumulative = np.cumsum(power) / total
        return float(np.interp(0.5, cumulative, freqs))


def romberg_quotient(ec: SwayMetrics, eo: SwayMetrics,
                     key: str = "a95") -> float:
    """
    Romberg 商：闭眼指标 / 睁眼指标。

    数值越大说明受试者对视觉信息的依赖越强，是跌倒风险的重要判据。
    """
    base = getattr(eo, key)
    if base <= 1e-9:
        return float("nan")
    return float(getattr(ec, key) / base)


def composite_sway_index(metrics: SwayMetrics,
                         reference: Optional[SwayMetrics] = None) -> float:
    """
    综合晃动指数：把幅度、速度、面积三类指标归一化后加权求和（越大越差）。

    参考指标用于个体内归一化，缺省时使用经验标度。
    """
    ref_a95 = reference.a95 if reference else 400.0
    ref_vel = reference.mean_velocity if reference else 40.0
    ref_rms = reference.rms_total if reference else 12.0
    return float(
        0.40 * metrics.a95 / max(ref_a95, 1e-6)
        + 0.35 * metrics.mean_velocity / max(ref_vel, 1e-6)
        + 0.25 * metrics.rms_total / max(ref_rms, 1e-6)
    )
