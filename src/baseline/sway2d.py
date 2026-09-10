"""
传统二维单目晃动估计（第一组实验基线）。

该方案代表临床上常见的低成本做法：只用单目视频的二维关键点，
以"人体在图像中的像素高度 + 已知身高"标定像素—毫米比例，
直接取髋中点二维轨迹作为重心晃动的近似。

其固有缺陷（本实验要验证的核心问题）
------------------------------------
相机安装高度与人体重心高度接近时，图像纵轴对"前后方向（AP）位移"的
敏感度极低（本实验参数下 20 mm 的 AP 位移仅引起约 0.5 像素变化，
与关键点检测噪声同量级），因此二维方法实际上只能观测到左右方向（ML）晃动，
无法反映前后方向晃动；同时髋中点并不等于质量加权的真实重心。
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from config import CFG, JOINT_INDEX, SimConfig
from src.rollingdepth.sway import SwayAnalyzer, SwayMetrics
from src.simulation.sensors import MonocularCamera, SensorStream


@dataclass
class TwoDStaticResult:
    """二维静态平衡估计结果。"""

    metrics: SwayMetrics                 # 二维平面晃动指标
    scale_mm_per_px: np.ndarray          # 逐帧像素—毫米标定系数
    sway_m: np.ndarray                   # (T, 2) 估计的晃动位移（米）
    hold_time: float                     # 单脚站立保持时间（s）
    body_px: np.ndarray                  # 逐帧人体像素高度


@dataclass
class TwoDTugResult:
    """二维 TUG 计时结果。"""

    tug_time: float                      # 总用时（s）
    stand_start: float                   # 起立起始时刻
    sit_end: float                       # 坐下结束时刻
    hip_height_mm: np.ndarray            # 髋中点估计高度曲线（毫米）
    phase_fraction: Dict[str, float] = field(default_factory=dict)


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    """按最后一维求均值，全缺失时返回 NaN（不触发 numpy 空切片警告）。"""
    mask = np.isfinite(values)
    counts = mask.sum(axis=-1)
    sums = np.where(mask, values, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / counts
    return np.where(counts == 0, np.nan, out)


class TwoDSwayEstimator:
    """基于二维关键点与身高标定的晃动估计器。"""

    def __init__(self, camera: MonocularCamera, cfg: SimConfig = CFG):
        self.camera = camera
        self.cfg = cfg
        self.analyzer = SwayAnalyzer(cfg.fps, cfg.detrend)

    # ------------------------------------------------------------------
    def _interp_nan(self, values: np.ndarray) -> np.ndarray:
        """沿时间轴对缺失值做线性插值。"""
        out = values.copy()
        idx = np.arange(out.shape[0])
        for j in range(out.shape[1] if out.ndim > 1 else 1):
            col = out[:, j] if out.ndim > 1 else out
            good = np.isfinite(col)
            if good.sum() == 0:
                col[:] = 0.0
            elif good.sum() == 1:
                col[:] = col[good][0]
            else:
                col[:] = np.interp(idx, idx[good], col[good])
            if out.ndim > 1:
                out[:, j] = col
            else:
                out = col
        return out

    def pixel_scale(self, stream: SensorStream, height_m: float) -> np.ndarray:
        """
        以"人体在图像中的像素高度"标定像素—毫米比例。

        这是传统二维方法的标准做法，隐含假设：人体直立、正对镜头、无前后位移。
        """
        kp = stream.keypoints2d
        head_v = kp[:, JOINT_INDEX["head"], 1]
        ankle_v = _row_nanmean(kp[:, [JOINT_INDEX["l_ankle"], JOINT_INDEX["r_ankle"]], 1])
        body_px = np.abs(head_v - ankle_v)
        body_px = self._interp_nan(body_px.reshape(-1, 1)).ravel()
        body_px = np.clip(body_px, 1.0, None)
        return height_m * 1000.0 / body_px

    def hip_track_px(self, stream: SensorStream) -> np.ndarray:
        """髋中点二维像素轨迹。"""
        kp = stream.keypoints2d
        hip = _row_nanmean(kp[:, [JOINT_INDEX["l_hip"], JOINT_INDEX["r_hip"]], :])
        return self._interp_nan(hip)

    # ------------------------------------------------------------------
    def estimate_static(self, stream: SensorStream, height_m: float,
                        hold_time: float) -> TwoDStaticResult:
        """静态平衡：二维髋中点晃动 → 平面晃动指标。"""
        scale = self.pixel_scale(stream, height_m)
        hip = self.hip_track_px(stream)

        # 像素位移 → 毫米位移；图像纵轴取负以对应前后方向
        disp = hip - hip.mean(axis=0, keepdims=True)
        sway_mm = np.stack([disp[:, 0], -disp[:, 1]], axis=1) * scale[:, None]
        sway_m = sway_mm / 1000.0

        # 构造 (T,3) 伪轨迹（z 恒为 0）以复用晃动分析器
        track = np.concatenate([sway_m, np.zeros((sway_m.shape[0], 1))], axis=1)
        metrics = self.analyzer.analyze(track)

        head_v = stream.keypoints2d[:, JOINT_INDEX["head"], 1]
        ankle_v = _row_nanmean(stream.keypoints2d[:, [JOINT_INDEX["l_ankle"],
                                                   JOINT_INDEX["r_ankle"]], 1])
        body_px = np.abs(head_v - ankle_v)
        return TwoDStaticResult(metrics=metrics, scale_mm_per_px=scale,
                                sway_m=sway_m, hold_time=hold_time,
                                body_px=self._interp_nan(body_px.reshape(-1, 1)).ravel())

    def estimate_tug(self, stream: SensorStream, height_m: float) -> TwoDTugResult:
        """动态平衡：由髋中点图像高度曲线估计 TUG 总用时。"""
        scale = self.pixel_scale(stream, height_m)
        hip = self.hip_track_px(stream)
        height_mm = -hip[:, 1] * scale          # 图像纵轴向下，取负得到"高度"

        fps = self.cfg.fps
        n = height_mm.shape[0]
        win = max(3, int(0.5 * fps))
        kernel = np.ones(win) / win
        smooth = np.convolve(height_mm, kernel, mode="same")

        seated = float(np.median(smooth[: max(3, int(1.0 * fps))]))
        mid_lo, mid_hi = int(n * 0.35), int(n * 0.65)
        standing = float(np.percentile(smooth[mid_lo:mid_hi], 85))
        threshold = seated + 0.25 * (standing - seated)

        above = smooth > threshold
        idx = np.where(above)[0]
        if idx.size == 0:
            return TwoDTugResult(tug_time=float(n / fps), stand_start=0.0,
                                 sit_end=float(n / fps), hip_height_mm=height_mm)
        stand_start = float(idx[0] / fps)
        sit_end = float(idx[-1] / fps)
        return TwoDTugResult(tug_time=max(sit_end - stand_start, 0.5),
                             stand_start=stand_start, sit_end=sit_end,
                             hip_height_mm=height_mm)
