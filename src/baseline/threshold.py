"""
传统二维方法的特征提取（第一组实验基线）。

传统方法只能观测到图像平面内的二维位移，因此其"晃动面积"是二维投影面积，
与三维重心置信椭圆面积不是同一量。量纲差异由 src/scoring.py 的
参考队列标定统一处理，保证与第二组实验可比。
"""

from typing import Dict

from src.rollingdepth.sway import SwayMetrics


def baseline_features(metrics: SwayMetrics, hold_time: float,
                      tug_time: float) -> Dict[str, float]:
    """提取传统二维方法的三项特征。"""
    return {
        "a95_2d": float(metrics.a95),
        "hold": float(hold_time),
        "tug": float(tug_time),
    }
