"""
评分标定（norming）与特征扣分计算——两组实验共用的评分机制。

设计动机
--------
传统二维方法测量的是"图像平面投影晃动"，本文方法测量的是"三维重心晃动"，
两者量纲与数值范围本就不同，若各自套用一套经验阈值，比较结果不可比。
因此本项目采用临床量表常用的做法：在独立的参考队列上完成量程标定，
把两种方法的原始风险指数映射到统一的 0~100 平衡能力评分与风险等级。

标定流程
--------
1. 在参考队列上分别提取两种方法的特征向量；
2. 对每个特征按队列分位数确定"良好端/不良端"锚点（默认 15%/85% 分位）；
3. 特征经线性斜坡映射为 0~1 扣分项，加权得到原始风险指数；
4. 用参考队列的真实评分均值/标准差对原始指数做仿射校正，
   使评分分布与真实平衡能力评分同量纲。

整个过程只使用独立参考队列（与测试队列不同随机种子）的信息，
不接触测试队列标签。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import CFG, SimConfig
from src.simulation.cohort import classify_risk


@dataclass(frozen=True)
class PenaltySpec:
    """一个特征扣分项的定义。"""

    key: str          # 特征名
    weight: float     # 权重
    higher_worse: bool  # True 表示数值越大越差


# 第一组（传统二维阈值法）使用的特征
BASELINE_SPECS: Tuple[PenaltySpec, ...] = (
    PenaltySpec("a95_2d", 0.40, True),      # 二维投影晃动面积
    PenaltySpec("hold", 0.30, False),       # 单脚站立保持时间
    PenaltySpec("tug", 0.30, True),         # TUG 总用时
)

# 第二组（本文方法）使用的特征
PROPOSED_SPECS: Tuple[PenaltySpec, ...] = (
    PenaltySpec("rms_ap", 0.16, True),      # 前后方向晃动 RMS（二维方法无法观测）
    PenaltySpec("a95_3d", 0.13, True),      # 三维重心晃动面积
    PenaltySpec("velocity", 0.10, True),    # 平均摆动速度
    PenaltySpec("romberg", 0.06, True),     # Romberg 商
    PenaltySpec("hold", 0.12, False),       # 单脚站立保持时间
    PenaltySpec("tug", 0.14, True),         # TUG 总用时
    PenaltySpec("stand", 0.06, True),       # 起立耗时
    PenaltySpec("turn", 0.07, True),        # 转身耗时
    PenaltySpec("step_length", 0.06, True), # 步长（不足为差）
    PenaltySpec("asymmetry", 0.05, True),   # 步态左右不对称
    PenaltySpec("gait_anomaly", 0.05, True),# GaitLLM 步态异常分
    PenaltySpec("cadence", 0.00, False),    # 步频（保留观测，权重为 0）
    PenaltySpec("efficiency", 0.00, False), # 路径效率（保留观测，权重为 0）
)


def _ramp(value: float, good: float, bad: float) -> float:
    """线性斜坡：good → 0，bad → 1。"""
    if abs(bad - good) < 1e-9:
        return 0.0
    return float(np.clip((value - good) / (bad - good), 0.0, 1.0))


@dataclass
class ScoreCalibration:
    """一种方法的评分标定参数（等效百分位等值法）。"""

    name: str
    specs: Tuple[PenaltySpec, ...]
    anchors: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    raw_reference: np.ndarray = field(default_factory=lambda: np.array([]))
    true_reference: np.ndarray = field(default_factory=lambda: np.array([]))
    affine_a: float = 1.0
    affine_b: float = 0.0

    def apply(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """由特征得到原始风险指数与逐项扣分。"""
        penalties: Dict[str, float] = {}
        weighted = 0.0
        total_w = 0.0
        for spec in self.specs:
            value = float(features.get(spec.key, float("nan")))
            if not np.isfinite(value):
                value = self.anchors.get(spec.key, (0.0, 1.0))[0]
            good, bad = self.anchors[spec.key]
            penalty = _ramp(value, good, bad) if spec.higher_worse else _ramp(value, bad, good)
            penalties[spec.key] = penalty
            weighted += spec.weight * penalty
            total_w += spec.weight
        return weighted / max(total_w, 1e-9), penalties

    def raw_index_to_score(self, raw: float) -> float:
        """
        把原始风险指数映射为 0~100 平衡评分。

        采用等效百分位等值法（equipercentile equating）：
        先在参考队列上求出该原始指数所处的百分位，再取参考队列真实评分
        在该百分位处的分位数作为最终评分。该方法保证评分分布与真实评分
        同分布，且完全保持原始指数的排序，是心理测量学中标准的量程等值方法。
        """
        if self.raw_reference.size < 4:
            return float(np.clip(self.affine_a * raw + self.affine_b, 0.0, 100.0))
        percentile = float(np.searchsorted(self.raw_reference, raw, side="left")) \
            / float(self.raw_reference.size)
        percentile = float(np.clip(percentile, 0.0, 1.0))
        return float(np.quantile(self.true_reference, percentile))

    def score(self, features: Dict[str, float]) -> Tuple[float, int, Dict[str, float]]:
        """由特征得到 0~100 平衡评分、风险等级与逐项扣分。"""
        index, penalties = self.apply(features)
        score = float(np.clip(self.raw_index_to_score(100.0 * (1.0 - index)), 0.0, 100.0))
        return score, classify_risk(score, CFG), penalties


def compute_anchors(feature_rows: Sequence[Dict[str, float]],
                    specs: Tuple[PenaltySpec, ...],
                    low_pct: float = 15.0, high_pct: float = 85.0) -> Dict[str, Tuple[float, float]]:
    """由参考队列的特征分布确定每个特征的锚点。"""
    anchors: Dict[str, Tuple[float, float]] = {}
    for spec in specs:
        values = np.array([row.get(spec.key, np.nan) for row in feature_rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 4:
            anchors[spec.key] = (0.0, 1.0)
            continue
        lo = float(np.percentile(values, low_pct))
        hi = float(np.percentile(values, high_pct))
        if abs(hi - lo) < 1e-9:
            hi = lo + max(abs(lo) * 0.2, 1.0)
        anchors[spec.key] = (lo, hi)
    return anchors


def fit_affine(calibration: ScoreCalibration,
               feature_rows: Sequence[Dict[str, float]],
               true_scores: Sequence[float]) -> ScoreCalibration:
    """
    用参考队列完成评分量程标定。

    同时保留线性仿射参数（作为样本过少时的兜底），并记录参考队列的
    原始指数分布与真实评分分布，用于等效百分位等值。
    """
    raw = np.array([100.0 * (1.0 - calibration.apply(row)[0]) for row in feature_rows])
    truth = np.asarray(true_scores, dtype=float)
    calibration.raw_reference = np.sort(raw)
    calibration.true_reference = np.sort(truth)

    raw_std = float(np.std(raw))
    if raw_std < 1e-6:
        calibration.affine_a, calibration.affine_b = 0.0, float(np.mean(truth))
    else:
        calibration.affine_a = float(np.std(truth) / raw_std)
        calibration.affine_b = float(np.mean(truth) - calibration.affine_a * np.mean(raw))
    return calibration
