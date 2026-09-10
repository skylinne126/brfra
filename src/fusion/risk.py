"""
多源特征融合与跌倒风险分级（第二组实验）。

把静态平衡（三维重心晃动）、动态平衡（TUG 阶段耗时）与步态语言
（GaitLLM 异常分）三类观测量统一组织为特征向量，再由 src/scoring.py
中的参考队列标定参数映射为 0~100 平衡能力评分与跌倒风险等级。

设计说明
--------
评分权重采用文献经验权重而非在测试集上学习，目的是让两组对照实验的差异
完全来自"观测量本身的信息量"（是否具备三维重心、四维轨迹与步态语言），
而不是来自更强的拟合能力，从而保证对照公平。
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from config import CFG, SimConfig
from src.mosca.tug import TugMetrics
from src.rollingdepth.sway import SwayMetrics
from src.scoring import PROPOSED_SPECS, ScoreCalibration


def proposed_features(static: SwayMetrics, hold_time: float, dynamic: TugMetrics,
                      gait_anomaly: float, romberg: float) -> Dict[str, float]:
    """提取本文方法的 13 维特征向量。"""
    return {
        "a95_3d": float(static.a95),
        "rms_ap": float(static.rms_ap),
        "velocity": float(static.mean_velocity),
        "romberg": float(romberg) if np.isfinite(romberg) else 2.5,
        "hold": float(hold_time),
        "tug": float(dynamic.total_time),
        "stand": float(dynamic.stand_up_time),
        "turn": float(dynamic.turn_time),
        "cadence": float(dynamic.cadence_spm),
        "step_length": float(dynamic.step_length),
        "asymmetry": float(dynamic.step_asymmetry),
        "efficiency": float(dynamic.path_efficiency),
        "gait_anomaly": float(gait_anomaly),
    }


@dataclass
class RiskAssessment:
    """融合评估结果。"""

    balance_score: float
    risk_level: int
    risk_name: str
    penalties: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        out = {"balance_score": self.balance_score, "risk_level": self.risk_level}
        out.update({f"penalty_{k}": v for k, v in self.penalties.items()})
        out.update({f"feature_{k}": v for k, v in self.features.items()})
        return out


def assess(features: Dict[str, float], calibration: ScoreCalibration,
           cfg: SimConfig = CFG) -> RiskAssessment:
    """由特征向量与标定参数给出平衡评分与风险等级。"""
    score, level, penalties = calibration.score(features)
    return RiskAssessment(balance_score=score, risk_level=level,
                          risk_name=cfg.risk_labels[level],
                          penalties=penalties, features=features)


def default_calibration() -> ScoreCalibration:
    """未经参考队列标定时的默认参数（仅用于单元调试）。"""
    return ScoreCalibration(name="本文方法", specs=PROPOSED_SPECS)
