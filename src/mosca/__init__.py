"""MoSca 复现模块：四维轨迹时序一致性优化、TUG 阶段切分与动态平衡指标。"""

from .trajectory4d import Trajectory4DOptimizer, FourDResult
from .phase import TugPhaseSegmenter, PhaseSegmentation
from .tug import TugMetrics, TugAnalyzer

__all__ = [
    "Trajectory4DOptimizer",
    "FourDResult",
    "TugPhaseSegmenter",
    "PhaseSegmentation",
    "TugMetrics",
    "TugAnalyzer",
]
