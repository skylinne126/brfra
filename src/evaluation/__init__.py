"""评价模块：与真值及金标准的一致性、分类性能统计。"""

from .metrics import EvaluationResult, evaluate, icc_2_1, confusion_matrix

__all__ = ["EvaluationResult", "evaluate", "icc_2_1", "confusion_matrix"]
