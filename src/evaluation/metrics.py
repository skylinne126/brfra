"""
评价指标：评分相关性、风险分级准确性与金标准一致性。

全部指标用 numpy 实现，不依赖额外机器学习库，便于复现。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩次（用于 Spearman 相关）。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # 处理并列
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i + 1, j + 2, dtype=float))
        i = j + 1
    return ranks


def pearson_r(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_r(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson_r(_rankdata(np.asarray(a, dtype=float)), _rankdata(np.asarray(b, dtype=float)))


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_class: int = 3) -> np.ndarray:
    cm = np.zeros((n_class, n_class), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_class: int = 3) -> float:
    cm = confusion_matrix(y_true, y_pred, n_class)
    f1s = []
    for c in range(n_class):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(np.mean(f1s))


def cohen_kappa(y_true: Sequence[int], y_pred: Sequence[int], n_class: int = 3) -> float:
    cm = confusion_matrix(y_true, y_pred, n_class).astype(float)
    n = cm.sum()
    if n <= 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = float(np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n))
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else float("nan")


def roc_auc(binary_true: Sequence[int], scores: Sequence[float]) -> float:
    """基于秩次的 AUC（Mann-Whitney U 统计量）。"""
    y = np.asarray(binary_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def icc_2_1(a: Sequence[float], b: Sequence[float]) -> float:
    """
    ICC(2,1)：双向随机效应、单一测量的一致性相关系数。

    用于评价评估结果与压力垫金标准之间的一致性。
    """
    x = np.stack([np.asarray(a, dtype=float), np.asarray(b, dtype=float)], axis=1)
    n, k = x.shape
    if n < 3:
        return float("nan")
    grand = x.mean()
    ms_rows = k * np.sum((x.mean(axis=1) - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((x.mean(axis=0) - grand) ** 2) / (k - 1)
    residual = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-12:
        return float("nan")
    return float((ms_rows - ms_error) / denom)


@dataclass
class EvaluationResult:
    """一组方法的评价结果。"""

    name: str
    pearson_r: float
    spearman_r: float
    mae: float
    rmse: float
    accuracy: float
    macro_f1: float
    kappa: float
    auc_high: float
    icc_cop: float
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=int))

    def to_dict(self) -> Dict[str, float]:
        return {
            "pearson_r": self.pearson_r, "spearman_r": self.spearman_r,
            "mae": self.mae, "rmse": self.rmse, "accuracy": self.accuracy,
            "macro_f1": self.macro_f1, "kappa": self.kappa,
            "auc_high": self.auc_high, "icc_cop": self.icc_cop,
        }

    def summary(self) -> str:
        return (f"{self.name}：r={self.pearson_r:.3f} ρ={self.spearman_r:.3f} "
                f"MAE={self.mae:.2f} 准确率={self.accuracy:.3f} "
                f"macro-F1={self.macro_f1:.3f} κ={self.kappa:.3f} "
                f"高风险AUC={self.auc_high:.3f} 压力垫ICC={self.icc_cop:.3f}")


def evaluate(name: str, y_true_score: Sequence[float], y_pred_score: Sequence[float],
             y_true_level: Sequence[int], y_pred_level: Sequence[int],
             cop_reference: Sequence[float], pred_cop_proxy: Sequence[float]) -> EvaluationResult:
    """
    计算一组方法的完整评价指标。

    cop_reference 为压力垫金标准晃动面积，pred_cop_proxy 为方法估计的晃动面积。
    """
    yt = np.asarray(y_true_score, dtype=float)
    yp = np.asarray(y_pred_score, dtype=float)
    tl = np.asarray(y_true_level, dtype=int)
    pl = np.asarray(y_pred_level, dtype=int)
    high_true = (tl == 2).astype(int)
    # 高风险得分：风险等级越高、评分越低，越可能是高风险
    high_score = -yp

    return EvaluationResult(
        name=name,
        pearson_r=pearson_r(yt, yp),
        spearman_r=spearman_r(yt, yp),
        mae=float(np.mean(np.abs(yt - yp))),
        rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
        accuracy=float(np.mean(tl == pl)),
        macro_f1=macro_f1(tl, pl),
        kappa=cohen_kappa(tl, pl),
        auc_high=roc_auc(high_true, high_score),
        icc_cop=icc_2_1(np.log(cop_reference), np.log(pred_cop_proxy)),
        confusion=confusion_matrix(tl, pl),
    )
