"""
虚拟老年受试者队列生成。

每个受试者包含四维潜在生理能力因子（姿势控制、下肢肌力、感觉整合、步态稳定性），
四者共同决定"真实平衡能力评分"与"真实跌倒风险等级"。该真值仅用于实验评价，
评估算法无法直接访问，只能通过模拟的单目视频观测反推，从而保证对照实验公平。
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import CFG, SimConfig


@dataclass
class Subject:
    """一名虚拟老年受试者。"""

    sid: int
    age: int
    height_cm: float
    mass_kg: float
    sex: str
    # 潜在生理能力因子（0~1，越大越好）
    postural_control: float        # 左右方向（ML）姿势控制能力
    ap_control: float              # 前后方向（AP）姿势控制能力
    muscle_strength: float         # 下肢肌力（起立与动态平衡）
    sensory_integration: float     # 感觉整合能力（闭眼条件下尤为关键）
    gait_stability: float          # 步态稳定性
    # 真值
    true_score: float              # 真实平衡能力评分（0~100）
    true_risk_level: int           # 0 低风险 / 1 中风险 / 2 高风险
    true_cop_a95: float            # 压力垫金标准：95% 置信椭圆面积（mm²）

    @property
    def true_risk_name(self) -> str:
        return CFG.risk_labels[self.true_risk_level]


def _truncated_normal(rng: np.random.Generator, mean: float, sd: float,
                      low: float, high: float) -> float:
    """截断正态采样，保证潜在因子落在物理合理区间内。"""
    for _ in range(50):
        value = rng.normal(mean, sd)
        if low <= value <= high:
            return float(value)
    return float(np.clip(value, low, high))


def _latent_from_age(rng: np.random.Generator, age: int, sd: float,
                     bias: float = 0.0) -> float:
    """能力因子随年龄线性衰退，并叠加个体差异。bias 用于构造偏健康的参考队列。"""
    mean = 0.82 + bias - 0.011 * (age - 65)   # 65 岁约 0.82，90 岁约 0.55
    return _truncated_normal(rng, mean, sd, 0.12, 0.98)


def classify_risk(score: float, cfg: SimConfig = CFG) -> int:
    """按平衡评分划分跌倒风险等级。"""
    low, high = cfg.risk_thresholds
    if score >= high:
        return 0
    if score >= low:
        return 1
    return 2


def generate_cohort(cfg: SimConfig = CFG) -> List[Subject]:
    """生成虚拟老年队列，真值评分与风险等级同时给出。"""
    rng = np.random.default_rng(cfg.seed)
    subjects: List[Subject] = []

    for sid in range(cfg.n_subjects):
        age = int(rng.integers(cfg.age_range[0], cfg.age_range[1] + 1))
        height = float(rng.uniform(*cfg.height_range))
        mass = float(rng.uniform(*cfg.mass_range))
        sex = "男" if rng.random() < 0.42 else "女"

        # 四维潜在能力：
        # 左右方向控制与感觉整合、肌力、步态稳定性相互关联；
        # 前后方向控制（AP）与左右方向控制仅中度相关（相关系数约 0.5），
        # 这一点是两组对照实验差异的关键来源——二维投影几乎观测不到 AP 晃动。
        posture = _latent_from_age(rng, age, 0.14, cfg.healthy_bias)
        ap_control = float(np.clip(0.50 * posture + 0.50 * _latent_from_age(
            rng, age, 0.16, cfg.healthy_bias), 0.10, 0.98))
        sensory = float(np.clip(0.55 * posture + 0.45 * _latent_from_age(
            rng, age, 0.15, cfg.healthy_bias), 0.12, 0.98))
        strength = _latent_from_age(rng, age, 0.14, cfg.healthy_bias)
        gait = float(np.clip(0.45 * strength + 0.55 * _latent_from_age(
            rng, age, 0.14, cfg.healthy_bias), 0.12, 0.98))

        # 真实平衡能力评分：文献经验的加权组合（0~100）
        true_score = 100.0 * (0.20 * posture + 0.22 * ap_control + 0.14 * sensory
                              + 0.16 * strength + 0.28 * gait)
        true_score += float(rng.normal(0.0, 1.8))          # 个体内随机波动
        true_score = float(np.clip(true_score, 5.0, 99.0))

        # 压力垫金标准：闭眼单脚站 95% 置信椭圆面积（mm²），能力越差面积越大
        cop_a95 = float(np.exp(np.log(180.0) + 1.6 * (1.0 - posture)
                               + 1.6 * (1.0 - ap_control)
                               + rng.normal(0.0, 0.18)))

        subjects.append(Subject(
            sid=sid, age=age, height_cm=height, mass_kg=mass, sex=sex,
            postural_control=posture, ap_control=ap_control,
            muscle_strength=strength,
            sensory_integration=sensory, gait_stability=gait,
            true_score=true_score,
            true_risk_level=classify_risk(true_score, cfg),
            true_cop_a95=cop_a95,
        ))

    return subjects


def cohort_statistics(subjects: List[Subject]) -> dict:
    """队列统计信息，用于报告首段描述。"""
    ages = np.array([s.age for s in subjects], dtype=float)
    scores = np.array([s.true_score for s in subjects], dtype=float)
    levels = np.array([s.true_risk_level for s in subjects], dtype=int)
    return {
        "n": len(subjects),
        "age_mean": float(ages.mean()),
        "age_std": float(ages.std()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "n_low": int((levels == 0).sum()),
        "n_mid": int((levels == 1).sum()),
        "n_high": int((levels == 2).sum()),
    }


def truth_arrays(subjects: List[Subject]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取真值数组：评分、风险等级、压力垫金标准面积。"""
    scores = np.array([s.true_score for s in subjects], dtype=float)
    levels = np.array([s.true_risk_level for s in subjects], dtype=int)
    cop = np.array([s.true_cop_a95 for s in subjects], dtype=float)
    return scores, levels, cop
