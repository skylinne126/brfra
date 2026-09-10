"""
步态序列离散化：把连续的步态参数转化为"步态句子"。

论文出处
--------
#837  Yang S, Wang J, Hou S, et al. Bridging Gait Recognition and Large
      Language Models Sequence Modeling[C]. Proceedings of the IEEE/CVF
      Conference on Computer Vision and Pattern Recognition (CVPR), 2025:
      3460-3469.

原论文提出 GaitLLM：用 Gait-to-Language（G2L）模块把步态序列转换为适合大语言
模型的文本形式，再用 Language-to-Gait（L2G）模块把模型输出映射回步态特征空间。
本模块复现其中的 G2L 思路——“步态序列 → 文本 token 序列”。

复现要点
--------
GaitLLM 的核心思想是把步态视为一种"语言"：先由步态参数得到离散 token，
再按时间/动作顺序拼接成句子，最后交给语言模型判断"这句话是否正常"。

本模块按固定槽位顺序（起立 → 步频 → 步长 → 步宽 → 对称性 → 转身 →
坐下 → 路径 → 摆动）把 TUG 指标量化成 token 序列，
量化阈值取自步态分析文献的常用切点。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.mosca.tug import TugMetrics

# 槽位顺序：与 TUG 动作的时间顺序一致
SLOT_ORDER: Tuple[str, ...] = (
    "起立", "步频", "步长", "步宽", "对称性", "转身", "坐下", "路径", "摆动",
)

# 每个槽位的量化分箱：(上界, token 名称)，按数值升序排列
BIN_TABLE: Dict[str, List[Tuple[float, str]]] = {
    "起立": [(1.5, "起立:迅速"), (2.5, "起立:正常"), (3.5, "起立:缓慢"), (1e9, "起立:很缓慢")],
    "步频": [(80.0, "步频:很慢"), (95.0, "步频:偏慢"), (120.0, "步频:正常"), (1e9, "步频:偏快")],
    "步长": [(0.35, "步长:很短"), (0.50, "步长:偏短"), (0.75, "步长:正常"), (1e9, "步长:偏长")],
    "步宽": [(0.06, "步宽:偏窄"), (0.15, "步宽:正常"), (0.22, "步宽:偏宽"), (1e9, "步宽:很宽")],
    "对称性": [(0.10, "对称:良好"), (0.25, "对称:轻度异常"), (1e9, "对称:明显异常")],
    "转身": [(1.8, "转身:流畅"), (3.0, "转身:尚可"), (4.5, "转身:迟疑"), (1e9, "转身:明显迟疑")],
    "坐下": [(1.5, "坐下:平稳"), (2.5, "坐下:正常"), (1e9, "坐下:缓慢")],
    # 路径效率为"越大越好"，单独处理
    "摆动": [(0.55, "摆动:正常"), (0.70, "摆动:偏长"), (1e9, "摆动:明显延长")],
}


@dataclass
class GaitSentence:
    """一条"步态句子"。"""

    tokens: List[str]
    features: Dict[str, float] = field(default_factory=dict)
    subject_id: int = -1

    def text(self, separator: str = " ") -> str:
        return separator.join(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


class GaitTokenizer:
    """步态参数 → 步态 token 序列。"""

    @staticmethod
    def _bin(value: float, bins: List[Tuple[float, str]]) -> str:
        for upper, token in bins:
            if value <= upper:
                return token
        return bins[-1][1]

    def tokenize(self, metrics: TugMetrics, subject_id: int = -1) -> GaitSentence:
        """由 TUG 指标构造步态句子。"""
        # 摆动期占比：以步频与步长推算的单步周期中摆动所占比例（近似）
        swing_ratio = self._swing_ratio(metrics)

        values = {
            "起立": metrics.stand_up_time,
            "步频": metrics.cadence_spm,
            "步长": metrics.step_length,
            "步宽": metrics.step_width,
            "对称性": metrics.step_asymmetry,
            "转身": metrics.turn_time,
            "坐下": metrics.sit_down_time,
            "路径": metrics.path_efficiency,
            "摆动": swing_ratio,
        }

        tokens: List[str] = []
        for slot in SLOT_ORDER:
            if slot == "路径":
                # 路径效率越大越好，分箱方向相反
                value = values[slot]
                if value >= 0.85:
                    tokens.append("路径:直行良好")
                elif value >= 0.65:
                    tokens.append("路径:轻度迂回")
                else:
                    tokens.append("路径:明显迂回")
            else:
                tokens.append(self._bin(values[slot], BIN_TABLE[slot]))
        return GaitSentence(tokens=tokens, features=values, subject_id=subject_id)

    @staticmethod
    def _swing_ratio(metrics: TugMetrics) -> float:
        """
        由步频与步长近似估计摆动期占比。

        步频越低、步长越短，通常对应更长的双支撑期与更短的摆动期。
        """
        if metrics.cadence_spm <= 1.0:
            return 1.0
        cycle = 60.0 / metrics.cadence_spm            # 单步周期（s）
        stride = max(metrics.step_length, 0.1)
        # 经验关系：正常步态摆动期约占步态周期 40%
        ratio = 0.40 * (stride / 0.55) / max(cycle / 0.55, 1e-6)
        return float(min(max(ratio, 0.15), 0.75))
