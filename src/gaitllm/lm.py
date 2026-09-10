"""
轻量步态语言模型：对"步态句子"计算似然与异常程度。

论文出处
--------
#837  Yang S, Wang J, Hou S, et al. Bridging Gait Recognition and Large
      Language Models Sequence Modeling[C]. Proceedings of the IEEE/CVF
      Conference on Computer Vision and Pattern Recognition (CVPR), 2025:
      3460-3469.

原论文用大语言模型对步态文本序列建模，其 L2G 模块把 LLM 输出映射回步态特征
空间以完成身份识别。本任务关注的是“步态偏离正常人群的程度”而非身份，因此把
L2G 分支替换为面向临床的异常判别与文本报告生成,并用可离线运行、可解释的
插值式二元语言模型替代大规模 LLM,保持“步态序列 → 语言建模”的方法学框架不变。

复现要点
--------
GaitLLM 用大语言模型对步态语句做判别；本复现以可离线运行、可解释的
插值式二元语言模型（interpolated bigram LM）替代，保持相同的方法学框架：
先在"正常步态语料"上学习 token 转移分布，再以句子的负对数似然
（困惑度）衡量该步态相对正常人群的偏离程度。

模型形式
--------
    P(t_k | t_{k-1}) = (c(t_{k-1}, t_k) + α · P_uni(t_k)) / (c(t_{k-1}) + α)
其中 α 为平滑系数，P_uni 为正常语料的一元分布。
句子得分 = Σ_k log P(t_k | t_{k-1})，再按标定语料的均值方差归一化为 0~100 异常分。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from config import CFG, SimConfig
from src.gaitllm.tokenizer import GaitSentence

START_TOKEN = "<S>"
END_TOKEN = "</S>"


@dataclass
class GaitScore:
    """一条步态句子的语言模型评分。"""

    loglik: float                       # 对数似然
    nll_per_token: float                # 每个 token 的平均负对数似然
    perplexity: float                   # 困惑度
    anomaly_score: float                # 步态异常分（0~100，越大越异常）
    slot_surprise: Dict[str, float] = field(default_factory=dict)

    def top_surprise(self, k: int = 3) -> List[Tuple[str, float]]:
        items = sorted(self.slot_surprise.items(), key=lambda kv: -kv[1])
        return items[:k]


class GaitLanguageModel:
    """步态语言模型（插值式二元模型）。"""

    def __init__(self, cfg: SimConfig = CFG):
        self.cfg = cfg
        self.alpha = cfg.lm_smoothing
        self.unigram: Dict[str, float] = {}
        self.bigram: Dict[str, Dict[str, int]] = {}
        self.prev_count: Dict[str, int] = {}
        self.calib_mean: float = 0.0
        self.calib_std: float = 1.0
        self.fitted: bool = False

    # ------------------------------------------------------------------
    def _build(self, sentences: List[GaitSentence]) -> None:
        """由语料统计一元/二元 token 分布。"""
        self.unigram = {}
        self.bigram = {}
        self.prev_count = {}
        uni_counts: Dict[str, int] = {}
        total = 0
        for sentence in sentences:
            prev = START_TOKEN
            for token in list(sentence.tokens) + [END_TOKEN]:
                uni_counts[token] = uni_counts.get(token, 0) + 1
                self.bigram.setdefault(prev, {})
                self.bigram[prev][token] = self.bigram[prev].get(token, 0) + 1
                self.prev_count[prev] = self.prev_count.get(prev, 0) + 1
                prev = token
                total += 1
        vocab_size = len(uni_counts)
        self.unigram = {t: (c + self.alpha) / (total + self.alpha * max(vocab_size, 1))
                        for t, c in uni_counts.items()}

    def fit(self, sentences: List[GaitSentence]) -> "GaitLanguageModel":
        """
        在正常步态语料上估计 token 转移分布。

        标定统计量采用留一法（leave-one-out）计算，即用"去掉该句后的语料"
        估计该句的负对数似然，从而得到无偏的样本外分布，
        避免直接使用训练集似然导致的异常分饱和。
        """
        self._build(sentences)
        self.fitted = True

        nlls: List[float] = []
        for i, sentence in enumerate(sentences):
            model = GaitLanguageModel(self.cfg)
            model._build([s for k, s in enumerate(sentences) if k != i])
            nlls.append(-model._loglik(sentence) / max(len(sentence), 1))
        if nlls:
            self.calib_mean = float(np.mean(nlls))
            self.calib_std = max(float(np.std(nlls)), 1e-3)
        return self

    # ------------------------------------------------------------------
    def _prob(self, prev: str, token: str) -> float:
        """插值式条件概率。"""
        denom = self.prev_count.get(prev, 0)
        count = self.bigram.get(prev, {}).get(token, 0)
        uni = self.unigram.get(token, 1e-6)
        return (count + self.alpha * uni) / (denom + self.alpha)

    def _loglik(self, sentence: GaitSentence) -> float:
        total = 0.0
        prev = START_TOKEN
        for token in list(sentence.tokens) + [END_TOKEN]:
            total += float(np.log(max(self._prob(prev, token), 1e-12)))
            prev = token
        return total

    def score(self, sentence: GaitSentence) -> GaitScore:
        """对一条步态句子打分。"""
        if not self.fitted:
            raise RuntimeError("步态语言模型尚未标定，请先调用 fit()。")

        loglik = self._loglik(sentence)
        nll = -loglik / max(len(sentence), 1)

        # 逐槽位"意外度"，用于解释是哪一步出现异常
        surprise: Dict[str, float] = {}
        prev = START_TOKEN
        from src.gaitllm.tokenizer import SLOT_ORDER
        for slot, token in zip(SLOT_ORDER, sentence.tokens):
            surprise[slot] = float(-np.log(max(self._prob(prev, token), 1e-12)))
            prev = token

        z = (nll - self.calib_mean) / self.calib_std
        anomaly = 100.0 * float(np.clip((z + 1.5) / 4.5, 0.0, 1.0))
        return GaitScore(
            loglik=float(loglik), nll_per_token=float(nll),
            perplexity=float(np.exp(nll)),
            anomaly_score=float(anomaly), slot_surprise=surprise,
        )
