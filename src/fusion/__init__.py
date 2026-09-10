"""特征融合模块：多源平衡指标特征提取、评分标定与跌倒风险分级。"""

from .risk import proposed_features, RiskAssessment, assess

__all__ = ["proposed_features", "RiskAssessment", "assess"]
