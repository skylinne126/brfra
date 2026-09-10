"""
通俗化文本报告生成。

把三维重心晃动指标、TUG 阶段耗时、步态 token 与语言模型异常分
组织成一段面向老年人及其家属的中文说明，并给出分级建议。
"""

from typing import Dict, List

from src.gaitllm.lm import GaitScore
from src.gaitllm.tokenizer import GaitSentence
from src.mosca.tug import TugMetrics
from src.rollingdepth.sway import SwayMetrics

# 正常参考值（同龄健康老年人参考区间）
NORMAL_REF = {
    "a95": 800.0,          # 95% 置信椭圆面积（mm²）
    "velocity": 25.0,      # 平均摆动速度（mm/s）
    "tug": 12.0,           # TUG 总用时（s）
    "turn": 2.5,           # 转身耗时（s）
    "stand": 2.0,          # 起立耗时（s）
}

ADVICE = {
    0: "当前平衡能力良好，建议保持每周 3 次以上的下肢力量与平衡训练，"
       "并每年复查一次。",
    1: "存在中度跌倒风险，建议在卫生间、床边加装扶手与防滑垫，"
       "并在康复师指导下进行每周 3 次以上的平衡与肌力训练，每 6 个月复查一次。",
    2: "存在高度跌倒风险，建议尽快进行专业平衡功能与步态评估，"
       "由康复医师制定干预方案，日常活动需有人陪同，居家环境应完成适老化改造。",
}


class ReportWriter:
    """自然语言报告生成器。"""

    @staticmethod
    def _ratio_text(value: float, reference: float, unit: str,
                    higher_is_worse: bool = True) -> str:
        ratio = value / max(reference, 1e-6)
        if higher_is_worse:
            if ratio < 0.9:
                level = "低于同龄参考值"
            elif ratio < 1.3:
                level = "处于同龄参考范围"
            elif ratio < 2.0:
                level = "高于同龄参考值"
            else:
                level = "明显高于同龄参考值"
        else:
            level = "处于同龄参考范围" if ratio >= 0.9 else "低于同龄参考值"
        return f"{value:.1f}{unit}（约为参考值的 {ratio:.2f} 倍，{level}）"

    def write(self, sentence: GaitSentence, score: GaitScore,
              static: SwayMetrics, dynamic: TugMetrics,
              balance_score: float, risk_level: int,
              risk_name: str, depth_note: str = "") -> str:
        """生成完整中文报告。"""
        lines: List[str] = []
        lines.append("【平衡能力与跌倒风险评估报告】")
        lines.append("")

        lines.append("一、静态平衡（闭眼单脚站立）")
        lines.append(
            "重心晃动 95% 置信椭圆面积：" +
            self._ratio_text(static.a95, NORMAL_REF["a95"], " mm²")
        )
        lines.append(
            "平均摆动速度：" +
            self._ratio_text(static.mean_velocity, NORMAL_REF["velocity"], " mm/s")
        )
        lines.append(
            f"前后方向晃动 RMS {static.rms_ap:.1f} mm，"
            f"左右方向晃动 RMS {static.rms_ml:.1f} mm。"
        )
        if depth_note:
            lines.append(depth_note)
        lines.append("")

        lines.append("二、动态平衡（起立—行走—转身—坐下）")
        lines.append(
            "TUG 总用时：" +
            self._ratio_text(dynamic.total_time, NORMAL_REF["tug"], " s")
        )
        lines.append(
            f"其中起立 {dynamic.stand_up_time:.1f} s、行走 "
            f"{dynamic.walk_out_time + dynamic.walk_back_time:.1f} s、"
            f"转身 {dynamic.turn_time:.1f} s、坐下 {dynamic.sit_down_time:.1f} s。"
        )
        lines.append(
            f"步数 {dynamic.step_count} 步，步频 {dynamic.cadence_spm:.0f} 步/分，"
            f"平均步长 {dynamic.step_length:.2f} m，步宽 {dynamic.step_width:.2f} m。"
        )
        lines.append(
            f"转身角速度峰值 {dynamic.turn_peak_rate:.0f} 度/秒，"
            f"路径效率 {dynamic.path_efficiency:.2f}。"
        )
        lines.append("")

        lines.append("三、步态语言分析")
        lines.append("步态句子：" + sentence.text(" "))
        lines.append(
            f"语言模型困惑度 {score.perplexity:.2f}，步态异常分 {score.anomaly_score:.1f} / 100。"
        )
        top = score.top_surprise(3)
        if top:
            detail = "、".join(f"{slot}（意外度 {value:.2f}）" for slot, value in top)
            lines.append(f"偏离正常步态最明显的环节：{detail}。")
        lines.append("")

        lines.append("四、综合结论")
        lines.append(f"平衡能力评分：{balance_score:.1f} / 100")
        lines.append(f"跌倒风险等级：{risk_name}")
        lines.append("建议：" + ADVICE.get(risk_level, ADVICE[1]))
        return "\n".join(lines)
