"""
方案 8-A 平衡能力与跌倒风险评估——代码复现主程序。

两组对照实验
------------
第一组（传统基线）：仅用单目二维关键点 + 身高标定估计晃动，配合临床阈值评分。
第二组（本文方案）：RollingDepth 三维重心 + MoSca 四维轨迹重建 +
                   GaitLLM 步态语言建模，多源特征融合评分。

两组实验使用完全相同的虚拟受试者队列与完全相同的传感观测数据；
评分标定在独立的参考队列上完成（见 src/scoring.py），保证对照公平。
运行后自动在 output/ 目录生成指标文件、控制台文本与全部图表。

用法
----
    python main.py                 # 完整队列（默认 120 人）
    python main.py --n 30          # 快速试跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 保证中文与单位符号在 Windows 控制台下正常输出
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from config import CFG, SimConfig
from src.baseline import TwoDSwayEstimator, baseline_features
from src.evaluation import evaluate
from src.fusion import assess, proposed_features
from src.gaitllm import GaitLanguageModel, GaitTokenizer, ReportWriter
from src.mosca import TugAnalyzer, TugPhaseSegmenter, Trajectory4DOptimizer
from src.rollingdepth import ComEstimator, MonocularDepthEstimator, SwayAnalyzer
from src.rollingdepth.sway import romberg_quotient
from src.scoring import (BASELINE_SPECS, PROPOSED_SPECS, ScoreCalibration,
                         compute_anchors, fit_affine)
from src.simulation import (MonocularCamera, generate_cohort,
                            simulate_quiet_stance, simulate_sensor_stream,
                            simulate_single_leg_stance, simulate_tug)
from src.simulation.cohort import Subject, cohort_statistics
from src.visualization import (plot_baseline_result, plot_com_reconstruction,
                               plot_gait_language, plot_method_comparison,
                               plot_tug_4d, render_text_shot, setup_style)


# ==========================================================================
# 单名受试者的完整评估流程
# ==========================================================================
class SubjectPipeline:
    """封装一名受试者的两组评估流程，避免重复仿真。"""

    def __init__(self, cfg: SimConfig = CFG):
        self.cfg = cfg
        self.camera = MonocularCamera.from_config(cfg)
        self.depth = MonocularDepthEstimator(self.camera, cfg)
        self.com = ComEstimator(cfg.fps, cfg)
        self.sway = SwayAnalyzer(cfg.fps, cfg.detrend)
        self.mosca = Trajectory4DOptimizer(cfg)
        self.segmenter = TugPhaseSegmenter(cfg.fps, cfg)
        self.tug = TugAnalyzer(cfg.fps, cfg)
        self.two_d = TwoDSwayEstimator(self.camera, cfg)
        self.tokenizer = GaitTokenizer()

    # ---------------- 仿真与观测 ----------------
    def simulate(self, subject: Subject) -> Dict[str, object]:
        """生成该受试者三类测试的真值轨迹与单目观测。"""
        cfg = self.cfg
        height_m = subject.height_cm / 100.0
        quiet = simulate_quiet_stance(subject, cfg, eyes_closed=False)
        sls = simulate_single_leg_stance(subject, cfg, eyes_closed=True)
        tug = simulate_tug(subject, cfg)

        seed = cfg.subject_seed(subject.sid)
        streams = {
            "quiet": simulate_sensor_stream(quiet["joints3d"], cfg, self.camera, seed=seed + 1),
            "sls": simulate_sensor_stream(sls["joints3d"], cfg, self.camera, seed=seed + 2),
            "tug": simulate_sensor_stream(tug["joints3d"], cfg, self.camera, seed=seed + 3),
        }
        return {"height_m": height_m, "quiet": quiet, "sls": sls, "tug": tug,
                "streams": streams}

    # ---------------- 第一组：传统二维方法 ----------------
    def run_baseline(self, sim: Dict[str, object]) -> Dict[str, object]:
        """传统二维方法：仅提取特征，评分由标定参数给出。"""
        height_m = sim["height_m"]
        sls_2d = self.two_d.estimate_static(sim["streams"]["sls"], height_m,
                                            sim["sls"]["hold_time"])
        tug_2d = self.two_d.estimate_tug(sim["streams"]["tug"], height_m)
        features = baseline_features(sls_2d.metrics, sls_2d.hold_time, tug_2d.tug_time)
        return {"features": features, "a95": sls_2d.metrics.a95,
                "tug_time": tug_2d.tug_time, "hold_time": sls_2d.hold_time,
                "metrics": sls_2d.metrics, "tug": tug_2d}

    def gold_cop_a95(self, sim: Dict[str, object]) -> float:
        """压力垫金标准：由真实 COP 信号计算的 95% 置信椭圆面积（mm²）。"""
        cop = np.asarray(sim["sls"]["cop"], dtype=float)      # 单位：米
        track = np.concatenate([cop - cop.mean(axis=0, keepdims=True),
                                np.zeros((cop.shape[0], 1))], axis=1)
        return float(self.sway.analyze(track).a95)

    # ---------------- 第二组：本文方法 ----------------
    def run_proposed(self, sim: Dict[str, object],
                     language_model: GaitLanguageModel | None,
                     want_artifacts: bool = False) -> Dict[str, object]:
        """RollingDepth + MoSca + GaitLLM 全流程。

        language_model 为 None 时只产出步态句子（用于构建标定语料）。
        """
        height_m = sim["height_m"]

        # ① RollingDepth：单目深度 → 三维关节 → 三维重心
        depth_sls = self.depth.estimate(sim["streams"]["sls"], height_m)
        com_sls = self.com.estimate(depth_sls.joints3d)
        static = self.sway.analyze(com_sls.com_centered)

        depth_quiet = self.depth.estimate(sim["streams"]["quiet"], height_m)
        com_quiet = self.com.estimate(depth_quiet.joints3d)
        quiet_metrics = self.sway.analyze(com_quiet.com_centered)
        romberg = romberg_quotient(static, quiet_metrics)

        # ② MoSca：四维轨迹重建 + TUG 阶段切分
        depth_tug = self.depth.estimate(sim["streams"]["tug"], height_m)
        four_d = self.mosca.reconstruct(depth_tug.joints3d, sim["streams"]["tug"].visible,
                                        height_m)
        com_tug = self.com.estimate(four_d.joints3d)
        seg = self.segmenter.segment(four_d.joints3d, com_tug.com_centered)
        dynamic = self.tug.analyze(four_d.joints3d, com_tug.com_centered, seg)

        # ③ GaitLLM：步态句子 → 语言模型异常分
        sentence = self.tokenizer.tokenize(dynamic, subject_id=-1)
        if language_model is None:
            return {"sentence": sentence, "dynamic": dynamic, "metrics": static,
                    "quiet_metrics": quiet_metrics, "romberg": romberg,
                    "hold_time": sim["sls"]["hold_time"]}

        gait_score = language_model.score(sentence)
        features = proposed_features(static, sim["sls"]["hold_time"], dynamic,
                                     gait_score.anomaly_score, romberg)

        result = {
            "features": features, "a95": static.a95,
            "hold_time": sim["sls"]["hold_time"], "tug_time": dynamic.total_time,
            "metrics": static, "quiet_metrics": quiet_metrics, "dynamic": dynamic,
            "sentence": sentence, "gait_score": gait_score, "romberg": romberg,
            "depth_mae": depth_sls.depth_mae, "joint_mae": depth_sls.joint_mae,
            "four_d": four_d, "segmentation": seg,
            "depth_sls": depth_sls, "stream_sls": sim["streams"]["sls"],
            "com_sls": com_sls, "depth_tug": depth_tug, "com_tug": com_tug,
        }
        if want_artifacts:
            result["true_sls"] = sim["sls"]
            result["true_tug"] = sim["tug"]
        return result


# ==========================================================================
# 参考队列：步态语料 + 评分标定
# ==========================================================================
def build_reference(cfg: SimConfig, pipeline: SubjectPipeline
                    ) -> Tuple[GaitLanguageModel, ScoreCalibration, ScoreCalibration]:
    """
    在独立参考队列上完成评分量程标定与 GaitLLM 语料标定。

    使用两个互不相同的参考队列：
    ① 代表性参考队列（与测试队列同分布）——用于两种方法的评分量程标定；
    ② 偏健康参考队列（社区健康老年人）——用于构建正常步态语料。
    两者随机种子与测试队列均不同，标定过程不接触测试队列标签。
    """
    # ① 评分量程标定
    ref_cfg = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + 7777,
                           "n_subjects": cfg.reference_subjects,
                           "healthy_bias": 0.0})
    cohort = generate_cohort(ref_cfg)

    base_rows, prop_rows, true_scores = [], [], []
    for subject in cohort:
        sim = pipeline.simulate(subject)
        base_rows.append(pipeline.run_baseline(sim)["features"])
        prop = pipeline.run_proposed(sim, None)
        true_scores.append(subject.true_score)
        prop_rows.append(prop)

    base_cal = ScoreCalibration(
        name="传统二维阈值法", specs=BASELINE_SPECS,
        anchors=compute_anchors(base_rows, BASELINE_SPECS))
    fit_affine(base_cal, base_rows, true_scores)

    # ② 正常步态语料标定
    lm_cfg = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + 5555,
                          "n_subjects": cfg.lm_reference_subjects,
                          "healthy_bias": cfg.reference_healthy_bias})
    lm_cohort = generate_cohort(lm_cfg)
    healthy = [s for s in lm_cohort if s.true_score >= cfg.lm_healthy_score]
    healthy = healthy[: cfg.lm_calibration_subjects]
    sentences = [pipeline.run_proposed(pipeline.simulate(s), None)["sentence"]
                 for s in healthy]
    language_model = GaitLanguageModel(cfg).fit(sentences)

    features = []
    for prop in prop_rows:
        gait_score = language_model.score(prop["sentence"])
        features.append(proposed_features(
            prop["metrics"], prop["hold_time"], prop["dynamic"],
            gait_score.anomaly_score, prop["romberg"]))
    prop_cal = ScoreCalibration(
        name="本文方法", specs=PROPOSED_SPECS,
        anchors=compute_anchors(features, PROPOSED_SPECS))
    fit_affine(prop_cal, features, true_scores)
    return language_model, base_cal, prop_cal


# ==========================================================================
# 主实验流程
# ==========================================================================
def run_experiment(cfg: SimConfig = CFG) -> Dict[str, object]:
    """执行两组对照实验并输出全部结果。"""
    t_start = time.time()
    setup_style()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SubjectPipeline(cfg)
    cohort = generate_cohort(cfg)
    stats = cohort_statistics(cohort)

    console: List[str] = []
    console.append("=" * 64)
    console.append("方案 8-A 平衡能力与跌倒风险评估 —— 代码复现实验")
    console.append("=" * 64)
    console.append(f"受试者人数：{stats['n']}    平均年龄：{stats['age_mean']:.1f} ± {stats['age_std']:.1f} 岁")
    console.append(f"真实评分分布：{stats['score_mean']:.1f} ± {stats['score_std']:.1f}"
                   f"    低/中/高风险人数：{stats['n_low']}/{stats['n_mid']}/{stats['n_high']}")
    console.append(f"采样帧率：{cfg.fps} fps    单目摄像头 {cfg.image_width}x{cfg.image_height}"
                   f"    距受试者 {cfg.camera_distance:.1f} m")
    console.append("")
    console.append(f"--- 正在独立参考队列（评分标定 {cfg.reference_subjects} 人 + "
                   f"步态语料 {cfg.lm_reference_subjects} 人）上完成标定 ---")

    language_model, base_cal, prop_cal = build_reference(cfg, pipeline)
    console.append(f"正常步态语料标定完成：平均负对数似然 "
                   f"{language_model.calib_mean:.3f} ± {language_model.calib_std:.3f}")
    console.append("")

    base_scores, base_levels, base_a95, base_tug = [], [], [], []
    prop_scores, prop_levels, prop_a95, prop_tug = [], [], [], []
    true_scores, true_levels, true_cop = [], [], []
    depth_maes, joint_maes = [], []
    bone_before, bone_after, jitter_before, jitter_after = [], [], [], []
    per_subject: List[Dict[str, object]] = []

    for i, subject in enumerate(cohort):
        sim = pipeline.simulate(subject)
        base = pipeline.run_baseline(sim)
        prop = pipeline.run_proposed(sim, language_model, want_artifacts=(i == 0))

        base_assess = assess(base["features"], base_cal, cfg)
        prop_assess = assess(prop["features"], prop_cal, cfg)

        base_scores.append(base_assess.balance_score)
        base_levels.append(base_assess.risk_level)
        base_a95.append(base["a95"]); base_tug.append(base["tug_time"])
        prop_scores.append(prop_assess.balance_score)
        prop_levels.append(prop_assess.risk_level)
        prop_a95.append(prop["a95"]); prop_tug.append(prop["tug_time"])
        true_scores.append(subject.true_score); true_levels.append(subject.true_risk_level)
        gold_cop = pipeline.gold_cop_a95(sim)
        true_cop.append(gold_cop)
        depth_maes.append(prop["depth_mae"]); joint_maes.append(prop["joint_mae"])
        bone_before.append(prop["four_d"].bone_error_before)
        bone_after.append(prop["four_d"].bone_error_after)
        jitter_before.append(prop["four_d"].jitter_before)
        jitter_after.append(prop["four_d"].jitter_after)

        per_subject.append({
            "sid": subject.sid, "age": subject.age, "sex": subject.sex,
            "true_score": round(subject.true_score, 2),
            "true_level": subject.true_risk_level,
            "true_cop_a95": round(gold_cop, 1),
            "true_tug": round(sim["tug"]["true_tug_time"], 2),
            "baseline_score": round(base_assess.balance_score, 2),
            "baseline_level": base_assess.risk_level,
            "baseline_a95": round(base["a95"], 1),
            "baseline_tug": round(base["tug_time"], 2),
            "proposed_score": round(prop_assess.balance_score, 2),
            "proposed_level": prop_assess.risk_level,
            "proposed_a95": round(prop["a95"], 1),
            "proposed_tug": round(prop["tug_time"], 2),
            "hold_time": round(prop["hold_time"], 2),
            "romberg": round(prop["romberg"], 3),
            "gait_anomaly": round(prop["gait_score"].anomaly_score, 2),
            "depth_mae_mm": round(prop["depth_mae"] * 1000, 1),
            "joint_mae_mm": round(prop["joint_mae"] * 1000, 1),
        })
        if (i + 1) % 20 == 0:
            console.append(f"  已完成 {i + 1}/{stats['n']} 名受试者…")

    console.append("")
    console.append(f"全部受试者处理完成，总用时 {time.time() - t_start:.1f} s")
    console.append("")

    base_eval = evaluate("传统二维阈值法", true_scores, base_scores, true_levels,
                         base_levels, true_cop, base_a95)
    prop_eval = evaluate("本文方法（RollingDepth+MoSca+GaitLLM）", true_scores,
                         prop_scores, true_levels, prop_levels, true_cop, prop_a95)

    console.append("=====【第一组：传统二维阈值法】=====")
    console.append(f"平衡评分与真值相关性 r：{base_eval.pearson_r:.3f}"
                   f"（Spearman ρ={base_eval.spearman_r:.3f}）")
    console.append(f"评分平均绝对误差 MAE：{base_eval.mae:.2f} 分")
    console.append(f"跌倒风险三分类准确率：{base_eval.accuracy:.3f}")
    console.append(f"macro-F1：{base_eval.macro_f1:.3f}    Cohen κ：{base_eval.kappa:.3f}")
    console.append(f"高风险识别 AUC：{base_eval.auc_high:.3f}")
    console.append(f"与压力垫金标准一致性 ICC(2,1)：{base_eval.icc_cop:.3f}")
    tug_mae_base = float(np.mean(np.abs(np.array(base_tug)
                                        - np.array([r["true_tug"] for r in per_subject]))))
    console.append(f"TUG 用时估计 MAE：{tug_mae_base:.2f} s")
    console.append("")
    console.append("=====【第二组：本文方法（RollingDepth + MoSca + GaitLLM）】=====")
    console.append(f"平衡评分与真值相关性 r：{prop_eval.pearson_r:.3f}"
                   f"（Spearman ρ={prop_eval.spearman_r:.3f}）")
    console.append(f"评分平均绝对误差 MAE：{prop_eval.mae:.2f} 分")
    console.append(f"跌倒风险三分类准确率：{prop_eval.accuracy:.3f}")
    console.append(f"macro-F1：{prop_eval.macro_f1:.3f}    Cohen κ：{prop_eval.kappa:.3f}")
    console.append(f"高风险识别 AUC：{prop_eval.auc_high:.3f}")
    console.append(f"与压力垫金标准一致性 ICC(2,1)：{prop_eval.icc_cop:.3f}")
    tug_mae_prop = float(np.mean(np.abs(np.array(prop_tug)
                                        - np.array([r["true_tug"] for r in per_subject]))))
    console.append(f"TUG 用时估计 MAE：{tug_mae_prop:.2f} s")
    console.append(f"单目深度估计平均误差：{np.mean(depth_maes) * 1000:.1f} mm")
    console.append(f"MoSca 四维重建骨骼长度误差：{np.mean(bone_before) * 1000:.1f} mm → "
                   f"{np.mean(bone_after) * 1000:.1f} mm")
    console.append(f"MoSca 四维重建时间抖动：{np.mean(jitter_before) * 1000:.2f} mm → "
                   f"{np.mean(jitter_after) * 1000:.2f} mm")
    console.append("")

    # ---------------- 示例报告 ----------------
    median_idx = int(np.argsort(true_scores)[len(true_scores) // 2])
    console.append(f"=====【示例报告：受试者 #{median_idx}】=====")
    example = pipeline.run_proposed(
        pipeline.simulate(cohort[median_idx]), language_model, want_artifacts=True)
    example_assess = assess(example["features"], prop_cal, cfg)
    report = ReportWriter().write(
        example["sentence"], example["gait_score"], example["metrics"],
        example["dynamic"], example_assess.balance_score, example_assess.risk_level,
        example_assess.risk_name,
        depth_note=f"三维重建关节平均误差 {example['joint_mae'] * 1000:.0f} mm。")
    console.append(report)

    console_text = "\n".join(console)
    (out_dir / "console_output.txt").write_text(console_text, encoding="utf-8")
    print(console_text)

    # ---------------- 图表 ----------------
    figures: Dict[str, Path] = {}
    figures["console"] = render_text_shot(
        console[:26], out_dir / "fig1_console.png", font_size=19)
    figures["baseline"] = plot_baseline_result(
        np.array(true_scores), np.array(base_scores), np.array(true_levels),
        np.array(base_levels), out_dir / "fig2_baseline.png",
        base_eval.pearson_r, base_eval.accuracy)
    figures["com"] = plot_com_reconstruction(
        example["true_sls"]["com3d"], example["com_sls"].com_centered,
        example["stream_sls"].true_depth, example["depth_sls"].depth,
        out_dir / "fig3_com.png", subject_label=f"（受试者 #{median_idx}）")
    figures["tug"] = plot_tug_4d(
        example["four_d"].joints3d, example["com_tug"].com_centered,
        example["segmentation"].phase, example["segmentation"].bounds,
        cfg.fps, out_dir / "fig4_tug4d.png")
    figures["gait"] = plot_gait_language(
        example["sentence"].text(), example["gait_score"].slot_surprise,
        example["gait_score"].anomaly_score, example["gait_score"].perplexity,
        out_dir / "fig5_gaitllm.png")
    figures["comparison"] = plot_method_comparison(
        base_eval.to_dict(), prop_eval.to_dict(), np.array(true_levels),
        np.array(base_scores), np.array(prop_scores), np.array(true_scores),
        out_dir / "fig6_comparison.png")

    # ---------------- 结果落盘 ----------------
    summary = {
        "cohort": stats,
        "baseline": base_eval.to_dict(),
        "proposed": prop_eval.to_dict(),
        "baseline_confusion": base_eval.confusion.tolist(),
        "proposed_confusion": prop_eval.confusion.tolist(),
        "baseline_tug_mae_s": tug_mae_base,
        "proposed_tug_mae_s": tug_mae_prop,
        "depth_mae_mm": float(np.mean(depth_maes) * 1000),
        "joint_mae_mm": float(np.mean(joint_maes) * 1000),
        "mosca_bone_error_mm": [float(np.mean(bone_before) * 1000),
                                float(np.mean(bone_after) * 1000)],
        "mosca_jitter_mm": [float(np.mean(jitter_before) * 1000),
                            float(np.mean(jitter_after) * 1000)],
        "calibration": {
            "baseline_anchors": {k: [float(a), float(b)] for k, (a, b)
                                 in base_cal.anchors.items()},
            "proposed_anchors": {k: [float(a), float(b)] for k, (a, b)
                                 in prop_cal.anchors.items()},
            "baseline_affine": [base_cal.affine_a, base_cal.affine_b],
            "proposed_affine": [prop_cal.affine_a, prop_cal.affine_b],
            "lm_calib_mean": language_model.calib_mean,
            "lm_calib_std": language_model.calib_std,
        },
        "runtime_sec": float(time.time() - t_start),
        "figures": {k: str(v) for k, v in figures.items()},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    header = list(per_subject[0].keys())
    with (out_dir / "subjects.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in per_subject:
            fh.write(",".join(str(row[k]) for k in header) + "\n")

    (out_dir / "example_report.txt").write_text(report, encoding="utf-8")

    return {"summary": summary, "example": example, "figures": figures,
            "baseline_eval": base_eval, "proposed_eval": prop_eval,
            "per_subject": per_subject, "console": console_text,
            "example_assess": example_assess}


def main() -> None:
    parser = argparse.ArgumentParser(description="方案 8-A 平衡能力与跌倒风险评估复现实验")
    parser.add_argument("--n", type=int, default=CFG.n_subjects, help="受试者人数")
    parser.add_argument("--seed", type=int, default=CFG.seed, help="随机种子")
    args = parser.parse_args()

    cfg = SimConfig(**{**CFG.__dict__, "n_subjects": args.n, "seed": args.seed})
    run_experiment(cfg)


if __name__ == "__main__":
    main()
