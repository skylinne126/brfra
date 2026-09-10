"""
图表绘制。

包含控制台文本截图渲染（模拟终端截图）与五类实验结果图：
传统方法评估结果、三维重心重建、MoSca 四维 TUG 轨迹与阶段切分、
GaitLLM 步态句子分析、两组方法综合对比。
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .style import PALETTE

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# --------------------------------------------------------------------------
# 文本截图渲染（替代终端/IDE 截图，便于复现）
# --------------------------------------------------------------------------
def render_text_shot(lines: Sequence[str], path: Path,
                     font_size: int = 20, padding: int = 24,
                     background: str = "#FFFFFF") -> Path:
    """把控制台输出渲染成等宽字体图片，模拟终端截图。"""
    from PIL import Image, ImageDraw

    from .style import load_pil_font

    # 等宽且支持中文（NSimSun）；非 Windows 环境自动回落
    font = load_pil_font(font_size)

    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * 1.22)
    width = max(int(font.getlength(line)) for line in lines) + 2 * padding
    width = max(width, 760)
    height = line_h * len(lines) + 2 * padding

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        color = "#1F2329"
        if line.startswith("====="):
            color = "#2E75B6"
        elif line.startswith("---"):
            color = "#8C8C8C"
        draw.text((padding, padding + i * line_h), line, font=font, fill=color)
    image.save(path)
    return path


# --------------------------------------------------------------------------
# 图 2：传统二维方法评估结果
# --------------------------------------------------------------------------
def plot_baseline_result(true_score: np.ndarray, pred_score: np.ndarray,
                         true_level: np.ndarray, pred_level: np.ndarray,
                         path: Path, r_value: float, accuracy: float) -> Path:
    """传统方法评分与真实风险对比（散点回归 + 混淆矩阵）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    colors = [PALETTE["baseline"], PALETTE["accent"], PALETTE["proposed"]]
    for level in (0, 1, 2):
        m = true_level == level
        ax.scatter(true_score[m], pred_score[m], s=34, alpha=0.78,
                   color=colors[level], edgecolor="white", linewidth=0.6,
                   label=["真实低风险", "真实中风险", "真实高风险"][level])
    lo, hi = 20, 100
    ax.plot([lo, hi], [lo, hi], "--", color="#8C8C8C", linewidth=1.1, label="理想一致线")
    coeff = np.polyfit(true_score, pred_score, 1)
    xs = np.linspace(true_score.min(), true_score.max(), 50)
    ax.plot(xs, np.polyval(coeff, xs), color=PALETTE["baseline"], linewidth=1.6,
            label=f"线性拟合 (r={r_value:.3f})")
    ax.set_xlabel("真实平衡能力评分")
    ax.set_ylabel("传统二维方法评分")
    ax.set_title("传统方法评分与真实评分对比")
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax = axes[1]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true_level, pred_level):
        cm[t, p] += 1
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "#1F2329",
                    fontsize=12)
    ax.set_xticks(range(3), ["预测低", "预测中", "预测高"])
    ax.set_yticks(range(3), ["真实低", "真实中", "真实高"])
    ax.set_title(f"风险分级混淆矩阵（准确率 {accuracy:.1%}）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 图 3：RollingDepth 三维重心重建
# --------------------------------------------------------------------------
def plot_com_reconstruction(true_com: np.ndarray, est_com: np.ndarray,
                            depth_true: np.ndarray, depth_est: np.ndarray,
                            path: Path, subject_label: str = "") -> Path:
    """三维重心轨迹重建与 95% 置信椭圆、深度估计误差。"""
    fig = plt.figure(figsize=(11.6, 4.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(true_com[:, 0] * 1000, true_com[:, 1] * 1000, true_com[:, 2] * 1000,
            color="#8C8C8C", linewidth=1.0, label="真实重心轨迹")
    ax.plot(est_com[:, 0] * 1000, est_com[:, 1] * 1000, est_com[:, 2] * 1000,
            color=PALETTE["proposed"], linewidth=1.2, alpha=0.9, label="RollingDepth 估计")
    ax.set_xlabel("左右 ML (mm)", labelpad=2)
    ax.set_ylabel("前后 AP (mm)", labelpad=2)
    ax.set_zlabel("竖直 (mm)", labelpad=2)
    ax.set_title("三维重心轨迹重建")
    ax.legend(fontsize=8)
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=8)

    ax = fig.add_subplot(1, 3, 2)
    for track, color, label in ((true_com, "#8C8C8C", "真实重心"),
                                (est_com, PALETTE["proposed"], "估计重心")):
        xy = (track[:, :2] - track[:, :2].mean(axis=0)) * 1000
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.9, alpha=0.85, label=label)
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals * 5.991)
        ax.add_patch(Ellipse(xy.mean(axis=0), width, height, angle=angle,
                             fill=False, edgecolor=color, linewidth=1.4,
                             linestyle="--"))
        ax.text(xy.mean(axis=0)[0], xy.mean(axis=0)[1],
                f"A95={np.pi * 5.991 * np.sqrt(max(np.linalg.det(cov), 0)):.0f} mm^2",
                fontsize=8, color=color)
    ax.set_aspect("equal")
    ax.set_xlabel("左右方向 (mm)")
    ax.set_ylabel("前后方向 (mm)")
    ax.set_title("水平晃动轨迹与 95% 置信椭圆")
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    err = np.abs(depth_est - depth_true).ravel()
    err = err[np.isfinite(err)] * 1000
    ax.hist(err, bins=40, color=PALETTE["proposed"], alpha=0.85, edgecolor="white")
    ax.axvline(np.median(err), color=PALETTE["baseline"], linestyle="--",
               label=f"中位误差 {np.median(err):.0f} mm")
    ax.set_xlabel("单目深度估计绝对误差 (mm)")
    ax.set_ylabel("频次")
    ax.set_title(f"深度估计误差分布{subject_label}")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.45, linestyle=":")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 图 4：MoSca 四维 TUG 轨迹与阶段切分
# --------------------------------------------------------------------------
def plot_tug_4d(joints3d: np.ndarray, com3d: np.ndarray, phase: np.ndarray,
                bounds: Dict[str, Tuple[float, float]], fps: float,
                path: Path) -> Path:
    """四维重建的 TUG 轨迹（按阶段着色）与阶段耗时。"""
    fig = plt.figure(figsize=(11.6, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    pelvis = joints3d[:, 12, :] if joints3d.shape[1] > 12 else com3d
    for name, color in PALETTE["phase"].items():
        m = phase == name
        if m.sum() == 0:
            continue
        ax.scatter(pelvis[m, 0], pelvis[m, 1], pelvis[m, 2], s=6,
                   color=color, label=name)
    ax.plot(com3d[:, 0], com3d[:, 1], com3d[:, 2], color="#4A4A4A",
            linewidth=0.8, alpha=0.6)
    ax.set_xlabel("左右 (m)", labelpad=2)
    ax.set_ylabel("前后 (m)", labelpad=2)
    ax.set_zlabel("高度 (m)", labelpad=2)
    ax.set_title("MoSca 四维重建：起立—行走—转身—坐下轨迹")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.view_init(elev=18, azim=-62)
    ax.tick_params(labelsize=8)

    ax = fig.add_subplot(1, 2, 2)
    order = ["SIT", "STAND_UP", "WALK_OUT", "TURN", "WALK_BACK", "SIT_DOWN"]
    names = {"SIT": "静坐", "STAND_UP": "起立", "WALK_OUT": "去程行走",
             "TURN": "转身", "WALK_BACK": "返程行走", "SIT_DOWN": "坐下"}
    durations = [float(np.sum(phase == k) / fps) for k in order]
    colors = [PALETTE["phase"][k] for k in order]
    bars = ax.barh([names[k] for k in order], durations, color=colors,
                   edgecolor="white", height=0.62)
    for bar, value in zip(bars, durations):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} s", va="center", fontsize=9)
    ax.set_xlabel("阶段耗时 (s)")
    ax.set_title(f"TUG 阶段切分结果（总用时 {sum(durations):.2f} s）")
    ax.grid(axis="x", alpha=0.45, linestyle=":")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 图 5：GaitLLM 步态句子分析
# --------------------------------------------------------------------------
def plot_gait_language(sentence_text: str, slot_surprise: Dict[str, float],
                       anomaly: float, perplexity: float, path: Path) -> Path:
    """步态句子 token 序列与各槽位意外度。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    tokens = sentence_text.split(" ")
    ax.axis("off")
    y = 0.92
    ax.text(0.02, 0.99, "步态句子（gait sentence）", fontsize=12, weight="bold",
            transform=ax.transAxes, va="top")
    for token in tokens:
        slot = token.split(":")[0]
        value = token.split(":")[1]
        normal = value in ("正常", "良好", "流畅", "迅速", "平稳", "直行良好", "尚可")
        color = "#2E75B6" if normal else ("#E8A33D" if value in ("偏慢", "偏短", "轻度异常", "轻度迂回", "偏长") else "#C0504D")
        ax.text(0.04, y, f"[{slot}]", fontsize=11, transform=ax.transAxes,
                va="center", color="#4A4A4A")
        ax.text(0.30, y, value, fontsize=11.5, transform=ax.transAxes,
                va="center", color=color, weight="bold")
        y -= 0.098
    ax.text(0.04, 0.03, f"困惑度 {perplexity:.2f}    步态异常分 {anomaly:.1f} / 100",
            fontsize=10.5, transform=ax.transAxes, color="#1F2329")

    ax = axes[1]
    slots = list(slot_surprise.keys())
    values = [slot_surprise[k] for k in slots]
    colors = ["#C0504D" if v > 1.2 else "#E8A33D" if v > 0.7 else "#2E75B6" for v in values]
    ax.bar(slots, values, color=colors, edgecolor="white")
    ax.set_ylabel("token 意外度（-log P）")
    ax.set_title("GaitLLM 各环节偏离正常步态的程度")
    ax.grid(axis="y", alpha=0.45, linestyle=":")
    ax.tick_params(axis="x", rotation=28, labelsize=9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 图 6：两组方法综合对比
# --------------------------------------------------------------------------
def plot_method_comparison(baseline: Dict[str, float], proposed: Dict[str, float],
                           true_level: np.ndarray, base_score: np.ndarray,
                           prop_score: np.ndarray, true_score: np.ndarray,
                           path: Path) -> Path:
    """雷达图 + 评分一致性散点 + ROC 曲线。"""
    fig = plt.figure(figsize=(12.0, 4.4))

    # ---- 雷达图 ----
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    labels = ["相关性", "准确率", "macro-F1", "高风险AUC", "压力垫一致性"]
    keys = ["pearson_r", "accuracy", "macro_f1", "auc_high", "icc_cop"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for data, color, name in ((baseline, PALETTE["baseline"], "传统二维方法"),
                              (proposed, PALETTE["proposed"], "本文方法")):
        values = [max(data[k], 0.0) for k in keys]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.8, label=name)
        ax.fill(angles, values, color=color, alpha=0.16)
    ax.set_xticks(angles[:-1], labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], fontsize=7.5)
    ax.set_title("综合评价指标对比", pad=16)
    ax.legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.18, -0.08))

    # ---- 散点一致性 ----
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(true_score, base_score, s=30, alpha=0.7, color=PALETTE["baseline"],
               label=f"传统方法 (r={baseline['pearson_r']:.3f})", edgecolor="white")
    ax.scatter(true_score, prop_score, s=30, alpha=0.7, color=PALETTE["proposed"],
               label=f"本文方法 (r={proposed['pearson_r']:.3f})", edgecolor="white")
    lo, hi = 20, 100
    ax.plot([lo, hi], [lo, hi], "--", color="#8C8C8C", linewidth=1.1)
    ax.set_xlabel("真实平衡能力评分")
    ax.set_ylabel("方法估计评分")
    ax.set_title("评分一致性对比")
    ax.grid(alpha=0.45, linestyle=":")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # ---- ROC 曲线 ----
    ax = fig.add_subplot(1, 3, 3)
    high = (true_level == 2).astype(int)
    for score, color, name in ((base_score, PALETTE["baseline"], "传统二维方法"),
                               (prop_score, PALETTE["proposed"], "本文方法")):
        fpr, tpr = _roc_curve(high, -score)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#8C8C8C", linewidth=1.0)
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真阳性率")
    ax.set_title("高风险识别 ROC 曲线")
    ax.grid(alpha=0.45, linestyle=":")
    ax.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _roc_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算 ROC 曲线坐标。"""
    order = np.argsort(-scores)
    y = y_true[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])
