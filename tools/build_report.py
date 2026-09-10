"""
按模板格式生成《平衡能力与跌倒风险评估代码复现》报告（含 LaTeX 公式）。

用法
----
    python tools/build_report.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import List

from report_lib import (FOOTER_XML, FONT_BODY, FONT_HEAD, OUT_DIR, REPORT_PATH,
                        S_BODY, S_REFERENCE, S_TITLE, ReportBuilder, caption,
                        code_block, ensure_template, equation, heading, para,
                        table, template_parts, toc_field, patch_content_types,
                        update_fields_with_word)

# 项目 GitHub 仓库地址：显示在标题下方（蓝色下划线超链接）。
# 若仓库地址不同，修改此处或运行时指定 --repo-url 即可。
REPO_URL = "https://github.com/skylinne126/brfra"

# --------------------------------------------------------------------------
# 关键公式（LaTeX 源码，转换为 Word 原生公式）
# --------------------------------------------------------------------------
EQ = {
    "depth_affine": r"z_i = a\,d_i + b",
    "ray": (r"\mathbf{P}_i = k_i z_i,\qquad "
            r"k_i=\left(\frac{u_i-c_x}{f},\;-\frac{v_i-c_y}{f},\;1\right)"),
    "seg_len": (r"L_{ij}=\left\|aA_{ij}+bB_{ij}\right\|,\qquad "
                r"A_{ij}=k_i d_i-k_j d_j,\qquad B_{ij}=k_i-k_j"),
    "scale_fit": (r"\min_{a,b}\;\sum_{ij}\left(\frac{\left\|aA_{ij}+bB_{ij}"
                  r"\right\|-\hat{L}_{ij}}{\hat{L}_{ij}}\right)^{2}"
                  r"+\lambda\left(\frac{a\bar{d}+b-z_0}{z_0}\right)^{2}"),
    "bone_fit": (r"\min_{z}\;\sum_{ij}\frac{\left(\left\|\mathbf{P}_i-"
                 r"\mathbf{P}_j\right\|-\hat{L}_{ij}\right)^{2}}{\hat{L}_{ij}^{2}}"
                 r"+\lambda\sum_{j}\frac{\left(z_j-z_j^{\mathrm{prior}}\right)^{2}}"
                 r"{\left(z_j^{\mathrm{prior}}\right)^{2}}"),
    "com": (r"\mathbf{r}_{\mathrm{com}}=\frac{\sum_{m=1}^{M}w_m\mathbf{r}_m}"
            r"{\sum_{m=1}^{M}w_m}"),
    "a95": r"A_{95}=\pi\,\chi_{0.95,\,2}^{2}\sqrt{\det\boldsymbol{\Sigma}}",
    "rms": (r"\sigma_{\mathrm{ML}}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}"
            r"\left(x_t-\bar{x}\right)^{2}}"),
    "romberg": r"R=\frac{A_{95}^{\mathrm{EC}}}{A_{95}^{\mathrm{EO}}}",
    "mosca_a": (r"\min_{\mathbf{X}}\;w_{\mathrm{data}}\sum_{t}m_t\left\|\mathbf{x}_t"
                r"-\mathbf{z}_t\right\|^{2}+w_{\mathrm{time}}\sum_{t}\left\|\mathbf{x}_{t+1}"
                r"-2\mathbf{x}_t+\mathbf{x}_{t-1}\right\|^{2}"),
    "mosca_b": (r"+\;w_{\mathrm{bone}}\sum_{t}\sum_{(i,j)}\frac{\left(\left\|\mathbf{x}_t^{i}"
                r"-\mathbf{x}_t^{j}\right\|-L_{ij}\right)^{2}}{L_{ij}^{2}}"),
    "bigram": (r"P\!\left(t_k\mid t_{k-1}\right)=\frac{c\!\left(t_{k-1},t_k\right)"
               r"+\alpha\,P_{\mathrm{uni}}\!\left(t_k\right)}{c\!\left(t_{k-1}\right)+\alpha}"),
    "ppl": (r"\mathrm{PPL}=\exp\left(-\frac{1}{N}\sum_{k=1}^{N}"
            r"\ln P\!\left(t_k\mid t_{k-1}\right)\right)"),
    "zscore": r"z=\frac{\mathrm{NLL}-\mu_{\mathrm{ref}}}{\sigma_{\mathrm{ref}}}",
    "score": (r"S=100\left(1-\frac{\sum_{i}\omega_i p_i}{\sum_{i}\omega_i}\right)"),
    "equate": (r"S=F_{\mathrm{true}}^{-1}\!\left(F_{\mathrm{ref}}\!\left(S_{\mathrm{raw}}"
               r"\right)\right)"),
}


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_cover(b: ReportBuilder, repo_url: str = "") -> None:
    b.add(para("平衡能力与跌倒风险评估代码复现", style=S_TITLE, font=FONT_HEAD))
    b.add(para("Balance Ability and Fall Risk Assessment (BAFRA)", center=True,
               size=24, space_after=60))
    b.add_repo_link(repo_url)
    b.add(para("8-A 平衡能力与跌倒风险评估", center=True, font=FONT_HEAD, size=24,
               space_after=40))
    b.add(para("对标 #876 RollingDepth、#882 MoSca、#837 GaitLLM（均为 CVPR 2025）",
               center=True, font=FONT_HEAD, size=24, space_after=60))
    b.add(para("组长：牛虹锦　　组员：朱天行　昝璎珏　骆拿云　余金昕",
               center=True, size=24, space_after=200))


def build_equations(b: ReportBuilder) -> None:
    """关键公式与符号说明（正文中的符号以行内公式呈现）。"""
    b.add(heading("   2.1.3 关键公式与符号说明", 3))
    b.add(para("本节给出复现过程中涉及的核心数学模型。公式中以 $\\mathbf{P}$ 表示关节三维坐标，"
               "$k$ 表示相机射线方向向量，$z$ 表示相机坐标系下的深度，$L$ 表示骨骼分段长度，"
               "带上标“^”者表示人体测量学先验值。", math=True))

    b.add(para("（1）单目深度尺度恢复", style=S_BODY, bold=True, font=FONT_HEAD))
    b.add(para("单目深度网络只能给出相对深度 $d_i$，与真实度量深度 $z_i$ 之间相差一个仿射变换，"
               "其中 $a$、$b$ 为未知的尺度系数与偏移系数：", math=True))
    b.add(equation(EQ["depth_affine"], 1))
    b.add(para("关节像素坐标 $(u_i,\\,v_i)$ 经相机内参 $(f,\\,c_x,\\,c_y)$ 反投影后，"
               "可沿射线方向表示为：", math=True))
    b.add(equation(EQ["ray"], 2))
    b.add(para("于是人体任一分段 $(i,\\,j)$ 的三维长度可写成 $(a,\\,b)$ 的函数：", math=True))
    b.add(equation(EQ["seg_len"], 3))
    b.add(para("以人体测量学先验长度 $\\hat{L}_{ij}$（分段长度比例 × 受试者身高）为监督，"
               "并引入“人体尺寸深度”锚定项 $z_0$（由各骨骼段的像素长度反推深度的中位数给出），"
               "以抑制静止场景下的解退化，得到尺度恢复目标函数：", math=True))
    b.add(equation(EQ["scale_fit"], 4))

    b.add(para("（2）骨骼约束下的深度优化", style=S_BODY, bold=True, font=FONT_HEAD))
    b.add(para("单目深度在光轴方向上仍存在逐帧抖动。在保持二维投影像素坐标完全不变"
               "（即仅沿光轴调整各关节深度 $z_j$）的前提下求解：", math=True))
    b.add(equation(EQ["bone_fit"], 5))

    b.add(para("（3）三维重心与晃动指标", style=S_BODY, bold=True, font=FONT_HEAD))
    b.add(para("按 Dempster 分段质量分布 $w_m$ 对 $M$ 个分段质心 $\\mathbf{r}_m$ 加权，"
               "得到人体三维重心：", math=True))
    b.add(equation(EQ["com"], 6))
    b.add(para("晃动的 95% 置信椭圆面积与左右方向晃动均方根分别为："))
    b.add(equation(EQ["a95"], 7))
    b.add(equation(EQ["rms"], 8))
    b.add(para("闭眼与睁眼条件下的晃动面积之比即 Romberg 商，用于衡量受试者对视觉信息的依赖："))
    b.add(equation(EQ["romberg"], 9))

    b.add(para("（4）MoSca 四维轨迹重建", style=S_BODY, bold=True, font=FONT_HEAD))
    b.add(para("在 RollingDepth 逐帧三维关节 $\\mathbf{z}_t$ 的基础上，引入时间二阶差分平滑项与"
               "骨骼长度一致性项，求解时间上连贯的四维轨迹 $\\mathbf{X}$：", math=True))
    b.add(equation(EQ["mosca_a"], 10))
    b.add(equation(EQ["mosca_b"]))
    b.add(para("式中 $m_t$ 为可见性掩码，$w_{\\mathrm{data}}$、$w_{\\mathrm{time}}$、"
               "$w_{\\mathrm{bone}}$ 分别为数据项、时序平滑项与骨骼约束项权重。", math=True))

    b.add(para("（5）GaitLLM 步态语言模型", style=S_BODY, bold=True, font=FONT_HEAD))
    b.add(para("把步态句子视为 token 序列 $t_1,\\dots,t_N$，"
               "用插值式二元语言模型刻画其转移概率：", math=True))
    b.add(equation(EQ["bigram"], 11))
    b.add(para("句子困惑度与归一化负对数似然偏离度（步态异常分）分别为："))
    b.add(equation(EQ["ppl"], 12))
    b.add(equation(EQ["zscore"], 13))

    b.add(para("（6）多源特征融合与评分标定", style=S_BODY, bold=True, font=FONT_HEAD))
    b.add(para("把各特征经参考队列锚点归一化为扣分项 $p_i$ 后加权求和，"
               "得到 0~100 平衡能力评分：", math=True))
    b.add(equation(EQ["score"], 14))
    b.add(para("为使两种方法的评分同量纲可比，采用等效百分位等值：先在参考队列上求出"
               "原始指数所处的百分位，再取参考队列真实评分在该百分位处的分位数：", math=True))
    b.add(equation(EQ["equate"], 15))


def build(repo_url: str = REPO_URL, update_fields: bool = True) -> Path:
    """生成报告 docx。"""
    metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    console_lines = (OUT_DIR / "console_output.txt").read_text(encoding="utf-8").split("\n")
    manifest = json.loads((OUT_DIR / "code_shots" / "manifest.json").read_text(encoding="utf-8"))

    base = metrics["baseline"]
    prop = metrics["proposed"]
    cohort = metrics["cohort"]

    work = OUT_DIR / "docx_build"
    if work.exists():
        shutil.rmtree(work)
    for sub in ("_rels", "docProps", "word/_rels", "word/theme", "word/media"):
        (work / sub).mkdir(parents=True)

    for rel in template_parts():
        shutil.copyfile(ensure_template() / rel, work / rel)
    (work / "word" / "footer1.xml").write_text(FOOTER_XML, encoding="utf-8")

    b = ReportBuilder(work / "word" / "media", hyperlink_url=repo_url)

    # ---------------- 封面与目录 ----------------
    build_cover(b, repo_url)
    b.add(para("目录", center=True, font=FONT_HEAD, size=32, space_before=200, space_after=120))
    b.add(toc_field())
    b.add(para("", space_after=0))

    # ---------------- 1 复现目标 ----------------
    b.add(heading(" 1 复现目标", 1))
    b.add(para(
        "本次代码复现面向医院老年科、养老院与社区健康中心的老年人站立平衡测试场景，"
        "仿真单目摄像头（压力垫为可选的非穿戴参考传感器）对老年人静态平衡与动态平衡的"
        "无穿戴式评估过程。实现传统二维关键点阈值法复现；在此基础上完成模型与算法改进，"
        "实现基于单目深度估计的三维重心估计、四维轨迹重建与步态语言建模的跌倒风险评估方案。"
        "构建可复用的平衡能力评估仿真框架，为无穿戴式跌倒风险筛查研究提供仿真工具支撑。"
        "具体目标包括："))
    goals = [
        "复现传统二维方法：仅用单目二维关键点与身高标定像素尺度，以髋中点平面位移近似"
        "重心晃动，配合临床阈值规则完成平衡评分与风险分级，获取各项量化指标作为对照基线。",
        "复现 #876 RollingDepth（Video Depth without Video Models, CVPR 2025）："
        "该文把单帧潜扩散模型改造成多帧深度估计器，并以基于优化的配准算法把深度片段"
        "拼接为时序一致的深度视频；本项目在其方法学框架下，用骨骼长度先验消解单目深度"
        "的仿射尺度歧义，结合 Dempster 分段质量分布估计人体三维重心，替代昂贵的测力板。",
        "复现 #882 MoSca（4D Motion Scaffolds, CVPR 2025）："
        "该文把视频提升为紧凑平滑的运动支架（Motion Scaffold）表示并做全局融合优化；"
        "本项目以 20 关节人体骨架充当运动支架，对逐帧三维关节做带时序正则与骨骼结构"
        "约束的四维重建，自动切分“起立—行走—转身—坐下”全过程并提取动态平衡指标。",
        "复现 #837 GaitLLM（Bridging Gait Recognition and LLMs, CVPR 2025）："
        "该文用 Gait-to-Language 模块把步态序列转为文本交给大语言模型建模；"
        "本项目复现其“步态→语言”思路，把步态参数离散化为“步态句子”，"
        "用步态语言模型判别步态异常，并生成面向老年人及其家属的通俗化文本报告。",
        "针对传统二维方法无法观测前后方向（AP）晃动、且缺少步态质量信息的缺陷，"
        "完成两处改进：①引入三维重心与前后方向晃动指标；②引入四维轨迹与步态语言特征。",
        "搭建对照仿真实验，保持仿真环境、队列规模、摄像参数与噪声水平完全一致，"
        "对比传统方案与本文方案的指标差异，验证改进策略有效性。",
        "实现三维重心轨迹、四维 TUG 轨迹与阶段切分、步态句子分析的可视化，"
        "输出控制台实验结果，完整保存评估指标与图像结果。",
    ]
    for text in goals:
        b.add(para(text, style=S_BODY, numid=1))

    # ---------------- 2 复现流程 ----------------
    b.add(heading(" 2 复现流程", 1))
    b.add(para(
        "本项目全部逻辑基于 Python 实现，使用 numpy 完成数值仿真与数组运算、"
        "scipy 完成非线性最小二乘与信号滤波、matplotlib 完成绘图可视化、"
        "pillow 完成文本截图渲染；共设计两组对照仿真实验。"))
    b.add(para("- 第一组：传统方案，单目二维关键点 + 身高标定 + 临床阈值评分"))
    b.add(para("- 第二组：本文方案，RollingDepth 三维重心 + MoSca 四维重建 + GaitLLM 步态语言建模"))

    b.add(heading("  2.1 仿真程序", 2))
    b.add(heading("   2.1.1 环境配置", 3))
    b.add(para("安装 PyCharm 开发工具并配置 Python 环境。", style=S_BODY, numid=2))
    b.add(para("安装项目所需第三方依赖库，在终端执行：", style=S_BODY, numid=2))
    b.add(code_block(["pip install -r requirements.txt"]))
    b.add(para("主要依赖项说明：", style=S_BODY, numid=2))
    b.add(para("- numpy：虚拟队列生成、运动学计算、指标统计与数组运算；"))
    b.add(para("- scipy：单目深度尺度恢复的非线性最小二乘、四维轨迹优化、Butterworth 低通滤波；"))
    b.add(para("- matplotlib：三维重心轨迹、四维 TUG 轨迹、步态句子分析等图表绘制；"))
    b.add(para("- pillow：控制台文本截图与附录源码截图渲染。"))
    b.add(para("程序运行后会自动生成 output 文件夹，用于存储指标文件与输出图片，无需手动新建。"))

    b.add(heading("   2.1.2 代码架构", 3))
    b.add(para("项目文件组织如下："))
    tree = (
        "balance_fall_risk/\n"
        "  ├─ main.py （仿真程序主入口，串联两组对照实验）\n"
        "  ├─ config.py （全局参数：队列、摄像头、噪声、算法超参）\n"
        "  ├─ src/\n"
        "  │   ├─ scoring.py （参考队列标定与统一评分机制）\n"
        "  │   ├─ simulation/ （虚拟队列、运动学、测试动作真值、传感噪声）\n"
        "  │   ├─ baseline/ （第一组：二维晃动估计）\n"
        "  │   ├─ rollingdepth/ （第二组：单目深度、三维重心、晃动分析）\n"
        "  │   ├─ mosca/ （第二组：四维重建、TUG 阶段切分、动态指标）\n"
        "  │   ├─ gaitllm/ （第二组：步态 token 化、语言模型、报告生成）\n"
        "  │   ├─ fusion/ （多源特征融合与风险分级）\n"
        "  │   ├─ evaluation/ （相关性、准确性、AUC、ICC 等评价指标）\n"
        "  │   └─ visualization/ （中文绘图样式与全部图表）\n"
        "  ├─ tools/ （代码截图渲染与报告生成脚本）\n"
        "  └─ output/ （程序自动生成，存放指标、控制台文本与图片）"
    )
    b.add(table([[tree]], [8306]))
    b.add(para("main.py 与各模块职责划分："))
    modules = [
        "仿真层 src/simulation/：generate_cohort() 生成含真实风险标签的虚拟老年队列；"
        "simulate_quiet_stance()、simulate_single_leg_stance()、simulate_tug() 生成"
        "睁眼双脚站立、闭眼单脚站立与 TUG 三类测试的三维真值轨迹；"
        "simulate_sensor_stream() 生成含关键点抖动、随机遮挡与单目深度仿射歧义的观测流。",
        "第一组 src/baseline/：TwoDSwayEstimator 以人体像素高度标定像素—毫米比例，"
        "取髋中点平面位移作为晃动近似，并估计 TUG 用时；baseline_features() 输出三项特征。",
        "第二组（RollingDepth）src/rollingdepth/：MonocularDepthEstimator 完成单目深度"
        "尺度恢复与骨骼约束优化；ComEstimator 按 Dempster 分段质量分布计算三维重心并低通滤波；"
        "SwayAnalyzer 计算晃动 RMS、95% 置信椭圆面积、平均摆动速度与中位功率频率。",
        "第二组（MoSca）src/mosca/：Trajectory4DOptimizer 求解带时序正则与骨骼约束的四维重建；"
        "TugPhaseSegmenter 自动切分 SIT/STAND_UP/WALK_OUT/TURN/WALK_BACK/SIT_DOWN；"
        "TugAnalyzer 统计阶段耗时、步数、步频、步长、步宽、转身角速度峰值等指标。",
        "第二组（GaitLLM）src/gaitllm/：GaitTokenizer 把步态参数量化成 9 个槽位的 token 序列；"
        "GaitLanguageModel 在正常步态语料上学习二元转移分布并计算困惑度与异常分；"
        "ReportWriter 生成通俗化中文报告。",
        "融合与评价 src/fusion/、src/evaluation/：proposed_features() 汇总 13 维特征，"
        "src/scoring.py 在独立参考队列上完成量程标定并输出 0~100 平衡评分与风险等级；"
        "evaluate() 计算相关性、MAE、三分类准确率、macro-F1、Cohen κ、高风险 AUC 与压力垫 ICC。",
    ]
    for text in modules:
        b.add(para(text, style=S_BODY))
    b.add(para("> 完整全部源代码详见附录A。"))

    # ---------------- 关键公式 ----------------
    build_equations(b)

    # ---------------- 复现流程 ----------------
    b.add(heading("   2.1.4 复现流程", 3))
    b.add(para("打开项目目录，在 PyCharm 打开 main.py，直接运行即可完成两组对照实验。"))

    b.add(heading("    第一组实验：传统二维阈值法复现", 4))
    b.add(para(
        f"仿真参数设置：{cohort['n']} 名虚拟老年受试者，平均年龄 "
        f"{cohort['age_mean']:.1f} ± {cohort['age_std']:.1f} 岁，"
        f"采样帧率 30 fps，单目摄像头 1280×720，安装高度 1.45 m，距受试者 3.6 m；"
        "测试协议为睁眼双脚站立 30 s、闭眼单脚站立（保持时间由能力决定）、"
        "以及 3 m 起立—行走—转身—坐下（TUG）。"))
    b.add(para("运行 main.py，程序执行流程："))
    steps_base = [
        "生成虚拟老年队列与单目观测流（含关键点抖动、随机遮挡与深度噪声）；",
        "以人体像素高度标定像素—毫米比例，提取髋中点二维轨迹并计算平面晃动指标；",
        "由髋中点图像高度曲线估计 TUG 总用时；",
        "按参考队列标定的阈值规则计算平衡能力评分与跌倒风险等级；",
        "控制台打印第一组全部评价指标，并保存仿真图表至 output 文件夹。",
    ]
    for text in steps_base:
        b.add(para(text, style=S_BODY, numid=3))
    b.add(para("> 控制台输出文本："))
    start = next(i for i, line in enumerate(console_lines) if "第一组" in line)
    end = next(i for i, line in enumerate(console_lines) if "第二组" in line)
    b.add(code_block(console_lines[start:end - 1]))
    b.add_image(OUT_DIR / "fig1_console.png", 5.2)
    b.add(caption("图 1 传统二维阈值法控制台输出"))
    b.add(caption("表 1 传统二维阈值法评价指标", keep_next=True))
    b.add(table([
        ["评价指标", "数值"],
        ["平衡评分与真值相关性 r", _fmt(base["pearson_r"])],
        ["Spearman 秩相关 ρ", _fmt(base["spearman_r"])],
        ["评分平均绝对误差 MAE", _fmt(base["mae"], 2) + " 分"],
        ["跌倒风险三分类准确率", _fmt(base["accuracy"])],
        ["macro-F1", _fmt(base["macro_f1"])],
        ["Cohen κ", _fmt(base["kappa"])],
        ["高风险识别 AUC", _fmt(base["auc_high"])],
        ["与压力垫金标准一致性 ICC(2,1)", _fmt(base["icc_cop"])],
        ["TUG 用时估计 MAE", _fmt(metrics["baseline_tug_mae_s"], 2) + " s"],
    ], [5064, 3242]))
    b.add(para(
        f"在 {cohort['n']} 名虚拟受试者上，传统二维方法的平衡评分与真实评分相关系数为 "
        f"{_fmt(base['pearson_r'])}，三分类准确率 {_fmt(base['accuracy'])}，"
        f"高风险识别 AUC 为 {_fmt(base['auc_high'])}，"
        f"但与压力垫金标准的一致性仅为 {_fmt(base['icc_cop'])}。"
        "该结果符合预期：传统方法依靠单脚站立保持时间与 TUG 用时，能够捕捉到相当一部分"
        "风险信息，但由于二维投影几乎观测不到前后方向晃动，其晃动面积与真实三维重心晃动"
        "在量纲与含义上均存在系统偏差，因而与金标准一致性很差。"))
    b.add(para(
        "需要特别说明的是，当相机安装高度接近人体重心高度时，图像纵轴对前后方向位移的"
        "敏感度极低——本实验参数下 20 mm 的前后方向位移仅引起约 0.5 像素变化，与关键点"
        "检测噪声处于同一量级，因此二维方法实际上只能观测到左右方向晃动。这正是引入"
        "单目深度估计与三维重心重建的直接动因。"))
    b.add_image(OUT_DIR / "fig2_baseline.png")
    b.add(caption("图 2 传统二维方法评分与真实评分对比及风险分级混淆矩阵"))

    b.add(heading("    第二组实验：本文方法复现（RollingDepth + MoSca + GaitLLM）", 4))
    b.add(para(
        "保持仿真参数完全不变（同一虚拟队列、同一传感噪声种子），切换为第二组代码，"
        "启用单目深度尺度恢复、三维重心估计、四维轨迹重建与步态语言建模。"))
    b.add(para("运行 main.py，程序执行流程："))
    steps_prop = [
        "在独立参考队列上标定 GaitLLM 正常步态语料与两种方法的评分量程；",
        "RollingDepth：用骨骼长度先验求解单目深度仿射尺度，经骨骼约束优化后反投影得到三维关节，"
        "再按分段质量分布加权得到三维重心轨迹；",
        "MoSca：对三维关节做四维时序一致性优化，自动切分 TUG 六个阶段并提取动态平衡与步态指标；",
        "GaitLLM：把步态指标量化成“步态句子”，由语言模型给出困惑度与步态异常分；",
        "融合 13 维特征，输出平衡能力评分与跌倒风险等级，并生成通俗化文本报告；",
        "控制台打印第二组全部指标，保存三维重心重建、四维轨迹、步态句子与对比图表。",
    ]
    for text in steps_prop:
        b.add(para(text, style=S_BODY, numid=4))
    b.add(para("> 控制台输出文本："))
    start = next(i for i, line in enumerate(console_lines) if "第二组" in line)
    end = next(i for i, line in enumerate(console_lines) if "示例报告" in line)
    b.add(code_block(console_lines[start:end - 1]))

    b.add_image(OUT_DIR / "fig3_com.png")
    b.add(caption("图 3 RollingDepth 三维重心轨迹重建与 95% 置信椭圆"))
    b.add_image(OUT_DIR / "fig4_tug4d.png")
    b.add(caption("图 4 MoSca 四维重建的 TUG 轨迹与阶段切分结果"))
    b.add_image(OUT_DIR / "fig5_gaitllm.png")
    b.add(caption("图 5 GaitLLM 步态句子与各环节偏离正常步态的程度"))

    b.add(caption("表 2 两组方法评价指标对比", keep_next=True))
    b.add(table([
        ["评价指标", "传统二维阈值法", "本文方法"],
        ["平衡评分相关性 r", _fmt(base["pearson_r"]), _fmt(prop["pearson_r"])],
        ["Spearman ρ", _fmt(base["spearman_r"]), _fmt(prop["spearman_r"])],
        ["评分 MAE（分）", _fmt(base["mae"], 2), _fmt(prop["mae"], 2)],
        ["三分类准确率", _fmt(base["accuracy"]), _fmt(prop["accuracy"])],
        ["macro-F1", _fmt(base["macro_f1"]), _fmt(prop["macro_f1"])],
        ["Cohen κ", _fmt(base["kappa"]), _fmt(prop["kappa"])],
        ["高风险识别 AUC", _fmt(base["auc_high"]), _fmt(prop["auc_high"])],
        ["压力垫一致性 ICC(2,1)", _fmt(base["icc_cop"]), _fmt(prop["icc_cop"])],
        ["TUG 用时 MAE（s）", _fmt(metrics["baseline_tug_mae_s"], 2),
         _fmt(metrics["proposed_tug_mae_s"], 2)],
    ], [3004, 2651, 2651]))
    b.add(para(
        f"本文方法把评分相关性由 {_fmt(base['pearson_r'])} 提升至 {_fmt(prop['pearson_r'])}，"
        f"三分类准确率由 {_fmt(base['accuracy'])} 提升至 {_fmt(prop['accuracy'])}，"
        f"macro-F1 由 {_fmt(base['macro_f1'])} 提升至 {_fmt(prop['macro_f1'])}，"
        f"Cohen κ 由 {_fmt(base['kappa'])} 提升至 {_fmt(prop['kappa'])}；"
        f"最显著的改进来自与压力垫金标准的一致性：ICC(2,1) 由 {_fmt(base['icc_cop'])} "
        f"提升至 {_fmt(prop['icc_cop'])}，说明三维重心晃动面积与真实压力中心晃动面积"
        "在绝对量纲上高度一致，而二维投影面积无法做到这一点。"))
    b.add(para(
        f"方法学层面，RollingDepth 单目深度估计的平均绝对误差为 "
        f"{metrics['depth_mae_mm']:.1f} mm；MoSca 四维重建把骨骼长度误差由 "
        f"{metrics['mosca_bone_error_mm'][0]:.1f} mm 降至 "
        f"{metrics['mosca_bone_error_mm'][1]:.1f} mm，时间抖动由 "
        f"{metrics['mosca_jitter_mm'][0]:.2f} mm 降至 "
        f"{metrics['mosca_jitter_mm'][1]:.2f} mm，说明时序正则与骨骼约束有效抑制了"
        "单目深度噪声，为阶段切分与步态指标提取提供了稳定的四维轨迹。"))
    b.add_image(OUT_DIR / "fig6_comparison.png")
    b.add(caption("图 6 两组方法综合评价指标、评分一致性与高风险识别 ROC 曲线"))
    b.add(para(
        "结合两组实验综合分析：第一组验证了传统二维阈值法在无穿戴场景下的可行性，"
        "但其观测量受投影几何限制，无法反映前后方向晃动与步态质量；"
        "第二组在相同仿真环境下引入三维重心、四维轨迹与步态语言特征，"
        "在评分相关性、分级准确率与金标准一致性上均有提升，更适配老年人跌倒风险筛查需求。"))

    # ---------------- 2.2 对标论文与本项目的对应关系 ----------------
    b.add(heading("  2.2 对标论文方法与本项目实现的对应关系", 2))
    b.add(para(
        "本项目复现的三篇工作均为 CVPR 2025 论文，分别解决“单目视频深度估计”"
        "“单目视频四维重建”与“步态序列语言化建模”三个问题。下表给出三篇论文的核心机制"
        "及其在本项目中的对应实现。"))
    b.add(caption("表 3 三篇对标论文核心机制与本项目复现要点", keep_next=True))
    mapping = [
        ["对标论文", "原论文核心机制", "本项目复现要点"],
        ["#876 RollingDepth\nVideo Depth without\nVideo Models\nCVPR 2025: 7233-7243",
         "由单帧潜扩散模型（LDM）派生的多帧深度估计器，"
         "把极短视频片段（典型为帧三元组）映射为“深度片段”；"
         "再用基于优化的鲁棒配准算法，把以不同帧率采样得到的深度片段"
         "拼接成时序一致、可处理数百帧长视频的深度视频。",
         "以逐帧相对深度为输入，用骨骼长度先验求解仿射尺度，"
         "承担原论文“片段级配准”的角色；以骨骼约束深度优化与滚动窗口"
         "时序平滑保证逐帧一致，替代视频基础模型，"
         "并额外把深度抬升为带分段质量的三维重心。"],
        ["#882 MoSca\nDynamic Gaussian Fusion\nvia 4D Motion Scaffolds\nCVPR 2025: 6165-6177",
         "把单目视频提升为紧凑、平滑的 Motion Scaffold（运动支架）表示以编码"
         "运动与形变，将几何与外观从形变场中解耦；把高斯锚定在支架上做全局融合，"
         "相机焦距与位姿由光束法平差求解。",
         "以 20 关节人体骨架充当运动支架：时间二阶差分项对应支架的平滑运动约束，"
         "骨骼长度一致性项对应支架的结构约束，全局四维优化对应支架全局融合；"
         "本任务只需轨迹、无需新视角合成，故略去高斯泼溅渲染分支。"
         "重建结果为 TUG 阶段切分与步态指标提供稳定输入。"],
        ["#837 GaitLLM\nBridging Gait Recognition\nand LLMs Sequence Modeling\nCVPR 2025: 3460-3469",
         "提出 Gait-to-Language（G2L）模块把步态序列转换为适合大语言模型的文本形式，"
         "再用 Language-to-Gait（L2G）模块把模型输出映射回步态特征空间，"
         "在 SUSTech1K、CCPG、Gait3D、GREW 等数据集上取得最优步态识别性能。",
         "复现 G2L 的“步态→语言”思路：把连续步态参数量化为 9 个槽位的 token 序列"
         "（步态句子）。因本任务关注临床异常而非身份，L2G 分支替换为异常判别与"
         "通俗化报告生成，并以可离线运行的插值式二元语言模型替代大规模 LLM，"
         "仍保留“以语言模型度量步态偏离程度”的方法学内核。"],
    ]
    b.add(table(mapping, [1850, 3278, 3178]))
    b.add(para(
        "三条技术路线在本项目中前后衔接，恰好对应原文由“视频”逐级升维到“语言”的递进关系："
        "RollingDepth 把单目视频抬升到三维（逐帧三维关节与三维重心）；"
        "MoSca 在此基础上把三维抬升到四维（时间一致的全过程轨迹与阶段切分）；"
        "GaitLLM 再把四维轨迹导出的步态指标抬升到语言（步态句子与异常判定）。"))
    b.add(para(
        "需要说明的是，三篇论文的原始任务分别是单目视频深度估计、动态场景新视角合成与"
        "步态身份识别，与老年人跌倒风险评估并不相同。本项目不直接调用三篇论文的预训练权重，"
        "而是在统一的虚拟老年队列与单目观测仿真环境下提取其方法学内核进行复现，"
        "并与传统二维阈值法做对照。这样既保留了论文的关键设计"
        "（片段级深度配准、运动支架全局优化、步态语言化），又使实验可控、可重复，"
        "便于定量分析各环节对最终跌倒风险评分的影响。"))

    # ---------------- 3 改进思路与未来研究方向 ----------------
    b.add(heading(" 3 改进思路与未来研究方向", 1))
    b.add(heading("    3.1 现有方法改进", 2))
    b.add(para(
        "完善深度估计的时序建模：RollingDepth 原文用基于优化的配准算法显式求解片段间的"
        "一致性变换，本项目则以全局仿射参数加骨骼约束投影近似该过程，对长时序缓慢漂移的"
        "建模仍较简单；后续可引入逐帧卡尔曼滤波或可微分光束法平差，"
        "进一步降低深度方向的低频漂移。"))
    b.add(para(
        "增强步态语言模型的表达力：GaitLLM 原文用大语言模型完成步态序列建模，"
        "本项目以插值式二元语言模型替代，只能刻画 token 间的一阶依赖；"
        "后续可引入小型预训练语言模型或条件随机场，并结合医学本体构建更细粒度的"
        "步态描述体系。"))
    b.add(para(
        "扩展对比基线：目前对比基线仅有传统二维阈值法，后续可加入压力垫 COP 指标、"
        "惯性传感器方案与临床量表（Berg 平衡量表、Tinetti 量表）作为多角度对照。"))

    b.add(heading("    3.2 功能扩展方向", 2))
    b.add(para(
        "增加连续随访与纵向监测：当前为单次评估，后续可支持同一受试者多次测试的纵向对比，"
        "输出平衡能力变化趋势与恶化预警，为干预效果评估提供依据。"))
    b.add(para(
        "多模态融合：在单目摄像头基础上接入压力垫、可穿戴惯性单元或毫米波雷达，"
        "研究不同模态在静态平衡、动态平衡与步态三个维度上的互补性，并给出低成本部署方案。"))
    b.add(para(
        "实时在线推理：当前为离线批量处理，后续可将管线重构为流式推理，"
        "在普通手机或边缘设备上实现实时评估与即时告警。"))

    b.add(heading("    3.3 理论研究深化", 2))
    b.add(para(
        "引入真实数据集：本项目全部数据由仿真生成，后续可在公开步态数据集"
        "（如 TUM GAID、KIMORE）与医院采集的真实数据上验证，减少仿真理想化带来的偏差。"
        "更进一步，可把三篇论文的公开实现（RollingDepth 深度估计器、MoSca 四维重建、"
        "GaitLLM 的 G2L/L2G 模块）接入本项目的评估框架，替换本项目的轻量复现实现，"
        "直接对比“原论文模型 + 临床下游任务”与“方法学内核复现”的差异。"))
    b.add(para(
        "量化不确定性与可解释性：研究深度估计误差如何传播到重心与评分，"
        "给出评分置信区间；同时利用步态语言模型的逐槽位意外度实现结果可解释。"))
    b.add(para(
        "大规模场景性能测试：分析受试者数量、视频时长增长时的时间复杂度，"
        "针对实时筛查场景研究模型轻量化与并行加速策略。"))

    b.add(heading("    3.4 工程应用拓展", 2))
    b.add(para(
        "开发交互式评估终端：当前依靠命令行运行，后续实现图形界面，"
        "支持摄像头实时采集、一键评估与报告导出，便于在养老机构与社区卫生服务中心部署。"))
    b.add(para(
        "模块封装与接口标准化：把队列仿真、深度估计、四维重建、步态语言建模封装为独立函数库，"
        "通过标准接口对接养老机构健康管理系统与保险公司健康管理平台。"))

    # ---------------- 参考文献 ----------------
    b.add(heading("参考文献", 1))
    references = [
        "Ke B, Narnhofer D, Huang S, Ke L, Peters T, Fragkiadaki K, Obukhov A, "
        "Schindler K. Video Depth without Video Models[C]. Proceedings of the "
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), "
        "2025: 7233-7243.",
        "Lei J, Weng Y, Harley A W, Guibas L, Daniilidis K. MoSca: Dynamic Gaussian "
        "Fusion from Casual Videos via 4D Motion Scaffolds[C]. Proceedings of the "
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), "
        "2025: 6165-6177.",
        "Yang S, Wang J, Hou S, Liu X, Cao C, Wang L, Huang Y. Bridging Gait "
        "Recognition and Large Language Models Sequence Modeling[C]. Proceedings "
        "of the IEEE/CVF Conference on Computer Vision and Pattern Recognition "
        "(CVPR), 2025: 3460-3469.",
        "Winter D A. Biomechanics and Motor Control of Human Movement[M]. 4th ed. "
        "Hoboken: John Wiley & Sons, 2009.",
        "Dempster W T. Space requirements of the seated operator[R]. "
        "Wright-Patterson Air Force Base: Wright Air Development Center, 1955.",
        "Prieto T E, Myklebust J B, Hoffmann R G, et al. Measures of postural "
        "steadiness: differences between healthy young and elderly adults[J]. "
        "IEEE Transactions on Biomedical Engineering, 1996, 43(9): 956-966.",
        "Podsiadlo D, Richardson S. The timed “Up & Go”: a test of basic functional "
        "mobility for frail elderly persons[J]. Journal of the American Geriatrics "
        "Society, 1991, 39(2): 142-148.",
        "Tinetti M E. Performance-oriented assessment of mobility problems in "
        "elderly patients[J]. Journal of the American Geriatrics Society, 1986, "
        "34(2): 119-126.",
        "Berg K, Wood-Dauphinee S, Williams J I, et al. Measuring balance in the "
        "elderly: validation of an instrument[J]. Canadian Journal of Public Health, "
        "1992, 83(S2): S7-S11.",
        "Mancini M, Horak F B. The relevance of clinical balance assessment tools "
        "to differentiate balance deficits[J]. European Journal of Physical and "
        "Rehabilitation Medicine, 2010, 46(2): 239-248.",
        "Beauchet O, Fantino B, Allali G, et al. Timed Up and Go test and risk of "
        "falls in older adults: a systematic review[J]. The Journal of Nutrition, "
        "Health & Aging, 2011, 15(10): 933-938.",
        "Sagawa Y, Turcot K, Armand S, et al. Biomechanics and physiological "
        "parameters used for gait assessment: a review[J]. Journal of "
        "NeuroEngineering and Rehabilitation, 2011, 8: 12.",
        "Kwolek B, Michalczuk A, Krzeszowski T, et al. Calibrated and synchronized "
        "multi-camera gait acquisition system with commodity RGB-D sensors[J]. "
        "Measurement Science and Technology, 2019, 30(9): 095105.",
    ]
    for text in references:
        b.add(para(text, style=S_REFERENCE, numid=6))

    # ---------------- 附录 A ----------------
    b.add(heading(" 附录A 完整源代码", 1))
    b.add(para(
        "附录给出核心算法与主流程源码截图；仿真层（src/simulation/）、"
        "评价指标（src/evaluation/）与可视化模块（src/visualization/）"
        "源码随工程目录一并提交，篇幅所限不在此重复列出。"))
    b.add_code_images(manifest, OUT_DIR / "code_shots")

    # ---------------- 打包 ----------------
    (work / "word" / "document.xml").write_text(b.document_xml(), encoding="utf-8")
    (work / "word" / "_rels" / "document.xml.rels").write_text(b.rels_xml(), encoding="utf-8")
    patch_content_types(work)

    if REPORT_PATH.exists():
        REPORT_PATH.unlink()
    with zipfile.ZipFile(REPORT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(work.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(work).as_posix())

    if update_fields and _try_update_fields():
        pass
    return REPORT_PATH


def _try_update_fields() -> bool:
    from report_lib import update_fields_with_word
    ok = update_fields_with_word(REPORT_PATH)
    print("目录与页码域已由 Word 更新。" if ok else
          "（提示）未检测到可用的 Word，目录将在打开文档时提示更新。")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按模板格式生成代码复现报告")
    parser.add_argument("--repo-url", default=REPO_URL,
                        help="显示在标题下方的项目仓库地址（蓝色下划线超链接）")
    parser.add_argument("--no-word", action="store_true",
                        help="跳过调用 Word 更新目录域（无 Office 环境时使用）")
    args = parser.parse_args()
    out = build(repo_url=args.repo_url, update_fields=not args.no_word)
    print(f"报告已生成：{out}")
