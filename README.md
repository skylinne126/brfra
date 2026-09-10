# BAFRA — 平衡能力与跌倒风险评估（方案 8-A 代码复现）

**Balance Ability and Fall Risk Assessment (BAFRA)**

面向医院老年科 / 养老院 / 社区健康中心的**无穿戴式**老年人平衡能力与跌倒风险评估复现项目。
仅需一台单目摄像头（压力垫为可选的非穿戴参考传感器），完成

- **静态平衡**：闭眼单脚站立、睁眼双脚站立的三维重心晃动量化；
- **动态平衡**：起立—行走—转身—坐下（TUG）全过程四维轨迹重建与阶段自动切分；
- **步态分析**：步态序列 → “步态句子” → 步态语言模型异常判别；
- **综合输出**：跌倒风险等级 + 平衡能力评分 + 通俗化中文报告。

复现三篇 **CVPR 2025** 论文（见下表），并与传统二维关键点阈值法做对照实验。

---

## 1. 复现的三篇工作

| 编号 | 论文 | 出处 | 在本项目中的作用 |
|------|------|------|------------------|
| #876 | **Video Depth without Video Models**（RollingDepth） | CVPR 2025: 7233-7243 | 单帧潜扩散模型派生的多帧深度估计 + 基于优化的深度片段配准；本项目以骨骼长度先验消解单目深度仿射尺度歧义，经骨骼约束优化估计三维重心 |
| #882 | **MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds** | CVPR 2025: 6165-6177 | 紧凑平滑的运动支架（Motion Scaffold）+ 全局融合优化；本项目以 20 关节骨架充当运动支架，用时序正则与骨骼约束做四维重建，恢复 TUG 全过程轨迹 |
| #837 | **Bridging Gait Recognition and Large Language Models Sequence Modeling**（GaitLLM） | CVPR 2025: 3460-3469 | Gait-to-Language（G2L）把步态序列转为文本、Language-to-Gait（L2G）映射回步态特征空间；本项目复现 G2L 思路，把步态离散化为“步态句子”并用语言模型判定异常 |

> 三篇论文的原始任务分别是单目视频深度估计、动态场景新视角合成与步态身份识别，
> 与跌倒风险评估并不相同。本项目**不直接调用其预训练权重**，而是提取方法学内核，
> 在统一的虚拟老年队列与单目观测仿真环境下复现，并与传统二维阈值法对照。

---

## 2. 快速开始

```bash
git clone https://github.com/skylinne126/brfra.git
cd brfra
pip install -r requirements.txt

python run_all.py            # 快速跑通（30 名受试者，约 1 分钟）
python run_all.py --full     # 论文完整队列（120 名受试者，约 3 分钟）
```

也可以分步执行：

```bash
python main.py --n 30               # ① 两组对照实验 → output/ 指标、CSV 与 6 张图
python tools/render_code_shots.py   # ② 生成附录 A 的源码截图
python tools/build_report.py        # ③ 生成 Word 代码复现报告
```

生成 Word 报告需要额外依赖（不生成报告时无需安装）：

```bash
pip install -r requirements-report.txt
```

### 运行环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.9 及以上（开发环境为 3.13） |
| 操作系统 | Windows / macOS / Linux 均可运行仿真与图表 |
| 必要依赖 | numpy、scipy、matplotlib、pillow |
| 报告公式 | `latex2mathml` + Office 自带的 `MML2OMML.XSL`；缺失时公式自动降级为纯文本，**不会报错** |
| 目录页码 | Windows 且安装 Word 时自动更新；否则打开文档后按 `Ctrl+A`、`F9` 手动更新 |

> 中文图表需要系统中文字体（Windows 自带宋体/黑体；Linux 可 `apt install fonts-noto-cjk`）。
> 找不到中文字体时程序仍可正常运行，只是图中汉字会显示为方框。

---

## 3. 实验结果

120 名虚拟老年受试者（平均年龄 79.9 ± 8.2 岁，低/中/高风险 19/79/22），
两组方法使用完全相同的队列与单目观测数据：

| 评价指标 | 传统二维阈值法 | 本文方法（RollingDepth + MoSca + GaitLLM） |
|---------|---------------|------------------------------------------|
| 平衡评分相关性 r | 0.894 | **0.896** |
| Spearman ρ | 0.892 | **0.910** |
| 评分 MAE（分） | 4.08 | **3.79** |
| 三分类准确率 | 0.792 | **0.858** |
| macro-F1 | 0.764 | **0.831** |
| Cohen κ | 0.623 | **0.728** |
| 高风险识别 AUC | 0.965 | **0.982** |
| 与压力垫金标准一致性 ICC(2,1) | 0.129 | **0.977** |

方法学指标：单目深度估计平均误差 91.1 mm；MoSca 四维重建把骨骼长度误差由
19.3 mm 降至 0.5 mm、时间抖动由 14.21 mm 降至 7.52 mm。

**关键结论**：当相机安装高度接近人体重心高度时，图像纵轴对前后方向（AP）位移的
敏感度极低（20 mm 的 AP 位移仅引起约 0.5 像素变化，与关键点噪声同量级），
因此二维方法实际上只能观测左右方向晃动，其估计面积与压力垫金标准几乎不相关
（ICC 0.13）；而三维重心晃动面积与金标准高度一致（ICC 0.98）。

---

## 4. 目录结构

```
brfra/
├─ main.py                          主入口：两组对照实验、指标输出与绘图
├─ config.py                        全局参数（队列、摄像头、噪声、算法超参）
├─ run_all.py                       一键跑通脚本
├─ requirements.txt                 核心依赖
├─ requirements-report.txt          生成 Word 报告的额外依赖
├─ assets/report_template/          报告排版模板资源（随工程分发）
├─ src/
│  ├─ scoring.py                    参考队列量程标定与统一评分机制
│  ├─ simulation/                   虚拟队列、运动学、测试动作真值、传感噪声
│  ├─ baseline/                     第一组：二维投影晃动估计
│  ├─ rollingdepth/                 #876 单目深度、三维重心、晃动分析
│  ├─ mosca/                        #882 四维重建、TUG 阶段切分、动态指标
│  ├─ gaitllm/                      #837 步态 token 化、语言模型、报告生成
│  ├─ fusion/                       多源特征融合与风险分级
│  ├─ evaluation/                   相关性、准确性、AUC、ICC 等评价指标
│  └─ visualization/                中文绘图样式与全部图表
├─ tools/
│  ├─ report_lib.py                 报告排版库（字体方案、OMML 公式、页脚、目录）
│  ├─ render_code_shots.py          生成 PyCharm 风格源码截图
│  └─ build_report.py               生成 Word 代码复现报告
└─ output/                          运行后自动生成（指标、CSV、图表、源码截图）
```

---

## 5. 方法概述

```
单目视频关键点
   │
   ├─[第一组] 二维关键点 + 身高标定 ──► 平面晃动指标 ──► 阈值评分 ──► 风险等级
   │
   └─[第二组]
        RollingDepth  相对深度 + 骨骼先验 ──► 仿射尺度恢复 ──► 三维关节
                      ──► Dempster 分段质量加权 ──► 三维重心轨迹
           │
        MoSca        时序二阶差分正则 + 骨骼长度一致性 ──► 四维轨迹
                      ──► TUG 阶段切分（起立/行走/转身/坐下）──► 动态与步态指标
           │
        GaitLLM      步态参数 ──► 9 槽位 token ──► “步态句子”
                      ──► 插值式二元语言模型 ──► 困惑度 / 步态异常分
           │
        融合          13 维特征 ──► 参考队列等效百分位等值 ──► 平衡评分 + 风险等级 + 文本报告
```

两组方法在**独立参考队列**上完成量程标定（`src/scoring.py`），保证对照公平。

---

## 6. 常见问题

**Q：跑完没有生成 Word 报告？**
A：先执行 `pip install -r requirements-report.txt`。若缺少 Office，公式会自动降级为纯文本，
报告仍能正常生成；目录页码需在 Word 中按 `Ctrl+A`、`F9` 更新。

**Q：图表里的中文变成方框？**
A：系统缺少中文字体。Windows 一般自带；Ubuntu 可 `sudo apt install fonts-noto-cjk`。

**Q：想改报告标题下方的仓库地址？**
A：`python tools/build_report.py --repo-url https://github.com/你的用户名/你的仓库`，
或直接修改 `tools/build_report.py` 中的 `REPO_URL` 常量。

**Q：运行时间？**
A：30 人约 1 分钟，120 人约 3 分钟（含参考队列标定）；瓶颈在单目深度尺度恢复的非线性最小二乘。

---

## 7. 参考文献

1. Ke B, Narnhofer D, Huang S, et al. Video Depth without Video Models[C]. CVPR, 2025: 7233-7243.
2. Lei J, Weng Y, Harley A W, et al. MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds[C]. CVPR, 2025: 6165-6177.
3. Yang S, Wang J, Hou S, et al. Bridging Gait Recognition and Large Language Models Sequence Modeling[C]. CVPR, 2025: 3460-3469.
4. Winter D A. Biomechanics and Motor Control of Human Movement[M]. 4th ed. Wiley, 2009.
5. Dempster W T. Space requirements of the seated operator[R]. Wright Air Development Center, 1955.
6. Prieto T E, Myklebust J B, Hoffmann R G, et al. Measures of postural steadiness[J]. IEEE TBME, 1996, 43(9): 956-966.
7. Podsiadlo D, Richardson S. The timed “Up & Go”[J]. JAGS, 1991, 39(2): 142-148.
8. Beauchet O, Fantino B, Allali G, et al. Timed Up and Go test and risk of falls in older adults[J]. JNHA, 2011, 15(10): 933-938.
