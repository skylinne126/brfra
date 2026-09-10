"""
方案 8-A 平衡能力与跌倒风险评估——全局配置。

本文件集中管理仿真实验的全部参数，保证第一组（传统 2D 阈值法）与
第二组（RollingDepth 3D 重心 + MoSca 4D 重建 + GaitLLM 步态语言建模）
两组实验在完全一致的仿真环境下运行，满足对照实验要求。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# --------------------------------------------------------------------------
# 人体骨骼模型：20 个关键点（COCO 扩展 + 足尖，用于 Dempster 分段质量加权）
# --------------------------------------------------------------------------
JOINT_NAMES: Tuple[str, ...] = (
    "head", "neck", "thorax", "pelvis",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hand", "r_hand",
    "l_hip", "r_hip", "l_knee", "r_knee",
    "l_ankle", "r_ankle", "l_toe", "r_toe",
)
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# --------------------------------------------------------------------------
# Dempster / Winter 人体分段质量分布（相对体重比例）与分段质心位置系数
# 分段质心位置系数指：质心到近端关节的距离 / 分段长度
# --------------------------------------------------------------------------
SEGMENT_DEFS = (
    # name, proximal, distal, mass_ratio, com_frac
    ("head", "head", "neck", 0.081, 0.500),
    ("trunk", "neck", "pelvis", 0.497, 0.500),
    ("upper_arm_l", "l_shoulder", "l_elbow", 0.028, 0.436),
    ("upper_arm_r", "r_shoulder", "r_elbow", 0.028, 0.436),
    ("forearm_l", "l_elbow", "l_wrist", 0.016, 0.430),
    ("forearm_r", "r_elbow", "r_wrist", 0.016, 0.430),
    ("hand_l", "l_wrist", "l_hand", 0.006, 0.506),
    ("hand_r", "r_wrist", "r_hand", 0.006, 0.506),
    ("thigh_l", "l_hip", "l_knee", 0.100, 0.433),
    ("thigh_r", "r_hip", "r_knee", 0.100, 0.433),
    ("shank_l", "l_knee", "l_ankle", 0.0465, 0.433),
    ("shank_r", "r_knee", "r_ankle", 0.0465, 0.433),
    ("foot_l", "l_ankle", "l_toe", 0.0145, 0.500),
    ("foot_r", "r_ankle", "r_toe", 0.0145, 0.500),
)

# Winter 人体测量学分段长度比例（相对身高 H）
SEGMENT_LENGTH_RATIO = {
    "head": 0.130,      # 头 + 颈
    "trunk": 0.288,     # 颈（肩峰）到髋
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.108,
    "thigh": 0.245,
    "shank": 0.246,
    "foot": 0.152,
}


@dataclass
class SimConfig:
    """仿真实验全局参数。"""

    # ---------------- 队列 ----------------
    n_subjects: int = 120
    seed: int = 20240908
    healthy_bias: float = 0.0                # 队列整体偏健康程度（参考队列用）
    age_range: Tuple[int, int] = (65, 92)
    height_range: Tuple[float, float] = (148.0, 180.0)   # cm
    mass_range: Tuple[float, float] = (42.0, 88.0)       # kg

    # ---------------- 采样与测试协议 ----------------
    fps: int = 30
    quiet_stance_duration: float = 30.0        # 睁眼双脚站立（s）
    single_leg_duration: float = 20.0          # 闭眼单脚站立（s）
    tug_distance: float = 3.0                  # TUG 直线行走距离（m）

    # ---------------- 摄像头与深度估计 ----------------
    camera_height: float = 1.45                # 摄像头安装高度（m）
    camera_distance: float = 3.60              # 受试者距摄像头距离（m）
    focal_px: float = 1120.0                   # 焦距（像素）
    image_width: int = 1280
    image_height: int = 720
    keypoint_noise_px: float = 2.2             # 2D 关键点检测抖动（像素）
    occlusion_prob: float = 0.06               # 单帧单关节遮挡概率
    depth_noise_ratio: float = 0.035           # 单目深度相对误差
    rolling_window: int = 15                   # RollingDepth 滚动窗口长度（帧）

    # ---------------- 信号处理 ----------------
    lowpass_cutoff_hz: float = 2.5             # 重心轨迹低通滤波截止频率
    savgol_window: int = 15                    # 三维关节时序平滑窗口（帧）
    detrend: bool = True                       # 是否去除均值（晃动分析标准流程）

    # ---------------- 骨骼约束深度优化 ----------------
    refine_iters: int = 300                    # 梯度下降迭代次数
    refine_prior_weight: float = 0.15          # 深度先验项权重
    refine_step: float = 0.02                  # Adam 学习率

    # ---------------- MoSca 四维重建 ----------------
    mosca_iters: int = 250                     # 四维优化迭代次数
    mosca_step: float = 0.02                   # Adam 学习率
    mosca_data_weight: float = 1.0             # 数据项权重
    mosca_time_weight: float = 1.5             # 时间二阶差分平滑权重
    mosca_bone_weight: float = 2.0             # 骨骼长度一致性权重

    # ---------------- TUG 阶段切分 ----------------
    walk_speed_threshold: float = 0.15         # 行走速度阈值（m/s）
    turn_rate_threshold: float = 30.0          # 转身角速度阈值（度/秒）
    foot_contact_speed: float = 0.12           # 足部着地判定进入阈值（m/s）
    foot_release_speed: float = 0.25           # 足部离地判定退出阈值（m/s）

    # ---------------- GaitLLM 步态语言模型 ----------------
    reference_subjects: int = 60               # 代表性参考队列人数（评分量程标定）
    lm_reference_subjects: int = 55            # 偏健康参考队列人数（步态语料标定）
    reference_healthy_bias: float = 0.12       # 参考队列偏健康（社区健康老年人）
    lm_calibration_subjects: int = 40          # 正常步态语料标定人数上限
    lm_healthy_score: float = 78.0             # 正常语料入选评分下限
    lm_smoothing: float = 1.0                  # 拉普拉斯平滑系数

    # ---------------- 风险分级 ----------------
    risk_thresholds: Tuple[float, float] = (55.0, 75.0)   # 低/中/高 分界
    risk_labels: Tuple[str, str, str] = ("低风险", "中风险", "高风险")

    # ---------------- 输出 ----------------
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    figure_dpi: int = 180

    def subject_seed(self, index: int) -> int:
        """保证同一受试者在两组实验中使用完全相同的随机种子。"""
        return self.seed + 1000 * (index + 1)


# 两组实验共用的固定实例
CFG = SimConfig()

# 中文绘图字体（Windows 内置）
CJK_FONT = "SimSun"
