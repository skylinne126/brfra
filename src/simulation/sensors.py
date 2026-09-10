"""
单目摄像头观测模型与传感噪声仿真。

模拟真实部署条件下的三类退化因素：
1. 2D 关键点检测抖动与随机遮挡；
2. 透视投影造成的深度方向信息丢失；
3. 单目深度估计网络固有的仿射尺度歧义（z = a·d_rel + b）与逐帧漂移。

压力垫为可选的"金标准"参考传感器，仅在评价阶段用于一致性校验。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from config import CFG, SimConfig


@dataclass
class MonocularCamera:
    """针孔相机模型（相机位于世界坐标 y 轴负方向，光轴指向 +y）。"""

    height: float
    distance: float
    focal_px: float
    width: int
    height_px: int

    @classmethod
    def from_config(cls, cfg: SimConfig = CFG) -> "MonocularCamera":
        return cls(height=cfg.camera_height, distance=cfg.camera_distance,
                   focal_px=cfg.focal_px, width=cfg.image_width,
                   height_px=cfg.image_height)

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height_px / 2.0

    def to_camera_frame(self, points_world: np.ndarray) -> np.ndarray:
        """世界坐标 → 相机坐标（z 为深度方向）。"""
        pts = np.asarray(points_world, dtype=float)
        return np.stack([
            pts[..., 0],
            pts[..., 2] - self.height,
            pts[..., 1] + self.distance,
        ], axis=-1)

    def from_camera_frame(self, points_cam: np.ndarray) -> np.ndarray:
        """相机坐标 → 世界坐标。"""
        pts = np.asarray(points_cam, dtype=float)
        return np.stack([
            pts[..., 0],
            pts[..., 2] - self.distance,
            pts[..., 1] + self.height,
        ], axis=-1)

    def project(self, points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """透视投影，返回像素坐标 (…,2) 与深度 (…)。"""
        cam = self.to_camera_frame(points_world)
        depth = np.clip(cam[..., 2], 1e-3, None)
        u = self.cx + self.focal_px * cam[..., 0] / depth
        v = self.cy - self.focal_px * cam[..., 1] / depth
        return np.stack([u, v], axis=-1), depth


@dataclass
class SensorStream:
    """一次测试采集到的全部单目观测。"""

    keypoints2d: np.ndarray        # (T, J, 2) 像素坐标，遮挡处为 NaN
    visible: np.ndarray            # (T, J) 可见性
    relative_depth: np.ndarray     # (T, J) 单目深度网络输出的相对深度
    true_depth: np.ndarray         # (T, J) 真值深度（仅用于误差评价）
    camera: MonocularCamera
    true_joints3d: np.ndarray      # (T, J, 3) 真值三维关节（仅用于误差评价）
    meta: Dict[str, float]

    @property
    def n_frames(self) -> int:
        return self.keypoints2d.shape[0]


def simulate_sensor_stream(true_joints3d: np.ndarray,
                           cfg: SimConfig = CFG,
                           camera: Optional[MonocularCamera] = None,
                           seed: int = 0,
                           occlusion_boost: float = 1.0) -> SensorStream:
    """
    由三维真值关节序列生成单目观测流。

    遮挡概率随人体远离摄像头、以及侧向（转身后）姿态而升高，
    对应真实场景中转身、弯腰时关节自遮挡加重的现象。
    """
    rng = np.random.default_rng(seed)
    cam = camera if camera is not None else MonocularCamera.from_config(cfg)
    n, j, _ = true_joints3d.shape

    pixels, depth = cam.project(true_joints3d)

    # ---- 2D 检测抖动：噪声随深度增大 ----
    depth_scale = depth / max(cfg.camera_distance, 1e-3)
    noise_sigma = cfg.keypoint_noise_px * np.clip(depth_scale, 0.6, 2.2)
    pixels = pixels + rng.normal(0.0, 1.0, size=pixels.shape) * noise_sigma[..., None]

    # ---- 遮挡：转身/侧向姿态下概率升高 ----
    lateral = np.abs(true_joints3d[..., 0] - true_joints3d[:, :1, 0])
    occ_prob = cfg.occlusion_prob * occlusion_boost * (1.0 + 1.4 * np.clip(lateral, 0, 0.4))
    visible = rng.random((n, j)) > occ_prob
    # 保证每一帧至少 12 个关键点可见，避免完全失效
    for i in range(n):
        if visible[i].sum() < 12:
            idx = rng.choice(j, size=12, replace=False)
            visible[i, idx] = True

    keypoints2d = pixels.copy()
    keypoints2d[~visible] = np.nan

    # ---- 单目深度估计：仿射尺度歧义 + 关节固定偏置 + 低频全局漂移 + 逐帧噪声 ----
    # 真实单目深度网络的误差具有明显的空间平滑性与时间相关性：
    # 关节固定偏置在去均值后自动抵消；全局漂移由 RollingDepth 的滚动尺度估计吸收；
    # 仅剩的逐帧小噪声由人体骨骼长度约束与时间平滑进一步抑制。
    a_true = float(rng.uniform(0.85, 1.35))
    b_true = float(rng.uniform(-0.45, 0.45))
    joint_bias = rng.normal(0.0, 0.006, size=(1, j))                     # 固定偏置 6 mm
    walk = np.cumsum(rng.normal(0.0, 1.0, size=(n, 1)), axis=0)
    walk = walk / (np.abs(walk).max() + 1e-9)
    global_drift = 0.020 * walk                                          # 全局漂移 ≤20 mm
    frame_noise = rng.normal(0.0, 0.011, size=depth.shape)               # 逐帧噪声 11 mm

    depth_obs = depth + joint_bias + global_drift + frame_noise
    rel = (depth_obs - b_true) / a_true
    rel[~visible] = np.nan

    meta = {
        "affine_a": a_true,
        "affine_b": b_true,
        "occlusion_rate": float(1.0 - visible.mean()),
        "mean_depth": float(np.nanmean(depth)),
    }
    return SensorStream(
        keypoints2d=keypoints2d,
        visible=visible,
        relative_depth=rel,
        true_depth=depth,
        camera=cam,
        true_joints3d=true_joints3d,
        meta=meta,
    )
