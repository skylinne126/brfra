"""
RollingDepth 复现：单目视频的三维人体深度估计。

论文出处
--------
#876  Ke B, Narnhofer D, Huang S, et al. Video Depth without Video Models[C].
      Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
      Recognition (CVPR), 2025: 7233-7243.

原论文的两个关键设计及其在本模块中的对应
----------------------------------------
(i)  由单帧潜扩散模型（LDM）派生的多帧深度估计器，把极短视频片段（典型为帧
     三元组）映射为“深度片段”——对应本模块逐帧相对深度的输入形式；
(ii) 基于优化的鲁棒配准算法，把以不同帧率采样得到的深度片段拼接为时序一致的
     深度视频——对应本模块的仿射尺度恢复与滚动窗口时序平滑。

复现要点
--------
1. 单目深度网络只能给出"相对深度"，存在仿射尺度歧义 z = a·d_rel + b；
2. 利用人体测量学先验（分段长度 / 身高比例）消解尺度歧义；
3. 以滚动时间窗口保证逐帧深度一致性，抑制单目深度的逐帧抖动；
4. 用骨骼长度约束对反投影结果做三维一致性优化。

尺度求解的数学形式
------------------
设第 i 个关节像素坐标为 (u_i, v_i)、相对深度为 d_i，其相机坐标为
    P_i = k_i · z_i ,  k_i = ((u_i - cx)/f, -(v_i - cy)/f, 1) ,  z_i = a·d_i + b
则任一分段长度满足
    L_ij = ‖a·A_ij + b·B_ij‖ ,  A_ij = k_i d_i - k_j d_j ,  B_ij = k_i - k_j
以先验长度 L̂_ij = ratio·身高 为监督，对 (a, b) 做鲁棒非线性最小二乘，
即可在无需测力板、无需已知拍摄距离的条件下恢复度量深度。

骨骼约束优化
------------
反投影得到的三维关节沿相机光轴存在残余抖动，进一步在保持二维投影不变
（仅调整深度 z_j）的前提下求解
    min_z  Σ_ij (‖P_i - P_j‖ - L̂_ij)² / L̂_ij²  +  λ Σ_j (z_j - z_j^prior)² / z_j^prior²
采用向量化梯度下降（Adam）求解，兼顾精度与运行效率。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from config import CFG, JOINT_INDEX, SimConfig
from src.rollingdepth.anthropometry import SCALE_PRIOR_PAIRS
from src.simulation.sensors import MonocularCamera, SensorStream

_PAIR_INDEX: List[Tuple[int, int, float]] = [
    (JOINT_INDEX[a], JOINT_INDEX[b], ratio) for a, b, ratio in SCALE_PRIOR_PAIRS
]


@dataclass
class DepthResult:
    """深度估计结果。"""

    joints3d: np.ndarray        # (T, J, 3) 反投影得到的三维关节
    depth: np.ndarray           # (T, J) 估计的度量深度
    scale_a: float              # 全局尺度系数 a
    scale_b: float              # 全局偏移系数 b
    valid: np.ndarray           # (T, J) 是否由观测直接支持
    depth_mae: float            # 与真值深度平均绝对误差（m，仅评价用）
    joint_mae: float            # 与真值三维关节平均误差（m，仅评价用）


class MonocularDepthEstimator:
    """基于人体尺寸先验与骨骼约束的单目度量深度估计器。"""

    def __init__(self, camera: MonocularCamera, cfg: SimConfig = CFG):
        self.camera = camera
        self.cfg = cfg

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """像素坐标 → 相机坐标系下的射线方向向量。"""
        f = self.camera.focal_px
        return np.stack([(u - self.camera.cx) / f, -(v - self.camera.cy) / f,
                         np.ones_like(u)], axis=-1)

    def _fill_missing(self, values: np.ndarray) -> np.ndarray:
        """对遮挡造成的缺失值沿时间轴线性插值填补。"""
        out = values.copy()
        n = out.shape[0]
        idx = np.arange(n)
        for j in range(out.shape[1]):
            col = out[:, j]
            good = np.isfinite(col)
            if good.sum() == 0:
                out[:, j] = 0.0
            elif good.sum() == 1:
                out[:, j] = col[good][0]
            else:
                out[:, j] = np.interp(idx, idx[good], col[good])
        return out

    # ------------------------------------------------------------------
    # 仿射尺度求解
    # ------------------------------------------------------------------
    def _affine_design(self, kp2d: np.ndarray, rel: np.ndarray,
                       height_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """构造长度残差的 A、B 矩阵与先验长度向量。"""
        vec_a, vec_b, priors = [], [], []
        for i, j, ratio in _PAIR_INDEX:
            ui, vi, di = kp2d[:, i, 0], kp2d[:, i, 1], rel[:, i]
            uj, vj, dj = kp2d[:, j, 0], kp2d[:, j, 1], rel[:, j]
            m = (np.isfinite(ui) & np.isfinite(vi) & np.isfinite(di)
                 & np.isfinite(uj) & np.isfinite(vj) & np.isfinite(dj))
            if not m.any():
                continue
            k_i = self._ray(ui[m], vi[m])
            k_j = self._ray(uj[m], vj[m])
            vec_a.append(k_i * di[m][:, None] - k_j * dj[m][:, None])
            vec_b.append(k_i - k_j)
            priors.append(np.full(int(m.sum()), ratio * height_m))
        if not vec_a:
            return np.zeros((0, 3)), np.zeros(0)
        return np.concatenate(vec_a), np.concatenate(priors)

    def _body_size_depth(self, kp2d: np.ndarray, height_m: float) -> float:
        """
        由"人体尺寸不变量"稳健估计人体所在深度。

        对每个骨骼段，已知其先验长度 L 与图像像素长度 l，则该段深度约为
        z = f·L / l。取所有段、所有帧的中位数，可抑制肢体朝向镜头造成的
        透视缩短影响，得到一个与姿态无关的绝对深度初值。
        """
        f = self.camera.focal_px
        estimates = []
        for i, j, ratio in _PAIR_INDEX:
            pixel_len = np.linalg.norm(kp2d[:, i, :] - kp2d[:, j, :], axis=1)
            ok = np.isfinite(pixel_len) & (pixel_len > 5.0)
            if ok.any():
                estimates.append(f * (ratio * height_m) / pixel_len[ok])
        if not estimates:
            return 3.6
        return float(np.median(np.concatenate(estimates)))

    def _fit_affine(self, kp2d: np.ndarray, rel: np.ndarray,
                    height_m: float) -> Tuple[float, float]:
        """
        鲁棒非线性最小二乘求解 (a, b)。

        除骨骼长度残差外，额外加入"人体尺寸深度"锚定项，避免受试者静止
        （二维观测几乎不变）时出现尺度解退化。
        """
        vec_a, priors = self._affine_design(kp2d, rel, height_m)
        if vec_a.shape[0] < 6:
            return 1.0, 0.0
        vec_b = []
        for i, j, _ratio in _PAIR_INDEX:
            ui, vi, di = kp2d[:, i, 0], kp2d[:, i, 1], rel[:, i]
            uj, vj, dj = kp2d[:, j, 0], kp2d[:, j, 1], rel[:, j]
            m = (np.isfinite(ui) & np.isfinite(vi) & np.isfinite(di)
                 & np.isfinite(uj) & np.isfinite(vj) & np.isfinite(dj))
            if not m.any():
                continue
            vec_b.append(self._ray(ui[m], vi[m]) - self._ray(uj[m], vj[m]))
        vec_b = np.concatenate(vec_b)

        z_anchor = self._body_size_depth(kp2d, height_m)
        rel_median = float(np.median(rel[np.isfinite(rel)]))
        a0 = max(z_anchor / max(abs(rel_median), 1e-3), 0.05)
        anchor_weight = 3.0

        def residual(params: np.ndarray) -> np.ndarray:
            a, b = params
            length = np.linalg.norm(a * vec_a + b * vec_b, axis=1)
            rel_res = (length - priors) / priors
            anchor_res = anchor_weight * (a * rel_median + b - z_anchor) / z_anchor
            return np.concatenate([rel_res, [anchor_res]])

        sol = least_squares(residual, x0=np.array([a0, 0.0]), method="trf",
                            bounds=([0.05, -15.0], [8.0, 15.0]), max_nfev=300)
        a, b = float(sol.x[0]), float(sol.x[1])
        if not np.isfinite(a) or a <= 1e-3:
            a, b = a0, 0.0
        return a, b

    # ------------------------------------------------------------------
    # 骨骼约束优化
    # ------------------------------------------------------------------
    def _skeleton_refine(self, ray: np.ndarray, z_prior: np.ndarray,
                         height_m: float) -> np.ndarray:
        """保持二维投影不变，沿光轴优化关节深度以满足骨骼长度先验。"""
        pairs = [(JOINT_INDEX[a], JOINT_INDEX[b], ratio * height_m)
                 for a, b, ratio in SCALE_PRIOR_PAIRS]
        z = z_prior.copy()
        m = np.zeros_like(z)
        v = np.zeros_like(z)
        lam = self.cfg.refine_prior_weight
        lr = self.cfg.refine_step
        floor = np.maximum(z_prior, 1e-3)
        for _ in range(self.cfg.refine_iters):
            grad = lam * (z - z_prior) / floor
            for i, j, length in pairs:
                p_i = ray[:, i] * z[:, [i]]
                p_j = ray[:, j] * z[:, [j]]
                vec = p_i - p_j
                norm = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
                rel_err = (norm - length) / length
                coeff = 2.0 * rel_err / (length * norm)
                grad[:, i] += coeff * vec[:, 2]
                grad[:, j] -= coeff * vec[:, 2]
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            z -= lr * m / (np.sqrt(v) + 1e-8)
            z = np.clip(z, 0.8, 12.0)
        return z

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def estimate(self, stream: SensorStream, height_m: float) -> DepthResult:
        """滚动窗口估计度量深度、骨骼约束优化并反投影得到三维关节。"""
        kp = stream.keypoints2d
        rel = stream.relative_depth

        # 遮挡帧时间插值
        kp_filled = np.stack([self._fill_missing(kp[..., k]) for k in range(2)], axis=-1)
        rel_filled = self._fill_missing(rel)

        # ① 全局仿射尺度恢复（消解单目深度尺度歧义）
        a, b = self._fit_affine(kp, rel, height_m)
        z_prior = np.clip(a * rel_filled + b, 0.8, 12.0)

        # ② 骨骼长度约束下的深度优化
        ray = self._ray(kp_filled[..., 0], kp_filled[..., 1])
        depth = self._skeleton_refine(ray, z_prior, height_m)

        points_cam = ray * depth[..., None]
        joints3d = self.camera.from_camera_frame(points_cam)

        # ③ 滚动时间窗口平滑：Savitzky-Golay 零相位滤波保持轨迹形状
        win = int(self.cfg.savgol_window)
        n = joints3d.shape[0]
        win = min(win if win % 2 == 1 else win + 1, n if n % 2 == 1 else n - 1)
        if win >= 5:
            joints3d = savgol_filter(joints3d, window_length=win, polyorder=2, axis=0)

        depth_mae = float(np.nanmean(np.abs(depth - stream.true_depth)))
        joint_mae = float(np.mean(np.linalg.norm(joints3d - stream.true_joints3d, axis=-1)))
        return DepthResult(joints3d=joints3d, depth=depth, scale_a=a, scale_b=b,
                           valid=stream.visible.copy(),
                           depth_mae=depth_mae, joint_mae=joint_mae)


def depth_scale_error(result: DepthResult, stream: SensorStream) -> float:
    """尺度恢复相对误差，用于方法学评价。"""
    est = result.depth
    true = stream.true_depth
    m = np.isfinite(est) & np.isfinite(true)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(est[m] - true[m]) / np.maximum(true[m], 1e-6)))
