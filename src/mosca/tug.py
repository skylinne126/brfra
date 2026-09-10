"""
TUG 动态平衡指标提取。

在阶段切分结果与四维重建轨迹基础上，计算动态平衡与步态的量化指标：
各阶段耗时、步数/步频/步幅/步宽、左右步幅对称性、转身角速度峰值、
行走路径长度与路径效率、行走过程中重心左右方向晃动等。
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import CFG, JOINT_INDEX, SimConfig
from src.mosca.phase import PhaseSegmentation


@dataclass
class TugMetrics:
    """TUG 动态平衡指标。"""

    total_time: float            # 总用时（s）
    stand_up_time: float         # 起立耗时（s）
    sit_down_time: float         # 坐下耗时（s）
    turn_time: float             # 转身耗时（s）
    walk_out_time: float         # 去程行走耗时（s）
    walk_back_time: float        # 返程行走耗时（s）
    step_count: int              # 步数
    cadence_spm: float           # 步频（步/分）
    step_length: float           # 平均步长（m，相邻两次落脚点间距离）
    step_width: float            # 平均步宽（m）
    step_asymmetry: float        # 左右步幅不对称指数
    turn_peak_rate: float        # 转身角速度峰值（度/秒）
    path_length: float           # 行走路径总长度（m）
    path_efficiency: float       # 路径效率（直线距离 / 路径长度）
    com_sway_lateral: float      # 行走中重心左右晃动 RMS（mm）
    pelvis_rise_velocity: float  # 起立时骨盆上升峰值速度（m/s）

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class TugAnalyzer:
    """TUG 指标分析器。"""

    def __init__(self, fps: float, cfg: SimConfig = CFG):
        self.fps = float(fps)
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _foot_strikes(self, joints3d: np.ndarray, side: str) -> np.ndarray:
        """
        检测足跟着地事件，返回着地帧序号。

        单目重建的高度存在随阶段漂移的整体偏移，绝对高度阈值不可靠，
        因此改用踝关节"水平运动速度"判定：支撑期脚在地面几乎静止，
        摆动期脚快速前移。采用施密特触发滞回（进入 0.12 m/s、退出 0.25 m/s）
        避免同一次着地被重复计数。
        """
        ankle = joints3d[:, JOINT_INDEX[f"{side}_ankle"], :]
        velocity = np.linalg.norm(np.gradient(ankle[:, :2], axis=0), axis=1) * self.fps
        win = max(3, int(0.10 * self.fps))
        kernel = np.ones(win) / win
        velocity = np.convolve(velocity, kernel, mode="same")

        enter, exit_ = self.cfg.foot_contact_speed, self.cfg.foot_release_speed
        strikes = []
        in_contact = False
        for i, value in enumerate(velocity):
            if not in_contact and value < enter:
                in_contact = True
                strikes.append(i)
            elif in_contact and value > exit_:
                in_contact = False
        return np.asarray(strikes, dtype=int)

    def _step_events(self, joints3d: np.ndarray, walk_mask: np.ndarray
                     ) -> Tuple[int, float, float]:
        """
        由骨盆竖直方向"起伏"计数步数（每次迈步骨盆升高一次）。

        相比直接检测足部着地，该方法对单目重建的高度整体偏移与噪声更稳健。

        返回 (步数, 步长, 左右不对称指数)。
        """
        fps = self.fps
        idx = np.where(walk_mask)[0]
        if idx.size < 5:
            return 0, 0.0, 0.0

        pelvis = joints3d[idx, JOINT_INDEX["pelvis"], :]
        z = pelvis[:, 2]
        win = max(3, int(0.15 * fps))
        kernel = np.ones(win) / win
        z = np.convolve(z, kernel, mode="same")

        # 局部极大值 = 一次迈步
        peaks = []
        min_gap = max(2, int(0.25 * fps))
        for i in range(1, len(z) - 1):
            if z[i] >= z[i - 1] and z[i] > z[i + 1]:
                if not peaks or i - peaks[-1] >= min_gap:
                    peaks.append(i)
                elif z[i] > z[peaks[-1]]:
                    peaks[-1] = i
        step_count = len(peaks)
        if step_count < 1:
            return 0, 0.0, 0.0

        path = float(np.sum(np.linalg.norm(np.diff(pelvis[:, :2], axis=0), axis=1)))

        # 左右步长不对称与步长：以脚相对骨盆的前后摆动幅度衡量
        direction = pelvis[-1, :2] - pelvis[0, :2]
        norm = np.linalg.norm(direction)
        ranges: List[float] = []
        if norm > 1e-6:
            unit = direction / norm
            for side in ("l", "r"):
                ankle = joints3d[idx, JOINT_INDEX[f"{side}_ankle"], :2]
                rel = (ankle - pelvis[:, :2]) @ unit
                rel = np.convolve(rel, kernel, mode="same")
                ranges.append(float(np.percentile(rel, 95) - np.percentile(rel, 5)))
        if len(ranges) == 2 and (ranges[0] + ranges[1]) > 1e-6:
            asymmetry = float(abs(ranges[0] - ranges[1]) / (ranges[0] + ranges[1]))
            step_length = float(np.mean(ranges))
        elif ranges:
            asymmetry, step_length = 0.0, float(ranges[0])
        else:
            asymmetry, step_length = 0.0, (path / step_count if step_count else 0.0)
        return step_count, step_length, asymmetry

    def analyze(self, joints3d: np.ndarray, com3d: np.ndarray,
                seg: PhaseSegmentation) -> TugMetrics:
        """计算全部 TUG 指标。"""
        fps = self.fps
        n = joints3d.shape[0]
        t = np.arange(n) / fps
        pelvis = joints3d[:, JOINT_INDEX["pelvis"], :]

        def dur(name: str) -> float:
            return seg.duration(name, fps)

        walk_mask = (seg.phase == "WALK_OUT") | (seg.phase == "WALK_BACK")
        walk_time = float(walk_mask.sum() / fps)

        # ---- 步态事件 ----
        step_count, step_length, asymmetry = self._step_events(joints3d, walk_mask)

        # ---- 步宽 ----
        lx = joints3d[:, JOINT_INDEX["l_ankle"], 0]
        rx = joints3d[:, JOINT_INDEX["r_ankle"], 0]
        step_width = float(np.mean(np.abs(lx[walk_mask] - rx[walk_mask]))) if walk_mask.any() else 0.0

        # ---- 转身 ----
        turn_mask = seg.phase == "TURN"
        turn_peak = float(seg.angular_velocity[turn_mask].max()) if turn_mask.any() else 0.0
        turn_peak = min(turn_peak, 360.0)

        # ---- 路径：分单程统计，避免往返路径相互抵消 ----
        def bout_efficiency(name: str) -> Tuple[float, float]:
            mask = seg.phase == name
            if mask.sum() < 4:
                return 0.0, 0.0
            path = pelvis[mask, :2]
            length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            straight = float(np.linalg.norm(path[-1] - path[0]))
            return length, (straight / length if length > 1e-6 else 0.0)

        len_out, eff_out = bout_efficiency("WALK_OUT")
        len_back, eff_back = bout_efficiency("WALK_BACK")
        if len_out >= len_back:
            path_length, efficiency = len_out, eff_out
        else:
            path_length, efficiency = len_back, eff_back

        # ---- 行走中重心左右晃动 ----
        if walk_mask.sum() > 5:
            p = pelvis[walk_mask, :2]
            direction = p[-1] - p[0]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                unit = direction / norm
                lateral = np.array([-unit[1], unit[0]])
                dev = (com3d[walk_mask, :2] - com3d[walk_mask, :2].mean(axis=0)) @ lateral
                com_sway = float(np.sqrt(np.mean(dev ** 2)) * 1000.0)
            else:
                com_sway = 0.0
        else:
            com_sway = 0.0

        # ---- 起立峰值速度 ----
        stand_mask = seg.phase == "STAND_UP"
        if stand_mask.sum() > 2:
            vel = np.gradient(pelvis[stand_mask, 2]) * fps
            rise_vel = float(np.max(vel))
        else:
            rise_vel = 0.0

        cadence = float(step_count / walk_time * 60.0) if walk_time > 0.5 else 0.0
        return TugMetrics(
            total_time=float(t[-1] + 1.0 / fps) if n else 0.0,
            stand_up_time=dur("STAND_UP"), sit_down_time=dur("SIT_DOWN"),
            turn_time=dur("TURN"), walk_out_time=dur("WALK_OUT"),
            walk_back_time=dur("WALK_BACK"),
            step_count=step_count, cadence_spm=cadence,
            step_length=step_length, step_width=step_width,
            step_asymmetry=asymmetry, turn_peak_rate=turn_peak,
            path_length=path_length, path_efficiency=efficiency,
            com_sway_lateral=com_sway, pelvis_rise_velocity=rise_vel,
        )
