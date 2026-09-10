"""
TUG（起立—行走—转身—坐下）阶段自动切分。

以四维重建轨迹为输入，通过重心高度曲线、水平运动速度与人体朝向角速度
三类事件构建状态机，自动识别 SIT / STAND_UP / WALK_OUT / TURN /
WALK_BACK / SIT_DOWN 六个阶段，为动态平衡指标提供阶段边界。
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from config import CFG, JOINT_INDEX, SimConfig
from src.mosca.trajectory4d import heading_series


@dataclass
class PhaseSegmentation:
    """阶段切分结果。"""

    phase: np.ndarray                  # (T,) 阶段标签
    bounds: Dict[str, Tuple[float, float]]   # 各阶段起止时刻（s）
    heading: np.ndarray                # (T,) 人体朝向角（弧度）
    angular_velocity: np.ndarray       # (T,) 朝向角速度（度/秒）
    speed: np.ndarray                  # (T,) 水平运动速度（米/秒）
    pelvis_height: np.ndarray          # (T,) 骨盆高度（米）

    def duration(self, name: str, fps: float) -> float:
        return float(np.sum(self.phase == name) / fps)


class TugPhaseSegmenter:
    """TUG 阶段切分器。"""

    def __init__(self, fps: float, cfg: SimConfig = CFG):
        self.fps = float(fps)
        self.cfg = cfg

    @staticmethod
    def _smooth(x: np.ndarray, win: int) -> np.ndarray:
        win = max(3, int(win))
        kernel = np.ones(win) / win
        return np.convolve(x, kernel, mode="same")

    def segment(self, joints3d: np.ndarray, com3d: np.ndarray) -> PhaseSegmentation:
        """由四维轨迹自动切分 TUG 阶段。"""
        fps = self.fps
        n = joints3d.shape[0]
        t = np.arange(n) / fps

        pelvis = joints3d[:, JOINT_INDEX["pelvis"], :]
        h = self._smooth(pelvis[:, 2], int(0.4 * fps))

        seated = float(np.median(h[: max(3, int(0.8 * fps))]))
        mid = h[int(n * 0.30):int(n * 0.70)]
        standing = float(np.percentile(mid, 90)) if mid.size else float(h.max())
        span = max(standing - seated, 1e-3)
        rise_thr = seated + 0.15 * span
        full_thr = seated + 0.85 * span

        above_rise = np.where(h > rise_thr)[0]
        above_full = np.where(h > full_thr)[0]
        if above_rise.size == 0 or above_full.size == 0:
            phase = np.array(["UNKNOWN"] * n, dtype=object)
            return PhaseSegmentation(phase=phase, bounds={}, heading=np.zeros(n),
                                     angular_velocity=np.zeros(n), speed=np.zeros(n),
                                     pelvis_height=h)

        stand_start = int(above_rise[0])
        stand_end = int(above_full[above_full >= stand_start][0])
        sit_start = int(above_full[above_full <= above_rise[-1]][-1])
        sit_end = int(above_rise[-1])

        heading = heading_series(joints3d)
        omega = np.gradient(np.degrees(heading)) * fps
        omega = self._smooth(np.abs(omega), max(3, int(0.30 * fps)))

        vel = np.gradient(pelvis[:, :2], axis=0) * fps
        speed = self._smooth(np.linalg.norm(vel, axis=1), max(3, int(0.30 * fps)))

        phase = np.array(["PAUSE"] * n, dtype=object)
        phase[:stand_start] = "SIT"
        phase[stand_start:stand_end] = "STAND_UP"
        phase[stand_end:sit_start] = "PAUSE"
        phase[sit_start:sit_end] = "SIT_DOWN"
        phase[sit_end:] = "SIT"

        # 在站立到坐下之间分离行走段与转身段：
        # 先由速度阈值定位两段行走，再把两者之间的时段交给朝向角速度进一步细化，
        # 若中间不存在明显转身窗口，则整体作为转身段。
        lo, hi = stand_end, sit_start
        if hi > lo:
            walk_flag = speed[lo:hi] > self.cfg.walk_speed_threshold
            bouts = self._contiguous(walk_flag)
            if len(bouts) >= 2:
                first, last = bouts[0], bouts[-1]
                mid_lo, mid_hi = lo + first[1], lo + last[0]
                phase[lo:mid_lo] = "WALK_OUT"
                phase[mid_hi:hi] = "WALK_BACK"
            else:
                mid_lo = lo + max(2, (hi - lo) // 4)
                mid_hi = hi - max(2, (hi - lo) // 4)
                phase[lo:mid_lo] = "WALK_OUT"
                phase[mid_hi:hi] = "WALK_BACK"

            if mid_hi > mid_lo:
                turn_flag = omega[mid_lo:mid_hi] > self.cfg.turn_rate_threshold
                turn_bouts = self._contiguous(turn_flag)
                if turn_bouts:
                    best = max(turn_bouts, key=lambda b: b[1] - b[0])
                    t0, t1 = mid_lo + best[0], mid_lo + best[1]
                    # 转身窗口之外的低速时段归入相邻行走段
                    phase[mid_lo:t0] = "WALK_OUT"
                    phase[t1:mid_hi] = "WALK_BACK"
                    phase[t0:t1] = "TURN"
                else:
                    phase[mid_lo:mid_hi] = "TURN"

        bounds: Dict[str, Tuple[float, float]] = {}
        for name in ("SIT", "STAND_UP", "WALK_OUT", "TURN", "WALK_BACK", "SIT_DOWN"):
            idx = np.where(phase == name)[0]
            if idx.size:
                bounds[name] = (float(t[idx[0]]), float(t[idx[-1]] + 1.0 / fps))
        return PhaseSegmentation(phase=phase, bounds=bounds, heading=heading,
                                 angular_velocity=omega, speed=speed, pelvis_height=h)

    @staticmethod
    def _contiguous(mask: np.ndarray) -> list:
        """返回布尔序列中的连续 True 区间列表 [(start, end), ...]。"""
        bouts = []
        start = None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                bouts.append((start, i))
                start = None
        if start is not None:
            bouts.append((start, len(mask)))
        return [b for b in bouts if b[1] - b[0] >= 3]
