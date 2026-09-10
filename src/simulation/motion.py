"""
测试动作真值生成：静态平衡（睁眼/闭眼双脚站立、闭眼单脚站立）与动态平衡（TUG）。

生成过程完全由受试者潜在能力因子驱动，输出三维关节轨迹、人体重心轨迹、
压力垫 COP 参考信号、TUG 阶段标签与步态事件，作为两组对照实验共用的真值。
"""

from typing import Dict, List, Tuple

import numpy as np

from config import CFG, JOINT_INDEX, SEGMENT_DEFS, SimConfig
from src.rollingdepth.anthropometry import STANDARD_MASS_RATIOS, compute_com
from src.simulation.cohort import Subject
from src.simulation.kinematics import SkeletonModel, build_standing_skeleton

# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------


def _true_mass_ratios(rng: np.random.Generator) -> Dict[str, float]:
    """
    个体化真实质量分布：在标准 Dempster 表基础上施加 ±8% 个体扰动。

    评估算法只能使用标准表，因此该扰动构成不可消除的人体测量学误差，
    使实验结论更接近真实工程情况。
    """
    ratios = {}
    for name, _, _, base, _ in SEGMENT_DEFS:
        ratios[name] = base * float(rng.normal(1.0, 0.08))
    total = sum(ratios.values())
    return {k: v / total for k, v in ratios.items()}


def _band_limited_sway(rng: np.random.Generator, n: int, dt: float,
                       rms: float, freqs: Tuple[Tuple[float, float], ...],
                       drift_weight: float = 0.55) -> np.ndarray:
    """生成指定 RMS 的多频段晃动信号（低频主导 + 分数布朗漂移）。"""
    t = np.arange(n) * dt
    sig = np.zeros(n)
    for freq, weight in freqs:
        sig += weight * np.sin(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))
    sig = sig / (np.std(sig) + 1e-12)
    drift = np.cumsum(rng.normal(0.0, 1.0, n))
    drift -= drift.mean()
    drift /= (np.std(drift) + 1e-12)
    mixed = sig + drift_weight * drift
    mixed = mixed / (np.std(mixed) + 1e-12)
    return rms * mixed


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """三次平滑插值，用于阶段内位移过渡。"""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _cop_from_com(com_xy: np.ndarray, instability: float,
                  rng: np.random.Generator, dt: float) -> np.ndarray:
    """
    由重心水平轨迹生成压力垫 COP 参考信号。

    真实 COP 在重心附近高频调节并略有超前，能力越差调节噪声越大。
    """
    n = com_xy.shape[0]
    win = max(3, int(round(0.10 / dt)))
    kernel = np.ones(win) / win
    smooth = np.stack([np.convolve(com_xy[:, k], kernel, mode="same") for k in range(2)], axis=1)
    lead = 0.55 * (com_xy - smooth)
    noise = rng.normal(0.0, 1.0, size=com_xy.shape)
    for k in range(2):
        noise[:, k] = np.convolve(noise[:, k], kernel, mode="same")
    noise = noise / (np.std(noise, axis=0) + 1e-12)
    return com_xy + lead + (0.0012 + 0.010 * instability) * noise


# --------------------------------------------------------------------------
# 静态平衡：睁眼/闭眼双脚站立
# --------------------------------------------------------------------------


def simulate_quiet_stance(subject: Subject, cfg: SimConfig = CFG,
                          eyes_closed: bool = False) -> Dict[str, np.ndarray]:
    """双脚站立静态平衡测试（可切换睁眼/闭眼）。"""
    rng = np.random.default_rng(cfg.subject_seed(subject.sid) + (31 if eyes_closed else 17))
    model = SkeletonModel.from_subject(subject)
    ratios = _true_mass_ratios(rng)

    n = int(cfg.quiet_stance_duration * cfg.fps)
    dt = 1.0 / cfg.fps

    pc = subject.postural_control
    ap = subject.ap_control
    si = subject.sensory_integration
    vision_factor = (1.0 + 1.35 * (1.0 - si)) if eyes_closed else 1.0

    rms_ml = (0.0032 + 0.0060 * (1.0 - pc)) * vision_factor
    rms_ap = (0.0045 + 0.0080 * (1.0 - ap)) * vision_factor
    rms_v = 0.0018 + 0.0022 * (1.0 - pc)

    sway_x = _band_limited_sway(rng, n, dt, rms_ml, ((0.22, 1.0), (0.55, 0.55), (1.35, 0.28)))
    sway_y = _band_limited_sway(rng, n, dt, rms_ap, ((0.18, 1.0), (0.45, 0.60), (1.05, 0.30)))
    sway_z = _band_limited_sway(rng, n, dt, rms_v, ((0.30, 1.0), (1.10, 0.40)), drift_weight=0.2)

    pelvis_ground = model.pelvis_to_ground()
    pelvis = np.zeros((n, 3), dtype=float)
    pelvis[:, 0] = sway_x
    pelvis[:, 1] = sway_y
    pelvis[:, 2] = pelvis_ground + sway_z

    foot_half = 0.100
    left_ankle = np.array([foot_half, 0.0, 0.0])
    right_ankle = np.array([-foot_half, 0.0, 0.0])

    joints = np.zeros((n, len(JOINT_INDEX), 3), dtype=float)
    for i in range(n):
        joints[i] = build_standing_skeleton(
            model, pelvis[i], yaw_rad=0.0, lean_rad=0.0,
            left_ankle=left_ankle, right_ankle=right_ankle,
        )

    com = compute_com(joints, ratios)
    cop = _cop_from_com(com[:, :2], 1.0 - pc, rng, dt)

    return {
        "t": np.arange(n) * dt,
        "joints3d": joints,
        "com3d": com,
        "cop": cop,
        "heading": np.zeros(n),
        "phase": np.array(["quiet_eo" if not eyes_closed else "quiet_ec"] * n),
        "model": model,
        "mass_ratios": ratios,
    }


def simulate_single_leg_stance(subject: Subject, cfg: SimConfig = CFG,
                               eyes_closed: bool = True) -> Dict[str, np.ndarray]:
    """闭眼单脚站立测试：右腿支撑，左腿抬起。"""
    rng = np.random.default_rng(cfg.subject_seed(subject.sid) + (59 if eyes_closed else 43))
    model = SkeletonModel.from_subject(subject)
    ratios = _true_mass_ratios(rng)

    # 单脚站立保持时间：能力不足者会提前放下抬起的脚，测试随即结束
    hold_time = 3.0 + 17.0 * (subject.postural_control ** 1.2) \
        * (0.65 + 0.35 * subject.sensory_integration)
    hold_time *= float(rng.lognormal(0.0, 0.22))
    hold_time = float(np.clip(hold_time, 1.5, cfg.single_leg_duration))

    n = max(45, int(hold_time * cfg.fps))
    dt = 1.0 / cfg.fps

    pc = subject.postural_control
    ap = subject.ap_control
    si = subject.sensory_integration
    vision_factor = (1.0 + 1.60 * (1.0 - si)) if eyes_closed else 1.0

    rms_ml = (0.0075 + 0.0210 * (1.0 - pc)) * vision_factor
    rms_ap = (0.0090 + 0.0215 * (1.0 - ap)) * vision_factor
    rms_v = 0.0025 + 0.0030 * (1.0 - pc)

    sway_x = _band_limited_sway(rng, n, dt, rms_ml, ((0.28, 1.0), (0.62, 0.60), (1.50, 0.30)))
    sway_y = _band_limited_sway(rng, n, dt, rms_ap, ((0.24, 1.0), (0.58, 0.55), (1.25, 0.32)))
    sway_z = _band_limited_sway(rng, n, dt, rms_v, ((0.35, 1.0), (1.20, 0.35)), drift_weight=0.2)

    pelvis_ground = model.pelvis_to_ground()
    pelvis = np.zeros((n, 3), dtype=float)
    pelvis[:, 0] = -0.035 + sway_x
    pelvis[:, 1] = 0.010 + sway_y
    pelvis[:, 2] = pelvis_ground - 0.012 + sway_z

    right_ankle = np.array([-0.045, 0.0, 0.0])
    left_ankle_base = np.array([0.055, 0.030, 0.130])
    joints = np.zeros((n, len(JOINT_INDEX), 3), dtype=float)
    for i in range(n):
        lift = left_ankle_base + np.array([0.35 * sway_x[i], 0.35 * sway_y[i], 0.30 * sway_z[i]])
        joints[i] = build_standing_skeleton(
            model, pelvis[i], yaw_rad=0.0, lean_rad=0.0,
            left_ankle=lift, right_ankle=right_ankle,
        )

    com = compute_com(joints, ratios)
    cop = _cop_from_com(com[:, :2], 1.0 - pc, rng, dt)

    return {
        "t": np.arange(n) * dt,
        "joints3d": joints,
        "com3d": com,
        "cop": cop,
        "heading": np.zeros(n),
        "phase": np.array(["sls_ec" if eyes_closed else "sls_eo"] * n),
        "model": model,
        "mass_ratios": ratios,
        "hold_time": hold_time,
    }


# --------------------------------------------------------------------------
# 动态平衡：起立—行走—转身—坐下（TUG）
# --------------------------------------------------------------------------


def _footfalls_straight(t_start: float, t_end: float, p_start: np.ndarray,
                        p_end: np.ndarray, step_len: float,
                        first_foot: str,
                        bias_by_foot: Dict[str, float] | None = None
                        ) -> Dict[str, List[Tuple[float, np.ndarray]]]:
    """直线行走的落脚事件表（左右脚交替）。

    bias_by_foot 为左右脚的落点前向偏置，用于模拟左右步长不对称。
    """
    bias_by_foot = bias_by_foot or {}
    dist = float(np.linalg.norm(p_end[:2] - p_start[:2]))
    n_steps = max(1, int(round(dist / step_len)))
    step_vec = (p_end - p_start) / n_steps
    norm = float(np.linalg.norm(step_vec[:2])) + 1e-9
    step_dir = step_vec / norm
    events: Dict[str, List[Tuple[float, np.ndarray]]] = {"l": [], "r": []}
    for k in range(1, n_steps + 1):
        t_k = t_start + (t_end - t_start) * k / n_steps
        foot = first_foot if (k % 2 == 1) else ("r" if first_foot == "l" else "l")
        pos = p_start + step_vec * k + step_dir * bias_by_foot.get(foot, 0.0)
        events[foot].append((float(t_k), pos))
    return events


def _footfalls_turn(t_start: float, t_end: float, center: np.ndarray,
                    radius: float, a_start: float, a_end: float,
                    n_steps: int, first_foot: str) -> Dict[str, List[Tuple[float, np.ndarray]]]:
    """原地转身的落脚事件表：落脚点沿圆弧分布。"""
    events: Dict[str, List[Tuple[float, np.ndarray]]] = {"l": [], "r": []}
    n_steps = max(1, n_steps)
    for k in range(1, n_steps + 1):
        t_k = t_start + (t_end - t_start) * k / n_steps
        ang = a_start + (a_end - a_start) * k / n_steps
        lateral = radius * 0.35 * (1.0 if k % 2 == 1 else -1.0)
        pos = center + np.array([radius * np.cos(ang), radius * np.sin(ang), 0.0])
        pos[:2] += lateral * np.array([-np.sin(ang), np.cos(ang)])
        foot = first_foot if (k % 2 == 1) else ("r" if first_foot == "l" else "l")
        events[foot].append((float(t_k), pos))
    return events


def _sample_foot(t: float, events: List[Tuple[float, np.ndarray]],
                 p_init: np.ndarray, swing_frac: float = 0.45) -> np.ndarray:
    """由落脚事件表插值出某一时刻的脚踝位置（含摆动期抬脚弧线）。"""
    if not events:
        return p_init.copy()
    if t <= events[0][0]:
        return p_init.copy() if len(events) == 1 else p_init.copy()
    for k in range(len(events) - 1):
        t0, p0 = events[k]
        t1, p1 = events[k + 1]
        if t0 <= t <= t1:
            cycle = max(t1 - t0, 1e-6)
            swing_start = t1 - swing_frac * cycle
            if t <= swing_start:
                return p0.copy()
            u = (t - swing_start) / max(swing_frac * cycle, 1e-6)
            base = p0 + _smoothstep(np.array([u]))[0] * (p1 - p0)
            lift = 0.045 * np.sin(np.pi * np.clip(u, 0.0, 1.0))
            return base + np.array([0.0, 0.0, lift])
    return events[-1][1].copy()


def simulate_tug(subject: Subject, cfg: SimConfig = CFG) -> Dict[str, np.ndarray]:
    """起立—行走—转身—坐下（TUG）全过程三维真值轨迹。"""
    rng = np.random.default_rng(cfg.subject_seed(subject.sid) + 71)
    model = SkeletonModel.from_subject(subject)
    ratios = _true_mass_ratios(rng)

    ms = subject.muscle_strength
    gs = subject.gait_stability

    seat_h = 0.45
    pelvis_ground = model.pelvis_to_ground()
    stand_time = 0.90 + 2.20 * (1.0 - ms)
    sit_time = 0.90 + 2.20 * (1.0 - ms)
    walk_speed = 0.45 + 0.75 * gs
    step_len = 0.30 + 0.35 * gs
    walk_time = cfg.tug_distance / walk_speed
    turn_time = 1.10 + 2.60 * (1.0 - gs)
    sit_hold = 1.60

    t0 = 0.0
    t_stand0 = t0 + sit_hold
    t_stand1 = t_stand0 + stand_time
    t_walk1 = t_stand1 + walk_time
    t_turn1 = t_walk1 + turn_time
    t_back1 = t_turn1 + walk_time
    t_sit1 = t_back1 + sit_time
    total = t_sit1 + 0.40

    n = int(total * cfg.fps)
    times = np.arange(n) / cfg.fps

    # ---------------- 骨盆轨迹关键帧 ----------------
    key_t = np.array([t0, t_stand0, t_stand1, t_walk1, t_turn1, t_back1, t_sit1, total])
    key_y = np.array([0.0, 0.0, 0.06, cfg.tug_distance, cfg.tug_distance + 0.04, 0.12, 0.02, 0.02])
    key_z = np.array([seat_h, seat_h, pelvis_ground, pelvis_ground,
                      pelvis_ground, pelvis_ground, seat_h, seat_h])
    key_yaw = np.array([0.0, 0.0, 0.0, 0.0, np.pi, np.pi, np.pi, np.pi])
    key_lean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    pelvis = np.zeros((n, 3), dtype=float)
    pelvis[:, 1] = np.interp(times, key_t, key_y)
    pelvis[:, 2] = np.interp(times, key_t, key_z)
    yaw = np.interp(times, key_t, key_yaw)

    # 起立/坐下过程的前倾角（起立前倾、坐下后倾）
    lean = np.zeros(n)
    for i, t in enumerate(times):
        if t_stand0 <= t <= t_stand1:
            u = (t - t_stand0) / max(stand_time, 1e-6)
            lean[i] = np.deg2rad(38.0 * (1.0 - ms) + 12.0) * np.sin(np.pi * u)
        elif t_back1 <= t <= t_sit1:
            u = (t - t_back1) / max(sit_time, 1e-6)
            lean[i] = -np.deg2rad(26.0 * (1.0 - ms) + 8.0) * np.sin(np.pi * u)

    # 行走期骨盆横向摆动与竖直起伏（步态稳定性越差摆动越大）
    walk_mask = ((times >= t_stand1) & (times <= t_walk1)) | ((times >= t_turn1) & (times <= t_back1))
    step_freq = walk_speed / step_len
    lat_amp = 0.012 + 0.028 * (1.0 - gs)
    vert_amp = 0.012 + 0.014 * (1.0 - gs)
    pelvis[:, 0] = lat_amp * np.sin(2.0 * np.pi * step_freq * times + rng.uniform(0, 2 * np.pi))
    pelvis[:, 2] += np.where(walk_mask, vert_amp * np.abs(np.sin(np.pi * step_freq * times)), 0.0)

    # 转身期骨盆绕小圆运动
    turn_mask = (times >= t_walk1) & (times <= t_turn1)
    if turn_mask.any():
        u = (times[turn_mask] - t_walk1) / max(turn_time, 1e-6)
        pelvis[turn_mask, 0] += 0.22 * np.sin(np.pi * u)
        pelvis[turn_mask, 1] += 0.18 * (1.0 - np.cos(np.pi * u)) * 0.5

    # 转身朝向角加抖动
    yaw = yaw + np.where(walk_mask, np.deg2rad(2.5) * (1.0 - gs)
                        * np.sin(2.0 * np.pi * step_freq * times + 0.7), 0.0)

    # ---------------- 落脚事件表 ----------------
    # 行走时双脚向中线收拢，步宽随步态稳定性下降而增大
    walk_half = 0.035 + 0.045 * (1.0 - gs)
    start_l = np.array([walk_half, 0.35, 0.0])
    start_r = np.array([-walk_half, 0.35, 0.0])
    far_l = np.array([walk_half, cfg.tug_distance, 0.0])
    far_r = np.array([-walk_half, cfg.tug_distance, 0.0])
    back_l = np.array([walk_half, 0.28, 0.0])
    back_r = np.array([-walk_half, 0.28, 0.0])

    # 左右步长不对称：步态稳定性越差，左右腿迈步长度差异越明显
    asym_ratio = 0.32 * (1.0 - gs)
    bias = 0.5 * asym_ratio * step_len
    bias_map = {"l": bias, "r": -bias}

    ev_out = _footfalls_straight(t_stand1, t_walk1, start_l, far_l, step_len, "l",
                                 bias_by_foot=bias_map)
    ev_out_r = _footfalls_straight(t_stand1, t_walk1, start_r, far_r, step_len, "r",
                                   bias_by_foot=bias_map)
    turn_center = np.array([0.0, cfg.tug_distance + 0.15, 0.0])
    ev_turn = _footfalls_turn(t_walk1, t_turn1, turn_center, 0.32, np.pi * 0.5, np.pi * 1.5,
                              max(2, int(round(turn_time / (1.0 / step_freq)))), "l")
    ev_turn_r = _footfalls_turn(t_walk1, t_turn1, turn_center, 0.32, np.pi * 1.5, np.pi * 0.5,
                                max(2, int(round(turn_time / (1.0 / step_freq)))), "r")
    ev_back = _footfalls_straight(t_turn1, t_back1, far_l, back_l, step_len, "r",
                                  bias_by_foot=bias_map)
    ev_back_r = _footfalls_straight(t_turn1, t_back1, far_r, back_r, step_len, "l",
                                    bias_by_foot=bias_map)

    left_events = ev_out["l"] + ev_turn["l"] + ev_back["l"]
    right_events = ev_out_r["r"] + ev_turn_r["r"] + ev_back_r["r"]
    left_events.sort(key=lambda e: e[0])
    right_events.sort(key=lambda e: e[0])

    # ---------------- 逐帧生成骨架 ----------------
    joints = np.zeros((n, len(JOINT_INDEX), 3), dtype=float)
    arm_amp = np.deg2rad(9.0 + 14.0 * gs)
    for i, t in enumerate(times):
        l_ankle = _sample_foot(t, left_events, start_l)
        r_ankle = _sample_foot(t, right_events, start_r)
        swing = arm_amp * np.sin(2.0 * np.pi * step_freq * t)
        joints[i] = build_standing_skeleton(
            model, pelvis[i], yaw_rad=float(yaw[i]), lean_rad=float(lean[i]),
            left_ankle=l_ankle, right_ankle=r_ankle,
            arm_swing_l=float(swing), arm_swing_r=float(-swing),
        )

    com = compute_com(joints, ratios)
    cop = _cop_from_com(com[:, :2], 1.0 - gs, rng, 1.0 / cfg.fps)

    # ---------------- 阶段标签 ----------------
    phase = np.empty(n, dtype=object)
    phase[times < t_stand0] = "SIT"
    phase[(times >= t_stand0) & (times < t_stand1)] = "STAND_UP"
    phase[(times >= t_stand1) & (times < t_walk1)] = "WALK_OUT"
    phase[(times >= t_walk1) & (times < t_turn1)] = "TURN"
    phase[(times >= t_turn1) & (times < t_back1)] = "WALK_BACK"
    phase[times >= t_back1] = "SIT_DOWN"

    return {
        "t": times,
        "joints3d": joints,
        "com3d": com,
        "cop": cop,
        "heading": yaw,
        "phase": phase,
        "phase_bounds": {
            "SIT": (t0, t_stand0),
            "STAND_UP": (t_stand0, t_stand1),
            "WALK_OUT": (t_stand1, t_walk1),
            "TURN": (t_walk1, t_turn1),
            "WALK_BACK": (t_turn1, t_back1),
            "SIT_DOWN": (t_back1, t_sit1),
        },
        "step_events": {"l": left_events, "r": right_events},
        "model": model,
        "mass_ratios": ratios,
        "true_tug_time": float(t_sit1),
    }
