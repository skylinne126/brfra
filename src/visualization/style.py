"""绘图全局样式：中文字体、字号与配色。"""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体候选（Windows / macOS / Linux 通用），按优先级排列。
# SimSun 的西文与数字为衬线体，与文档正文的 Times New Roman + 宋体 组合一致。
CJK_FONT_CANDIDATES: Tuple[str, ...] = (
    "SimSun", "宋体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑",
    "Noto Sans CJK SC", "Noto Serif CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "AR PL UMing CN",
    "PingFang SC", "Heiti SC", "Arial Unicode MS",
)

# PIL 渲染截图时使用的等宽字体（文件名, ttc 索引）
MONO_FONT_CANDIDATES: Tuple[Tuple[str, int], ...] = (
    ("simsun.ttc", 1), ("simsun.ttc", 0),          # NSimSun（等宽中文）
    ("consola.ttf", 0),
    ("msyh.ttc", 0), ("simhei.ttf", 0),
    ("DejaVuSansMono.ttf", 0), ("NotoSansCJK-Regular.ttc", 0),
    ("wqy-zenhei.ttc", 0), ("arial.ttf", 0),
)

FONT_SEARCH_ROOTS = (
    Path("C:/Windows/Fonts"),
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/System/Library/Fonts"), Path("/Library/Fonts"),
    Path.home() / "Library/Fonts",
    Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
)


def find_font_file(name: str) -> Optional[Path]:
    """在常见字体目录中查找字体文件（Linux 下按子目录递归查找）。"""
    for root in FONT_SEARCH_ROOTS:
        if not root.exists():
            continue
        direct = root / name
        if direct.exists():
            return direct
        try:
            for found in root.rglob(name):
                return found
        except OSError:
            continue
    return None


def load_pil_font(size: int,
                  candidates: Sequence[Tuple[str, int]] = MONO_FONT_CANDIDATES):
    """加载一个支持中文的 PIL 字体；全部缺失时退回 PIL 默认字体（不报错）。"""
    from PIL import ImageFont

    for name, index in candidates:
        path = find_font_file(name)
        if path is None:
            continue
        try:
            return ImageFont.truetype(str(path), size, index=index)
        except (OSError, TypeError):
            continue
    return ImageFont.load_default()

PALETTE = {
    "baseline": "#C0504D",     # 传统方法：暖红
    "proposed": "#2E75B6",     # 本文方法：蓝
    "accent": "#7F9F3F",       # 辅助：橄榄绿
    "grid": "#D9D9D9",
    "text": "#1F2329",
    "phase": {
        "SIT": "#BFBFBF", "STAND_UP": "#E8A33D", "WALK_OUT": "#2E75B6",
        "TURN": "#C0504D", "WALK_BACK": "#4F9D5B", "SIT_DOWN": "#8E6BB5",
        "PAUSE": "#D9D9D9",
    },
}


def setup_style() -> None:
    """
    设置中文绘图样式。

    采用宋体（SimSun）作为主字体——其西文与数字为衬线体，与文档正文的
    “Times New Roman + 宋体”组合保持一致；数学符号使用 STIX 字体。
    非 Windows 环境自动回落到系统可用的中文字体。
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(CJK_FONT_CANDIDATES) + ["DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10.5
    plt.rcParams["axes.edgecolor"] = "#8C8C8C"
    plt.rcParams["axes.labelcolor"] = PALETTE["text"]
    plt.rcParams["text.color"] = PALETTE["text"]
    plt.rcParams["xtick.color"] = PALETTE["text"]
    plt.rcParams["ytick.color"] = PALETTE["text"]
    plt.rcParams["grid.color"] = PALETTE["grid"]
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
