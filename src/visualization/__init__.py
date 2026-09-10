"""可视化模块：论文风格图表与文本截图渲染。"""

from .style import setup_style, PALETTE
from .plots import (
    render_text_shot,
    plot_baseline_result,
    plot_com_reconstruction,
    plot_tug_4d,
    plot_gait_language,
    plot_method_comparison,
)

__all__ = [
    "setup_style",
    "PALETTE",
    "render_text_shot",
    "plot_baseline_result",
    "plot_com_reconstruction",
    "plot_tug_4d",
    "plot_gait_language",
    "plot_method_comparison",
]
