"""
生成 PyCharm 风格源码截图，用于代码复现报告的附录 A。

用 PIL 把工程源码按行渲染为带行号、语法高亮的图片，
视觉风格对齐 PyCharm 浅色主题（关键字蓝、字符串绿、注释灰、数字蓝紫）。

用法
----
    python tools/render_code_shots.py                 # 默认每张 70 行
    python tools/render_code_shots.py --lines 55      # 指定每张行数
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import token as TOKEN
import tokenize
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SHOT_DIR = PROJECT_ROOT / "output" / "code_shots"

from src.visualization.style import load_pil_font  # noqa: E402  （需先设置 sys.path）

# 附录中出现顺序（核心算法与主流程；仿真层、评价与可视化模块随工程一并提交）
SOURCE_ORDER: Tuple[str, ...] = (
    "src/rollingdepth/anthropometry.py",
    "src/rollingdepth/depth.py",
    "src/rollingdepth/com.py",
    "src/rollingdepth/sway.py",
    "src/mosca/trajectory4d.py",
    "src/mosca/phase.py",
    "src/mosca/tug.py",
    "src/gaitllm/tokenizer.py",
    "src/gaitllm/lm.py",
    "src/gaitllm/report.py",
    "src/baseline/sway2d.py",
    "src/fusion/risk.py",
    "src/scoring.py",
    "main.py",
)

# PyCharm 浅色主题配色
COLORS: Dict[str, str] = {
    "default": "#1A1A1A",
    "keyword": "#0033B3",
    "string": "#067D17",
    "comment": "#8C8C8C",
    "number": "#1750EB",
    "builtin": "#7A3E9D",
    "decorator": "#9E880D",
    "def": "#00627A",
    "lineno": "#AEB3B8",
    "bg": "#FFFFFF",
    "gutter": "#F7F7F7",
    "rule": "#E6E6E6",
}

KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
    "del", "elif", "else", "except", "False", "finally", "for", "from", "global",
    "if", "import", "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass",
    "raise", "return", "True", "try", "while", "with", "yield",
}
BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
    "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "tuple",
    "zip", "isinstance", "getattr", "setattr", "super", "type", "map", "filter",
}

# 等宽中文字体缺失的符号，渲染时替换为等价写法，避免出现方框
SYMBOL_MAP = str.maketrans({
    "\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2190": "<-",
    "\u00b2": "^2", "\u00b1": "+/-", "\u2248": "~", "\u00b5": "u",
    "\u00b0": "deg", "\u00d7": "x", "\u2265": ">=", "\u2264": "<=",
    "\u00b7": ".", "\u03c9": "w", "\u03c3": "sigma", "\u03bb": "lambda",
    "\u0394": "Delta", "\u03a3": "Sum", "\u2260": "!=", "\u221a": "sqrt",
    "\u221e": "inf", "\u2191": "^", "\u2193": "v",
})


def _sanitize(text: str) -> str:
    """把字体缺失的符号替换为等价 ASCII 写法。"""
    return text.translate(SYMBOL_MAP)


def _font(size: int):
    """
    选择等宽且支持中文的字体。

    Consolas 不含中文字形，会渲染成方框，因此优先使用 NSimSun
    （等宽中文字体，ASCII 为半宽）；非 Windows 环境自动回落到系统可用字体。
    """
    return load_pil_font(size)


def _color_for(token_type: int, text: str, prev_type: int,
               prev_type_text: str = "") -> str:
    if token_type == TOKEN.COMMENT:
        return COLORS["comment"]
    if token_type == TOKEN.STRING:
        return COLORS["string"]
    if token_type == TOKEN.NUMBER:
        return COLORS["number"]
    if token_type == TOKEN.NAME:
        if text in KEYWORDS:
            return COLORS["keyword"]
        if prev_type == TOKEN.NAME and prev_type_text == "def":
            return COLORS["def"]
        if text in BUILTINS:
            return COLORS["builtin"]
        return COLORS["default"]
    if token_type == TOKEN.OP and text == "@":
        return COLORS["decorator"]
    return COLORS["default"]


def source_spans(source: str) -> Tuple[List[str], List[List[Tuple[int, int, str]]]]:
    """返回每行文本与每行的彩色区间列表（输入应为已做符号替换的源码）。"""
    lines = source.split("\n")
    spans: List[List[Tuple[int, int, str]]] = [[] for _ in lines]
    prev_type = -1
    prev_type_text = ""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        tokens = []

    for tok in tokens:
        ttype, text = tok.type, tok.string
        (srow, scol), (erow, ecol) = tok.start, tok.end
        if ttype in (TOKEN.NEWLINE, TOKEN.NL, TOKEN.INDENT, TOKEN.DEDENT,
                     TOKEN.ENCODING, TOKEN.ENDMARKER):
            prev_type, prev_type_text = ttype, text
            continue
        color = _color_for(ttype, text, prev_type, prev_type_text)
        if srow == erow:
            if 1 <= srow <= len(spans):
                spans[srow - 1].append((scol, ecol, color))
        else:
            for row in range(srow, min(erow, len(spans)) + 1):
                start = scol if row == srow else 0
                end = ecol if row == erow else len(lines[row - 1])
                spans[row - 1].append((start, end, color))
        prev_type, prev_type_text = ttype, text
    return lines, spans


def render_chunk(lines: List[str], spans: List[List[Tuple[int, int, str]]],
                 first_line_no: int, out_path: Path, font_size: int = 12,
                 width: int = 820, pad: int = 14) -> Path:
    """把一段源码渲染为图片。"""
    font = _font(font_size)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 5
    gutter_w = 62
    height = pad * 2 + line_h * len(lines)

    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, gutter_w, height], fill=COLORS["gutter"])
    draw.line([gutter_w, 0, gutter_w, height], fill=COLORS["rule"], width=1)

    for i, line in enumerate(lines):
        y = pad + i * line_h
        lineno = str(first_line_no + i)
        draw.text((gutter_w - 10, y), lineno, font=font, fill=COLORS["lineno"], anchor="ra")
        if not spans[i]:
            draw.text((gutter_w + 12, y), line, font=font, fill=COLORS["default"])
            continue
        for start, end, color in sorted(spans[i]):
            start = max(0, min(start, len(line)))
            end = max(start, min(end, len(line)))
            if end <= start:
                continue
            x = gutter_w + 12 + font.getlength(line[:start])
            draw.text((x, y), line[start:end], font=font, fill=color)

    image.save(out_path)
    return out_path


def build_shots(lines_per_shot: int = 80, font_size: int = 11) -> List[Dict[str, object]]:
    """渲染全部源码，返回截图清单。"""
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []
    index = 0
    for rel in SOURCE_ORDER:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        source = _sanitize(path.read_text(encoding="utf-8"))
        lines, spans = source_spans(source)
        total = len(lines)
        for start in range(0, total, lines_per_shot):
            chunk = lines[start:start + lines_per_shot]
            chunk_spans = spans[start:start + lines_per_shot]
            index += 1
            name = f"code_{index:02d}_{Path(rel).stem}_{start + 1}.png"
            render_chunk(chunk, chunk_spans, start + 1, SHOT_DIR / name, font_size)
            manifest.append({
                "index": index, "file": rel, "start_line": start + 1,
                "end_line": min(start + lines_per_shot, total),
                "image": name,
            })
    (SHOT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成附录代码截图")
    parser.add_argument("--lines", type=int, default=80, help="每张截图包含的代码行数")
    parser.add_argument("--font", type=int, default=11, help="字号")
    args = parser.parse_args()
    manifest = build_shots(args.lines, args.font)
    print(f"已生成 {len(manifest)} 张代码截图，输出目录：{SHOT_DIR}")


if __name__ == "__main__":
    main()
