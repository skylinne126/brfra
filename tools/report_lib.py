"""
按模板格式生成《平衡能力与跌倒风险评估代码复现》报告。

排版优化
--------
1. 字体分层：正文宋体 + Times New Roman，各级标题黑体 + Arial，
   控制台/代码块 Consolas + 宋体（保持等宽对齐），表头黑体加粗；
2. 公式：以 LaTeX 书写核心数学模型，经 latex2mathml 转 MathML、
   再由 Office 自带的 MML2OMML.XSL 转为 OMML，作为 Word 原生公式插入；
3. 版式：居中题注、页码页脚、自动目录（生成后调用 Word 更新域）。

复用模板 docx 中的样式表、编号表、主题与页面设置，只重新生成 document.xml，
从而保证排版与模板一致。

用法
----
    python tools/build_report.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from lxml import etree

try:                                    # 公式排版为可选项，缺失时自动降级
    import latex2mathml.converter as l2m
except ImportError:                     # pragma: no cover
    l2m = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
# 报表模板排版资源随工程一起分发（assets/report_template/），保证 clone 后即可运行
TEMPLATE_DIR = PROJECT_ROOT / "assets" / "report_template"
# 可选：若存在原始模板 docx，则优先从它重新解包（便于替换模板）
TEMPLATE_DOCX_CANDIDATES = (
    PROJECT_ROOT / "assets" / "report_template.docx",
    WORKSPACE / "群智感知下的应急任务分配优化代码复现.docx",
)
OUT_DIR = PROJECT_ROOT / "output"
REPORT_PATH = PROJECT_ROOT / "平衡能力与跌倒风险评估代码复现.docx"

# MathML → OMML 转换表由 Office 自带；按常见安装路径依次探测
MML2OMML_CANDIDATES = (
    Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files\Microsoft Office\root\Office15\MML2OMML.XSL"),
    Path("/Applications/Microsoft Word.app/Contents/Resources/MML2OMML.XSL"),
    Path("/usr/share/microsoft/office/MML2OMML.XSL"),
)

TEMPLATE_PARTS = (
    "[Content_Types].xml", "_rels/.rels",
    "docProps/app.xml", "docProps/core.xml", "docProps/custom.xml",
    "word/styles.xml", "word/numbering.xml", "word/settings.xml",
    "word/fontTable.xml", "word/theme/theme1.xml",
)


def find_mml2omml() -> Optional[Path]:
    """探测 Office 自带的 MathML→OMML 转换表，找不到时返回 None。"""
    env = os.environ.get("MML2OMML_XSL")
    if env and Path(env).exists():
        return Path(env)
    for path in MML2OMML_CANDIDATES:
        if path.exists():
            return path
    return None


def find_font(candidates: Sequence[Tuple[str, int]]) -> Optional[Path]:
    """
    在 Windows / macOS / Linux 常见字体目录中查找第一个可用字体。

    candidates 为 (文件名, ttc 索引) 序列；返回找到的字体路径。
    """
    roots = [
        Path("C:/Windows/Fonts"),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
        Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
        Path("/System/Library/Fonts"), Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
    ]
    for name, _index in candidates:
        for root in roots:
            if not root.exists():
                continue
            direct = root / name
            if direct.exists():
                return direct
            for found in root.rglob(name):        # Linux 下字体按目录分层存放
                return found
    return None


# --------------------------------------------------------------------------
# 样式 ID（与模板 styles.xml 一致）
# --------------------------------------------------------------------------
S_TITLE = "14"
S_H1 = "2"
S_H2 = "3"
S_H3 = "4"
S_H4 = "5"
S_CAPTION = "11"
S_BODY = "12"
S_APPENDIX = "19"
S_FIGURE = "20"
S_REFERENCE = "21"

# --------------------------------------------------------------------------
# 字体方案
# --------------------------------------------------------------------------
FONT_BODY = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
             'w:eastAsia="宋体" w:cs="Times New Roman"/>')
FONT_HEAD = ('<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="黑体" w:cs="Arial"/>')
FONT_MONO = ('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="宋体" '
             'w:cs="Consolas"/>')

EMU_PER_INCH = 914400
CONTENT_WIDTH_IN = 5.72
CONTENT_TWIPS = 8306


def esc(text: str) -> str:
    """XML 转义。"""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def run(text: str, bold: bool = False, italic: bool = False,
        font: str = FONT_BODY, size: Optional[int] = None) -> str:
    """构造文本运行块。size 单位为半磅。"""
    props = font
    if bold:
        props += "<w:b/>"
    if italic:
        props += "<w:i/>"
    if size:
        props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    pieces = text.split("\n")
    body = ""
    for i, piece in enumerate(pieces):
        if i:
            body += "<w:br/>"
        body += f'<w:t xml:space="preserve">{esc(piece)}</w:t>'
    return f'<w:r><w:rPr>{props}</w:rPr>{body}</w:r>'


def para(text: str = "", style: Optional[str] = None, numid: Optional[int] = None,
         first_line: Optional[int] = None, center: bool = False,
         bottom_border: bool = False, bold: bool = False,
         font: str = FONT_BODY, size: Optional[int] = None,
         space_before: Optional[int] = None, space_after: Optional[int] = None,
         line: Optional[int] = None, keep_next: bool = False,
         math: bool = False) -> str:
    """构造一个段落（pPr 子元素严格按 schema 顺序排列）。

    math=True 时，文本中以 $...$ 包裹的 LaTeX 片段会渲染为行内 Word 公式。
    """
    props: List[str] = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if keep_next:
        props.append("<w:keepNext/>")
    if numid is not None:
        props.append(f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{numid}"/></w:numPr>')
    if bottom_border:
        props.append('<w:pBdr><w:bottom w:val="single" w:color="auto" w:sz="4" '
                     'w:space="0"/></w:pBdr>')
    if space_before is not None or space_after is not None or line is not None:
        attrs = ""
        if space_before is not None:
            attrs += f' w:before="{space_before}"'
        if space_after is not None:
            attrs += f' w:after="{space_after}"'
        if line is not None:
            attrs += f' w:line="{line}" w:lineRule="auto"'
        props.append(f"<w:spacing{attrs}/>")
    if first_line is not None or numid is not None:
        indent = first_line if first_line is not None else 480
        props.append(f'<w:ind w:left="0" w:leftChars="0" w:firstLine="{indent}" '
                     f'w:firstLineChars="0"/>')
    if center:
        props.append('<w:jc w:val="center"/>')
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    if not text:
        body = ""
    elif math:
        body = rich_text(text, bold=bold, font=font, size=size)
    else:
        body = run(text, bold=bold, font=font, size=size)
    return f"<w:p>{ppr}{body}</w:p>"


def heading(text: str, level: int) -> str:
    """各级标题统一使用黑体 + Arial。"""
    style = {1: S_H1, 2: S_H2, 3: S_H3, 4: S_H4}[level]
    return para(text, style=style, font=FONT_HEAD)


def caption(text: str, keep_next: bool = False) -> str:
    """图题、表题：黑体、居中；表题置 keepNext 以避免与表格分页。"""
    return para(text, style=S_CAPTION, center=True, font=FONT_HEAD,
                keep_next=keep_next)


def image_para(rid: str, cx: int, cy: int, style: str = S_FIGURE) -> str:
    """插入一张居中、随文的图片。"""
    drawing = (
        '<w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="114300" distR="114300" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{abs(hash(rid)) % 9000 + 100}" name="图片"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:nvPicPr><pic:cNvPr id="0" name="图片"/><pic:cNvPicPr>'
        '<a:picLocks noChangeAspect="1"/></pic:cNvPicPr></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
        '<a:ln><a:noFill/></a:ln></pic:spPr></pic:pic></a:graphicData>'
        '</a:graphic></wp:inline></w:drawing></w:r>'
    )
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/><w:keepNext/>' \
           f'<w:jc w:val="center"/></w:pPr>{drawing}</w:p>'


def table(rows: Sequence[Sequence[str]], widths: Sequence[int],
          header: bool = True) -> str:
    """构造带边框的表格（表头黑体加粗居中）。"""
    total = sum(widths)
    borders = ("<w:tblBorders>"
               '<w:top w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
               '<w:left w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
               '<w:bottom w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
               '<w:right w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
               '<w:insideH w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
               '<w:insideV w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
               "</w:tblBorders>")
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblStyle w:val="15"/>'
           f'<w:tblW w:w="{total}" w:type="dxa"/>'
           f'<w:tblInd w:w="0" w:type="dxa"/>{borders}'
           '<w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:top w:w="30" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>'
           '<w:bottom w:w="30" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar>'
           f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
    for r, row in enumerate(rows):
        cells = []
        for c, text in enumerate(row):
            is_head = header and r == 0
            content = run(text, bold=is_head,
                          font=FONT_HEAD if is_head else FONT_BODY) if text else ""
            # 含换行的单元格（如目录树）使用单倍行距并左对齐，保证树形结构紧凑
            multiline = "\n" in text
            tight = '<w:spacing w:line="240" w:lineRule="auto"/>' if multiline else ""
            align = "left" if (multiline or (c == 0 and not is_head)) else "center"
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[c]}" w:type="dxa"/>'
                + ('<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>' if is_head else "")
                + '<w:vAlign w:val="center"/></w:tcPr>'
                '<w:p><w:pPr><w:ind w:left="0" w:firstLine="0"/>'
                f'{tight}<w:jc w:val="{align}"/></w:pPr>'
                f'{content}</w:p></w:tc>')
        out.append(f'<w:tr>{"".join(cells)}</w:tr>')
    out.append("</w:tbl>")
    return "".join(out)


def code_block(lines: Sequence[str]) -> str:
    """等宽代码块：单倍行距、零段间距，最后一行加下边框（与模板一致）。"""
    out = []
    last = len(lines) - 1
    for i, line in enumerate(lines):
        out.append(para(line, style=S_BODY, line=240,
                        space_before=0, space_after=0,
                        bottom_border=(i == last), font=FONT_MONO, size=20))
    return "".join(out)


# --------------------------------------------------------------------------
# LaTeX 公式 → Word 原生 OMML（可降级：缺少依赖时退化为普通文本）
# --------------------------------------------------------------------------
_XSLT = None
_XSLT_CHECKED = False
MATH_AVAILABLE = True


def _xslt():
    """惰性加载 MML2OMML 转换表；不可用时返回 None。"""
    global _XSLT, _XSLT_CHECKED, MATH_AVAILABLE
    if _XSLT_CHECKED:
        return _XSLT
    _XSLT_CHECKED = True
    if l2m is None:
        MATH_AVAILABLE = False
        print("（提示）未安装 latex2mathml，公式将以纯文本形式插入；"
              "如需 Word 原生公式请执行：pip install -r requirements-report.txt")
        return None
    xsl = find_mml2omml()
    if xsl is None:
        MATH_AVAILABLE = False
        print("（提示）未找到 Office 自带的 MML2OMML.XSL，公式将以纯文本形式插入。")
        return None
    _XSLT = etree.XSLT(etree.parse(str(xsl)))
    return _XSLT


def _latex_to_plain(latex: str) -> str:
    """降级方案：把 LaTeX 粗略转成可读的 Unicode 文本。"""
    text = latex
    for src, dst in ((r"\times", "×"), (r"\cdot", "·"), (r"\sum", "Σ"),
                     (r"\min", "min"), (r"\max", "max"), (r"\exp", "exp"),
                     (r"\ln", "ln"), (r"\pi", "π"), (r"\lambda", "λ"),
                     (r"\alpha", "α"), (r"\omega", "ω"), (r"\chi", "χ"),
                     (r"\sigma", "σ"), (r"\mu", "μ"), (r"\Sigma", "Σ"),
                     (r"\hat{L}", "L̂"), (r"\bar{d}", "d̄"), (r"\bar{x}", "x̄"),
                     (r"\mathbf", ""), (r"\mathrm", ""), (r"\left", ""),
                     (r"\right", ""), (r"\|", "‖"), (r"\;", " "), (r"\,", " "),
                     (r"\!", ""), (r"\quad", "  "), (r"\qquad", "    ")):
        text = text.replace(src, dst)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def omml(latex: str) -> str:
    """
    把 LaTeX 公式转换为 Word 原生公式（OMML）。

    若运行环境缺少 latex2mathml 或 Office 转换表，则退化为带样式的纯文本，
    保证报告生成流程不会中断。
    """
    xslt = _xslt()
    if xslt is None:
        return f'<w:r><w:rPr>{FONT_BODY}<w:i/></w:rPr>' \
               f'<w:t xml:space="preserve">{esc(_latex_to_plain(latex))}</w:t></w:r>'
    mathml = l2m.convert(latex)
    node = etree.fromstring(mathml.encode("utf-8"))
    return etree.tostring(xslt(node).getroot(), encoding="unicode")


_MATH_SPAN = re.compile(r"\$(.+?)\$")


def rich_text(text: str, bold: bool = False, font: str = FONT_BODY,
              size: Optional[int] = None) -> str:
    """
    构造"正文 + 行内公式"混合的文本内容。

    以 $...$ 包裹的 LaTeX 片段渲染为 Word 行内公式，其余部分为普通文本，
    从而让符号（如 z_i、\\hat{L}_{ij}）以数学字体呈现，符合学术排版惯例。
    """
    pieces: List[str] = []
    pos = 0
    for match in _MATH_SPAN.finditer(text):
        if match.start() > pos:
            pieces.append(run(text[pos:match.start()], bold=bold, font=font, size=size))
        pieces.append(omml(match.group(1)))
        pos = match.end()
    if pos < len(text):
        pieces.append(run(text[pos:], bold=bold, font=font, size=size))
    return "".join(pieces)


def equation(latex: str, number: Optional[int] = None) -> str:
    """居中显示公式，公式编号右对齐（使用居中式制表位）。"""
    tabs = (f'<w:tabs><w:tab w:val="center" w:pos="{CONTENT_TWIPS // 2}"/>'
            f'<w:tab w:val="right" w:pos="{CONTENT_TWIPS}"/></w:tabs>')
    tail = (f'<w:r><w:tab/></w:r><w:r><w:rPr>{FONT_BODY}</w:rPr>'
            f'<w:t>({number})</w:t></w:r>') if number else ""
    return (f'<w:p><w:pPr>{tabs}<w:spacing w:before="120" w:after="120"/>'
            '<w:jc w:val="left"/></w:pPr>'
            f'<w:r><w:tab/></w:r>{omml(latex)}{tail}</w:p>')


def picture_size(path: Path, width_in: float = CONTENT_WIDTH_IN) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        w, h = image.size
    cx = int(width_in * EMU_PER_INCH)
    return cx, int(cx * h / w)


FOOTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<w:p><w:pPr><w:jc w:val="center"/>'
    f'<w:rPr>{FONT_BODY}<w:sz w:val="18"/></w:rPr></w:pPr>'
    f'<w:r><w:rPr>{FONT_BODY}<w:sz w:val="18"/></w:rPr><w:t xml:space="preserve">- </w:t></w:r>'
    '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
    '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    f'<w:r><w:rPr>{FONT_BODY}<w:sz w:val="18"/></w:rPr><w:t>1</w:t></w:r>'
    '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    f'<w:r><w:rPr>{FONT_BODY}<w:sz w:val="18"/></w:rPr><w:t xml:space="preserve"> -</w:t></w:r>'
    "</w:p></w:ftr>"
)

FOOTER_RID = "rId6"
HYPERLINK_RID = "rId7"


def toc_field() -> str:
    """插入自动目录域（生成后由 Word 更新域填充页码）。"""
    return (
        '<w:p><w:pPr><w:pStyle w:val="12"/>'
        '<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="left"/></w:pPr>'
        '<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:rPr>{FONT_BODY}</w:rPr>'
        '<w:t>目录将在 Word 中打开时自动更新（Ctrl+A 后按 F9）。</w:t></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
    )


class ReportBuilder:
    """报告构造器：收集段落并维护图片关系。"""

    def __init__(self, media_dir: Path, hyperlink_url: str = ""):
        self.parts: List[str] = []
        self.media_dir = media_dir
        self.rels: List[Tuple[str, str]] = []
        self.hyperlink_url = hyperlink_url
        self._rid = 100

    def add(self, xml: str) -> None:
        self.parts.append(xml)

    def add_repo_link(self, url: str, text: str = "") -> None:
        """
        在标题下方插入仓库地址超链接：蓝色、带下划线、小字号。

        使用显式字符格式（模板未定义 Hyperlink 字符样式），
        并在 document.xml.rels 中登记 External 关系。
        """
        if not url:
            return
        label = text or url
        style = (f'{FONT_BODY}<w:color w:val="0563C1"/><w:u w:val="single"/>'
                 '<w:sz w:val="18"/><w:szCs w:val="18"/>')
        self.parts.append(
            '<w:p><w:pPr><w:spacing w:after="160"/>'
            '<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
            f'<w:hyperlink r:id="{HYPERLINK_RID}" w:history="1">'
            f'<w:r><w:rPr>{style}</w:rPr><w:t xml:space="preserve">{esc(label)}</w:t></w:r>'
            '</w:hyperlink></w:p>')

    def add_image(self, path: Path, width_in: float = CONTENT_WIDTH_IN) -> None:
        self._rid += 1
        rid = f"rId{self._rid}"
        shutil.copyfile(path, self.media_dir / path.name)
        self.rels.append((rid, f"media/{path.name}"))
        self.add(image_para(rid, *picture_size(path, width_in)))

    def add_code_images(self, manifest: List[Dict[str, object]], shot_dir: Path) -> None:
        """插入附录源码截图，并配以文件名与行号范围的题注。"""
        counter = 0
        for item in manifest:
            path = shot_dir / str(item["image"])
            if not path.exists():
                continue
            counter += 1
            self._rid += 1
            rid = f"rId{self._rid}"
            shutil.copyfile(path, self.media_dir / path.name)
            self.rels.append((rid, f"media/{path.name}"))
            self.add(image_para(rid, *picture_size(path, 5.25)))
            self.add(caption(
                f"代码 A-{counter}  {item['file']}（第 {item['start_line']}–{item['end_line']} 行）"))

    def document_xml(self) -> str:
        sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="{FOOTER_RID}"/>'
                '<w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" '
                'w:header="851" w:footer="992" w:gutter="0"/>'
                '<w:cols w:space="425" w:num="1"/>'
                '<w:docGrid w:type="lines" w:linePitch="312" w:charSpace="0"/></w:sectPr>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:document '
            'xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            'xmlns:o="urn:schemas-microsoft-com:office:office" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:v="urn:schemas-microsoft-com:vml" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
            'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/'
            'wordprocessingDrawing" '
            'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
            'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
            'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
            'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
            'mc:Ignorable="w14 wp14">'
            f'<w:body>{"".join(self.parts)}{sect}</w:body></w:document>'
        )

    def rels_xml(self) -> str:
        base = (
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/settings" Target="settings.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
            '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>'
            '<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
            f'<Relationship Id="{FOOTER_RID}" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
        )
        if self.hyperlink_url:
            base += (f'<Relationship Id="{HYPERLINK_RID}" Type="http://schemas.'
                     'openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
                     f'Target="{esc(self.hyperlink_url)}" TargetMode="External"/>')
        for rid, target in self.rels:
            base += (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                     f'officeDocument/2006/relationships/image" Target="{target}"/>')
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
                f'2006/relationships">{base}</Relationships>')


def template_parts() -> tuple:
    """需要从模板 docx 复用的排版资源清单。"""
    return TEMPLATE_PARTS


def ensure_template() -> Path:
    """
    确保模板排版资源可用。

    优先使用随工程分发的 assets/report_template/；
    若该目录不完整，则尝试从原始模板 docx 自动解包。
    """
    if (TEMPLATE_DIR / "word" / "styles.xml").exists():
        return TEMPLATE_DIR
    for candidate in TEMPLATE_DOCX_CANDIDATES:
        if candidate.exists():
            TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as zf:
                zf.extractall(TEMPLATE_DIR)
            return TEMPLATE_DIR
    raise FileNotFoundError(
        f"未找到报告模板排版资源：{TEMPLATE_DIR}\n"
        "请确认 assets/report_template/ 目录随工程一并下载。")


def patch_content_types(work: Path) -> None:
    """在 [Content_Types].xml 中登记页脚部件。"""
    path = work / "[Content_Types].xml"
    xml = path.read_text(encoding="utf-8")
    if "footer1.xml" in xml:
        return
    override = ('<Override PartName="/word/footer1.xml" ContentType="application/'
                'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')
    xml = xml.replace("</Types>", override + "</Types>")
    path.write_text(xml, encoding="utf-8")


def update_fields_with_word(path: Path) -> bool:
    """调用 Word 更新目录域与页码域，使交付的文档目录带有真实页码。"""
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return False
    try:
        word = win32com.client.Dispatch("Word.Application")
    except Exception:                     # noqa: BLE001
        return False
    try:
        word.Visible = False
        doc = word.Documents.Open(str(path))
        doc.Fields.Update()
        for i in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents(i).Update()
        doc.Repaginate()
        doc.Save()
        doc.Close(False)
        return True
    finally:
        word.Quit()
