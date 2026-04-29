from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING


@dataclass
class BBox:
    """
    统一坐标框表示，采用 PDF 页面坐标系：
    左上角 (x0, y0)，右下角 (x1, y1)
    """
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:  # 或 Tuple[float, float]
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

from typing import Literal
@dataclass
class TextBlock:
    """
    统一文本块结构：
    - 原生文本 / OCR 文本 / 融合文本 都统一为这个对象
    """
    text: str
    bbox: BBox
    page_num: int

    # 来源
    source: Literal["native", "ocr", "fused"] = "native" 

    # 结构标签（后续 layout_engine 可修改）
    block_type: Literal["title", "text", "table", "header", "footer", "page_num", "caption", "image"] = "text"

    # OCR / 融合可信度
    confidence: float = 1.0

    # 原生文本特征（native 路有用）
    font: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False

    # 排序 / 结构信息
    reading_order: Optional[int] = None
    section: Optional[str] = None

    # 额外信息
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "page_num": self.page_num,
            "source": self.source,
            "block_type": self.block_type,
            "confidence": self.confidence,
            "font": self.font,
            "font_size": self.font_size,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "reading_order": self.reading_order,
            "section": self.section,
            "meta": self.meta,
        }


@dataclass
class PageData:
    """
    单页数据容器
    """
    page_num: int
    width: float
    height: float

    if TYPE_CHECKING:
        import numpy as np

    # 页面图像（供 OCR）
    image: Optional["np.ndarray"] = None
    image_path: Optional[str] = None   # 可选保存路径

    # 双路原始结果
    native_blocks: List[TextBlock] = field(default_factory=list)
    ocr_blocks: List[TextBlock] = field(default_factory=list)

    # 中间结果 / 最终结果
    fused_blocks: List[TextBlock] = field(default_factory=list)
    cleaned_blocks: List[TextBlock] = field(default_factory=list)

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
            "native_blocks": [b.to_dict() for b in self.native_blocks],
            "ocr_blocks": [b.to_dict() for b in self.ocr_blocks],
            "fused_blocks": [b.to_dict() for b in self.fused_blocks],
            "cleaned_blocks": [b.to_dict() for b in self.cleaned_blocks],
            "meta": self.meta,
        }


@dataclass
class DocumentData:
    """
    整个 PDF 文档数据容器
    """
    file_path: str
    pages: List[PageData] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "pages": [p.to_dict() for p in self.pages],
            "meta": self.meta,
        }