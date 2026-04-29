# YourPDFLoader/parser/pdf_processor.py

import io
import os
from typing import List, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from ocr_base_loader.schema import BBox, TextBlock, PageData, DocumentData


class PDFProcessor:
    """
    PDF 双路原始数据生成器：
    1. 页面渲染为图片（供 OCR）
    2. 提取原生可复制文本块（供 native 路）
    """

    def __init__(
        self,
        dpi: int = 200,
        keep_image: bool = True,        #生成图片
        save_images: bool = False,      #保存图片至本地
        image_dir: Optional[str] = None,
        merge_spans_to_line: bool = True,
        strip_text: bool = True,
    ):
        """
        Args:
            dpi: 页面渲染分辨率
            keep_image: 是否将页面图像保存在内存中（np.ndarray）
            save_images: 是否将渲染后的页面图片保存到本地
            image_dir: 图片保存目录（当 save_images=True 时生效）
            merge_spans_to_line: 是否将 PyMuPDF 的 span 合并为 line 级 block
            strip_text: 是否对文本做 strip()
        """
        self.dpi = dpi
        self.keep_image = keep_image
        self.save_images = save_images
        self.image_dir = image_dir
        self.merge_spans_to_line = merge_spans_to_line
        self.strip_text = strip_text

    # =========================
    # 对外主入口
    # =========================
    def process(self, pdf_path: str) -> DocumentData:
        """
        处理整个 PDF，输出统一 DocumentData

        Args:
            pdf_path: PDF 文件路径

        Returns:
            DocumentData
        """
        doc = fitz.open(pdf_path)

        document = DocumentData(
            file_path=pdf_path,
            meta={
                "page_count": len(doc),
                "dpi": self.dpi,
            }
        )

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            width = float(page.rect.width)
            height = float(page.rect.height)

            image = None
            image_path = None

            # 路 A：页面渲染为图片
            if self.keep_image or self.save_images:
                image = self.render_page_to_image(page)

            if self.save_images:
                image_path = self.save_page_image(
                    image=image,
                    pdf_path=pdf_path,
                    page_num=page_num
                )

            # 路 B：提取原生文本块
            native_blocks = self.extract_native_blocks(
                page=page,
                page_num=page_num
            )

            page_data = PageData(
                page_num=page_num,
                width=width,
                height=height,
                image=image if self.keep_image else None,
                image_path=image_path,
                native_blocks=native_blocks,
                meta={
                    "rotation": page.rotation,
                }
            )

            document.pages.append(page_data)

        doc.close()
        return document

    # =========================
    # 页面渲染
    # =========================
    def render_page_to_image(self, page: fitz.Page) -> np.ndarray:
        """
        将 PDF 页面渲染为 RGB 图像（numpy array）

        Args:
            page: fitz.Page

        Returns:
            np.ndarray, shape=(H, W, 3), RGB
        """
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        # 直接从 samples 构造 numpy，跳过 PNG 编解码
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return image.copy()  # copy 保证内存独立

    def save_page_image(self, image: np.ndarray, pdf_path: str, page_num: int) -> str:
        """
        保存页面图像到本地

        Args:
            image: 页面图像（RGB）
            pdf_path: PDF 路径
            page_num: 页码（1-based）

        Returns:
            图片保存路径
        """
        if self.image_dir is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            self.image_dir = os.path.join(os.path.dirname(pdf_path), f"{base_name}_images")

        os.makedirs(self.image_dir, exist_ok=True)

        image_name = f"page_{page_num:04d}.png"
        image_path = os.path.join(self.image_dir, image_name)

        Image.fromarray(image).save(image_path)
        return image_path

    # =========================
    # 原生文本提取
    # =========================
    def extract_native_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """
        提取页面原生文本块，统一输出为 TextBlock 列表

        优先使用 page.get_text("dict")：
        - 保留 block / line / span 层级信息
        - 可获取 bbox / font / size / flags

        Args:
            page: fitz.Page
            page_num: 页码（1-based）

        Returns:
            List[TextBlock]
        """
        text_dict = page.get_text("dict")
        blocks = []

        for block_idx, block in enumerate(text_dict.get("blocks", [])):
            # type == 0 表示文本块；其他类型可能是图片等
            if block.get("type", 0) != 0:
                continue

            if self.merge_spans_to_line:
                blocks.extend(
                    self._extract_line_level_blocks(
                        block=block,
                        page_num=page_num,
                        block_idx=block_idx
                    )
                )
            else:
                blocks.extend(
                    self._extract_span_level_blocks(
                        block=block,
                        page_num=page_num,
                        block_idx=block_idx
                    )
                )

        return blocks

    def _extract_line_level_blocks(
        self,
        block: dict,
        page_num: int,
        block_idx: int
    ) -> List[TextBlock]:
        """
        将 PyMuPDF 的 span 合并为 line 级 TextBlock

        优点：
        - 更适合后续 OCR/native 融合
        - 粒度较稳定
        - 更适合 chunk 前结构化处理
        """
        results = []

        for line_idx, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            if not spans:
                continue

            texts = []
            x0_list, y0_list, x1_list, y1_list = [], [], [], []

            font_sizes = []
            fonts = []
            bold_flags = []
            italic_flags = []

            for span_idx, span in enumerate(spans):
                text = span.get("text", "")
                if self.strip_text:
                    text = text.strip()

                if not text:
                    continue

                texts.append(text)

                sx0, sy0, sx1, sy1 = span.get("bbox", [0, 0, 0, 0])
                x0_list.append(float(sx0))
                y0_list.append(float(sy0))
                x1_list.append(float(sx1))
                y1_list.append(float(sy1))

                font_sizes.append(float(span.get("size", 0)))
                fonts.append(span.get("font", ""))

                flags = int(span.get("flags", 0))
                bold_flags.append(self._is_bold(flags, span.get("font", "")))
                italic_flags.append(self._is_italic(flags, span.get("font", "")))

            if not texts:
                continue

            merged_text = self._merge_line_text(texts)

            bbox = BBox(
                x0=min(x0_list),
                y0=min(y0_list),
                x1=max(x1_list),
                y1=max(y1_list),
            )

            main_font = self._most_common(fonts)
            avg_font_size = float(np.mean(font_sizes)) if font_sizes else None
            is_bold = any(bold_flags)
            is_italic = any(italic_flags)

            block_obj = TextBlock(
                text=merged_text,
                bbox=bbox,
                page_num=page_num,
                source="native",
                block_type="text",
                confidence=1.0,
                font=main_font,
                font_size=avg_font_size,
                is_bold=is_bold,
                is_italic=is_italic,
                meta={
                    "block_idx": block_idx,
                    "line_idx": line_idx,
                    "span_count": len(spans),
                    "extract_level": "line",
                }
            )

            results.append(block_obj)

        return results

    def _extract_span_level_blocks(
        self,
        block: dict,
        page_num: int,
        block_idx: int
    ) -> List[TextBlock]:
        """
        直接输出 span 级 TextBlock
        """
        results = []

        for line_idx, line in enumerate(block.get("lines", [])):
            for span_idx, span in enumerate(line.get("spans", [])):
                text = span.get("text", "")
                if self.strip_text:
                    text = text.strip()

                if not text:
                    continue

                x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])

                font = span.get("font", "")
                font_size = float(span.get("size", 0))
                flags = int(span.get("flags", 0))

                block_obj = TextBlock(
                    text=text,
                    bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                    page_num=page_num,
                    source="native",
                    block_type="text",
                    confidence=1.0,
                    font=font,
                    font_size=font_size,
                    is_bold=self._is_bold(flags, font),
                    is_italic=self._is_italic(flags, font),
                    meta={
                        "block_idx": block_idx,
                        "line_idx": line_idx,
                        "span_idx": span_idx,
                        "extract_level": "span",
                    }
                )

                results.append(block_obj)

        return results

    # =========================
    # 工具函数
    # =========================
    @staticmethod
    @staticmethod
    def _merge_line_text(texts: List[str]) -> str:
        if not texts:
            return ""
        result = []
        for i, t in enumerate(texts):
            if i == 0:
                result.append(t)
            else:
                prev = result[-1]
            # 中文字符之间不加空格
                if prev and t and '\u4e00' <= prev[-1] <= '\u9fff' and '\u4e00' <= t[0] <= '\u9fff':
                    result.append(t)
                else:
                    result.append(" " + t)
        return "".join(result).strip()

    @staticmethod
    def _most_common(items: List[str]) -> Optional[str]:
        """
        返回列表中出现频率最高的元素
        """
        if not items:
            return None
        return max(set(items), key=items.count)

    @staticmethod
    def _is_bold(flags: int, font_name: str = "") -> bool:
        """
        粗略判断是否加粗
        """
        font_name = (font_name or "").lower()
        return ("bold" in font_name) or bool(flags & 2**4)

    @staticmethod
    def _is_italic(flags: int, font_name: str = "") -> bool:
        """
        粗略判断是否斜体
        """
        font_name = (font_name or "").lower()
        return ("italic" in font_name) or ("oblique" in font_name) or bool(flags & 2**1)
'''    
if __name__ == "__main__":
    import traceback

    # 【请换成你的PDF路径】
    test_pdf_path = r"D:\Aprojet\py\RAG_Agent\financial_rag_agent\dataset\raw_pdf\半导体\H2_AN202502281643614026_1.pdf"

    try:
        print("=" * 50)
        print("开始测试 PDFProcessor ...")
        print(f"测试文件：{test_pdf_path}")

        # 初始化（关键：keep_image=True 必须开）
        processor = PDFProcessor(
            dpi=200,
            keep_image=True,    # 必须开启，才会生成图片
            save_images=False,  # 不保存本地，避免干扰
            merge_spans_to_line=True
        )

        # 处理PDF
        doc_data = processor.process(pdf_path=test_pdf_path)

        print(f"总页数：{len(doc_data.pages)}")
        if len(doc_data.pages) == 0:
            print("❌ 错误：PDF 没有任何页面！")
            exit(1)

        # 取第一页
        page = doc_data.pages[0]
        print(f"第1页尺寸：{page.width} x {page.height}")

        # 查看图片
        img = page.image
        print(f"图片是否为 None：{img is None}")

        if img is not None:
            print(f"图片形状 (H, W, C)：{img.shape}")
            print(f"图片数据类型：{img.dtype}")

            # 关键判断：是否是空尺寸图片（导致OpenCV报错）
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                print("❌ 错误：图片宽/高为 0！")
            else:
                print("✅ 图片生成成功！完全正常！")

        print("=" * 50)

    except Exception as e:
        print("❌ 测试失败：")
        print(e)
        traceback.print_exc()
'''