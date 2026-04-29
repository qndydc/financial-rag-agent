# -*- coding: utf-8 -*-
"""
增强版 PDF 解析器
- 普通文本：PyMuPDF
- 表格：pdfplumber
- 支持复杂双栏/多栏、表格提取、页眉页脚过滤、正文与表格去重
- 保留原有输入输出格式
"""

import os
import re
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))  # 添加项目根目录到路径
from configs import rag_config


class PyMuPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.doc = fitz.open(file_path)
        self.plumber_doc = pdfplumber.open(file_path)
        self.file_stem = Path(file_path).stem
        self.doc_id = f"{self.file_stem}_{len(self.doc)}pages"
        
    # =========================
    # 基础工具
    # =========================
    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _normalize_cell(cell) -> str:
        if cell is None:
            return ""
        text = str(cell).replace("\n", "<br>")
        text = re.sub(r"[ \t]+", " ", text).strip()
        # Markdown 表格转义
        text = text.replace("|", r"\|")
        return text

    @staticmethod
    def _bbox_intersects(b1: Tuple[float, float, float, float],
                         b2: Tuple[float, float, float, float],
                         tolerance: float = 2.0) -> bool:
        """
        判断两个 bbox 是否相交（带少量容忍度）
        bbox: (x0, y0, x1, y1)
        """
        ax0, ay0, ax1, ay1 = b1
        bx0, by0, bx1, by1 = b2

        ax0 -= tolerance
        ay0 -= tolerance
        ax1 += tolerance
        ay1 += tolerance

        bx0 -= tolerance
        by0 -= tolerance
        bx1 += tolerance
        by1 += tolerance

        horizontal = not (ax1 < bx0 or bx1 < ax0)
        vertical = not (ay1 < by0 or by1 < ay0)
        return horizontal and vertical

    @staticmethod
    def _is_probably_page_number(text: str) -> bool:
        text = text.strip()
        if re.fullmatch(r"\d{1,3}", text):
            return True
        if re.fullmatch(r"-\s*\d{1,3}\s*-", text):
            return True
        if re.fullmatch(r"第\s*\d+\s*页", text):
            return True
        return False

    @staticmethod
    def _is_header_or_footer(block: dict, page_height: float) -> bool:
        """
        简单页眉页脚过滤
        """
        x0, y0, x1, y1 = block["bbox"]
        text = block["text"].strip()

        if not text:
            return True

        # 顶部/底部边缘区域
        top_margin = page_height * 0.06
        bottom_margin = page_height * 0.94

        if y1 <= top_margin:
            return True
        if y0 >= bottom_margin:
            return True

        if PyMuPDFLoader._is_probably_page_number(text):
            return True

        return False

    # =========================
    # 文档级 metadata 提取
    # =========================
    def _safe_extract_plain_text_from_page(self, page_idx: int) -> str:
        try:
            page = self.doc[page_idx]
            text = page.get_text("text")
            return self._clean_text(text)
        except Exception:
            return ""

    def _get_front_pages_text(self, max_pages: int = 3) -> str:
        texts = []
        total_pages = min(max_pages, len(self.doc))
        for i in range(total_pages):
            txt = self._safe_extract_plain_text_from_page(i)
            if txt:
                texts.append(txt)
        return "\n".join(texts).strip()

    def _extract_broker_from_filename(self) -> Optional[str]:
        """
        适配常见命名：
        中信证券_银行行业深度报告_2024.pdf
        招商证券-计算机行业专题-20240510.pdf
        """
        stem = self.file_stem

        # 常见分隔符优先
        parts = re.split(r"[_\-—]+", stem)
        if parts:
            candidate = parts[0].strip()
            if 2 <= len(candidate) <= 20 and any(k in candidate for k in ["证券", "研究所", "投顾", "基金", "资管"]):
                return candidate

        # 兜底：直接匹配“xx证券”
        m = re.search(r"([\u4e00-\u9fa5A-Za-z]{2,20}证券)", stem)
        if m:
            return m.group(1)

        return None

    def _extract_report_date(self, text: str) -> Optional[str]:
        """
        输出尽量统一成 YYYY-MM-DD
        """
        if not text:
            return None

        patterns = [
            r"(20\d{2})[-/\.年](\d{1,2})[-/\.月](\d{1,2})日?",
            r"(20\d{2})(\d{2})(\d{2})",
        ]

        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                y, mo, d = m.group(1), m.group(2), m.group(3)
                try:
                    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
                except Exception:
                    continue
        return None

    def _extract_title(self, text: str) -> Optional[str]:
        """
        从前几页中提取一个较像标题的文本：
        - 优先取较靠前、较长但不过长的一行
        - 排除日期、页码、纯机构名、风险提示等
        """
        if not text:
            return None

        lines = [self._clean_text(x) for x in text.splitlines()]
        lines = [x for x in lines if x]

        bad_keywords = [
            "请务必阅读正文之后的免责条款部分",
            "风险提示",
            "目录",
            "证券研究报告",
            "行业评级",
            "投资评级",
        ]

        candidates = []
        for line in lines[:80]:
            if len(line) < 6 or len(line) > 40:
                continue
            if self._is_probably_page_number(line):
                continue
            if self._extract_report_date(line):
                continue
            if any(k in line for k in bad_keywords):
                continue
            # 排除纯机构名
            if re.fullmatch(r"[\u4e00-\u9fa5A-Za-z]{2,20}(证券|研究所|基金|资管)", line):
                continue

            candidates.append(line)

        if not candidates:
            return None

        # 优先包含“行业/深度/专题/点评/研究/报告”等标题词
        priority_keywords = ["深度", "专题", "点评", "研究", "报告", "行业", "公司", "策略", "跟踪"]
        for c in candidates:
            if any(k in c for k in priority_keywords):
                return c

        return candidates[0]

    def _extract_company(self, text: str, title: Optional[str] = None) -> Optional[str]:
        """
        轻量启发式：
        - 优先从标题里抓“公司名（股份/集团/银行/科技等）”
        - 再从前几页文本中抓
        """
        source_texts = []
        if title:
            source_texts.append(title)
        if text:
            source_texts.append(text[:3000])

        patterns = [
            r"([\u4e00-\u9fa5A-Za-z]{2,20}(银行|证券|集团|股份|科技|信息|药业|电子|能源|通信|汽车|工业|软件|智能|航空|半导体))",
            r"([\u4e00-\u9fa5]{2,12})（\d{6}(?:\.SH|\.SZ)?）",
            r"([\u4e00-\u9fa5]{2,12})\(\d{6}(?:\.SH|\.SZ)?\)",
        ]

        for src in source_texts:
            for pattern in patterns:
                m = re.search(pattern, src)
                if m:
                    return m.group(1)

        return None

    def _extract_document_level_metadata(self) -> Dict:
        """
        文档级别 metadata，只提一次，然后复用于每一页。
        """
        front_text = self._get_front_pages_text(max_pages=3)

        broker = self._extract_broker_from_filename()
        title = self._extract_title(front_text)
        report_date = self._extract_report_date(front_text) or self._extract_report_date(self.file_stem)
        company = self._extract_company(front_text, title=title)

        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "file_stem": self.file_stem,
            "file_path": self.file_path,
            "source": self.file_name,
            "parse_engine": "pymupdf_pdfplumber_enhanced",
            "broker": broker,
            "title": title,
            "report_date": report_date,
            "company": company,
        }
    
    # =========================
    # 表格处理（pdfplumber）
    # =========================
    def _extract_tables_with_bboxes(self, page_idx: int) -> List[Dict]:
        """
        提取当前页表格：
        返回:
        [
            {
                "bbox": (x0, y0, x1, y1),
                "markdown": "..."
            },
            ...
        ]
        注意：pdfplumber bbox 是 (x0, top, x1, bottom)，与 PyMuPDF 坐标系兼容（左上原点）
        """
        page = self.plumber_doc.pages[page_idx]
        tables_data = []

        # 尽量显式给出 table_settings，适应研报常见表格
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
        }

        try:
            tables = page.find_tables(table_settings=table_settings)
        except Exception:
            tables = []

        for tb in tables:
            try:
                raw_table = tb.extract()
            except Exception:
                raw_table = None

            if not raw_table:
                continue

            # 清理空行空列
            cleaned_rows = []
            max_cols = 0
            for row in raw_table:
                if row is None:
                    continue
                normalized = [self._normalize_cell(c) for c in row]
                if any(cell.strip() for cell in normalized):
                    cleaned_rows.append(normalized)
                    max_cols = max(max_cols, len(normalized))

            if not cleaned_rows or max_cols == 0:
                continue

            # 对齐列数
            aligned_rows = []
            for row in cleaned_rows:
                row = row + [""] * (max_cols - len(row))
                aligned_rows.append(row)

            md = self._table_to_markdown(aligned_rows)
            if not md.strip():
                continue

            bbox = tb.bbox  # (x0, top, x1, bottom)
            tables_data.append({
                "bbox": bbox,
                "markdown": md
            })

        return tables_data

    def _table_to_markdown(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""

        # 处理全空
        rows = [[self._normalize_cell(c) for c in row] for row in rows]
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            return ""

        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]

        # 默认第一行作为表头；若第一行明显不是表头，也保留这种简单策略，避免过拟合
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []

        md_lines = []
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * n_cols) + " |")

        for row in body:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    # =========================
    # 文本块处理（PyMuPDF）
    # =========================
    def _extract_text_blocks(self, page_idx: int,
                             table_bboxes: Optional[List[Tuple[float, float, float, float]]] = None) -> List[Dict]:
        """
        用 PyMuPDF 提取文本块，并排除表格区域、页眉页脚
        返回:
        [
            {
                "bbox": (x0,y0,x1,y1),
                "text": "...",
            }
        ]
        """
        page = self.doc[page_idx]
        page_dict = page.get_text("dict")
        page_height = page.rect.height

        blocks = []
        table_bboxes = table_bboxes or []

        for blk in page_dict.get("blocks", []):
            if blk.get("type") != 0:  # 只处理文本块
                continue

            bbox = tuple(blk.get("bbox", (0, 0, 0, 0)))
            lines = []
            for line in blk.get("lines", []):
                spans = []
                for sp in line.get("spans", []):
                    txt = sp.get("text", "")
                    if txt:
                        spans.append(txt)
                line_text = "".join(spans).strip()
                if line_text:
                    lines.append(line_text)

            text = "\n".join(lines).strip()
            text = self._clean_text(text)

            block_item = {
                "bbox": bbox,
                "text": text
            }

            if not text:
                continue

            if self._is_header_or_footer(block_item, page_height):
                continue

            # 排除与表格重叠的正文块
            overlapped = False
            for tb in table_bboxes:
                if self._bbox_intersects(bbox, tb, tolerance=3.0):
                    overlapped = True
                    break
            if overlapped:
                continue

            blocks.append(block_item)

        return blocks

    def _detect_columns(self, blocks: List[Dict], page_width: float) -> List[List[Dict]]:
        """
        简单多栏检测：
        - 统计 block 中心点 x 分布
        - 若左右半区都有较多块，则视为双栏
        - 否则单栏
        注：这是启发式，不做过拟合
        """
        if not blocks:
            return []

        mid_x = page_width / 2
        left_blocks = []
        right_blocks = []
        cross_blocks = []

        for blk in blocks:
            x0, y0, x1, y1 = blk["bbox"]
            center_x = (x0 + x1) / 2
            width = x1 - x0

            # 跨中线的大块，通常是标题/整页宽表述
            if x0 < mid_x < x1 or width > page_width * 0.7:
                cross_blocks.append(blk)
            elif center_x <= mid_x:
                left_blocks.append(blk)
            else:
                right_blocks.append(blk)

        # 双栏判断条件：左右都有一定数量块
        if len(left_blocks) >= 3 and len(right_blocks) >= 3:
            # 跨栏块作为单独“前置流”，再接左栏、右栏
            # 但阅读顺序中，跨栏标题可能出现在上方，因此最终还要按整体 y 再融合
            return [cross_blocks, left_blocks, right_blocks]

        return [blocks]

    def sort_blocks_by_row(self, blocks: List) -> List[str]:
        """
        保留原方法名，但内部升级为：
        - 输入兼容 PyMuPDF blocks(tuple) 或自定义 dict blocks
        - 输出为按阅读顺序组织的文本行列表
        """
        normalized_blocks = []

        # 兼容旧版 tuple blocks: (x0, y0, x1, y1, text, block_no, block_type, ...)
        for blk in blocks:
            if isinstance(blk, dict):
                text = self._clean_text(blk.get("text", ""))
                bbox = tuple(blk.get("bbox", (0, 0, 0, 0)))
            else:
                if len(blk) < 5:
                    continue
                text = self._clean_text(blk[4])
                bbox = (blk[0], blk[1], blk[2], blk[3])

            if not text:
                continue

            normalized_blocks.append({
                "bbox": bbox,
                "text": text
            })

        if not normalized_blocks:
            return []

        # 这里无法直接知道 page_width，因此用 x 最大值估算
        page_width = max(b["bbox"][2] for b in normalized_blocks) + 1
        column_groups = self._detect_columns(normalized_blocks, page_width)

        ordered_blocks = []
        if len(column_groups) == 1:
            ordered_blocks = sorted(
                column_groups[0],
                key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1))
            )
        else:
            # cross / left / right 各自按 y,x 排
            cross_blocks = sorted(column_groups[0], key=lambda b: (b["bbox"][1], b["bbox"][0]))
            left_blocks = sorted(column_groups[1], key=lambda b: (b["bbox"][1], b["bbox"][0]))
            right_blocks = sorted(column_groups[2], key=lambda b: (b["bbox"][1], b["bbox"][0]))

            # 为避免跨栏标题被错位，先按顶部跨栏块，再左栏，再右栏
            ordered_blocks = cross_blocks + left_blocks + right_blocks

        # 段落合并：相近块拼接成自然段
        merged_lines = []
        prev_bbox = None
        prev_text = None

        for blk in ordered_blocks:
            text = blk["text"]
            bbox = blk["bbox"]

            if prev_text is None:
                prev_text = text
                prev_bbox = bbox
                continue

            px0, py0, px1, py1 = prev_bbox
            x0, y0, x1, y1 = bbox

            vertical_gap = y0 - py1
            same_column = abs(x0 - px0) < 40

            # 若同列且垂直间距较小，拼成一个段落
            if same_column and vertical_gap <= 12:
                # 避免硬拼标题
                if prev_text.endswith(("：", ":", "；", ";")):
                    prev_text = prev_text + "\n" + text
                else:
                    prev_text = prev_text + "\n" + text
                prev_bbox = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
            else:
                merged_lines.append(prev_text)
                prev_text = text
                prev_bbox = bbox

        if prev_text:
            merged_lines.append(prev_text)

        return merged_lines

    # =========================
    # 页面内容融合
    # =========================
    def _merge_page_elements(self,
                             text_blocks: List[Dict],
                             tables: List[Dict]) -> List[Dict]:
        """
        把正文块和表格按页面 y 坐标融合
        输出:
        [
            {"type": "text", "y": ..., "content": "..."},
            {"type": "table", "y": ..., "content": "..."},
        ]
        """
        elements = []

        # 文本块先做排序与段落化
        text_lines = self.sort_blocks_by_row(text_blocks)

        # 这里为了让文本与表格整体按页面位置融合，需要重新用原 blocks 的顶部 y
        # 简化策略：按 block 顺序对应文本行不够可靠，因此改为直接按 text_blocks 形成段落元素
        # 所以这里重新做一个更轻量的按列后块顺序排列
        if text_blocks:
            page_width = max(b["bbox"][2] for b in text_blocks) + 1
            column_groups = self._detect_columns(text_blocks, page_width)

            ordered_text_blocks = []
            if len(column_groups) == 1:
                ordered_text_blocks = sorted(column_groups[0], key=lambda b: (b["bbox"][1], b["bbox"][0]))
            else:
                cross_blocks = sorted(column_groups[0], key=lambda b: (b["bbox"][1], b["bbox"][0]))
                left_blocks = sorted(column_groups[1], key=lambda b: (b["bbox"][1], b["bbox"][0]))
                right_blocks = sorted(column_groups[2], key=lambda b: (b["bbox"][1], b["bbox"][0]))
                ordered_text_blocks = cross_blocks + left_blocks + right_blocks

            merged_blocks = []
            prev = None
            for blk in ordered_text_blocks:
                if prev is None:
                    prev = dict(blk)
                    continue

                px0, py0, px1, py1 = prev["bbox"]
                x0, y0, x1, y1 = blk["bbox"]
                vertical_gap = y0 - py1
                same_column = abs(x0 - px0) < 40

                if same_column and vertical_gap <= 12:
                    prev["text"] = prev["text"] + "\n" + blk["text"]
                    prev["bbox"] = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
                else:
                    merged_blocks.append(prev)
                    prev = dict(blk)

            if prev is not None:
                merged_blocks.append(prev)

            for blk in merged_blocks:
                elements.append({
                    "type": "text",
                    "y": blk["bbox"][1],
                    "content": blk["text"]
                })

        for tb in tables:
            x0, y0, x1, y1 = tb["bbox"]
            elements.append({
                "type": "table",
                "y": y0,
                "content": tb["markdown"]
            })

        elements.sort(key=lambda x: x["y"])
        return elements

    def extract_page_markdown(self, page_idx: int) -> str:
        """
        提取单页内容，输出 Markdown
        """
        # 1. 提表格
        tables = self._extract_tables_with_bboxes(page_idx)
        table_bboxes = [tb["bbox"] for tb in tables]

        # 2. 提正文（排除表格区域）
        text_blocks = self._extract_text_blocks(page_idx, table_bboxes=table_bboxes)

        # 3. 融合页面元素
        elements = self._merge_page_elements(text_blocks, tables)

        # 4. 输出 markdown
        md_lines = []
        prev_type = None

        for elem in elements:
            content = self._clean_text(elem["content"])
            if not content:
                continue

            if elem["type"] == "table":
                # 表格前后加空行
                if md_lines and md_lines[-1] != "":
                    md_lines.append("")
                md_lines.append(content)
                md_lines.append("")
            else:
                # 普通文本
                if prev_type == "text" and md_lines and md_lines[-1] != "":
                    md_lines.append("")
                md_lines.append(content)

            prev_type = elem["type"]

        md_content = "\n".join(md_lines).strip()
        return md_content

    def load_all_pages(self) -> List[Dict]:
        """
        读取全部页面，返回带元数据的结构化列表
        """
        pages_content = []
        doc_level_meta = self._extract_document_level_metadata()

        for page_idx in range(len(self.doc)):
            md_text = self.extract_page_markdown(page_idx)

            if not md_text.strip():
                continue

            meta = {
                **doc_level_meta,
                "page_num": page_idx + 1,  # 页码从1开始

            }

            pages_content.append({
                "content": md_text,
                "metadata": meta
            })

        self.doc.close()
        self.plumber_doc.close()
        return pages_content


# 快速测试
if __name__ == "__main__":
    # 随便放一个测试 PDF
    test_pdf = os.path.join(rag_config.RAW_PDF_DIR, r"H2_AN202502281643614026_1.pdf")
    if os.path.exists(test_pdf):
        loader = PyMuPDFLoader(test_pdf)
        docs = loader.load_all_pages()

        # 打印第一页看看效果
        if docs:
            print("=" * 50)
            print(f"文件名: {docs[0]['metadata']['source']}")
            print(f"页码: {docs[0]['metadata']['page_num']}")
            print("=" * 50)
            print(docs[0]["content"])
            if len(docs) > 9:
                print(docs[8]["content"])  # 保留你的输出习惯
                print(docs[8]["metadata"])
        else:
            breakpoint()
    else:
        print(f"测试文件不存在：{test_pdf}")
