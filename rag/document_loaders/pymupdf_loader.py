# -*- coding: utf-8 -*-
"""
基础版 PDF 解析器
使用 PyMuPDF (fitz) 提取文本 + 页码 + 简单排版
输出结构化 Markdown 格式，为后续 Chunk 做准备
缺陷：表格，图片，双列分布等无法处理。
PDF 不是「按行 / 按表格存内容」，而是按「绘制顺序」存一个个独立的文本块（Text Block）
比如双列，pdf先处理左边一列,即按文本块的 y 坐标排序，y 相同再按 x 坐标排序（x=n,y=0）
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # 添加项目根目录到路径
from configs import rag_config



class PyMuPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.doc = fitz.open(file_path)

    def sort_blocks_by_row(self, blocks: List) -> List[str]:
        """
        核心修复：按行分组+按列排序，解决表格顺序错乱
        逻辑：
        1. 按y坐标阈值分组（同一行的文本块y坐标接近）
        2. 每组内按x坐标从左到右排序（左列→右列）
        3. 合并成按行对齐的文本
        """
        # 1. 提取有效文本块（过滤空块）
        text_blocks = []
        for blk in blocks:
            if blk[4].strip():
                # 记录：(y坐标中点, x坐标起点, 文本内容)
                y_mid = (blk[1] + blk[3]) / 2  # 用块的垂直中点代表行位置
                x_start = blk[0]
                text = blk[4].strip()
                text_blocks.append((y_mid, x_start, text))

        # 2. 按y坐标分组（同一行的块y差小于阈值，阈值可根据PDF调整）
        row_threshold = 5  # 单位：点，同一行的y差不超过5个点
        rows = []
        current_row = []
        # 先按y坐标升序排序（从上到下）
        text_blocks.sort(key=lambda x: x[0])

        for blk in text_blocks:
            if not current_row:
                current_row.append(blk)
            else:
                # 比较当前块和当前行第一个块的y差
                if abs(blk[0] - current_row[0][0]) < row_threshold:
                    current_row.append(blk)
                else:
                    # 新行，保存当前行
                    rows.append(current_row)
                    current_row = [blk]
        # 保存最后一行
        if current_row:
            rows.append(current_row)

        # 3. 每行内按x坐标升序排序（从左到右），合并文本
        sorted_lines = []
        for row in rows:
            # 按x坐标排序（左列→右列）
            row.sort(key=lambda x: x[1])
            # 合并同一行的文本，用制表符分隔（方便转Markdown表格）
            line_text = "\t".join([blk[2] for blk in row])
            sorted_lines.append(line_text)

        return sorted_lines

    def extract_page_markdown(self, page_idx: int) -> str:
        """
        修复版：提取单页内容，表格按行对齐，输出Markdown
        """
        page = self.doc[page_idx]
        blocks = page.get_text("blocks")  # 还是用blocks，但用自定义排序

        # 用修复后的排序逻辑
        sorted_lines = self.sort_blocks_by_row(blocks)

        # 识别表格（两列/多列，包含"指"的行），转Markdown表格
        md_lines = []
        in_table = False
        table_rows = []

        for line in sorted_lines:
            # 检测表格行：包含"指"，且有制表符（两列分隔）
            if "\t" in line and "指" in line:
                if not in_table:
                    in_table = True
                    # 表格表头（如果是释义表，自动加表头）
                    table_rows.append("| 术语 | 释义 |")
                    table_rows.append("|------|------|")
                # 拆分两列，转Markdown表格行
                cols = line.split("\t", 1)  # 只按第一个制表符拆分，避免释义内的制表符
                term = cols[0].strip()
                meaning = cols[1].strip()
                table_rows.append(f"| {term} | {meaning} |")
            else:
                # 非表格行，直接作为普通段落
                if in_table:
                    # 表格结束，先把表格写入
                    md_lines.extend(table_rows)
                    md_lines.append("")
                    in_table = False
                    table_rows = []
                md_lines.append(line)

        # 处理最后可能剩余的表格
        if in_table and table_rows:
            md_lines.extend(table_rows)

        # 合并成最终Markdown
        md_content = "\n\n".join(md_lines)
        return md_content

    def load_all_pages(self) -> List[Dict]:
        """
        读取全部页面，返回带元数据的结构化列表
        """
        pages_content = []

        for page_idx in range(len(self.doc)):
            md_text = self.extract_page_markdown(page_idx)
            if not md_text.strip():
                continue

            meta = {
                "source": self.file_name,
                "page_num": page_idx + 1,  # 页码从1开始
                "file_path": self.file_path,
                "parse_engine": "pymupdf_fixed"
            }

            pages_content.append({
                "content": md_text,
                "metadata": meta
            })

        self.doc.close()
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
            print(docs[9]["content"])  # 打印第5页看看表格效果
        else:
            breakpoint()
    else:

        print(f"测试文件不存在：{test_pdf}")
