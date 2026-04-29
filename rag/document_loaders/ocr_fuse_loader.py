"""
Financial Report PDF Loader
支持双路输入（原生文本+OCR）、版面分析、智能融合
适配RAG流程的专业PDF读取器
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile

try:
    import pdfplumber
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install pdfplumber PyMuPDF Pillow pytesseract numpy --break-system-packages")


@dataclass
class TextBlock:
    """文本块数据结构"""
    text: str
    page: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    block_type: str  # 'text', 'table', 'title', 'image_caption', 'footer', 'header'
    confidence: float = 1.0  # OCR置信度
    source: str = 'native'  # 'native' or 'ocr'
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def area(self) -> float:
        """计算bbox面积"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def iou(self, other: 'TextBlock') -> float:
        """计算与另一个block的IOU（交并比）"""
        if self.page != other.page:
            return 0.0
        
        x0 = max(self.bbox[0], other.bbox[0])
        y0 = max(self.bbox[1], other.bbox[1])
        x1 = min(self.bbox[2], other.bbox[2])
        y1 = min(self.bbox[3], other.bbox[3])
        
        if x1 < x0 or y1 < y0:
            return 0.0
        
        intersection = (x1 - x0) * (y1 - y0)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0


class FinancialPDFLoader:
    """金融研报PDF加载器"""
    
    def __init__(
        self,
        pdf_path: str,
        use_ocr: bool = True,
        ocr_language: str = 'chi_sim+eng',
        dpi: int = 300,
        enable_table_detection: bool = True,
        enable_layout_analysis: bool = True
    ):
        """
        初始化PDF加载器
        
        Args:
            pdf_path: PDF文件路径
            use_ocr: 是否启用OCR（双路模式）
            ocr_language: OCR语言，默认中英文
            dpi: 图像渲染DPI，影响OCR质量
            enable_table_detection: 是否检测表格
            enable_layout_analysis: 是否进行版面分析
        """
        self.pdf_path = pdf_path
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.dpi = dpi
        self.enable_table_detection = enable_table_detection
        self.enable_layout_analysis = enable_layout_analysis
        
        # 验证文件存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def extract_native_text(self) -> List[TextBlock]:
        """
        提取原生PDF文本（保留位置信息）
        使用pdfplumber进行版面感知提取
        """
        blocks = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # 提取文本块（保留bbox）
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False
                )
                
                if not words:
                    continue
                
                # 按行聚合文字
                lines = self._group_words_into_lines(words)
                
                for line in lines:
                    text = line['text'].strip()
                    if not text:
                        continue
                    
                    bbox = (line['x0'], line['top'], line['x1'], line['bottom'])
                    
                    # 简单的块类型判断
                    block_type = self._classify_block_type(text, bbox, page.height)
                    
                    blocks.append(TextBlock(
                        text=text,
                        page=page_num,
                        bbox=bbox,
                        block_type=block_type,
                        source='native',
                        confidence=1.0
                    ))
                
                # 提取表格
                if self.enable_table_detection:
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue
                        
                        # 将表格转换为markdown格式
                        table_text = self._table_to_markdown(table)
                        
                        # 表格bbox（简化处理，使用整个页面宽度）
                        bbox = (0, 0, page.width, page.height)
                        
                        blocks.append(TextBlock(
                            text=table_text,
                            page=page_num,
                            bbox=bbox,
                            block_type='table',
                            source='native',
                            confidence=1.0,
                            metadata={'table_index': table_idx}
                        ))
        
        return blocks
    
    def extract_with_ocr(self) -> List[TextBlock]:
        """
        OCR路径提取
        将每页转换为图像后进行OCR识别
        """
        if not self.use_ocr:
            return []
        
        blocks = []
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 渲染页面为图像
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # 转换为PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # OCR识别（获取详细的bbox信息）
            ocr_data = pytesseract.image_to_data(
                img,
                lang=self.ocr_language,
                output_type=pytesseract.Output.DICT
            )
            
            # 按行聚合OCR结果
            lines = self._group_ocr_into_lines(ocr_data, pix.width, pix.height, page.rect.width, page.rect.height)
            
            for line in lines:
                text = line['text'].strip()
                if not text or len(text) < 2:  # 过滤太短的文本
                    continue
                
                block_type = self._classify_block_type(text, line['bbox'], page.rect.height)
                
                blocks.append(TextBlock(
                    text=text,
                    page=page_num,
                    bbox=line['bbox'],
                    block_type=block_type,
                    source='ocr',
                    confidence=line['confidence']
                ))
        
        doc.close()
        return blocks
    
    def _group_words_into_lines(self, words: List[Dict]) -> List[Dict]:
        """将单词聚合成行"""
        if not words:
            return []
        
        # 按y坐标排序
        words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        lines = []
        current_line = {
            'text': words[0]['text'],
            'x0': words[0]['x0'],
            'x1': words[0]['x1'],
            'top': words[0]['top'],
            'bottom': words[0]['bottom']
        }
        
        for word in words[1:]:
            # 如果y坐标接近，认为是同一行
            if abs(word['top'] - current_line['top']) < 5:
                current_line['text'] += ' ' + word['text']
                current_line['x0'] = min(current_line['x0'], word['x0'])
                current_line['x1'] = max(current_line['x1'], word['x1'])
                current_line['bottom'] = max(current_line['bottom'], word['bottom'])
            else:
                lines.append(current_line)
                current_line = {
                    'text': word['text'],
                    'x0': word['x0'],
                    'x1': word['x1'],
                    'top': word['top'],
                    'bottom': word['bottom']
                }
        
        lines.append(current_line)
        return lines
    
    def _group_ocr_into_lines(
        self,
        ocr_data: Dict,
        img_width: int,
        img_height: int,
        page_width: float,
        page_height: float
    ) -> List[Dict]:
        """将OCR结果聚合成行，并转换坐标到PDF空间"""
        lines = []
        current_line = None
        
        scale_x = page_width / img_width
        scale_y = page_height / img_height
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if not text or conf < 0:
                continue
            
            # 转换bbox到PDF坐标
            x0 = ocr_data['left'][i] * scale_x
            y0 = ocr_data['top'][i] * scale_y
            x1 = (ocr_data['left'][i] + ocr_data['width'][i]) * scale_x
            y1 = (ocr_data['top'][i] + ocr_data['height'][i]) * scale_y
            
            if current_line is None:
                current_line = {
                    'text': text,
                    'bbox': (x0, y0, x1, y1),
                    'confidence': conf,
                    'count': 1
                }
            elif abs(y0 - current_line['bbox'][1]) < 10:  # 同一行
                current_line['text'] += ' ' + text
                current_line['bbox'] = (
                    min(current_line['bbox'][0], x0),
                    min(current_line['bbox'][1], y0),
                    max(current_line['bbox'][2], x1),
                    max(current_line['bbox'][3], y1)
                )
                current_line['confidence'] += conf
                current_line['count'] += 1
            else:
                # 计算平均置信度
                current_line['confidence'] /= current_line['count']
                current_line['confidence'] /= 100.0  # 转换为0-1
                lines.append(current_line)
                
                current_line = {
                    'text': text,
                    'bbox': (x0, y0, x1, y1),
                    'confidence': conf,
                    'count': 1
                }
        
        if current_line:
            current_line['confidence'] /= current_line['count']
            current_line['confidence'] /= 100.0
            lines.append(current_line)
        
        return lines
    
    def _classify_block_type(self, text: str, bbox: Tuple, page_height: float) -> str:
        """简单的块类型分类"""
        # 页眉页脚判断（基于位置）
        y_pos = bbox[1]
        if y_pos < page_height * 0.08:
            return 'header'
        if y_pos > page_height * 0.92:
            return 'footer'
        
        # 标题判断（简单规则）
        if len(text) < 50 and (
            text.isupper() or
            re.match(r'^[一二三四五六七八九十\d]+[、\.]', text) or
            re.match(r'^第[一二三四五六七八九十\d]+[章节]', text)
        ):
            return 'title'
        
        return 'text'
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """将表格转换为Markdown格式"""
        if not table:
            return ""
        
        lines = []
        for i, row in enumerate(table):
            # 清洗单元格
            cells = [str(cell).strip() if cell else '' for cell in row]
            lines.append('| ' + ' | '.join(cells) + ' |')
            
            # 添加表头分隔符
            if i == 0:
                lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
        
        return '\n'.join(lines)
    
    def merge_and_deduplicate(
        self,
        native_blocks: List[TextBlock],
        ocr_blocks: List[TextBlock],
        iou_threshold: float = 0.5,
        text_similarity_threshold: float = 0.8
    ) -> List[TextBlock]:
        """
        融合去重：合并原生文本和OCR结果
        
        策略：
        1. 优先使用原生文本（质量更高）
        2. 对于原生文本缺失的区域，使用OCR补充
        3. 基于IOU和文本相似度去重
        """
        merged_blocks = []
        used_ocr_indices = set()
        
        # 按页分组处理
        native_by_page = {}
        ocr_by_page = {}
        
        for block in native_blocks:
            if block.page not in native_by_page:
                native_by_page[block.page] = []
            native_by_page[block.page].append(block)
        
        for i, block in enumerate(ocr_blocks):
            if block.page not in ocr_by_page:
                ocr_by_page[block.page] = []
            ocr_by_page[block.page].append((i, block))
        
        # 处理每一页
        all_pages = set(native_by_page.keys()) | set(ocr_by_page.keys())
        
        for page_num in sorted(all_pages):
            native_page_blocks = native_by_page.get(page_num, [])
            ocr_page_blocks = ocr_by_page.get(page_num, [])
            
            # 1. 添加所有原生文本块
            merged_blocks.extend(native_page_blocks)
            
            # 2. 检查OCR块，如果与原生块重叠度低，则添加
            for ocr_idx, ocr_block in ocr_page_blocks:
                is_duplicate = False
                
                for native_block in native_page_blocks:
                    iou = ocr_block.iou(native_block)
                    
                    # 如果位置重叠且文本相似，认为是重复
                    if iou > iou_threshold:
                        text_sim = self._text_similarity(
                            ocr_block.text,
                            native_block.text
                        )
                        if text_sim > text_similarity_threshold:
                            is_duplicate = True
                            used_ocr_indices.add(ocr_idx)
                            break
                
                # 如果不是重复，添加OCR块（可能是图片区域的文字）
                if not is_duplicate and ocr_block.confidence > 0.6:
                    merged_blocks.append(ocr_block)
                    used_ocr_indices.add(ocr_idx)
        
        return merged_blocks
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算（基于字符集合的Jaccard相似度）"""
        if not text1 or not text2:
            return 0.0
        
        # 移除空格和标点
        clean1 = re.sub(r'[^\w]', '', text1.lower())
        clean2 = re.sub(r'[^\w]', '', text2.lower())
        
        if not clean1 or not clean2:
            return 0.0
        
        set1 = set(clean1)
        set2 = set(clean2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_layout(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        版面分析：检测多栏布局、段落等
        
        基本策略：
        1. 检测页面是否为多栏布局
        2. 对文本块进行排序（阅读顺序）
        3. 识别段落边界
        """
        if not self.enable_layout_analysis:
            return blocks
        
        # 按页分组
        blocks_by_page = {}
        for block in blocks:
            if block.page not in blocks_by_page:
                blocks_by_page[block.page] = []
            blocks_by_page[block.page].append(block)
        
        analyzed_blocks = []
        
        for page_num, page_blocks in sorted(blocks_by_page.items()):
            # 检测是否为多栏布局
            is_multi_column = self._detect_multi_column(page_blocks)
            
            if is_multi_column:
                # 多栏布局：按列分组后排序
                sorted_blocks = self._sort_multi_column_blocks(page_blocks)
            else:
                # 单栏布局：简单按y坐标排序
                sorted_blocks = sorted(page_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
            
            analyzed_blocks.extend(sorted_blocks)
        
        return analyzed_blocks
    
    def _detect_multi_column(self, blocks: List[TextBlock]) -> bool:
        """检测是否为多栏布局"""
        if len(blocks) < 5:
            return False
        
        # 统计文本块的x中心位置
        text_blocks = [b for b in blocks if b.block_type == 'text']
        if len(text_blocks) < 5:
            return False
        
        x_centers = [(b.bbox[0] + b.bbox[2]) / 2 for b in text_blocks]
        
        # 使用简单的聚类判断
        x_centers_sorted = sorted(x_centers)
        median = x_centers_sorted[len(x_centers_sorted) // 2]
        
        left_count = sum(1 for x in x_centers if x < median * 0.8)
        right_count = sum(1 for x in x_centers if x > median * 1.2)
        
        # 如果左右两侧都有足够的文本块，认为是多栏
        return left_count > 2 and right_count > 2
    
    def _sort_multi_column_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """对多栏布局的文本块进行排序"""
        # 简单策略：根据x坐标分成左右两列，然后分别按y排序
        if not blocks:
            return []
        
        x_centers = [(b.bbox[0] + b.bbox[2]) / 2 for b in blocks]
        median_x = sorted(x_centers)[len(x_centers) // 2]
        
        left_blocks = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 < median_x]
        right_blocks = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 >= median_x]
        
        left_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        right_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        
        # 交错合并（先左列第一段，再右列第一段，以此类推）
        # 这里简化为：所有左列 + 所有右列
        return left_blocks + right_blocks
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        return text.strip()
    
    def to_markdown(self, blocks: List[TextBlock]) -> str:
        """转换为Markdown格式"""
        lines = []
        current_page = -1
        
        for block in blocks:
            # 添加页面分隔符
            if block.page != current_page:
                if current_page >= 0:
                    lines.append('\n---\n')
                lines.append(f'## Page {block.page + 1}\n')
                current_page = block.page
            
            # 根据block类型格式化
            if block.block_type == 'title':
                lines.append(f'\n### {self.clean_text(block.text)}\n')
            elif block.block_type == 'table':
                lines.append(f'\n{block.text}\n')
            elif block.block_type in ['header', 'footer']:
                lines.append(f'\n*{self.clean_text(block.text)}*\n')
            else:
                lines.append(self.clean_text(block.text))
        
        return '\n'.join(lines)
    
    def to_json(self, blocks: List[TextBlock]) -> Dict:
        """转换为JSON格式"""
        return {
            'document': os.path.basename(self.pdf_path),
            'total_blocks': len(blocks),
            'blocks': [asdict(block) for block in blocks]
        }
    
    def to_chunks(
        self,
        blocks: List[TextBlock],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, any]]:
        """
        转换为文本块列表，适合RAG的embedding和检索
        
        Args:
            blocks: 文本块列表
            chunk_size: 每个chunk的最大字符数
            chunk_overlap: chunk之间的重叠字符数
            
        Returns:
            包含文本、元数据的chunk列表
        """
        chunks = []
        
        # 按页和块类型分组
        for block in blocks:
            # 跳过页眉页脚（通常不重要）
            if block.block_type in ['header', 'footer']:
                continue
            
            text = self.clean_text(block.text)
            
            # 如果文本块本身就很小，直接作为一个chunk
            if len(text) <= chunk_size:
                chunks.append({
                    'text': text,
                    'metadata': {
                        'page': block.page,
                        'block_type': block.block_type,
                        'source': block.source,
                        'confidence': block.confidence,
                        'bbox': block.bbox
                    }
                })
            else:
                # 大文本块需要切分
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_text = text[start:end]
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'page': block.page,
                            'block_type': block.block_type,
                            'source': block.source,
                            'confidence': block.confidence,
                            'bbox': block.bbox,
                            'chunk_index': len(chunks)
                        }
                    })
                    
                    start = end - chunk_overlap
        
        return chunks
    
    def load(
        self,
        output_format: str = 'chunks'
    ) -> Union[str, Dict, List[Dict]]:
        """
        完整加载流程
        
        Args:
            output_format: 输出格式 'markdown', 'json', 'chunks'
            
        Returns:
            根据output_format返回不同格式的数据
        """
        print(f"📄 正在加载PDF: {self.pdf_path}")
        
        # 1. 提取原生文本
        print("🔍 提取原生文本...")
        native_blocks = self.extract_native_text()
        print(f"   ✓ 提取 {len(native_blocks)} 个原生文本块")
        
        # 2. OCR提取（如果启用）
        ocr_blocks = []
        if self.use_ocr:
            print("🖼️  OCR识别中...")
            ocr_blocks = self.extract_with_ocr()
            print(f"   ✓ 提取 {len(ocr_blocks)} 个OCR文本块")
        
        # 3. 融合去重
        print("🔄 融合去重...")
        merged_blocks = self.merge_and_deduplicate(native_blocks, ocr_blocks)
        print(f"   ✓ 融合后共 {len(merged_blocks)} 个文本块")
        
        # 4. 版面分析
        if self.enable_layout_analysis:
            print("📐 版面分析...")
            merged_blocks = self.analyze_layout(merged_blocks)
            print(f"   ✓ 版面分析完成")
        
        # 5. 输出
        print(f"📤 生成 {output_format} 格式输出...")
        if output_format == 'markdown':
            result = self.to_markdown(merged_blocks)
        elif output_format == 'json':
            result = self.to_json(merged_blocks)
        elif output_format == 'chunks':
            result = self.to_chunks(merged_blocks)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        print("✅ 加载完成!")
        return result


if __name__ == "__main__":
    import sys

    # ====== 🔧 配置区（直接改这里）======
    CONFIG = {
        "pdf_path": r"D:\Aprojet\py\RAG_Agent\financial_rag_agent\dataset\raw_pdf\半导体\H2_AN202502281643614026_1.pdf",   # ← 直接改这里
        "output_dir": "outputs",
        "use_ocr": True,
        "dpi": 300,
        "ocr_language": "chi_sim+eng",
        "enable_table_detection": True,
        "enable_layout_analysis": True,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    # ===================================

    # 👉 如果命令行传了路径，就覆盖配置（但不用 --）
    if len(sys.argv) > 1:
        CONFIG["pdf_path"] = sys.argv[1]

    pdf_path = CONFIG["pdf_path"]

    if not os.path.exists(pdf_path):
        print(f"❌ PDF 文件不存在: {pdf_path}")
        sys.exit(1)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("=" * 80)
    print("🚀 Financial PDF Loader 测试")
    print(f"📄 文件: {pdf_path}")
    print(f"🖼️ OCR: {CONFIG['use_ocr']}")
    print("=" * 80)

    loader = FinancialPDFLoader(
        pdf_path=pdf_path,
        use_ocr=CONFIG["use_ocr"],
        ocr_language=CONFIG["ocr_language"],
        dpi=CONFIG["dpi"],
        enable_table_detection=CONFIG["enable_table_detection"],
        enable_layout_analysis=CONFIG["enable_layout_analysis"]
    )

    # ====== 核心流程（只跑一次）======
    native_blocks = loader.extract_native_text()

    ocr_blocks = []
    if CONFIG["use_ocr"]:
        ocr_blocks = loader.extract_with_ocr()

    merged_blocks = loader.merge_and_deduplicate(native_blocks, ocr_blocks)

    if CONFIG["enable_layout_analysis"]:
        merged_blocks = loader.analyze_layout(merged_blocks)

    # ====== 输出 ======
    markdown = loader.to_markdown(merged_blocks)
    json_data = loader.to_json(merged_blocks)
    chunks = loader.to_chunks(
        merged_blocks,
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"]
    )

    stem = Path(pdf_path).stem
    out_dir = CONFIG["output_dir"]

    md_path = os.path.join(out_dir, f"{stem}.md")
    json_path = os.path.join(out_dir, f"{stem}.json")
    chunks_path = os.path.join(out_dir, f"{stem}_chunks.json")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # ====== 验证输出 ======
    print("\n✅ 完成")
    print(f"🧱 blocks: {len(merged_blocks)}")
    print(f"📦 chunks: {len(chunks)}")
    print(f"📄 markdown: {md_path}")

    print("\n🔍 样例 chunk:")
    for i, c in enumerate(chunks[:3]):
        print(f"\n[{i}] page={c['metadata']['page']} | {c['metadata']['block_type']}")
        print(c["text"][:150])
        
    