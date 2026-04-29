# -*- coding: utf-8 -*-
"""
金融研报 Markdown 文本切分器
基于 langchain-chatchat 中文切分逻辑优化
专门适配：带表格、标题、页码的 Markdown 文本
保证：标题不拆分、表格不拆分、句子完整、语义连贯
"""

import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # 添加项目根目录到路径
from configs import rag_config



class ChineseMarkdownTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, **kwargs):
        # 从配置文件读取默认值
        chunk_size = chunk_size or rag_config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or rag_config.CHUNK_OVERLAP

        # 切分优先级：先按 Markdown 结构 → 再按段落 → 再按句子 → 再按标点
        separators = [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]

        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True,
            is_separator_regex=True,
            **kwargs
        )

    def _clean_text_before_split(self, text: str) -> str:
        """
        预清洗：
        1. 清理测试打印中的 ===== 分隔线
        2. 清理孤立页码，如 1 / 219
        3. 清理多余空白
        """
        text = text.replace("\xa0", " ")

        # 去掉测试输出中可能残留的分隔线
        text = re.sub(r"^\s*=+\s*$", "", text, flags=re.MULTILINE)

        # 去掉孤立页码，如：1 / 219
        text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # 去掉过多空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def split_text(self, text: str) -> List[str]:
        """
        重写 split_text：
        1. 先保护整张 Markdown 表格不被拆分
        2. 再做中文递归切分
        3. 最后还原表格并清洗文本
        """
        text = self._clean_text_before_split(text)

        # ========== 步骤1：保护整张 Markdown 表格 ==========
        # 匹配连续表格块，而不是单行
        # 例如：
        # | A | B |
        # |---|---|
        # | 1 | 2 |
        table_block_pattern = re.compile(
            r'((?:^\|.*\|\s*$\n?)+)',
            re.MULTILINE
        )

        table_placeholders = {}

        def replace_table_block(match):
            table_block = match.group(1).strip()
            placeholder = f"[[TABLE_BLOCK_{len(table_placeholders)}]]"
            table_placeholders[placeholder] = table_block
            return placeholder

        text = table_block_pattern.sub(replace_table_block, text)

        # ========== 步骤2：执行递归切分 ==========
        chunks = super().split_text(text)

        # ========== 步骤3：还原表格 + 清洗 ==========
        final_chunks = []
        for chunk in chunks:
            # 还原表格块
            for ph, table_block in table_placeholders.items():
                chunk = chunk.replace(ph, table_block)

            # 清洗空白
            chunk = chunk.strip()
            chunk = re.sub(r"\n{3,}", "\n\n", chunk)

            if chunk:
                final_chunks.append(chunk)

        return final_chunks


# 快速测试
if __name__ == "__main__":
    # 模拟你上一步提取的 Markdown 文本
    test_md = """
海光信息技术股份有限公司2024 年年度报告

| | 2024年末 | 2023年末 | 本期末比<br>上年同期<br>末增减<br>（%） | 2022年末 |
| --- | --- | --- | --- | --- |
| 归属于上市公司股<br>东的净资产 | 20,250,959,179.95 | 18,705,083,962.67 | 8.26 | 17,053,149,859.87 |
| 总资产 | 28,559,492,036.59 | 22,902,547,952.79 | 24.70 | 21,934,487,694.40 |

(二) 主要财务指标

| 主要财务指标 | 2024年 | 2023年 | 本期比上年同期增<br>减(%) | 2022年 |
| --- | --- | --- | --- | --- |
| 基本每股收益（元／股） | 0.83 | 0.54 | 53.70 | 0.38 |
| 稀释每股收益（元／股） | 0.83 | 0.54 | 53.70 | 0.38 |
| 扣除非经常性损益后的基本每股<br>收益（元／股） | 0.78 | 0.49 | 59.18 | 0.35 |
| 加权平均净资产收益率（%） | 9.92 | 7.11 | 增加2.81个百分点 | 8.49 |
| 扣除非经常性损益后的加权平均<br>净资产收益率（%） | 9.32 | 6.40 | 增加2.92个百分点 | 7.91 |
| 研发投入占营业收入的比例（%） | 37.61 | 46.74 | 减少9.13个百分点 | 40.33 |

√适用 □不适用
1. 营业收入、归属于上市公司股东的净利润、归属于上市公司股东的扣除非经常性损益的净
利润，同比实现较快增长，主要系报告期内，公司围绕通用计算市场，持续保持高强度的研发投
入，不断实现技术创新、产品性能提升，获得用户更广泛认可，进一步拓展了产品的应用领域，
加之国产化市场占比进一步提升，促进公司业绩显著增长。
2. 基本每股收益、稀释每股收益、扣除非经常性损益后的基本每股收益，较上年同期大幅上
升，主要得益于公司产品的市场销售大幅提升，收入和利润显著增长，带动每股盈利水平整体提
升。
3. 报告期内，公司持续加大研发投入力度，研发投入同比增长22.63%，由于营业收入增幅高
于研发投入增长，研发投入占营业收入的比例同比有所下降。
七、境内外会计准则下会计数据差异
(一) 同时按照国际会计准则与按中国会计准则披露的财务报告中净利润和归属于上市公司股东
的净资产差异情况
□适用 √不适用
(二) 同时按照境外会计准则与按中国会计准则披露的财务报告中净利润和归属于上市公司股东
的净资产差异情况
□适用 √不适用
(三) 境内外会计准则差异的说明：
□适用 √不适用
9 / 219
    """

    splitter = ChineseMarkdownTextSplitter()
    chunks = splitter.split_text(test_md)

    print("=" * 60)
    print("切分结果：")
    print("=" * 60)
    for i, chunk in enumerate(chunks):
        print(f"\n【Chunk {i+1}】")
        print(chunk)
