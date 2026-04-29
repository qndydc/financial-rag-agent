import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import Any, List, Optional, Sequence

# ======================================
# 🔥 关键：用 langchain_core 官方基类
# ======================================
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents import BaseDocumentCompressor  # 最新官方基类

# pydantic 兼容新旧版
try:
    from pydantic.v1 import Field, PrivateAttr
except ImportError:
    from pydantic import Field, PrivateAttr

from sentence_transformers import CrossEncoder

from configs import model_config

RERANKER_MODEL_NAME = model_config.RERANKER_MODEL_NAME
RERANKER_TOP_N = model_config.RERANKER_TOP_N
RERANKER_DEVICE = model_config.RERANKER_DEVICE
RERANKER_MAX_LENGTH = model_config.RERANKER_MAX_LENGTH
RERANKER_BATCH_SIZE = model_config.RERANKER_BATCH_SIZE


class LangchainReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder Reranker."""

    model_name_or_path: str = Field()
    top_n: int = Field()
    device: str = Field()
    max_length: int = Field()
    batch_size: int = Field()
    num_workers: int = Field()
    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name_or_path: str = RERANKER_MODEL_NAME,
        top_n: int = RERANKER_TOP_N,
        device: str = RERANKER_DEVICE,
        max_length: int = RERANKER_MAX_LENGTH,
        batch_size: int = RERANKER_BATCH_SIZE,
        num_workers: int = 0,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            top_n=top_n,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self._model = CrossEncoder(
            model_name=model_name_or_path,
            max_length=max_length,   # 仍保留，作为最后一道保险
            device=device,
        )

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_markdown_blocks(self, text: str) -> List[str]:
        """
        先按 Markdown/段落结构粗切：
        - 标题块
        - 表格块
        - 普通段落块
        """
        text = self._normalize_text(text)
        if not text:
            return []

        lines = text.splitlines()
        blocks = []
        current = []
        in_table = False

        def flush():
            nonlocal current
            if current:
                block = "\n".join(current).strip()
                if block:
                    blocks.append(block)
                current = []

        for line in lines:
            s = line.strip()

            is_table_line = s.startswith("|") and s.endswith("|")
            is_heading = s.startswith("#")

            if is_table_line:
                if not in_table:
                    flush()
                    in_table = True
                current.append(line)
                continue

            if in_table and not is_table_line:
                flush()
                in_table = False

            if is_heading:
                flush()
                current.append(line)
                continue

            if s == "":
                flush()
                continue

            current.append(line)

        flush()
        return blocks

    def _split_long_doc(
        self,
        text: str,
        max_chars: int = 320,
        overlap: int = 60,
        max_segments: int = 8,
    ) -> List[str]:
        """
        文档先按 Markdown 结构切，再按句子切；过长再滑窗。
        """
        text = self._normalize_text(text)
        if not text:
            return []

        blocks = self._split_markdown_blocks(text)
        if not blocks:
            return []

        segments = []

        for block in blocks:
            if len(block) <= max_chars:
                segments.append(block)
                continue

            # 按句子切
            parts = re.split(r"(?<=[。！？；;.!?\n])", block)
            parts = [p.strip() for p in parts if p.strip()]

            current = ""
            for part in parts:
                if len(current) + len(part) <= max_chars:
                    current += part
                else:
                    if current:
                        segments.append(current)

                    if len(part) <= max_chars:
                        current = part
                    else:
                        # 极长句再滑窗切
                        start = 0
                        step = max(1, max_chars - overlap)
                        while start < len(part):
                            seg = part[start:start + max_chars].strip()
                            if seg:
                                segments.append(seg)
                            start += step
                        current = ""

            if current:
                segments.append(current)

        # 去空、去重、限制段数
        cleaned = []
        seen = set()
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if seg in seen:
                continue
            seen.add(seg)
            cleaned.append(seg)

        return cleaned[:max_segments]

    def _aggregate_scores(self, scores: List[float]) -> float:
        """
        同一个原文档多个 pair 分数聚合。
        用 max 最稳：只要某一段强相关，这个 doc 就该被保留。
        """
        if not scores:
            return float("-inf")
        return max(scores)

    def _is_noise_doc(self, text: str) -> bool:
        text = self._normalize_text(text)

        if not text or len(text) < 30:
            return True

        bad_keywords = [
            "本报告采用基本无氯气漂染纸浆",
            "年报封面",
            "封面由AI工具",
            "中式卷轴徐徐展开",
            "书写壮丽诗篇",
            "环保纸印刷",
            "目录",
            "释义",
        ]

        if any(k in text for k in bad_keywords):
            return True

        # 乱码比例过滤
        total = len(text)
        valid_chars = sum(
            ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
            for ch in text
        )

        if total > 0 and valid_chars / total < 0.45:
            return True

        return False

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using CrossEncoder rerank.
        兼容长 query / 长 doc：
        - 短 query 不拆
        - 长 query 切成多个 query segments
        - 长 doc 切成多个 doc segments
        - 对所有 (q_seg, d_seg) 组合打分
        - 分数聚合回原始 doc
        """
        if len(documents) == 0:
            return []

         # 先过滤明显噪声
        doc_list = [
            doc for doc in list(documents)
            if not self._is_noise_doc(doc.page_content)
        ]

        query_segments = [self._normalize_text(query)]

        if not query_segments:
            return doc_list[: self.top_n]

        all_pairs = []
        pair_doc_indices = []

        for doc_idx, doc in enumerate(doc_list):
            doc_text = self._normalize_text(doc.page_content)
            doc_segments = self._split_long_doc(doc_text)

            if not doc_segments:
                continue

            for q_seg in query_segments:
                for d_seg in doc_segments:
                    all_pairs.append([q_seg, d_seg])
                    pair_doc_indices.append(doc_idx)

        if not all_pairs:
            return doc_list[: self.top_n]

        results = self._model.predict( # 用cross-encoder模型对所有 (query segment, doc segment) 对进行打分
            sentences=all_pairs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            convert_to_tensor=True,
        )

        # 聚合同一 doc 的多个 pair 分数
        doc_score_map = {i: [] for i in range(len(doc_list))}
        for score, doc_idx in zip(results.tolist(), pair_doc_indices):
            doc_score_map[doc_idx].append(float(score))

        scored_docs = []
        for doc_idx, scores in doc_score_map.items():
            if not scores:
                continue
            final_score = self._aggregate_scores(scores)
            doc = doc_list[doc_idx]
            doc.metadata["score"] = final_score
            scored_docs.append((final_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_k = self.top_n if self.top_n < len(scored_docs) else len(scored_docs)
        final_results = [doc for _, doc in scored_docs[:top_k]]

        return final_results