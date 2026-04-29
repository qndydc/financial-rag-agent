# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool


class RagSearchAdapter:
    """
    将底层 RAG search_fn 统一封装成：
    1. 普通 Python 调用接口
    2. LangChain Tool 接口
    """

    def __init__(self, search_fn: Callable):
        self.search_fn = search_fn

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        """
        给 node 直接调用的普通接口
        返回结构化 docs，而不是纯字符串
        """
        results = self.search_fn(
            query=query,
            mode=mode,
            use_reranker=use_reranker,
        )

        docs = self._normalize_results(results)
        return {
            "query": query,
            "docs": docs,
        }

    def as_langchain_tool(self):
        """
        兼容 function-calling 的 tool 版本
        """

        @tool("rag_search")
        def rag_search(query: str) -> str:
            """
            搜索金融研报知识库，返回与问题最相关的研报内容片段。
            """
            result = self.search(query=query)
            docs = result["docs"]

            if not docs:
                return "未检索到相关研报内容。"

            formatted = []
            for i, d in enumerate(docs, 1):
                formatted.append(
                    f"[片段{i}]\n"
                    f"文件名: {d.get('file_name', '')}\n"
                    f"页码: {d.get('page_num', '')}\n"
                    f"内容: {d.get('content', '')}\n"
                )
            return "\n".join(formatted)

        return rag_search

    def _normalize_results(self, results: Any) -> List[Dict[str, Any]]:
        """
        将底层 RAG 返回结果统一标准化
        兼容：
        - List[Document]
        - Dict
        - List[Dict]
        """
        normalized_docs: List[Dict[str, Any]] = []

        if not results:
            return normalized_docs

        # 如果底层直接返回 dict，尝试取 docs
        if isinstance(results, dict):
            results = results.get("docs", results.get("documents", []))

        # 单个 Document
        if isinstance(results, Document):
            results = [results]

        # List[Document] / List[Dict]
        if isinstance(results, list):
            for idx, item in enumerate(results):
                if isinstance(item, Document):
                    normalized_docs.append(
                        {
                            "doc_id": item.metadata.get("doc_id", f"doc_{idx}"),
                            "chunk_id": item.metadata.get("chunk_id", f"chunk_{idx}"),
                            "file_name": item.metadata.get("file_name", ""),
                            "page_num": item.metadata.get("page_num", ""),
                            "content": item.page_content,
                            "score": item.metadata.get("score", None),
                            "metadata": item.metadata,
                        }
                    )
                elif isinstance(item, dict):
                    normalized_docs.append(
                        {
                            "doc_id": item.get("doc_id", f"doc_{idx}"),
                            "chunk_id": item.get("chunk_id", f"chunk_{idx}"),
                            "file_name": item.get("file_name", item.get("metadata", {}).get("file_name", "")),
                            "page_num": item.get("page_num", item.get("metadata", {}).get("page_num", "")),
                            "content": item.get("content", item.get("page_content", "")),
                            "score": item.get("score", item.get("metadata", {}).get("score", None)),
                            "metadata": item.get("metadata", {}),
                        }
                    )

        return normalized_docs


def create_rag_search_adapter(search_fn: Callable) -> RagSearchAdapter: #用一个class将search_fn保存起来
    return RagSearchAdapter(search_fn)