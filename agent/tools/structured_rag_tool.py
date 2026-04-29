# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Callable

from langchain_core.documents import Document
from langchain_core.tools import tool


class StructuredRagSearchAdapter:
    """
    接收包装好的 search_fn，支持 structured query：
    - 多 sub_queries 分别调用已有 search_fn(query, mode, use_reranker)
    - 标准化底层返回结果
    - 去重
    - 后过滤
    - 均衡融合
    """

    def __init__(self, search_fn: Callable, final_top_k: int = 5):
        if not callable(search_fn):
            raise TypeError("search_fn 必须是可调用对象")
        self.search_fn = search_fn
        self.final_top_k = final_top_k

    def search(
        self,
        structured_query: Dict[str, Any],
        mode: str = "hybrid",
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        if not isinstance(structured_query, dict):
            raise TypeError("structured_query 必须是 dict")

        sub_queries = structured_query.get("sub_queries", []) or []
        rewritten_query = structured_query.get("rewritten_query", "") or ""
        filters = structured_query.get("filters", {}) or {}
        exclude_terms = structured_query.get("exclude_terms", []) or []
        required_terms = structured_query.get("required_terms", []) or []

        if not sub_queries:
            if rewritten_query:
                sub_queries = [rewritten_query]
            else:
                raise ValueError("structured_query 至少需要 sub_queries 或 rewritten_query")

        all_docs: List[Dict[str, Any]] = []

        for sq in sub_queries: #对输入的每一个子查询都调用一次底层检索接口，得到的结果会带上一个字段 "retrieved_by" 标记是由哪个子查询检索到的
            raw_results = self.search_fn(
                query=sq,
                mode=mode,
                use_reranker=use_reranker,
            )
            docs = self._normalize_results(raw_results)

            for d in docs:
                d["retrieved_by"] = sq

            docs = self._apply_filters(docs, filters)
            docs = self._apply_exclude_terms(docs, exclude_terms)
            docs = self._apply_required_terms(docs, required_terms)

            all_docs.extend(docs)

        docs = self._deduplicate_docs(all_docs)
        docs = self._balance_docs_by_subquery(
            docs,
            sub_queries=sub_queries,
            final_top_k=structured_query.get("top_k", self.final_top_k),
        )

        return {
            "query": structured_query,
            "docs": docs,
        }

    def as_langchain_tool(self):
        @tool("structured_rag_search")
        def structured_rag_search(query_json: str) -> str:
            """
            搜索金融研报知识库，支持结构化复杂查询 JSON。
            """
            import json

            structured_query = json.loads(query_json)
            result = self.search(structured_query=structured_query)
            docs = result["docs"]

            if not docs:
                return "未检索到相关研报内容。"

            formatted = []
            for i, d in enumerate(docs, 1):
                formatted.append(
                    f"[片段{i}]\n"
                    f"文件名: {d.get('file_name', '')}\n"
                    f"页码: {d.get('page_num', '')}\n"
                    f"子查询: {d.get('retrieved_by', '')}\n"
                    f"内容: {d.get('content', '')}\n"
                )
            return "\n".join(formatted)

        return structured_rag_search

    def _normalize_results(self, results: Any) -> List[Dict[str, Any]]: #同样的函数在rag_tool里也有
        """
        将底层 search_fn 返回统一标准化
        兼容：
        - List[Document]
        - Dict
        - List[Dict]
        - 单个 Document
        """
        normalized_docs: List[Dict[str, Any]] = []

        if not results:
            return normalized_docs

        if isinstance(results, dict):
            results = results.get("docs", results.get("documents", []))

        if isinstance(results, Document):
            results = [results]

        if isinstance(results, list):
            for idx, item in enumerate(results):
                if isinstance(item, Document):
                    metadata = item.metadata or {}
                    normalized_docs.append(
                        {
                            "doc_id": metadata.get("doc_id", f"doc_{idx}"),
                            "chunk_id": metadata.get("chunk_id", f"chunk_{idx}"),
                            "file_name": metadata.get("file_name", ""),
                            "page_num": metadata.get("page_num", ""),
                            "content": item.page_content,
                            "score": metadata.get("score", None),
                            "metadata": metadata,
                        }
                    )
                elif isinstance(item, dict):
                    metadata = item.get("metadata", {}) or {}
                    normalized_docs.append(
                        {
                            "doc_id": item.get("doc_id", f"doc_{idx}"),
                            "chunk_id": item.get("chunk_id", f"chunk_{idx}"),
                            "file_name": item.get("file_name", metadata.get("file_name", "")),
                            "page_num": item.get("page_num", metadata.get("page_num", "")),
                            "content": item.get("content", item.get("page_content", "")),
                            "score": item.get("score", metadata.get("score", None)),
                            "metadata": metadata,
                        }
                    )

        return normalized_docs

    def _apply_filters(self, docs: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not filters:
            return docs

        values = [str(v).strip() for v in filters.values() if v not in ("", None, [], {})]
        if not values:
            return docs

        filtered = []
        for d in docs:
            text = f"{d.get('content', '')} {d.get('metadata', {})}".lower()
            if all(v.lower() in text for v in values):
                filtered.append(d)

        return filtered if filtered else docs

    def _apply_exclude_terms(self, docs: List[Dict[str, Any]], exclude_terms: List[str]) -> List[Dict[str, Any]]:
        if not exclude_terms:
            return docs

        filtered = []
        for d in docs:
            text = d.get("content", "").lower()
            if any(term.lower() in text for term in exclude_terms):
                continue
            filtered.append(d)
        return filtered

    def _apply_required_terms(self, docs: List[Dict[str, Any]], required_terms: List[str]) -> List[Dict[str, Any]]:
        if not required_terms:
            return docs

        filtered = []
        for d in docs:
            text = d.get("content", "").lower()
            if all(term.lower() in text for term in required_terms):
                filtered.append(d)

        return filtered if filtered else docs

    def _deduplicate_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = {}
        for d in docs:
            key = (
                str(d.get("file_name", "")),
                str(d.get("page_num", "")),
                str(d.get("chunk_id", "")),
            )
            if key not in seen:
                seen[key] = d
        return list(seen.values())

    def _balance_docs_by_subquery(
        self,
        docs: List[Dict[str, Any]],
        sub_queries: List[str],
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        buckets = {sq: [] for sq in sub_queries}
        others = []

        for d in docs:
            sq = d.get("retrieved_by")
            if sq in buckets:
                buckets[sq].append(d)
            else:
                others.append(d)

        result = []
        seen = set()

        while len(result) < final_top_k:
            moved = False
            for sq in sub_queries:
                if buckets[sq]:
                    d = buckets[sq].pop(0)
                    key = (
                        str(d.get("file_name", "")),
                        str(d.get("page_num", "")),
                        str(d.get("chunk_id", "")),
                    )
                    if key not in seen:
                        seen.add(key)
                        result.append(d)
                        moved = True
                        if len(result) >= final_top_k:
                            break
            if not moved:
                break

        if len(result) < final_top_k:
            for d in others:
                key = (
                    str(d.get("file_name", "")),
                    str(d.get("page_num", "")),
                    str(d.get("chunk_id", "")),
                )
                if key not in seen:
                    seen.add(key)
                    result.append(d)
                    if len(result) >= final_top_k:
                        break

        return result


def create_structured_rag_search_adapter(search_fn: Callable, final_top_k: int = 5) -> StructuredRagSearchAdapter:
    return StructuredRagSearchAdapter(
        search_fn=search_fn,
        final_top_k=final_top_k,
    )


if __name__ == "__main__":
    import json
    import time
    import sys
    from pathlib import Path
    from collections import Counter

    # 让本地直接运行时能找到项目根目录
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))

    print("=" * 100)
    print("StructuredRagSearchAdapter 多路召回测试启动")
    print("=" * 100)

    try:
        # 你需要按自己项目里的真实路径改这几个 import
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from rag.retrievers.hybrid_retriever import create_hybrid_retriever
        from rag import load_vector_store, create_hybrid_retriever
        from configs import rag_config
    except Exception as e:
        print("[ERROR] 导入底层 RAG 模块失败")
        print("请检查以下模块路径是否和你的项目一致：")
        print(" - rag.retrievers.hybrid_retriever.create_hybrid_retriever")
        print(" - rag.retrievers.vectorstore_loader.load_vectorstore_and_documents")
        print(f"详细错误: {e}")
        raise

    try:
        print("[TEST] 正在加载向量库和文档...")
        vs_path = rag_config.VECTOR_STORE_DIR
        docs_path = f"{rag_config.VECTOR_STORE_DIR}/all_documents.json"

        # 1. 加载 RAG
        print("[Agent] 正在加载向量库...")
        vectorstore = load_vector_store(vs_path)

        def load_all_documents(json_path: str) -> List[Document]:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return [
                Document(
                    page_content=d["page_content"],
                    metadata=d.get("metadata", {}),
                )
                for d in data
            ]
        all_documents = load_all_documents(docs_path)

        print("[TEST] 正在构建底层 search_fn...")
        search_fn = create_hybrid_retriever(
            vectorstore=vectorstore,
            all_documents=all_documents,
        )

        adapter = create_structured_rag_search_adapter(
            search_fn=search_fn,
            final_top_k=6,
        )

        # -----------------------------
        # 测试样例 1：对比类问题
        # -----------------------------
        structured_query_1 = {
            "task_type": "contrast",
            "original_query": "对比一下上海航天汽车机电股份有限公司以及海光信息在2024年的营销、收入、市场等方面的差距",
            "rewritten_query": "上海航天汽车机电股份有限公司 海光信息 2024 营销 收入 市场 对比",
            "entities": ["上海航天汽车机电股份有限公司", "海光信息"],
            "metrics": ["营销", "收入", "市场"],
            "sub_queries": [
                "上海航天汽车机电股份有限公司 2024 营销 收入 市场",
                "海光信息 2024 营销 收入 市场",
            ],
            "filters": {
                "year": "2024"
            },
            "exclude_terms": [],
            "required_terms": [],
            "top_k": 6,
        }

        # -----------------------------
        # 测试样例 2：多条件 + 排除词
        # -----------------------------
        structured_query_2 = {
            "task_type": "multi_constraint_exclude",
            "original_query": "找海光信息2024年与营业收入、毛利率、研发费用有关的信息，但尽量排除资产负债表和现金流表",
            "rewritten_query": "海光信息 2024 营业收入 毛利率 研发费用 排除资产负债表 现金流",
            "entities": ["海光信息"],
            "metrics": ["营业收入", "毛利率", "研发费用"],
            "sub_queries": [
                "海光信息 2024 营业收入",
                "海光信息 2024 毛利率",
                "海光信息 2024 研发费用",
            ],
            "filters": {
                "year": "2024"
            },
            "exclude_terms": ["资产负债表", "现金流量表", "现金流"],
            "required_terms": [],
            "top_k": 6,
        }

        test_cases = [
            ("测试1-对比类多路召回", structured_query_1),
            ("测试2-多条件+排除词多路召回", structured_query_2),
        ]

        for case_name, sq in test_cases:
            print("\n" + "=" * 100)
            print(f"{case_name}")
            print("=" * 100)
            print("[Structured Query]")
            print(json.dumps(sq, ensure_ascii=False, indent=2))

            # 先逐个子查询单独测，便于看每一路召回效果
            print("\n[Step 1] 各子查询单独召回结果")
            per_subquery_stats = []

            for idx, sub_q in enumerate(sq.get("sub_queries", []), 1):
                start = time.perf_counter()
                raw_docs = search_fn(query=sub_q, mode="hybrid", use_reranker=True)
                elapsed = time.perf_counter() - start

                normalized_docs = adapter._normalize_results(raw_docs)

                print(f"\n  ({idx}) 子查询: {sub_q}")
                print(f"      原始返回文档数: {len(normalized_docs)}")
                print(f"      耗时: {elapsed:.4f}s")

                if normalized_docs:
                    for j, d in enumerate(normalized_docs[:5], 1):
                        print(
                            f"      - Top{j}: file={d.get('file_name', '')}, "
                            f"page={d.get('page_num', '')}, "
                            f"score={d.get('score', None)}"
                        )
                else:
                    print("      - 无召回结果")

                per_subquery_stats.append(
                    {
                        "sub_query": sub_q,
                        "count": len(normalized_docs),
                        "elapsed_sec": elapsed,
                    }
                )

            # 再跑 structured adapter
            print("\n[Step 2] 结构化多路召回 + 融合 + 去重 + 均衡")
            start = time.perf_counter()
            result = adapter.search(
                structured_query=sq,
                mode="hybrid",
                use_reranker=True,
            )
            elapsed = time.perf_counter() - start

            docs = result["docs"]

            print(f"最终返回文档数: {len(docs)}")
            print(f"总耗时: {elapsed:.4f}s")

            if not docs:
                print("[WARN] 最终无文档返回")
                continue

            subquery_counter = Counter()
            file_counter = Counter()

            print("\n[Final Docs]")
            for i, d in enumerate(docs, 1):
                retrieved_by = d.get("retrieved_by", "")
                file_name = d.get("file_name", "")
                page_num = d.get("page_num", "")
                score = d.get("score", None)
                content_preview = d.get("content", "").replace("\n", " ")[:120]

                subquery_counter[retrieved_by] += 1
                file_counter[file_name] += 1

                print(
                    f"[{i}] file={file_name} | page={page_num} | "
                    f"score={score} | retrieved_by={retrieved_by}"
                )
                print(f"    preview: {content_preview}")

            print("\n[Summary]")
            print("各子查询在最终结果中的贡献：")
            for k, v in subquery_counter.items():
                print(f" - {k}: {v}")

            print("最终结果文件分布：")
            for k, v in file_counter.items():
                print(f" - {k}: {v}")

        print("\n" + "=" * 100)
        print("StructuredRagSearchAdapter 多路召回测试完成")
        print("=" * 100)

    except Exception as e:
        print("\n[ERROR] StructuredRagSearchAdapter 测试失败")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        raise
    