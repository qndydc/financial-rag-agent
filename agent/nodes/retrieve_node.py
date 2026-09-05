# -*- coding: utf-8 -*-
from typing import Dict, List, Any

from agent.call_lifecycle import call_lifecycle
from agent.state.agent_state import AgentState
from configs import model_config


def build_retrieve_node(rag_adapter, structured_rag_adapter):
    """
    通过闭包把 rag_adapter 注入节点

    说明：
    1. 这里只接受结构化检索对象 structured_query
    2. 不再兼容旧版 rewritten_query / user_input 直传检索
    3. rag_adapter.search(query=structured_query) 需要返回：
       {
           "docs": [...],
           "meta": {...}
       }
    """

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        structured_query = state.get("structured_query", {})
        state_rewritten_query = state.get("rewritten_query", "")
        user_input = state["user_input"]
        task_type = structured_query.get("task_type", "")

        # 定义一个简单状态，简单状态直接普通查询，复杂状态走结构化查询
        if task_type != "fact":  # 如果不是普通查询，就走结构化检索
            def invoke_search(structured_query, mode, use_reranker):
                return structured_rag_adapter.search(
                    structured_query,
                    mode=mode,
                    use_reranker=use_reranker,
                )

            outcome = call_lifecycle.execute(
                "structured_rag_search",
                invoke_search,
                {
                    "structured_query": structured_query,
                    "mode": "hybrid",
                    "use_reranker": True,
                },
                argument_retryable=(
                    state.get("argument_repair_count", 0)
                    < state.get("argument_repair_limit", model_config.ARGUMENT_REPAIR_LIMIT)
                ),
                empty_retryable=(
                    state.get("generalization_count", 0)
                    < state.get("generalization_limit", model_config.EMPTY_RESULT_RETRY_LIMIT)
                ),
                is_empty=lambda value: not value or not value.get("docs", []),
                summarize=lambda value: {"doc_count": len((value or {}).get("docs", []))},
            )
            print(f"[retrieve_node] 使用结构化检索，structured_query = {structured_query}")
        else:
            query = state_rewritten_query or user_input

            def invoke_search(query, mode, use_reranker):
                return rag_adapter.search(
                    query,
                    mode=mode,
                    use_reranker=use_reranker,
                )

            outcome = call_lifecycle.execute(
                "rag_search",
                invoke_search,
                {
                    "query": query,
                    "mode": "hybrid",
                    "use_reranker": True,
                },
                argument_retryable=(
                    state.get("argument_repair_count", 0)
                    < state.get("argument_repair_limit", model_config.ARGUMENT_REPAIR_LIMIT)
                ),
                empty_retryable=(
                    state.get("generalization_count", 0)
                    < state.get("generalization_limit", model_config.EMPTY_RESULT_RETRY_LIMIT)
                ),
                is_empty=lambda value: not value or not value.get("docs", []),
                summarize=lambda value: {"doc_count": len((value or {}).get("docs", []))},
            )
            print(f"[retrieve_node] 使用普通检索，query = {query}")

        observation = outcome.observation.model_dump()
        result = outcome.value or {}
        docs = result.get("docs", [])
        meta = result.get("meta", {})

        citations = _build_citations(docs)

        # 便于调试：记录本轮实际执行了哪些子查询
        sub_queries = structured_query.get("sub_queries", [])
        structured_rewritten_query = structured_query.get("rewritten_query", "")
        filters = structured_query.get("filters", {})
        exclude_terms = structured_query.get("exclude_terms", [])
        required_terms = structured_query.get("required_terms", [])

        print(f"[retrieve_node] task_type = {task_type}")
        print(f"[retrieve_node] rewritten_query = {structured_rewritten_query or state_rewritten_query}")
        print(f"[retrieve_node] sub_queries = {sub_queries}")
        print(f"[retrieve_node] filters = {filters}")
        print(f"[retrieve_node] exclude_terms = {exclude_terms}")
        print(f"[retrieve_node] required_terms = {required_terms}")
        print(f"[retrieve_node] docs_count = {len(docs)}")

        if docs:
            top1 = docs[0]
            print(f"[retrieve_node] top1_file = {top1.get('file_name', '')}")
            print(f"[retrieve_node] top1_page = {top1.get('page_num', '')}")
            print(f"[retrieve_node] top1_score = {top1.get('score', None)}")
            print(f"[retrieve_node] top1_retrieved_by = {top1.get('retrieved_by', '')}")

        return {
            "retrieved_docs": docs,
            "citations": citations,
            "last_observation": observation,
            "observations": [observation],
            "debug_info": {
                **state.get("debug_info", {}),
                "retrieved_count": len(docs),
                "retrieved_task_type": task_type,
                "retrieved_query": structured_rewritten_query or state_rewritten_query or user_input,
                "retrieved_sub_queries": sub_queries,
                "retrieve_filters": filters,
                "retrieve_exclude_terms": exclude_terms,
                "retrieve_required_terms": required_terms,
                "retrieve_meta": meta,
            }
        }

    return retrieve_node


def _build_citations(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    保留你原来的 citation 结构，同时补充 retrieved_by 方便调试多路召回来源
    """
    citations = []

    for d in docs:
        citations.append(
            {
                "file_name": d.get("file_name", ""),
                "page_num": d.get("page_num", ""),
                "chunk_id": d.get("chunk_id", ""),
                "snippet": d.get("content", "")[:180],
                "retrieved_by": d.get("retrieved_by", ""),
            }
        )

    return citations
