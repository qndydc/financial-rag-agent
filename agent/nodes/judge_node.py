# -*- coding: utf-8 -*-
from typing import Dict

from agent.state.agent_state import AgentState


def judge_node(state: AgentState) -> Dict:
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    print(f"[judge_node] retry_count = {retry_count}")
    print(f"[judge_node] docs_count = {len(docs)}")

    if not docs:
        return {
            "retrieval_success": False,
            "fallback_reason": "没有检索到相关研报内容。",
            "debug_info": {
                **state.get("debug_info", {}),
                "judge_reason": "no_docs",
            }
        }

    top1 = docs[0]
    top1_score = top1.get("score", None)

    print(f"[judge_node] top1_score = {top1_score}")
    print(f"[judge_node] top1_file = {top1.get('file_name', '')}")

    # 只有当 score 真的存在时，才做阈值判断
    if top1_score is not None:
        if isinstance(top1_score, (int, float)) and top1_score < 0.3:
            return {
                "retrieval_success": False,
                "fallback_reason": f"检索结果相关性过低（top1 score={top1_score:.4f}）。",
                "debug_info": {
                    **state.get("debug_info", {}),
                    "judge_reason": "low_score",
                    "top1_score": top1_score,
                }
            }

    # 没有 score 但有 docs，先认为通过
    return {
        "retrieval_success": True,
        "debug_info": {
            **state.get("debug_info", {}),
            "judge_reason": "pass",
            "retrieved_count": len(docs),
            "top1_score": top1_score,
        }
    }