# -*- coding: utf-8 -*-
import re
from typing import Dict

from agent.state.agent_state import AgentState


CHAT_PATTERNS = [
    r"你是谁",
    r"你是什么模型",
    r"介绍一下你自己",
    r"请介绍.*你自己",
    r"你好",
    r"您好",
    r"hello",
    r"\bhi\b",
    r"你能做什么",
]

UNCLEAR_PATTERNS = [
    r"这个怎么样",
    r"这个呢",
    r"那这个呢",
    r"那它呢",
    r"这个公司怎么样",
]


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    # 去掉常见中英文标点和多余空格
    text = re.sub(r"[？?！!，,。.；;：:\s]+", "", text)
    return text


def _match_any(text: str, patterns) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def route_node(state: AgentState) -> Dict:
    raw_input = state.get("user_input", "").strip()
    norm_input = _normalize_text(raw_input)

    # 1. 闲聊 / 身份类问题
    if _match_any(norm_input, [p.replace(".*", "") for p in CHAT_PATTERNS]) or _match_any(raw_input.lower(), CHAT_PATTERNS):
        print(f"[route_node] chat")
        return {
            "intent": "chat",
            "debug_info": {
                **state.get("debug_info", {}),
                "route_intent": "chat",
                "route_input": raw_input,
            }
        }

    # 2. 明显过于模糊
    if _match_any(norm_input, [p.replace(".*", "") for p in UNCLEAR_PATTERNS]) or len(norm_input) <= 2:
        print(f"[route_node] unclear")
        return {
            "intent": "unclear",
            "fallback_reason": "问题过于模糊，缺少明确的公司、行业、时间或指标信息。",
            "debug_info": {
                **state.get("debug_info", {}),
                "route_intent": "unclear",
                "route_input": raw_input,
            }
        }

    # 3. 默认走金融 RAG
    print(f"[route_node] rag_qa")
    return {
        "intent": "rag_qa",
        "debug_info": {
            **state.get("debug_info", {}),
            "route_intent": "rag_qa",
            "route_input": raw_input,
        }
    }