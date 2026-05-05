# -*- coding: utf-8 -*-
from datetime import date
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state.agent_state import AgentState
from agent.llm.base_llm import get_llm


ANSWER_SYSTEM_PROMPT = """
你是金融研报分析助手。

你的任务：
基于提供的研报检索片段回答用户问题，并可参考少量历史对话记忆理解上下文。

要求：
1. 只能依据给定材料回答，不要编造
2. 历史记忆仅用于辅助理解上下文，不可替代当前检索到的研报材料
3. 如果历史记忆与当前检索材料冲突，以当前检索材料为准
4. 优先给出简洁、专业、结论明确的回答
5. 如果材料不足，请明确说明“根据当前检索到的研报内容，暂无足够证据……”
6. 回答最后附“来源”列表，格式如下：
   - 文件名，第X页
7. 今天日期是：{current_date}
"""

CHAT_SYSTEM_PROMPT = """
你是一个通用AI助手。

你的任务：
结合必要的历史对话信息，简单回复用户问题

要求：
1. 直接回答用户问题，不要解释
2. 不要附加任何来源信息
3. 不要提及“研报”或“检索”。
4. 如果历史信息与当前问题无关，则忽略历史信息。
5. 今天日期是：{current_date}
"""

ANSWER_USER_PROMPT = """
【问题类型】
{intent}

【用户问题】
{user_input}

【相关历史记忆】
{memory_context}

【检索到的研报片段】
{context}

请基于上述材料作答。
如果历史记忆与当前检索材料不一致，以当前检索材料为准。
"""

CHAT_USER_PROMPT = """
【历史对话摘要】
{memory_context}

【用户问题】
{user_input}

请直接回答。
"""


def build_answer_node(history_manager):
    """
    answer_node 内部自行调用 history_manager 获取相关 memory，
    不依赖 state 中额外传 memory_context。
    """

    def answer_node(state: AgentState) -> Dict:
        user_input = state["user_input"]
        intent = state.get("intent", "rag_qa")
        session_id = state.get("session_id", "")
        docs = state.get("retrieved_docs", [])
        citations = state.get("citations") or []
        rewritten_query = state.get("rewritten_query", "")
        structured_query = state.get("structured_query", {}) or {}

        llm = get_llm()

        # answer阶段用于找相关memory的query，优先使用 rewrite 结果
        memory_query = _choose_memory_query(
            user_input=user_input,
            rewritten_query=rewritten_query,
            structured_query=structured_query,
            intent=intent,
        )

        if intent == "chat":
            relevant_memory = history_manager.get_relevant_memory(
                session_id=session_id,
                query=memory_query,
                top_k=1,
                min_score=0.30,
                search_turns=6,
                max_question_chars=80,
                max_answer_chars=220,
            )

            memory_context = _build_chat_memory_context(relevant_memory)

            messages = [
                SystemMessage(
                    content=CHAT_SYSTEM_PROMPT.format(
                        current_date=date.today().isoformat()
                    )
                ),
                HumanMessage(
                    content=CHAT_USER_PROMPT.format(
                        memory_context=memory_context or "无",
                        user_input=user_input,
                    )
                ),
            ]
        else:
            if not docs:
                return {
                    "answer": "根据当前检索到的研报内容，暂无足够证据回答该问题。",
                }

            # rag场景：只注入极少量相关memory，降低token消耗
            relevant_memory = history_manager.get_relevant_memory(
                session_id=session_id,
                query=memory_query,
                top_k=1,
                min_score=0.35,
                search_turns=6,
                max_question_chars=60,
                max_answer_chars=180,
            )

            memory_context = _build_rag_memory_context(relevant_memory)
            context = _build_context(docs)

            messages = [
                SystemMessage(
                    content=ANSWER_SYSTEM_PROMPT.format(
                        current_date=date.today().isoformat()
                    )
                ),
                HumanMessage(
                    content=ANSWER_USER_PROMPT.format(
                        intent=intent,
                        user_input=user_input,
                        memory_context=memory_context or "无",
                        context=context,
                    )
                ),
            ]

        response = llm.invoke(messages)
        answer_text = getattr(response, "content", str(response)).strip()

        if intent != "chat" and "来源" not in answer_text:
            answer_text = answer_text.rstrip() + "\n\n来源：\n" + _format_sources(citations)

        return {"answer": answer_text}

    return answer_node


def _choose_memory_query(
    user_input: str,
    rewritten_query: str,
    structured_query: Dict,
    intent: str,
) -> str:
    """
    选择 answer 阶段检索相关历史 memory 的查询词。
    尽量复用 rewrite 的结果，避免和 retrieval 分治冲突。
    """
    if intent == "chat":
        return rewritten_query or user_input

    task_type = structured_query.get("task_type", "")
    structured_rewritten_query = structured_query.get("rewritten_query", "")

    # 对复杂问题，优先使用结构化rewrite后的query
    if task_type and task_type != "fact":
        return structured_rewritten_query or rewritten_query or user_input

    # 普通问题
    return rewritten_query or user_input


def _build_chat_memory_context(memories: List[Dict]) -> str:
    """
    chat 模式下的历史摘要，允许稍微自然一点。
    """
    if not memories:
        return "无"

    lines = []
    for i, m in enumerate(memories, 1):
        lines.append(
            f"[历史{i}] 用户之前问：{m.get('question', '')}\n"
            f"[历史{i}] 之前回答：{m.get('answer', '')}"
        )
    return "\n\n".join(lines)


def _build_rag_memory_context(memories: List[Dict]) -> str:
    """
    rag 模式下的相关历史记忆，必须更短、更克制。
    """
    if not memories:
        return "无"

    lines = []
    for i, m in enumerate(memories, 1):
        lines.append(
            f"[记忆{i}] 历史问题：{m.get('question', '')}\n"
            f"[记忆{i}] 历史回答：{m.get('answer', '')}"
        )
    return "\n\n".join(lines)


def _build_context(docs: List[Dict], max_chars_per_doc: int = 500) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        content = d.get("content", "")
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."

        blocks.append(
            f"[片段{i}]\n"
            f"文件名: {d.get('file_name', '')}\n"
            f"页码: {d.get('page_num', '')}\n"
            f"内容: {content}\n"
        )
    return "\n".join(blocks)


def _format_sources(citations: List[Dict]) -> str:
    if not citations:
        return "- 无"

    lines = []
    seen = set()
    for c in citations:
        key = (c.get("file_name", ""), c.get("page_num", ""))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {c.get('file_name', '')}，第{c.get('page_num', '')}页")
    return "\n".join(lines)