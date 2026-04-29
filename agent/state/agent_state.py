# -*- coding: utf-8 -*-
from typing import Annotated, Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    金融 RAG Agent 全局状态
    """

    # LangGraph 消息流（保留，便于后续扩展）
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 输入
    user_input: str
    session_id: str

    # 多轮上下文
    chat_history: List[BaseMessage]

    # 路由 NEW
    intent: str   # rag_qa / chat / unclear

    # 检索 query
    rewritten_query: str
    retrieve_queries: List[str]
    structured_query: Dict[str, Any]

    # 检索结果
    retrieved_docs: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]

    # 检索质量控制 NEW 
    retrieval_success: bool
    fallback_reason: str

    # 重试控制 NEW
    retry_count: int
    max_retries: int

    # 最终输出
    answer: str

    # 调试
    debug_info: Dict[str, Any]
    error: Optional[str]