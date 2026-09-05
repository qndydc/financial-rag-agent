# -*- coding: utf-8 -*-
import operator
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

    # 分类恢复预算：临时错误由 call_lifecycle 在单次调用内处理
    argument_repair_count: int
    generalization_count: int
    argument_repair_limit: int
    generalization_limit: int

    # 调用生命周期记录
    observations: Annotated[List[Dict[str, Any]], operator.add]
    last_observation: Dict[str, Any]
    recovery_action: str

    # 最终输出
    answer: str

    # 调试
    debug_info: Dict[str, Any]
    error: Optional[str]
