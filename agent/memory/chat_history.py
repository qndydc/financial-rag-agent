# -*- coding: utf-8 -*-
"""
多会话对话历史管理

基于内存存储，按 session_id 隔离。
生产环境可替换为 Redis / 数据库持久化方案。
"""
from typing import Dict, List, Any
from difflib import SequenceMatcher
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ChatHistoryManager:
    """
    多路并发安全的对话历史管理器。

    使用滑动窗口限制每个会话的历史长度，
    避免长对话导致 Token 超限。
    """

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: 保留最近 N 轮对话（1 轮 = 1条 Human + 1条 AI）
        """
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self.max_turns = max_turns

    # ──────────────────────────────────────────
    # 基本操作
    # ──────────────────────────────────────────

    def get(self, session_id: str) -> List[BaseMessage]:
        """获取指定会话的历史消息列表（不包含系统消息）。"""
        return list(self._sessions.get(session_id, []))

    def add(self, session_id: str, messages: List[BaseMessage]) -> None:
        """
        向指定会话追加消息，并执行滑动窗口截断。

        Args:
            session_id: 会话唯一标识符
            messages:   本轮新增消息（通常是 [HumanMessage, AIMessage]）
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].extend(messages)

        # 滑动窗口：保留最近 max_turns 轮（成对截断，避免切断对话语义）
        max_messages = self.max_turns * 2
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]

    def clear(self, session_id: str) -> None:
        """清除指定会话的所有历史。"""
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """清除所有会话历史。"""
        self._sessions.clear()

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────

    def list_sessions(self) -> List[str]:
        """返回当前所有活跃会话 ID 列表。"""
        return list(self._sessions.keys())

    def session_length(self, session_id: str) -> int:
        """返回指定会话的消息条数。"""
        return len(self._sessions.get(session_id, []))

    def get_last_n_turns(self, session_id: str, n: int) -> List[BaseMessage]:
        """获取最近 n 轮对话（2n 条消息）。"""
        history = self._sessions.get(session_id, [])
        return history[-(n * 2):]

    # ──────────────────────────────────────────
    # 新增：answer 内部使用的轻量相关记忆提取
    # ──────────────────────────────────────────

    def get_relevant_memory(
        self,
        session_id: str,
        query: str,
        top_k: int = 1,
        min_score: float = 0.35,
        search_turns: int = 6,
        max_question_chars: int = 80,
        max_answer_chars: int = 220,
    ) -> List[Dict[str, Any]]:
        """
        从最近若干轮对话中抽取最相关的少量历史 QA。
        用于 answer 节点临时拼 prompt，不写回 state。

        Returns:
            [
                {
                    "question": "...",
                    "answer": "...",
                    "score": 0.78
                }
            ]
        """
        history = self.get_last_n_turns(session_id, search_turns)
        if not history:
            return []

        qa_pairs: List[Dict[str, str]] = []
        i = 0
        while i < len(history) - 1:
            if isinstance(history[i], HumanMessage) and isinstance(history[i + 1], AIMessage):
                q = self._safe_text(history[i].content)
                a = self._safe_text(history[i + 1].content)
                qa_pairs.append({"question": q, "answer": a})
                i += 2
            else:
                i += 1

        if not qa_pairs:
            return []

        normalized_query = self._normalize(query)
        scored = []

        for qa in qa_pairs:
            q = qa["question"]
            a = qa["answer"]

            score_q = self._similarity(normalized_query, self._normalize(q))
            score_a = self._similarity(normalized_query, self._normalize(a[:300]))
            final_score = max(score_q, score_a * 0.8)

            if final_score >= min_score:
                scored.append(
                    {
                        "question": self._truncate(q, max_question_chars),
                        "answer": self._truncate(a, max_answer_chars),
                        "score": round(final_score, 4),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ──────────────────────────────────────────
    # 内部函数
    # ──────────────────────────────────────────

    def _safe_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return str(content)

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."