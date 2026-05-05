# -*- coding: utf-8 -*-
"""
金融 RAG Agent 主编排器（增强骨架版）

Graph 结构：

    START
      ↓
  load_history
      ↓
     route
   ┌────┼───────┐
   │    │       │
   │    │       └──────────────→ fallback
   │    │
   │    └──────────────→ answer   (chat)
   │
   └→ rewrite → retrieve → judge
                           ├─ answer
                           ├─ rewrite (retry)
                           └─ fallback
      ↓
  save_history
      ↓
     END
"""
import json
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from configs import rag_config
from rag import load_vector_store, create_hybrid_retriever

from agent.state.agent_state import AgentState
from agent.llm.base_llm import get_llm
from agent.tools.rag_tool import create_rag_search_adapter
from agent.tools.structured_rag_tool import create_structured_rag_search_adapter
from agent.memory.chat_history import ChatHistoryManager

from agent.nodes.rewrite_node import rewrite_node, increment_retry_node
from agent.nodes.retrieve_node import build_retrieve_node
from agent.nodes.answer_node import build_answer_node
from agent.nodes.route_node import route_node
from agent.nodes.judge_node import judge_node
from agent.nodes.fallback_node import fallback_node

import asyncio
from typing import AsyncGenerator, Dict, Any

class FinancialRAGAgent:
    """
    金融研报 RAG Agent
    """

    def __init__(
        self,
        vector_store_path: str = None,
        all_docs_json_path: str = None,
        max_turns: int = 10,
        max_retries: int = 1,
    ):
        vs_path = vector_store_path or rag_config.VECTOR_STORE_DIR
        docs_path = all_docs_json_path or f"{rag_config.VECTOR_STORE_DIR}/all_documents.json"

        # 1. 加载 RAG
        print("[Agent] 正在加载向量库...")
        vectorstore = load_vector_store(vs_path)
        all_documents = self._load_all_documents(docs_path)
        self.search_fn = create_hybrid_retriever(
            vectorstore, all_documents
        )  # 这里search_fn已经变成了内部函数search，封装了向量检索、BM25检索和Reranker的完整流程
        print(f"[Agent] 向量库加载完成，共 {len(all_documents)} 个文档块。")

        # 2. 创建 RAG Adapter
        self.rag_adapter = create_rag_search_adapter(self.search_fn)  # 将底层search函数封装成python接口
        self.structured_rag_adapter = create_structured_rag_search_adapter(self.search_fn)  # 为结构化检索创建独立的适配器

        # 3. 对话历史
        self.memory = ChatHistoryManager(max_turns=max_turns)

        # 4. 默认 LLM（节点内也可直接 get_llm）
        self.llm = get_llm()
        self.max_retries = max_retries

        # 5. 构建图
        self.graph = self._build_graph()
        print("[Agent] 初始化完成，Agent 就绪。")

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # --- node 1: load_history ---
        def load_history_node(state: AgentState) -> dict:
            session_id = state["session_id"]
            user_input = state["user_input"]

            history = self.memory.get_last_n_turns(
                session_id, n=3
            )  # 从内存中获取历史对话，返回一个消息列表（不包含系统消息）,不考虑轮数
            messages = history + [HumanMessage(content=user_input)]

            print(f"[load_history_node] session_id = {session_id}")
            print(f"[load_history_node] history_len = {len(history)}")

            return {
                "chat_history": history,
                "messages": messages,
                "retry_count": state.get("retry_count", 0),
                "max_retries": state.get("max_retries", self.max_retries),
                "debug_info": {
                    **state.get("debug_info", {}),
                    "history_count": len(history),
                }
            }

        # ---node 2：save_history, 在 answer_node 和 fallback_node 之后执行，负责把本轮对话（用户输入 + 模型回答）保存到历史中
        def save_history_node(state: AgentState) -> dict:
            session_id = state["session_id"]
            user_input = state["user_input"]
            answer = state.get("answer", "").strip()

            self.memory.add(
                session_id,
                [
                    HumanMessage(content=user_input),
                    AIMessage(content=answer),
                ]
            )

            return {}

        # --- other node ---
        retrieve_node = build_retrieve_node(self.rag_adapter, self.structured_rag_adapter)  # 通过闭包把 rag_adapter 注入节点
        answer_node = build_answer_node(self.memory)

        # 先加节点
        workflow.add_node("load_history", load_history_node)
        workflow.add_node("route", route_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("increment_retry", increment_retry_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("judge", judge_node)
        workflow.add_node("ans", answer_node)
        workflow.add_node("fallback", fallback_node)
        workflow.add_node("save_history", save_history_node)

        #                   |<--------  retry  --------|
        # load_his -> route |-> rewrite -> retrieve -> |judge
        #                   |----------  chat  ------->|->ans      |--> save_his
        #                   |-------- bad query -------|->fallback |

        # 再定义节点间的边（顺序执行）
        workflow.set_entry_point("load_history")
        workflow.add_edge("load_history", "route")
        workflow.add_conditional_edges(
            "route",
            self._route_after_route,  # 根据 route_node 的输出 intent 决定下一步走向，chat 直接答，unclear 直接 fallback，rag_qa 走 rewrite
            {
                "chat": "ans",      # chat模式直接走回答节点，不检索，不涉及query结构重写
                "unclear": "fallback",
                "rag_qa": "rewrite",  # 涉及到qa才走后续节点多次召回
            }
        )

        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "judge")
        workflow.add_conditional_edges(
            "judge",
            self._route_after_judge,  # 根据 judge_node 的输出 retrieval_success 和 retry_count 决定下一步走向，
            {                         # 成功直接答，失败且未超重试次数走 rewrite 重试，失败且超重试次数走 fallback
                "ans": "ans",
                "rewrite": "increment_retry",
                "fallback": "fallback",
            }
        )

        workflow.add_edge("increment_retry", "rewrite")
        workflow.add_edge("ans", "save_history")
        workflow.add_edge("fallback", "save_history")
        workflow.add_edge("save_history", END)

        return workflow.compile()

    def _route_after_route(self, state: AgentState) -> str:
        return state.get("intent", "rag_qa")

    def _route_after_judge(self, state: AgentState) -> str:
        retreival_success = state.get("retrieval_success", False)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.max_retries)

        if retreival_success:
            return "ans"
        if retry_count < max_retries:
            return "rewrite"
        return "fallback"

    def chat(self, user_input: str, session_id: str = "default") -> str:
        result = self.graph.invoke(
            {
                "user_input": user_input,
                "session_id": session_id,
                "retry_count": 0,
                "max_retries": self.max_retries,
                "debug_info": {},
            }
        )

        answer = result.get("answer", "抱歉，未能生成回答。")
        return answer

    def clear_history(self, session_id: str = "default") -> None:
        self.memory.clear(session_id)

    def list_sessions(self):
        return self.memory.list_sessions()

    @staticmethod
    def _load_all_documents(json_path: str) -> List[Document]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(
                page_content=d["page_content"],
                metadata=d.get("metadata", {}),
            )
            for d in data
        ]
    
    # ------------------------------
    # 【新增】流式输出 支持：答案 + 引用 + 证据】
    # ------------------------------


    # 【后端最终版 SSE 输出，
    async def stream_chat(self, user_input: str, session_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
        final_answer = ""
        evidences = []

        async for event in self.graph.astream({
            "user_input": user_input,
            "session_id": session_id,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "debug_info": {},
        }):
            node_name = list(event.keys())[0]
            data = event[node_name]

            if data is None:
                continue

            # 1. 流式输出答案
            if "answer" in data and data["answer"]:
                yield {
                    "type": "answer",
                    "content": data["answer"]
                }

            # 2. 输出证据（检索到的 chunks）
            if "contexts" in data and data["contexts"]:
                evidences = []
                for idx, doc in enumerate(data["contexts"]):
                    evidences.append({
                        "id": idx,
                        "page": doc.metadata.get("page", "未知"),
                        "pdf": doc.metadata.get("source", "文档"),
                        "content": doc.page_content,
                        "score": doc.metadata.get("score", 0.0)
                    })
                yield {
                    "type": "evidences",
                    "list": evidences
                }
                
    def eval_chat(self, user_input: str, session_id: str = "ragas_eval") -> dict:
        """
        Ragas 评测专用接口。

        返回：
        - answer: 最终答案
        - contexts: 检索到的 chunk 文本列表
        - context_docs: 原始 Document 列表
        - intent / rewritten_query / retrieval_success / debug_info: 调试字段
        """
        result = self.graph.invoke(
            {
                "user_input": user_input,
                "session_id": session_id,
                "retry_count": 0,
                "max_retries": self.max_retries,
                "debug_info": {},
            }
        )

        context_docs = result.get("contexts", []) or []

        return {
            "answer": result.get("answer", "抱歉，未能生成回答。"),
            "contexts": [doc.page_content for doc in context_docs],
            "context_docs": context_docs,
            "intent": result.get("intent"),
            "rewritten_query": result.get("rewritten_query"),
            "retrieval_success": result.get("retrieval_success"),
            "debug_info": result.get("debug_info", {}),
        }


if __name__ == "__main__":
    """
    本地调试入口：
    直接运行 python agent/orchestrator.py
    """

    import traceback
    import time

    print("=" * 80)
    print("FinancialRAGAgent 本地调试启动")
    print("=" * 80)

    try:
        agent = FinancialRAGAgent()  # 实例化 Agent，加载向量库和模型，构建图create graph

        test_session_id = "debug_session"

        # 你可以先放几个固定测试问题
        test_questions = [
            "海光信息2024年净利润是多少？",
            "对比一下海光信息和上海航天汽车机电公司的财务表现",
            "那同比增速呢？",
            "你好，你是什么模型？请介绍你的版本",
        ]

        for i, question in enumerate(test_questions, 1):
            print("\n" + "-" * 80)
            print(f"[Round {i}] 用户问题: {question}")
            print("-" * 80)

            answer = agent.chat(question, session_id=test_session_id)  # 调用chat交给大模型输出答案

            print("[Agent Answer]")
            print(answer)

            time.sleep(3)

        print("\n" + "=" * 80)
        print("多轮固定问题测试完成")
        print("=" * 80)

        # 可选：进入交互式调试
        print("\n进入交互模式，输入 quit / exit 退出。\n")

        while True:
            user_input = input("User> ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("已退出调试模式。")
                break

            if not user_input:
                continue

            answer = agent.chat(user_input, session_id=test_session_id)

            print("\nAssistant>")
            print(answer)
            print()

    except Exception as e:
        print("\n[ERROR] Agent 调试运行失败")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print("\n详细堆栈如下：")
        traceback.print_exc()