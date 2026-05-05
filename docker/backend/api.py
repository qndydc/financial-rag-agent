# -*- coding: utf-8 -*-
# api.py  【修复完整可运行版】
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from typing import Optional, AsyncGenerator, Dict, Any

# 直接导入你的 Agent
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # 将上级目录加入路径
from agent.orchestrator import FinancialRAGAgent

# ======================================
# 1. 初始化 FastAPI app
# ======================================
app = FastAPI(title="金融 RAG Agent API", version="1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# 2. 全局初始化 Agent
# ======================================
print("[API] 正在初始化 RAG Agent...")
agent = FinancialRAGAgent()
print("[API] 初始化完成！")

# ======================================
# 3. 请求体结构
# ======================================
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

# ======================================
# 4. 普通对话接口
# ======================================
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        answer = agent.chat(request.query, session_id=request.session_id)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================
# 5. 🔥 SSE 流式对话接口
# ======================================
@app.post("/api/stream/chat")
async def stream_chat(request: ChatRequest):
    try:
        async def event_generator():
            async for chunk in agent.stream_chat(
                request.query,
                session_id=request.session_id
            ):
                # ✅ 必须加这两行，过滤 None，防止崩溃
                if chunk is None:
                    continue
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        return EventSourceResponse(event_generator())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================
# 6. 清空历史
# ======================================
@app.post("/api/clear_history")
async def clear_history(request: ChatRequest):
    agent.clear_history(request.session_id)
    return {"status": "success"}

# ======================================
# 【正确启动】
# ======================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )