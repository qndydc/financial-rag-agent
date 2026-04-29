# -*- coding: utf-8 -*-
"""
LLM 工厂模块

根据 model_config.LLM_MODE 自动切换 API 模式 / 本地模型。
所有 Agent 组件统一通过 get_llm() 获取 LLM 实例，便于统一维护。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from configs import model_config


def get_llm(
    temperature: float = None,
    max_tokens: int = None,
    streaming: bool = False,
) -> BaseChatModel:
    """
    LLM 工厂函数，返回 LangChain 兼容的 ChatModel 实例。

    Args:
        temperature: 生成温度，None 时使用 model_config 默认值
        max_tokens:  最大生成 Token 数，None 时使用 model_config 默认值
        streaming:   是否启用流式输出（用于 stream_chat）

    Returns:
        BaseChatModel 实例（已绑定 API Key 和 Base URL）
    """
    temp   = temperature if temperature is not None else model_config.GENERATION_TEMPERATURE
    tokens = max_tokens  if max_tokens  is not None else model_config.GENERATION_MAX_TOKENS

    if model_config.LLM_MODE == "api":
        return ChatOpenAI(
            model=model_config.LLM_MODEL_NAME,
            temperature=temp,
            max_tokens=tokens,
            streaming=streaming,
            openai_api_key=model_config.OPENAI_API_KEY,
            openai_api_base=model_config.OPENAI_BASE_URL,
        )

    elif model_config.LLM_MODE == "local":
        # 预留本地模型接口（vLLM / Ollama / HuggingFace Pipeline）
        raise NotImplementedError(
            "本地大模型尚未集成。\n"
            "可选方案：\n"
            "  1. 设置 LLM_MODE=api 并填写 OPENAI_API_KEY\n"
            "  2. 在此处接入 vLLM / Ollama 的 OpenAI 兼容接口\n"
            "  3. 使用 langchain_community.llms.HuggingFacePipeline"
        )

    else:
        raise ValueError(
            f"不支持的 LLM_MODE：'{model_config.LLM_MODE}'，"
            "请在 .env 中设置 LLM_MODE=api 或 LLM_MODE=local"
        )
