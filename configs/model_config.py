# -*- coding: utf-8 -*-
"""
大模型 & 向量模型配置
支持本地模型 / API 模型切换
可扩展：增加更多模型厂商、多模型路由、量化配置等
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class ModelConfig:
    # ---------------------------
    # LLM 通用配置
    # ---------------------------
    # 运行模式: "api" 或 "local"
    LLM_MODE = os.getenv("LLM_MODE", "api")

    # API 模型配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
#####################################################################################还没有配置本地大模型
    # 本地大模型配置（后期扩展）
    LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "")
    LOCAL_LLM_DEVICE = os.getenv("LOCAL_LLM_DEVICE", "cuda")  # cuda / cpu
    LOCAL_LLM_DTYPE = "bfloat16"  # auto, float16, bfloat16, float32
#####################################################################################还没有配置本地大模型

    # ---------------------------
    # Embedding 模型配置
    # ---------------------------
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5"
    )
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")  # cuda / cpu
    EMBEDDING_NORMALIZE = True  # 是否归一化

    # ---------------------------
    # Reranker 模型配置
    # ---------------------------
    RERANKER_MODEL_NAME = os.getenv(
        "RERANKER_MODEL_NAME",
        "BAAI/bge-reranker-base"
    )
    RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")
    RERANKER_TOP_N = 10  # 重排后保留多少条
    RERANKER_MAX_LENGTH = 1024                   # 输入文本最大长度（越长越准，越慢）
    RERANKER_BATCH_SIZE = 32                     # 批处理大小（不爆显存就开大）

    # ---------------------------
    # 通用生成参数
    # ---------------------------
    GENERATION_TEMPERATURE = 0.1
    GENERATION_MAX_TOKENS = 2048
    GENERATION_TOP_P = 0.9


model_config = ModelConfig()