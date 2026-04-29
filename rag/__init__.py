# -*- coding: utf-8 -*-
"""
配置模块统一出口
方便其他模块直接导入：
from configs import model_config, rag_config
"""

# rag/__init__.py

# ==========================
# 向量库相关
# ==========================
from .embeddings.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
)


# ==========================
# 检索模块
# ==========================
from .retrievers.vector_retriever import create_vector_retriever
from .retrievers.bm25_retriever import create_bm25_retriever
from .retrievers.reranker import LangchainReranker
from .retrievers.hybrid_retriever import create_hybrid_retriever

# ==========================
# 暴露给外部使用的接口
# ==========================
__all__ = [
    # 向量库
    "create_vector_store",
    "save_vector_store",
    "load_vector_store", #就这个有用


    # 检索
    "create_vector_retriever",
    "create_bm25_retriever",
    "LangchainReranker",
    "create_hybrid_retriever", #就这个有用
]