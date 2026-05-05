import re
from langchain_core.documents import Document
from typing import List
from langchain_classic.retrievers import EnsembleRetriever

# ------------------------------------------------------------------------------
# ✅ 关键优化：只导入类型，不初始化模型
# ------------------------------------------------------------------------------
from typing import Optional
from .vector_retriever import create_vector_retriever
from .bm25_retriever import create_bm25_retriever

# ✅ Reranker 只在第一次检索时加载（懒加载！）
from .reranker import LangchainReranker

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs import rag_config
from configs import model_config

VECTOR_SEARCH_TOP_K = rag_config.VECTOR_SEARCH_TOP_K
SCORE_THRESHOLD = rag_config.SCORE_THRESHOLD
BM25_SEARCH_TOP_K = rag_config.BM25_SEARCH_TOP_K
RERANKER_MODEL_NAME = model_config.RERANKER_MODEL_NAME
RERANKER_DEVICE = model_config.RERANKER_DEVICE
RERANKER_TOP_N = model_config.RERANKER_TOP_N
RERANKER_MAX_LENGTH = model_config.RERANKER_MAX_LENGTH
RERANKER_BATCH_SIZE = model_config.RERANKER_BATCH_SIZE
FINAL_RETURN_CHUNK_COUNT = rag_config.FINAL_RETURN_CHUNK_COUNT
HYBRID_SEARCH_WEIGHT_VECTOR = rag_config.HYBRID_SEARCH_WEIGHT_VECTOR
HYBRID_SEARCH_WEIGHT_BM25 = rag_config.HYBRID_SEARCH_WEIGHT_BM25

# ------------------------------------------------------------------------------
# 懒加载检索器 + 懒加载重排模型
# ------------------------------------------------------------------------------
def create_hybrid_retriever(
    vectorstore,
    all_documents: List[Document],
    score_threshold=SCORE_THRESHOLD,
    reranker_model_path: str = RERANKER_MODEL_NAME,
    device=RERANKER_DEVICE
):
    # 这些创建极快，不耗时
    vector_ret = create_vector_retriever(vectorstore, top_k=VECTOR_SEARCH_TOP_K, score_threshold=score_threshold)
    print("vector初始化完成")
    bm25_ret = create_bm25_retriever(all_documents, top_k=BM25_SEARCH_TOP_K)
    print("bm25初始化完成")
    hybrid_ret = EnsembleRetriever(retrievers=[vector_ret, bm25_ret], weights=[HYBRID_SEARCH_WEIGHT_VECTOR, HYBRID_SEARCH_WEIGHT_BM25])
    print("hybrid初始化完成")

    # ====================== ✅ 核心懒加载 ======================
    # reranker 只在第一次真正检索时才加载
    # ==========================================================
    reranker = None

    def get_reranker():
        nonlocal reranker
        if reranker is None:
            reranker = LangchainReranker(
                model_name_or_path=reranker_model_path,
                top_n=RERANKER_TOP_N,
                device=device,
                max_length=RERANKER_MAX_LENGTH,
                batch_size=RERANKER_BATCH_SIZE,
            )
        return reranker

    # --------------------------------------------------------------------------
    # 搜索函数：第一次调用才加载模型
    # --------------------------------------------------------------------------
    def search(
        query: str,
        mode: str = "hybrid",
        use_reranker: bool = True
    ) -> List[Document]:

        # 1. 检索
        if mode == "vector":
            docs = vector_ret.invoke(query)
        elif mode == "bm25":
            docs = bm25_ret.invoke(query)
        elif mode == "hybrid":
            docs = hybrid_ret.invoke(query)
        else:
            raise ValueError(f"不支持的检索模式：{mode}")

        # 2. 重排（第一次才加载模型）
        if use_reranker:
            reranker = get_reranker()
            docs = reranker.compress_documents(docs, query)

        final_docs = docs[:FINAL_RETURN_CHUNK_COUNT]
        return final_docs

    return search
