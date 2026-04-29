import jieba
# ====================== 🔥 修复在这里 ======================
# 新版 BM25 移到了 langchain_community
from langchain_community.retrievers import BM25Retriever
# ==========================================================
from langchain_core.documents import Document
from typing import List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs import rag_config

BM25_SEARCH_TOP_K = rag_config.BM25_SEARCH_TOP_K

def chinese_tokenizer(text: str):
    text = text.replace("\n", " ")
    return list(jieba.cut_for_search(text))

def create_bm25_retriever(documents: List[Document], top_k=BM25_SEARCH_TOP_K):
    retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=chinese_tokenizer
    )
    retriever.k = top_k
    return retriever