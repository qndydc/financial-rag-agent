import re
import pickle
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever  # 关键
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from configs import rag_config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

VECTOR_SEARCH_TOP_K = rag_config.VECTOR_SEARCH_TOP_K
# 配置
TOP_K = VECTOR_SEARCH_TOP_K
BM25_INDEX_PATH = Path(__file__).parent / "bm25_okapi_index.pkl"

## ==========================
# 超快中文分词（无 jieba）
# ==========================
def fast_chinese_tokenize(text: str) -> List[str]:
    text = text.replace("\n", " ").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fa5]|[0-9]+", text)
    return [t for t in tokens if t]

# ==========================
# 🔥 关键：继承 LangChain BaseRetriever
# 这样才能放进 EnsembleRetriever
# ==========================
class BM25Retriever(BaseRetriever):
    bm25: BM25Okapi
    documents: List[Document]
    top_k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        tokens = fast_chinese_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(scores, self.documents), key=lambda x: -x[0])
        return [doc for _, doc in ranked[:self.top_k]]

# ==========================
# 保存 / 加载
# ==========================
def save_bm25(bm25, docs, path=BM25_INDEX_PATH):
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)

def load_bm25(path=BM25_INDEX_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["documents"]

# ==========================
# 创建函数（你原来的调用方式不变）
# ==========================
def create_bm25_retriever(documents: List[Document], top_k=TOP_K, force_rebuild=False):
    if not force_rebuild and BM25_INDEX_PATH.exists():
        bm25, docs = load_bm25()
        return BM25Retriever(bm25=bm25, documents=docs, top_k=top_k)

    texts = [d.page_content for d in documents]
    tokenized = [fast_chinese_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    save_bm25(bm25, documents)

    return BM25Retriever(bm25=bm25, documents=documents, top_k=top_k)