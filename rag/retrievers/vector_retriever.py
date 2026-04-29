from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from langchain_core.documents import Document
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs import rag_config

VECTOR_SEARCH_TOP_K = rag_config.VECTOR_SEARCH_TOP_K
SCORE_THRESHOLD = rag_config.SCORE_THRESHOLD

# 自定义向量检索器 → 继承 LangChain 标准 VectorStoreRetriever
class VectorRetriever(VectorStoreRetriever):
    # 重写核心检索方法（LangChain 规定必须实现的接口）
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # 如果使用【相似度分数阈值】模式
        if self.search_type == "similarity_score_threshold":
            # 执行检索：返回 (文档, 相似度分数)
            docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs)
            #定义阈值
            threshold = self.search_kwargs.get("score_threshold", SCORE_THRESHOLD)
            return [d for d, s in docs_and_scores if s >= threshold]
        return self.vectorstore.similarity_search(query, **self.search_kwargs)
# 工厂函数：一键创建向量检索器（解耦、统一入口）
def create_vector_retriever(vectorstore, top_k=VECTOR_SEARCH_TOP_K, score_threshold=SCORE_THRESHOLD):
    return VectorRetriever(
        vectorstore=vectorstore,
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": score_threshold}
    )