###########################################  测评函数  #####################################################
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from configs import rag_config
from rag import load_vector_store
from rag import create_hybrid_retriever
import json

# ==========================
# 🔥 核心：从缓存加载所有 Document（你 pipeline 已生成好）
# ==========================
def get_all_documents():
    cache_path = os.path.join(rag_config.VECTOR_STORE_DIR, "all_documents.json")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在！请先运行 rag_pipeline.py\n路径：{cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 直接转回 LangChain Document 格式
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in data
    ]

# ==========================
# 全流程最终测试（兼容版）
# ==========================
def test_rag_full_flow(
    query: str = "对比一下海光信息和上海航天汽车机电公司的财务表现。",
    search_mode: str = "hybrid",  # vector / bm25 / hybrid
    use_reranker: bool = True
):
    print("=" * 80)
    print("🔥  RAG 全流程最终兼容性测试（完全适配你的字典Loader）")
    print("=" * 80)

    try:
        # 1. 加载向量库
        print("🔹 加载向量库...")
        vs = load_vector_store(rag_config.VECTOR_STORE_DIR)

        # 2. 加载缓存好的全部 Document（给 BM25 使用）
        print("🔹 加载缓存文档（Document格式）...")
        all_docs = get_all_documents()
        print(f"✅ 总文档数量：{len(all_docs)}")

        # 3. 创建检索器
        print("🔹 初始化混合检索器...")
        search = create_hybrid_retriever(
            vectorstore=vs,
            all_documents=all_docs,
            score_threshold=0.3,
            device="cuda"  # 没有GPU就写 "cpu"
        )

        # 4. 执行检索
        print(f"\n🔍 查询：{query}")
        print(f"🧩 模式：{search_mode} | Reranker：{use_reranker}")
        
        docs = search(
            query=query,
            mode=search_mode,
            use_reranker=use_reranker
        )

        # 5. 输出最终给 LLM 的结果
        print(f"\n✅ 最终返回给LLM：{len(docs)} 条")
        print("=" * 80)

        for i, doc in enumerate(docs):
            print(f"\n【结果 {i+1}】")
            print(f"文件：{doc.metadata.get('source', '未知')}")
            print(f"页码：{doc.metadata.get('page_num', '未知')}")
            print(f"相关性分数：{doc.metadata.get('score', 'N/A')}")
            print(f"内容：{doc.page_content[:180]}...")
            print("-" * 70)

        print("\n🎉 全流程兼容测试完成 ✅")
        print("✅ 你的 RAG 系统：字典 → Document → 召回 → 精排 完全打通！")
        print(docs[0].metadata)
        print(docs[0])
    except Exception as e:
        print(f"\n❌ 出错：{e}")
        import traceback
        traceback.print_exc()

# ==========================
# 运行
# ==========================
if __name__ == "__main__":
    test_rag_full_flow(
        query="海光信息2024年净利润是多少？",
        search_mode="hybrid",
        use_reranker=True
    )