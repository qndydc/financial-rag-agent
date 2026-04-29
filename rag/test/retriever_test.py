###########################################  检索测评函数  #####################################################
import os
import sys
import json
import time
import traceback
from typing import List, Dict, Any
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


from langchain_core.documents import Document
from configs import rag_config
from rag import load_vector_store
from rag import create_hybrid_retriever


# ==========================
# 从缓存加载所有 Document
# ==========================
def get_all_documents():
    cache_path = os.path.join(rag_config.VECTOR_STORE_DIR, "all_documents.json")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在！请先运行 rag_pipeline.py\n路径：{cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in data
    ]


# ==========================
# 读取测试集
# ==========================
def load_eval_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"测试集不存在：{dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("测试集格式错误：最外层必须是 list")

    for item in data:
        if "query" not in item or "gt" not in item:
            raise ValueError(f"测试集样本缺少 query 或 gt 字段：{item}")

    return data


# ==========================
# 从 Document 中提取 chunk_id
# 根据你的 metadata 字段名改这里即可
# ==========================
def get_chunk_id(doc: Document) -> str:
    metadata = doc.metadata

    # 优先读取你显式保存的 chunk_id
    if "chunk_id" in metadata:
        return str(metadata["chunk_id"])

    # 如果没有 chunk_id，可以临时用 source + page_num + chunk_index 拼一个
    source = metadata.get("source", "unknown_source")
    page_num = metadata.get("page_num", metadata.get("page", "unknown_page"))
    chunk_index = metadata.get("chunk_index", metadata.get("chunk_id", "unknown_chunk"))

    return f"{source}::page_{page_num}::chunk_{chunk_index}"


# ==========================
# 计算单条样本指标
# ==========================
def compute_single_metrics(retrieved_docs: List[Document], gt_chunk_ids: List[str]):
    retrieved_chunk_ids = [get_chunk_id(doc) for doc in retrieved_docs]

    gt_set = set(gt_chunk_ids)
    retrieved_set = set(retrieved_chunk_ids)

    hit_gt = gt_set & retrieved_set

    hit_count_for_precision = sum(
        1 for cid in retrieved_chunk_ids if cid in gt_set
    )

    recall = len(hit_gt) / len(gt_set) if len(gt_set) > 0 else 0.0
    precision = hit_count_for_precision / len(retrieved_chunk_ids) if len(retrieved_chunk_ids) > 0 else 0.0
    noise_ratio = 1.0 - precision if len(retrieved_chunk_ids) > 0 else 0.0

    return {
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "hit_gt": list(hit_gt),
        "hit_count": len(hit_gt),
        "gt_count": len(gt_set),
        "retrieved_count": len(retrieved_chunk_ids),
        "recall": recall,
        "precision": precision,
        "noise_ratio": noise_ratio,
    }


# ==========================
# 批量检索评测主函数
# ==========================
def evaluate_retriever(
    dataset_path: str,
    search_mode: str = "hybrid",      # vector / bm25 / hybrid
    use_reranker: bool = True,
    score_threshold: float = 0.3,
    device: str = "cuda",
    save_detail_path: str = None
):
    print("=" * 90)
    print("🔥 RAG 检索模块批量测评")
    print("=" * 90)
    print(f"检索模式：{search_mode}")
    print(f"是否使用 reranker：{use_reranker}")
    print(f"测试集路径：{dataset_path}")

    try:
        # 1. 加载测试集
        eval_data = load_eval_dataset(dataset_path)
        print(f"✅ 测试样本数量：{len(eval_data)}")

        # 2. 加载向量库
        print("\n🔹 加载向量库...")
        vs = load_vector_store(rag_config.VECTOR_STORE_DIR)

        # 3. 加载全部文档
        print("🔹 加载缓存文档...")
        all_docs = get_all_documents()
        print(f"✅ 总文档数量：{len(all_docs)}")

        # 4. 创建检索器
        print("🔹 初始化检索器...")
        search = create_hybrid_retriever(
            vectorstore=vs,
            all_documents=all_docs,
            score_threshold=score_threshold,
            device=device
        )

        all_recalls = []
        all_precisions = []
        all_noise_ratios = []
        all_latencies = []

        detail_results = []

        # 5. 循环评测
        for idx, item in enumerate(eval_data, start=1):
            query = item["query"]
            gt = item["gt"]

            print("\n" + "-" * 90)
            print(f"样本 {idx}/{len(eval_data)}")
            print(f"Query: {query}")
            print(f"GT数量: {len(gt)}")

            start_time = time.perf_counter()

            docs = search(
                query=query,
                mode=search_mode,
                use_reranker=use_reranker
            )

            latency = time.perf_counter() - start_time

            metrics = compute_single_metrics(docs, gt)

            all_recalls.append(metrics["recall"])
            all_precisions.append(metrics["precision"])
            all_noise_ratios.append(metrics["noise_ratio"])
            all_latencies.append(latency)

            print(f"召回文档数：{metrics['retrieved_count']}")
            print(f"命中GT数：{metrics['hit_count']} / {metrics['gt_count']}")
            print(f"Recall：{metrics['recall']:.4f}")
            print(f"Precision：{metrics['precision']:.4f}")
            print(f"Noise Ratio：{metrics['noise_ratio']:.4f}")
            print(f"Latency：{latency:.4f} 秒")

            detail_results.append({
                "query": query,
                "gt": gt,
                "retrieved_chunk_ids": metrics["retrieved_chunk_ids"],
                "hit_gt": metrics["hit_gt"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "noise_ratio": metrics["noise_ratio"],
                "latency": latency,
                "search_mode": search_mode,
                "use_reranker": use_reranker,
            })

        # 6. 汇总结果
        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
        avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
        avg_noise_ratio = sum(all_noise_ratios) / len(all_noise_ratios) if all_noise_ratios else 0.0
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

        print("\n" + "=" * 90)
        print("📊 检索测评汇总")
        print("=" * 90)
        print(f"Search Mode：{search_mode}")
        print(f"Use Reranker：{use_reranker}")
        print(f"样本数量：{len(eval_data)}")
        print(f"平均 Recall：{avg_recall:.4f}")
        print(f"平均 Precision：{avg_precision:.4f}")
        print(f"平均 Noise Ratio：{avg_noise_ratio:.4f}")
        print(f"平均检索延迟：{avg_latency:.4f} 秒")
        print("=" * 90)

        summary = {
            "search_mode": search_mode,
            "use_reranker": use_reranker,
            "sample_count": len(eval_data),
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "avg_noise_ratio": avg_noise_ratio,
            "avg_latency": avg_latency,
            "details": detail_results,
        }

        return summary

    except Exception as e:
        print(f"\n❌ 测评出错：{e}")
        traceback.print_exc()


# ==========================
# 运行入口
# ==========================
if __name__ == "__main__":
    evaluate_retriever(
        dataset_path="d:\\Aprojet\\py\\RAG_Agent\\financial_rag_agent\\rag\\test\\test_dataset\\retrieval_gt_dataset_dif.json",
        search_mode="hybrid", #vector / bm25 / hybrid
        use_reranker=True,
        score_threshold=0.3,
        device="cuda",
    )