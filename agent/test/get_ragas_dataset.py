# -*- coding: utf-8 -*-

"""
读取 agent/test/testset.json，
用 FinancialRAGAgent 全流程跑 query，
生成 Ragas 评测数据集：

(query, answer, retrieved_contexts, reference)

运行：
python -m agent.test.build_ragas_dataset
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from agent.orchestrator import FinancialRAGAgent


CURRENT_DIR = Path(__file__).resolve().parent

INPUT_PATH = CURRENT_DIR / "testset.json"
OUTPUT_PATH = CURRENT_DIR / "ragas_dataset.json"


def load_testset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("testset.json 必须是 JSON list 格式")

    return data


def save_json(data: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_query(item: Dict[str, Any]) -> str:
    return item.get("query") or item.get("question") or item.get("user_input")


def get_reference(item: Dict[str, Any]) -> str:
    return item.get("reference") or item.get("answer") or item.get("gt")


def build_ragas_dataset(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> List[Dict[str, Any]]:
    testset = load_testset(input_path)

    print("=" * 80)
    print(f"加载测试集：{input_path}")
    print(f"样本数量：{len(testset)}")
    print("=" * 80)

    agent = FinancialRAGAgent()

    ragas_dataset = []

    for idx, item in enumerate(testset, start=1):
        query = get_query(item)
        reference = get_reference(item)

        if not query:
            print(f"[Skip] 第 {idx} 条缺少 query")
            continue

        if not reference:
            print(f"[Skip] 第 {idx} 条缺少 reference")
            continue

        session_id = f"ragas_eval_{uuid.uuid4().hex}"

        print("\n" + "-" * 80)
        print(f"[{idx}/{len(testset)}] Query: {query}")

        result = agent.eval_chat(
            user_input=query,
            session_id=session_id,
        )

        answer = result.get("answer", "")
        retrieved_contexts = result.get("contexts", [])

        row = {
            "query": query,
            "answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,

            # 可选调试字段
            "intent": result.get("intent"),
            "rewritten_query": result.get("rewritten_query"),
            "retrieval_success": result.get("retrieval_success"),
            "context_count": len(retrieved_contexts),
        }

        ragas_dataset.append(row)

        print(f"Answer: {answer[:200]}")
        print(f"Retrieved contexts: {len(retrieved_contexts)}")

    save_json(ragas_dataset, output_path)

    print("\n" + "=" * 80)
    print("Ragas dataset 构建完成")
    print(f"输出文件：{output_path}")
    print(f"有效样本数：{len(ragas_dataset)}")
    print("=" * 80)

    return ragas_dataset


if __name__ == "__main__":
    build_ragas_dataset()