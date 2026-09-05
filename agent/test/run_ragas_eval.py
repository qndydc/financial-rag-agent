# -*- coding: utf-8 -*-

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    FactualCorrectness,
    LLMContextRecall,
    ResponseRelevancy,
)

from agent.llm.base_llm import get_llm
from agent.orchestrator import FinancialRAGAgent
from configs import model_config
from rag.embeddings.embed_model import LocalEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTSET_PATH = Path(__file__).resolve().parent / "testset.jsonl"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"


def load_testset(path: Path) -> List[Dict[str, Any]]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            query = item.get("query") or item.get("question")
            reference = item.get("reference")
            if not query or not reference:
                raise ValueError(
                    f"{path}:{line_number} 必须包含 query（或 question）和 reference"
                )

            samples.append({"query": query, "reference": reference})

    if not samples:
        raise ValueError(f"评测集为空：{path}")

    return samples


def build_dataset(
    agent: FinancialRAGAgent, testset: List[Dict[str, Any]]
) -> EvaluationDataset:
    rows = []

    for i, item in enumerate(testset, 1):
        query = item["query"]
        reference = item["reference"]

        print(f"[{i}/{len(testset)}] {query}")

        result = agent.eval_chat(
            user_input=query,
            session_id=f"eval_{uuid.uuid4().hex}",
        )

        rows.append(
            {
                "user_input": query,
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": reference,
            }
        )

    return EvaluationDataset.from_list(rows)


def save_results(df) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result_json = REPORT_DIR / "ragas_result.json"
    result_csv = REPORT_DIR / "ragas_result.csv"
    summary_md = REPORT_DIR / "ragas_summary.md"

    df.to_json(result_json, orient="records", force_ascii=False, indent=2)
    df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    metrics = {
        column: float(df[column].mean())
        for column in df.select_dtypes(include="number").columns
    }
    if "faithfulness" in metrics:
        metrics["hallucination_rate_proxy"] = 1 - metrics["faithfulness"]

    lines = [
        "# RAGAS 评测汇总",
        "",
        f"- 样本数：{len(df)}",
        f"- 评判模型：{model_config.LLM_MODEL_NAME}",
        f"- 相关性向量模型：{model_config.EMBEDDING_MODEL_NAME}",
        "",
    ]
    lines.extend(f"- {name}: {value:.4f}" for name, value in metrics.items())
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n评测明细：{result_json}")
    print(f"CSV 明细：{result_csv}")
    print(f"指标汇总：{summary_md}")


def main() -> None:
    agent = FinancialRAGAgent()
    testset = load_testset(TESTSET_PATH)
    dataset = build_dataset(agent, testset)

    evaluator_llm = LangchainLLMWrapper(get_llm(temperature=0, streaming=False))
    evaluator_embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    df = result.to_pandas()

    print("\n========== RAGAS Metrics ==========")
    for column in df.select_dtypes(include="number").columns:
        print(f"{column}: {df[column].mean():.4f}")

    if "faithfulness" in df.columns:
        print(f"hallucination_rate_proxy: {1 - df['faithfulness'].mean():.4f}")

    save_results(df)


if __name__ == "__main__":
    main()
