# -*- coding: utf-8 -*-

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextRecall,
    FactualCorrectness,
)

from agent.orchestrator import FinancialRAGAgent


TESTSET_PATH = Path(__file__).resolve().parent / "testset.jsonl"


def load_testset(path: Path) -> List[Dict[str, Any]]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    return samples


def build_dataset(agent: FinancialRAGAgent, testset: List[Dict[str, Any]]) -> Dataset:
    rows = []

    for i, item in enumerate(testset, 1):
        question = item["question"]
        reference = item["reference"]

        print(f"[{i}/{len(testset)}] {question}")

        result = agent.eval_chat(
            user_input=question,
            session_id=f"eval_{uuid.uuid4().hex}",
        )

        rows.append({
            "user_input": question,
            "response": result["answer"],
            "retrieved_contexts": result["contexts"],
            "reference": reference,
        })

    return Dataset.from_list(rows)


def main():
    agent = FinancialRAGAgent()
    testset = load_testset(TESTSET_PATH)

    dataset = build_dataset(agent, testset)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            FactualCorrectness(),
        ],
        raise_exceptions=False,
    )

    df = result.to_pandas()

    print("\n========== Ragas Metrics ==========")

    for col in df.columns:
        if df[col].dtype.kind in "if":
            print(f"{col}: {df[col].mean():.4f}")

    if "faithfulness" in df.columns:
        hallucination_rate = 1 - df["faithfulness"].mean()
        print(f"hallucination_rate: {hallucination_rate:.4f}")


if __name__ == "__main__":
    main()