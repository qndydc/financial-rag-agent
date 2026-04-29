# -*- coding: utf-8 -*-

import os
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm


# =========================
# 配置
# =========================

BASE_MODEL_PATH = "D:/Aprojet/Model/bge-reranker-base"

CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = CURRENT_DIR / "reranker_train_data.jsonl"
OUTPUT_DIR = CURRENT_DIR / "output" / "bge-reranker-base-finance"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 随机种子
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 读取三元组数据
# =========================

def load_train_data(path: Path) -> List[Dict[str, str]]:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            query = item["query"]

            if "pos" in item and "neg" in item:
                positive = item["pos"][0]
                negative = item["neg"][0]
            else:
                positive = item["positive_doc"]
                negative = item["negative_doc"]

            if not query or not positive or not negative:
                continue

            data.append({
                "query": query,
                "positive": positive,
                "negative": negative,
            })

    return data


# =========================
# Dataset
# =========================

class RerankerTrainDataset(Dataset):
    """
    每条样本返回：
    query
    positive passage
    negative passage
    """

    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": item["negative"],
        }


def collate_fn(batch, tokenizer):
    queries = []
    passages = []
    labels = []

    for item in batch:
        query = item["query"]

        queries.append(query)
        passages.append(item["positive"])
        labels.append(1.0)

        queries.append(query)
        passages.append(item["negative"])
        labels.append(0.0)

    encoded = tokenizer(
        queries,
        passages,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    labels = torch.tensor(labels, dtype=torch.float)

    return encoded, labels


# =========================
# 训练函数
# =========================

def train():
    set_seed(SEED)

    print("=" * 80)
    print("BGE Reranker 微调启动")
    print("=" * 80)
    print(f"base model: {BASE_MODEL_PATH}")
    print(f"train data: {TRAIN_DATA_PATH}")
    print(f"output dir: {OUTPUT_DIR}")
    print(f"device: {DEVICE}")

    train_data = load_train_data(TRAIN_DATA_PATH)
    print(f"训练样本数：{len(train_data)}")

    if len(train_data) == 0:
        raise ValueError("训练数据为空，请检查 reranker_train_data.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=1,
    )

    model.to(DEVICE)
    model.train()

    dataset = RerankerTrainDataset(train_data)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = len(dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for encoded, labels in progress_bar:
            encoded = {
                k: v.to(DEVICE)
                for k, v in encoded.items()
            }
            labels = labels.to(DEVICE)

            outputs = model(**encoded)
            logits = outputs.logits.squeeze(-1)

            loss = loss_fn(logits, labels) # score(query, positive_chunk) > score(query, negative_chunk)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        epoch_output_dir = OUTPUT_DIR / f"checkpoint-epoch-{epoch + 1}"
        epoch_output_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)

        print(f"checkpoint saved to: {epoch_output_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n训练完成")
    print(f"最终模型已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    train()