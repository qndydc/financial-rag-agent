# -*- coding: utf-8 -*-
"""
RAG 模块核心配置
包含文档解析、分块、检索、向量库、页码溯源等参数
全部可扩展，支持后期加过滤策略、召回策略、混合检索权重等
"""

import os


class RagConfig:
    # ---------------------------
    # 文档路径配置
    # ---------------------------
    # 原始 PDF 存放目录
    RAW_PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset/raw_pdf")
    # 向量库持久化路径
    VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset/vector_store")

    # ---------------------------
    # PDF 解析 & DeepDoc 配置
    # ---------------------------
    # 是否启用表格解析
    PARSE_TABLES = True
    # 是否保留页码（金融场景必须 True）
    KEEP_PAGE_NUMBER = True
    # 解析后输出格式：markdown / text
    PARSE_OUTPUT_FORMAT = "markdown"

    # ---------------------------
    # 文本分块配置
    # ---------------------------
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 50
    # 分块策略：paragraph / sentence / section
    CHUNK_SPLIT_STRATEGY = "paragraph"
    # 是否按标题层级结构化分块
    CHUNK_BY_HEADER = True

    # ---------------------------
    # 检索配置
    # ---------------------------
    # 向量召回数量 & 阈值
    VECTOR_SEARCH_TOP_K = 10
    SCORE_THRESHOLD = 0.1
    # BM25 召回数量
    BM25_SEARCH_TOP_K = 10
    # 最终融合后返回给 LLM 的 chunk 数量
    FINAL_RETURN_CHUNK_COUNT = 30

    # 混合检索权重（后期可扩展）
    HYBRID_SEARCH_WEIGHT_VECTOR = 0.8
    HYBRID_SEARCH_WEIGHT_BM25 = 0.2

    # ---------------------------
    # 元数据 & 溯源配置
    # ---------------------------
    # 强制携带元数据字段
    REQUIRED_METADATA_FIELDS = [
        "source",
        "file_path",
        "page_num",
        "publish_date",
        "company",
        "broker"
    ]
    # 回答时是否强制引用原文片段
    ENABLE_ORIGIN_QUOTE = True

rag_config = RagConfig()