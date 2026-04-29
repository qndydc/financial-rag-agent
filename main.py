from configs import model_config, rag_config

if __name__ == "__main__":
    print("=== 模型配置 ===")
    print("LLM 模式:", model_config.LLM_MODE)
    print("Embedding 模型:", model_config.EMBEDDING_MODEL_NAME)

    print("\n=== RAG 配置 ===")
    print("PDF 目录:", rag_config.RAW_PDF_DIR)
    print("分块大小:", rag_config.CHUNK_SIZE)
    print("是否保留页码:", rag_config.KEEP_PAGE_NUMBER)