financial_rag_agent/
├── agent/              # Agent 核心逻辑
│   ├── nodes/         # LangGraph 工作流节点
│   │   ├── retrieve_node.py   # 检索节点
│   │   ├── rewrite_node.py    # 查询改写节点
│   │   ├── judge_node.py      # 检索质量判决节点
│   │   ├── route_node.py      # 路由节点
│   │   ├── answer_node.py     # 答案生成节点
│   │   └── ...
│   ├── orchestrator.py        # FinancialRAGAgent 主类
│   ├── state/                 # 状态管理
│   ├── tools/                 # RAG 工具封装
│   └── llm/                   # LLM 适配层
├── rag/               # RAG 核心组件
│   ├── retrievers/    # 检索器
│   │   ├── hybrid_retriever.py   # 混合检索（向量+BM25）
│   │   ├── vector_retriever.py   # 向量检索（FAISS）
│   │   ├── bm25_retriever.py     # BM25 检索
│   │   └── reranker.py           # 重排序模型
│   ├── document_loaders/  # PDF 解析
│   ├── embeddings/        # 嵌入模型
│   └── text_splitters/    # 文本分块
├── configs/           # 配置文件
│   ├── rag_config.py    # RAG 参数
│   └── model_config.py  # 模型配置
├── dataset/           # 数据目录
│   ├── raw_pdf/       # 原始 PDF 研报
│   └── vector_store/  # 向量库
└── docker/            # Docker 部署