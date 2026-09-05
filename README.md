# 🚀 Financial RAG Agent
**面向金融分析师的智能研报分析与问答系统 | 基于 RAG + Agent 架构**

---

## 📖 项目简介
Financial RAG Agent 是一个面向金融分析师的智能研报分析系统，基于 RAG（Retrieval-Augmented Generation）与 Agent 架构构建，能够从大量券商研报 PDF 中自动解析结构化内容，通过混合检索（向量 + BM25 + Reranker）精准召回相关段落，并结合大语言模型生成具备可解释性的答案。

系统支持多轮对话、查询改写、证据筛选与判断，并在最终回答中标注引用来源与页码，实现**可溯源**的专业级问答体验。同时，项目提供完整的前后端 Demo（FastAPI + Vue + SSE 流式输出）、Docker 一键部署方案以及基于 RAGAS 的自动化评测体系，适用于金融研究、投研辅助及垂直领域 RAG 系统开发实践。

---

## ✅ 核心功能
- 📄 **多 PDF 研报解析与结构化处理**
- 🧠 **RAG + Agent 多轮对话**（基于 LangGraph DAG 工作流）
- 🔍 **Hybrid Retrieval**（向量检索 + BM25 + 微调 Reranker 精排）
- 📌 **引用溯源**（自动标注文档来源 + 页码 + 段落）
- ⚡ **SSE 流式输出**（打字机式实时生成）
- 🧩 **Query Rewrite + Retrieval Judge**（查询改写 + 检索判决）
- 📊 **RAGAS 自动化评测**（忠实度、相关性、幻觉率评估）
- 🐳 **Docker 一键部署**（前后端一体化）
- 🖥️ **分析师工作台 UI**（Vue + 三栏布局）

---

## ⚡ Quick Start（推荐）
### 一键启动前后端
```bash
docker compose -f docker/docker-compose.yml up --build
```

### 🌐 访问地址
| 服务         | 地址                          |
|------------|-------------------------------|
| 前端（Vue） | http://localhost:5173         |
| 后端（FastAPI） | http://localhost:8000       |
| API 文档    | http://localhost:8000/docs    |

### 🛑 停止服务
```bash
docker compose -f docker/docker-compose.yml down
```

### 🔄 后台运行（可选）
```bash
docker compose -f docker/docker-compose.yml up -d --build
```

---

## 🧪 Development Mode（开发模式）
适用于调试代码或二次开发。

### 方式一：Docker 分开启动
#### 启动后端
```bash
docker build -t financial-rag-backend -f docker/backend/Dockerfile .

docker run -p 8000:8000 \
  --env-file .env \
  financial-rag-backend
```

#### 启动前端
```bash
cd docker/frontend
docker build -t financial-rag-frontend .
docker run -p 5173:5173 financial-rag-frontend
```

### 方式二：本地开发（推荐）
#### 启动后端
```bash
pip install -r requirements.txt

uvicorn docker.backend.api:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

#### 启动前端
```bash
cd docker/frontend
npm install
npm run dev
```

---

## 📦 项目结构（关键部分）
```
docker/
├── docker-compose.yml    #  compose 编排
├── backend/              # 后端 FastAPI
│   ├── api.py
│   └── Dockerfile
└── frontend/             # 前端 Vue
    ├── Dockerfile
    └── src/
```

---

## ⚙️ 环境配置
创建项目根目录 `.env` 文件：
```env
OPENAI_API_KEY=your_api_key
EMBEDDING_MODEL=your_embedding_model
VECTOR_STORE_PATH=data/vector_store
```

---

## 📚 数据准备
首次运行前，构建向量库：
```bash
python scripts/build_vector_store.py
```

---

## 📊 RAGAS 评测
安装依赖并在 `.env` 中配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和
`LLM_MODEL_NAME`。评测用例位于 `agent/test/testset.jsonl`，每行至少包含
`query` 和 `reference`。

运行自动化评测：
```bash
python -m agent.test.run_ragas_eval
```

评测过程分为两步：先让 `FinancialRAGAgent.eval_chat()` 为每个问题生成
`response` 和 `retrieved_contexts`，再由独立的评判模型计算以下指标：

- `faithfulness`：回答中的事实能否由检索上下文支持。
- `response_relevancy`：回答是否切中问题。
- `context_recall`：检索上下文是否覆盖参考答案中的信息。
- `factual_correctness`：生成回答与参考答案的事实是否一致。

`hallucination_rate_proxy = 1 - faithfulness` 只是由忠实度推导出的代理指标，
不是单独标注得到的真实幻觉率。比较不同版本时，应使用同一测试集、评判模型和参数。

### 输出路径
```
evaluation/reports/
├── ragas_result.json
├── ragas_result.csv
└── ragas_summary.md
```

---

## 🧠 系统架构
```
用户问题
   ↓
Agent（LangGraph DAG 工作流）
   ↓
Query Rewrite（查询改写/子问题拆分）
   ↓
Hybrid Retrieval（BM25 + Vector + Rerank）
   ↓
Retrieval Judge（检索判决/重试机制）
   ↓
LLM 生成 + 引用标注
   ↓
带页码/来源的最终答案
```

---

## ⚠️ 常见问题
### 1️⃣ Docker 路径错误
如果出现：
```
cannot find docker/xxx
```
请检查 `docker-compose.yml`：
```yaml
build:
  context: ..
  dockerfile: docker/backend/Dockerfile
```

### 2️⃣ SSE 不流式
Nginx 需要配置：
```
proxy_buffering off;
```

### 3️⃣ 端口冲突
修改映射端口：
```yaml
ports:
  - "8001:8000"
```

### 4️⃣ 向量库/模型缺失
- 已运行 `build_vector_store.py`
- `.env` 配置正确
- 模型/向量库路径存在

---

## 📌 Roadmap
- [ ] PDF 原文高亮定位
- [ ] 多文档对比分析
- [ ] 表格问答增强
- [ ] 金融领域 Reranker 微调
- [ ] 多用户与权限系统

---

## 📄 License
**MIT License**  
自由使用、修改、分发，保留版权声明即可。

---

## ⭐ Support
如果这个项目对你有帮助，欢迎 **Star ⭐** 或提 **Issue / PR** 🙌

---
