'''
读取所有pdf文件，将其切分并转换成向量，最后创建向量数据库存入其中
'''

import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from document_loaders.pymupdf_texttable_loader import PyMuPDFLoader
from text_splitters.chinese_splitter import ChineseMarkdownTextSplitter
from rag import create_vector_store, save_vector_store
import sys
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录到路径
from configs.rag_config import rag_config

RAW_PDF_DIR = rag_config.RAW_PDF_DIR
VECTOR_STORE_DIR = rag_config.VECTOR_STORE_DIR

def run_pdf_to_vector(
    pdf_dir: str = RAW_PDF_DIR,
    save_path: str = VECTOR_STORE_DIR,
    recursive: bool = True
):
    """
    完整 RAG 构建流水线：
    批量PDF → 按页读取（带页码/文件名）→ 文本切分 → 向量化 → 存入向量库
    全程保留元数据，不丢失任何来源信息
    """
    # ======================
    # 1. 扫描所有 PDF 文件
    # ======================
    pdf_paths = []
    if recursive:
        for root, _, files in os.walk(pdf_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_paths.append(str(Path(root) / file))
    else:
        pdf_paths = [
            str(Path(pdf_dir) / f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith(".pdf")
        ]

    if not pdf_paths:
        print("❌ 未找到任何 PDF 文件，请检查路径：", pdf_dir)
        return

    print(f"✅ 找到 {len(pdf_paths)} 个 PDF 文件，开始处理...\n")

    # ======================
    # 2. 读取所有 PDF（保留 content + metadata）
    # ======================
    all_pages = []  # 结构：[{"content": str, "metadata": dict}, ...]

    for path in pdf_paths:
        print(f"正在读取：{Path(path).name}")
        loader = PyMuPDFLoader(file_path=path)
        pages = loader.load_all_pages()  # 你的loader自带metadata
        print(f"   └── 共 {len(pages)} 页\n")
        all_pages.extend(pages)

    print(f"✅ 全部PDF读取完成，总页数：{len(all_pages)}")

    # ======================
    # 3. 文本切分
    # ======================
    splitter = ChineseMarkdownTextSplitter()
    chunks_with_meta = []

    for page in all_pages:
        text = page["content"]
        meta = page["metadata"]  # 直接用loader自带的元数据（文件名、页码、路径）

        if not text.strip():
            continue

        # 你的纯字符串切分
        chunks = splitter.split_text(text)

        # 每个小chunk都继承这一页的元数据
        for idx, chunk in enumerate(chunks):
            chunk_meta = dict(meta)  # ⚠️ 必须 copy，不能直接用 meta

            chunk_meta["chunk_id"] = f"{meta['doc_id']}_p{meta['page_num']}_c{idx+1}"

            chunks_with_meta.append({
                "text": chunk,
                "metadata": chunk_meta
            })

    print(f"✅ 切分完成，总片段数量：{len(chunks_with_meta)}")

    # ======================
    # 把 chunks_with_meta 转换成一个额外的 List[Document]
    # 后面 BM25 / reranker 需要
    # ======================
    from langchain_core.documents import Document
    all_documents_for_bm25 = []
    for item in chunks_with_meta:
        all_documents_for_bm25.append(Document(
            page_content=item["text"],
            metadata=item["metadata"]
        ))
    import json
    cache_path = os.path.join(save_path, "all_documents.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in all_documents_for_bm25
        ], f, ensure_ascii=False, indent=2)

    print("\n🎉 全部完成！json数据库已保存到：", save_path)

    # ======================
    # 4. 生成向量库并保存（带元数据一起存）
    # ======================
    print("\n正在生成向量库并保存...")
    vs = create_vector_store(chunks_with_meta)
    save_vector_store(vs, save_path)

    print("\n🎉 全部完成！向量库已保存到：", save_path)


# 直接运行此文件即可开始构建知识库
if __name__ == "__main__":
    run_pdf_to_vector(
        pdf_dir=RAW_PDF_DIR,
        save_path=VECTOR_STORE_DIR,
        recursive=True
    )