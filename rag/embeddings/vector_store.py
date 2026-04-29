from langchain_community.vectorstores import FAISS
from rag.embeddings.embed_model import LocalEmbeddings
import os

def create_vector_store(chunks_with_meta: list):
    """
    传入带元数据的chunk列表，创建向量库
    结构：[{"text": str, "metadata": dict}, ...]
    """
    embedding = LocalEmbeddings()

    # 分离文本与元数据
    texts = [item["text"] for item in chunks_with_meta]
    metadatas = [item["metadata"] for item in chunks_with_meta]

    # 文本向量化 + 元数据一起存储
    return FAISS.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas
    )

def save_vector_store(vector_store: FAISS, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"✅ 向量库已保存至：{save_path}")

def load_vector_store(db_path: str): #加载数据库
    embedding = LocalEmbeddings()
    return FAISS.load_local(
        folder_path=db_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
