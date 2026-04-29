'''
本地 Embedding 模型
所有参数自动从 config_model.py 读取
'''
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

# ====================== 🔥 修复在这里 ======================
# 抛弃 langchain_core.pydantic_v1，直接用标准 pydantic v1
try:
    from pydantic.v1 import BaseModel, Field, root_validator
except ImportError:
    from pydantic import BaseModel, Field, root_validator

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
# ==========================================================

# 👇 导入你的配置文件（关键！）
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs import model_config

logger = logging.getLogger(__name__)


class LocalEmbeddings(BaseModel, Embeddings):
    """
    本地 Embedding 模型（完整版，对齐 langchain-chatchat）
    所有参数自动从 config_model.py 读取
    支持：批量处理、GPU保护、归一化、超长文本安全
    """

    client: Any = Field(default=None, exclude=True)

    model: str = model_config.EMBEDDING_MODEL_NAME
    device: str = model_config.EMBEDDING_DEVICE
    normalize_embeddings: bool = model_config.EMBEDDING_NORMALIZE

    embedding_ctx_length: int = 8192
    chunk_size: int = 16
    show_progress_bar: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        all_required_fields = set(cls.__fields__.keys())
        extra = values.get("model_kwargs", {})

        for key in list(values.keys()):
            if key not in all_required_fields:
                extra[key] = values.pop(key)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            values["client"] = SentenceTransformer(
                model_name_or_path=values["model"],
                device=values["device"],
                **values.get("model_kwargs", {})
            )
            logger.info(f"✅ 向量模型加载成功：{values['model']}")
        except Exception as e:
            logger.error(f"❌ 模型加载失败：{str(e)}")
            raise RuntimeError(f"加载模型失败：{e}")
        return values

    def _get_len_safe_embeddings(
        self, texts: List[str], chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        _chunk_size = chunk_size or self.chunk_size
        client = self.client
        embeddings = []

        for i in range(0, len(texts), _chunk_size):
            batch = texts[i:i + _chunk_size]
            vecs = client.encode(
                batch,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            ).tolist()
            embeddings.extend(vecs)
        return embeddings

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        return self._get_len_safe_embeddings(texts, chunk_size=chunk_size)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
'''
# ====================== 测试代码：检查本地Embedding是否安装成功 ======================
if __name__ == "__main__":
    import sys
    import logging

    # 开启日志，方便看成功/失败信息
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        print("\n" + "="*60)
        print("🧪 开始测试本地 Embedding 模型加载与向量生成")
        print("="*60 + "\n")

        # 1. 实例化 Embedding 模型（自动加载你 config 里的模型）
        embed = LocalEmbeddings()

        # 2. 测试一句话生成向量
        test_text = "你好，这是一条测试文本，用于检查本地Embedding模型是否正常工作"
        vector = embed.embed_query(test_text)

        # 3. 输出结果
        print(f"\n✅ 测试成功！")
        print(f"📌 测试句子：{test_text}")
        print(f"📊 向量维度：{len(vector)}")
        print(f"🧬 向量前5个值：{vector[:5]}")
        print("\n🎉 你的本地 Embedding 模型环境完全正常，可以正常使用！")

    except Exception as e:
        print(f"\n❌ 测试失败！错误信息：\n{str(e)}")
        print("\n💡 可能原因：")
        print("   1. sentence-transformers 未安装：pip install sentence-transformers")
        print("   2. 模型路径配置错误（检查 model_config.py）")
        print("   3. 模型文件损坏或未下载完整")
        print("   4. PyTorch / GPU 环境异常")
        sys.exit(1)
'''