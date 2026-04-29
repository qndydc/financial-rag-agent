# -*- coding: utf-8 -*-
"""
配置模块统一出口
方便其他模块直接导入：
from configs import model_config, rag_config
"""

from .model_config import model_config
from .rag_config import rag_config

__all__ = [
    "model_config",
    "rag_config"
]