# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StructuredQuery(BaseModel):
    """
    统一结构化检索对象
    """

    task_type: str = Field(default="fact",description="查询类型：fact / compare / multi_aspect / filter / exclude / aggregation")

    original_query: str = Field(default="", description="原始用户问题")

    rewritten_query: str = Field(default="",description="归一化后的总检索语句，保留实体、年份、指标、主题词"
                                 )
    sub_queries: List[str] = Field(default_factory=list, description="拆解后的子查询列表, 多路召回的核心字段，每个元素都应是可独立检索的短查询")

    filters: Dict[str, Any] = Field(default_factory=dict, description="结构化过滤条件")

    use_history: bool = Field(default=False, description="是否依赖历史补全")
