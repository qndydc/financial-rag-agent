# -*- coding: utf-8 -*-
from typing import Dict

from agent.state.agent_state import AgentState


def fallback_node(state: AgentState) -> Dict:
    """
    统一兜底节点：
    - 问题太模糊
    - 检索失败
    - 检索结果不足以支持回答
    """
    reason = state.get("fallback_reason", "").strip()
    user_input = state.get("user_input", "").strip()

    if not reason:
        reason = "当前检索到的研报内容不足以支撑可靠回答。"

    answer = (
        f"抱歉，我暂时不能可靠回答这个问题。\n\n"
        f"原因：{reason}\n\n"
        f"你可以尝试把问题问得更具体一些，例如补充：\n"
        f"- 公司名\n"
        f"- 时间范围\n"
        f"- 具体指标（如营收、净利润、毛利率、同比增速、评级等）"
    )

    return {"answer": answer}