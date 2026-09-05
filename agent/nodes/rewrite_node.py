# -*- coding: utf-8 -*-
import json
from typing import Dict, List
import re

from langchain_core.messages import HumanMessage, SystemMessage
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from agent.state.agent_state import AgentState
from agent.llm.base_llm import get_llm
from agent.call_lifecycle import call_lifecycle, validate_structured_query


QUERY_REWRITE_SYSTEM_PROMPT = """
你是金融研报问答系统的查询改写器。
请把用户问题改写为适合文档检索的 JSON。
只输出 JSON，不要解释。

输出格式：
{
  "task_type": "fact | compare | multi_aspect | filter | aggregation",
  "rewritten_query": "适合检索的查询",
  "sub_queries": ["子查询1", "子查询2"],
  "filters": {},
  "use_history": false
}

要求：
1. rewritten_query 保留公司、指标、年份、主题词
2. sub_queries 最多 3 个，只有在确实需要拆分时才拆
3. 若问题不依赖历史，use_history=false
4. 若没有 filters，输出 {}
"""


QUERY_REWRITE_USER_PROMPT = """
【最近对话历史】
__history_text__

【当前用户问题】
__user_input__

请输出结构化检索 JSON：
"""


def rewrite_node(state: AgentState) -> Dict:
    user_input = (state.get("user_input") or "").strip()
    history = state.get("chat_history", [])

    if not user_input:
        print("[rewrite_node] 用户输入为空，跳过改写")
        return _build_result(
            user_input="",
            rewritten_query="",
            sub_queries=[],
            task_type="fact",
            filters={},
            use_history=False,
            debug_info=state.get("debug_info", {}),
            rewrite_mode="empty",
        )

    # 1) 简单问题：直接通过，不产生模型调用 observation
    if not _need_any_rewrite(user_input, history):
        print("[rewrite_node] 问题简单，直接通过")
        return _build_result(
            user_input=user_input,
            rewritten_query=user_input,
            sub_queries=[user_input],
            task_type="fact",
            filters={},
            use_history=False,
            debug_info=state.get("debug_info", {}),
            rewrite_mode="direct",
        )
    '''
    # 3) 轻度上下文依赖：规则改写
    if _need_rule_rewrite(user_input, history):
        print("[rewrite_node] 问题需要规则改写")
        rewritten_query = _simple_rewrite(user_input, history)
        return _build_result(
            user_input=user_input,
            rewritten_query=rewritten_query,
            sub_queries=[rewritten_query] if rewritten_query else [user_input],
            task_type="fact",
            filters={},
            use_history=(rewritten_query != user_input),
            debug_info=state.get("debug_info", {}),
            rewrite_mode="rule",
        )
    '''
    # 2) 复杂问题：通过统一生命周期调用 LLM
    history_text = _format_history_for_rewrite(history)
    print(f"[rewrite_node] 调用 LLM 进行改写，历史文本: {history_text}")
    llm = get_llm()
    messages = [
        SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=QUERY_REWRITE_USER_PROMPT
            .replace("__history_text__", history_text)
            .replace("__user_input__", user_input)
        ),
    ]

    def invoke_and_validate(messages):
        response = llm.invoke(messages)
        text = getattr(response, "content", str(response)).strip()
        return parse_structured_query_response(text, user_input)

    outcome = call_lifecycle.execute(
        "llm.query_rewrite",
        invoke_and_validate,
        {"messages": messages},
        argument_retryable=(
            state.get("argument_repair_count", 0)
            < state.get("argument_repair_limit", 1)
        ),
        is_empty=lambda value: not value,
    )
    observation = outcome.observation.model_dump()

    if outcome.value is None:
        return {
            "last_observation": observation,
            "observations": [observation],
            "debug_info": {
                **state.get("debug_info", {}),
                "rewrite_mode": "llm_failed",
                "rewrite_error": observation["result"],
            },
        }

    result = build_query_state(
        outcome.value,
        debug_info=state.get("debug_info", {}),
        rewrite_mode="llm",
    )
    result.update({
        "last_observation": observation,
        "observations": [observation],
    })
    return result


def parse_structured_query_response(text: str, user_input: str) -> Dict:
    parsed = _safe_parse_json(text)
    return validate_structured_query(parsed, original_query=user_input)


def build_query_state(payload: Dict, debug_info: Dict, rewrite_mode: str) -> Dict:
    return _build_result(
        user_input=payload["original_query"],
        rewritten_query=payload["rewritten_query"],
        sub_queries=payload["sub_queries"],
        task_type=payload["task_type"],
        filters=payload["filters"],
        use_history=payload["use_history"],
        debug_info=debug_info,
        rewrite_mode=rewrite_mode,
        exclude_terms=payload.get("exclude_terms", []),
        required_terms=payload.get("required_terms", []),
        top_k=payload.get("top_k", 5),
    )


def _build_result(
    user_input: str,
    rewritten_query: str,
    sub_queries: List[str],
    task_type: str,
    filters: Dict,
    use_history: bool,
    debug_info: Dict,
    rewrite_mode: str,
    exclude_terms: List[str] = None,
    required_terms: List[str] = None,
    top_k: int = 5,
) -> Dict:
    structured_query = {
        "task_type": task_type,
        "original_query": user_input,
        "rewritten_query": rewritten_query,
        "sub_queries": sub_queries,
        "filters": filters,
        "use_history": use_history,
        "exclude_terms": exclude_terms or [],
        "required_terms": required_terms or [],
        "top_k": top_k,
    }

    return {
        "structured_query": structured_query,
        "rewritten_query": rewritten_query,
        "retrieve_queries": sub_queries if sub_queries else [user_input],
        "debug_info": {
            **debug_info,
            "rewrite_mode": rewrite_mode,
            "structured_query": structured_query,
        }
    }


def _format_history_for_rewrite(history: List, max_turn: int = 4, max_chars: int = 300) -> str:
    if not history:
        return "无"

    user_msgs = []
    for msg in reversed(history):
        if getattr(msg, "type", "") == "human":
            content = (getattr(msg, "content", "") or "").strip()
            if content:
                user_msgs.append(f"用户: {content[:120]}")
        if len(user_msgs) >= max_turn:
            break

    user_msgs.reverse()
    text = "\n".join(user_msgs)
    return text[:max_chars] if text else "无"


def _safe_parse_json(text: str) -> Dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    return json.loads(text)


def _need_any_rewrite(user_input: str, history: List) -> bool:
    text = (user_input or "").strip()
    if not text:
        return False

    pronouns = ["它", "他", "她","他们","她们","它们", "这家", "那家", "该公司", "其", "这个", "那个", "前者", "后者", "那","那么", "然后", "再", "接着", "另外", "还有", "以及", "如何", "多少", "吗"]
    complex_terms = ["对比", "比较", "分别", "各自", "汇总", "总结", "筛选", "排除", "不包括", "除了", "排名", "前五", "前十", "top", "bottom","比", "较", "更", "多", "少"]

    if any(p in text for p in pronouns):
        return True
    if any(t in text for t in complex_terms):
        return True
    if len(text) > 15 and not _looks_self_contained(text):
        return True

    return False


def _need_rule_rewrite(user_input: str, history: List[SystemMessage]) -> bool:
    """
    判断当前问题是否需要做“单 query 规则改写”。
    设计原则：
    - 所有依赖上文、但不需要拆成多query的问题，都尽量归到这里
    - 如果没有历史，直接返回 False
    """
    if not user_input or not history:
        return False

    text = user_input.strip()

    # 过长、信息完整的问题，通常不需要规则改写
    if len(text) >= 25 and _looks_self_contained(text):
        return False

    # 1. 明显代词 / 指代
    pronoun_patterns = [
        r"\b它\b", r"\b他\b", r"\b她\b", r"\b这\b", r"\b那\b",
        r"这个", r"那个", r"这家", r"那家", r"该公司", r"这家公司", r"那家公司",
        r"前者", r"后者", r"其", r"其中",
    ]

    # 2. 明显承接问法
    continuation_patterns = [
        r"^那",
        r"^那么",
        r"^然后",
        r"^再",
        r"^接着",
        r"^另外",
        r"^还有",
        r"^以及",
        r".*呢[？?]?$",
        r".*怎么样[？?]?$",
        r".*如何[？?]?$",
        r".*多少[？?]?$",
        r".*吗[？?]?$",
    ]

    # 3. 典型省略式短问句：只有指标/时间/限定词，没有主体
    elliptical_patterns = [
        r"^(同比|环比|增速|增长率|毛利率|净利率|净利润|营收|收入|利润|归母净利润|扣非净利润).*$",
        r"^(今年|去年|明年|202\d年|Q[1-4]|[1-4]季度|上半年|下半年).*$",
        r"^(按季度|按年度|分产品|分地区|海外|国内|主营业务|资产负债率|现金流).*$",
    ]

    # 4. 很短的追问句，通常依赖上文
    short_follow_up = len(text) <= 12

    if any(re.search(p, text) for p in pronoun_patterns):
        return True

    if any(re.search(p, text) for p in continuation_patterns):
        return True

    if any(re.search(p, text) for p in elliptical_patterns):
        return True

    # 短句且不自洽，优先认为依赖历史
    if short_follow_up and not _looks_self_contained(text):
        return True

    return False


def _looks_self_contained(text: str) -> bool:
    """
    判断一个问题是否已经基本自包含：
    - 含有明确主体 + 明确指标 / 动作
    这里只做很轻量规则，不追求完美。
    """
    has_company_like = bool(re.search(r"(公司|集团|银行|证券|保险|科技|信息|股份|控股|有限)", text))
    has_metric_like = bool(re.search(
        r"(营收|收入|利润|净利润|归母净利润|毛利率|净利率|同比|环比|估值|PE|PB|ROE|现金流|资产负债率)",
        text
    ))

    # 有明显公司名风格 + 有指标，通常算完整问题
    return has_company_like and has_metric_like


def _simple_rewrite(user_input: str, history: List) -> str:
    user_input = (user_input or "").strip()
    if not user_input:
        return ""

    if not history:
        return user_input

    last_user_msgs = [
        getattr(m, "content", "").strip()
        for m in history
        if getattr(m, "type", "") == "human" and getattr(m, "content", "").strip()
    ]
    last_user_query = last_user_msgs[-1] if last_user_msgs else ""

    if not last_user_query:
        return user_input

    pronouns = ["它", "这家公司", "那家", "其", "该公司", "这个", "那个", "前者", "后者", "同比", "环比", "相比"]

    if any(p in user_input for p in pronouns):
        return f"{last_user_query} {user_input}".strip()

    return user_input
