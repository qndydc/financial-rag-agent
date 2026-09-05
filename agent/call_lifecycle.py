# -*- coding: utf-8 -*-
"""工具与模型调用的统一生命周期管理。"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator
from typing_extensions import Annotated

from configs import model_config


QueryText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=model_config.MAX_USER_QUERY_LENGTH,
    ),
]


class ErrorType(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SERVICE_UNAVAILABLE = "service_unavailable"
    EMPTY_RESULT = "empty_result"
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    TOOL_NOT_FOUND = "tool_not_found"
    RISK_DENIED = "risk_denied"
    INTERNAL_ERROR = "internal_error"


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    MODEL = "model"
    HIGH = "high"


class CallObservation(BaseModel):
    """写回 AgentState 的稳定调用记录。"""

    model_config = ConfigDict(extra="forbid")

    tool: str
    status: Literal["success", "failed"]
    result: Any
    error_type: Optional[str] = None
    retryable: bool = False
    attempts: int = 1
    duration_ms: int = 0


class LLMCallArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    messages: List[Any] = Field(min_length=1, max_length=50)


class RagSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: QueryText
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    use_reranker: bool = True


class StructuredQueryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task_type: Literal[
        "fact",
        "compare",
        "contrast",
        "multi_aspect",
        "multi_constraint_exclude",
        "filter",
        "exclude",
        "aggregation",
    ]
    original_query: QueryText
    rewritten_query: QueryText
    sub_queries: List[QueryText] = Field(min_length=1, max_length=3)
    filters: Dict[str, Any]
    use_history: bool
    exclude_terms: List[QueryText] = Field(default_factory=list, max_length=10)
    required_terms: List[QueryText] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=5, ge=1, le=30)

    @model_validator(mode="after")
    def ensure_json_sized_filters(self) -> "StructuredQueryPayload":
        try:
            encoded = json.dumps(self.filters, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("filters 必须是可序列化的 JSON 对象") from exc
        if len(encoded) > 4000:
            raise ValueError("filters 内容过长")
        return self


class StructuredRagSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    structured_query: StructuredQueryPayload
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    use_reranker: bool = True


@dataclass(frozen=True)
class CallSpec:
    name: str
    args_model: Type[BaseModel]
    risk_level: RiskLevel
    timeout_seconds: float
    max_attempts: int


@dataclass
class CallOutcome:
    value: Any
    observation: CallObservation


def validate_structured_query(payload: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """用可信原问题覆盖模型输出，并进行严格、禁止额外字段的校验。"""
    if not isinstance(payload, dict):
        raise TypeError("结构化查询必须是 JSON object")
    candidate = dict(payload)
    candidate["original_query"] = original_query
    validated = StructuredQueryPayload.model_validate(candidate, strict=True)
    return _model_values(validated)


def friendly_error_message(observation: Dict[str, Any]) -> str:
    error_type = observation.get("error_type")
    messages = {
        ErrorType.INVALID_ARGUMENTS.value: "调用参数不符合工具要求，自动修复后仍无法执行。",
        ErrorType.TIMEOUT.value: "服务响应超时，已完成自动重试，请稍后再试。",
        ErrorType.RATE_LIMITED.value: "服务当前请求过多，已完成自动重试，请稍后再试。",
        ErrorType.SERVICE_UNAVAILABLE.value: "依赖服务暂时不可用，已完成自动重试，请稍后再试。",
        ErrorType.EMPTY_RESULT.value: "调用成功，但没有返回可用结果。",
        ErrorType.BAD_REQUEST.value: "依赖服务拒绝了本次请求，请检查服务配置。",
        ErrorType.UNAUTHORIZED.value: "依赖服务认证失败，请检查凭据配置。",
        ErrorType.FORBIDDEN.value: "依赖服务拒绝访问，请检查账号权限。",
        ErrorType.TOOL_NOT_FOUND.value: "请求的工具不存在或未启用。",
        ErrorType.RISK_DENIED.value: "高风险操作未获得可信授权，已阻止执行。",
        ErrorType.INTERNAL_ERROR.value: "调用过程中发生不可恢复的内部错误。",
    }
    return messages.get(error_type, "调用失败，暂时无法完成请求。")


class CallLifecycle:
    """校验、风险控制、超时、重试、分类和 observation 生成。"""

    def __init__(
        self,
        *,
        specs: Optional[List[CallSpec]] = None,
        max_concurrency: Optional[int] = None,
        backoff_base_seconds: Optional[float] = None,
        backoff_max_seconds: Optional[float] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self._specs: Dict[str, CallSpec] = {}
        self._sleep = sleep_fn
        self._random = random_fn
        self._backoff_base = (
            backoff_base_seconds
            if backoff_base_seconds is not None
            else model_config.CALL_BACKOFF_BASE_SECONDS
        )
        self._backoff_max = (
            backoff_max_seconds
            if backoff_max_seconds is not None
            else model_config.CALL_BACKOFF_MAX_SECONDS
        )
        concurrency = max_concurrency or model_config.CALL_MAX_CONCURRENCY
        self._executor = ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="call-lifecycle")
        for spec in specs or _default_specs():
            self.register(spec)

    def register(self, spec: CallSpec) -> None:
        self._specs[spec.name] = spec

    def execute(
        self,
        tool: str,
        func: Callable[..., Any],
        args: Dict[str, Any],
        *,
        approved_high_risk: bool = False,
        argument_retryable: bool = True,
        empty_retryable: bool = True,
        is_empty: Optional[Callable[[Any], bool]] = None,
        summarize: Optional[Callable[[Any], Any]] = None,
    ) -> CallOutcome:
        started = time.monotonic()
        spec = self._specs.get(tool)
        if spec is None:
            return self._failure(
                tool,
                ErrorType.TOOL_NOT_FOUND,
                "工具未注册",
                started,
                attempts=0,
                retryable=False,
            )

        if spec.risk_level == RiskLevel.HIGH and not approved_high_risk:
            return self._failure(
                tool,
                ErrorType.RISK_DENIED,
                "高风险工具缺少可信授权",
                started,
                attempts=0,
                retryable=False,
            )

        try:
            validated = spec.args_model.model_validate(args, strict=True)
            validated_args = _model_values(validated)
        except (ValidationError, TypeError, ValueError) as exc:
            return self._failure(
                tool,
                ErrorType.INVALID_ARGUMENTS,
                _error_summary(exc),
                started,
                attempts=0,
                retryable=argument_retryable,
            )

        attempts = 0
        while attempts < max(1, spec.max_attempts):
            attempts += 1
            future = self._executor.submit(func, **validated_args)
            try:
                value = future.result(timeout=spec.timeout_seconds)
                empty = is_empty(value) if is_empty else _default_is_empty(value)
                result_summary = summarize(value) if summarize else _summarize_result(value)
                if empty:
                    observation = CallObservation(
                        tool=tool,
                        status="success",
                        result=result_summary,
                        error_type=ErrorType.EMPTY_RESULT.value,
                        retryable=empty_retryable,
                        attempts=attempts,
                        duration_ms=_duration_ms(started),
                    )
                    return CallOutcome(value=value, observation=observation)

                observation = CallObservation(
                    tool=tool,
                    status="success",
                    result=result_summary,
                    error_type=None,
                    retryable=False,
                    attempts=attempts,
                    duration_ms=_duration_ms(started),
                )
                return CallOutcome(value=value, observation=observation)
            except FutureTimeoutError as exc:
                future.cancel()
                error_type = ErrorType.TIMEOUT
                error = exc
            except Exception as exc:  # 分类后只对明确的临时错误重试
                error_type = classify_error(exc)
                error = exc

            if error_type in _TRANSIENT_ERRORS and attempts < spec.max_attempts:
                retry_after = _retry_after_seconds(error)
                self._sleep(retry_after if retry_after is not None else self._backoff(attempts))
                continue

            retryable = error_type == ErrorType.INVALID_ARGUMENTS and argument_retryable
            return self._failure(
                tool,
                error_type,
                _error_summary(error),
                started,
                attempts=attempts,
                retryable=retryable,
            )

        return self._failure(
            tool,
            ErrorType.INTERNAL_ERROR,
            "调用未产生结果",
            started,
            attempts=attempts,
            retryable=False,
        )

    def _failure(
        self,
        tool: str,
        error_type: ErrorType,
        message: Any,
        started: float,
        *,
        attempts: int,
        retryable: bool,
    ) -> CallOutcome:
        observation = CallObservation(
            tool=tool,
            status="failed",
            result={"error": message},
            error_type=error_type.value,
            retryable=retryable,
            attempts=attempts,
            duration_ms=_duration_ms(started),
        )
        return CallOutcome(value=None, observation=observation)

    def _backoff(self, attempt: int) -> float:
        base = min(self._backoff_max, self._backoff_base * (2 ** max(0, attempt - 1)))
        return base + (base * 0.25 * self._random())


def classify_error(exc: Exception) -> ErrorType:
    if isinstance(exc, (json.JSONDecodeError, ValidationError, TypeError)):
        return ErrorType.INVALID_ARGUMENTS
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, FutureTimeoutError)):
        return ErrorType.TIMEOUT

    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    if status_code == 429:
        return ErrorType.RATE_LIMITED
    if status_code in {408, 500, 502, 503, 504}:
        return ErrorType.SERVICE_UNAVAILABLE
    if status_code == 400:
        return ErrorType.BAD_REQUEST
    if status_code == 401:
        return ErrorType.UNAUTHORIZED
    if status_code == 403:
        return ErrorType.FORBIDDEN

    class_name = type(exc).__name__.lower()
    if "timeout" in class_name:
        return ErrorType.TIMEOUT
    if "ratelimit" in class_name or "rate_limit" in class_name:
        return ErrorType.RATE_LIMITED
    if "connection" in class_name or "serviceunavailable" in class_name:
        return ErrorType.SERVICE_UNAVAILABLE
    return ErrorType.INTERNAL_ERROR


_TRANSIENT_ERRORS = {
    ErrorType.TIMEOUT,
    ErrorType.RATE_LIMITED,
    ErrorType.SERVICE_UNAVAILABLE,
}


def _default_specs() -> List[CallSpec]:
    llm_timeout = model_config.LLM_CALL_TIMEOUT_SECONDS
    tool_timeout = model_config.TOOL_CALL_TIMEOUT_SECONDS
    attempts = model_config.CALL_MAX_ATTEMPTS
    return [
        CallSpec("llm.query_rewrite", LLMCallArgs, RiskLevel.MODEL, llm_timeout, attempts),
        CallSpec("llm.query_repair", LLMCallArgs, RiskLevel.MODEL, llm_timeout, attempts),
        CallSpec("llm.query_generalize", LLMCallArgs, RiskLevel.MODEL, llm_timeout, attempts),
        CallSpec("llm.answer", LLMCallArgs, RiskLevel.MODEL, llm_timeout, attempts),
        CallSpec("rag_search", RagSearchArgs, RiskLevel.READ_ONLY, tool_timeout, attempts),
        CallSpec("structured_rag_search", StructuredRagSearchArgs, RiskLevel.READ_ONLY, tool_timeout, attempts),
    ]


def _model_values(model: BaseModel) -> Dict[str, Any]:
    """保留 LangChain message 等对象，不通过 model_dump 将其转换为 dict。"""
    return {
        field: _validated_value(getattr(model, field))
        for field in model.__class__.model_fields
    }


def _validated_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _model_values(value)
    if isinstance(value, list):
        return [_validated_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_validated_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _validated_value(item) for key, item in value.items()}
    return value


def _default_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"no_data", "not_found"}
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def _summarize_result(value: Any) -> Any:
    if isinstance(value, dict):
        docs = value.get("docs")
        if isinstance(docs, list):
            return {"doc_count": len(docs)}
        return {"keys": sorted(str(key) for key in value.keys())[:20]}
    content = getattr(value, "content", None)
    if content is not None:
        return {"content_chars": len(str(content))}
    if isinstance(value, str):
        return {"content_chars": len(value)}
    if isinstance(value, (list, tuple, set)):
        return {"item_count": len(value)}
    return {"type": type(value).__name__}


def _error_summary(exc: Exception) -> Any:
    if isinstance(exc, ValidationError):
        try:
            return exc.errors(include_url=False, include_input=False)
        except TypeError:
            return str(exc)[:1000]
    text = str(exc) or type(exc).__name__
    text = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1***", text)
    return text[:1000]


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, min(float(value), model_config.CALL_BACKOFF_MAX_SECONDS))
    except (TypeError, ValueError):
        return None


def _duration_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


call_lifecycle = CallLifecycle()
