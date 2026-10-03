# -*- coding: utf-8 -*-
import importlib.util
import sys
import time
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "agent" / "call_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("call_lifecycle_under_test", MODULE_PATH)
call_lifecycle_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = call_lifecycle_module
SPEC.loader.exec_module(call_lifecycle_module)

ATOMIC_MODULE_PATH = PROJECT_ROOT / "agent" / "tools" / "atomic_tools.py"
ATOMIC_SPEC = importlib.util.spec_from_file_location("atomic_tools_under_test", ATOMIC_MODULE_PATH)
atomic_tools_module = importlib.util.module_from_spec(ATOMIC_SPEC)
sys.modules[ATOMIC_SPEC.name] = atomic_tools_module
ATOMIC_SPEC.loader.exec_module(atomic_tools_module)

CallLifecycle = call_lifecycle_module.CallLifecycle
CallSpec = call_lifecycle_module.CallSpec
ErrorType = call_lifecycle_module.ErrorType
RagSearchArgs = call_lifecycle_module.RagSearchArgs
StructuredRagSearchArgs = call_lifecycle_module.StructuredRagSearchArgs
RiskLevel = call_lifecycle_module.RiskLevel
validate_structured_query = call_lifecycle_module.validate_structured_query
FinancialAtomicTools = atomic_tools_module.FinancialAtomicTools


class HttpFailure(Exception):
    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


class FakeDocument:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class FakeAdapter:
    def __init__(self, kind):
        self.kind = kind
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {
            "docs": [{"content": self.kind}],
            "meta": {"adapter": self.kind},
        }


def search_spec(*, attempts=3, timeout=0.5, risk=RiskLevel.READ_ONLY):
    return CallSpec(
        name="test_search",
        args_model=RagSearchArgs,
        risk_level=risk,
        timeout_seconds=timeout,
        max_attempts=attempts,
    )


class CallLifecycleTest(unittest.TestCase):
    def build_lifecycle(self, spec, sleeps=None):
        return CallLifecycle(
            specs=[spec],
            max_concurrency=4,
            backoff_base_seconds=0.01,
            backoff_max_seconds=0.02,
            sleep_fn=(sleeps.append if sleeps is not None else lambda _: None),
            random_fn=lambda: 0.0,
        )

    def test_dirty_arguments_are_blocked_before_execution(self):
        calls = []
        lifecycle = self.build_lifecycle(search_spec())

        outcome = lifecycle.execute(
            "test_search",
            lambda **kwargs: calls.append(kwargs),
            {"query": "海光信息", "mode": "hybrid", "use_reranker": True, "hidden": "value"},
        )

        self.assertEqual(calls, [])
        self.assertEqual(outcome.observation.status, "failed")
        self.assertEqual(outcome.observation.error_type, ErrorType.INVALID_ARGUMENTS.value)
        self.assertTrue(outcome.observation.retryable)
        self.assertEqual(outcome.observation.attempts, 0)

    def test_high_risk_call_requires_trusted_approval(self):
        calls = []
        lifecycle = self.build_lifecycle(search_spec(risk=RiskLevel.HIGH))

        outcome = lifecycle.execute(
            "test_search",
            lambda **kwargs: calls.append(kwargs),
            {"query": "test", "mode": "hybrid", "use_reranker": True},
        )

        self.assertEqual(calls, [])
        self.assertEqual(outcome.observation.error_type, ErrorType.RISK_DENIED.value)
        self.assertFalse(outcome.observation.retryable)

    def test_transient_error_retries_with_identical_arguments(self):
        calls = []
        sleeps = []
        lifecycle = self.build_lifecycle(search_spec(attempts=3), sleeps=sleeps)

        def flaky_search(query, mode, use_reranker):
            calls.append((query, mode, use_reranker))
            if len(calls) < 3:
                raise HttpFailure(503)
            return {"docs": [{"content": "ok"}]}

        outcome = lifecycle.execute(
            "test_search",
            flaky_search,
            {"query": "海光信息", "mode": "hybrid", "use_reranker": True},
            is_empty=lambda value: not value["docs"],
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(len(set(calls)), 1)
        self.assertEqual(sleeps, [0.01, 0.02])
        self.assertEqual(outcome.observation.status, "success")
        self.assertEqual(outcome.observation.attempts, 3)

    def test_permanent_error_does_not_retry(self):
        expected_types = {
            400: ErrorType.BAD_REQUEST.value,
            401: ErrorType.UNAUTHORIZED.value,
            403: ErrorType.FORBIDDEN.value,
        }
        for status_code, expected_type in expected_types.items():
            with self.subTest(status_code=status_code):
                calls = []
                lifecycle = self.build_lifecycle(search_spec(attempts=3))

                def permanent_failure(**kwargs):
                    calls.append(kwargs)
                    raise HttpFailure(status_code)

                outcome = lifecycle.execute(
                    "test_search",
                    permanent_failure,
                    {"query": "test", "mode": "hybrid", "use_reranker": True},
                )

                self.assertEqual(len(calls), 1)
                self.assertEqual(outcome.observation.error_type, expected_type)
                self.assertFalse(outcome.observation.retryable)

    def test_429_is_retried(self):
        calls = []
        lifecycle = self.build_lifecycle(search_spec(attempts=2))

        def rate_limited_once(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise HttpFailure(429)
            return {"docs": [{"content": "ok"}]}

        outcome = lifecycle.execute(
            "test_search",
            rate_limited_once,
            {"query": "test", "mode": "hybrid", "use_reranker": True},
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome.observation.status, "success")

    def test_empty_result_is_successful_but_recoverable_once(self):
        lifecycle = self.build_lifecycle(search_spec())
        outcome = lifecycle.execute(
            "test_search",
            lambda **kwargs: {"docs": []},
            {"query": "test", "mode": "hybrid", "use_reranker": True},
            is_empty=lambda value: not value["docs"],
            empty_retryable=True,
        )

        observation = outcome.observation.model_dump()
        self.assertEqual(observation["status"], "success")
        self.assertEqual(observation["error_type"], ErrorType.EMPTY_RESULT.value)
        self.assertTrue(observation["retryable"])
        for field in ("tool", "status", "result", "error_type", "retryable"):
            self.assertIn(field, observation)

    def test_timeout_is_bounded_and_exhausts_retry_budget(self):
        calls = []
        lifecycle = self.build_lifecycle(search_spec(attempts=2, timeout=0.005))

        def slow_search(**kwargs):
            calls.append(kwargs)
            time.sleep(0.03)
            return {"docs": []}

        outcome = lifecycle.execute(
            "test_search",
            slow_search,
            {"query": "test", "mode": "hybrid", "use_reranker": True},
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome.observation.error_type, ErrorType.TIMEOUT.value)
        self.assertFalse(outcome.observation.retryable)
        self.assertEqual(outcome.observation.attempts, 2)

    def test_unknown_tool_never_executes(self):
        lifecycle = self.build_lifecycle(search_spec())
        outcome = lifecycle.execute("missing", lambda: 1, {})
        self.assertEqual(outcome.observation.error_type, ErrorType.TOOL_NOT_FOUND.value)
        self.assertEqual(outcome.observation.attempts, 0)

    def test_structured_query_is_strict_and_original_query_is_trusted(self):
        payload = {
            "task_type": "fact",
            "original_query": "模型伪造的问题",
            "rewritten_query": "海光信息 净利润",
            "sub_queries": ["海光信息 净利润"],
            "filters": {},
            "use_history": False,
        }
        validated = validate_structured_query(payload, "用户真实问题")
        self.assertEqual(validated["original_query"], "用户真实问题")

        payload["smuggled_parameter"] = "blocked"
        with self.assertRaises(ValidationError):
            validate_structured_query(payload, "用户真实问题")

    def test_nested_structured_query_reaches_tool_as_plain_dict(self):
        spec = CallSpec(
            name="structured_test",
            args_model=StructuredRagSearchArgs,
            risk_level=RiskLevel.READ_ONLY,
            timeout_seconds=0.5,
            max_attempts=1,
        )
        lifecycle = self.build_lifecycle(spec)
        received = []
        structured_query = {
            "task_type": "fact",
            "original_query": "海光信息净利润",
            "rewritten_query": "海光信息 净利润",
            "sub_queries": ["海光信息 净利润"],
            "filters": {},
            "use_history": False,
        }

        def tool(structured_query, mode, use_reranker):
            received.append(structured_query)
            return {"docs": [{"content": "ok"}]}

        outcome = lifecycle.execute(
            "structured_test",
            tool,
            {
                "structured_query": structured_query,
                "mode": "hybrid",
                "use_reranker": True,
            },
        )
        self.assertEqual(outcome.observation.status, "success")
        self.assertIsInstance(received[0], dict)

    def test_registered_handler_is_invoked_by_name(self):
        lifecycle = CallLifecycle(
            max_concurrency=1,
            sleep_fn=lambda _: None,
            random_fn=lambda: 0.0,
        )
        lifecycle.register_handler(
            "metric_calculate",
            lambda operation, values, precision: {
                "operation": operation,
                "result": sum(values),
                "precision": precision,
            },
        )
        outcome = lifecycle.execute_registered(
            "metric_calculate",
            {"operation": "sum", "values": [1, 2, 3], "precision": 2},
        )
        self.assertEqual(outcome.observation.status, "success")
        self.assertEqual(outcome.value["result"], 6)


class AtomicToolsTest(unittest.TestCase):
    def setUp(self):
        self.simple_adapter = FakeAdapter("simple")
        self.structured_adapter = FakeAdapter("structured")
        self.documents = [
            FakeDocument(
                f"chunk-{index}",
                {
                    "doc_id": "report-1",
                    "chunk_id": f"report-1-c{index}",
                    "file_name": "海光信息2024年报.pdf",
                    "title": "海光信息2024年报",
                    "company": "海光信息",
                    "report_date": "2024-12-31",
                    "page_num": index,
                },
            )
            for index in range(1, 4)
        ]
        self.tools = FinancialAtomicTools(
            self.simple_adapter,
            self.structured_adapter,
            self.documents,
        )

    def test_report_search_selects_expected_adapter(self):
        simple = self.tools.report_search(query="净利润")
        structured = self.tools.report_search(
            query="对比",
            structured_query={"task_type": "compare", "sub_queries": ["a", "b"]},
        )
        self.assertEqual(simple["meta"]["search_kind"], "simple")
        self.assertEqual(structured["meta"]["search_kind"], "structured")
        self.assertEqual(len(self.simple_adapter.calls), 1)
        self.assertEqual(len(self.structured_adapter.calls), 1)

    def test_context_expand_returns_bounded_neighbors(self):
        result = self.tools.context_expand("report-1", "report-1-c2", before=1, after=1)
        self.assertEqual([doc["distance"] for doc in result["docs"]], [-1, 0, 1])
        self.assertEqual(result["docs"][1]["relation"], "target")

    def test_report_catalog_deduplicates_and_filters(self):
        result = self.tools.report_catalog(company="海光", year="2024")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["reports"][0]["doc_id"], "report-1")

    def test_metric_calculate_uses_decimal_arithmetic(self):
        growth = self.tools.metric_calculate("growth_rate", [120, 100], precision=2)
        margin = self.tools.metric_calculate("gross_margin", [200, 120], precision=2)
        self.assertEqual(growth["result"], 20.0)
        self.assertEqual(growth["unit"], "%")
        self.assertEqual(margin["result"], 40.0)


if __name__ == "__main__":
    unittest.main()
