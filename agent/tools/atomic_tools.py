# -*- coding: utf-8 -*-
"""金融 RAG 的四个只读原子工具实现。"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, Iterable, List, Optional


class FinancialAtomicTools:
    def __init__(self, rag_adapter, structured_rag_adapter, documents: Iterable[Any]):
        self.rag_adapter = rag_adapter
        self.structured_rag_adapter = structured_rag_adapter
        self._documents = list(documents)
        self._docs_by_id = self._build_document_index(self._documents)
        self._catalog = self._build_catalog(self._documents)

    def handlers(self) -> Dict[str, Callable[..., Dict[str, Any]]]:
        return {
            "report_search": self.report_search,
            "context_expand": self.context_expand,
            "report_catalog": self.report_catalog,
            "metric_calculate": self.metric_calculate,
        }

    def report_search(
        self,
        query: Optional[str] = None,
        structured_query: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        structured_query = structured_query or None
        if structured_query and structured_query.get("task_type") != "fact":
            result = self.structured_rag_adapter.search(
                structured_query,
                mode=mode,
                use_reranker=use_reranker,
            )
            search_kind = "structured"
        else:
            effective_query = query or (structured_query or {}).get("rewritten_query", "")
            result = self.rag_adapter.search(
                effective_query,
                mode=mode,
                use_reranker=use_reranker,
            )
            search_kind = "simple"

        return {
            **result,
            "meta": {
                **(result.get("meta", {}) or {}),
                "search_kind": search_kind,
            },
        }

    def context_expand(
        self,
        doc_id: str,
        chunk_id: str,
        before: int = 1,
        after: int = 1,
    ) -> Dict[str, Any]:
        documents = self._docs_by_id.get(doc_id, [])
        target_index = next(
            (
                index
                for index, document in enumerate(documents)
                if str((getattr(document, "metadata", {}) or {}).get("chunk_id", "")) == chunk_id
            ),
            None,
        )
        if target_index is None:
            return {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "docs": [],
                "reason": "not_found",
            }

        start = max(0, target_index - before)
        end = min(len(documents), target_index + after + 1)
        expanded = [
            self._normalize_document(document, index, target_index)
            for index, document in enumerate(documents[start:end], start=start)
        ]
        return {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "docs": expanded,
            "meta": {
                "before": target_index - start,
                "after": end - target_index - 1,
            },
        }

    def report_catalog(
        self,
        company: Optional[str] = None,
        year: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        company_norm = (company or "").strip().lower()
        reports = []
        for report in self._catalog:
            searchable = " ".join(
                str(report.get(key, ""))
                for key in ("company", "title", "file_name", "broker")
            ).lower()
            report_date = str(report.get("report_date", ""))
            if company_norm and company_norm not in searchable:
                continue
            if year and not report_date.startswith(year) and year not in searchable:
                continue
            reports.append(dict(report))
            if len(reports) >= limit:
                break
        return {
            "reports": reports,
            "count": len(reports),
            "filters": {"company": company, "year": year},
        }

    def metric_calculate(
        self,
        operation: str,
        values: List[float],
        precision: int = 4,
    ) -> Dict[str, Any]:
        decimals = [Decimal(str(value)) for value in values]
        percent_result = False

        if operation == "sum":
            result = sum(decimals, Decimal("0"))
            formula = " + ".join(str(value) for value in decimals)
        elif operation == "difference":
            result = decimals[0] - decimals[1]
            formula = f"{decimals[0]} - {decimals[1]}"
        elif operation == "ratio":
            result = decimals[0] / decimals[1]
            formula = f"{decimals[0]} / {decimals[1]}"
        elif operation == "growth_rate":
            result = (decimals[0] - decimals[1]) / abs(decimals[1]) * Decimal("100")
            formula = f"({decimals[0]} - {decimals[1]}) / abs({decimals[1]}) * 100"
            percent_result = True
        elif operation == "gross_margin":
            result = (decimals[0] - decimals[1]) / decimals[0] * Decimal("100")
            formula = f"({decimals[0]} - {decimals[1]}) / {decimals[0]} * 100"
            percent_result = True
        else:
            raise ValueError(f"不支持的计算操作：{operation}")

        quantizer = Decimal("1").scaleb(-precision)
        rounded = result.quantize(quantizer, rounding=ROUND_HALF_UP)
        return {
            "operation": operation,
            "values": values,
            "formula": formula,
            "result": float(rounded),
            "unit": "%" if percent_result else None,
            "precision": precision,
        }

    @staticmethod
    def _build_document_index(documents: Iterable[Any]) -> Dict[str, List[Any]]:
        index: Dict[str, List[Any]] = defaultdict(list)
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            doc_id = str(metadata.get("doc_id") or metadata.get("source") or metadata.get("file_name") or "")
            if doc_id:
                index[doc_id].append(document)
        return dict(index)

    @staticmethod
    def _build_catalog(documents: Iterable[Any]) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        seen = set()
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            doc_id = str(metadata.get("doc_id") or metadata.get("source") or metadata.get("file_name") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            reports.append({
                "doc_id": doc_id,
                "file_name": metadata.get("file_name", metadata.get("source", "")),
                "title": metadata.get("title", ""),
                "company": metadata.get("company", ""),
                "broker": metadata.get("broker", ""),
                "report_date": metadata.get("report_date", metadata.get("publish_date", "")),
            })
        return reports

    @staticmethod
    def _normalize_document(document: Any, index: int, target_index: int) -> Dict[str, Any]:
        metadata = dict(getattr(document, "metadata", {}) or {})
        return {
            "doc_id": metadata.get("doc_id", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "file_name": metadata.get("file_name", metadata.get("source", "")),
            "page_num": metadata.get("page_num", metadata.get("page", "")),
            "content": getattr(document, "page_content", ""),
            "metadata": metadata,
            "relation": "target" if index == target_index else "neighbor",
            "distance": index - target_index,
        }
