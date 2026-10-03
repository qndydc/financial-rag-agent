"""验证 CLI 分发和启动前检查；不下载模型、不调用外部 API。"""
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import main as cli


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model_config = types.SimpleNamespace(LLM_MODE="api", OPENAI_API_KEY="test-key")
        self.rag_config = types.SimpleNamespace(
            RAW_PDF_DIR=str(self.directory / "pdf"),
            VECTOR_STORE_DIR=str(self.directory / "index"),
        )
        config = types.ModuleType("configs")
        config.model_config = self.model_config
        config.rag_config = self.rag_config
        self.config_patch = patch.dict(sys.modules, {"configs": config})
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def make_index(self):
        directory = Path(self.rag_config.VECTOR_STORE_DIR)
        directory.mkdir()
        for name in ("index.faiss", "index.pkl", "all_documents.json"):
            (directory / name).touch()
        return directory

    def test_invalid_configuration_stops_before_agent_initialization(self):
        with patch.dict(sys.modules, {"agent.orchestrator": None}):
            self.model_config.OPENAI_API_KEY = ""
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(cli.main(["chat"]), 1)
            self.assertIn("OPENAI_API_KEY", errors.getvalue())
            self.model_config.OPENAI_API_KEY = "test-key"
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(cli.main(["chat"]), 1)
            self.assertIn("python main.py build", errors.getvalue())

    def test_single_question_uses_selected_index_and_session(self):
        store = self.make_index()
        module = types.ModuleType("agent.orchestrator")
        module.FinancialRAGAgent = MagicMock()
        agent = module.FinancialRAGAgent.return_value
        agent.chat.return_value = "测试答案"
        with patch.dict(sys.modules, {"agent": types.ModuleType("agent"), "agent.orchestrator": module}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = cli.main(["chat", "--question", " 净利润？ ", "--session-id", "research"])
        self.assertEqual(code, 0)
        module.FinancialRAGAgent.assert_called_once_with(
            vector_store_path=str(store), all_docs_json_path=str(store / "all_documents.json")
        )
        agent.chat.assert_called_once_with("净利润？", session_id="research")
        self.assertIn("测试答案", output.getvalue())

    def test_default_interactive_mode_clears_history_and_exits(self):
        self.make_index()
        module = types.ModuleType("agent.orchestrator")
        module.FinancialRAGAgent = MagicMock()
        with patch.dict(sys.modules, {"agent": types.ModuleType("agent"), "agent.orchestrator": module}):
            with patch("builtins.input", side_effect=["/clear", "exit"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main([]), 0)
        module.FinancialRAGAgent.return_value.clear_history.assert_called_once_with("cli")
        module.FinancialRAGAgent.return_value.chat.assert_not_called()

    def test_build_does_not_require_model_api_key(self):
        pdf_dir = Path(self.rag_config.RAW_PDF_DIR)
        pdf_dir.mkdir()
        (pdf_dir / "report.PDF").touch()
        self.model_config.OPENAI_API_KEY = ""
        module = types.ModuleType("rag.rag_pipline")
        module.run_pdf_to_vector = MagicMock()
        with patch.dict(sys.modules, {"rag": types.ModuleType("rag"), "rag.rag_pipline": module}):
            self.assertEqual(cli.main(["build"]), 0)
        module.run_pdf_to_vector.assert_called_once_with(
            pdf_dir=str(pdf_dir), save_path=self.rag_config.VECTOR_STORE_DIR
        )


if __name__ == "__main__":
    unittest.main()
