"""项目 CLI：构建知识库、单次问答或交互对话。"""
import argparse
import sys
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(description="金融研报 RAG Agent")
    commands = parser.add_subparsers(dest="command")
    chat = commands.add_parser("chat", help="启动问答；不指定问题时进入交互模式")
    chat.add_argument("--question", help="执行一次问答后退出")
    chat.add_argument("--session-id", default="cli", help="会话标识，默认 cli")
    chat.add_argument("--vector-store", help="已有向量库目录")
    build = commands.add_parser("build", help="解析 PDF 并构建知识库")
    build.add_argument("--pdf-dir", help="PDF 目录，默认 dataset/raw_pdf")
    build.add_argument("--save-path", help="索引输出目录，默认 dataset/vector_store")
    return parser


def main(argv=None):
    args = create_parser().parse_args(argv)
    try:
        from configs import model_config, rag_config

        if args.command == "build":
            pdf_dir = Path(args.pdf_dir or rag_config.RAW_PDF_DIR)
            if not pdf_dir.is_dir() or not any(pdf_dir.rglob("*.[pP][dD][fF]")):
                raise ValueError(f"目录中没有 PDF：{pdf_dir}。请先放入研报文件。")
            from rag.rag_pipline import run_pdf_to_vector

            run_pdf_to_vector(
                pdf_dir=str(pdf_dir),
                save_path=args.save_path or rag_config.VECTOR_STORE_DIR,
            )
            return 0

        if model_config.LLM_MODE != "api":
            raise ValueError("当前 CLI 支持 API 模型，请在 .env 中设置 LLM_MODE=api。")
        if not model_config.OPENAI_API_KEY.strip():
            raise ValueError("请先在 .env 中填写 OPENAI_API_KEY。")
        store_path = Path(getattr(args, "vector_store", None) or rag_config.VECTOR_STORE_DIR)
        missing = [name for name in ("index.faiss", "index.pkl", "all_documents.json")
                   if not (store_path / name).is_file()]
        if missing:
            raise ValueError(f"知识库缺少 {', '.join(missing)}；请先运行 python main.py build。")
        question = getattr(args, "question", None)
        if question is not None and not question.strip():
            raise ValueError("问题不能为空。")
        from agent.orchestrator import FinancialRAGAgent

        agent = FinancialRAGAgent(
            vector_store_path=str(store_path),
            all_docs_json_path=str(store_path / "all_documents.json"),
        )
        session_id = getattr(args, "session_id", "cli")
        if question is not None:
            print(agent.chat(question.strip(), session_id=session_id))
            return 0

        print("请输入金融研报问题。/clear 清空当前会话；quit 或 exit 退出。")
        while True:
            try:
                question = input("你 > ").strip()
            except EOFError:
                break
            if question.lower() in {"quit", "exit"}:
                break
            if question == "/clear":
                agent.clear_history(session_id)
                print("当前会话已清空。")
            elif question:
                print(f"助手 > {agent.chat(question, session_id=session_id)}\n")
        return 0
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130
    except (ImportError, ValueError, FileNotFoundError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
