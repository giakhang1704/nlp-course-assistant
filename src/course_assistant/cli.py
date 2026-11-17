"""CLI for independently reviewable Part I outputs."""
import argparse
import json
from pathlib import Path
import unittest
import sys
from .config import PROJECT_ROOT, load_config, resource_path
from .exporters import read_sentences, write_lines, write_jsonl
from .part1 import run_part1, verify_samples
from .parsing.generator import SentenceGenerator


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cli = argparse.ArgumentParser(description="CO3085 Course Assistant — CFG, Earley and semantic handoff")
    cli.add_argument("--config", type=Path)
    cli.add_argument("--rag-config", type=Path)
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Produce and verify Part I outputs")
    run.add_argument("--input", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--skip-sample-verification", action="store_true")
    inspect = commands.add_parser("inspect", help="Inspect parse tree, frame and retrieval query (no QA)")
    inspect.add_argument("sentence")
    parse = commands.add_parser("parse", help="One bracketed tree or () per input line")
    parse.add_argument("--input", type=Path)
    parse.add_argument("--output", type=Path)
    parse.add_argument("--api-output", type=Path)
    generate = commands.add_parser("generate")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--output", type=Path)
    grammar = commands.add_parser("grammar")
    grammar.add_argument("--output", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--samples", type=Path)
    commands.add_parser("test", help="Run local regression tests; no API/network calls")
    commands.add_parser("build-kb", help="Build allowlisted KB, chunks and TF-IDF index")
    for name in ("retrieve", "ask"):
        qa_command = commands.add_parser(name, help="Retrieve evidence" if name == "retrieve" else "Answer with grounded generation by the local LLM (Ollama)")
        qa_command.add_argument("question")
        qa_command.add_argument("--top-k", type=int)
        qa_command.add_argument("--output-dir", type=Path)
    batch = commands.add_parser("qa-batch", help="Process one question per input line")
    batch.add_argument("--input", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path)
    batch.add_argument("--retrieve-only", action="store_true")
    batch.add_argument("--resume", action="store_true", help="Continue a matching checkpoint")
    batch.add_argument("--retry-errors", action="store_true", help="Retry failed questions when resuming")
    dialogue = commands.add_parser("dialogue", help="Run grouped multi-turn sessions and export Part III outputs")
    dialogue.add_argument("--input", type=Path, default=PROJECT_ROOT / "input/dialogues.jsonl")
    dialogue.add_argument("--output-dir", type=Path)
    dialogue.add_argument("--retrieve-only", action="store_true")
    chat = commands.add_parser("chat", help="Interactive session with local dialogue memory")
    chat.add_argument("--session-id", default="interactive")
    chat.add_argument("--output-dir", type=Path)
    chat.add_argument("--retrieve-only", action="store_true")
    evaluation = commands.add_parser("evaluate-retrieval", help="Evaluate labeled development retrieval cases, no LLM calls")
    evaluation.add_argument("--input", type=Path)
    evaluation.add_argument("--output-dir", type=Path)
    final = commands.add_parser("evaluate", help="Part IV: run the final test set and write output/evaluation.txt")
    final.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "output/evaluation-final")
    final.add_argument("--output", type=Path)
    final.add_argument("--score-only", action="store_true", help="Score existing traces without running the pipeline")
    review = commands.add_parser("prepare-review", help="Create human QA grading CSV without API calls")
    review.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data/evaluation/retrieval_test.jsonl")
    review.add_argument("--trace", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    scoring = commands.add_parser("score-qa", help="Calculate QA status and manually graded content metrics")
    scoring.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data/evaluation/retrieval_test.jsonl")
    scoring.add_argument("--trace", type=Path, required=True)
    scoring.add_argument("--review", type=Path)
    scoring.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args(argv)
    try:
        if args.command == "test":
            suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
            return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        config = load_config(args.config)
        if args.command in {"prepare-review", "score-qa"}:
            from .evaluation.qa_review import prepare_review, score_qa
            result = (prepare_review(args.labels, args.trace, args.output) if args.command == "prepare-review"
                      else score_qa(args.labels, args.trace, args.output_dir, args.review))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "evaluate" and args.score_only:
            from .evaluation.final import score
            output = args.output or resource_path(config["output"]) / "evaluation.txt"
            print(json.dumps(score(PROJECT_ROOT, args.run_dir, output), ensure_ascii=False, indent=2))
            return 0
        if args.command in {"build-kb", "retrieve", "ask", "qa-batch", "evaluate-retrieval", "dialogue", "chat", "evaluate"}:
            from .rag.qa import CourseAssistant, load_rag_config, export_qa
            from .retrieval.index import build_index
            rag_config = load_rag_config(args.rag_config)
            if args.command == "build-kb":
                print(json.dumps(build_index(PROJECT_ROOT, rag_config), ensure_ascii=False, indent=2))
                return 0
            # Honor the same grammar/lexicon/schema options as Part I.
            from .parsing.api import CourseParser
            qa_parser = CourseParser(resource_path(config["grammar"]), resource_path(config["lexicon"]),
                                     resource_path(config["intent_schema"]), max_tokens=config["max_tokens"])
            assistant = CourseAssistant(config=rag_config, parser=qa_parser)
            if args.command == "evaluate":
                from .evaluation.final import run_final, score
                run_final(PROJECT_ROOT, assistant, qa_parser, args.run_dir,
                          progress=lambda line: print(line, file=sys.stderr, flush=True))
                output = args.output or resource_path(config["output"]) / "evaluation.txt"
                print(json.dumps(score(PROJECT_ROOT, args.run_dir, output), ensure_ascii=False, indent=2))
                return 0
            if args.command in {"dialogue", "chat"}:
                from .dialogue.manager import DialogueAssistant
                from .dialogue.exporter import run_dialogues, export_dialogue, check_destination
                from .knowledge.loaders import read_jsonl
                from .rag.batch import ERROR_STATUSES, STOP_ERRORS
                manager = DialogueAssistant(assistant)
                destination = args.output_dir or resource_path(config["output"])
                if args.command == "dialogue":
                    dialogues = read_jsonl(args.input)
                    records = run_dialogues(manager, dialogues, destination, args.retrieve_only,
                                           progress=lambda line: print(line, file=sys.stderr, flush=True))
                    total = sum(len(d["turns"]) for d in dialogues)
                else:
                    check_destination(destination)
                    manager.state(args.session_id)
                    records = []
                    print("Course Assistant — /reset: xóa ngữ cảnh; /exit: kết thúc.")
                    try:
                        while True:
                            try:
                                question = input("Bạn> ")
                            except EOFError:
                                break
                            if question.strip().casefold() in {"/exit", "/quit"}:
                                break
                            result = manager.ask(question, args.session_id, args.retrieve_only)
                            records.append(result)
                            export_dialogue(records, destination)
                            print(f"Assistant [{result['status']}]> " + (result["answer"] or result["error"] or "Đã truy xuất; xem dialogue.txt."))
                            if result["error"] in STOP_ERRORS:
                                break
                    except KeyboardInterrupt:
                        print("\nĐã dừng hội thoại; các lượt trước đã được lưu.")
                    total = len(records)
                print(json.dumps({"output_dir": str(destination), "saved_turns": len(records),
                                  "total_turns": total, "errors": sum(r["status"] in ERROR_STATUSES for r in records)}, ensure_ascii=False, indent=2))
                return 1 if len(records) < total or any(r["status"] in ERROR_STATUSES for r in records) else 0
            if args.command == "evaluate-retrieval":
                from .evaluation.retrieval import evaluate_retrieval
                summary = evaluate_retrieval(assistant, args.input or PROJECT_ROOT / "data/evaluation/retrieval_dev.jsonl",
                                             args.output_dir or resource_path(config["output"]))
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 0
            questions = read_sentences(args.input) if args.command == "qa-batch" else [args.question]
            if args.command == "qa-batch":
                from .rag.batch import run_batch, ERROR_STATUSES
                destination = args.output_dir or resource_path(config["output"]) / "qa-batch"
                results = run_batch(assistant, questions, destination, args.retrieve_only, args.resume, args.retry_errors)
                print(json.dumps({"output_dir": str(destination), "saved": len(results), "total": len(questions),
                                  "errors": sum(r["status"] in ERROR_STATUSES for r in results)}, ensure_ascii=False, indent=2))
                return 1 if len(results) < len(questions) or any(r["status"] in ERROR_STATUSES for r in results) else 0
            results = [assistant.ask(args.question, retrieve_only=args.command == "retrieve", k=args.top_k)]
            export_qa(results, args.output_dir or resource_path(config["output"]))
            print(json.dumps(results if args.command == "qa-batch" else results[0], ensure_ascii=False, indent=2))
            return 1 if any(r["status"] in {"generation_error", "llm_unavailable"} for r in results) else 0
        # Validate limits before writing or loading any generated output.
        limit = getattr(args, "limit", None)
        limit = config["sample_limit"] if limit is None else limit
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10000:
            raise ValueError("limit must be an integer in [1, 10000]")
        from .parsing.api import CourseParser
        parser = CourseParser(resource_path(config["grammar"]), resource_path(config["lexicon"]),
                              resource_path(config["intent_schema"]), max_tokens=config["max_tokens"])
        output_dir = resource_path(config["output"])
        if args.command == "run":
            report = run_part1(parser, args.input or resource_path(config["input"]),
                               args.output_dir or output_dir, limit,
                               resource_path(config["evaluation"]),
                               PROJECT_ROOT / "data/evaluation/legacy_queries.txt",
                               PROJECT_ROOT / "data/evaluation/independent_cases.jsonl",
                               verify=not args.skip_sample_verification)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1 if report["sample_verification"]["failures"] else 0
        if args.command == "inspect":
            print(json.dumps(parser.parse(args.sentence), ensure_ascii=False, indent=2))
        elif args.command == "parse":
            results = [parser.parse(s) for s in read_sentences(args.input or resource_path(config["input"]))]
            write_lines(args.output or output_dir / "parse-results.txt", (r["parse_tree"] for r in results))
            if args.api_output:
                write_jsonl(args.api_output, results)
            print(json.dumps({"sentences": len(results), "parsed": sum(r["is_valid"] for r in results)}))
        elif args.command == "generate":
            write_lines(args.output or output_dir / "samples.txt", SentenceGenerator(parser.grammar).generate(limit))
        elif args.command == "grammar":
            write_lines(args.output or output_dir / "grammar.txt", parser.grammar.export().splitlines())
        elif args.command == "verify":
            result = verify_samples(parser, read_sentences(args.samples or output_dir / "samples.txt"))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result["failures"] else 0
    except (ValueError, OSError, RuntimeError) as error:
        cli.exit(1, f"Error: {error}\n")
    except ImportError:
        cli.exit(1, "Missing Part II dependency. Run: python -m pip install -r requirements.txt\n")
    except KeyboardInterrupt:
        print("Interrupted; completed batch questions are checkpointed. Use --resume to continue.", file=sys.stderr)
        return 130
    return 0
