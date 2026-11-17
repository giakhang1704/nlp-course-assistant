"""File contract and checkout execution regression tests."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from course_assistant.config import PROJECT_ROOT, DEFAULT_GRAMMAR_PATH
from course_assistant.exporters import read_sentences


class CheckoutContractTests(unittest.TestCase):
    def command(self, *args, cwd=None):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "run.py"), *args],
                              capture_output=True, text=True, encoding="utf-8", cwd=cwd, timeout=40)

    def test_other_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command("inspect", "Môn NLP có bao nhiêu tín chỉ?", cwd=directory)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(json.loads(result.stdout)["is_valid"])
            self.assertTrue(DEFAULT_GRAMMAR_PATH.exists())

    def test_bad_limit_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "samples.txt"
            destination.write_text("KEEP", encoding="utf-8")
            result = self.command("generate", "--limit", "10001", "--output", str(destination))
            self.assertNotEqual(0, result.returncode)
            self.assertEqual("KEEP", destination.read_text(encoding="utf-8"))

    def test_bom_empty_lines_and_unknown_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            destination = Path(directory) / "trees.txt"
            source.write_text("\ufeffCFG là gì?\n\nCFG là gì? 🙂\n", encoding="utf-8")
            result = self.command("parse", "--input", str(source), "--output", str(destination))
            self.assertEqual(0, result.returncode, result.stderr)
            trees = destination.read_text(encoding="utf-8").splitlines()
            self.assertEqual(3, len(trees))
            self.assertTrue(trees[0].startswith("(S "))
            self.assertEqual(["()", "()"], trees[1:])
            self.assertEqual(3, len(read_sentences(source)))

    def test_dialogue_cli_keeps_original_and_resolved_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sessions.jsonl"
            output = root / "result"
            source.write_text(json.dumps({"session_id": "cli", "turns": [
                "Chương 3 nói về gì?", "Giải thích chương đó kỹ hơn."]}, ensure_ascii=False) + "\n", encoding="utf-8")
            result = self.command("dialogue", "--input", str(source), "--retrieve-only", "--output-dir", str(output))
            self.assertEqual(0, result.returncode, result.stderr)
            rows = [json.loads(line) for line in (output / "dialogue-trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual("retrieved", rows[-1]["status"])
            self.assertEqual("Giải thích chương đó kỹ hơn.", rows[-1]["original_question"])
            self.assertIn("Chương 3", rows[-1]["effective_question"])

    def test_interactive_chat_handles_reset_exit_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(PROJECT_ROOT / "run.py"),
                "chat", "--retrieve-only", "--output-dir", directory],
                input="Chương 3 nói về gì?\n/reset\nGiải thích chương đó kỹ hơn.\n/exit\n",
                capture_output=True, text=True, encoding="utf-8", timeout=50)
            self.assertEqual(0, result.returncode, result.stderr)
            rows = [json.loads(line) for line in (Path(directory) / "dialogue-trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["retrieved", "session_reset", "clarification_needed"], [r["status"] for r in rows])


if __name__ == "__main__":
    unittest.main()
