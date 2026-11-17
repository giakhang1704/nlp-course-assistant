"""Part IV: run the final test set and write output/evaluation.txt automatically.

Gold: data/evaluation/final (built by scripts/build_eval_set.py).
Scoring is automatic and reproducible:
- parser: valid/() against parser_expected, intent and entities against the labels;
- retrieval: every evidence group has a document in the retrieved top-k;
- answer: an answerable question is correct when the status is `answered`, every
  `must_contain` key fact appears in the answer and at least one citation points to
  a gold document; other questions are correct when the system refuses
  (`insufficient_information`) or asks back (`clarification_needed`) as expected;
- dialogue turn: the same answer rule plus the resolved reference equals the gold target.
The first failing stage is reported as the cause of each error.
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from ..exporters import write_json, write_lines

TECHNICAL = {"generation_error", "llm_unavailable", "not_run"}
EXPECTED = {"retrieved": "answered"}


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text):
    return unicodedata.normalize("NFC", text or "").casefold()


def has_fact(answer, item):
    """`item` is 'a|b|c'; one alternative must occur (left word boundary;
    right boundary only after a digit, so 'probabilit' matches 'probability'
    but 'chương 1' does not match 'chương 10')."""
    text = norm(answer)
    for alt in item.split("|"):
        alt = norm(alt.strip())
        pattern = r"(?<!\w)" + re.escape(alt) + (r"(?!\d)" if alt[-1:].isdigit() else "")
        if re.search(pattern, text):
            return True
    return False


def echoes_question(answer, question):
    """True when the answer is only the question again (citations/sample prefix removed)."""
    def bare(text):
        text = re.sub(r"\[c\d+\]|theo dữ liệu mẫu của project:?", " ", norm(text))
        return " ".join(re.findall(r"\w+", text))
    return bool(question) and bare(answer) == bare(question)


def missing_facts(answer, must):
    return [item for item in must if not has_fact(answer, item)]


def retrieved_docs(trace):
    return [h["document_id"] for h in trace.get("retrieved_chunks", [])]


def evidence_hit(case, trace):
    docs = retrieved_docs(trace)
    return all(any(d in docs for d in group) for group in case["evidence_groups"])


def judge(case, trace, parse=None):
    """Return (correct, cause). `case` uses expected_status/must_contain/evidence_groups."""
    expected = EXPECTED.get(case["expected_status"], case["expected_status"])
    status = trace.get("status", "not_run")
    llm_called = bool(trace.get("generation_attempts"))
    if status == "not_run":
        return False, "chưa chạy"
    if status in TECHNICAL and trace.get("error") == "llm_empty_answer":
        # The model returned insufficient_information with an empty answer:
        # a refusal, rejected by output validation (deterministic at temperature 0).
        status, llm_called = "insufficient_information", True
    elif status in TECHNICAL:
        return False, "lỗi kỹ thuật khi gọi LLM"
    if expected == "answered":
        if status == "answered" and echoes_question(trace.get("answer", ""), trace.get("question") or case.get("question", "")):
            return False, "LLM lặp lại câu hỏi, không trả lời"
        if status == "answered":
            missing = missing_facts(trace.get("answer", ""), case.get("must_contain", []))
            if missing:
                return False, "LLM trả lời thiếu/sai ý chính"
            gold = {d for g in case.get("evidence_groups", []) for d in g}
            cited = {c.get("document_id") for c in trace.get("citations", [])}
            if gold and not cited & gold:
                return False, "citation không trỏ tới nguồn đúng"
            return True, None
        if parse is not None and case.get("parser_expected") and not parse.get("is_valid"):
            return False, "parser không nhận câu"
        if case.get("evidence_groups") and not evidence_hit(case, trace):
            return False, "truy xuất trượt tài liệu đúng"
        if status == "clarification_needed":
            return False, "hỏi lại dù câu đã đủ thông tin"
        return False, ("LLM từ chối dù context có thông tin" if llm_called
                       else "evidence policy chặn trước LLM")
    if status == expected:
        return True, None
    if status == "answered":
        return False, ("trả lời câu không có thông tin (hallucination)" if expected == "insufficient_information"
                       else "tự đoán đối tượng thay vì hỏi lại")
    return False, f"trạng thái {status} thay vì {expected}"


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else f"{a}/0"


def short(text, n=220):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def score(root, run_dir, output_path):
    root, run_dir = Path(root), Path(run_dir)
    gold = root / "data/evaluation/final"
    pcases = read_jsonl(gold / "parser_cases.jsonl")
    qcases = read_jsonl(gold / "qa_cases.jsonl")
    dialogues = read_jsonl(gold / "dialogues.jsonl")
    parsed = {r["sentence"]: r for r in read_jsonl(run_dir / "parser/parser-api.jsonl")}
    trace = {r["question"]: r for r in read_jsonl(run_dir / "qa/qa-trace.jsonl")}
    dtrace = {(r["session_id"], r["turn_id"]): r for r in read_jsonl(run_dir / "dialogue/dialogue-trace.jsonl")}
    part1_path = root / "output/part1-evaluation.json"
    part1 = json.loads(part1_path.read_text(encoding="utf-8")) if part1_path.is_file() else None
    model = next((r.get("model") for r in trace.values() if r.get("model")), "-")
    L = []
    w = L.append
    summary = {}

    w("=" * 78)
    w("COURSE ASSISTANT (CO3085) - ĐÁNH GIÁ HỆ THỐNG (PHẦN IV)")
    w("=" * 78)
    w("Pipeline: Grammar/Earley parser -> semantic frame -> intent/entity -> query -> TF-IDF retrieval")
    w(f"          -> evidence policy -> LLM local ({model}, Ollama, temperature 0) -> kiểm tra citation.")
    w("Sinh bằng: python -B run.py evaluate   (bộ kiểm thử: data/evaluation/final).")
    w("Chấm tự động. Câu có đáp án được tính ĐÚNG khi: trạng thái answered + chứa mọi ý chính")
    w("(must_contain) + ít nhất một citation trỏ tới tài liệu nguồn chuẩn. Câu thiếu thông tin/")
    w("ngoài miền đúng khi hệ thống từ chối; câu thiếu ngữ cảnh đúng khi hệ thống hỏi lại.")
    w("Bộ kiểm thử đã được xem khi phát triển hệ thống; không phải bộ kiểm tra độc lập.")
    w("")

    # 1. Test set
    w("1. TẬP KIỂM THỬ")
    w("-" * 78)
    official = sum(c["source"] == "official" for c in pcases)
    w(f"Câu đơn: {len(pcases)} câu = {official} câu mẫu của giảng viên (sample_queries.txt) + {len(pcases) - official} câu của nhóm.")
    w(f"  Chấm parser: {len(pcases)} câu; chấm hỏi đáp: {len(qcases)} câu (câu sai cấu trúc chỉ chấm parser).")
    for cat, n in Counter(c["category"] for c in pcases).most_common():
        w(f"  - {cat:<20} {n}")
    para = Counter(c["paraphrase_group"] for c in pcases if c["paraphrase_group"])
    w(f"  - nhóm diễn đạt lại: {len(para)} nhóm x 3 cách hỏi = {sum(para.values())} câu (nằm trong answerable)")
    w(f"Hội thoại: {len(dialogues)} phiên / {sum(len(d['turns']) for d in dialogues)} lượt "
      f"({sum(d['source'] == 'official' for d in dialogues)} phiên mẫu của giảng viên + "
      f"{sum(d['source'] != 'official' for d in dialogues)} phiên của nhóm).")
    w("")

    # 2. Part I and parser
    w("2. PHẦN I - GRAMMAR, SINH CÂU, PARSER")
    w("-" * 78)
    if part1:
        sv = part1.get("sample_verification", {})
        w(f"Grammar: {part1['productions']} production; thuật toán {part1['algorithm']}.")
        w(f"Sinh câu: {part1['generated_count']} câu, {part1['unique_generated_count']} câu khác nhau; "
          f"parse lại {sv.get('count', '?')} câu, lỗi {len(sv.get('failures', []))}.")
        w(f"input/sentences.txt: {part1['valid_input_count']}/{part1['input_count']} câu parse được.")
    p_ok = i_ok = e_ok = i_n = 0
    p_err = []
    for c in pcases:
        r = parsed.get(c["question"], {})
        valid = bool(r.get("is_valid"))
        if valid == c["parser_expected"]:
            p_ok += 1
        else:
            p_err.append(f"{c['id']} mong đợi {'cây' if c['parser_expected'] else '()'}, nhận {'cây' if valid else '()'}: {c['question']}")
        if c["parser_expected"] and valid and c["intent_expected"]:
            i_n += 1
            got = (r.get("semantic_frame") or {}).get("intent") or r.get("intent_type")
            i_ok += got == c["intent_expected"]
            if got != c["intent_expected"]:
                p_err.append(f"{c['id']} intent {got} (nhãn {c['intent_expected']}): {c['question']}")
            ids = {e["id"] for e in r.get("entities", [])}
            lacking = sorted(set(c["entities_expected"]) - ids)
            e_ok += not lacking
            if lacking:
                p_err.append(f"{c['id']} thiếu entity {lacking}: {c['question']}")
    off = [c for c in pcases if c["source"] == "official"]
    off_ok = sum(bool(parsed.get(c["question"], {}).get("is_valid")) == c["parser_expected"] for c in off)
    w(f"Bộ kiểm thử: đúng cây/() {pct(p_ok, len(pcases))}; trên câu thuộc grammar có nhãn: "
      f"intent đúng {pct(i_ok, i_n)}, entity đúng {pct(e_ok, i_n)}.")
    w(f"Riêng {len(off)} câu mẫu của giảng viên: đúng cây/() {pct(off_ok, len(off))}.")
    for line in p_err:
        w("  LỖI " + line)
    w("Câu ra () vẫn được thử bằng truy xuất thô (raw_question): grammar không quyết định câu có trả lời được hay không.")
    w("")
    summary["parser"] = {"validity": [p_ok, len(pcases)], "intent": [i_ok, i_n], "entity": [e_ok, i_n]}

    # 3. Retrieval
    w("3. TRUY XUẤT")
    w("-" * 78)
    ans = [c for c in qcases if c["expected_status"] == "retrieved"]
    misses = [c for c in ans if not evidence_hit(c, trace.get(c["question"], {}))]
    w(f"Tài liệu nguồn chuẩn có trong top-5: {pct(len(ans) - len(misses), len(ans))} câu có đáp án.")
    for c in misses:
        w(f"  TRƯỢT {c['id']}: {c['question']} | cần {c['evidence_groups']} | lấy được "
          f"{retrieved_docs(trace.get(c['question'], {}))[:3]}")
    w("")
    summary["retrieval_hit"] = [len(ans) - len(misses), len(ans)]

    # 4. QA
    rows = []
    for c in qcases:
        t = trace.get(c["question"], {})
        correct, cause = judge(c, t, parsed.get(c["question"]))
        rows.append({**c, "trace": t, "status": t.get("status", "not_run"), "correct": correct, "cause": cause})
    n_ok = sum(r["correct"] for r in rows)
    w("4. HỎI ĐÁP (RAG + LLM)")
    w("-" * 78)
    w(f"Số câu kiểm thử: {len(rows)}; chưa chạy: {sum(r['status'] == 'not_run' for r in rows)}.")
    w(f"SỐ CÂU TRẢ LỜI ĐÚNG: {pct(n_ok, len(rows))}")
    w(f"Trạng thái đúng: {pct(sum(r['status'] == EXPECTED.get(r['expected_status'], r['expected_status']) for r in rows), len(rows))}")
    for title, key in [("Theo nguồn", lambda r: "giảng viên" if r["source"] == "official" else "nhóm"),
                       ("Theo loại câu", lambda r: r["category"]),
                       ("Câu có đáp án, theo intent", lambda r: r["intent_expected"] if r["category"] == "answerable" else None)]:
        w(title + ":")
        groups = defaultdict(list)
        for r in rows:
            if key(r):
                groups[key(r)].append(r)
        for name, items in sorted(groups.items()):
            w(f"  {name:<22} {pct(sum(r['correct'] for r in items), len(items))}")
    answered = [r for r in rows if r["status"] == "answered"]
    cited_gold = [r for r in answered if {c.get("document_id") for c in r["trace"].get("citations", [])}
                  & {d for g in r["evidence_groups"] for d in g}]
    w(f"Câu được trả lời có citation: {pct(sum(bool(r['trace'].get('citations')) for r in answered), len(answered))}; "
      f"citation trúng nguồn chuẩn: {pct(len(cited_gold), sum(bool(r['evidence_groups']) for r in answered))}.")
    times = sorted(r["trace"]["generation_seconds"] for r in rows if r["trace"].get("generation_seconds"))
    if times:
        w(f"Thời gian sinh (CPU): trung vị {times[len(times) // 2]:.1f}s, tối đa {times[-1]:.1f}s, {len(times)} lần gọi LLM.")
    w("")
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r["correct"])
    summary["qa"] = {"correct": [n_ok, len(rows)],
                     "by_category": {k: [sum(v), len(v)] for k, v in sorted(by_category.items())}}

    # 5. Paraphrases
    w("5. CÁCH DIỄN ĐẠT KHÁC NHAU CỦA CÙNG YÊU CẦU")
    w("-" * 78)
    groups = defaultdict(list)
    for r in rows:
        if r["paraphrase_group"]:
            groups[r["paraphrase_group"]].append(r)
    for name, items in sorted(groups.items()):
        w(f"{name}: đúng {sum(r['correct'] for r in items)}/{len(items)}")
        for r in items:
            w(f"  {r['id']} {'cây' if r['parser_expected'] else '() '} {'ĐÚNG' if r['correct'] else 'SAI '} {r['question']}"
              + (f"  [{r['cause']}]" if r["cause"] else ""))
    w(f"Nhóm đúng cả 3 cách hỏi: {sum(all(r['correct'] for r in v) for v in groups.values())}/{len(groups)}")
    w("")

    # 6. Dialogue
    w("6. NGỮ CẢNH HỘI THOẠI (PHẦN III)")
    w("-" * 78)
    ref_n = ref_ok = d_ok = d_n = 0
    d_errors = []
    for d in dialogues:
        w(f"{d['session_id']} ({'giảng viên' if d['source'] == 'official' else 'nhóm'})")
        for gt in d["gold_turns"]:
            t = dtrace.get((d["session_id"], gt["turn_index"]), {})
            focus = gt["expected_focus_document_ids"]
            correct, cause = judge({**gt, "evidence_groups": [focus] if focus else []}, t)
            used = [e["id"] for e in t.get("context_used", [])]
            note = ""
            if gt["reference_targets"]:
                ref_n += 1
                good = sorted(used) == sorted(gt["reference_targets"])
                ref_ok += good
                note = f" | tham chiếu -> {used or '-'}" + ("" if good else f" (cần {gt['reference_targets']})")
                if not good and t:
                    correct, cause = False, "giải tham chiếu sai"
            d_n += 1
            d_ok += correct
            w(f"  lượt {gt['turn_index']} [{t.get('status', 'not_run')}] {'ĐÚNG' if correct else 'SAI '} "
              f"{d['turns'][gt['turn_index'] - 1]}{note}")
            if t.get("effective_question") and t["effective_question"] != d["turns"][gt["turn_index"] - 1]:
                w(f"      câu sau khi giải tham chiếu: {t['effective_question']}")
            w(f"      trả lời: {short(t.get('answer'), 160)}")
            if not correct:
                d_errors.append((d["session_id"], gt["turn_index"], cause))
    w(f"Giải tham chiếu đúng: {pct(ref_ok, ref_n)}; lượt hội thoại đúng: {pct(d_ok, d_n)}")
    w("")
    summary["dialogue"] = {"references": [ref_ok, ref_n], "turns": [d_ok, d_n]}

    # 7. Examples
    w("7. VÍ DỤ (LLM/RAG)")
    w("-" * 78)

    def example(title, r):
        t = r["trace"]
        w(f"[{title}] {r['id']}: {r['question']}")
        w(f"  Trạng thái: {r['status']}; tài liệu truy xuất: {', '.join(retrieved_docs(t)[:3]) or '-'}")
        w(f"  Citation: {', '.join(c.get('document_id', '') for c in t.get('citations', [])) or '-'}")
        w(f"  Trả lời: {short(t.get('answer'), 400)}")
    good = [r for r in rows if r["correct"] and r["status"] == "answered"]
    picks = [next((r for r in good if r["source"] == "official"), None),
             next((r for r in good if r["paraphrase_group"] and not r["parser_expected"]), None),
             next((r for r in good if r["source"] != "official" and not r["parser_expected"]
                   and not r["paraphrase_group"]), None)]
    for r in [p for p in picks if p]:
        example("Đúng và có căn cứ", r)
    for r in [r for r in rows if r["category"] == "missing_information" and r["correct"]][:2]:
        example("KB không có thông tin", r)
    risky = [r for r in rows if r["expected_status"] == "insufficient_information" and r["trace"].get("generation_attempts")]
    w(f"Nguy cơ hallucination: {len(risky)}/{sum(r['expected_status'] == 'insufficient_information' for r in rows)} câu "
      f"thiếu thông tin/ngoài miền lọt qua evidence policy và được gửi tới LLM; LLM từ chối đúng "
      f"{sum(r['correct'] for r in risky)}/{len(risky)}.")
    for r in risky[:3]:
        example("Nguy cơ hallucination", r)
    for r in [r for r in rows if r["cause"] == "trả lời câu không có thông tin (hallucination)"]:
        example("Hallucination", r)
    wrong_answers = [r for r in rows if r["status"] == "answered" and not r["correct"]]
    w(f"Câu được trả lời nhưng sai/thiếu ý (có citation nên trông như có căn cứ): {len(wrong_answers)}.")
    for r in wrong_answers:
        example("Trả lời sai có citation", r)
    w("")

    # 8. Errors
    w("8. CÁC TRƯỜNG HỢP SAI VÀ NGUYÊN NHÂN")
    w("-" * 78)
    wrong = [r for r in rows if not r["correct"]]
    for cause, n in Counter(r["cause"] for r in wrong).most_common():
        w(f"  {cause:<46} {n}")
    for session, turn, cause in d_errors:
        w(f"  (hội thoại {session} lượt {turn}) {cause}")
    w("")
    for r in wrong:
        t = r["trace"]
        w(f"- {r['id']} [{r['category']}, {r['status']}] {r['question']}")
        w(f"    Nguyên nhân: {r['cause']}")
        if r["cause"] == "LLM trả lời thiếu/sai ý chính":
            w(f"    Thiếu: {missing_facts(t.get('answer', ''), r['must_contain'])}; đáp án chuẩn: {r['reference_answer']}")
        if r["cause"] == "truy xuất trượt tài liệu đúng":
            w(f"    Cần {r['evidence_groups']}, lấy được {retrieved_docs(t)[:3]}")
        w(f"    Trả lời: {short(t.get('answer'))}")
    w("")

    # 9. Assessment
    w("9. ƯU ĐIỂM VÀ HẠN CHẾ")
    w("-" * 78)
    for line in [
        "Ưu điểm:",
        "- Grammar ngữ nghĩa cho cây cú pháp kèm intent, entity, loại câu và logical form; câu thuộc grammar được truy vấn đúng trường dữ liệu.",
        "- Evidence policy chặn trước LLM các câu không có bằng chứng; LLM chỉ trả lời từ context đánh số [C1..], citation được kiểm tra lại.",
        "- Câu ngoài grammar vẫn được thử bằng truy xuất thô nên không mất câu hỏi tự nhiên.",
        "- Chạy hoàn toàn local (Ollama), tái lập được, không phụ thuộc API.",
        "Hạn chế:",
        "- Grammar chỉ phủ các mẫu câu đã viết; nhiều cách diễn đạt tự nhiên ra () và phụ thuộc vào TF-IDF.",
        "- TF-IDF dựa trên từ: số viết bằng chữ ('tuần thứ năm') hoặc tên riêng không có trong KB ('ChatGPT' thay cho AI) làm trượt tài liệu.",
        "- Slide bài giảng chỉ có 11 chương; Chương 12 chỉ có tóm tắt và subtopics từ dữ liệu mẫu nên câu hỏi chi tiết về chương này dễ thiếu bằng chứng.",
        "- Câu ngoài miền/thiếu thông tin có từ trùng với KB có thể lọt tới LLM; khi đó an toàn phụ thuộc vào việc model tự từ chối.",
        "- Ngữ cảnh hội thoại chỉ xử lý các dạng tham chiếu định sẵn (này/đó/nó/chương sau/hai cái đó) trong 3 lượt gần nhất.",
        "- Chấm tự động bằng ý chính + citation không phát hiện được câu trả lời đúng ý nhưng diễn đạt khác từ khóa, hoặc thêm chi tiết sai bên cạnh ý đúng.",
        "- Model 4B chạy CPU chậm (vài chục giây/câu).",
    ]:
        w(line)
    write_lines(output_path, L)
    write_json(run_dir / "evaluation-summary.json", summary)
    # The required Part II/III files show the same run that was evaluated.
    import shutil
    for source in [run_dir / "qa/answer.txt", run_dir / "qa/retrieval.txt", run_dir / "dialogue/dialogue.txt"]:
        if source.is_file():
            shutil.copy2(source, Path(output_path).parent / source.name)
    return summary


def run_final(root, assistant, parser, run_dir, progress=None):
    """Run parser, QA batch and dialogues on the final set (resuming QA if a checkpoint exists)."""
    from ..exporters import write_jsonl
    from ..rag.batch import run_batch
    from ..dialogue.manager import DialogueAssistant
    from ..dialogue.exporter import run_dialogues
    root, run_dir = Path(root), Path(run_dir)
    sentences = (root / "input/sentences_eval.txt").read_text(encoding="utf-8").splitlines()
    questions = (root / "input/questions_eval.txt").read_text(encoding="utf-8").splitlines()
    dialogues = read_jsonl(root / "data/evaluation/final/dialogues.jsonl")
    results = [parser.parse(s) for s in sentences if s.strip()]
    write_lines(run_dir / "parser/parse-results.txt", (r["parse_tree"] for r in results))
    write_jsonl(run_dir / "parser/parser-api.jsonl", results)
    qa_dir = run_dir / "qa"
    run_batch(assistant, [q for q in questions if q.strip()], qa_dir, resume=(qa_dir / "batch.json").is_file(),
              retry_errors=(qa_dir / "batch.json").is_file(), progress=progress)
    done = read_jsonl(run_dir / "dialogue/dialogue-trace.jsonl")
    if len(done) < sum(len(d["turns"]) for d in dialogues):
        run_dialogues(DialogueAssistant(assistant), dialogues, run_dir / "dialogue", progress=progress)
